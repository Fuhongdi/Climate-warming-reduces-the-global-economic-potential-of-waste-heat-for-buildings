"""Compare priority-queue network selection against exact HiGHS mixed-integer linear programming across national systems."""
from __future__ import annotations

import json
import time
import highspy
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from code.common.climate_windows import BASELINE_WINDOW
from code.common.paths import DATA_PROCESSED
from code.matching.solver import (
    TechConfig,
    evaluate_all_alpha_options_for_link,
    evaluate_best_design_for_link,
    solve_country_celf,
    solve_country_upper_bound_lp,
)
from code.matching.zone_climate_evaluator import (
    evaluate_zones_and_global_demand,
    load_window_daily_climate,
)

TEST_COUNTRIES = ("CHN", "RUS", "DEU", "FIN", "NLD", "GRC")


def _solve_component_milp(
    df_comp: pd.DataFrame,
    q_init_m_gj_d: dict[str, np.ndarray],
    zone_dmq_gj_d: dict[str, np.ndarray],
    zone_meta: dict[str, dict[str, float]],
    cfg: TechConfig,
) -> tuple[float, float]:
  """Solve exact component-level FCTP-MILP with 360-state source constraints."""
  if len(df_comp) == 0:
    return 0.0, 0.0

  cand_rows = list(df_comp.to_dict("records"))
  if len(cand_rows) == 1:
    row = cand_rows[0]
    zid = str(row["zone_id"])
    zm = zone_meta[zid]
    if zm["c_ind_usd_per_gj"] <= zm["c_net_usd_per_gj"]:
      return 0.0, 0.0
    sid = str(row["stream_id"])
    res = evaluate_best_design_for_link(
        ell_km=float(row["ell_km"]),
        upgrade_mode=str(row["upgrade_mode"]),
        q_rem_m_gj_d=q_init_m_gj_d[sid],
        d_zone_mq_gj_d=zone_dmq_gj_d[zid],
        d_zone_peak_gj_d=zm["peak_gj_d"],
        d_zone_ann_gj=zm["ann_gj"],
        alpha_rem=1.0,
        lambda_dist=zm["lambda_dist"],
        c_ind_usd_per_gj=zm["c_ind_usd_per_gj"],
        cf_co2_t_per_gj=zm["cf_co2_t_per_gj"],
        c_net_usd_per_gj=zm["c_net_usd_per_gj"],
        p_elec_usd_per_mwh=zm["p_elec_usd_per_mwh"],
        p_gas_usd_per_gj=zm["p_gas_usd_per_gj"],
        has_gas=bool(zm["has_gas"]),
        grid_ef_tco2_per_mwh=zm["grid_ef_tco2_per_mwh"],
        cap_scale_civil=zm["cap_scale_civil"],
        cap_scale_equip=zm["cap_scale_equip"],
        cfg=cfg,
        sector=str(row.get("sector", "")),
        subsector_detail=str(row.get("subsector_detail", "")),
    )
    if res is None:
      return 0.0, 0.0
    return float(res["net_benefit_usd_yr"]), float(res["P_net_waste_heat_gj_yr"])

  # Replay component CELF once to record exact realized residual source profiles on shared sources
  zm_comp = {str(z): zone_meta[str(z)] for z in df_comp["zone_id"].unique()}
  q_comp = {str(s): q_init_m_gj_d[str(s)] for s in df_comp["stream_id"].unique()}
  sel_comp, _ = solve_country_celf(df_comp, q_comp, zone_dmq_gj_d, zm_comp, cfg)

  q_rem_track = {s: np.repeat(arr.reshape(12), 30).astype(np.float64) for s, arr in q_comp.items()}
  celf_edge_choice: dict[tuple[str, str], tuple[np.ndarray, float]] = {}
  for x in sel_comp:
    s_k = str(x["stream_id"])
    z_k = str(x["zone_id"])
    a_k = float(x["alpha"])
    celf_edge_choice[(s_k, z_k)] = (q_rem_track[s_k].copy(), a_k)
    # Reconstruct exact 360-state draw of this selected link
    zm = zone_meta[z_k]
    r_chk = evaluate_best_design_for_link(
        ell_km=float(x["ell_km"]),
        upgrade_mode=str(x["upgrade_mode"]),
        q_rem_m_gj_d=q_rem_track[s_k],
        d_zone_mq_gj_d=zone_dmq_gj_d[z_k],
        d_zone_peak_gj_d=zm["peak_gj_d"],
        d_zone_ann_gj=zm["ann_gj"],
        alpha_rem=a_k,
        lambda_dist=zm["lambda_dist"],
        c_ind_usd_per_gj=zm["c_ind_usd_per_gj"],
        cf_co2_t_per_gj=zm["cf_co2_t_per_gj"],
        c_net_usd_per_gj=zm["c_net_usd_per_gj"],
        p_elec_usd_per_mwh=zm["p_elec_usd_per_mwh"],
        p_gas_usd_per_gj=zm["p_gas_usd_per_gj"],
        has_gas=bool(zm["has_gas"]),
        grid_ef_tco2_per_mwh=zm["grid_ef_tco2_per_mwh"],
        cap_scale_civil=zm["cap_scale_civil"],
        cap_scale_equip=zm["cap_scale_equip"],
        cfg=cfg,
        sector=str(x.get("sector", "")),
        subsector_detail=str(x.get("subsector_detail", "")),
    )
    if r_chk is not None:
      q_rem_track[s_k] = np.maximum(0.0, q_rem_track[s_k] - np.asarray(r_chk["q_used_360"], dtype=np.float64))

  src_deg = df_comp["stream_id"].value_counts().to_dict()
  options: list[dict[str, object]] = []
  warm_indices: list[int] = []

  for e_idx, row in enumerate(cand_rows):
    zid = str(row["zone_id"])
    zm = zone_meta[zid]
    if zm["c_ind_usd_per_gj"] <= zm["c_net_usd_per_gj"]:
      continue
    sid = str(row["stream_id"])
    is_shared = src_deg.get(sid, 1) >= 2
    q_base_360 = np.repeat(q_init_m_gj_d[sid].reshape(12), 30).astype(np.float64)

    seen_keys: set[tuple[float, float, float]] = set()
    scales = (1.0, 0.75, 0.50, 0.25) if is_shared else (1.0,)
    for sc in scales:
      alpha_opts = evaluate_all_alpha_options_for_link(
          ell_km=float(row["ell_km"]),
          upgrade_mode=str(row["upgrade_mode"]),
          q_rem_m_gj_d=q_base_360 * sc,
          d_zone_mq_gj_d=zone_dmq_gj_d[zid],
          d_zone_peak_gj_d=zm["peak_gj_d"],
          d_zone_ann_gj=zm["ann_gj"],
          lambda_dist=zm["lambda_dist"],
          c_ind_usd_per_gj=zm["c_ind_usd_per_gj"],
          cf_co2_t_per_gj=zm["cf_co2_t_per_gj"],
          c_net_usd_per_gj=zm["c_net_usd_per_gj"],
          p_elec_usd_per_mwh=zm["p_elec_usd_per_mwh"],
          p_gas_usd_per_gj=zm["p_gas_usd_per_gj"],
          has_gas=bool(zm["has_gas"]),
          grid_ef_tco2_per_mwh=zm["grid_ef_tco2_per_mwh"],
          cap_scale_civil=zm["cap_scale_civil"],
          cap_scale_equip=zm["cap_scale_equip"],
          cfg=cfg,
          sector=str(row.get("sector", "")),
          subsector_detail=str(row.get("subsector_detail", "")),
          all_capacities=is_shared,
      )
      for res in alpha_opts:
        key = (
            round(float(res["alpha"]), 4),
            round(float(res["P_net_waste_heat_gj_yr"]), 2),
            round(float(res["net_benefit_usd_yr"]), 2),
        )
        if key in seen_keys:
          continue
        seen_keys.add(key)
        options.append({
            "e_idx": e_idx,
            "sid": sid,
            "zid": zid,
            "alpha": float(res["alpha"]),
            "net_benefit": float(res["net_benefit_usd_yr"]),
            "P_gj": float(res["P_net_waste_heat_gj_yr"]),
            "q_used_360": np.asarray(res["q_used_360"], dtype=np.float64),
        })

    # Ensure exact realized curtailment profile from 360-state CELF is represented in the discrete option set
    if (sid, zid) in celf_edge_choice:
      q_real, a_real = celf_edge_choice[(sid, zid)]
      r_real = evaluate_best_design_for_link(
          ell_km=float(row["ell_km"]),
          upgrade_mode=str(row["upgrade_mode"]),
          q_rem_m_gj_d=q_real,
          d_zone_mq_gj_d=zone_dmq_gj_d[zid],
          d_zone_peak_gj_d=zm["peak_gj_d"],
          d_zone_ann_gj=zm["ann_gj"],
          alpha_rem=a_real,
          lambda_dist=zm["lambda_dist"],
          c_ind_usd_per_gj=zm["c_ind_usd_per_gj"],
          cf_co2_t_per_gj=zm["cf_co2_t_per_gj"],
          c_net_usd_per_gj=zm["c_net_usd_per_gj"],
          p_elec_usd_per_mwh=zm["p_elec_usd_per_mwh"],
          p_gas_usd_per_gj=zm["p_gas_usd_per_gj"],
          has_gas=bool(zm["has_gas"]),
          grid_ef_tco2_per_mwh=zm["grid_ef_tco2_per_mwh"],
          cap_scale_civil=zm["cap_scale_civil"],
          cap_scale_equip=zm["cap_scale_equip"],
          cfg=cfg,
          sector=str(row.get("sector", "")),
          subsector_detail=str(row.get("subsector_detail", "")),
      )
      if r_real is not None:
        key = (
            round(float(r_real["alpha"]), 4),
            round(float(r_real["P_net_waste_heat_gj_yr"]), 2),
            round(float(r_real["net_benefit_usd_yr"]), 2),
        )
        matched_idx = None
        for idx_o in range(len(options) - 1, -1, -1):
          if options[idx_o]["e_idx"] != e_idx:
            break
          k_o = (
              round(float(options[idx_o]["alpha"]), 4),
              round(float(options[idx_o]["P_gj"]), 2),
              round(float(options[idx_o]["net_benefit"]), 2),
          )
          if k_o == key:
            matched_idx = idx_o
            break
        if matched_idx is None:
          matched_idx = len(options)
          options.append({
              "e_idx": e_idx,
              "sid": sid,
              "zid": zid,
              "alpha": float(r_real["alpha"]),
              "net_benefit": float(r_real["net_benefit_usd_yr"]),
              "P_gj": float(r_real["P_net_waste_heat_gj_yr"]),
              "q_used_360": np.asarray(r_real["q_used_360"], dtype=np.float64),
          })
        warm_indices.append(matched_idx)

  if not options:
    return 0.0, 0.0

  n_vars = len(options)
  h = highspy.Highs()
  h.setOptionValue("output_flag", False)
  h.setOptionValue("mip_rel_gap", 1e-4)
  h.setOptionValue("time_limit", 60.0)
  h.setOptionValue("threads", 4)

  h.addVars(n_vars, np.zeros(n_vars), np.ones(n_vars))
  costs = np.array([float(opt["net_benefit"]) for opt in options], dtype=np.float64)
  v_indices = np.arange(n_vars, dtype=np.int32)
  h.changeColsCost(n_vars, v_indices, costs)
  h.changeColsIntegrality(n_vars, v_indices, np.full(n_vars, highspy.HighsVarType.kInteger))
  h.changeObjectiveSense(highspy.ObjSense.kMaximize)

  edges_u = sorted({int(opt["e_idx"]) for opt in options})
  zids_u = sorted({str(opt["zid"]) for opt in options})
  # Only sources with >=2 distinct active edges in options can experience cross-link contention
  sid_edges: dict[str, set[int]] = {}
  for opt in options:
    sid_edges.setdefault(str(opt["sid"]), set()).add(int(opt["e_idx"]))
  shared_sids_u = sorted([s for s, eset in sid_edges.items() if len(eset) >= 2])

  e_map = {e: r for r, e in enumerate(edges_u)}
  z_map = {z: len(edges_u) + r for r, z in enumerate(zids_u)}
  s_base = len(edges_u) + len(zids_u)
  s_map = {s: s_base + 360 * r for r, s in enumerate(shared_sids_u)}
  n_rows = s_base + 360 * len(shared_sids_u)

  r_idx_list: list[int] = []
  c_idx_list: list[int] = []
  val_list: list[float] = []
  upper = np.zeros(n_rows, dtype=np.float64)
  upper[:len(edges_u)] = 1.0
  upper[len(edges_u):s_base] = 1.0
  for s, r0 in s_map.items():
    upper[r0:r0 + 360] = np.repeat(q_init_m_gj_d[s].reshape(12), 30)

  for v_i, opt in enumerate(options):
    r_idx_list.append(e_map[int(opt["e_idx"])])
    c_idx_list.append(v_i)
    val_list.append(1.0)

    r_idx_list.append(z_map[str(opt["zid"])])
    c_idx_list.append(v_i)
    val_list.append(float(opt["alpha"]))

    sid_v = str(opt["sid"])
    if sid_v in s_map:
      r0 = s_map[sid_v]
      q360 = np.asarray(opt["q_used_360"], dtype=np.float64)
      nz = np.flatnonzero(q360 > 1e-9)
      if len(nz) > 0:
        r_idx_list.extend((r0 + nz).tolist())
        c_idx_list.extend([v_i] * len(nz))
        val_list.extend(q360[nz].tolist())

  mat = csr_matrix((val_list, (r_idx_list, c_idx_list)), shape=(n_rows, n_vars))
  inf = highspy.kHighsInf
  h.addRows(
      n_rows,
      np.full(n_rows, -inf),
      upper,
      int(mat.nnz),
      mat.indptr.astype(np.int32),
      mat.indices.astype(np.int32),
      mat.data.astype(np.float64),
  )

  if warm_indices:
    sol_init = highspy.HighsSolution()
    w_vec = np.zeros(n_vars, dtype=np.float64)
    w_vec[warm_indices] = 1.0
    sol_init.col_value = w_vec.tolist()
    h.setSolution(sol_init)

  h.run()
  sol = np.asarray(h.getSolution().col_value, dtype=np.float64)
  tot_N = float(np.sum(sol * costs))
  tot_P = float(sum(sol[v_i] * float(options[v_i]["P_gj"]) for v_i in range(n_vars)))
  return tot_N, tot_P


def solve_country_exact_milp(
    df_cands: pd.DataFrame,
    q_init_m_gj_d: dict[str, np.ndarray],
    zone_dmq_gj_d: dict[str, np.ndarray],
    zone_meta: dict[str, dict[str, float]],
    cfg: TechConfig,
) -> tuple[float, float, float]:
  """Decompose national bipartite graph into disconnected <=100 km components and solve exact 360-state MILP."""
  t0 = time.perf_counter()
  if len(df_cands) == 0:
    return 0.0, 0.0, 0.0

  # Filter positive-margin zones upfront
  valid_z = {z for z, zm in zone_meta.items() if zm["c_ind_usd_per_gj"] > zm["c_net_usd_per_gj"]}
  df_act = df_cands[df_cands["zone_id"].isin(valid_z)].reset_index(drop=True)
  if len(df_act) == 0:
    return 0.0, 0.0, time.perf_counter() - t0

  sids = sorted(df_act["stream_id"].unique())
  zids = sorted(df_act["zone_id"].unique())
  sid_to_i = {s: idx for idx, s in enumerate(sids)}
  zid_to_j = {z: idx for idx, z in enumerate(zids)}
  n_s, n_z = len(sids), len(zids)
  n_e = len(df_act)

  edge_s = df_act["stream_id"].map(sid_to_i).to_numpy(dtype=np.int32)
  edge_z = df_act["zone_id"].map(zid_to_j).to_numpy(dtype=np.int32)
  adj = csr_matrix(
      (np.ones(2 * n_e, dtype=np.int8),
       (np.concatenate([edge_s, n_s + edge_z]), np.concatenate([n_s + edge_z, edge_s]))),
      shape=(n_s + n_z, n_s + n_z),
  )
  _, comp_labels = connected_components(adj, directed=False, return_labels=True)
  edge_comp = comp_labels[edge_s]

  order_e = np.argsort(edge_comp, kind="mergesort")
  comp_sorted = edge_comp[order_e]
  splits = np.flatnonzero(np.diff(comp_sorted)) + 1
  comp_groups = np.split(order_e, splits)

  tot_N = 0.0
  tot_P = 0.0
  for e_grp in comp_groups:
    n_c, p_c = _solve_component_milp(
        df_act.iloc[e_grp].reset_index(drop=True),
        q_init_m_gj_d,
        zone_dmq_gj_d,
        zone_meta,
        cfg,
    )
    tot_N += n_c
    tot_P += p_c

  return tot_N, tot_P, time.perf_counter() - t0


def main() -> None:
  sinks_dir = DATA_PROCESSED / "sinks_2025"
  out_dir = DATA_PROCESSED / "validation_4_milp"
  out_dir.mkdir(parents=True, exist_ok=True)

  df_sources = pd.read_parquet(DATA_PROCESSED / "sources_2025/sources_2025_streams.parquet")
  df_sources = df_sources[df_sources["iso3"].isin(TEST_COUNTRIES)].reset_index(drop=True)
  df_zones = pd.read_parquet(sinks_dir / "urban_heat_zones.parquet")
  df_reg_params = pd.read_parquet(sinks_dir / "calibrated_region_parameters_2025.parquet")
  df_cands = pd.read_parquet(
      sinks_dir / "candidates_2025.parquet",
      filters=[("iso3", "in", list(TEST_COUNTRIES))],
  )
  df_cands = df_cands.merge(df_sources[["stream_id", "subsector_detail"]], on="stream_id", how="left")

  df_z_sub = df_zones[df_zones["iso3"].isin(TEST_COUNTRIES)].copy()
  df_rp_sub = df_reg_params[df_reg_params["iso3"].isin(TEST_COUNTRIES)].copy()
  del df_zones

  obs_cache_npz = DATA_PROCESSED / "scenarios_2025/_climate_cache/gswp3-era5__obsclim__2006_2025__e2025__b14__ret0.npz"
  if obs_cache_npz.exists():
    print("1. Loading precomputed 2006-2025 observed 360-state climate cache (<0.2 GiB)...", flush=True)
    from code.scenarios.run_global_scenarios import _build_country_zone_meta
    reg_valid = df_reg_params[df_reg_params["valid_for_study"]].copy().set_index("region_id")
    df_z_sub = df_z_sub[df_z_sub["region_id"].isin(reg_valid.index)].reset_index(drop=True)
    zset = set(df_z_sub["zone_id"].astype(str))
    with np.load(obs_cache_npz, allow_pickle=True) as npz:
      zids_all = npz["zids"].astype(str)
      ann_all = npz["z_ann_gj"]
      pk_all = npz["z_peak_gj_d"]
      dmq_all = npz["z_dmq"]
      e25_all = npz["z_ashp_el_2025"]
      e50_all = npz["z_ashp_el_2050"]
      z_stats = {}
      zone_dmq = {}
      for idx in range(len(zids_all)):
        zk = str(zids_all[idx])
        if zk in zset:
          z_stats[zk] = (float(ann_all[idx]), float(pk_all[idx]), float(e25_all[idx]), float(e50_all[idx]))
          zone_dmq[zk] = np.asarray(dmq_all[idx], dtype=np.float64)
      df_reg_D = pd.DataFrame({
          "iso3": npz["reg_iso3"].astype(str),
          "ann_tot_gj": npz["reg_tot_gj"].astype(np.float64),
      })
    zone_meta = _build_country_zone_meta(
        df_z_iso=df_z_sub,
        z_stats=z_stats,
        reg_valid=reg_valid,
        discount_rate=0.07,
        use_retail_prices=False,
        price_multiplier=1.0,
        is_4gdh=False,
        use_2050_dh_tech=False,
        use_2050_indiv_tech=False,
        use_2050_grid_ef=False,
    )
  else:
    df_weights = pd.read_parquet(sinks_dir / "urban_zone_cell_weights.parquet")
    df_reg_cells = pd.read_parquet(sinks_dir / "ghsl_region_cell_volumes.parquet")
    df_rc_sub = df_reg_cells[df_reg_cells["iso3"].isin(TEST_COUNTRIES)].copy()
    print("1. Loading 2006-2025 observed daily BAIT & TAS...", flush=True)
    b_d, t_d, m_d = load_window_daily_climate("gswp3-era5__obsclim", BASELINE_WINDOW.years)
    print(f"2. Evaluating zone demands and counterfactual costs on {TEST_COUNTRIES}...", flush=True)
    zone_meta, zone_dmq, df_reg_D = evaluate_zones_and_global_demand(
        b_d, t_d, m_d, df_z_sub, df_weights, df_rc_sub, df_rp_sub,
    )

  cfg = TechConfig()
  rows = []
  for iso3 in TEST_COUNTRIES:
    s_sub = df_sources[df_sources["iso3"] == iso3]
    c_sub = df_cands[(df_cands["iso3"] == iso3) & df_cands["zone_id"].isin(zone_meta)].reset_index(drop=True)
    z_ids_iso = set(df_z_sub.loc[df_z_sub["iso3"] == iso3, "zone_id"])
    zm_iso = {z: zone_meta[z] for z in z_ids_iso if z in zone_meta}
    zdmq_iso = {z: zone_dmq[z] for z in zm_iso}

    q_init = {
        str(r.stream_id): np.array([getattr(r, f"q3g_m{m+1:02d}_gj_d") for m in range(12)], dtype=np.float64)
        for r in s_sub.itertuples(index=False)
    }
    Q_tot_pj = float(s_sub["q3g_annual_gj"].sum() / 1e6)
    D_tot_pj = float(df_reg_D.loc[df_reg_D["iso3"] == iso3, "ann_tot_gj"].sum() / 1e6)

    t0 = time.perf_counter()
    sel_celf, P_celf_gj = solve_country_celf(c_sub, q_init, zdmq_iso, zm_iso, cfg)
    t_celf = time.perf_counter() - t0
    N_celf_musd = float(sum(float(x["net_benefit_usd_yr"]) for x in sel_celf) / 1e6)
    P_celf_pj = P_celf_gj / 1e6

    t1 = time.perf_counter()
    U_lp_gj, _ = solve_country_upper_bound_lp(c_sub, q_init, zdmq_iso, zm_iso, cfg)
    t_lp = time.perf_counter() - t1
    U_lp_pj = U_lp_gj / 1e6

    N_milp_usd, P_milp_gj, t_milp = solve_country_exact_milp(c_sub, q_init, zdmq_iso, zm_iso, cfg)
    N_milp_musd = N_milp_usd / 1e6
    P_milp_pj = P_milp_gj / 1e6

    n_signed_diff_pct = 100.0 * (N_celf_musd - N_milp_musd) / max(1e-9, N_milp_musd) if N_milp_musd > 0 else 0.0
    p_signed_diff_pct = 100.0 * (P_celf_pj - P_milp_pj) / max(1e-9, P_milp_pj) if P_milp_pj > 0 else 0.0
    n_rel_diff_pct = abs(n_signed_diff_pct)
    p_rel_diff_pct = abs(p_signed_diff_pct)

    assert -1e-6 <= P_celf_pj <= U_lp_pj + 1e-6 <= Q_tot_pj + 1e-5, f"Ordering violated in {iso3}!"
    assert -1e-6 <= P_milp_pj <= U_lp_pj + 1e-5, f"MILP P exceeded LP U in {iso3}: {P_milp_pj} > {U_lp_pj}"
    assert N_celf_musd <= N_milp_musd * 1.0002 + 1e-3, (
        f"CELF N ({N_celf_musd}) exceeded MILP N ({N_milp_musd}) in {iso3}!"
    )

    rec = {
        "iso3": iso3,
        "n_sources": int(len(s_sub)),
        "n_zones": int(len(zm_iso)),
        "n_candidates": int(len(c_sub)),
        "Q_pj_yr": Q_tot_pj,
        "D_pj_yr": D_tot_pj,
        "U_lp_pj_yr": U_lp_pj,
        "P_celf_pj_yr": P_celf_pj,
        "P_milp_pj_yr": P_milp_pj,
        "P_signed_diff_pct": p_signed_diff_pct,
        "P_rel_diff_pct": p_rel_diff_pct,
        "N_celf_million_usd_yr": N_celf_musd,
        "N_milp_million_usd_yr": N_milp_musd,
        "N_signed_diff_pct": n_signed_diff_pct,
        "N_rel_diff_pct": n_rel_diff_pct,
        "celf_sec": t_celf,
        "lp_sec": t_lp,
        "milp_sec": t_milp,
    }
    rows.append(rec)
    print("  ", rec, flush=True)

  df_out = pd.DataFrame(rows)
  df_out.to_csv(out_dir / "validation_4_milp_3countries.csv", index=False)
  df_out.to_csv(out_dir / "validation_4_milp_6countries.csv", index=False)
  (out_dir / "validation_4_summary.json").write_text(json.dumps(rows, indent=2))
  print(f"\n=== VALIDATION 4 ({', '.join(TEST_COUNTRIES)}) COMPLETE ===", flush=True)
  print(df_out.to_string(index=False), flush=True)


if __name__ == "__main__":
  main()
