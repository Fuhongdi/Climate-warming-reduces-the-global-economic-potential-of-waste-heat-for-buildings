"""Compare priority-queue network selection against exact HiGHS mixed-integer linear programming across national systems."""
from __future__ import annotations

import json
import time
import highspy
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from code.common.climate_windows import BASELINE_WINDOW
from code.common.paths import DATA_PROCESSED
from code.matching.solver import (
    TechConfig,
    evaluate_all_alpha_options_for_link,
    solve_country_celf,
    solve_country_upper_bound_lp,
)
from code.matching.zone_climate_evaluator import (
    evaluate_zones_and_global_demand,
    load_window_daily_climate,
)

TEST_COUNTRIES = ("FIN", "NLD", "GRC")


def solve_country_exact_milp(
    df_cands: pd.DataFrame,
    q_init_m_gj_d: dict[str, np.ndarray],
    zone_dmq_gj_d: dict[str, np.ndarray],
    zone_meta: dict[str, dict[str, float]],
    cfg: TechConfig,
) -> tuple[float, float, float]:
  """Build & solve exact HiGHS FCTP-MILP over discretized (alpha, C) options; return (N_usd, P_gj, solve_sec)."""
  t0 = time.perf_counter()
  options = []
  for e_idx, row in enumerate(df_cands.to_dict("records")):
    zid = str(row["zone_id"])
    zm = zone_meta[zid]
    if zm["c_ind_usd_per_gj"] <= zm["c_net_usd_per_gj"]:
      continue
    sid = str(row["stream_id"])
    alpha_opts = evaluate_all_alpha_options_for_link(
        ell_km=float(row["ell_km"]),
        upgrade_mode=str(row["upgrade_mode"]),
        q_rem_m_gj_d=q_init_m_gj_d[sid],
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
    )
    for res in alpha_opts:
      options.append({
          "e_idx": e_idx,
          "sid": sid,
          "zid": zid,
          "alpha": res["alpha"],
          "net_benefit": res["net_benefit_usd_yr"],
          "P_gj": res["P_net_waste_heat_gj_yr"],
          "q_used_m": np.array([res[f"q_used_m{m+1:02d}_gj_d"] for m in range(12)], dtype=np.float64),
      })

  if not options:
    return 0.0, 0.0, time.perf_counter() - t0

  n_vars = len(options)
  h = highspy.Highs()
  h.setOptionValue("output_flag", False)
  h.setOptionValue("mip_rel_gap", 1e-3)
  h.setOptionValue("time_limit", 90.0)
  h.setOptionValue("threads", 16)

  h.addVars(n_vars, np.zeros(n_vars), np.ones(n_vars))
  costs = np.array([opt["net_benefit"] for opt in options], dtype=np.float64)
  v_indices = np.arange(n_vars, dtype=np.int32)
  h.changeColsCost(n_vars, v_indices, costs)
  h.changeColsIntegrality(n_vars, v_indices, np.full(n_vars, highspy.HighsVarType.kInteger))
  h.changeObjectiveSense(highspy.ObjSense.kMaximize)

  edges_u = sorted({opt["e_idx"] for opt in options})
  zids_u = sorted({opt["zid"] for opt in options})
  sids_u = sorted({opt["sid"] for opt in options})
  e_map = {e: r for r, e in enumerate(edges_u)}
  z_map = {z: len(edges_u) + r for r, z in enumerate(zids_u)}
  s_base = len(edges_u) + len(zids_u)
  s_map = {s: s_base + 12 * r for r, s in enumerate(sids_u)}
  n_rows = s_base + 12 * len(sids_u)

  r_idx_list = []
  c_idx_list = []
  val_list = []
  upper = np.zeros(n_rows, dtype=np.float64)
  upper[:len(edges_u)] = 1.0
  upper[len(edges_u):s_base] = 1.0
  for s, r0 in s_map.items():
    upper[r0:r0 + 12] = q_init_m_gj_d[s]

  for v_i, opt in enumerate(options):
    r_idx_list.append(e_map[opt["e_idx"]])
    c_idx_list.append(v_i)
    val_list.append(1.0)
    r_idx_list.append(z_map[opt["zid"]])
    c_idx_list.append(v_i)
    val_list.append(float(opt["alpha"]))
    r0 = s_map[opt["sid"]]
    for m in range(12):
      qv = float(opt["q_used_m"][m])
      if qv > 0.0:
        r_idx_list.append(r0 + m)
        c_idx_list.append(v_i)
        val_list.append(qv)

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
  h.run()
  sol = np.asarray(h.getSolution().col_value, dtype=np.float64)
  tot_N = float(np.sum(sol * costs))
  tot_P = float(sum(sol[v_i] * options[v_i]["P_gj"] for v_i in range(n_vars)))
  return tot_N, tot_P, time.perf_counter() - t0


def main() -> None:
  sinks_dir = DATA_PROCESSED / "sinks_2025"
  out_dir = DATA_PROCESSED / "validation_4_milp"
  out_dir.mkdir(parents=True, exist_ok=True)

  df_sources = pd.read_parquet(DATA_PROCESSED / "sources_2025/sources_2025_streams.parquet")
  df_zones = pd.read_parquet(sinks_dir / "urban_heat_zones.parquet")
  df_weights = pd.read_parquet(sinks_dir / "urban_zone_cell_weights.parquet")
  df_reg_cells = pd.read_parquet(sinks_dir / "ghsl_region_cell_volumes.parquet")
  df_reg_params = pd.read_parquet(sinks_dir / "calibrated_region_parameters_2025.parquet")
  df_cands = pd.read_parquet(sinks_dir / "candidates_2025.parquet")

  df_z_sub = df_zones[df_zones["iso3"].isin(TEST_COUNTRIES)].copy()
  df_rc_sub = df_reg_cells[df_reg_cells["iso3"].isin(TEST_COUNTRIES)].copy()
  df_rp_sub = df_reg_params[df_reg_params["iso3"].isin(TEST_COUNTRIES)].copy()

  print("1. Loading 2006-2025 observed daily BAIT & TAS...", flush=True)
  b_d, t_d, m_d = load_window_daily_climate("gswp3-era5__obsclim", BASELINE_WINDOW.years)

  print("2. Evaluating zone demands and counterfactual costs on FIN, NLD, GRC...", flush=True)
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
    N_celf_musd = float(sum(x["net_benefit_usd_yr"] for x in sel_celf) / 1e6)
    P_celf_pj = P_celf_gj / 1e6

    t1 = time.perf_counter()
    U_lp_gj, _ = solve_country_upper_bound_lp(c_sub, q_init, zdmq_iso, zm_iso, cfg)
    t_lp = time.perf_counter() - t1
    U_lp_pj = U_lp_gj / 1e6

    N_milp_usd, P_milp_gj, t_milp = solve_country_exact_milp(c_sub, q_init, zdmq_iso, zm_iso, cfg)
    N_milp_musd = N_milp_usd / 1e6
    P_milp_pj = P_milp_gj / 1e6

    n_rel_diff_pct = 100.0 * abs(N_celf_musd - N_milp_musd) / max(1e-9, N_milp_musd) if N_milp_musd > 0 else 0.0
    p_rel_diff_pct = 100.0 * abs(P_celf_pj - P_milp_pj) / max(1e-9, P_milp_pj) if P_milp_pj > 0 else 0.0
    assert -1e-6 <= P_celf_pj <= U_lp_pj + 1e-6 <= Q_tot_pj + 1e-5, f"Ordering violated in {iso3}!"

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
        "P_rel_diff_pct": p_rel_diff_pct,
        "N_celf_million_usd_yr": N_celf_musd,
        "N_milp_million_usd_yr": N_milp_musd,
        "N_rel_diff_pct": n_rel_diff_pct,
        "celf_sec": t_celf,
        "lp_sec": t_lp,
        "milp_sec": t_milp,
    }
    rows.append(rec)
    print("  ", rec, flush=True)

  df_out = pd.DataFrame(rows)
  df_out.to_csv(out_dir / "validation_4_milp_3countries.csv", index=False)
  (out_dir / "validation_4_summary.json").write_text(json.dumps(rows, indent=2))
  print("\n=== VALIDATION 4 (FIN, NLD, GRC) COMPLETE ===", flush=True)
  print(df_out.to_string(index=False), flush=True)


if __name__ == "__main__":
  main()
