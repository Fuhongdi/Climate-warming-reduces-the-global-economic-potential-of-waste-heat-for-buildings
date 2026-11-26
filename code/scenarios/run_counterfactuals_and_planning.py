#!/usr/bin/env python3
"""Run within-configuration 2050 warming decompositions, fixed-COP mechanism counterfactuals, cogeneration fuel allocation sensitivities, out-of-sample forward-climate network planning and 360-quantile solver diagnostics."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import sys
import tempfile
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MALLOC_TRIM_THRESHOLD_", "65536")

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("WASTE_HEAT_ROOT", Path(__file__).resolve().parents[2]))
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from code.common.parameters import (
    DH_SUBSTATION_2025,
    DISCOUNT_RATE_CENTRAL,
    INDIV_ASHP_2025,
    INDIV_GAS_BOILER_2025,
    NATURAL_GAS_EF_TCO2_PER_GJ_LHV,
    PEAKING_ELEC_BOILER_2025,
    PEAKING_GAS_BOILER_2025,
    PIPE_CAPEX_COEF_EUR_PER_KM,
    PIPE_CAPEX_EXPONENT,
    PIPE_FIXED_OM_EUR_PER_MW_KM_YR,
    PIPE_LIFETIME_YR,
    PIPE_PUMP_ELEC_MWH_PER_MWH_TH,
    PIPE_VAR_OM_EUR_PER_MWH_KM,
    STATION_LIFETIME_YR,
    USD_PER_EUR_2025,
)
from code.common.paths import DATA_PROCESSED, GJ_PER_MW_DAY, GJ_PER_MWH
from code.matching.locked_asset_evaluator import evaluate_locked_portfolio
from code.matching.solver import DAYS_MONTH, TechConfig, crf, get_upgrade_params, solve_country_celf
import code.scenarios.run_global_scenarios as rgs

MODS = ["gfdl-esm4", "ipsl-cm6a-lr", "mpi-esm1-2-hr", "mri-esm2-0", "ukesm1-0-ll"]
CLIM_DIR = DATA_PROCESSED / "scenarios_2025/_climate_cache"
U_DIR = DATA_PROCESSED / "scenarios_2025/_u_cache"
SCEN_DIR = DATA_PROCESSED / "scenarios_2025"
N_WORKERS = 4


def evaluate_portfolio_general(
    df_links: pd.DataFrame,
    q_init_m_gj_d: dict[str, np.ndarray],
    zone_dmq_fut: dict[str, np.ndarray],
    zone_meta_fut: dict[str, dict[str, float]],
    zone_meta_design: dict[str, dict[str, float]],
    discount_rate: float = 0.07,
    enforce_fut_peak_adequacy: bool = True,
) -> pd.DataFrame:
  """Evaluate any locked portfolio (historical or forward-designed) across all 360 (month, quantile) states."""
  if len(df_links) == 0:
    return pd.DataFrame()

  cfg = TechConfig(discount_rate=discount_rate, is_4gdh=False, use_2050_dh_tech=False)
  a_pipe = crf(discount_rate, PIPE_LIFETIME_YR)
  a_eq = crf(discount_rate, STATION_LIFETIME_YR)

  q_rem = {
      sid: (arr.reshape(12, 30).copy() if arr.size == 360 else np.broadcast_to(arr.reshape(12)[:, None], (12, 30)).copy())
      for sid, arr in q_init_m_gj_d.items()
  }
  out_rows = []

  for r in df_links.itertuples(index=False):
    sid = str(r.stream_id)
    zid = str(r.zone_id)
    if zid not in zone_meta_fut or zid not in zone_meta_design or sid not in q_rem:
      continue

    zm_f = zone_meta_fut[zid]
    zm_d = zone_meta_design[zid]
    dmq_f = zone_dmq_fut[zid]
    n_q = dmq_f.shape[1]
    w_mq = DAYS_MONTH[:, None] / float(n_q)

    alpha = float(r.alpha)
    c_mw = float(r.c_mw)
    c_src_mw = float(r.c_src_mw)
    c_pk_mw_locked = float(r.c_pk_mw)
    ell_km = float(r.ell_km)
    eta_tot = float(r.eta_tot)
    up_mode = str(r.upgrade_mode)
    has_gas = bool(zm_d["has_gas"])
    cap_civ = float(zm_d["cap_scale_civil"])
    cap_eq = float(zm_d["cap_scale_equip"])
    p_el = float(zm_d["p_elec_usd_per_mwh"])
    p_ng = float(zm_d["p_gas_usd_per_gj"])
    grid_ef = float(zm_d["grid_ef_tco2_per_mwh"])

    kappa, cop, eq_capex_eur, eq_fom_eur, eq_vom_eur = get_upgrade_params(up_mode, cfg)
    pk_sp = PEAKING_GAS_BOILER_2025 if has_gas else PEAKING_ELEC_BOILER_2025
    a_pk = crf(discount_rate, pk_sp.lifetime_yr)
    a_sub = crf(discount_rate, DH_SUBSTATION_2025.lifetime_yr)

    c_d_gj = c_mw * GJ_PER_MW_DAY
    src_avail_mq = kappa * eta_tot * q_rem[sid]
    sub_dmq_f = alpha * dmq_f
    h_mq_f = np.minimum(c_d_gj, np.minimum(src_avail_mq, sub_dmq_f))
    H_deliv_f_gj = float(np.sum(h_mq_f * w_mq))

    q_used_mq = h_mq_f / max(1e-9, kappa * eta_tot)
    q_rem[sid] = np.maximum(0.0, q_rem[sid] - q_used_mq)

    c_pk_eff = c_pk_mw_locked
    extra_sub_cap_usd = 0.0
    extra_sub_ann_usd = 0.0
    extra_sub_fom_el_usd = 0.0
    if enforce_fut_peak_adequacy:
      sub_peak_mw_fut = alpha * float(zm_f["peak_gj_d"]) / GJ_PER_MW_DAY
      sub_peak_mw_des = alpha * float(zm_d["peak_gj_d"]) / GJ_PER_MW_DAY
      min_sup_mw_fut = min(c_mw, float(np.min(src_avail_mq)) / GJ_PER_MW_DAY)
      req_pk_mw = max(0.0, sub_peak_mw_fut - min_sup_mw_fut)
      c_pk_eff = max(c_pk_mw_locked, req_pk_mw)
      if sub_peak_mw_fut > sub_peak_mw_des:
        d_peak = sub_peak_mw_fut - sub_peak_mw_des
        extra_sub_cap_usd = DH_SUBSTATION_2025.capex_eur_per_mw_h * d_peak * USD_PER_EUR_2025 * cap_eq
        extra_sub_fom_usd = DH_SUBSTATION_2025.fixed_om_eur_per_mw_h_yr * d_peak * USD_PER_EUR_2025 * cap_eq
        extra_sub_el_usd = (DH_SUBSTATION_2025.aux_elec_mwh_per_mw_h_yr * d_peak) * p_el
        extra_sub_fom_el_usd = extra_sub_fom_usd + extra_sub_el_usd
        extra_sub_ann_usd = a_sub * extra_sub_cap_usd + extra_sub_fom_el_usd

    pipe_cap_usd = PIPE_CAPEX_COEF_EUR_PER_KM * (c_src_mw ** PIPE_CAPEX_EXPONENT) * ell_km * USD_PER_EUR_2025 * cap_civ
    pipe_fom_usd = PIPE_FIXED_OM_EUR_PER_MW_KM_YR * c_src_mw * ell_km * USD_PER_EUR_2025 * cap_civ
    eq_cap_usd = eq_capex_eur * c_src_mw * USD_PER_EUR_2025 * cap_eq
    eq_fom_usd = eq_fom_eur * c_src_mw * USD_PER_EUR_2025 * cap_eq
    pk_cap_usd = pk_sp.capex_eur_per_mw_h * c_pk_eff * USD_PER_EUR_2025 * cap_eq
    pk_fom_usd = pk_sp.fixed_om_eur_per_mw_h_yr * c_pk_eff * USD_PER_EUR_2025 * cap_eq

    locked_dist_sub_ann_usd = alpha * float(zm_d["ann_gj"]) * float(zm_d["c_net_usd_per_gj"]) + extra_sub_ann_usd
    locked_dist_sub_capex_usd = alpha * float(zm_d["dist_sub_capex_usd_full"]) + extra_sub_cap_usd
    total_sunk_capex_usd = pipe_cap_usd + eq_cap_usd + pk_cap_usd + locked_dist_sub_capex_usd

    locked_fixed_ann_usd = (
        locked_dist_sub_ann_usd
        + a_pipe * pipe_cap_usd + pipe_fom_usd
        + a_eq * eq_cap_usd + eq_fom_usd
        + a_pk * pk_cap_usd + pk_fom_usd
    )

    sub_D_f_gj = alpha * float(zm_f["ann_gj"])
    H_src_f_mwh = (H_deliv_f_gj / max(1e-9, eta_tot)) / GJ_PER_MWH
    pipe_vom_usd = PIPE_VAR_OM_EUR_PER_MWH_KM * H_src_f_mwh * ell_km * USD_PER_EUR_2025
    eq_vom_usd = eq_vom_eur * H_src_f_mwh * USD_PER_EUR_2025
    pump_el_mwh = PIPE_PUMP_ELEC_MWH_PER_MWH_TH * H_src_f_mwh
    hp_el_mwh = (H_src_f_mwh / cop) if math.isfinite(cop) else 0.0

    H_pk_f_gj = max(0.0, sub_D_f_gj - H_deliv_f_gj)
    pk_vom_usd = pk_sp.var_om_non_elec_eur_per_mwh_h * (H_pk_f_gj / GJ_PER_MWH) * USD_PER_EUR_2025
    if pk_sp.fuel == "gas":
      pk_fuel_gj = H_pk_f_gj / pk_sp.efficiency_lhv
      pk_el_mwh = pk_sp.aux_elec_mwh_per_mwh_h * (H_pk_f_gj / GJ_PER_MWH)
      pk_energy_usd = pk_fuel_gj * p_ng + pk_el_mwh * p_el
      pk_co2_t = pk_fuel_gj * NATURAL_GAS_EF_TCO2_PER_GJ_LHV + pk_el_mwh * grid_ef
    else:
      pk_el_mwh = (H_pk_f_gj / GJ_PER_MWH) / pk_sp.efficiency_lhv
      pk_energy_usd = pk_el_mwh * p_el
      pk_co2_t = pk_el_mwh * grid_ef

    tot_el_mwh = hp_el_mwh + pump_el_mwh
    var_cost_f_usd = pipe_vom_usd + eq_vom_usd + tot_el_mwh * p_el + pk_vom_usd + pk_energy_usd
    tot_dh_cost_f_usd = locked_fixed_ann_usd + var_cost_f_usd

    avoided_ind_f_usd = sub_D_f_gj * float(zm_f["c_ind_usd_per_gj"])
    net_benefit_f_usd = avoided_ind_f_usd - tot_dh_cost_f_usd

    ongoing_cash_f_usd = (
        alpha * float(zm_d.get("dist_sub_fom_el_usd_full", 0.0)) + extra_sub_fom_el_usd
        + pipe_fom_usd + eq_fom_usd + pk_fom_usd + var_cost_f_usd
    )
    op_margin_f_usd = avoided_ind_f_usd - ongoing_cash_f_usd

    dh_co2_f_t = tot_el_mwh * grid_ef + pk_co2_t
    co2_abated_f_t = sub_D_f_gj * float(zm_f["cf_co2_t_per_gj"]) - dh_co2_f_t

    out_rows.append({
        "iso3": str(r.iso3),
        "stream_id": sid,
        "zone_id": zid,
        "alpha": alpha,
        "c_mw": c_mw,
        "total_sunk_capex_usd": total_sunk_capex_usd,
        "sub_D_fut_gj_yr": sub_D_f_gj,
        "P_fut_gj_yr": H_deliv_f_gj / kappa,
        "net_benefit_fut_usd_yr": net_benefit_f_usd,
        "operating_margin_excl_sunk_fut_usd_yr": op_margin_f_usd,
        "is_loss_making_fut": bool(net_benefit_f_usd < 0.0),
        "is_operating_loss_fut": bool(op_margin_f_usd < 0.0),
        "co2_abated_fut_t_yr": co2_abated_f_t,
        "closure_risk_co2_t_yr": max(0.0, co2_abated_f_t) if net_benefit_f_usd < 0.0 else 0.0,
    })

  return pd.DataFrame(out_rows)


def _init_lean_2025_worker() -> None:
  """Initialize 2025-only worker tables (<0.85 GiB shared via fork)."""
  rgs._init_solver_worker()


def _worker_scenario(scen_task: dict) -> dict:
  """Solve all active countries for one counterfactual/planning scenario and release .npz cache immediately."""
  isos_active: list[str] = scen_task["isos_active"]
  tag = str(scen_task["tag"])
  P_gj_tot = 0.0
  N_usd_tot = 0.0
  co2_t_tot = 0.0
  keep_links_all: list[dict] = []
  try:
    for iso in sorted(isos_active):
      res = _worker_generic({**scen_task, "iso3": iso})
      P_gj_tot += float(res["P_gj"])
      N_usd_tot += float(res["N_usd"])
      co2_t_tot += float(res["co2_t"])
      if res["links"]:
        keep_links_all.extend(res["links"])
    return {"tag": tag, "P_ej": P_gj_tot / 1e9, "N_busd": N_usd_tot / 1e9, "co2_mt": co2_t_tot / 1e6, "links": keep_links_all}
  finally:
    rgs._SOLV_INIT["clim_lru"].clear()  # type: ignore[union-attr]
    rgs._trim_memory()


def _worker_generic(task: dict) -> dict:
  kind = str(task["kind"])
  iso3 = str(task["iso3"])

  src_by_iso = rgs._SOLV_INIT["src_by_iso"]
  reg_valid = rgs._SOLV_INIT["reg_valid"]
  z_by_iso, c_by_iso = rgs._SOLV_INIT["epoch_data"][2025]

  df_z_iso = z_by_iso.get(iso3)
  df_s_iso = src_by_iso.get(iso3)
  df_c_iso = c_by_iso.get(iso3)
  if df_z_iso is None or df_s_iso is None or df_c_iso is None or df_s_iso.empty or df_c_iso.empty:
    return {"tag": task["tag"], "iso3": iso3, "P_gj": 0.0, "N_usd": 0.0, "co2_t": 0.0, "links": []}

  retire_fossil = bool(task.get("retire_fossil", False))
  is_4gdh = bool(task.get("is_4gdh", False))
  use_2050_dh = bool(task.get("use_2050_dh", False))
  use_2050_ind = bool(task.get("use_2050_ind", False))
  use_2050_ef = bool(task.get("use_2050_ef", False))
  chp_mode = str(task.get("chp_mode", "residual"))
  monthly_ded = bool(task.get("use_monthly_source_deduction", False))

  if retire_fossil:
    df_s_iso = df_s_iso[~df_s_iso["is_fossil_power"]]
    df_c_iso = df_c_iso[~df_c_iso["is_fossil_power"]].reset_index(drop=True)
    if df_s_iso.empty or df_c_iso.empty:
      return {"tag": task["tag"], "iso3": iso3, "P_gj": 0.0, "N_usd": 0.0, "co2_t": 0.0, "links": []}

  if kind == "fixed_cop":
    c_stats_fut, z_dmq = rgs._get_country_clim_slice(task["clim_npz_fut"], iso3)
    c_stats_base, _ = rgs._get_country_clim_slice(task["clim_npz_base"], iso3)
    c_stats = {}
    for zid, (ann_f, pk_f, el25_f, el50_f) in c_stats_fut.items():
      if zid in c_stats_base:
        ann_b, _, el25_b, el50_b = c_stats_base[zid]
        ratio = ann_f / max(1.0, ann_b) if ann_b > 1.0 else 1.0
        c_stats[zid] = (ann_f, pk_f, el25_b * ratio, el50_b * ratio)
      else:
        c_stats[zid] = (ann_f, pk_f, el25_f, el50_f)
  else:
    c_stats, z_dmq = rgs._get_country_clim_slice(task["clim_npz"], iso3)

  if not z_dmq:
    return {"tag": task["tag"], "iso3": iso3, "P_gj": 0.0, "N_usd": 0.0, "co2_t": 0.0, "links": []}

  zm_iso = rgs._build_country_zone_meta(
      df_z_iso=df_z_iso,
      z_stats=c_stats,
      reg_valid=reg_valid,
      discount_rate=DISCOUNT_RATE_CENTRAL,
      use_retail_prices=False,
      price_multiplier=1.0,
      is_4gdh=is_4gdh,
      use_2050_dh_tech=use_2050_dh,
      use_2050_indiv_tech=use_2050_ind,
      use_2050_grid_ef=use_2050_ef,
  )
  q_col = "q4g_m" if is_4gdh else "q3g_m"
  m_cols = [f"{q_col}{m:02d}_gj_d" for m in range(1, 13)]
  q_init = {
      str(r.stream_id): np.array([getattr(r, mc) for mc in m_cols], dtype=np.float64)
      for r in df_s_iso.itertuples(index=False)
  }
  cfg = TechConfig(
      discount_rate=DISCOUNT_RATE_CENTRAL,
      is_4gdh=is_4gdh,
      use_2050_dh_tech=use_2050_dh,
      chp_mode=chp_mode,
      use_monthly_source_deduction=monthly_ded,
  )
  selected, P_gj = solve_country_celf(df_c_iso, q_init, z_dmq, zm_iso, cfg)

  N_usd = sum(float(x["net_benefit_usd_yr"]) for x in selected)
  co2_t = sum(float(x["co2_abated_t_yr"]) for x in selected)
  keep_links = selected if bool(task.get("return_links", False)) else []
  if keep_links:
    z_reg_map = dict(zip(df_z_iso["zone_id"].astype(str), df_z_iso["region_id"].astype(str)))
    for x in keep_links:
      x["iso3"] = iso3
      x["region_id"] = z_reg_map.get(str(x["zone_id"]), "")
  return {"tag": task["tag"], "iso3": iso3, "P_gj": float(P_gj), "N_usd": N_usd, "co2_t": co2_t, "links": keep_links}


def _load_full_zm_dmq_custom(
    npz_weights: list[tuple[str, float]],
    df_zones: pd.DataFrame,
    reg_valid: pd.DataFrame,
    fixed_cop_base_npz: str | None = None,
) -> tuple[dict[str, dict[str, float]], dict[str, np.ndarray], dict[str, bool]]:
  acc_ann, acc_pk, acc_el25, acc_el50, acc_dmq = {}, {}, {}, {}, {}
  for npz_p, w in npz_weights:
    with np.load(npz_p, allow_pickle=True) as npz:
      zids = npz["zids"].astype(str)
      ann = npz["z_ann_gj"]
      pk = npz["z_peak_gj_d"]
      dmq = npz["z_dmq"]
      e25 = npz["z_ashp_el_2025"]
      e50 = npz["z_ashp_el_2050"]
      for i in range(len(zids)):
        z = str(zids[i])
        acc_ann[z] = acc_ann.get(z, 0.0) + w * float(ann[i])
        acc_pk[z] = acc_pk.get(z, 0.0) + w * float(pk[i])
        acc_el25[z] = acc_el25.get(z, 0.0) + w * float(e25[i])
        acc_el50[z] = acc_el50.get(z, 0.0) + w * float(e50[i])
        if z not in acc_dmq:
          acc_dmq[z] = w * np.asarray(dmq[i], dtype=np.float64)
        else:
          acc_dmq[z] = acc_dmq[z] + w * np.asarray(dmq[i], dtype=np.float64)

  if fixed_cop_base_npz is not None:
    with np.load(fixed_cop_base_npz, allow_pickle=True) as bnpz:
      bzids = bnpz["zids"].astype(str)
      bann = bnpz["z_ann_gj"]
      be25 = bnpz["z_ashp_el_2025"]
      be50 = bnpz["z_ashp_el_2050"]
      for i in range(len(bzids)):
        z = str(bzids[i])
        if z in acc_ann and float(bann[i]) > 1.0:
          rat = acc_ann[z] / float(bann[i])
          acc_el25[z] = float(be25[i]) * rat
          acc_el50[z] = float(be50[i]) * rat

  z_stats = {z: (acc_ann[z], acc_pk[z], acc_el25[z], acc_el50[z]) for z in acc_ann}
  zm = rgs._build_country_zone_meta(
      df_z_iso=df_zones,
      z_stats=z_stats,
      reg_valid=reg_valid,
      discount_rate=DISCOUNT_RATE_CENTRAL,
      use_retail_prices=False,
      price_multiplier=1.0,
      is_4gdh=False,
      use_2050_dh_tech=False,
      use_2050_indiv_tech=False,
      use_2050_grid_ef=False,
  )

  # Track whether building ASHP is the cheaper decentralized benchmark in each zone (using exact f_space_heat_dd)
  a_gb = crf(DISCOUNT_RATE_CENTRAL, INDIV_GAS_BOILER_2025.lifetime_yr)
  a_hp = crf(DISCOUNT_RATE_CENTRAL, INDIV_ASHP_2025.lifetime_yr)
  is_ashp_comp = {}
  for zrow in df_zones.itertuples(index=False):
    zid = str(zrow.zone_id)
    if zid not in z_stats:
      continue
    ann_gj = max(1.0, z_stats[zid][0])
    pk_mw = max(ann_gj / 365.25, z_stats[zid][1]) / GJ_PER_MW_DAY
    el25 = z_stats[zid][2]
    rp = reg_valid.loc[str(zrow.region_id)]
    has_gas = bool(rp["has_gas"])
    p_el = float(rp["p_elec_pretax_usd_per_mwh"])
    p_ng = float(rp["p_gas_pretax_usd_per_gj"])
    cap_eq = float(rp["cap_scale_equip"])
    if not has_gas or not np.isfinite(p_ng) or p_ng <= 0.0:
      is_ashp_comp[zid] = True
    else:
      dhw_mw = (float(zrow.volume_m3) * float(rp["w_c_gj_per_m3_day"])) / GJ_PER_MW_DAY
      f_sh = float(getattr(zrow, "f_space_heat_dd", 1.0))
      hp_eff_mw = min(pk_mw, dhw_mw) + f_sh * max(0.0, pk_mw - dhw_mw)
      hp_c = (
          a_hp * INDIV_ASHP_2025.capex_eur_per_mw_h * hp_eff_mw * USD_PER_EUR_2025 * cap_eq
          + INDIV_ASHP_2025.fixed_om_eur_per_mw_h_yr * hp_eff_mw * USD_PER_EUR_2025 * cap_eq
          + ((el25 / GJ_PER_MWH) + INDIV_ASHP_2025.aux_elec_mwh_per_mw_h_yr * hp_eff_mw) * p_el
      ) / ann_gj
      gb_c = (
          a_gb * INDIV_GAS_BOILER_2025.capex_eur_per_mw_h * pk_mw * USD_PER_EUR_2025 * cap_eq
          + INDIV_GAS_BOILER_2025.fixed_om_eur_per_mw_h_yr * pk_mw * USD_PER_EUR_2025 * cap_eq
          + (ann_gj / INDIV_GAS_BOILER_2025.efficiency_or_scop_ref) * p_ng
          + (INDIV_GAS_BOILER_2025.aux_elec_mwh_per_mw_h_yr * pk_mw) * p_el
      ) / ann_gj
      is_ashp_comp[zid] = bool(hp_c <= gb_c)

  return zm, acc_dmq, is_ashp_comp


def main() -> None:
  t_start = time.perf_counter()
  results: dict[str, object] = {}

  df_sources = pd.read_parquet(DATA_PROCESSED / "sources_2025/sources_2025_streams.parquet")
  df_zones = pd.read_parquet(DATA_PROCESSED / "sinks_2025/urban_heat_zones.parquet")
  df_rp = pd.read_parquet(DATA_PROCESSED / "sinks_2025/calibrated_region_parameters_2025.parquet")
  reg_valid = df_rp[df_rp["valid_for_study"]].copy().set_index("region_id")
  df_zones_valid = df_zones[df_zones["region_id"].isin(reg_valid.index)].reset_index(drop=True)
  del df_zones, df_rp

  df_cands_iso = pd.read_parquet(DATA_PROCESSED / "sinks_2025/candidates_2025.parquet", columns=["iso3"])
  cand_counts = df_cands_iso["iso3"].value_counts().to_dict()
  del df_cands_iso
  isos_lpt = sorted(df_sources["iso3"].dropna().unique(), key=lambda x: cand_counts.get(x, 0), reverse=True)
  isos_active = [iso for iso in isos_lpt if cand_counts.get(iso, 0) > 0]
  rgs._trim_memory()
  print(f"[Init] Loaded {len(isos_active)} active countries (RSS={rgs._process_tree_rss_gb():.2f} GiB)", flush=True)

  # Ensure 2006-2025 ret1 climate caches exist for Part A
  clim_tasks = []
  for m in MODS:
    ck = f"{m}__ssp245__2006_2025__e2025__b14__ret1"
    out_p = CLIM_DIR / f"{ck}.npz"
    if not out_p.exists():
      clim_tasks.append({
          "clim_key": ck,
          "out_npz": str(out_p),
          "run_id": f"{m}__ssp245",
          "start_year": 2006,
          "end_year": 2025,
          "ghsl_epoch": 2025,
          "bait_threshold_c": 14.0,
          "use_envelope_retrofit": True,
      })
  if clim_tasks:
    print(f"[Part A] Building {len(clim_tasks)} missing ret1 2006-2025 climate caches...", flush=True)
    rgs._init_clim_worker()
    for ct in clim_tasks:
      rgs._compute_one_clim_projection(ct)
    rgs._CLIM_INIT.clear()

  r4_cfgs = [
      ("S3_4gdh_standalone", dict(retire_fossil=False, use_retrofit=False, is_4gdh=True, use_2050_dh=True, use_2050_ind=False, use_2050_ef=False)),
      ("S4_4gdh_symmetric", dict(retire_fossil=False, use_retrofit=False, is_4gdh=True, use_2050_dh=True, use_2050_ind=True, use_2050_ef=False)),
      ("S2_envelope_retrofit", dict(retire_fossil=False, use_retrofit=True, is_4gdh=False, use_2050_dh=False, use_2050_ind=False, use_2050_ef=False)),
      ("S1_fossil_retire", dict(retire_fossil=True, use_retrofit=False, is_4gdh=False, use_2050_dh=False, use_2050_ind=False, use_2050_ef=True)),
      ("S5_combined_all", dict(retire_fossil=True, use_retrofit=True, is_4gdh=True, use_2050_dh=True, use_2050_ind=True, use_2050_ef=True)),
  ]

  d_tot_by_ck: dict[str, float] = {}
  for m in MODS:
    for ret_tag in ("ret0", "ret1"):
      ck = f"{m}__ssp245__2006_2025__e2025__b14__{ret_tag}"
      with np.load(CLIM_DIR / f"{ck}.npz", allow_pickle=True) as npz:
        d_tot_by_ck[ck] = float(np.sum(npz["reg_tot_gj"]))

  # Build scenario-level tasks (< 15 GiB RAM across 6 forked workers)
  scen_tasks: list[dict] = []

  def _add_scen(tag: str, kind: str, extra: dict) -> None:
    scen_tasks.append({"kind": kind, "tag": tag, "isos_active": isos_active, **extra})

  # 3A. Part A: Exact 2006-2025 model-own baselines for ALL 5 R4 configurations (S1..S5)
  for cname, kw in r4_cfgs:
    ret_tag = "ret1" if kw["use_retrofit"] else "ret0"
    kw_clean = {k: v for k, v in kw.items() if k != "use_retrofit"}
    for m in MODS:
      ck = f"{m}__ssp245__2006_2025__e2025__b14__{ret_tag}"
      _add_scen(f"R4_MODBASE__{cname}__{m}", "standard", {"clim_npz": str(CLIM_DIR / f"{ck}.npz"), **kw_clean})

  # 3B. Part B: Fixed ASHP COP on resized 2025 system & 2050 S5_combined_all (5 GCMs 2046-2065)
  for m in MODS:
    ck_b0 = str(CLIM_DIR / f"{m}__ssp245__2006_2025__e2025__b14__ret0.npz")
    ck_f0 = str(CLIM_DIR / f"{m}__ssp245__2046_2065__e2025__b14__ret0.npz")
    _add_scen(f"FIXCOP_R1__{m}", "fixed_cop", {
        "clim_npz_base": ck_b0, "clim_npz_fut": ck_f0,
        "retire_fossil": False, "is_4gdh": False, "use_2050_dh": False, "use_2050_ind": False, "use_2050_ef": False,
    })
    ck_b1 = str(CLIM_DIR / f"{m}__ssp245__2006_2025__e2025__b14__ret1.npz")
    ck_f1 = str(CLIM_DIR / f"{m}__ssp245__2046_2065__e2025__b14__ret1.npz")
    _add_scen(f"FIXCOP_S5__{m}", "fixed_cop", {
        "clim_npz_base": ck_b1, "clim_npz_fut": ck_f1,
        "retire_fossil": True, "is_4gdh": True, "use_2050_dh": True, "use_2050_ind": True, "use_2050_ef": True,
    })

  # 3C. Part C: All three CHP heat-accounting conventions across historical & future trajectories
  chp_traj_scens = [
      ("obs_1946_1965", "gswp3-era5__obsclim__1946_1965__e2025__b14__ret0"),
      ("obs_2006_2025", "gswp3-era5__obsclim__2006_2025__e2025__b14__ret0"),
  ] + [(f"{m}_245_0625", f"{m}__ssp245__2006_2025__e2025__b14__ret0") for m in MODS] \
    + [(f"{m}_245_4665", f"{m}__ssp245__2046_2065__e2025__b14__ret0") for m in MODS] \
    + [(f"{m}_370_0625", f"{m}__ssp370__2006_2025__e2025__b14__ret0") for m in MODS] \
    + [(f"{m}_370_8100", f"{m}__ssp370__2081_2100__e2025__b14__ret0") for m in MODS]

  for chp_tag, chp_mode in [
      ("CHP_PWRLOSS", "power_loss_015"),
      ("CHP_FUELBOTH", "fuel_alloc"),
      ("CHP_FUELCO2", "fuel_co2_only"),
  ]:
    for stag, ck in chp_traj_scens:
      _add_scen(f"{chp_tag}__{stag}", "standard", {"clim_npz": str(CLIM_DIR / f"{ck}.npz"), "chp_mode": chp_mode})

  # 3D. Part D: Blended forward-climate planning designs
  pv_20 = (1.0 - (1.0 + DISCOUNT_RATE_CENTRAL) ** (-20)) / DISCOUNT_RATE_CENTRAL
  pv_40 = (1.0 - (1.0 + DISCOUNT_RATE_CENTRAL) ** (-40)) / DISCOUNT_RATE_CENTRAL
  w_near = pv_20 / pv_40
  w_mid = (pv_40 - pv_20) / pv_40

  _tmp_ctx = tempfile.TemporaryDirectory(prefix="fwd_clim_")
  fwd_tmp_dir = Path(_tmp_ctx.name)
  fwd_design_weights: dict[str, list[tuple[str, float]]] = {}
  for m_hold in MODS:
    others = [m for m in MODS if m != m_hold]
    w_loo_pv = []
    for mo in others:
      w_loo_pv.append((str(CLIM_DIR / f"{mo}__ssp245__2026_2045__e2025__b14__ret0.npz"), w_near / 4.0))
      w_loo_pv.append((str(CLIM_DIR / f"{mo}__ssp245__2046_2065__e2025__b14__ret0.npz"), w_mid / 4.0))
    fwd_design_weights[f"FWD_LOO_PV__{m_hold}"] = w_loo_pv
    fwd_design_weights[f"FWD_LOO_MID__{m_hold}"] = [
        (str(CLIM_DIR / f"{mo}__ssp245__2046_2065__e2025__b14__ret0.npz"), 0.25) for mo in others
    ]
    fwd_design_weights[f"FWD_PERF_PV__{m_hold}"] = [
        (str(CLIM_DIR / f"{m_hold}__ssp245__2026_2045__e2025__b14__ret0.npz"), w_near),
        (str(CLIM_DIR / f"{m_hold}__ssp245__2046_2065__e2025__b14__ret0.npz"), w_mid),
    ]

  print(f"[Part D] Writing {len(fwd_design_weights)} compact blended forward .npz caches...", flush=True)
  for ftag, nw_list in fwd_design_weights.items():
    out_fnpz = fwd_tmp_dir / f"{ftag}.npz"
    zids, z_iso3 = None, None
    ann_w, pk_w, el25_w, el50_w, dmq_w = 0.0, 0.0, 0.0, 0.0, 0.0
    for npz_p, w in nw_list:
      with np.load(npz_p, allow_pickle=True) as npz:
        if zids is None:
          zids = npz["zids"]
          z_iso3 = npz["z_iso3"]
        ann_w = ann_w + w * npz["z_ann_gj"]
        pk_w = pk_w + w * npz["z_peak_gj_d"]
        el25_w = el25_w + w * npz["z_ashp_el_2025"]
        el50_w = el50_w + w * npz["z_ashp_el_2050"]
        dmq_w = dmq_w + w * npz["z_dmq"].astype(np.float32)
    np.savez(
        out_fnpz,
        zids=zids, z_iso3=z_iso3,
        z_ann_gj=ann_w, z_peak_gj_d=pk_w, z_dmq=dmq_w.astype(np.float32),
        z_ashp_el_2025=el25_w, z_ashp_el_2050=el50_w,
    )
    _add_scen(ftag, "standard", {"clim_npz": str(out_fnpz), "return_links": True})

  # 3E. Part E: Legacy monthly-mean source deduction vs exact 360-state deduction diagnostic
  for stag, ck in [("obs_2006_2025", "gswp3-era5__obsclim__2006_2025__e2025__b14__ret0")] + [
      (f"{m}_245_4665", f"{m}__ssp245__2046_2065__e2025__b14__ret0") for m in MODS
  ]:
    _add_scen(f"MONTHLY_DED__{stag}", "standard", {
        "clim_npz": str(CLIM_DIR / f"{ck}.npz"),
        "use_monthly_source_deduction": True,
        "return_links": (stag == "obs_2006_2025"),
    })

  print(
      f"[All Parts] Submitting {len(scen_tasks)} counterfactual/planning scenarios across {N_WORKERS} workers "
      f"(RSS={rgs._process_tree_rss_gb():.2f} GiB)...",
      flush=True,
  )

  agg: dict[str, dict[str, float]] = {}
  links_by_tag: dict[str, list[dict]] = {}

  import gc
  use_fork = sys.platform.startswith("linux")
  if use_fork:
    _init_lean_2025_worker()
    gc.collect()
    gc.freeze()

  ctx = mp.get_context("fork") if use_fork else mp.get_context("spawn")
  with ProcessPoolExecutor(
      max_workers=N_WORKERS,
      mp_context=ctx,
      initializer=_init_lean_2025_worker,
  ) as ex:
    futs = [ex.submit(_worker_scenario, st) for st in scen_tasks]
    for idx_s, fut in enumerate(as_completed(futs), 1):
      res = fut.result()
      tg = str(res["tag"])
      agg[tg] = {"P_ej": float(res["P_ej"]), "N_busd": float(res["N_busd"]), "co2_mt": float(res["co2_mt"])}
      if res["links"]:
        links_by_tag[tg] = res["links"]
      if idx_s == 1 or idx_s % 10 == 0 or idx_s == len(scen_tasks):
        print(
            f"  [Counterfactual scenario {idx_s}/{len(scen_tasks)}] {tg}: P={res['P_ej']:.3f} EJ, "
            f"RSS={rgs._process_tree_rss_gb():.2f} GiB ({time.perf_counter()-t_start:.1f}s)",
            flush=True,
        )

  if use_fork:
    gc.unfreeze()
  rgs._SOLV_INIT.clear()
  rgs._trim_memory()
  print(f"[All Parts] Scenario solver tasks finished in {time.perf_counter() - t_start:.1f}s!", flush=True)

  # --------------------------------------------------------------------------
  # POST-PROCESS PART A: R4 2050 Configurations Climate Effect (Obs vs Mod-Own)
  # --------------------------------------------------------------------------
  df_glob = pd.read_parquet(SCEN_DIR / "global_scenarios_summary.parquet").set_index("scenario_id")
  r1_obs_u = float(df_glob.loc["R1__obs__2006_2025", "U_ej_yr"])
  r1_mod_u_ratio = [float(df_glob.loc[f"R1__{m}__ssp245__2006_2025", "U_ej_yr"]) / r1_obs_u for m in MODS]

  part_a = {}
  for cname, kw in r4_cfgs:
    ret_tag = "ret1" if kw["use_retrofit"] else "ret0"
    obs_row = df_glob.loc[f"R4__{cname}__obs__obsclim__2006_2025"]
    fut_rows = [df_glob.loc[f"R4__{cname}__{m}__ssp245__2046_2065"] for m in MODS]

    D0_obs, T0_obs, P0_obs = float(obs_row["D_total_ej_yr"]), float(obs_row["U_ej_yr"]), float(obs_row["P_ej_yr"])
    D0_m = [d_tot_by_ck[f"{m}__ssp245__2006_2025__e2025__b14__{ret_tag}"] / 1e9 for m in MODS]
    T0_m = [T0_obs * rat for rat in r1_mod_u_ratio]
    P0_m = [agg[f"R4_MODBASE__{cname}__{m}"]["P_ej"] for m in MODS]
    N0_m = [agg[f"R4_MODBASE__{cname}__{m}"]["N_busd"] for m in MODS]
    C0_m = [agg[f"R4_MODBASE__{cname}__{m}"]["co2_mt"] for m in MODS]

    D1_m = [float(r["D_total_ej_yr"]) for r in fut_rows]
    T1_m = [float(r["U_ej_yr"]) for r in fut_rows]
    P1_m = [float(r["P_ej_yr"]) for r in fut_rows]
    N1_m = [float(r["net_benefit_billion_usd_yr"]) for r in fut_rows]
    C1_m = [float(r["co2_abated_mt_yr"]) for r in fut_rows]

    dp_mod = [100.0 * (p1 - p0) / p0 for p0, p1 in zip(P0_m, P1_m)]
    dp_abs_mod = [p1 - p0 for p0, p1 in zip(P0_m, P1_m)]
    t1_mod = [100.0 * (d1 - d0) / d0 for d0, d1 in zip(D0_m, D1_m)]
    t2_mod = [100.0 * ((t1 - t0) / t0 - (d1 - d0) / d0) for d0, d1, t0, t1 in zip(D0_m, D1_m, T0_m, T1_m)]
    t3_mod = [100.0 * ((p1 - p0) / p0 - (t1 - t0) / t0) for t0, t1, p0, p1 in zip(T0_m, T1_m, P0_m, P1_m)]

    dp_obs = [100.0 * (p1 - P0_obs) / P0_obs for p1 in P1_m]
    t1_obs = [100.0 * (d1 - D0_obs) / D0_obs for d1 in D1_m]
    t2_obs = [100.0 * ((t1 - T0_obs) / T0_obs - (d1 - D0_obs) / D0_obs) for d1, t1 in zip(D1_m, T1_m)]
    t3_obs = [100.0 * ((p1 - P0_obs) / P0_obs - (t1 - T0_obs) / T0_obs) for t1, p1 in zip(T1_m, P1_m)]

    part_a[cname] = {
        "obs_2006_2025": {
            "D_ej": D0_obs, "T_ej": T0_obs, "P_ej": P0_obs,
            "net_saving_busd": float(obs_row["net_benefit_billion_usd_yr"]),
            "co2_mt": float(obs_row["co2_abated_mt_yr"]),
        },
        "mod_2006_2025": {
            "D_mean_ej": float(np.mean(D0_m)),
            "T_mean_ej": float(np.mean(T0_m)),
            "P_mean_ej": float(np.mean(P0_m)),
            "P_min_max_ej": [float(np.min(P0_m)), float(np.max(P0_m))],
            "net_saving_mean_busd": float(np.mean(N0_m)),
            "co2_mean_mt": float(np.mean(C0_m)),
        },
        "fut_2046_2065": {
            "D_mean_ej": float(np.mean(D1_m)),
            "T_mean_ej": float(np.mean(T1_m)),
            "P_mean_ej": float(np.mean(P1_m)),
            "P_min_max_ej": [float(np.min(P1_m)), float(np.max(P1_m))],
            "net_saving_mean_busd": float(np.mean(N1_m)),
            "co2_mean_mt": float(np.mean(C1_m)),
        },
        "decomp_vs_mod_2006_2025": {
            "dP_ej_mean": float(np.mean(dp_abs_mod)),
            "dP_pct_mean": float(np.mean(dp_mod)),
            "dP_pct_min_max": [float(np.min(dp_mod)), float(np.max(dp_mod))],
            "t1_demand_pp": float(np.mean(t1_mod)),
            "t2_seasonal_pp": float(np.mean(t2_mod)),
            "t3_econ_viability_pp": float(np.mean(t3_mod)),
        },
        "decomp_vs_obs_2006_2025": {
            "dP_ej_mean": float(np.mean(P1_m) - P0_obs),
            "dP_pct_mean": float(np.mean(dp_obs)),
            "dP_pct_min_max": [float(np.min(dp_obs)), float(np.max(dp_obs))],
            "t1_demand_pp": float(np.mean(t1_obs)),
            "t2_seasonal_pp": float(np.mean(t2_obs)),
            "t3_econ_viability_pp": float(np.mean(t3_obs)),
        },
    }
  results["PartA_R4_2050_Climate_Effects"] = part_a

  # --------------------------------------------------------------------------
  # POST-PROCESS PART B: Fixed Building ASHP COP Mechanism Split
  # --------------------------------------------------------------------------
  df_locked_base = pd.read_parquet(SCEN_DIR / "R1__obs__2006_2025_selected_links.parquet")
  q_init_all = {
      str(r.stream_id): np.array([getattr(r, f"q3g_m{m+1:02d}_gj_d") for m in range(12)], dtype=np.float64)
      for r in df_sources.itertuples(index=False)
  }
  obs_base_npz = str(CLIM_DIR / "gswp3-era5__obsclim__2006_2025__e2025__b14__ret0.npz")
  zm_obs_0, _, is_ashp_0 = _load_full_zm_dmq_custom([(obs_base_npz, 1.0)], df_zones_valid, reg_valid)

  locked_full_rows, locked_fixcop_rows = [], []
  ashp_zone_full_nb, ashp_zone_fix_nb, gb_zone_full_nb, gb_zone_fix_nb = [], [], [], []
  ashp_zone_cap_full, ashp_zone_cap_fix, gb_zone_cap_full, gb_zone_cap_fix = [], [], [], []
  switch_to_ashp_count = []

  for m in MODS:
    near_npz = str(CLIM_DIR / f"{m}__ssp245__2026_2045__e2025__b14__ret0.npz")
    fut_npz = str(CLIM_DIR / f"{m}__ssp245__2046_2065__e2025__b14__ret0.npz")
    zm_near_full, dmq_n, _ = _load_full_zm_dmq_custom([(near_npz, 1.0)], df_zones_valid, reg_valid)
    zm_near_fix, _, _ = _load_full_zm_dmq_custom([(near_npz, 1.0)], df_zones_valid, reg_valid, fixed_cop_base_npz=obs_base_npz)
    zm_full, dmq_f, is_ashp_f = _load_full_zm_dmq_custom([(fut_npz, 1.0)], df_zones_valid, reg_valid)
    zm_fix, _, _ = _load_full_zm_dmq_custom([(fut_npz, 1.0)], df_zones_valid, reg_valid, fixed_cop_base_npz=obs_base_npz)

    res_near_full = evaluate_locked_portfolio(df_locked_base, q_init_all, dmq_n, zm_near_full, zm_obs_0, DISCOUNT_RATE_CENTRAL)
    res_near_fix = evaluate_locked_portfolio(df_locked_base, q_init_all, dmq_n, zm_near_fix, zm_obs_0, DISCOUNT_RATE_CENTRAL)
    res_full = evaluate_locked_portfolio(df_locked_base, q_init_all, dmq_f, zm_full, zm_obs_0, DISCOUNT_RATE_CENTRAL)
    res_fix = evaluate_locked_portfolio(df_locked_base, q_init_all, dmq_f, zm_fix, zm_obs_0, DISCOUNT_RATE_CENTRAL)

    for dr_n, dr, store in [(res_near_full, res_full, locked_full_rows), (res_near_fix, res_fix, locked_fixcop_rows)]:
      lm = dr["is_loss_making_fut"]
      npv_40 = pv_20 * dr_n["net_benefit_fut_usd_yr"].to_numpy() + (pv_40 - pv_20) * dr["net_benefit_fut_usd_yr"].to_numpy()
      npv_neg = npv_40 < 0.0
      store.append({
          "model": m,
          "P_0_ej": float(dr["P_0_gj_yr"].sum() / 1e9),
          "P_fut_ej": float(dr["P_fut_gj_yr"].sum() / 1e9),
          "NB_0_busd": float(dr["net_benefit_0_usd_yr"].sum() / 1e9),
          "NB_fut_busd": float(dr["net_benefit_fut_usd_yr"].sum() / 1e9),
          "n_loss": int(lm.sum()),
          "loss_share_pct": 100.0 * float(lm.mean()),
          "exposed_capex_busd": float(dr.loc[lm, "total_sunk_capex_usd"].sum() / 1e9),
          "n_lifetime_npv_neg": int(npv_neg.sum()),
          "lifetime_npv_neg_capex_busd": float(dr.loc[npv_neg, "total_sunk_capex_usd"].sum() / 1e9),
          "foregone_co2_mt": float(dr["foregone_co2_abatement_t_yr"].sum() / 1e6),
          "closure_risk_co2_mt": float(dr["stranded_reversion_co2_t_yr"].sum() / 1e6),
      })

    z_s = res_full["zone_id"].astype(str)
    m_ashp_any = z_s.map(lambda z: bool(is_ashp_0.get(z, False) or is_ashp_f.get(z, False))).to_numpy()
    m_switch = z_s.map(lambda z: bool((not is_ashp_0.get(z, False)) and is_ashp_f.get(z, False))).to_numpy()
    switch_to_ashp_count.append(int(m_switch.sum()))

    nb0_a = float(res_full.loc[m_ashp_any, "net_benefit_0_usd_yr"].sum() / 1e9)
    ashp_zone_full_nb.append(100.0 * (float(res_full.loc[m_ashp_any, "net_benefit_fut_usd_yr"].sum() / 1e9) - nb0_a) / max(1e-9, nb0_a))
    ashp_zone_fix_nb.append(100.0 * (float(res_fix.loc[m_ashp_any, "net_benefit_fut_usd_yr"].sum() / 1e9) - nb0_a) / max(1e-9, nb0_a))
    ashp_zone_cap_full.append(float(res_full.loc[m_ashp_any & res_full["is_loss_making_fut"], "total_sunk_capex_usd"].sum() / 1e9))
    ashp_zone_cap_fix.append(float(res_fix.loc[m_ashp_any & res_fix["is_loss_making_fut"], "total_sunk_capex_usd"].sum() / 1e9))

    nb0_g = float(res_full.loc[~m_ashp_any, "net_benefit_0_usd_yr"].sum() / 1e9)
    gb_zone_full_nb.append(100.0 * (float(res_full.loc[~m_ashp_any, "net_benefit_fut_usd_yr"].sum() / 1e9) - nb0_g) / max(1e-9, nb0_g))
    gb_zone_fix_nb.append(100.0 * (float(res_fix.loc[~m_ashp_any, "net_benefit_fut_usd_yr"].sum() / 1e9) - nb0_g) / max(1e-9, nb0_g))
    gb_zone_cap_full.append(float(res_full.loc[(~m_ashp_any) & res_full["is_loss_making_fut"], "total_sunk_capex_usd"].sum() / 1e9))
    gb_zone_cap_fix.append(float(res_fix.loc[(~m_ashp_any) & res_fix["is_loss_making_fut"], "total_sunk_capex_usd"].sum() / 1e9))

  df_lf = pd.DataFrame(locked_full_rows)
  df_lx = pd.DataFrame(locked_fixcop_rows)

  r1_p0_m = [float(df_glob.loc[f"R1__{m}__ssp245__2006_2025", "P_ej_yr"]) for m in MODS]
  r1_pf_full = [float(df_glob.loc[f"R1__{m}__ssp245__2046_2065", "P_ej_yr"]) for m in MODS]
  r1_pf_fix = [agg[f"FIXCOP_R1__{m}"]["P_ej"] for m in MODS]
  r1_t0_m = [float(df_glob.loc[f"R1__{m}__ssp245__2006_2025", "U_ej_yr"]) for m in MODS]
  r1_tf_m = [float(df_glob.loc[f"R1__{m}__ssp245__2046_2065", "U_ej_yr"]) for m in MODS]

  s5_t0_obs = float(df_glob.loc["R4__S5_combined_all__obs__obsclim__2006_2025", "U_ej_yr"])
  s5_p0_m = [agg[f"R4_MODBASE__S5_combined_all__{m}"]["P_ej"] for m in MODS]
  s5_pf_full = [float(df_glob.loc[f"R4__S5_combined_all__{m}__ssp245__2046_2065", "P_ej_yr"]) for m in MODS]
  s5_pf_fix = [agg[f"FIXCOP_S5__{m}"]["P_ej"] for m in MODS]
  s5_t0_m = [s5_t0_obs * rat for rat in r1_mod_u_ratio]
  s5_tf_m = [float(df_glob.loc[f"R4__S5_combined_all__{m}__ssp245__2046_2065", "U_ej_yr"]) for m in MODS]

  results["PartB_Fixed_ASHP_COP_Split"] = {
      "locked_networks_2046_2065": {
          "NB0_busd": float(df_lf["NB_0_busd"].iloc[0]),
          "NB_fut_full_mean_busd": float(df_lf["NB_fut_busd"].mean()),
          "NB_fut_fixcop_mean_busd": float(df_lx["NB_fut_busd"].mean()),
          "dNB_full_pct": float(100.0 * (df_lf["NB_fut_busd"].mean() - df_lf["NB_0_busd"].iloc[0]) / df_lf["NB_0_busd"].iloc[0]),
          "dNB_fixcop_pct": float(100.0 * (df_lx["NB_fut_busd"].mean() - df_lx["NB_0_busd"].iloc[0]) / df_lx["NB_0_busd"].iloc[0]),
          "n_loss_full_mean": float(df_lf["n_loss"].mean()),
          "n_loss_fixcop_mean": float(df_lx["n_loss"].mean()),
          "loss_share_full_pct": float(df_lf["loss_share_pct"].mean()),
          "loss_share_fixcop_pct": float(df_lx["loss_share_pct"].mean()),
          "exposed_capex_full_busd": float(df_lf["exposed_capex_busd"].mean()),
          "exposed_capex_fixcop_busd": float(df_lx["exposed_capex_busd"].mean()),
          "n_lifetime_npv_neg_full_mean": float(df_lf["n_lifetime_npv_neg"].mean()),
          "n_lifetime_npv_neg_fixcop_mean": float(df_lx["n_lifetime_npv_neg"].mean()),
          "lifetime_npv_neg_capex_full_busd": float(df_lf["lifetime_npv_neg_capex_busd"].mean()),
          "lifetime_npv_neg_capex_fixcop_busd": float(df_lx["lifetime_npv_neg_capex_busd"].mean()),
          "closure_co2_full_mt": float(df_lf["closure_risk_co2_mt"].mean()),
          "closure_co2_fixcop_mt": float(df_lx["closure_risk_co2_mt"].mean()),
          "ashp_competing_subset": {
              "n_links_switching_gb_to_ashp_mean": float(np.mean(switch_to_ashp_count)),
              "dNB_full_pct": float(np.mean(ashp_zone_full_nb)),
              "dNB_fixcop_pct": float(np.mean(ashp_zone_fix_nb)),
              "exposed_capex_full_busd": float(np.mean(ashp_zone_cap_full)),
              "exposed_capex_fixcop_busd": float(np.mean(ashp_zone_cap_fix)),
          },
          "gas_boiler_only_subset": {
              "dNB_full_pct": float(np.mean(gb_zone_full_nb)),
              "dNB_fixcop_pct": float(np.mean(gb_zone_fix_nb)),
              "exposed_capex_full_busd": float(np.mean(gb_zone_cap_full)),
              "exposed_capex_fixcop_busd": float(np.mean(gb_zone_cap_fix)),
          },
      },
      "resized_2025_system_2046_2065": {
          "P_2006_mean_ej": float(np.mean(r1_p0_m)),
          "P_2046_full_mean_ej": float(np.mean(r1_pf_full)),
          "P_2046_fixcop_mean_ej": float(np.mean(r1_pf_fix)),
          "dP_full_pct": float(np.mean([100.0 * (pf - p0) / p0 for p0, pf in zip(r1_p0_m, r1_pf_full)])),
          "dP_fixcop_pct": float(np.mean([100.0 * (pf - p0) / p0 for p0, pf in zip(r1_p0_m, r1_pf_fix)])),
          "t3_econ_full_pp": float(np.mean([100.0 * ((pf - p0) / p0 - (tf - t0) / t0) for p0, pf, t0, tf in zip(r1_p0_m, r1_pf_full, r1_t0_m, r1_tf_m)])),
          "t3_econ_fixcop_pp": float(np.mean([100.0 * ((pf - p0) / p0 - (tf - t0) / t0) for p0, pf, t0, tf in zip(r1_p0_m, r1_pf_fix, r1_t0_m, r1_tf_m)])),
      },
      "resized_2050_zero_fossil_S5_2046_2065": {
          "P_2006_mean_ej": float(np.mean(s5_p0_m)),
          "P_2046_full_mean_ej": float(np.mean(s5_pf_full)),
          "P_2046_fixcop_mean_ej": float(np.mean(s5_pf_fix)),
          "dP_full_pct": float(np.mean([100.0 * (pf - p0) / p0 for p0, pf in zip(s5_p0_m, s5_pf_full)])),
          "dP_fixcop_pct": float(np.mean([100.0 * (pf - p0) / p0 for p0, pf in zip(s5_p0_m, s5_pf_fix)])),
          "t3_econ_full_pp": float(np.mean([100.0 * ((pf - p0) / p0 - (tf - t0) / t0) for p0, pf, t0, tf in zip(s5_p0_m, s5_pf_full, s5_t0_m, s5_tf_m)])),
          "t3_econ_fixcop_pp": float(np.mean([100.0 * ((pf - p0) / p0 - (tf - t0) / t0) for p0, pf, t0, tf in zip(s5_p0_m, s5_pf_fix, s5_t0_m, s5_tf_m)])),
      },
  }

  # --------------------------------------------------------------------------
  # POST-PROCESS PART C: All Three CHP Heat-Accounting Conventions
  # --------------------------------------------------------------------------
  def _summarize_chp(prefix: str) -> dict[str, object]:
    p_1946 = agg[f"{prefix}__obs_1946_1965"]["P_ej"]
    p_2006 = agg[f"{prefix}__obs_2006_2025"]["P_ej"]
    n_2006 = agg[f"{prefix}__obs_2006_2025"]["N_busd"]
    c_1946 = agg[f"{prefix}__obs_1946_1965"]["co2_mt"]
    c_2006 = agg[f"{prefix}__obs_2006_2025"]["co2_mt"]

    p_245_0 = [agg[f"{prefix}__{m}_245_0625"]["P_ej"] for m in MODS]
    p_245_f = [agg[f"{prefix}__{m}_245_4665"]["P_ej"] for m in MODS]
    n_245_f = [agg[f"{prefix}__{m}_245_4665"]["N_busd"] for m in MODS]
    c_245_0 = [agg[f"{prefix}__{m}_245_0625"]["co2_mt"] for m in MODS]
    c_245_f = [agg[f"{prefix}__{m}_245_4665"]["co2_mt"] for m in MODS]

    p_370_0 = [agg[f"{prefix}__{m}_370_0625"]["P_ej"] for m in MODS]
    p_370_f = [agg[f"{prefix}__{m}_370_8100"]["P_ej"] for m in MODS]
    c_370_0 = [agg[f"{prefix}__{m}_370_0625"]["co2_mt"] for m in MODS]
    c_370_f = [agg[f"{prefix}__{m}_370_8100"]["co2_mt"] for m in MODS]

    return {
        "P_1946_ej": p_1946,
        "P_2006_ej": p_2006,
        "NB_2006_busd": n_2006,
        "co2_1946_mt": c_1946,
        "co2_2006_mt": c_2006,
        "P_2046_ssp245_mean_ej": float(np.mean(p_245_f)),
        "P_2046_ssp245_min_max_ej": [float(np.min(p_245_f)), float(np.max(p_245_f))],
        "dP_2046_vs_obs_pct": float(np.mean([100.0 * (x - p_2006) / max(1e-9, p_2006) for x in p_245_f])),
        "dP_2046_vs_mod0_pct": float(np.mean([100.0 * (a - b) / max(1e-9, b) for a, b in zip(p_245_f, p_245_0)])),
        "NB_2046_ssp245_mean_busd": float(np.mean(n_245_f)),
        "co2_2046_ssp245_mean_mt": float(np.mean(c_245_f)),
        "dco2_2046_vs_obs_pct": float(np.mean([100.0 * (x - c_2006) / max(1e-9, c_2006) for x in c_245_f])),
        "dco2_2046_vs_mod0_pct": float(np.mean([100.0 * (a - b) / max(1e-9, b) for a, b in zip(c_245_f, c_245_0)])),
        "P_2081_ssp370_mean_ej": float(np.mean(p_370_f)),
        "dP_2081_ssp370_vs_mod0_pct": float(np.mean([100.0 * (a - b) / max(1e-9, b) for a, b in zip(p_370_f, p_370_0)])),
        "co2_2081_ssp370_mean_mt": float(np.mean(c_370_f)),
        "dco2_2081_ssp370_vs_mod0_pct": float(np.mean([100.0 * (a - b) / max(1e-9, b) for a, b in zip(c_370_f, c_370_0)])),
    }

  results["PartC_CHP_Fuel_Allocation"] = {
      "power_loss_015": _summarize_chp("CHP_PWRLOSS"),
      "fuel_cost_and_co2": _summarize_chp("CHP_FUELBOTH"),
      "fuel_co2_only": _summarize_chp("CHP_FUELCO2"),
  }

  # --------------------------------------------------------------------------
  # POST-PROCESS PART D: Forward-Climate vs Historical-Climate Planning
  # --------------------------------------------------------------------------
  fut_cache = {}
  for m in MODS:
    for wid in ("2026_2045", "2046_2065"):
      p_npz = str(CLIM_DIR / f"{m}__ssp245__{wid}__e2025__b14__ret0.npz")
      zm_w, dmq_w, _ = _load_full_zm_dmq_custom([(p_npz, 1.0)], df_zones_valid, reg_valid)
      fut_cache[(m, wid)] = (zm_w, dmq_w)

  hist_eval = {}
  for m in MODS:
    zm_n, dmq_n = fut_cache[(m, "2026_2045")]
    zm_m, dmq_m = fut_cache[(m, "2046_2065")]
    dr_hn = evaluate_portfolio_general(df_locked_base, q_init_all, dmq_n, zm_n, zm_obs_0, DISCOUNT_RATE_CENTRAL, True)
    dr_hm = evaluate_portfolio_general(df_locked_base, q_init_all, dmq_m, zm_m, zm_obs_0, DISCOUNT_RATE_CENTRAL, True)
    npv_40_h = pv_20 * dr_hn["net_benefit_fut_usd_yr"].to_numpy() + (pv_40 - pv_20) * dr_hm["net_benefit_fut_usd_yr"].to_numpy()
    npv_neg_h = npv_40_h < 0.0
    for wid, dr_h in [("2026_2045", dr_hn), ("2046_2065", dr_hm)]:
      hist_eval[(m, wid)] = {
          "n_links": len(dr_h),
          "capex_busd": float(dr_h["total_sunk_capex_usd"].sum() / 1e9),
          "NB_busd_yr": float(dr_h["net_benefit_fut_usd_yr"].sum() / 1e9),
          "P_ej_yr": float(dr_h["P_fut_gj_yr"].sum() / 1e9),
          "co2_mt_yr": float(dr_h["co2_abated_fut_t_yr"].sum() / 1e6),
          "n_loss": int(dr_h["is_loss_making_fut"].sum()),
          "n_op_loss": int(dr_h["is_operating_loss_fut"].sum()),
          "exposed_capex_busd": float(dr_h.loc[dr_h["is_loss_making_fut"], "total_sunk_capex_usd"].sum() / 1e9),
          "n_lifetime_npv_neg": int(npv_neg_h.sum()),
          "lifetime_npv_neg_capex_busd": float(dr_hm.loc[npv_neg_h, "total_sunk_capex_usd"].sum() / 1e9),
      }

  part_d = {
      "pv_factor_yr1_20": pv_20,
      "pv_factor_yr21_40": pv_40 - pv_20,
      "pv_factor_total_40yr": pv_40,
      "historical_plan": {
          "n_links_mean": float(np.mean([hist_eval[(m, "2046_2065")]["n_links"] for m in MODS])),
          "capex_busd": float(np.mean([hist_eval[(m, "2026_2045")]["capex_busd"] for m in MODS])),
          "P_2026_2045_ej_yr": float(np.mean([hist_eval[(m, "2026_2045")]["P_ej_yr"] for m in MODS])),
          "P_2046_2065_ej_yr": float(np.mean([hist_eval[(m, "2046_2065")]["P_ej_yr"] for m in MODS])),
          "NB_2026_2045_busd_yr": float(np.mean([hist_eval[(m, "2026_2045")]["NB_busd_yr"] for m in MODS])),
          "NB_2046_2065_busd_yr": float(np.mean([hist_eval[(m, "2046_2065")]["NB_busd_yr"] for m in MODS])),
          "NPV_40yr_7pct_busd": float(np.mean([
              pv_20 * hist_eval[(m, "2026_2045")]["NB_busd_yr"] + (pv_40 - pv_20) * hist_eval[(m, "2046_2065")]["NB_busd_yr"]
              for m in MODS
          ])),
          "cum_40yr_undisc_busd": float(np.mean([
              20.0 * hist_eval[(m, "2026_2045")]["NB_busd_yr"] + 20.0 * hist_eval[(m, "2046_2065")]["NB_busd_yr"]
              for m in MODS
          ])),
          "exposed_capex_2046_2065_busd": float(np.mean([hist_eval[(m, "2046_2065")]["exposed_capex_busd"] for m in MODS])),
          "n_loss_2046_2065": float(np.mean([hist_eval[(m, "2046_2065")]["n_loss"] for m in MODS])),
          "n_operating_loss_2046_2065": float(np.mean([hist_eval[(m, "2046_2065")]["n_op_loss"] for m in MODS])),
          "n_lifetime_npv_neg": float(np.mean([hist_eval[(m, "2046_2065")]["n_lifetime_npv_neg"] for m in MODS])),
          "lifetime_npv_neg_capex_busd": float(np.mean([hist_eval[(m, "2046_2065")]["lifetime_npv_neg_capex_busd"] for m in MODS])),
          "co2_2026_2045_mt_yr": float(np.mean([hist_eval[(m, "2026_2045")]["co2_mt_yr"] for m in MODS])),
          "co2_2046_2065_mt_yr": float(np.mean([hist_eval[(m, "2046_2065")]["co2_mt_yr"] for m in MODS])),
          "cum_40yr_co2_mt": float(np.mean([
              20.0 * hist_eval[(m, "2026_2045")]["co2_mt_yr"] + 20.0 * hist_eval[(m, "2046_2065")]["co2_mt_yr"]
              for m in MODS
          ])),
      },
  }

  hist_link_c = {
      (str(r.stream_id), str(r.zone_id)): float(r.c_mw)
      for r in df_locked_base.itertuples(index=False)
  }

  for scheme in ("FWD_LOO_PV", "FWD_LOO_MID", "FWD_PERF_PV"):
    s_rows = []
    for m in MODS:
      ftag = f"{scheme}__{m}"
      df_fwd = pd.DataFrame(links_by_tag[ftag])
      zm_des, _, _ = _load_full_zm_dmq_custom(fwd_design_weights[ftag], df_zones_valid, reg_valid)

      zm_near, dmq_near = fut_cache[(m, "2026_2045")]
      zm_mid, dmq_mid = fut_cache[(m, "2046_2065")]
      dr_near = evaluate_portfolio_general(df_fwd, q_init_all, dmq_near, zm_near, zm_des, DISCOUNT_RATE_CENTRAL, True)
      dr_mid = evaluate_portfolio_general(df_fwd, q_init_all, dmq_mid, zm_mid, zm_des, DISCOUNT_RATE_CENTRAL, True)

      fwd_link_c = {
          (str(r.stream_id), str(r.zone_id)): float(r.c_mw)
          for r in dr_mid.itertuples(index=False)
      }
      n_dropped = sum(1 for k in hist_link_c if k not in fwd_link_c)
      n_downsized = sum(1 for k, c0 in hist_link_c.items() if k in fwd_link_c and fwd_link_c[k] < c0 - 1e-3)

      nb_n = float(dr_near["net_benefit_fut_usd_yr"].sum() / 1e9)
      nb_m = float(dr_mid["net_benefit_fut_usd_yr"].sum() / 1e9)
      p_n = float(dr_near["P_fut_gj_yr"].sum() / 1e9)
      p_m = float(dr_mid["P_fut_gj_yr"].sum() / 1e9)
      c_n = float(dr_near["co2_abated_fut_t_yr"].sum() / 1e6)
      c_m = float(dr_mid["co2_abated_fut_t_yr"].sum() / 1e6)

      h_n = hist_eval[(m, "2026_2045")]["NB_busd_yr"]
      h_m = hist_eval[(m, "2046_2065")]["NB_busd_yr"]
      hp_n = hist_eval[(m, "2026_2045")]["P_ej_yr"]
      hp_m = hist_eval[(m, "2046_2065")]["P_ej_yr"]
      hc_n = hist_eval[(m, "2026_2045")]["co2_mt_yr"]
      hc_m = hist_eval[(m, "2046_2065")]["co2_mt_yr"]

      npv_40_f = pv_20 * dr_near["net_benefit_fut_usd_yr"].to_numpy() + (pv_40 - pv_20) * dr_mid["net_benefit_fut_usd_yr"].to_numpy()
      npv_neg_f = npv_40_f < 0.0

      s_rows.append({
          "model": m,
          "n_links": len(dr_mid),
          "n_dropped_vs_hist": n_dropped,
          "n_downsized_vs_hist": n_downsized,
          "capex_busd": float(dr_near["total_sunk_capex_usd"].sum() / 1e9),
          "capex_saved_vs_hist_busd": hist_eval[(m, "2026_2045")]["capex_busd"] - float(dr_near["total_sunk_capex_usd"].sum() / 1e9),
          "P_2026_2045_ej_yr": p_n,
          "P_2046_2065_ej_yr": p_m,
          "dP_2026_2045_vs_hist_ej_yr": p_n - hp_n,
          "dP_2046_2065_vs_hist_ej_yr": p_m - hp_m,
          "dP_2046_2065_vs_hist_pct": 100.0 * (p_m - hp_m) / max(1e-9, hp_m),
          "NB_2026_2045_busd_yr": nb_n,
          "NB_2046_2065_busd_yr": nb_m,
          "dNB_2026_2045_busd_yr": nb_n - h_n,
          "dNB_2046_2065_busd_yr": nb_m - h_m,
          "avoidable_NPV_7pct_busd": pv_20 * (nb_n - h_n) + (pv_40 - pv_20) * (nb_m - h_m),
          "avoidable_cum_40yr_undisc_busd": 20.0 * (nb_n - h_n) + 20.0 * (nb_m - h_m),
          "exposed_capex_2046_2065_busd": float(dr_mid.loc[dr_mid["is_loss_making_fut"], "total_sunk_capex_usd"].sum() / 1e9),
          "n_loss_2046_2065": int(dr_mid["is_loss_making_fut"].sum()),
          "n_operating_loss_2046_2065": int(dr_mid["is_operating_loss_fut"].sum()),
          "n_lifetime_npv_neg": int(npv_neg_f.sum()),
          "lifetime_npv_neg_capex_busd": float(dr_mid.loc[npv_neg_f, "total_sunk_capex_usd"].sum() / 1e9),
          "co2_2026_2045_mt_yr": c_n,
          "co2_2046_2065_mt_yr": c_m,
          "dco2_2026_2045_vs_hist_mt_yr": c_n - hc_n,
          "dco2_2046_2065_vs_hist_mt_yr": c_m - hc_m,
          "dco2_2046_2065_vs_hist_pct": 100.0 * (c_m - hc_m) / max(1e-9, hc_m),
          "cum_40yr_co2_mt": 20.0 * c_n + 20.0 * c_m,
          "dcum_40yr_co2_vs_hist_mt": 20.0 * (c_n - hc_n) + 20.0 * (c_m - hc_m),
          "closure_risk_co2_2046_2065_mt_yr": float(dr_mid["closure_risk_co2_t_yr"].sum() / 1e6),
      })
    df_sr = pd.DataFrame(s_rows)
    part_d[scheme] = {col: float(df_sr[col].mean()) for col in df_sr.columns if col != "model"}
    part_d[scheme]["avoidable_NPV_min_max_busd"] = [float(df_sr["avoidable_NPV_7pct_busd"].min()), float(df_sr["avoidable_NPV_7pct_busd"].max())]
    part_d[scheme]["avoidable_cum_40yr_min_max_busd"] = [float(df_sr["avoidable_cum_40yr_undisc_busd"].min()), float(df_sr["avoidable_cum_40yr_undisc_busd"].max())]
  results["PartD_Avoidable_Planning_Cost"] = part_d

  # --------------------------------------------------------------------------
  # POST-PROCESS PART E: Exact 360-State vs Legacy Monthly-Mean Source Deduction
  # --------------------------------------------------------------------------
  p0_360 = float(df_glob.loc["R1__obs__2006_2025", "P_ej_yr"])
  pf_360 = float(np.mean(r1_pf_full))
  p0_mon = agg["MONTHLY_DED__obs_2006_2025"]["P_ej"]
  pf_mon = float(np.mean([agg[f"MONTHLY_DED__{m}_245_4665"]["P_ej"] for m in MODS]))
  results["PartE_Check5_12x30_Diagnostic"] = {
      "P0_360state_exact_ej": p0_360,
      "P0_monthly_deduction_ej": p0_mon,
      "P0_overuse_eliminated_pct": 100.0 * (p0_mon - p0_360) / max(1e-9, p0_mon),
      "P_2046_360state_mean_ej": pf_360,
      "P_2046_monthly_mean_ej": pf_mon,
      "dP_vs_obs_360state_pct": 100.0 * (pf_360 - p0_360) / max(1e-9, p0_360),
      "dP_vs_obs_monthly_pct": 100.0 * (pf_mon - p0_mon) / max(1e-9, p0_mon),
      "n_links_2006_360state": len(df_locked_base),
      "n_links_2006_monthly": len(links_by_tag.get("MONTHLY_DED__obs_2006_2025", [])),
  }

  results["total_elapsed_sec"] = round(time.perf_counter() - t_start, 2)
  out_json = SCEN_DIR / "counterfactual_and_planning_summary.json"
  out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
  print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
  main()
