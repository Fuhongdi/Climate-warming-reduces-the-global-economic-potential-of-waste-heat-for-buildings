"""Evaluate fixed 40-year district heating networks sized under present climate across future warming windows."""
from __future__ import annotations

import math
import numpy as np
import pandas as pd

from code.common.parameters import (
    DH_SUBSTATION_2025,
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
from code.common.paths import GJ_PER_MW_DAY, GJ_PER_MWH
from code.matching.solver import DAYS_MONTH, TechConfig, crf, get_upgrade_params


def evaluate_locked_portfolio(
    df_locked_links: pd.DataFrame,
    q_init_m_gj_d: dict[str, np.ndarray],
    zone_dmq_fut: dict[str, np.ndarray],
    zone_meta_fut: dict[str, dict[str, float]],
    zone_meta_base: dict[str, dict[str, float]],
    discount_rate: float = 0.07,
) -> pd.DataFrame:
  """Simulate locked baseline connections (in original selection order) under a future climate window."""
  if len(df_locked_links) == 0:
    return pd.DataFrame()

  cfg = TechConfig(discount_rate=discount_rate, is_4gdh=False, use_2050_dh_tech=False)
  a_pipe = crf(discount_rate, PIPE_LIFETIME_YR)
  a_eq = crf(discount_rate, STATION_LIFETIME_YR)
  a_dist = crf(discount_rate, 40.0)
  a_sub = crf(discount_rate, DH_SUBSTATION_2025.lifetime_yr)

  q_rem = {
      sid: (arr.reshape(12, 30).copy() if arr.size == 360 else np.broadcast_to(arr.reshape(12)[:, None], (12, 30)).copy())
      for sid, arr in q_init_m_gj_d.items()
  }
  out_rows = []

  for r in df_locked_links.itertuples(index=False):
    sid = str(r.stream_id)
    zid = str(r.zone_id)
    if zid not in zone_meta_fut or sid not in q_rem:
      continue

    zm_f = zone_meta_fut[zid]
    zm_0 = zone_meta_base[zid]
    dmq_f = zone_dmq_fut[zid]
    n_q = dmq_f.shape[1]
    w_mq = DAYS_MONTH[:, None] / float(n_q)

    alpha = float(r.alpha)
    c_mw = float(r.c_mw)
    c_src_mw = float(r.c_src_mw)
    c_pk_mw = float(r.c_pk_mw)
    ell_km = float(r.ell_km)
    eta_tot = float(r.eta_tot)
    up_mode = str(r.upgrade_mode)
    has_gas = bool(zm_0["has_gas"])
    cap_civ = float(zm_0["cap_scale_civil"])
    cap_eq = float(zm_0["cap_scale_equip"])
    p_el = float(zm_0["p_elec_usd_per_mwh"])
    p_ng = float(zm_0["p_gas_usd_per_gj"])
    grid_ef = float(zm_0["grid_ef_tco2_per_mwh"])

    kappa, cop, eq_capex_eur, eq_fom_eur, eq_vom_eur = get_upgrade_params(up_mode, cfg)
    pk_sp = PEAKING_GAS_BOILER_2025 if has_gas else PEAKING_ELEC_BOILER_2025
    a_pk = crf(discount_rate, pk_sp.lifetime_yr)

    # Sunk / locked annualized fixed costs (transmission + source upgrade + peaking + urban dist & substation)
    pipe_cap_usd = PIPE_CAPEX_COEF_EUR_PER_KM * (c_src_mw ** PIPE_CAPEX_EXPONENT) * ell_km * USD_PER_EUR_2025 * cap_civ
    pipe_fom_usd = PIPE_FIXED_OM_EUR_PER_MW_KM_YR * c_src_mw * ell_km * USD_PER_EUR_2025 * cap_civ
    eq_cap_usd = eq_capex_eur * c_src_mw * USD_PER_EUR_2025 * cap_eq
    eq_fom_usd = eq_fom_eur * c_src_mw * USD_PER_EUR_2025 * cap_eq
    pk_cap_usd = pk_sp.capex_eur_per_mw_h * c_pk_mw * USD_PER_EUR_2025 * cap_eq
    pk_fom_usd = pk_sp.fixed_om_eur_per_mw_h_yr * c_pk_mw * USD_PER_EUR_2025 * cap_eq

    # Baseline urban distribution + substation fixed annual cost locked at baseline sizing:
    locked_dist_sub_ann_usd = alpha * float(zm_0["ann_gj"]) * float(zm_0["c_net_usd_per_gj"])
    locked_dist_sub_capex_usd = alpha * float(zm_0["dist_sub_capex_usd_full"])
    total_sunk_capex_usd = pipe_cap_usd + eq_cap_usd + pk_cap_usd + locked_dist_sub_capex_usd

    locked_fixed_ann_usd = (
        locked_dist_sub_ann_usd
        + a_pipe * pipe_cap_usd + pipe_fom_usd
        + a_eq * eq_cap_usd + eq_fom_usd
        + a_pk * pk_cap_usd + pk_fom_usd
    )

    # Future daily dispatch on locked capacity (c_mw, alpha) across all 360 (month, quantile) states
    c_d_gj = c_mw * GJ_PER_MW_DAY
    src_avail_mq = kappa * eta_tot * q_rem[sid]
    sub_dmq_f = alpha * dmq_f
    h_mq_f = np.minimum(c_d_gj, np.minimum(src_avail_mq, sub_dmq_f))
    H_deliv_f_gj = float(np.sum(h_mq_f * w_mq))

    # Update remaining source waste heat across all 360 states for subsequent locked links sharing source sid
    q_used_mq = h_mq_f / max(1e-9, kappa * eta_tot)
    q_rem[sid] = np.maximum(0.0, q_rem[sid] - q_used_mq)

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

    # Ongoing operating cashflow excluding sunk capital recovery charges
    dist_sub_fom_el_0 = alpha * float(zm_0.get("dist_sub_fom_el_usd_full", 0.0))
    ongoing_cash_cost_f_usd = dist_sub_fom_el_0 + pipe_fom_usd + eq_fom_usd + pk_fom_usd + var_cost_f_usd
    operating_margin_excl_sunk_f_usd = avoided_ind_f_usd - ongoing_cash_cost_f_usd

    sub_D_0_gj = float(r.sub_demand_gj_yr)
    c_ind_0_usd_gj = float(zm_0["c_ind_usd_per_gj"])
    c_ind_f_usd_gj = float(zm_f["c_ind_usd_per_gj"])
    lcoh_0_usd_gj = float(r.tot_dh_cost_usd_yr) / max(1.0, sub_D_0_gj)
    lcoh_f_usd_gj = tot_dh_cost_f_usd / max(1.0, sub_D_f_gj)

    dh_co2_f_t = tot_el_mwh * grid_ef + pk_co2_t
    co2_abated_f_t = sub_D_f_gj * float(zm_f["cf_co2_t_per_gj"]) - dh_co2_f_t
    co2_abated_0_t = float(r.co2_abated_t_yr)

    out_rows.append({
        "iso3": str(r.iso3),
        "region_id": str(r.region_id),
        "stream_id": sid,
        "zone_id": zid,
        "sector": str(r.sector),
        "alpha": alpha,
        "c_mw": c_mw,
        "total_sunk_capex_usd": total_sunk_capex_usd,
        "sub_D_0_gj_yr": sub_D_0_gj,
        "sub_D_fut_gj_yr": sub_D_f_gj,
        "P_0_gj_yr": float(r.P_net_waste_heat_gj_yr),
        "P_fut_gj_yr": H_deliv_f_gj / kappa,
        "c_ind_0_usd_per_gj": c_ind_0_usd_gj,
        "c_ind_fut_usd_per_gj": c_ind_f_usd_gj,
        "lcoh_0_usd_per_gj": lcoh_0_usd_gj,
        "lcoh_fut_usd_per_gj": lcoh_f_usd_gj,
        "delta_lcoh_usd_per_gj": lcoh_f_usd_gj - lcoh_0_usd_gj,
        "net_benefit_0_usd_yr": float(r.net_benefit_usd_yr),
        "net_benefit_fut_usd_yr": net_benefit_f_usd,
        "operating_margin_excl_sunk_fut_usd_yr": operating_margin_excl_sunk_f_usd,
        "is_loss_making_fut": bool(net_benefit_f_usd < 0.0),
        "is_operating_loss_fut": bool(operating_margin_excl_sunk_f_usd < 0.0),
        "co2_abated_0_t_yr": co2_abated_0_t,
        "co2_abated_fut_t_yr": co2_abated_f_t,
        "foregone_co2_abatement_t_yr": co2_abated_0_t - co2_abated_f_t,
        "stranded_reversion_co2_t_yr": max(0.0, co2_abated_f_t) if net_benefit_f_usd < 0.0 else 0.0,
    })

  return pd.DataFrame(out_rows)
