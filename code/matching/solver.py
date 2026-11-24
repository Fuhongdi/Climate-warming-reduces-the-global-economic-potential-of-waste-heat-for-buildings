"""National multiperiod linear programming technical potential solver, priority-queue economic screening and MILP benchmark."""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import math

import highspy
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from code.common.parameters import (
    HP_INDUSTRIAL_POWER_2025,
    HP_INDUSTRIAL_POWER_2050,
    HP_WASTEWATER_2025,
    HP_WASTEWATER_2050,
    LTDH_LOSS_MULTIPLIER,
    LTDH_NETWORK_CAPEX_MULTIPLIER,
    NATURAL_GAS_EF_TCO2_PER_GJ_LHV,
    PEAKING_ELEC_BOILER_2025,
    PEAKING_ELEC_BOILER_2050,
    PEAKING_GAS_BOILER_2025,
    PEAKING_GAS_BOILER_2050,
    PIPE_CAPEX_COEF_EUR_PER_KM,
    PIPE_CAPEX_EXPONENT,
    PIPE_FIXED_OM_EUR_PER_MW_KM_YR,
    PIPE_LIFETIME_YR,
    PIPE_PUMP_ELEC_MWH_PER_MWH_TH,
    PIPE_VAR_OM_EUR_PER_MWH_KM,
    SOURCE_HEX_STATION_CAPEX_EUR_PER_MW,
    SOURCE_PUMP_STATION_CAPEX_EUR_PER_MW,
    SOURCE_STATION_FIXED_OM_FRAC_YR,
    STATION_LIFETIME_YR,
    USD_PER_EUR_2025,
    pipe_loss_per_km,
)
from code.common.paths import GJ_PER_MW_DAY, GJ_PER_MWH

DAYS_MONTH = np.array([31.0, 28.25, 31.0, 30.0, 31.0, 30.0, 31.0, 31.0, 30.0, 31.0, 30.0, 31.0], dtype=np.float64)
DAYS_YEAR = float(DAYS_MONTH.sum())  # 365.25 exact calendar mean over 20-year windows

# Candidate grid of sub-shares alpha and capacity-to-peak sizing ratios for per-link optimization
ALPHA_GRID = np.array([0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.65, 0.80, 1.00], dtype=np.float64)
CAP_PEAK_RATIOS = np.array([0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 1.00], dtype=np.float64)


def crf(discount_rate: float, lifetime_yr: float) -> float:
  """Capital recovery factor r(1+r)^L / ((1+r)^L - 1)."""
  if discount_rate <= 0.0:
    return 1.0 / lifetime_yr
  x = (1.0 + discount_rate) ** lifetime_yr
  return (discount_rate * x) / (x - 1.0)


@dataclass(frozen=True)
class TechConfig:
  discount_rate: float = 0.07
  is_4gdh: bool = False
  use_2050_dh_tech: bool = False
  price_multiplier: float = 1.0


def get_upgrade_params(upgrade_mode: str, cfg: TechConfig) -> tuple[float, float, float, float, float]:
  """Return (kappa, cop_or_inf, capex_eur_per_mw_src, fixed_om_eur_per_mw_yr, var_om_eur_per_mwh_src)."""
  if upgrade_mode == "direct_hex":
    capex = SOURCE_HEX_STATION_CAPEX_EUR_PER_MW + SOURCE_PUMP_STATION_CAPEX_EUR_PER_MW
    fom = capex * SOURCE_STATION_FIXED_OM_FRAC_YR
    return 1.0, math.inf, capex, fom, 0.0
  if upgrade_mode == "hp_wastewater":
    sp = HP_WASTEWATER_2050 if cfg.use_2050_dh_tech else HP_WASTEWATER_2025
  else:
    sp = HP_INDUSTRIAL_POWER_2050 if cfg.use_2050_dh_tech else HP_INDUSTRIAL_POWER_2025
  capex = sp.capex_eur_per_mw_h + SOURCE_PUMP_STATION_CAPEX_EUR_PER_MW
  fom = sp.fixed_om_eur_per_mw_h_yr + SOURCE_PUMP_STATION_CAPEX_EUR_PER_MW * SOURCE_STATION_FIXED_OM_FRAC_YR
  return sp.kappa, sp.cop, capex, fom, sp.var_om_eur_per_mwh_h


def _evaluate_grid_for_link(
    ell_km: float,
    upgrade_mode: str,
    q_rem_m_gj_d: np.ndarray,
    d_zone_mq_gj_d: np.ndarray,
    d_zone_peak_gj_d: float,
    d_zone_ann_gj: float,
    alphas_in: np.ndarray,
    lambda_dist: float,
    c_ind_usd_per_gj: float,
    cf_co2_t_per_gj: float,
    c_net_usd_per_gj: float,
    p_elec_usd_per_mwh: float,
    p_gas_usd_per_gj: float,
    has_gas: bool,
    grid_ef_tco2_per_mwh: float,
    cap_scale_civil: float,
    cap_scale_equip: float,
    cfg: TechConfig,
) -> tuple[np.ndarray, dict[str, np.ndarray]] | None:
  """Vectorized evaluation across (alphas_in, CAP_PEAK_RATIOS). Exact mathematical parity with scalar loop."""
  unit_margin = c_ind_usd_per_gj - c_net_usd_per_gj
  if unit_margin <= 0.0 or d_zone_ann_gj <= 0.0 or len(alphas_in) == 0:
    return None
  if float(q_rem_m_gj_d.max()) <= 0.0:
    return None

  kappa, cop, eq_capex_eur, eq_fom_eur, eq_vom_eur = get_upgrade_params(upgrade_mode, cfg)
  dist_loss_eff = lambda_dist * (LTDH_LOSS_MULTIPLIER if cfg.is_4gdh else 1.0)
  if dist_loss_eff >= 0.95:
    return None

  q_Avail_m = kappa * (1.0 - dist_loss_eff) * q_rem_m_gj_d  # (12,)
  q_max_mw = float(q_Avail_m.max()) / GJ_PER_MW_DAY
  if q_max_mw < 0.05:
    return None

  a_max = float(alphas_in[-1])
  max_sub_peak_mw = (a_max * d_zone_peak_gj_d) / GJ_PER_MW_DAY
  if max_sub_peak_mw <= 0.01:
    return None

  # Exact fixed-CAPEX lower bound check (zero approximation: all omitted O&M/energy/peaking costs >= 0)
  c_upper_max = max(0.1, min(max_sub_peak_mw, q_max_mw * 1.05))
  l_min = pipe_loss_per_km(c_upper_max, is_4gdh=cfg.is_4gdh)
  eta_tot_max = ((1.0 - l_min) ** ell_km) * (1.0 - dist_loss_eff)
  if eta_tot_max <= 0.05:
    return None

  a_pipe = crf(cfg.discount_rate, PIPE_LIFETIME_YR)
  a_eq = crf(cfg.discount_rate, STATION_LIFETIME_YR)
  net_cap_mult = LTDH_NETWORK_CAPEX_MULTIPLIER if cfg.is_4gdh else 1.0
  c_src_min = 0.1 / eta_tot_max
  min_fixed_annual_usd = (
      a_pipe * PIPE_CAPEX_COEF_EUR_PER_KM * (c_src_min ** PIPE_CAPEX_EXPONENT) * ell_km * net_cap_mult * cap_scale_civil
      + a_eq * eq_capex_eur * c_src_min * cap_scale_equip
  ) * USD_PER_EUR_2025
  if a_max * d_zone_ann_gj * unit_margin <= min_fixed_annual_usd:
    return None

  sub_peak_mw_a = (alphas_in * d_zone_peak_gj_d) / GJ_PER_MW_DAY
  valid_a = sub_peak_mw_a > 0.01
  if not np.any(valid_a):
    return None
  alphas = alphas_in[valid_a]
  sub_peak_mw_a = sub_peak_mw_a[valid_a]
  sub_ann_d_a = alphas * d_zone_ann_gj
  avoided_ind_a = sub_ann_d_a * c_ind_usd_per_gj
  dist_sub_a = sub_ann_d_a * c_net_usd_per_gj

  # (K_a, 8) candidate capacities
  cap_upper_a = np.minimum(sub_peak_mw_a, q_max_mw * 1.05)[:, None]
  c_mw = np.clip(sub_peak_mw_a[:, None] * CAP_PEAK_RATIOS[None, :], 0.1, cap_upper_a)

  l_base = np.where(c_mw <= 20.0, 0.010, np.where(c_mw <= 100.0, 0.0025, 0.0010))
  if cfg.is_4gdh:
    l_base = l_base * LTDH_LOSS_MULTIPLIER
  eta_pipe = (1.0 - l_base) ** ell_km         # (K_a, 8)
  eta_tot = eta_pipe * (1.0 - dist_loss_eff)  # (K_a, 8)

  c_d_gj = c_mw * GJ_PER_MW_DAY                                      # (K_a, 8)
  src_avail_m = eta_pipe[:, :, None] * q_Avail_m[None, None, :]      # (K_a, 8, 12)
  sup_cap_m = np.minimum(c_d_gj[:, :, None], src_avail_m)            # (K_a, 8, 12)

  # Broadcast over monthly quantiles: (K_a, 8, 12, Q)
  h_mq = np.minimum(sup_cap_m[:, :, :, None], alphas[:, None, None, None] * d_zone_mq_gj_d[None, None, :, :])
  h_m_mean = np.mean(h_mq, axis=3)                                   # (K_a, 8, 12)
  H_deliv_gj = np.sum(h_m_mean * DAYS_MONTH[None, None, :], axis=2)  # (K_a, 8)

  valid_grid = (eta_tot > 0.05) & (H_deliv_gj > 0.0) & ((avoided_ind_a > dist_sub_a)[:, None])
  if not np.any(valid_grid):
    return None

  if has_gas:
    pk_sp = PEAKING_GAS_BOILER_2050 if cfg.use_2050_dh_tech else PEAKING_GAS_BOILER_2025
  else:
    pk_sp = PEAKING_ELEC_BOILER_2050 if cfg.use_2050_dh_tech else PEAKING_ELEC_BOILER_2025
  a_pk = crf(cfg.discount_rate, pk_sp.lifetime_yr)

  p_el = p_elec_usd_per_mwh * cfg.price_multiplier
  p_ng = p_gas_usd_per_gj * cfg.price_multiplier

  eta_safe = np.where(valid_grid, eta_tot, 1.0)
  c_src_mw = c_mw / eta_safe
  H_src_gj = H_deliv_gj / eta_safe
  H_src_mwh = H_src_gj / GJ_PER_MWH

  pipe_capex_usd = (
      PIPE_CAPEX_COEF_EUR_PER_KM * (c_src_mw ** PIPE_CAPEX_EXPONENT) * ell_km
      * net_cap_mult * USD_PER_EUR_2025 * cap_scale_civil
  )
  pipe_fom_usd = PIPE_FIXED_OM_EUR_PER_MW_KM_YR * c_src_mw * ell_km * USD_PER_EUR_2025 * cap_scale_civil
  pipe_vom_usd = PIPE_VAR_OM_EUR_PER_MWH_KM * H_src_mwh * ell_km * USD_PER_EUR_2025
  pump_el_mwh = PIPE_PUMP_ELEC_MWH_PER_MWH_TH * H_src_mwh

  eq_capex_usd = eq_capex_eur * c_src_mw * USD_PER_EUR_2025 * cap_scale_equip
  eq_fom_usd = eq_fom_eur * c_src_mw * USD_PER_EUR_2025 * cap_scale_equip
  eq_vom_usd = eq_vom_eur * H_src_mwh * USD_PER_EUR_2025
  hp_el_mwh = (H_src_mwh / cop) if math.isfinite(cop) else np.zeros_like(H_src_mwh)

  H_pk_gj = np.maximum(0.0, sub_ann_d_a[:, None] - H_deliv_gj)
  min_supply_mw = np.minimum(c_mw, np.min(src_avail_m, axis=2) / GJ_PER_MW_DAY)
  c_pk_mw = np.maximum(0.0, sub_peak_mw_a[:, None] - min_supply_mw)
  pk_capex_usd = pk_sp.capex_eur_per_mw_h * c_pk_mw * USD_PER_EUR_2025 * cap_scale_equip
  pk_fom_usd = pk_sp.fixed_om_eur_per_mw_h_yr * c_pk_mw * USD_PER_EUR_2025 * cap_scale_equip
  H_pk_mwh = H_pk_gj / GJ_PER_MWH
  pk_vom_usd = pk_sp.var_om_non_elec_eur_per_mwh_h * H_pk_mwh * USD_PER_EUR_2025

  if pk_sp.fuel == "gas":
    pk_fuel_gj = H_pk_gj / pk_sp.efficiency_lhv
    pk_el_mwh = pk_sp.aux_elec_mwh_per_mwh_h * H_pk_mwh
    pk_energy_usd = pk_fuel_gj * p_ng + pk_el_mwh * p_el
    pk_co2_t = pk_fuel_gj * NATURAL_GAS_EF_TCO2_PER_GJ_LHV + pk_el_mwh * grid_ef_tco2_per_mwh
  else:
    pk_el_mwh = H_pk_mwh / pk_sp.efficiency_lhv
    pk_energy_usd = pk_el_mwh * p_el
    pk_co2_t = pk_el_mwh * grid_ef_tco2_per_mwh

  tot_el_mwh = hp_el_mwh + pump_el_mwh
  tot_dh_cost_usd = (
      dist_sub_a[:, None]
      + a_pipe * pipe_capex_usd + pipe_fom_usd + pipe_vom_usd
      + a_eq * eq_capex_usd + eq_fom_usd + eq_vom_usd
      + tot_el_mwh * p_el
      + a_pk * pk_capex_usd + pk_fom_usd + pk_vom_usd + pk_energy_usd
  )
  net_benefit = np.where(valid_grid, avoided_ind_a[:, None] - tot_dh_cost_usd, -1.0)
  if float(net_benefit.max()) <= 0.0:
    return None

  dh_co2_t = tot_el_mwh * grid_ef_tco2_per_mwh + pk_co2_t
  avoided_co2_t = (sub_ann_d_a * cf_co2_t_per_gj)[:, None] - dh_co2_t

  arrays = {
      "alphas": alphas,
      "c_mw": c_mw,
      "c_src_mw": c_src_mw,
      "c_pk_mw": c_pk_mw,
      "eta_tot": eta_tot,
      "eta_safe": eta_safe,
      "kappa_scalar": np.array([kappa]),
      "H_deliv_gj": H_deliv_gj,
      "H_pk_gj": H_pk_gj,
      "sub_ann_d_a": sub_ann_d_a,
      "tot_dh_cost_usd": tot_dh_cost_usd,
      "avoided_ind_a": avoided_ind_a,
      "total_capex_usd": pipe_capex_usd + eq_capex_usd + pk_capex_usd,
      "avoided_co2_t": avoided_co2_t,
      "h_m_mean": h_m_mean,
  }
  return net_benefit, arrays


def _extract_design_dict(ia: int, ic: int, best_n: float, arrays: dict[str, np.ndarray]) -> dict[str, float]:
  kappa = float(arrays["kappa_scalar"][0])
  eta_tot = float(arrays["eta_tot"][ia, ic])
  H_deliv = float(arrays["H_deliv_gj"][ia, ic])
  q_used_m = arrays["h_m_mean"][ia, ic, :] / (kappa * eta_tot)
  return {
      "net_benefit_usd_yr": float(best_n),
      "c_mw": float(arrays["c_mw"][ia, ic]),
      "c_src_mw": float(arrays["c_src_mw"][ia, ic]),
      "c_pk_mw": float(arrays["c_pk_mw"][ia, ic]),
      "alpha": float(arrays["alphas"][ia]),
      "eta_tot": eta_tot,
      "kappa": kappa,
      "H_deliv_gj_yr": H_deliv,
      "P_net_waste_heat_gj_yr": float(H_deliv / kappa),
      "H_peaking_gj_yr": float(arrays["H_pk_gj"][ia, ic]),
      "sub_demand_gj_yr": float(arrays["sub_ann_d_a"][ia]),
      "tot_dh_cost_usd_yr": float(arrays["tot_dh_cost_usd"][ia, ic]),
      "avoided_ind_cost_usd_yr": float(arrays["avoided_ind_a"][ia]),
      "total_capex_usd": float(arrays["total_capex_usd"][ia, ic]),
      "co2_abated_t_yr": float(arrays["avoided_co2_t"][ia, ic]),
      **{f"q_used_m{m+1:02d}_gj_d": float(q_used_m[m]) for m in range(12)},
  }


def evaluate_best_design_for_link(
    ell_km: float,
    upgrade_mode: str,
    q_rem_m_gj_d: np.ndarray,
    d_zone_mq_gj_d: np.ndarray,
    d_zone_peak_gj_d: float,
    d_zone_ann_gj: float,
    alpha_rem: float,
    lambda_dist: float,
    c_ind_usd_per_gj: float,
    cf_co2_t_per_gj: float,
    c_net_usd_per_gj: float,
    p_elec_usd_per_mwh: float,
    p_gas_usd_per_gj: float,
    has_gas: bool,
    grid_ef_tco2_per_mwh: float,
    cap_scale_civil: float,
    cap_scale_equip: float,
    cfg: TechConfig,
) -> dict[str, float] | None:
  """Optimize (C_MW, alpha <= alpha_rem) for candidate link (i, z) and return best N > 0 design (or None)."""
  if alpha_rem <= 1e-6:
    return None
  alphas = ALPHA_GRID[ALPHA_GRID <= alpha_rem + 1e-9]
  if len(alphas) == 0 or alphas[-1] < alpha_rem - 1e-6:
    alphas = np.append(alphas, alpha_rem)

  out = _evaluate_grid_for_link(
      ell_km=ell_km,
      upgrade_mode=upgrade_mode,
      q_rem_m_gj_d=q_rem_m_gj_d,
      d_zone_mq_gj_d=d_zone_mq_gj_d,
      d_zone_peak_gj_d=d_zone_peak_gj_d,
      d_zone_ann_gj=d_zone_ann_gj,
      alphas_in=alphas,
      lambda_dist=lambda_dist,
      c_ind_usd_per_gj=c_ind_usd_per_gj,
      cf_co2_t_per_gj=cf_co2_t_per_gj,
      c_net_usd_per_gj=c_net_usd_per_gj,
      p_elec_usd_per_mwh=p_elec_usd_per_mwh,
      p_gas_usd_per_gj=p_gas_usd_per_gj,
      has_gas=has_gas,
      grid_ef_tco2_per_mwh=grid_ef_tco2_per_mwh,
      cap_scale_civil=cap_scale_civil,
      cap_scale_equip=cap_scale_equip,
      cfg=cfg,
  )
  if out is None:
    return None
  net_benefit, arrays = out
  flat_best = int(np.argmax(net_benefit))
  ia, ic = divmod(flat_best, net_benefit.shape[1])
  best_n = float(net_benefit[ia, ic])
  if best_n <= 0.0:
    return None
  return _extract_design_dict(ia, ic, best_n, arrays)


def evaluate_all_alpha_options_for_link(
    ell_km: float,
    upgrade_mode: str,
    q_rem_m_gj_d: np.ndarray,
    d_zone_mq_gj_d: np.ndarray,
    d_zone_peak_gj_d: float,
    d_zone_ann_gj: float,
    lambda_dist: float,
    c_ind_usd_per_gj: float,
    cf_co2_t_per_gj: float,
    c_net_usd_per_gj: float,
    p_elec_usd_per_mwh: float,
    p_gas_usd_per_gj: float,
    has_gas: bool,
    grid_ef_tco2_per_mwh: float,
    cap_scale_civil: float,
    cap_scale_equip: float,
    cfg: TechConfig,
) -> list[dict[str, float]]:
  """Return optimal C design for EVERY feasible alpha in ALPHA_GRID with N > 0 in a single vectorized call."""
  out = _evaluate_grid_for_link(
      ell_km=ell_km,
      upgrade_mode=upgrade_mode,
      q_rem_m_gj_d=q_rem_m_gj_d,
      d_zone_mq_gj_d=d_zone_mq_gj_d,
      d_zone_peak_gj_d=d_zone_peak_gj_d,
      d_zone_ann_gj=d_zone_ann_gj,
      alphas_in=ALPHA_GRID,
      lambda_dist=lambda_dist,
      c_ind_usd_per_gj=c_ind_usd_per_gj,
      cf_co2_t_per_gj=cf_co2_t_per_gj,
      c_net_usd_per_gj=c_net_usd_per_gj,
      p_elec_usd_per_mwh=p_elec_usd_per_mwh,
      p_gas_usd_per_gj=p_gas_usd_per_gj,
      has_gas=has_gas,
      grid_ef_tco2_per_mwh=grid_ef_tco2_per_mwh,
      cap_scale_civil=cap_scale_civil,
      cap_scale_equip=cap_scale_equip,
      cfg=cfg,
  )
  if out is None:
    return []
  net_benefit, arrays = out
  results: list[dict[str, float]] = []
  for ia in range(net_benefit.shape[0]):
    ic = int(np.argmax(net_benefit[ia, :]))
    best_n = float(net_benefit[ia, ic])
    if best_n > 0.0:
      results.append(_extract_design_dict(ia, ic, best_n, arrays))
  return results


def solve_country_celf(
    df_cands: pd.DataFrame,
    q_init_m_gj_d: dict[str, np.ndarray],
    zone_dmq_gj_d: dict[str, np.ndarray],
    zone_meta: dict[str, dict[str, float]],
    cfg: TechConfig,
) -> tuple[list[dict[str, object]], float]:
  """Run CELF lazy priority-queue greedy matching for one country; return (selected_links, P_gj_yr)."""
  if len(df_cands) == 0:
    return [], 0.0

  q_rem = {sid: arr.copy() for sid, arr in q_init_m_gj_d.items()}
  alpha_rem = {zid: 1.0 for zid in zone_meta}

  # Initial evaluation of all candidate links
  heap: list[tuple[float, int, int, dict[str, float]]] = []
  cand_rows = list(df_cands.to_dict("records"))

  for idx, row in enumerate(cand_rows):
    zid = str(row["zone_id"])
    zm = zone_meta[zid]
    if zm["c_ind_usd_per_gj"] <= zm["c_net_usd_per_gj"]:
      continue
    sid = str(row["stream_id"])
    res = evaluate_best_design_for_link(
        ell_km=float(row["ell_km"]),
        upgrade_mode=str(row["upgrade_mode"]),
        q_rem_m_gj_d=q_rem[sid],
        d_zone_mq_gj_d=zone_dmq_gj_d[zid],
        d_zone_peak_gj_d=zm["peak_gj_d"],
        d_zone_ann_gj=zm["ann_gj"],
        alpha_rem=alpha_rem[zid],
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
    if res is not None and res["net_benefit_usd_yr"] > 0.0:
      heapq.heappush(heap, (-res["net_benefit_usd_yr"], idx, 0, res))

  src_ver = {sid: 0 for sid in q_rem}
  zone_ver = {zid: 0 for zid in alpha_rem}
  selected: list[dict[str, object]] = []
  total_P_gj = 0.0

  while heap:
    neg_n, idx, eval_stamp, res = heapq.heappop(heap)
    row = cand_rows[idx]
    sid = str(row["stream_id"])
    zid = str(row["zone_id"])
    cur_stamp = src_ver[sid] + zone_ver[zid]
    if eval_stamp == cur_stamp:
      alpha_sel = res["alpha"]
      alpha_rem[zid] = max(0.0, alpha_rem[zid] - alpha_sel)
      for m in range(12):
        q_rem[sid][m] = max(0.0, q_rem[sid][m] - res[f"q_used_m{m+1:02d}_gj_d"])
      src_ver[sid] += 1
      zone_ver[zid] += 1
      rec = {**row, **res}
      selected.append(rec)
      total_P_gj += res["P_net_waste_heat_gj_yr"]
    else:
      if alpha_rem[zid] <= 1e-6 or float(q_rem[sid].max()) <= 1e-3:
        continue
      zm = zone_meta[zid]
      new_res = evaluate_best_design_for_link(
          ell_km=float(row["ell_km"]),
          upgrade_mode=str(row["upgrade_mode"]),
          q_rem_m_gj_d=q_rem[sid],
          d_zone_mq_gj_d=zone_dmq_gj_d[zid],
          d_zone_peak_gj_d=zm["peak_gj_d"],
          d_zone_ann_gj=zm["ann_gj"],
          alpha_rem=alpha_rem[zid],
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
      if new_res is not None and new_res["net_benefit_usd_yr"] > 0.0:
        heapq.heappush(heap, (-new_res["net_benefit_usd_yr"], idx, src_ver[sid] + zone_ver[zid], new_res))

  return selected, total_P_gj


def solve_country_upper_bound_lp(
    df_cands: pd.DataFrame,
    q_init_m_gj_d: dict[str, np.ndarray],
    zone_dmq_gj_d: dict[str, np.ndarray],
    zone_meta: dict[str, dict[str, float]],
    cfg: TechConfig,
) -> tuple[float, dict[str, float]]:
  """Compute cost-free physical upper bound U [GJ/yr] via exact HiGHS LP across all (m, q) states."""
  if len(df_cands) == 0:
    return 0.0, {zid: 0.0 for zid in zone_meta}

  sids = sorted(df_cands["stream_id"].unique())
  zids = sorted(df_cands["zone_id"].unique())
  sid_to_i = {s: idx for idx, s in enumerate(sids)}
  zid_to_j = {z: idx for idx, z in enumerate(zids)}
  n_s, n_z = len(sids), len(zids)
  n_edges = len(df_cands)

  l_km_min = pipe_loss_per_km(200.0, is_4gdh=cfg.is_4gdh)
  dist_mult = LTDH_LOSS_MULTIPLIER if cfg.is_4gdh else 1.0

  edge_s = df_cands["stream_id"].map(sid_to_i).values.astype(np.int32)
  edge_z = df_cands["zone_id"].map(zid_to_j).values.astype(np.int32)
  ell_arr = df_cands["ell_km"].values.astype(np.float64)
  z_lam = np.array([zone_meta[z]["lambda_dist"] for z in zids], dtype=np.float64)
  eta_tot_arr = ((1.0 - l_km_min) ** ell_arr) * (1.0 - z_lam[edge_z] * dist_mult)

  # Map upgrade_mode to kappa vectorially
  k_hex, _, _, _, _ = get_upgrade_params("direct_hex", cfg)
  k_ww, _, _, _, _ = get_upgrade_params("hp_wastewater", cfg)
  k_ip, _, _, _, _ = get_upgrade_params("hp_ind_power", cfg)
  um_vals = df_cands["upgrade_mode"].values
  kap_arr = np.where(um_vals == "direct_hex", k_hex, np.where(um_vals == "hp_wastewater", k_ww, k_ip))

  inv_kappa = 1.0 / kap_arr
  src_coef = 1.0 / np.maximum(1e-9, kap_arr * eta_tot_arr)

  sample_dmq = next(iter(zone_dmq_gj_d.values()))
  n_q = sample_dmq.shape[1]
  S_mat_full = np.vstack([q_init_m_gj_d[s] for s in sids])               # (n_s, 12)
  Z_mat_full = np.stack([zone_dmq_gj_d[z] for z in zids], axis=0)        # (n_z, 12, n_q)
  w_m_arr = DAYS_MONTH / float(n_q)

  # Decompose bipartite graph into exact disconnected components (zero coupling across >100 km gaps)
  adj = csr_matrix(
      (np.ones(2 * n_edges, dtype=np.int8),
       (np.concatenate([edge_s, n_s + edge_z]), np.concatenate([n_s + edge_z, edge_s]))),
      shape=(n_s + n_z, n_s + n_z),
  )
  n_comp, comp_labels = connected_components(adj, directed=False, return_labels=True)
  edge_comp = comp_labels[edge_s]

  zone_U_arr = np.zeros(n_z, dtype=np.float64)
  total_U_gj = 0.0
  inf = highspy.kHighsInf

  # Group edge indices by component in O(n_edges log n_edges)
  order_e = np.argsort(edge_comp, kind="mergesort")
  comp_sorted = edge_comp[order_e]
  splits = np.flatnonzero(np.diff(comp_sorted)) + 1
  comp_edge_groups = np.split(order_e, splits)

  for e_grp in comp_edge_groups:
    m_e = len(e_grp)
    if m_e == 1:
      # Exact analytical solution for isolated 1-source 1-zone component
      ei = int(e_grp[0])
      si = int(edge_s[ei])
      zi = int(edge_z[ei])
      # max h/kappa s.t. h/(kappa*eta) <= q_s(m), h <= D_z(m,q)  =>  h/kappa = min(eta * q_s(m), D_z(m,q) / kappa)
      sup_lim = (eta_tot_arr[ei] * S_mat_full[si, :])[:, None]          # (12, 1)
      dem_lim = Z_mat_full[zi, :, :] * inv_kappa[ei]                    # (12, n_q)
      u_val = float(np.sum(np.minimum(sup_lim, dem_lim) * w_m_arr[:, None]))
      total_U_gj += u_val
      zone_U_arr[zi] += u_val
      continue

    s_uniq, s_loc = np.unique(edge_s[e_grp], return_inverse=True)
    z_uniq, z_loc = np.unique(edge_z[e_grp], return_inverse=True)
    s_loc_i32 = s_loc.astype(np.int32)
    z_loc_i32 = z_loc.astype(np.int32)
    cs, cz = len(s_uniq), len(z_uniq)
    c_rows = cs + cz
    ik_sub = inv_kappa[e_grp]
    sc_sub = src_coef[e_grp]
    ell_sub = ell_arr[e_grp]

    S_sub = S_mat_full[s_uniq, :]        # (cs, 12)
    Z_sub = Z_mat_full[z_uniq, :, :]     # (cz, 12, n_q)
    row_indices_all = np.arange(c_rows, dtype=np.int32)
    lower_all = np.full(c_rows, -inf, dtype=np.float64)
    upper_all = np.zeros(c_rows, dtype=np.float64)
    z_u_sub = np.zeros(cz, dtype=np.float64)

    if m_e > 4000:
      # Exact Dantzig LP Sifting with in-place addCols warm-started basis retention
      ord_s = np.lexsort((ell_sub, s_loc_i32))
      cuts_s = np.r_[0, np.flatnonzero(np.diff(s_loc_i32[ord_s])) + 1]
      rank_s = np.empty(m_e, dtype=np.int32)
      rank_s[ord_s] = np.arange(m_e, dtype=np.int32) - np.repeat(cuts_s, np.diff(np.r_[cuts_s, m_e]))

      ord_z = np.lexsort((ell_sub, z_loc_i32))
      cuts_z = np.r_[0, np.flatnonzero(np.diff(z_loc_i32[ord_z])) + 1]
      rank_z = np.empty(m_e, dtype=np.int32)
      rank_z[ord_z] = np.arange(m_e, dtype=np.int32) - np.repeat(cuts_z, np.diff(np.r_[cuts_z, m_e]))

      active = np.zeros(m_e, dtype=bool)
      col_list: list[np.ndarray] = []

      h = highspy.Highs()
      h.setOptionValue("output_flag", False)
      h.setOptionValue("threads", 1)
      h.changeObjectiveSense(highspy.ObjSense.kMaximize)
      h.addRows(
          c_rows, lower_all, np.zeros(c_rows, dtype=np.float64),
          0, np.zeros(c_rows, dtype=np.int32), np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float64),
      )

      def _append_cols(new_cols: np.ndarray) -> None:
        k = len(new_cols)
        if k == 0:
          return
        active[new_cols] = True
        col_list.append(new_cols)
        starts = np.arange(0, 2 * k, 2, dtype=np.int32)
        idx_nz = np.empty(2 * k, dtype=np.int32)
        idx_nz[0::2] = s_loc_i32[new_cols]
        idx_nz[1::2] = cs + z_loc_i32[new_cols]
        val_nz = np.empty(2 * k, dtype=np.float64)
        val_nz[0::2] = sc_sub[new_cols]
        val_nz[1::2] = 1.0
        h.addCols(
            k,
            ik_sub[new_cols],
            np.zeros(k, dtype=np.float64),
            np.full(k, inf, dtype=np.float64),
            2 * k,
            starts,
            idx_nz,
            val_nz,
        )

      seed_cols = np.flatnonzero((rank_s < 8) | (rank_z < 6))
      _append_cols(seed_cols)
      idx_a = np.concatenate(col_list)

      def _find_violated(cur_idx_a: np.ndarray, sol_a: np.ndarray, rd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pos = sol_a > 1e-3
        if np.any(pos):
          sa = s_loc_i32[cur_idx_a][pos]
          za = z_loc_i32[cur_idx_a][pos]
          lhs = sc_sub[cur_idx_a][pos] * rd[sa] + rd[cs + za]
          err_pos = float(np.mean(np.abs(ik_sub[cur_idx_a][pos] - lhs)))
          err_neg = float(np.mean(np.abs(ik_sub[cur_idx_a][pos] + lhs)))
          sgn = 1.0 if err_pos <= err_neg else -1.0
        else:
          sgn = 1.0 if float(np.sum(rd)) >= 0.0 else -1.0
        rc = ik_sub - sgn * (sc_sub * rd[s_loc_i32] + rd[cs + z_loc_i32])
        rc[active] = -1.0
        return np.flatnonzero(rc > 1e-6), rc

      first_solve = True
      # Pre-sift in-place on 3 representative seasonal anchors (retaining optimal basis!)
      for am, aq in ((0, n_q - 1), (3, n_q // 2), (6, 0)):
        upper_all[:cs] = S_sub[:, am]
        upper_all[cs:] = Z_sub[:, am, aq]
        if float(upper_all.max()) <= 0.0:
          continue
        h.changeRowsBounds(c_rows, row_indices_all, lower_all, upper_all)
        for _ in range(20):
          h.run()
          if first_solve:
            h.setOptionValue("presolve", "off")
            first_solve = False
          sol_a = np.asarray(h.getSolution().col_value, dtype=np.float64)
          rd = np.asarray(h.getSolution().row_dual, dtype=np.float64)
          viol, rc = _find_violated(idx_a, sol_a, rd)
          if len(viol) == 0:
            break
          if len(viol) > 4000:
            viol = viol[np.argsort(-rc[viol])[:4000]]
          _append_cols(viol)
          idx_a = np.concatenate(col_list)

      prev_upper: np.ndarray | None = None
      prev_sol: np.ndarray | None = None

      for m in range(12):
        w_d = float(w_m_arr[m])
        upper_all[:cs] = S_sub[:, m]
        for q in range(n_q):
          upper_all[cs:] = Z_sub[:, m, q]
          if prev_upper is not None and prev_sol is not None and len(prev_sol) == len(idx_a) and np.max(np.abs(upper_all - prev_upper)) < 1e-9:
            sol_a = prev_sol
          else:
            h.changeRowsBounds(c_rows, row_indices_all, lower_all, upper_all)
            while True:
              h.run()
              if first_solve:
                h.setOptionValue("presolve", "off")
                first_solve = False
              sol_a = np.asarray(h.getSolution().col_value, dtype=np.float64)
              rd = np.asarray(h.getSolution().row_dual, dtype=np.float64)
              viol, rc = _find_violated(idx_a, sol_a, rd)
              if len(viol) == 0:
                break
              if len(viol) > 4000:
                viol = viol[np.argsort(-rc[viol])[:4000]]
              _append_cols(viol)
              idx_a = np.concatenate(col_list)
            prev_upper = upper_all.copy()
            prev_sol = sol_a
          net_wh_e = sol_a * ik_sub[idx_a] * w_d
          total_U_gj += float(net_wh_e.sum())
          z_u_sub += np.bincount(z_loc_i32[idx_a], weights=net_wh_e, minlength=cz)
    else:
      h = highspy.Highs()
      h.setOptionValue("output_flag", False)
      h.setOptionValue("threads", 1)
      h.addVars(m_e, np.zeros(m_e), np.full(m_e, inf))
      h.changeColsCost(m_e, np.arange(m_e, dtype=np.int32), ik_sub)
      h.changeObjectiveSense(highspy.ObjSense.kMaximize)

      r_idx = np.concatenate([s_loc_i32, cs + z_loc_i32])
      c_idx = np.concatenate([np.arange(m_e, dtype=np.int32), np.arange(m_e, dtype=np.int32)])
      vals = np.concatenate([sc_sub, np.ones(m_e, dtype=np.float64)])
      mat = csr_matrix((vals, (r_idx, c_idx)), shape=(c_rows, m_e))
      h.addRows(
          c_rows,
          lower_all,
          np.zeros(c_rows),
          int(mat.nnz),
          mat.indptr.astype(np.int32),
          mat.indices.astype(np.int32),
          mat.data.astype(np.float64),
      )

      first_solve = True
      prev_upper = None
      prev_sol = None

      for m in range(12):
        w_d = float(w_m_arr[m])
        upper_all[:cs] = S_sub[:, m]
        for q in range(n_q):
          upper_all[cs:] = Z_sub[:, m, q]
          if prev_upper is not None and prev_sol is not None and np.max(np.abs(upper_all - prev_upper)) < 1e-9:
            sol = prev_sol
          else:
            h.changeRowsBounds(c_rows, row_indices_all, lower_all, upper_all)
            h.run()
            if first_solve:
              h.setOptionValue("presolve", "off")
              first_solve = False
            sol = np.asarray(h.getSolution().col_value, dtype=np.float64)
            prev_upper = upper_all.copy()
            prev_sol = sol
          net_wh_e = sol * ik_sub * w_d
          total_U_gj += float(net_wh_e.sum())
          z_u_sub += np.bincount(z_loc_i32, weights=net_wh_e, minlength=cz)

    zone_U_arr[z_uniq] += z_u_sub

  zone_U_gj = {z: 0.0 for z in zone_meta}
  for j_z, zid in enumerate(zids):
    zone_U_gj[zid] = float(zone_U_arr[j_z])

  return total_U_gj, zone_U_gj
