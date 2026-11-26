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
W_360 = np.repeat(DAYS_MONTH / 30.0, 30)  # (360,) day weights across 12 months x 30 empirical quantiles

# Uniform 0.05-step sub-share grid (closed under 1 - alpha so CELF and MILP share identical discrete alpha domain)
ALPHA_GRID = np.round(np.linspace(0.05, 1.00, 20), 2)
ALPHA_COARSE = np.array([0.05, 0.15, 0.30, 0.50, 0.70, 0.85, 1.00], dtype=np.float64)
CAP_PEAK_RATIOS = np.array([0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 1.00], dtype=np.float64)

# DEA typical extraction steam turbine power-loss factor C_v [MWh_e / MWh_th] and IPCC 2006 LHV fuel specs
CHP_POWER_LOSS_CV: float = 0.15
CHP_FUEL_SPECS: dict[str, tuple[float, float, float]] = {
    "power_coal": (0.15 / 0.39, 0.0946, 3.50),     # (GJ_fuel / GJ_th = Cv/eta_e, EF tCO2/GJ_fuel, fuel USD/GJ)
    "power_gas": (0.15 / 0.52, 0.0561, -1.0),      # -1.0 => use regional natural gas price p_gas_usd_per_gj
    "power_oil": (0.15 / 0.38, 0.0733, 12.00),
    "power_biomass": (0.15 / 0.30, 0.0, 5.00),
    "power_waste": (0.15 / 0.28, 0.0450, 0.00),
}


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
  chp_mode: str = "residual"  # "residual", "power_loss_015", "fuel_alloc", "fuel_co2_only"
  use_monthly_source_deduction: bool = False


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
    sector: str = "",
    subsector_detail: str = "",
) -> tuple[np.ndarray, dict[str, np.ndarray]] | None:
  """Vectorized evaluation across (alphas_in, CAP_PEAK_RATIOS) over all 360 (month, quantile) states."""
  unit_margin = c_ind_usd_per_gj - c_net_usd_per_gj
  if unit_margin <= 0.0 or d_zone_ann_gj <= 0.0 or len(alphas_in) == 0:
    return None

  q_raw_max = float(q_rem_m_gj_d.max())
  if q_raw_max <= 0.0:
    return None

  kappa, cop, eq_capex_eur, eq_fom_eur, eq_vom_eur = get_upgrade_params(upgrade_mode, cfg)
  dist_loss_eff = lambda_dist * (LTDH_LOSS_MULTIPLIER if cfg.is_4gdh else 1.0)
  if dist_loss_eff >= 0.95:
    return None

  avail_factor = kappa * (1.0 - dist_loss_eff)
  q_max_mw = (avail_factor * q_raw_max) / GJ_PER_MW_DAY
  if q_max_mw < 0.05:
    return None

  a_max = float(alphas_in[-1])
  max_sub_peak_mw = (a_max * d_zone_peak_gj_d) / GJ_PER_MW_DAY
  if max_sub_peak_mw <= 0.01:
    return None

  # Exact fixed-CAPEX lower bound check BEFORE any array allocations
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
  c_d_gj = c_mw * GJ_PER_MW_DAY               # (K_a, 8)

  is_12 = q_rem_m_gj_d.size == 12
  if is_12:
    # Fast L2-cache path for untouched sources (exact 360-state integral, 15 KB intermediate tensors)
    q_Avail_m = avail_factor * q_rem_m_gj_d.reshape(12)
    src_avail_m = eta_pipe[:, :, None] * q_Avail_m[None, None, :]     # (K_a, 8, 12)
    sup_cap_m = np.minimum(c_d_gj[:, :, None], src_avail_m)           # (K_a, 8, 12)
    h_mq = np.minimum(sup_cap_m[:, :, :, None], alphas[:, None, None, None] * d_zone_mq_gj_d[None, None, :, :])
    H_deliv_gj = np.sum(np.mean(h_mq, axis=3) * DAYS_MONTH[None, None, :], axis=2)
    min_src_gj_d = np.min(src_avail_m, axis=2)
    sup_cap_360 = None
  else:
    # Exact 360-state path for partially depleted shared sources
    q_Avail_360 = avail_factor * q_rem_m_gj_d.reshape(360)
    src_avail_360 = eta_pipe[:, :, None] * q_Avail_360[None, None, :]  # (K_a, 8, 360)
    sup_cap_360 = np.minimum(c_d_gj[:, :, None], src_avail_360)        # (K_a, 8, 360)
    d_zone_360 = d_zone_mq_gj_d.reshape(360)
    h_360_all = np.minimum(sup_cap_360, alphas[:, None, None] * d_zone_360[None, None, :])
    H_deliv_gj = h_360_all @ W_360
    min_src_gj_d = np.min(src_avail_360, axis=2)
    sup_cap_m = None

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
  min_supply_mw = np.minimum(c_mw, min_src_gj_d / GJ_PER_MW_DAY)
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

  # Alternative CHP extraction heat accounting conventions (for power direct_hex streams)
  chp_cost_usd = 0.0
  chp_co2_t = 0.0
  if cfg.chp_mode != "residual" and sector == "power" and upgrade_mode == "direct_hex":
    if cfg.chp_mode == "power_loss_015":
      lost_el_mwh = CHP_POWER_LOSS_CV * H_src_mwh
      chp_cost_usd = lost_el_mwh * p_el
      chp_co2_t = lost_el_mwh * grid_ef_tco2_per_mwh
    elif cfg.chp_mode in ("fuel_alloc", "fuel_co2_only"):
      fuel_gj_per_gj, ef_t_per_gj, p_fuel_fixed = CHP_FUEL_SPECS.get(
          subsector_detail, (0.15 / 0.39, 0.0946, 3.50)
      )
      inc_fuel_gj = fuel_gj_per_gj * H_src_gj
      chp_co2_t = inc_fuel_gj * ef_t_per_gj
      if cfg.chp_mode == "fuel_alloc":
        p_fuel_base = p_gas_usd_per_gj if (p_fuel_fixed < 0.0 and p_gas_usd_per_gj > 0.0) else max(3.50, p_fuel_fixed)
        chp_cost_usd = inc_fuel_gj * (p_fuel_base * cfg.price_multiplier)

  tot_el_mwh = hp_el_mwh + pump_el_mwh
  tot_dh_cost_usd = (
      dist_sub_a[:, None]
      + a_pipe * pipe_capex_usd + pipe_fom_usd + pipe_vom_usd
      + a_eq * eq_capex_usd + eq_fom_usd + eq_vom_usd
      + tot_el_mwh * p_el
      + a_pk * pk_capex_usd + pk_fom_usd + pk_vom_usd + pk_energy_usd
      + chp_cost_usd
  )
  net_benefit = np.where(valid_grid, avoided_ind_a[:, None] - tot_dh_cost_usd, -1.0)
  if float(net_benefit.max()) <= 0.0:
    return None

  dh_co2_t = tot_el_mwh * grid_ef_tco2_per_mwh + pk_co2_t + chp_co2_t
  avoided_co2_t = (sub_ann_d_a * cf_co2_t_per_gj)[:, None] - dh_co2_t

  arrays: dict[str, object] = {
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
      "sup_cap_m": sup_cap_m,
      "sup_cap_360": sup_cap_360,
      "d_zone_mq": d_zone_mq_gj_d,
  }
  return net_benefit, arrays  # type: ignore[return-value]


def _extract_design_dict(ia: int, ic: int, best_n: float, arrays: dict[str, np.ndarray]) -> dict[str, object]:
  kappa = float(arrays["kappa_scalar"][0])
  eta_tot = float(arrays["eta_tot"][ia, ic])
  alpha_v = float(arrays["alphas"][ia])
  H_deliv = float(arrays["H_deliv_gj"][ia, ic])
  dmq = arrays["d_zone_mq"]
  if arrays.get("sup_cap_m") is not None:
    h_360_best = np.minimum(arrays["sup_cap_m"][ia, ic, :, None], alpha_v * dmq).reshape(360)
  else:
    h_360_best = np.minimum(arrays["sup_cap_360"][ia, ic, :], alpha_v * dmq.reshape(360))
  q_used_360 = h_360_best / (kappa * eta_tot)
  q_used_m = q_used_360.reshape(12, 30).mean(axis=1)
  return {
      "net_benefit_usd_yr": float(best_n),
      "c_mw": float(arrays["c_mw"][ia, ic]),
      "c_src_mw": float(arrays["c_src_mw"][ia, ic]),
      "c_pk_mw": float(arrays["c_pk_mw"][ia, ic]),
      "alpha": alpha_v,
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
      "q_used_360": q_used_360,
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
    sector: str = "",
    subsector_detail: str = "",
) -> dict[str, object] | None:
  """Optimize (C_MW, alpha <= alpha_rem) via coarse-to-fine 0.05-grid search and return best N > 0 design."""
  if alpha_rem <= 1e-6:
    return None
  alphas_c = ALPHA_COARSE[ALPHA_COARSE <= alpha_rem + 1e-9]
  if len(alphas_c) == 0 or alphas_c[-1] < alpha_rem - 1e-6:
    alphas_c = np.append(alphas_c, alpha_rem)

  kw = dict(
      ell_km=ell_km,
      upgrade_mode=upgrade_mode,
      q_rem_m_gj_d=q_rem_m_gj_d,
      d_zone_mq_gj_d=d_zone_mq_gj_d,
      d_zone_peak_gj_d=d_zone_peak_gj_d,
      d_zone_ann_gj=d_zone_ann_gj,
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
      sector=sector,
      subsector_detail=subsector_detail,
  )
  out = _evaluate_grid_for_link(alphas_in=alphas_c, **kw)  # type: ignore[arg-type]
  if out is None:
    return None
  net_benefit, arrays = out
  flat_best = int(np.argmax(net_benefit))
  ia, ic = divmod(flat_best, net_benefit.shape[1])
  best_n = float(net_benefit[ia, ic])
  if best_n <= 0.0:
    return None

  best_a = float(arrays["alphas"][ia])
  # Local ±0.05 refinement on ALPHA_GRID around coarse winner
  nbrs = ALPHA_GRID[(ALPHA_GRID <= alpha_rem + 1e-9) & (np.abs(ALPHA_GRID - best_a) <= 0.101) & (np.abs(ALPHA_GRID - best_a) > 1e-6)]
  if len(nbrs) > 0:
    out_r = _evaluate_grid_for_link(alphas_in=nbrs, **kw)  # type: ignore[arg-type]
    if out_r is not None:
      nb_r, arr_r = out_r
      fb_r = int(np.argmax(nb_r))
      ia_r, ic_r = divmod(fb_r, nb_r.shape[1])
      if float(nb_r[ia_r, ic_r]) > best_n:
        best_n = float(nb_r[ia_r, ic_r])
        ia, ic, arrays = ia_r, ic_r, arr_r

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
    sector: str = "",
    subsector_detail: str = "",
    all_capacities: bool = False,
) -> list[dict[str, object]]:
  """Return feasible positive-benefit designs across ALPHA_GRID (either best C per alpha or all non-dominated C)."""
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
      sector=sector,
      subsector_detail=subsector_detail,
  )
  if out is None:
    return []
  net_benefit, arrays = out
  results: list[dict[str, object]] = []
  for ia in range(net_benefit.shape[0]):
    if not all_capacities:
      ic = int(np.argmax(net_benefit[ia, :]))
      best_n = float(net_benefit[ia, ic])
      if best_n > 0.0:
        results.append(_extract_design_dict(ia, ic, best_n, arrays))
    else:
      best_so_far = 0.0
      for ic in range(net_benefit.shape[1]):
        nb = float(net_benefit[ia, ic])
        # Smaller ic uses <= source peak capacity; keep strictly improving capacities up to optimum
        if nb > best_so_far + 1e-3:
          best_so_far = nb
          results.append(_extract_design_dict(ia, ic, nb, arrays))
  return results


def solve_country_celf(
    df_cands: pd.DataFrame,
    q_init_m_gj_d: dict[str, np.ndarray],
    zone_dmq_gj_d: dict[str, np.ndarray],
    zone_meta: dict[str, dict[str, float]],
    cfg: TechConfig,
) -> tuple[list[dict[str, object]], float]:
  """Run CELF lazy priority-queue greedy matching across all 360 states; return (selected_links, P_gj_yr)."""
  if len(df_cands) == 0:
    return [], 0.0

  # Keep untouched sources as compact (12,) arrays; promote to (360,) only when a stream is actually tapped!
  q_rem = {sid: arr.copy() for sid, arr in q_init_m_gj_d.items()}
  alpha_rem = {zid: 1.0 for zid in zone_meta}

  # Initial evaluation of all candidate links using exact vectorized fixed-cost pre-screen + zero-copy arrays
  heap: list[tuple[float, int, int, dict[str, object]]] = []
  zids_arr = df_cands["zone_id"].to_numpy()
  sids_arr = df_cands["stream_id"].to_numpy()
  ells_arr = df_cands["ell_km"].to_numpy(dtype=np.float64)
  ums_arr = df_cands["upgrade_mode"].to_numpy()
  secs_arr = df_cands["sector"].to_numpy() if "sector" in df_cands.columns else np.full(len(df_cands), "")
  subs_arr = df_cands["subsector_detail"].to_numpy() if "subsector_detail" in df_cands.columns else np.full(len(df_cands), "")

  # Exact vectorized pre-screen identical to lines 117-156 of _evaluate_grid_for_link (eliminates ~98% dead Python calls)
  u_zids, z_inv = np.unique(zids_arr, return_inverse=True)
  z_umarg = np.array([zone_meta[str(z)]["c_ind_usd_per_gj"] - zone_meta[str(z)]["c_net_usd_per_gj"] for z in u_zids], dtype=np.float64)[z_inv]
  z_dann = np.array([zone_meta[str(z)]["ann_gj"] for z in u_zids], dtype=np.float64)[z_inv]
  z_dpk = np.array([zone_meta[str(z)]["peak_gj_d"] for z in u_zids], dtype=np.float64)[z_inv]
  z_ldist = np.array([zone_meta[str(z)]["lambda_dist"] for z in u_zids], dtype=np.float64)[z_inv]
  z_cciv = np.array([zone_meta[str(z)]["cap_scale_civil"] for z in u_zids], dtype=np.float64)[z_inv]
  z_ceq = np.array([zone_meta[str(z)]["cap_scale_equip"] for z in u_zids], dtype=np.float64)[z_inv]

  u_sids, s_inv = np.unique(sids_arr, return_inverse=True)
  s_qmax = np.array([float(q_rem[str(s)].max()) for s in u_sids], dtype=np.float64)[s_inv]

  u_ums, um_inv = np.unique(ums_arr, return_inverse=True)
  um_kap = np.array([get_upgrade_params(str(u), cfg)[0] for u in u_ums], dtype=np.float64)[um_inv]
  um_eqc = np.array([get_upgrade_params(str(u), cfg)[2] for u in u_ums], dtype=np.float64)[um_inv]

  dloss_mult = LTDH_LOSS_MULTIPLIER if cfg.is_4gdh else 1.0
  net_cap_mult = LTDH_NETWORK_CAPEX_MULTIPLIER if cfg.is_4gdh else 1.0
  a_pipe_v = crf(cfg.discount_rate, PIPE_LIFETIME_YR)
  a_eq_v = crf(cfg.discount_rate, STATION_LIFETIME_YR)

  dloss_eff = z_ldist * dloss_mult
  q_max_mw_v = (um_kap * (1.0 - dloss_eff) * s_qmax) / GJ_PER_MW_DAY
  max_spk_mw_v = z_dpk / GJ_PER_MW_DAY
  c_up_max_v = np.maximum(0.1, np.minimum(max_spk_mw_v, q_max_mw_v * 1.05))
  l_min_v = np.where(c_up_max_v <= 20.0, 0.010, np.where(c_up_max_v <= 100.0, 0.0025, 0.0010)) * dloss_mult
  eta_max_v = ((1.0 - l_min_v) ** ells_arr) * (1.0 - dloss_eff)
  eta_safe_v = np.where(eta_max_v > 0.05, eta_max_v, 1.0)
  c_src_min_v = 0.1 / eta_safe_v
  min_fix_ann_v = (
      a_pipe_v * PIPE_CAPEX_COEF_EUR_PER_KM * (c_src_min_v ** PIPE_CAPEX_EXPONENT) * ells_arr * net_cap_mult * z_cciv
      + a_eq_v * um_eqc * c_src_min_v * z_ceq
  ) * USD_PER_EUR_2025

  viable_indices = np.flatnonzero(
      (z_umarg > 0.0)
      & (z_dann > 0.0)
      & (s_qmax > 0.0)
      & (dloss_eff < 0.95)
      & (q_max_mw_v >= 0.05)
      & (max_spk_mw_v > 0.01)
      & (eta_max_v > 0.05)
      & ((z_dann * z_umarg) > min_fix_ann_v)
  )

  for idx_np in viable_indices:
    idx = int(idx_np)
    zid = str(zids_arr[idx])
    zm = zone_meta[zid]
    sid = str(sids_arr[idx])
    res = evaluate_best_design_for_link(
        ell_km=float(ells_arr[idx]),
        upgrade_mode=str(ums_arr[idx]),
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
        sector=str(secs_arr[idx]),
        subsector_detail=str(subs_arr[idx]),
    )
    if res is not None and float(res["net_benefit_usd_yr"]) > 0.0:
      heapq.heappush(heap, (-float(res["net_benefit_usd_yr"]), idx, 0, res))

  src_ver = {sid: 0 for sid in q_rem}
  zone_ver = {zid: 0 for zid in alpha_rem}
  selected: list[dict[str, object]] = []
  total_P_gj = 0.0

  while heap:
    neg_n, idx, eval_stamp, res = heapq.heappop(heap)
    sid = str(sids_arr[idx])
    zid = str(zids_arr[idx])
    cur_stamp = src_ver[sid] + zone_ver[zid]
    if eval_stamp == cur_stamp:
      alpha_sel = float(res["alpha"])
      alpha_rem[zid] = max(0.0, alpha_rem[zid] - alpha_sel)
      q_used_360 = np.asarray(res.pop("q_used_360"), dtype=np.float64)
      if cfg.use_monthly_source_deduction:
        q_used_m = q_used_360.reshape(12, 30).mean(axis=1)
        if q_rem[sid].size == 12:
          q_rem[sid] = np.maximum(0.0, q_rem[sid] - q_used_m)
        else:
          q_rem[sid] = np.maximum(0.0, q_rem[sid] - np.repeat(q_used_m, 30))
      else:
        if q_rem[sid].size == 12:
          q_rem[sid] = np.maximum(0.0, np.repeat(q_rem[sid].reshape(12), 30) - q_used_360)
        else:
          q_rem[sid] = np.maximum(0.0, q_rem[sid] - q_used_360)
      src_ver[sid] += 1
      zone_ver[zid] += 1
      rec = {**df_cands.iloc[idx].to_dict(), **res}
      selected.append(rec)
      total_P_gj += float(res["P_net_waste_heat_gj_yr"])
    else:
      if alpha_rem[zid] <= 1e-6 or float(q_rem[sid].max()) <= 1e-3:
        continue
      zm = zone_meta[zid]
      new_res = evaluate_best_design_for_link(
          ell_km=float(ells_arr[idx]),
          upgrade_mode=str(ums_arr[idx]),
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
          sector=str(secs_arr[idx]),
          subsector_detail=str(subs_arr[idx]),
      )
      if new_res is not None and float(new_res["net_benefit_usd_yr"]) > 0.0:
        heapq.heappush(heap, (-float(new_res["net_benefit_usd_yr"]), idx, src_ver[sid] + zone_ver[zid], new_res))

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
