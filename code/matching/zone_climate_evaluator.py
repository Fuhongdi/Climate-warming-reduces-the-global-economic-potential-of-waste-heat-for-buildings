"""Evaluate urban heat zone daily demands, counterfactual individual costs, & global D for any 20-yr window.

Loads daily populated-cell BAIT and TAS (applying 2006–2025 monthly bias offsets for ISIMIP3b runs),
projects exact zone daily demand D_z(d) and zone daily outdoor temperature T_z(d) across all ~7,305 days
via sparse cell-weight matrix multiplication, and computes:
  - Total global building heat demand D [EJ/yr] and per-region D_c [GJ/yr] (for Eq. 6 decomposition)
  - Per-zone annual demand D_z [GJ/yr], cold-snap design peak D^peak_z [GJ/d] (99.5th percentile),
    and (12, 30) monthly empirical quantile demand distribution D_z_mq [GJ/d]
  - Per-zone urban distribution + DEA Sheet 203 substation cost c^net_z [2025 USD/GJ]
  - Per-zone counterfactual individual heating unit cost c^ind_z [2025 USD/GJ] and CO2 factor [tCO2/GJ],
    choosing min(DEA Sheet 202 condensing gas boiler, DEA Sheet 207 weather-compensated ASHP evaluated
    day-by-day with Brudermueller et al., 2025, Nat. Commun. field COP curve).
"""
from __future__ import annotations

from pathlib import Path
import netCDF4 as nc
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from code.common.parameters import (
    BAIT_THRESHOLD_CENTRAL_C,
    DH_SUBSTATION_2025,
    DH_SUBSTATION_2050,
    INDIV_ASHP_2025,
    INDIV_ASHP_2050,
    INDIV_GAS_BOILER_2025,
    INDIV_GAS_BOILER_2050,
    LTDH_NETWORK_CAPEX_MULTIPLIER,
    NATURAL_GAS_EF_TCO2_PER_GJ_LHV,
    USD_PER_EUR_2025,
)
from code.common.paths import DATA_PROCESSED, GJ_PER_MW_DAY, GJ_PER_MWH
from code.matching.solver import crf

N_QUANTILES = 30
DIST_LIFETIME_YR = 40.0


def ashp_daily_cop(tas_c: np.ndarray, use_2050_indiv_tech: bool = False) -> np.ndarray:
  """Field-calibrated radiator-supply (55 C) apartment ASHP daily COP vs outdoor air temperature T [C].

  Follows Brudermueller et al. (2025, Nat. Commun. 16, 2893) & Staffell et al. (2012, Energy Environ. Sci.),
  anchored to DEA Sheet 207 reference seasonal efficiency (2.98889 in 2025; 4.29333 in 2050).
  """
  base = np.clip(2.35 + 0.065 * tas_c + 0.00045 * (tas_c ** 2), 1.35, 4.80)
  if use_2050_indiv_tech:
    return base * (INDIV_ASHP_2050.efficiency_or_scop_ref / INDIV_ASHP_2025.efficiency_or_scop_ref)
  return base


def load_window_daily_climate(
    run_id: str,
    years: list[int] | tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Load (N_days, 52160) float32 arrays (bait_c, tas_c, months_1to12) for observed or bias-aligned ISIMIP3b run.

  run_id:
    - 'gswp3-era5__obsclim' for observed windows
    - '{model}__{ssp}' (e.g. 'gfdl-esm4__ssp245') for ISIMIP3b runs (uses {model}__historical for yr <= 2014
      and adds 2006–2025 monthly bias alignment offsets).
  """
  bait_dir = DATA_PROCESSED / "bait"
  tas_dir = DATA_PROCESSED / "tas"
  is_obs = run_id == "gswp3-era5__obsclim"

  b_off = None
  t_off = None
  if not is_obs:
    with nc.Dataset(DATA_PROCESSED / "climate_aligned/monthly_bias_offsets_2006_2025.nc", "r") as ds_off:
      runs = [str(ds_off["run"][i]) for i in range(ds_off.dimensions["run"].size)]
      r_idx = runs.index(run_id)
      b_off = np.asarray(ds_off["bait_offset_c"][r_idx, :, :], dtype=np.float32)  # (12, 52160)
      t_off = np.asarray(ds_off["tas_offset_c"][r_idx, :, :], dtype=np.float32)   # (12, 52160)
    model_name, _ = run_id.split("__")

  bait_list = []
  tas_list = []
  month_list = []

  for yr in years:
    ser = "gswp3-era5__obsclim" if is_obs else (f"{model_name}__historical" if yr <= 2014 else run_id)
    with nc.Dataset(bait_dir / ser / f"year_{yr}.nc", "r") as dsb:
      tvar = dsb["time"]
      raw_d = nc.num2date(tvar[:], tvar.units, getattr(tvar, "calendar", "proleptic_gregorian"))
      m_yr = np.array([d.month for d in raw_d], dtype=np.int32)
      b_yr = np.asarray(dsb["bait_c"][:, :], dtype=np.float32)
    with nc.Dataset(tas_dir / ser / f"year_{yr}.nc", "r") as dst:
      t_yr = np.asarray(dst["tas_c"][:, :], dtype=np.float32)

    if not is_obs and b_off is not None and t_off is not None:
      b_yr = b_yr + b_off[m_yr - 1, :]
      t_yr = t_yr + t_off[m_yr - 1, :]

    bait_list.append(b_yr)
    tas_list.append(t_yr)
    month_list.append(m_yr)

  return np.concatenate(bait_list, axis=0), np.concatenate(tas_list, axis=0), np.concatenate(month_list, axis=0)


def evaluate_zones_and_global_demand(
    bait_daily: np.ndarray,                 # (N_days, 52160) float32
    tas_daily: np.ndarray,                  # (N_days, 52160) float32
    months_daily: np.ndarray,               # (N_days,) int32 in 1..12
    df_zones: pd.DataFrame,
    df_zone_weights: pd.DataFrame,
    df_reg_cells: pd.DataFrame,
    df_reg_params: pd.DataFrame,
    bait_threshold_c: float = BAIT_THRESHOLD_CENTRAL_C,
    discount_rate: float = 0.07,
    use_retail_prices: bool = False,
    price_multiplier: float = 1.0,
    is_4gdh: bool = False,
    use_2050_dh_tech: bool = False,
    use_2050_indiv_tech: bool = False,
    use_2050_grid_ef: bool = False,
    kc_retrofit_mult_by_region: dict[str, float] | None = None,
    ghsl_epoch: int = 2025,
) -> tuple[dict[str, dict[str, float]], dict[str, np.ndarray], pd.DataFrame]:
  """Compute zone_meta, zone_dmq_gj_d, and regional total demand table for one 20-year window."""
  n_days = bait_daily.shape[0]
  n_years = n_days / 365.25

  # 1. Cell-level mean annual HDD [C*day/yr] for total regional demand D_c (all urban + rural buildings)
  hdd_daily = np.maximum(np.float32(bait_threshold_c) - bait_daily, np.float32(0.0))
  cell_ann_hdd = hdd_daily.sum(axis=0, dtype=np.float64) / n_years  # (52160,)

  reg_valid = df_reg_params[df_reg_params["valid_for_study"]].copy().set_index("region_id")
  if abs(bait_threshold_c - 12.0) < 1e-6 and "k_c_12c_gj_per_m3_hdd" in reg_valid.columns:
    kc_col = "k_c_12c_gj_per_m3_hdd"
  elif abs(bait_threshold_c - 16.0) < 1e-6 and "k_c_16c_gj_per_m3_hdd" in reg_valid.columns:
    kc_col = "k_c_16c_gj_per_m3_hdd"
  else:
    kc_col = "k_c_gj_per_m3_hdd"

  if kc_retrofit_mult_by_region:
    kc_eff = pd.Series(
        {rid: row[kc_col] * kc_retrofit_mult_by_region.get(rid, 1.0) for rid, row in reg_valid.iterrows()}
    )
  else:
    kc_eff = reg_valid[kc_col].copy()
  wc_eff = reg_valid["w_c_gj_per_m3_day"]

  v_col = "v25_m3" if ghsl_epoch == 2025 else "v20_m3"
  rc = df_reg_cells[df_reg_cells["region_id"].isin(reg_valid.index)].copy()
  rc["kc"] = rc["region_id"].map(kc_eff)
  rc["wc"] = rc["region_id"].map(wc_eff)
  rc["ann_sh_gj"] = rc[v_col] * rc["kc"] * cell_ann_hdd[rc["cell_idx"].values]
  rc["ann_dhw_gj"] = rc[v_col] * rc["wc"] * 365.25
  rc["ann_tot_gj"] = rc["ann_sh_gj"] + rc["ann_dhw_gj"]
  df_reg_D = rc.groupby(["region_id", "iso3"], as_index=False)[["ann_sh_gj", "ann_dhw_gj", "ann_tot_gj"]].sum()

  # 2. Sparse matrix projection from 52,160 cells to N_zones
  df_z = df_zones[df_zones["region_id"].isin(reg_valid.index)].reset_index(drop=True)
  zids = df_z["zone_id"].tolist()
  zid_to_i = {z: i for i, z in enumerate(zids)}
  n_z = len(zids)

  w_sub = df_zone_weights[df_zone_weights["zone_id"].isin(zid_to_i)].copy()
  w_rows = w_sub["zone_id"].map(zid_to_i).values.astype(np.int32)
  w_cols = w_sub["cell_idx"].values.astype(np.int32)
  w_vols = w_sub["volume_m3"].values.astype(np.float64)

  z_vol_tot = df_z["volume_m3"].values.astype(np.float64)
  W_vol = csr_matrix((w_vols, (w_rows, w_cols)), shape=(n_z, 52160))
  W_frac = csr_matrix((w_vols / np.maximum(1.0, z_vol_tot[w_rows]), (w_rows, w_cols)), shape=(n_z, 52160))

  z_kc = df_z["region_id"].map(kc_eff).values.astype(np.float64)
  z_wc = df_z["region_id"].map(wc_eff).values.astype(np.float64)

  # Evaluate in chunks of 730 days to keep peak RAM < 1.5 GB even for 30,000 zones!
  z_d_daily = np.empty((n_days, n_z), dtype=np.float32)
  z_ashp_el_gj_yr = np.zeros(n_z, dtype=np.float64)

  for d0 in range(0, n_days, 730):
    d1 = min(n_days, d0 + 730)
    # hdd_chunk: (chunk, 52160) -> (chunk, n_z) = hdd_chunk @ W_vol.T
    zhdd_v = hdd_daily[d0:d1, :].astype(np.float64) @ W_vol.T  # volume-weighted degree-days [m3 * C]
    zd = zhdd_v * z_kc[None, :] + (z_vol_tot * z_wc)[None, :]  # [GJ/d]
    z_d_daily[d0:d1, :] = zd.astype(np.float32)

    ztas = tas_daily[d0:d1, :].astype(np.float64) @ W_frac.T   # zone daily air temperature [C]
    zcop = ashp_daily_cop(ztas, use_2050_indiv_tech=use_2050_indiv_tech)
    z_ashp_el_gj_yr += np.sum(zd / zcop, axis=0) / n_years

  z_ann_gj = z_d_daily.sum(axis=0, dtype=np.float64) / n_years
  z_peak_gj_d = np.percentile(z_d_daily, 99.5, axis=0).astype(np.float64)

  # Monthly 30-quantile empirical demand distribution (n_z, 12, N_QUANTILES)
  q_probs = (np.arange(N_QUANTILES, dtype=np.float64) + 0.5) / float(N_QUANTILES) * 100.0
  z_dmq = np.zeros((n_z, 12, N_QUANTILES), dtype=np.float64)
  for m in range(1, 13):
    m_slice = z_d_daily[months_daily == m, :]  # (~600, n_z)
    # np.percentile returns (N_QUANTILES, n_z) -> transpose to (n_z, N_QUANTILES)
    z_dmq[:, m - 1, :] = np.percentile(m_slice, q_probs, axis=0).T

  # 3. Equipment & network unit costs for each zone
  sub_sp = DH_SUBSTATION_2050 if use_2050_dh_tech else DH_SUBSTATION_2025
  gb_sp = INDIV_GAS_BOILER_2050 if use_2050_indiv_tech else INDIV_GAS_BOILER_2025
  hp_sp = INDIV_ASHP_2050 if use_2050_indiv_tech else INDIV_ASHP_2025

  a_dist = crf(discount_rate, DIST_LIFETIME_YR)
  a_sub = crf(discount_rate, sub_sp.lifetime_yr)
  a_gb = crf(discount_rate, gb_sp.lifetime_yr)
  a_hp = crf(discount_rate, hp_sp.lifetime_yr)
  net_cap_mult = LTDH_NETWORK_CAPEX_MULTIPLIER if is_4gdh else 1.0

  zone_meta: dict[str, dict[str, float]] = {}
  zone_dmq_dict: dict[str, np.ndarray] = {}

  el_col = "p_elec_retail_usd_per_mwh" if use_retail_prices else "p_elec_pretax_usd_per_mwh"
  ng_col = "p_gas_retail_usd_per_gj" if use_retail_prices else "p_gas_pretax_usd_per_gj"
  ef_col = "grid_ef_2050_tco2_per_mwh" if use_2050_grid_ef else "grid_ef_2025_tco2_per_mwh"

  for idx_z, zrow in enumerate(df_z.itertuples(index=False)):
    zid = str(zrow.zone_id)
    rid = str(zrow.region_id)
    rp = reg_valid.loc[rid]

    ann_gj = max(1.0, float(z_ann_gj[idx_z]))
    peak_gj_d = max(ann_gj / 365.25, float(z_peak_gj_d[idx_z]))
    peak_mw = peak_gj_d / GJ_PER_MW_DAY

    p_el = float(rp[el_col]) * price_multiplier
    p_ng = float(rp[ng_col]) * price_multiplier
    has_gas = bool(rp["has_gas"])
    grid_ef = float(rp[ef_col])
    cap_civ = float(rp["cap_scale_civil"])
    cap_eq = float(rp["cap_scale_equip"])

    # Urban distribution + building DH substation cost c^net_z [USD/GJ]
    dist_cap_usd = float(zrow.persson_dist_capex_eur2025) * net_cap_mult * USD_PER_EUR_2025 * cap_civ
    dist_fom_usd = float(zrow.persson_dist_fom_eur2025_yr) * net_cap_mult * USD_PER_EUR_2025 * cap_civ
    sub_cap_usd = sub_sp.capex_eur_per_mw_h * peak_mw * USD_PER_EUR_2025 * cap_eq
    sub_fom_usd = sub_sp.fixed_om_eur_per_mw_h_yr * peak_mw * USD_PER_EUR_2025 * cap_eq
    sub_el_usd = (sub_sp.aux_elec_mwh_per_mw_h_yr * peak_mw) * p_el
    c_net_usd_gj = (a_dist * dist_cap_usd + dist_fom_usd + a_sub * sub_cap_usd + sub_fom_usd + sub_el_usd) / ann_gj

    # Counterfactual Option B: Weather-compensated Apartment ASHP (DEA Sheet 207)
    hp_cap_usd = hp_sp.capex_eur_per_mw_h * peak_mw * USD_PER_EUR_2025 * cap_eq
    hp_fom_usd = hp_sp.fixed_om_eur_per_mw_h_yr * peak_mw * USD_PER_EUR_2025 * cap_eq
    hp_el_mwh = (float(z_ashp_el_gj_yr[idx_z]) / GJ_PER_MWH) + (hp_sp.aux_elec_mwh_per_mw_h_yr * peak_mw)
    c_ashp_usd_gj = (a_hp * hp_cap_usd + hp_fom_usd + hp_el_mwh * p_el) / ann_gj
    co2_ashp_t_gj = (hp_el_mwh * grid_ef) / ann_gj

    # Counterfactual Option A: Condensing Gas Boiler (DEA Sheet 202, if gas network exists)
    if has_gas and np.isfinite(p_ng) and p_ng > 0.0:
      gb_cap_usd = gb_sp.capex_eur_per_mw_h * peak_mw * USD_PER_EUR_2025 * cap_eq
      gb_fom_usd = gb_sp.fixed_om_eur_per_mw_h_yr * peak_mw * USD_PER_EUR_2025 * cap_eq
      gb_fuel_gj = ann_gj / gb_sp.efficiency_or_scop_ref
      gb_aux_mwh = gb_sp.aux_elec_mwh_per_mw_h_yr * peak_mw
      c_gb_usd_gj = (a_gb * gb_cap_usd + gb_fom_usd + gb_fuel_gj * p_ng + gb_aux_mwh * p_el) / ann_gj
      co2_gb_t_gj = (gb_fuel_gj * NATURAL_GAS_EF_TCO2_PER_GJ_LHV + gb_aux_mwh * grid_ef) / ann_gj
      if c_gb_usd_gj < c_ashp_usd_gj:
        c_ind_usd_gj = c_gb_usd_gj
        cf_co2_t_gj = co2_gb_t_gj
      else:
        c_ind_usd_gj = c_ashp_usd_gj
        cf_co2_t_gj = co2_ashp_t_gj
    else:
      c_ind_usd_gj = c_ashp_usd_gj
      cf_co2_t_gj = co2_ashp_t_gj

    zone_meta[zid] = {
        "ann_gj": ann_gj,
        "peak_gj_d": peak_gj_d,
        "lambda_dist": float(zrow.lambda_dist_3gdh),
        "c_ind_usd_per_gj": float(c_ind_usd_gj),
        "cf_co2_t_per_gj": float(cf_co2_t_gj),
        "c_net_usd_per_gj": float(c_net_usd_gj),
        "p_elec_usd_per_mwh": float(rp[el_col]),
        "p_gas_usd_per_gj": float(rp[ng_col]) if has_gas else 0.0,
        "has_gas": 1.0 if has_gas else 0.0,
        "grid_ef_tco2_per_mwh": grid_ef,
        "cap_scale_civil": cap_civ,
        "cap_scale_equip": cap_eq,
        "dist_sub_capex_usd_full": float(dist_cap_usd + sub_cap_usd),
    }
    zone_dmq_dict[zid] = z_dmq[idx_z]

  return zone_meta, zone_dmq_dict, df_reg_D
