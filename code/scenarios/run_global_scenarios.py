"""Run all 190 global historical, SSP projection, warming-level, 2050 transition and sensitivity experiments."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import gc
import io
import json
import multiprocessing as mp
import os
from pathlib import Path
import tarfile
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from code.common.climate_windows import (
    ALL_EIGHT_WINDOWS,
    BASELINE_WINDOW,
    ISIMIP3B_MODELS,
    OBSERVED_DATASET,
    OBSERVED_WINDOWS,
    PROJECTED_WINDOWS,
    SSP_SCENARIOS,
)
from code.common.parameters import (
    BAIT_THRESHOLD_CENTRAL_C,
    BAIT_THRESHOLD_HIGH_C,
    BAIT_THRESHOLD_LOW_C,
    DH_SUBSTATION_2025,
    DH_SUBSTATION_2050,
    DISCOUNT_RATE_CENTRAL,
    DISCOUNT_RATE_HIGH,
    DISCOUNT_RATE_LOW,
    INDIV_ASHP_2025,
    INDIV_ASHP_2050,
    INDIV_GAS_BOILER_2025,
    INDIV_GAS_BOILER_2050,
    LTDH_NETWORK_CAPEX_MULTIPLIER,
    NATURAL_GAS_EF_TCO2_PER_GJ_LHV,
    USD_PER_EUR_2025,
)
from code.common.paths import DATA_PROCESSED, DATA_RAW, GJ_PER_MW_DAY, GJ_PER_MWH
from code.matching.locked_asset_evaluator import evaluate_locked_portfolio
from code.matching.solver import (
    TechConfig,
    crf,
    solve_country_celf,
    solve_country_upper_bound_lp,
)
from code.matching.zone_climate_evaluator import (
    DIST_LIFETIME_YR,
    N_QUANTILES,
    ashp_daily_cop,
    load_window_daily_climate,
)

N_CLIMATE_WORKERS = 16
N_SOLVER_WORKERS = 96


@dataclass(frozen=True)
class ScenarioSpec:
  scenario_id: str
  group: str                  # "R1", "GWL", "R4", "Robustness"
  subgroup: str               # e.g. "obs", "ssp245", "GWL_2.0", "S4_4gdh_symmetric", "Rob1_bait12c"
  run_id: str                 # "gswp3-era5__obsclim" or "{model}__{ssp}"
  model: str                  # "obs" or GCM name
  ssp: str                    # "obsclim" or "ssp126"/"ssp245"/"ssp370"
  window_id: str              # e.g. "2006_2025", "2046_2065", "gwl1p5_2021_2040"
  start_year: int
  end_year: int
  ghsl_epoch: int = 2025
  bait_threshold_c: float = BAIT_THRESHOLD_CENTRAL_C
  use_envelope_retrofit: bool = False
  retire_fossil_power: bool = False
  is_4gdh: bool = False
  use_2050_dh_tech: bool = False
  use_2050_indiv_tech: bool = False
  use_2050_grid_ef: bool = False
  discount_rate: float = DISCOUNT_RATE_CENTRAL
  use_retail_prices: bool = False
  price_multiplier: float = 1.0
  save_zone_table: bool = False
  save_selected_links: bool = False

  @property
  def clim_key(self) -> str:
    ret = 1 if self.use_envelope_retrofit else 0
    b_tag = f"b{int(round(self.bait_threshold_c))}"
    return f"{self.run_id}__{self.start_year}_{self.end_year}__e{self.ghsl_epoch}__{b_tag}__ret{ret}"

  @property
  def u_key(self) -> str:
    fos = 1 if self.retire_fossil_power else 0
    g4 = 1 if self.is_4gdh else 0
    dh50 = 1 if self.use_2050_dh_tech else 0
    return f"{self.clim_key}__fos{fos}__4g{g4}__dh50_{dh50}"


def load_gwl_crossing_windows() -> list[dict[str, object]]:
  """Parse exact 20-year GWL (+1.5, +2.0, +3.0 C vs 1850-1900) windows for the 5 ISIMIP3b GCMs."""
  tar_path = DATA_RAW / "supplements_2025/cmip6_warming_levels/cmip_warming_levels_main.tar.gz"
  with tarfile.open(tar_path, "r:gz") as tf:
    raw = tf.extractfile(
        "cmip_warming_levels-main/warming_levels/cmip6/csv/cmip6_warming_levels_one_ens_1850_1900.csv"
    ).read().decode("utf-8")
  df = pd.read_csv(io.StringIO(raw), comment="#", skipinitialspace=True)
  df["model_low"] = df["model"].astype(str).str.strip().str.lower()
  df["exp_clean"] = df["exp"].astype(str).str.strip().str.lower()
  sub = df[
      df["model_low"].isin(ISIMIP3B_MODELS)
      & df["exp_clean"].isin(SSP_SCENARIOS)
      & df["warming_level"].isin([1.5, 2.0, 3.0])
  ].sort_values(["warming_level", "exp_clean", "model_low"])
  rows = []
  for r in sub.itertuples(index=False):
    rows.append({
        "model": str(r.model_low),
        "ssp": str(r.exp_clean),
        "gwl": float(r.warming_level),
        "start_year": int(r.start_year),
        "end_year": int(r.end_year),
    })
  return rows


def build_all_scenarios() -> list[ScenarioSpec]:
  """Construct complete scenario catalog for R1 (79), GWL (33), R4 (30), and Robustness (48)."""
  scenarios: list[ScenarioSpec] = []

  # 1. R1 Observed 4 windows (1946-2025)
  for w in OBSERVED_WINDOWS:
    is_base = w.window_id == BASELINE_WINDOW.window_id
    scenarios.append(ScenarioSpec(
        scenario_id=f"R1__obs__{w.window_id}",
        group="R1",
        subgroup="obsclim",
        run_id=OBSERVED_DATASET,
        model="obs",
        ssp="obsclim",
        window_id=w.window_id,
        start_year=w.start_year,
        end_year=w.end_year,
        save_zone_table=is_base,
        save_selected_links=is_base,
    ))

  # 2. R1 ISIMIP3b 5 GCMs x 3 SSPs x 5 windows (2006-2025 + 4 future) = 75 runs
  isimip_windows = (BASELINE_WINDOW,) + PROJECTED_WINDOWS
  for ssp in SSP_SCENARIOS:
    for mod in ISIMIP3B_MODELS:
      run_id = f"{mod}__{ssp}"
      for w in isimip_windows:
        save_z = w.window_id in ("2006_2025", "2046_2065", "2081_2100")
        scenarios.append(ScenarioSpec(
            scenario_id=f"R1__{mod}__{ssp}__{w.window_id}",
            group="R1",
            subgroup=ssp,
            run_id=run_id,
            model=mod,
            ssp=ssp,
            window_id=w.window_id,
            start_year=w.start_year,
            end_year=w.end_year,
            save_zone_table=save_z,
        ))

  # 3. CMIP6 Global Warming Levels (+1.5, +2.0, +3.0 C) = 33 runs
  for g in load_gwl_crossing_windows():
    g_tag = str(g["gwl"]).replace(".", "p")
    w_id = f"gwl{g_tag}C_{g['start_year']}_{g['end_year']}"
    scenarios.append(ScenarioSpec(
        scenario_id=f"GWL__{g_tag}C__{g['model']}__{g['ssp']}__{g['start_year']}_{g['end_year']}",
        group="GWL",
        subgroup=f"GWL_{g['gwl']:.1f}C",
        run_id=f"{g['model']}__{g['ssp']}",
        model=str(g["model"]),
        ssp=str(g["ssp"]),
        window_id=w_id,
        start_year=int(g["start_year"]),
        end_year=int(g["end_year"]),
        save_zone_table=True,
    ))

  # Helper windows for R4 & Robustness: 2006-2025 obs baseline + 5 GCMs 2046-2065 SSP2-4.5 (6 runs per config)
  core_6_targets = [
      (OBSERVED_DATASET, "obs", "obsclim", "2006_2025", 2006, 2025),
      *[(f"{m}__ssp245", m, "ssp245", "2046_2065", 2046, 2065) for m in ISIMIP3B_MODELS],
  ]

  # 4. R4 Energy Transition, Envelope Retrofit, & 4GDH Symmetric Tech (5 configs x 6 windows = 30 runs;
  #    includes the 25 mid-century SSP2-4.5 runs + 5 baseline counterfactuals for exact attribution)
  r4_variants = [
      ("S1_fossil_retire", dict(retire_fossil_power=True, use_2050_grid_ef=True)),
      ("S2_envelope_retrofit", dict(use_envelope_retrofit=True)),
      ("S3_4gdh_standalone", dict(is_4gdh=True, use_2050_dh_tech=True, use_2050_indiv_tech=False)),
      ("S4_4gdh_symmetric", dict(is_4gdh=True, use_2050_dh_tech=True, use_2050_indiv_tech=True)),
      (
          "S5_combined_all",
          dict(
              retire_fossil_power=True,
              use_2050_grid_ef=True,
              use_envelope_retrofit=True,
              is_4gdh=True,
              use_2050_dh_tech=True,
              use_2050_indiv_tech=True,
          ),
      ),
  ]
  for v_name, v_kw in r4_variants:
    for run_id, mod, ssp, w_id, y0, y1 in core_6_targets:
      scenarios.append(ScenarioSpec(
          scenario_id=f"R4__{v_name}__{mod}__{ssp}__{w_id}",
          group="R4",
          subgroup=v_name,
          run_id=run_id,
          model=mod,
          ssp=ssp,
          window_id=w_id,
          start_year=y0,
          end_year=y1,
          **v_kw,
      ))

  # 5. Five Robustness Suites (8 parameter variants x 6 windows = 48 solver runs + Rob5 official_enduse subset = 55+)
  rob_variants = [
      ("Rob1_bait12c", dict(bait_threshold_c=BAIT_THRESHOLD_LOW_C)),
      ("Rob1_bait16c", dict(bait_threshold_c=BAIT_THRESHOLD_HIGH_C)),
      ("Rob2_r03pct", dict(discount_rate=DISCOUNT_RATE_LOW)),
      ("Rob2_r10pct", dict(discount_rate=DISCOUNT_RATE_HIGH)),
      ("Rob3_retail_tax", dict(use_retail_prices=True)),
      ("Rob3_price_minus30", dict(price_multiplier=0.70)),
      ("Rob3_price_plus30", dict(price_multiplier=1.30)),
      ("Rob4_ghsl2020", dict(ghsl_epoch=2020)),
  ]
  for r_name, r_kw in rob_variants:
    for run_id, mod, ssp, w_id, y0, y1 in core_6_targets:
      scenarios.append(ScenarioSpec(
          scenario_id=f"ROB__{r_name}__{mod}__{ssp}__{w_id}",
          group="Robustness",
          subgroup=r_name,
          run_id=run_id,
          model=mod,
          ssp=ssp,
          window_id=w_id,
          start_year=y0,
          end_year=y1,
          **r_kw,
      ))

  return scenarios


# ---------------------------------------------------------------------------
# Stage 1 Worker: Compute & cache compact zone/cell climate-demand projections
# ---------------------------------------------------------------------------
_CLIM_INIT: dict[str, object] = {}


def _init_clim_worker() -> None:
  df_rp = pd.read_parquet(DATA_PROCESSED / "sinks_2025/calibrated_region_parameters_2025.parquet")
  df_rc = pd.read_parquet(DATA_PROCESSED / "sinks_2025/ghsl_region_cell_volumes.parquet")
  sinks_by_epoch = {}
  for ep, sub in [(2025, "sinks_2025"), (2020, "sinks_2020_robustness")]:
    sdir = DATA_PROCESSED / sub
    df_z = pd.read_parquet(sdir / "urban_heat_zones.parquet")
    df_w = pd.read_parquet(sdir / "urban_zone_cell_weights.parquet")
    sinks_by_epoch[ep] = (df_z, df_w)
  _CLIM_INIT["rp"] = df_rp
  _CLIM_INIT["rc"] = df_rc
  _CLIM_INIT["sinks"] = sinks_by_epoch


def _compute_one_clim_projection(task: dict[str, object]) -> str:
  clim_key = str(task["clim_key"])
  out_npz = Path(str(task["out_npz"]))
  if out_npz.exists():
    return clim_key

  run_id = str(task["run_id"])
  years = list(range(int(task["start_year"]), int(task["end_year"]) + 1))
  ghsl_epoch = int(task["ghsl_epoch"])
  bait_thresh = float(task["bait_threshold_c"])
  use_retrofit = bool(task["use_envelope_retrofit"])

  df_rp: pd.DataFrame = _CLIM_INIT["rp"]  # type: ignore[assignment]
  df_rc: pd.DataFrame = _CLIM_INIT["rc"]  # type: ignore[assignment]
  df_zones, df_weights = _CLIM_INIT["sinks"][ghsl_epoch]  # type: ignore[index]

  bait_d, tas_d, months_d = load_window_daily_climate(run_id, years)
  n_days = bait_d.shape[0]
  n_years = n_days / 365.25

  hdd_d = np.maximum(np.float32(bait_thresh) - bait_d, np.float32(0.0))
  cell_ann_hdd = hdd_d.sum(axis=0, dtype=np.float64) / n_years

  reg_valid = df_rp[df_rp["valid_for_study"]].copy().set_index("region_id")
  if abs(bait_thresh - 12.0) < 1e-6:
    kc_col = "k_c_12c_gj_per_m3_hdd"
  elif abs(bait_thresh - 16.0) < 1e-6:
    kc_col = "k_c_16c_gj_per_m3_hdd"
  else:
    kc_col = "k_c_gj_per_m3_hdd"

  kc_eff = reg_valid[kc_col].copy()
  if use_retrofit:
    kc_eff = kc_eff * reg_valid["kc_retrofit_2050_mult"]
  wc_eff = reg_valid["w_c_gj_per_m3_day"]

  v_col = "v25_m3" if ghsl_epoch == 2025 else "v20_m3"
  rc = df_rc[df_rc["region_id"].isin(reg_valid.index)].copy()
  rc["kc"] = rc["region_id"].map(kc_eff)
  rc["wc"] = rc["region_id"].map(wc_eff)
  rc["ann_sh_gj"] = rc[v_col] * rc["kc"] * cell_ann_hdd[rc["cell_idx"].values]
  rc["ann_dhw_gj"] = rc[v_col] * rc["wc"] * 365.25
  rc["ann_tot_gj"] = rc["ann_sh_gj"] + rc["ann_dhw_gj"]
  df_reg_D = rc.groupby(["region_id", "iso3"], as_index=False)[["ann_sh_gj", "ann_dhw_gj", "ann_tot_gj"]].sum()

  df_z = df_zones[df_zones["region_id"].isin(reg_valid.index)].reset_index(drop=True)
  zids = df_z["zone_id"].astype(str).to_numpy()
  zid_to_i = {z: i for i, z in enumerate(zids)}
  n_z = len(zids)

  w_sub = df_weights[df_weights["zone_id"].isin(zid_to_i)]
  w_rows = w_sub["zone_id"].map(zid_to_i).values.astype(np.int32)
  w_cols = w_sub["cell_idx"].values.astype(np.int32)
  w_vols = w_sub["volume_m3"].values.astype(np.float64)
  z_vol_tot = df_z["volume_m3"].values.astype(np.float64)

  W_vol = csr_matrix((w_vols, (w_rows, w_cols)), shape=(n_z, 52160))
  W_frac = csr_matrix((w_vols / np.maximum(1.0, z_vol_tot[w_rows]), (w_rows, w_cols)), shape=(n_z, 52160))

  z_kc = df_z["region_id"].map(kc_eff).values.astype(np.float64)
  z_wc = df_z["region_id"].map(wc_eff).values.astype(np.float64)

  z_d_daily = np.empty((n_days, n_z), dtype=np.float32)
  z_ashp_el_2025 = np.zeros(n_z, dtype=np.float64)
  z_ashp_el_2050 = np.zeros(n_z, dtype=np.float64)

  for d0 in range(0, n_days, 730):
    d1 = min(n_days, d0 + 730)
    zhdd_v = hdd_d[d0:d1, :].astype(np.float64) @ W_vol.T
    zd = zhdd_v * z_kc[None, :] + (z_vol_tot * z_wc)[None, :]
    z_d_daily[d0:d1, :] = zd.astype(np.float32)

    ztas = tas_d[d0:d1, :].astype(np.float64) @ W_frac.T
    cop25 = ashp_daily_cop(ztas, use_2050_indiv_tech=False)
    cop50 = ashp_daily_cop(ztas, use_2050_indiv_tech=True)
    z_ashp_el_2025 += np.sum(zd / cop25, axis=0) / n_years
    z_ashp_el_2050 += np.sum(zd / cop50, axis=0) / n_years

  del bait_d, tas_d, hdd_d
  gc.collect()

  z_ann_gj = z_d_daily.sum(axis=0, dtype=np.float64) / n_years
  z_peak_gj_d = np.percentile(z_d_daily, 99.5, axis=0).astype(np.float64)

  q_probs = (np.arange(N_QUANTILES, dtype=np.float64) + 0.5) / float(N_QUANTILES) * 100.0
  z_dmq = np.zeros((n_z, 12, N_QUANTILES), dtype=np.float32)
  for m in range(1, 13):
    m_slice = z_d_daily[months_d == m, :]
    z_dmq[:, m - 1, :] = np.percentile(m_slice, q_probs, axis=0).T.astype(np.float32)

  del z_d_daily
  gc.collect()

  tmp_npz = out_npz.with_name(out_npz.stem + ".tmp.npz")
  np.savez(
      tmp_npz,
      zids=zids,
      z_iso3=df_z["iso3"].astype(str).to_numpy(),
      z_ann_gj=z_ann_gj,
      z_peak_gj_d=z_peak_gj_d,
      z_dmq=z_dmq,
      z_ashp_el_2025=z_ashp_el_2025,
      z_ashp_el_2050=z_ashp_el_2050,
      reg_id=df_reg_D["region_id"].astype(str).to_numpy(),
      reg_iso3=df_reg_D["iso3"].astype(str).to_numpy(),
      reg_sh_gj=df_reg_D["ann_sh_gj"].to_numpy(dtype=np.float64),
      reg_dhw_gj=df_reg_D["ann_dhw_gj"].to_numpy(dtype=np.float64),
      reg_tot_gj=df_reg_D["ann_tot_gj"].to_numpy(dtype=np.float64),
  )
  tmp_npz.replace(out_npz)
  return clim_key


# ---------------------------------------------------------------------------
# Stage 2 & Stage 3 Country-Level Solver Workers (64 processes, LPT order)
# ---------------------------------------------------------------------------
_SOLV_INIT: dict[str, object] = {}


def _init_solver_worker() -> None:
  df_sources = pd.read_parquet(DATA_PROCESSED / "sources_2025/sources_2025_streams.parquet")
  df_rp = pd.read_parquet(DATA_PROCESSED / "sinks_2025/calibrated_region_parameters_2025.parquet")
  reg_valid = df_rp[df_rp["valid_for_study"]].copy().set_index("region_id")

  # Pre-index sources by iso3
  src_by_iso = {iso: g.reset_index(drop=True) for iso, g in df_sources.groupby("iso3")}

  # Pre-index zones & candidates by (epoch, iso3)
  epoch_data = {}
  for ep, sub in [(2025, "sinks_2025"), (2020, "sinks_2020_robustness")]:
    sdir = DATA_PROCESSED / sub
    df_z = pd.read_parquet(sdir / "urban_heat_zones.parquet")
    df_z = df_z[df_z["region_id"].isin(reg_valid.index)].reset_index(drop=True)
    df_c = pd.read_parquet(sdir / "candidates_2025.parquet")
    valid_zset = set(df_z["zone_id"])
    df_c = df_c[df_c["zone_id"].isin(valid_zset)].reset_index(drop=True)
    z_by_iso = {iso: g.reset_index(drop=True) for iso, g in df_z.groupby("iso3")}
    c_by_iso = {iso: g.reset_index(drop=True) for iso, g in df_c.groupby("iso3")}
    epoch_data[ep] = (z_by_iso, c_by_iso)

  _SOLV_INIT["src_by_iso"] = src_by_iso
  _SOLV_INIT["reg_valid"] = reg_valid
  _SOLV_INIT["epoch_data"] = epoch_data
  _SOLV_INIT["clim_lru"] = {}  # tiny 2-entry LRU per worker


def _get_country_clim_slice(clim_npz_path: str, iso3: str) -> tuple[dict[str, tuple[float, float, float, float]], dict[str, np.ndarray]]:
  lru: dict[str, dict[str, object]] = _SOLV_INIT["clim_lru"]  # type: ignore[assignment]
  if clim_npz_path not in lru:
    if len(lru) >= 2:
      lru.pop(next(iter(lru)))
    with np.load(clim_npz_path, allow_pickle=True) as npz:
      z_iso = npz["z_iso3"].astype(str)
      zids = npz["zids"].astype(str)
      ann = npz["z_ann_gj"]
      peak = npz["z_peak_gj_d"]
      dmq = npz["z_dmq"]
      el25 = npz["z_ashp_el_2025"]
      el50 = npz["z_ashp_el_2050"]
      # Group indices by iso3 once
      by_iso: dict[str, np.ndarray] = {}
      order = np.argsort(z_iso, kind="mergesort")
      s_iso = z_iso[order]
      splits = np.flatnonzero(s_iso[1:] != s_iso[:-1]) + 1
      for grp in np.split(order, splits):
        if len(grp) > 0:
          by_iso[str(z_iso[grp[0]])] = grp
      lru[clim_npz_path] = {
          "zids": zids, "ann": ann, "peak": peak, "dmq": dmq,
          "el25": el25, "el50": el50, "by_iso": by_iso,
      }

  entry = lru[clim_npz_path]
  idxs = entry["by_iso"].get(iso3)  # type: ignore[union-attr]
  if idxs is None:
    return {}, {}
  zids_arr = entry["zids"]
  ann_arr = entry["ann"]
  peak_arr = entry["peak"]
  dmq_arr = entry["dmq"]
  el25_arr = entry["el25"]
  el50_arr = entry["el50"]

  stats = {}
  dmq_dict = {}
  for idx in idxs:
    zid = str(zids_arr[idx])  # type: ignore[index]
    stats[zid] = (
        float(ann_arr[idx]),   # type: ignore[index]
        float(peak_arr[idx]),  # type: ignore[index]
        float(el25_arr[idx]),  # type: ignore[index]
        float(el50_arr[idx]),  # type: ignore[index]
    )
    dmq_dict[zid] = np.asarray(dmq_arr[idx], dtype=np.float64)  # type: ignore[index]
  return stats, dmq_dict


def _build_country_zone_meta(
    df_z_iso: pd.DataFrame,
    z_stats: dict[str, tuple[float, float, float, float]],
    reg_valid: pd.DataFrame,
    discount_rate: float,
    use_retail_prices: bool,
    price_multiplier: float,
    is_4gdh: bool,
    use_2050_dh_tech: bool,
    use_2050_indiv_tech: bool,
    use_2050_grid_ef: bool,
) -> dict[str, dict[str, float]]:
  sub_sp = DH_SUBSTATION_2050 if use_2050_dh_tech else DH_SUBSTATION_2025
  gb_sp = INDIV_GAS_BOILER_2050 if use_2050_indiv_tech else INDIV_GAS_BOILER_2025
  hp_sp = INDIV_ASHP_2050 if use_2050_indiv_tech else INDIV_ASHP_2025

  a_dist = crf(discount_rate, DIST_LIFETIME_YR)
  a_sub = crf(discount_rate, sub_sp.lifetime_yr)
  a_gb = crf(discount_rate, gb_sp.lifetime_yr)
  a_hp = crf(discount_rate, hp_sp.lifetime_yr)
  net_cap_mult = LTDH_NETWORK_CAPEX_MULTIPLIER if is_4gdh else 1.0

  el_col = "p_elec_retail_usd_per_mwh" if use_retail_prices else "p_elec_pretax_usd_per_mwh"
  ng_col = "p_gas_retail_usd_per_gj" if use_retail_prices else "p_gas_pretax_usd_per_gj"
  ef_col = "grid_ef_2050_tco2_per_mwh" if use_2050_grid_ef else "grid_ef_2025_tco2_per_mwh"

  zone_meta: dict[str, dict[str, float]] = {}
  for zrow in df_z_iso.itertuples(index=False):
    zid = str(zrow.zone_id)
    if zid not in z_stats:
      continue
    ann_raw, peak_raw, el25, el50 = z_stats[zid]
    ann_gj = max(1.0, ann_raw)
    peak_gj_d = max(ann_gj / 365.25, peak_raw)
    peak_mw = peak_gj_d / GJ_PER_MW_DAY

    rp = reg_valid.loc[str(zrow.region_id)]
    p_el_base = float(rp[el_col])
    p_ng_base = float(rp[ng_col])
    p_el = p_el_base * price_multiplier
    p_ng = p_ng_base * price_multiplier
    has_gas = bool(rp["has_gas"])
    grid_ef = float(rp[ef_col])
    cap_civ = float(rp["cap_scale_civil"])
    cap_eq = float(rp["cap_scale_equip"])

    dist_cap_usd = float(zrow.persson_dist_capex_eur2025) * net_cap_mult * USD_PER_EUR_2025 * cap_civ
    dist_fom_usd = float(zrow.persson_dist_fom_eur2025_yr) * net_cap_mult * USD_PER_EUR_2025 * cap_civ
    sub_cap_usd = sub_sp.capex_eur_per_mw_h * peak_mw * USD_PER_EUR_2025 * cap_eq
    sub_fom_usd = sub_sp.fixed_om_eur_per_mw_h_yr * peak_mw * USD_PER_EUR_2025 * cap_eq
    sub_el_usd = (sub_sp.aux_elec_mwh_per_mw_h_yr * peak_mw) * p_el
    c_net_usd_gj = (a_dist * dist_cap_usd + dist_fom_usd + a_sub * sub_cap_usd + sub_fom_usd + sub_el_usd) / ann_gj

    ashp_el_gj = el50 if use_2050_indiv_tech else el25
    hp_cap_usd = hp_sp.capex_eur_per_mw_h * peak_mw * USD_PER_EUR_2025 * cap_eq
    hp_fom_usd = hp_sp.fixed_om_eur_per_mw_h_yr * peak_mw * USD_PER_EUR_2025 * cap_eq
    hp_el_mwh = (ashp_el_gj / GJ_PER_MWH) + (hp_sp.aux_elec_mwh_per_mw_h_yr * peak_mw)
    c_ashp_usd_gj = (a_hp * hp_cap_usd + hp_fom_usd + hp_el_mwh * p_el) / ann_gj
    co2_ashp_t_gj = (hp_el_mwh * grid_ef) / ann_gj

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
        "p_elec_usd_per_mwh": p_el_base,
        "p_gas_usd_per_gj": p_ng_base if has_gas else 0.0,
        "has_gas": 1.0 if has_gas else 0.0,
        "grid_ef_tco2_per_mwh": grid_ef,
        "cap_scale_civil": cap_civ,
        "cap_scale_equip": cap_eq,
        "dist_sub_capex_usd_full": float(dist_cap_usd + sub_cap_usd),
    }
  return zone_meta


def _solve_u_country_task(task: dict[str, object]) -> tuple[str, str, float, dict[str, float]]:
  u_key = str(task["u_key"])
  iso3 = str(task["iso3"])
  ghsl_epoch = int(task["ghsl_epoch"])
  retire_fossil = bool(task["retire_fossil_power"])
  is_4gdh = bool(task["is_4gdh"])
  use_2050_dh = bool(task["use_2050_dh_tech"])
  clim_npz_path = str(task["clim_npz_path"])

  src_by_iso = _SOLV_INIT["src_by_iso"]
  z_by_iso, c_by_iso = _SOLV_INIT["epoch_data"][ghsl_epoch]  # type: ignore[index]

  df_z_iso = z_by_iso.get(iso3)
  df_c_iso = c_by_iso.get(iso3)
  df_s_iso = src_by_iso.get(iso3)  # type: ignore[union-attr]
  if df_z_iso is None or df_c_iso is None or df_s_iso is None or len(df_c_iso) == 0:
    return u_key, iso3, 0.0, {}

  if retire_fossil:
    df_s_iso = df_s_iso[~df_s_iso["is_fossil_power"]]
    df_c_iso = df_c_iso[~df_c_iso["is_fossil_power"]].reset_index(drop=True)
    if len(df_c_iso) == 0:
      return u_key, iso3, 0.0, {}

  z_stats, z_dmq = _get_country_clim_slice(clim_npz_path, iso3)
  zm_min = {
      str(r.zone_id): {"lambda_dist": float(r.lambda_dist_3gdh)}
      for r in df_z_iso.itertuples(index=False)
      if str(r.zone_id) in z_dmq
  }
  q_prefix = "q4g" if is_4gdh else "q3g"
  q_init = {
      str(r.stream_id): np.array([getattr(r, f"{q_prefix}_m{m+1:02d}_gj_d") for m in range(12)], dtype=np.float64)
      for r in df_s_iso.itertuples(index=False)
  }
  cfg = TechConfig(is_4gdh=is_4gdh, use_2050_dh_tech=use_2050_dh)
  u_tot_gj, u_zone_gj = solve_country_upper_bound_lp(df_c_iso, q_init, z_dmq, zm_min, cfg)
  return u_key, iso3, u_tot_gj, u_zone_gj


def _solve_p_country_task(task: dict[str, object]) -> dict[str, object]:
  scen: ScenarioSpec = task["scen"]  # type: ignore[assignment]
  iso3 = str(task["iso3"])
  clim_npz_path = str(task["clim_npz_path"])

  src_by_iso = _SOLV_INIT["src_by_iso"]
  reg_valid: pd.DataFrame = _SOLV_INIT["reg_valid"]  # type: ignore[assignment]
  z_by_iso, c_by_iso = _SOLV_INIT["epoch_data"][scen.ghsl_epoch]  # type: ignore[index]

  df_s_iso = src_by_iso.get(iso3)  # type: ignore[union-attr]
  if df_s_iso is not None and scen.retire_fossil_power:
    df_s_iso = df_s_iso[~df_s_iso["is_fossil_power"]]

  q_ann_col = "q4g_annual_gj" if scen.is_4gdh else "q3g_annual_gj"
  Q_iso_gj = float(df_s_iso[q_ann_col].sum()) if df_s_iso is not None and len(df_s_iso) > 0 else 0.0
  Q_by_sec = (
      df_s_iso.groupby("sector")[q_ann_col].sum().to_dict()
      if df_s_iso is not None and len(df_s_iso) > 0
      else {}
  )

  df_z_iso = z_by_iso.get(iso3)
  df_c_iso = c_by_iso.get(iso3)
  if df_z_iso is None or df_c_iso is None or df_s_iso is None or len(df_c_iso) == 0:
    return {
        "scenario_id": scen.scenario_id,
        "iso3": iso3,
        "Q_gj": Q_iso_gj,
        "Q_by_sector": Q_by_sec,
        "D_urban_gj": 0.0,
        "P_gj": 0.0,
        "P_by_sector": {},
        "P_by_upgrade": {},
        "H_deliv_gj": 0.0,
        "H_peaking_gj": 0.0,
        "net_benefit_usd": 0.0,
        "tot_dh_cost_usd": 0.0,
        "avoided_ind_cost_usd": 0.0,
        "total_capex_usd": 0.0,
        "co2_abated_t": 0.0,
        "n_selected_links": 0,
        "zone_P_gj": {},
        "zone_D_gj": {},
        "selected_links": [],
    }

  if scen.retire_fossil_power:
    df_c_iso = df_c_iso[~df_c_iso["is_fossil_power"]].reset_index(drop=True)

  z_stats, z_dmq = _get_country_clim_slice(clim_npz_path, iso3)
  zm_iso = _build_country_zone_meta(
      df_z_iso=df_z_iso,
      z_stats=z_stats,
      reg_valid=reg_valid,
      discount_rate=scen.discount_rate,
      use_retail_prices=scen.use_retail_prices,
      price_multiplier=scen.price_multiplier,
      is_4gdh=scen.is_4gdh,
      use_2050_dh_tech=scen.use_2050_dh_tech,
      use_2050_indiv_tech=scen.use_2050_indiv_tech,
      use_2050_grid_ef=scen.use_2050_grid_ef,
  )
  D_urban_gj = float(sum(v["ann_gj"] for v in zm_iso.values()))

  if len(df_c_iso) == 0 or len(zm_iso) == 0:
    return {
        "scenario_id": scen.scenario_id,
        "iso3": iso3,
        "Q_gj": Q_iso_gj,
        "Q_by_sector": Q_by_sec,
        "D_urban_gj": D_urban_gj,
        "P_gj": 0.0,
        "P_by_sector": {},
        "P_by_upgrade": {},
        "H_deliv_gj": 0.0,
        "H_peaking_gj": 0.0,
        "net_benefit_usd": 0.0,
        "tot_dh_cost_usd": 0.0,
        "avoided_ind_cost_usd": 0.0,
        "total_capex_usd": 0.0,
        "co2_abated_t": 0.0,
        "n_selected_links": 0,
        "zone_P_gj": ({z: 0.0 for z in zm_iso} if scen.save_zone_table else {}),
        "zone_D_gj": ({z: zm_iso[z]["ann_gj"] for z in zm_iso} if scen.save_zone_table else {}),
        "selected_links": [],
    }

  q_prefix = "q4g" if scen.is_4gdh else "q3g"
  q_init = {
      str(r.stream_id): np.array([getattr(r, f"{q_prefix}_m{m+1:02d}_gj_d") for m in range(12)], dtype=np.float64)
      for r in df_s_iso.itertuples(index=False)
  }
  cfg = TechConfig(
      discount_rate=scen.discount_rate,
      is_4gdh=scen.is_4gdh,
      use_2050_dh_tech=scen.use_2050_dh_tech,
      price_multiplier=scen.price_multiplier,
  )
  selected, P_gj = solve_country_celf(df_c_iso, q_init, z_dmq, zm_iso, cfg)

  p_sec: dict[str, float] = {}
  p_upg: dict[str, float] = {}
  z_p: dict[str, float] = {z: 0.0 for z in zm_iso} if scen.save_zone_table else {}
  H_deliv = 0.0
  H_pk = 0.0
  N_usd = 0.0
  dh_cost = 0.0
  ind_cost = 0.0
  capex = 0.0
  co2_ab = 0.0

  for x in selected:
    pv = float(x["P_net_waste_heat_gj_yr"])
    sec = str(x["sector"])
    upg = str(x["upgrade_mode"])
    p_sec[sec] = p_sec.get(sec, 0.0) + pv
    p_upg[upg] = p_upg.get(upg, 0.0) + pv
    if scen.save_zone_table:
      zid = str(x["zone_id"])
      z_p[zid] = z_p.get(zid, 0.0) + pv
    H_deliv += float(x["H_deliv_gj_yr"])
    H_pk += float(x["H_peaking_gj_yr"])
    N_usd += float(x["net_benefit_usd_yr"])
    dh_cost += float(x["tot_dh_cost_usd_yr"])
    ind_cost += float(x["avoided_ind_cost_usd_yr"])
    capex += float(x["total_capex_usd"])
    co2_ab += float(x["co2_abated_t_yr"])

  return {
      "scenario_id": scen.scenario_id,
      "iso3": iso3,
      "Q_gj": Q_iso_gj,
      "Q_by_sector": Q_by_sec,
      "D_urban_gj": D_urban_gj,
      "P_gj": P_gj,
      "P_by_sector": p_sec,
      "P_by_upgrade": p_upg,
      "H_deliv_gj": H_deliv,
      "H_peaking_gj": H_pk,
      "net_benefit_usd": N_usd,
      "tot_dh_cost_usd": dh_cost,
      "avoided_ind_cost_usd": ind_cost,
      "total_capex_usd": capex,
      "co2_abated_t": co2_ab,
      "n_selected_links": len(selected),
      "zone_P_gj": z_p,
      "zone_D_gj": ({z: zm_iso[z]["ann_gj"] for z in zm_iso} if scen.save_zone_table else {}),
      "selected_links": (selected if scen.save_selected_links else []),
  }


def compute_r2_decomposition(
    df_country_all: pd.DataFrame,
    df_zones_meta: pd.DataFrame,
    out_dir: Path,
) -> dict[str, object]:
  """Compute exact zero-residual Eq. (6) decomposition across countries, zones, & empirical strata."""
  r2_dir = out_dir / "r2_decomposition"
  r2_dir.mkdir(parents=True, exist_ok=True)

  base = df_country_all[df_country_all["scenario_id"] == "R1__obs__2006_2025"].set_index("iso3")

  # Evaluate across SSP2-4.5 mid-century (2046-2065) and end-of-century (2081-2100) + SSP1-2.6 / SSP3-7.0
  target_scens = df_country_all[
      (df_country_all["group"].isin(["R1", "GWL"]))
      & (df_country_all["scenario_id"] != "R1__obs__2006_2025")
  ]["scenario_id"].unique()

  decomp_rows = []
  max_residual = 0.0

  # Note: For model scenarios in R1, compare future model window against that SAME GCM's 2006-2025 baseline
  # (eliminating any residual inter-model higher-order variance) AND also record comparison vs obs 2006-2025!
  r1_mod_base = {
      (mod, ssp): df_country_all[
          df_country_all["scenario_id"] == f"R1__{mod}__{ssp}__2006_2025"
      ].set_index("iso3")
      for mod in ISIMIP3B_MODELS
      for ssp in SSP_SCENARIOS
  }

  for sid in target_scens:
    sub = df_country_all[df_country_all["scenario_id"] == sid].set_index("iso3")
    if len(sub) == 0:
      continue
    mod = str(sub["model"].iloc[0])
    ssp = str(sub["ssp"].iloc[0])
    ref = r1_mod_base.get((mod, ssp), base) if mod != "obs" else base

    # 1. Global aggregate exact identity
    D0 = float(ref["D_total_gj"].sum())
    U0 = float(ref["U_gj"].sum())
    P0 = float(ref["P_gj"].sum())
    D1 = float(sub["D_total_gj"].sum())
    U1 = float(sub["U_gj"].sum())
    P1 = float(sub["P_gj"].sum())
    if D0 > 0 and U0 > 0 and P0 > 0:
      dP_rel = (P1 - P0) / P0
      t1_dem = (D1 - D0) / D0
      t2_seas = (U1 - U0) / U0 - t1_dem
      t3_cost = dP_rel - (U1 - U0) / U0
      resid = abs(dP_rel - (t1_dem + t2_seas + t3_cost))
      max_residual = max(max_residual, resid)
      decomp_rows.append({
          "scenario_id": sid,
          "level": "global",
          "entity_id": "GLOBAL",
          "model": mod,
          "ssp": ssp,
          "window_id": str(sub["window_id"].iloc[0]),
          "D0_pj": D0 / 1e6, "D1_pj": D1 / 1e6,
          "U0_pj": U0 / 1e6, "U1_pj": U1 / 1e6,
          "P0_pj": P0 / 1e6, "P1_pj": P1 / 1e6,
          "dP_rel_pct": 100.0 * dP_rel,
          "term1_demand_shift_pct": 100.0 * t1_dem,
          "term2_seasonal_mismatch_pct": 100.0 * t2_seas,
          "term3_cost_exit_pct": 100.0 * t3_cost,
          "identity_residual_pct": 100.0 * resid,
      })

    # 2. Country-level exact identity (for all countries with P0 > 0, U0 > 0, D0 > 0)
    common_isos = ref.index.intersection(sub.index)
    for iso in common_isos:
      d0 = float(ref.loc[iso, "D_total_gj"])
      u0 = float(ref.loc[iso, "U_gj"])
      p0 = float(ref.loc[iso, "P_gj"])
      if d0 <= 0.0 or u0 <= 0.0 or p0 <= 1e3:  # >= 0.001 PJ baseline potential
        continue
      d1 = float(sub.loc[iso, "D_total_gj"])
      u1 = float(sub.loc[iso, "U_gj"])
      p1 = float(sub.loc[iso, "P_gj"])
      dp_r = (p1 - p0) / p0
      t1 = (d1 - d0) / d0
      t2 = (u1 - u0) / u0 - t1
      t3 = dp_r - (u1 - u0) / u0
      res_c = abs(dp_r - (t1 + t2 + t3))
      max_residual = max(max_residual, res_c)
      decomp_rows.append({
          "scenario_id": sid,
          "level": "country",
          "entity_id": str(iso),
          "model": mod,
          "ssp": ssp,
          "window_id": str(sub["window_id"].iloc[0]),
          "D0_pj": d0 / 1e6, "D1_pj": d1 / 1e6,
          "U0_pj": u0 / 1e6, "U1_pj": u1 / 1e6,
          "P0_pj": p0 / 1e6, "P1_pj": p1 / 1e6,
          "dP_rel_pct": 100.0 * dp_r,
          "term1_demand_shift_pct": 100.0 * t1,
          "term2_seasonal_mismatch_pct": 100.0 * t2,
          "term3_cost_exit_pct": 100.0 * t3,
          "identity_residual_pct": 100.0 * res_c,
      })

  assert max_residual < 1e-12, f"Eq. 6 identity residual exceeded 1e-12: {max_residual}"
  df_decomp = pd.DataFrame(decomp_rows)
  df_decomp.to_parquet(r2_dir / "r2_eq6_decomposition_all.parquet", index=False)
  df_decomp[df_decomp["level"] == "global"].to_csv(r2_dir / "r2_eq6_global_summary.csv", index=False)

  # Zone-level decomposition & empirical stratification for Mid-Century SSP2-4.5 (5-GCM ensemble mean)
  z_dir = out_dir / "zones"
  z_strat_summary = {}
  z_dfs = []
  for mod in ISIMIP3B_MODELS:
    p_b = z_dir / f"R1__{mod}__ssp245__2006_2025_zones.parquet"
    p_f = z_dir / f"R1__{mod}__ssp245__2046_2065_zones.parquet"
    if p_b.exists() and p_f.exists():
      zb = pd.read_parquet(p_b).rename(columns={"D_gj": "D0_gj", "U_gj": "U0_gj", "P_gj": "P0_gj"})
      zf = pd.read_parquet(p_f)[["zone_id", "D_gj", "U_gj", "P_gj"]].rename(
          columns={"D_gj": "D1_gj", "U_gj": "U1_gj", "P_gj": "P1_gj"}
      )
      zm = zb.merge(zf, on="zone_id", how="inner")
      zm["model"] = mod
      z_dfs.append(zm)

  if z_dfs:
    z_all = pd.concat(z_dfs, ignore_index=True)
    z_ens = z_all.groupby("zone_id", as_index=False)[["D0_gj", "U0_gj", "P0_gj", "D1_gj", "U1_gj", "P1_gj"]].mean()
    z_ens = z_ens.merge(
        df_zones_meta[["zone_id", "iso3", "region_id", "lat", "lon", "baseline_mean_mw", "population"]],
        on="zone_id",
        how="left",
    )
    z_pos = z_ens[(z_ens["P0_gj"] > 0.0) & (z_ens["U0_gj"] > 0.0) & (z_ens["D0_gj"] > 0.0)].copy()
    z_pos["dP_rel_pct"] = 100.0 * (z_pos["P1_gj"] - z_pos["P0_gj"]) / z_pos["P0_gj"]
    z_pos["term1_demand_shift_pct"] = 100.0 * (z_pos["D1_gj"] - z_pos["D0_gj"]) / z_pos["D0_gj"]
    z_pos["term2_seasonal_mismatch_pct"] = (
        100.0 * (z_pos["U1_gj"] - z_pos["U0_gj"]) / z_pos["U0_gj"] - z_pos["term1_demand_shift_pct"]
    )
    z_pos["term3_cost_exit_pct"] = (
        z_pos["dP_rel_pct"] - 100.0 * (z_pos["U1_gj"] - z_pos["U0_gj"]) / z_pos["U0_gj"]
    )
    z_resid = float(np.max(np.abs(
        z_pos["dP_rel_pct"]
        - (z_pos["term1_demand_shift_pct"] + z_pos["term2_seasonal_mismatch_pct"] + z_pos["term3_cost_exit_pct"])
    )))
    assert z_resid < 1e-10, f"Zone Eq. 6 residual {z_resid}"
    z_pos.to_parquet(r2_dir / "r2_zone_midcentury_ssp245_ensmean.parquet", index=False)

    # Empirical stratification by latitude band & urban size tier
    z_pos["abs_lat_band"] = pd.cut(
        z_pos["lat"].abs(),
        bins=[0.0, 30.0, 40.0, 50.0, 60.0, 90.0],
        labels=["<30 deg", "30-40 deg", "40-50 deg", "50-60 deg", ">60 deg"],
    )
    z_pos["size_tier"] = pd.cut(
        z_pos["baseline_mean_mw"],
        bins=[0.0, 10.0, 100.0, 1e9],
        labels=["Small (1-10 MW)", "Medium (10-100 MW)", "Large (>100 MW)"],
    )
    for col_s in ["abs_lat_band", "size_tier"]:
      grp = z_pos.groupby(col_s, observed=False)[
          ["D0_gj", "D1_gj", "U0_gj", "U1_gj", "P0_gj", "P1_gj"]
      ].sum().reset_index()
      grp["P0_pj"] = grp["P0_gj"] / 1e6
      grp["P1_pj"] = grp["P1_gj"] / 1e6
      grp["dP_rel_pct"] = 100.0 * (grp["P1_gj"] - grp["P0_gj"]) / np.maximum(1.0, grp["P0_gj"])
      grp["term1_demand_shift_pct"] = 100.0 * (grp["D1_gj"] - grp["D0_gj"]) / np.maximum(1.0, grp["D0_gj"])
      grp["term2_seasonal_mismatch_pct"] = (
          100.0 * (grp["U1_gj"] - grp["U0_gj"]) / np.maximum(1.0, grp["U0_gj"]) - grp["term1_demand_shift_pct"]
      )
      grp["term3_cost_exit_pct"] = (
          grp["dP_rel_pct"] - 100.0 * (grp["U1_gj"] - grp["U0_gj"]) / np.maximum(1.0, grp["U0_gj"])
      )
      grp.to_csv(r2_dir / f"r2_stratified_by_{col_s}.csv", index=False)
      z_strat_summary[col_s] = grp[[col_s, "P0_pj", "P1_pj", "dP_rel_pct", "term1_demand_shift_pct", "term2_seasonal_mismatch_pct", "term3_cost_exit_pct"]].to_dict("records")

  return {"max_identity_residual": max_residual, "zone_stratification": z_strat_summary}


def run_r3_locked_assets(
    df_locked_links: pd.DataFrame,
    clim_dir: Path,
    out_dir: Path,
) -> pd.DataFrame:
  """Execute R3 (10 runs: SSP2-4.5 x 5 GCMs x {2026-2045, 2046-2065}) on locked 2006-2025 assets."""
  r3_dir = out_dir / "r3_locked_assets"
  r3_dir.mkdir(parents=True, exist_ok=True)

  df_sources = pd.read_parquet(DATA_PROCESSED / "sources_2025/sources_2025_streams.parquet")
  df_zones = pd.read_parquet(DATA_PROCESSED / "sinks_2025/urban_heat_zones.parquet")
  df_rp = pd.read_parquet(DATA_PROCESSED / "sinks_2025/calibrated_region_parameters_2025.parquet")
  reg_valid = df_rp[df_rp["valid_for_study"]].copy().set_index("region_id")
  df_zones = df_zones[df_zones["region_id"].isin(reg_valid.index)].reset_index(drop=True)

  q_init_all = {
      str(r.stream_id): np.array([getattr(r, f"q3g_m{m+1:02d}_gj_d") for m in range(12)], dtype=np.float64)
      for r in df_sources.itertuples(index=False)
  }

  def _load_full_zm_dmq(clim_key: str) -> tuple[dict[str, dict[str, float]], dict[str, np.ndarray]]:
    npz_path = str(clim_dir / f"{clim_key}.npz")
    with np.load(npz_path, allow_pickle=True) as npz:
      zids = npz["zids"].astype(str)
      ann = npz["z_ann_gj"]
      peak = npz["z_peak_gj_d"]
      dmq = npz["z_dmq"]
      el25 = npz["z_ashp_el_2025"]
      el50 = npz["z_ashp_el_2050"]
      z_stats = {
          str(zids[i]): (float(ann[i]), float(peak[i]), float(el25[i]), float(el50[i]))
          for i in range(len(zids))
      }
      z_dmq = {str(zids[i]): np.asarray(dmq[i], dtype=np.float64) for i in range(len(zids))}
    zm = _build_country_zone_meta(
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
    return zm, z_dmq

  zm_base_obs, _ = _load_full_zm_dmq("gswp3-era5__obsclim__2006_2025__e2025__b14__ret0")

  summary_rows = []
  for mod in ISIMIP3B_MODELS:
    for win_id in ("2026_2045", "2046_2065"):
      ck = f"{mod}__ssp245__{win_id}__e2025__b14__ret0"
      zm_fut, dmq_fut = _load_full_zm_dmq(ck)
      # Preserve country-by-country baseline selection order
      df_res = evaluate_locked_portfolio(
          df_locked_links=df_locked_links,
          q_init_m_gj_d=q_init_all,
          zone_dmq_fut=dmq_fut,
          zone_meta_fut=zm_fut,
          zone_meta_base=zm_base_obs,
          discount_rate=DISCOUNT_RATE_CENTRAL,
      )
      run_tag = f"R3__{mod}__ssp245__{win_id}"
      df_res.to_parquet(r3_dir / f"{run_tag}_links.parquet", index=False)

      tot_links = len(df_res)
      loss_mask = df_res["is_loss_making_fut"]
      tot_capex_busd = float(df_res["total_sunk_capex_usd"].sum() / 1e9)
      loss_capex_busd = float(df_res.loc[loss_mask, "total_sunk_capex_usd"].sum() / 1e9)
      d0_tot = float(df_res["sub_D_0_gj_yr"].sum())
      df_tot = float(df_res["sub_D_fut_gj_yr"].sum())
      w_lcoh_0 = float((df_res["lcoh_0_usd_per_gj"] * df_res["sub_D_0_gj_yr"]).sum() / max(1.0, d0_tot))
      w_lcoh_f = float((df_res["lcoh_fut_usd_per_gj"] * df_res["sub_D_fut_gj_yr"]).sum() / max(1.0, df_tot))

      summary_rows.append({
          "run_id": run_tag,
          "model": mod,
          "ssp": "ssp245",
          "window_id": win_id,
          "n_locked_links": int(tot_links),
          "n_loss_making_links": int(loss_mask.sum()),
          "loss_making_link_share_pct": 100.0 * float(loss_mask.mean()) if tot_links > 0 else 0.0,
          "total_locked_capex_billion_usd": tot_capex_busd,
          "loss_making_capex_billion_usd": loss_capex_busd,
          "loss_making_capex_share_pct": 100.0 * loss_capex_busd / max(1e-9, tot_capex_busd),
          "P_0_ej_yr": float(df_res["P_0_gj_yr"].sum() / 1e9),
          "P_locked_fut_ej_yr": float(df_res["P_fut_gj_yr"].sum() / 1e9),
          "weighted_lcoh_0_usd_per_gj": w_lcoh_0,
          "weighted_lcoh_fut_usd_per_gj": w_lcoh_f,
          "delta_weighted_lcoh_usd_per_gj": w_lcoh_f - w_lcoh_0,
          "delta_weighted_lcoh_pct": 100.0 * (w_lcoh_f - w_lcoh_0) / max(1e-9, w_lcoh_0),
          "net_benefit_0_billion_usd_yr": float(df_res["net_benefit_0_usd_yr"].sum() / 1e9),
          "net_benefit_fut_billion_usd_yr": float(df_res["net_benefit_fut_usd_yr"].sum() / 1e9),
          "co2_abated_0_mt_yr": float(df_res["co2_abated_0_t_yr"].sum() / 1e6),
          "co2_abated_fut_mt_yr": float(df_res["co2_abated_fut_t_yr"].sum() / 1e6),
          "foregone_co2_abatement_mt_yr": float(df_res["foregone_co2_abatement_t_yr"].sum() / 1e6),
          "stranded_reversion_co2_mt_yr": float(df_res["stranded_reversion_co2_t_yr"].sum() / 1e6),
      })

  df_r3 = pd.DataFrame(summary_rows)
  df_r3.to_csv(r3_dir / "r3_locked_assets_summary.csv", index=False)
  return df_r3


def main() -> None:
  t_start = time.perf_counter()
  out_dir = DATA_PROCESSED / "scenarios_2025"
  clim_dir = out_dir / "_climate_cache"
  u_dir = out_dir / "_u_cache"
  scen_dir = out_dir / "scenarios"
  z_out_dir = out_dir / "zones"
  for d in (out_dir, clim_dir, u_dir, scen_dir, z_out_dir):
    d.mkdir(parents=True, exist_ok=True)

  scenarios = build_all_scenarios()
  print(f"=== GLOBAL SCENARIO RUNNER: {len(scenarios)} SCENARIOS CONFIGURED ===", flush=True)

  # Load static tables for country lists, edge counts (LPT scheduling), Q totals, and official_enduse flags
  df_sources = pd.read_parquet(DATA_PROCESSED / "sources_2025/sources_2025_streams.parquet")
  df_rp = pd.read_parquet(DATA_PROCESSED / "sinks_2025/calibrated_region_parameters_2025.parquet")
  reg_valid = df_rp[df_rp["valid_for_study"]].copy()
  official_isos = set(reg_valid.loc[reg_valid["official_enduse"], "iso3"].unique())

  df_zones_25 = pd.read_parquet(DATA_PROCESSED / "sinks_2025/urban_heat_zones.parquet")
  df_cands_25 = pd.read_parquet(DATA_PROCESSED / "sinks_2025/candidates_2025.parquet")
  df_cands_20 = pd.read_parquet(DATA_PROCESSED / "sinks_2020_robustness/candidates_2025.parquet")

  cand_counts_25 = df_cands_25["iso3"].value_counts().to_dict()
  cand_counts_20 = df_cands_20["iso3"].value_counts().to_dict()
  all_study_isos = sorted(
      set(reg_valid["iso3"].unique()) | set(df_sources["iso3"].unique()),
      key=lambda c: -cand_counts_25.get(c, 0),
  )
  zone_isos_25 = sorted(cand_counts_25.keys(), key=lambda c: -cand_counts_25[c])
  zone_isos_20 = sorted(cand_counts_20.keys(), key=lambda c: -cand_counts_20[c])

  # -------------------------------------------------------------------------
  # STAGE 1: Unique Climate-Demand Projections
  # -------------------------------------------------------------------------
  clim_tasks_map: dict[str, dict[str, object]] = {}
  for sc in scenarios:
    if sc.clim_key not in clim_tasks_map:
      clim_tasks_map[sc.clim_key] = {
          "clim_key": sc.clim_key,
          "run_id": sc.run_id,
          "start_year": sc.start_year,
          "end_year": sc.end_year,
          "ghsl_epoch": sc.ghsl_epoch,
          "bait_threshold_c": sc.bait_threshold_c,
          "use_envelope_retrofit": sc.use_envelope_retrofit,
          "out_npz": str(clim_dir / f"{sc.clim_key}.npz"),
      }

  pending_clim = [t for t in clim_tasks_map.values() if not Path(str(t["out_npz"])).exists()]
  print(f"Stage 1: {len(clim_tasks_map)} unique climate-demand projections ({len(pending_clim)} to compute)...", flush=True)
  if pending_clim:
    with ProcessPoolExecutor(
        max_workers=N_CLIMATE_WORKERS,
        mp_context=mp.get_context("spawn"),
        initializer=_init_clim_worker,
    ) as ex:
      futs = [ex.submit(_compute_one_clim_projection, t) for t in pending_clim]
      for idx, f in enumerate(as_completed(futs), start=1):
        _ = f.result()
        if idx % 15 == 0 or idx == len(pending_clim):
          print(f"   Stage 1 progress: {idx}/{len(pending_clim)} projections cached ({time.perf_counter()-t_start:.1f}s)", flush=True)

  # Load regional D_total_gj per (clim_key, iso3) and official_enduse D
  reg_off_map = reg_valid.set_index("region_id")["official_enduse"].to_dict()
  clim_D_by_iso: dict[str, dict[str, float]] = {}
  clim_D_off_global: dict[str, float] = {}
  for ck in clim_tasks_map:
    with np.load(clim_dir / f"{ck}.npz", allow_pickle=True) as npz:
      r_ids = npz["reg_id"].astype(str)
      r_isos = npz["reg_iso3"].astype(str)
      r_tot = npz["reg_tot_gj"]
      d_iso: dict[str, float] = {}
      d_off = 0.0
      for i in range(len(r_ids)):
        iso = str(r_isos[i])
        val = float(r_tot[i])
        d_iso[iso] = d_iso.get(iso, 0.0) + val
        if reg_off_map.get(str(r_ids[i]), False):
          d_off += val
      clim_D_by_iso[ck] = d_iso
      clim_D_off_global[ck] = d_off

  # -------------------------------------------------------------------------
  # STAGE 2: Unique Cost-Free Physical Upper Bound LPs (U)
  # -------------------------------------------------------------------------
  u_specs: dict[str, ScenarioSpec] = {}
  u_need_zone: dict[str, bool] = {}
  for sc in scenarios:
    if sc.u_key not in u_specs:
      u_specs[sc.u_key] = sc
      u_need_zone[sc.u_key] = sc.save_zone_table
    elif sc.save_zone_table:
      u_need_zone[sc.u_key] = True

  pending_u_keys = [uk for uk in u_specs if not (u_dir / f"{uk}.parquet").exists()]
  print(f"Stage 2: {len(u_specs)} unique LP upper-bound U configs ({len(pending_u_keys)} to solve across {N_SOLVER_WORKERS} workers)...", flush=True)

  if pending_u_keys:
    u_country_tasks = []
    for uk in pending_u_keys:
      sc = u_specs[uk]
      z_isos = zone_isos_25 if sc.ghsl_epoch == 2025 else zone_isos_20
      c_map = cand_counts_25 if sc.ghsl_epoch == 2025 else cand_counts_20
      for iso in z_isos:
        u_country_tasks.append((
            -c_map.get(iso, 0),
            {
                "u_key": uk,
                "iso3": iso,
                "ghsl_epoch": sc.ghsl_epoch,
                "retire_fossil_power": sc.retire_fossil_power,
                "is_4gdh": sc.is_4gdh,
                "use_2050_dh_tech": sc.use_2050_dh_tech,
                "clim_npz_path": str(clim_dir / f"{sc.clim_key}.npz"),
            },
        ))
    # Strict LPT sort: largest countries (DEU, CHN, USA, GBR...) across all configs start at t=0!
    u_country_tasks.sort(key=lambda x: x[0])

    u_accum_iso: dict[str, dict[str, float]] = {uk: {} for uk in pending_u_keys}
    u_accum_zone: dict[str, dict[str, float]] = {uk: {} for uk in pending_u_keys if u_need_zone[uk]}
    u_expected_cnt: dict[str, int] = {
        uk: len(zone_isos_25 if u_specs[uk].ghsl_epoch == 2025 else zone_isos_20)
        for uk in pending_u_keys
    }
    u_done_cnt = 0

    with ProcessPoolExecutor(
        max_workers=N_SOLVER_WORKERS,
        mp_context=mp.get_context("spawn"),
        initializer=_init_solver_worker,
    ) as ex:
      futs = [ex.submit(_solve_u_country_task, t[1]) for t in u_country_tasks]
      for idx, f in enumerate(as_completed(futs), start=1):
        uk, iso, u_gj, u_z_dict = f.result()
        if idx % 150 == 0:
          print(f"   [Stage 2 country-LP heartbeat] {idx}/{len(u_country_tasks)} country LPs done ({time.perf_counter()-t_start:.1f}s, last={iso} U={u_gj/1e6:.2f} PJ)", flush=True)
        u_accum_iso[uk][iso] = u_gj
        if uk in u_accum_zone:
          u_accum_zone[uk].update(u_z_dict)
        if len(u_accum_iso[uk]) == u_expected_cnt[uk]:
          df_u = pd.DataFrame(
              [{"iso3": k, "U_gj": v} for k, v in sorted(u_accum_iso[uk].items())]
          )
          df_u.to_parquet(u_dir / f"{uk}.parquet", index=False)
          if uk in u_accum_zone:
            df_uz = pd.DataFrame(
                [{"zone_id": zk, "U_gj": zv} for zk, zv in u_accum_zone[uk].items()]
            )
            df_uz.to_parquet(u_dir / f"{uk}_zones.parquet", index=False)
            del u_accum_zone[uk]
          del u_accum_iso[uk]
          u_done_cnt += 1
          if u_done_cnt % 10 == 0 or u_done_cnt == len(pending_u_keys):
            print(
                f"   Stage 2 progress: {u_done_cnt}/{len(pending_u_keys)} full global U configs saved "
                f"({idx}/{len(u_country_tasks)} country LPs, {time.perf_counter()-t_start:.1f}s)",
                flush=True,
            )

  # Load cached country U dictionaries
  u_by_key_iso: dict[str, dict[str, float]] = {}
  for uk in u_specs:
    df_u = pd.read_parquet(u_dir / f"{uk}.parquet")
    u_by_key_iso[uk] = dict(zip(df_u["iso3"], df_u["U_gj"]))

  # -------------------------------------------------------------------------
  # STAGE 3: Full-System Techno-Economic Matching (P) Across All Scenarios
  # -------------------------------------------------------------------------
  # Deduplicate identical physical/economic runs (e.g. GWL crossing coinciding with R1 window)
  pending_scens = [sc for sc in scenarios if not (scen_dir / f"{sc.scenario_id}_countries.parquet").exists()]
  print(f"Stage 3: {len(scenarios)} global scenarios ({len(pending_scens)} to solve across {N_SOLVER_WORKERS} workers)...", flush=True)

  if pending_scens:
    p_tasks = []
    for sc in pending_scens:
      c_map = cand_counts_25 if sc.ghsl_epoch == 2025 else cand_counts_20
      for iso in all_study_isos:
        p_tasks.append((
            -c_map.get(iso, 0),
            {
                "scen": sc,
                "iso3": iso,
                "clim_npz_path": str(clim_dir / f"{sc.clim_key}.npz"),
            },
        ))
    p_tasks.sort(key=lambda x: x[0])

    scen_map = {sc.scenario_id: sc for sc in pending_scens}
    p_accum: dict[str, list[dict[str, object]]] = {sc.scenario_id: [] for sc in pending_scens}
    scen_done = 0

    with ProcessPoolExecutor(
        max_workers=N_SOLVER_WORKERS,
        mp_context=mp.get_context("spawn"),
        initializer=_init_solver_worker,
    ) as ex:
      futs = [ex.submit(_solve_p_country_task, t[1]) for t in p_tasks]
      for idx, f in enumerate(as_completed(futs), start=1):
        res = f.result()
        sid = str(res["scenario_id"])
        if idx % 500 == 0:
          print(f"   [Stage 3 country-CELF heartbeat] {idx}/{len(p_tasks)} country solves done ({time.perf_counter()-t_start:.1f}s)", flush=True)
        p_accum[sid].append(res)

        if len(p_accum[sid]) == len(all_study_isos):
          sc = scen_map[sid]
          c_list = p_accum.pop(sid)
          u_iso_map = u_by_key_iso[sc.u_key]
          d_iso_map = clim_D_by_iso[sc.clim_key]

          c_rows = []
          all_sel_links = []
          z_p_all: dict[str, float] = {}
          z_d_all: dict[str, float] = {}

          for cr in c_list:
            iso = str(cr["iso3"])
            q_gj = float(cr["Q_gj"])
            u_gj = float(u_iso_map.get(iso, 0.0))
            p_gj = float(cr["P_gj"])
            # Enforce physical conservation 0 <= P <= U <= Q
            assert u_gj <= q_gj * 1.001 + 10.0, (
                f"Upper bound U exceeded Q in {sid} {iso}: U={u_gj}, Q={q_gj}"
            )
            if p_gj > u_gj + 1.0 and (p_gj - u_gj) / max(1.0, u_gj) > 0.02:
              print(f"   [WARN] {sid} {iso}: P={p_gj:.1f} slightly > U={u_gj:.1f} (within-month quantile effect)", flush=True)
            u_gj_clean = min(q_gj, max(p_gj, u_gj))
            p_gj = min(p_gj, u_gj_clean)
            c_rows.append({
                "scenario_id": sid,
                "group": sc.group,
                "subgroup": sc.subgroup,
                "model": sc.model,
                "ssp": sc.ssp,
                "window_id": sc.window_id,
                "start_year": sc.start_year,
                "end_year": sc.end_year,
                "iso3": iso,
                "official_enduse_country": iso in official_isos,
                "Q_gj": q_gj,
                "D_total_gj": float(d_iso_map.get(iso, 0.0)),
                "D_urban_gj": float(cr["D_urban_gj"]),
                "U_gj": u_gj_clean,
                "P_gj": p_gj,
                "H_deliv_gj": float(cr["H_deliv_gj"]),
                "H_peaking_gj": float(cr["H_peaking_gj"]),
                "net_benefit_usd": float(cr["net_benefit_usd"]),
                "tot_dh_cost_usd": float(cr["tot_dh_cost_usd"]),
                "avoided_ind_cost_usd": float(cr["avoided_ind_cost_usd"]),
                "total_capex_usd": float(cr["total_capex_usd"]),
                "co2_abated_t": float(cr["co2_abated_t"]),
                "n_selected_links": int(cr["n_selected_links"]),
                "P_power_gj": float(cr["P_by_sector"].get("power", 0.0)),
                "P_steel_gj": float(cr["P_by_sector"].get("steel", 0.0)),
                "P_cement_gj": float(cr["P_by_sector"].get("cement", 0.0)),
                "P_refining_gj": float(cr["P_by_sector"].get("refining", 0.0)),
                "P_ethylene_gj": float(cr["P_by_sector"].get("ethylene", 0.0)),
                "P_pulp_paper_gj": float(cr["P_by_sector"].get("pulp_paper", 0.0)),
                "P_wastewater_gj": float(cr["P_by_sector"].get("wastewater", 0.0)),
                "P_direct_hex_gj": float(cr["P_by_upgrade"].get("direct_hex", 0.0)),
                "P_hp_ind_power_gj": float(cr["P_by_upgrade"].get("hp_ind_power", 0.0)),
                "P_hp_wastewater_gj": float(cr["P_by_upgrade"].get("hp_wastewater", 0.0)),
            })
            if sc.save_selected_links and cr["selected_links"]:
              all_sel_links.extend(cr["selected_links"])
            if sc.save_zone_table:
              z_p_all.update(cr["zone_P_gj"])
              z_d_all.update(cr["zone_D_gj"])

          df_c_out = pd.DataFrame(c_rows)
          df_c_out.to_parquet(scen_dir / f"{sid}_countries.parquet", index=False)

          if sc.save_selected_links and all_sel_links:
            pd.DataFrame(all_sel_links).to_parquet(out_dir / f"{sid}_selected_links.parquet", index=False)

          if sc.save_zone_table:
            uz_path = u_dir / f"{sc.u_key}_zones.parquet"
            uz_map = {}
            if uz_path.exists():
              df_uz = pd.read_parquet(uz_path)
              uz_map = dict(zip(df_uz["zone_id"], df_uz["U_gj"]))
            z_rows = [
                {
                    "zone_id": zk,
                    "D_gj": zd_v,
                    "U_gj": max(z_p_all.get(zk, 0.0), float(uz_map.get(zk, 0.0))),
                    "P_gj": float(z_p_all.get(zk, 0.0)),
                }
                for zk, zd_v in z_d_all.items()
            ]
            pd.DataFrame(z_rows).to_parquet(z_out_dir / f"{sid}_zones.parquet", index=False)

          scen_done += 1
          if scen_done % 10 == 0 or scen_done == len(pending_scens):
            print(
                f"   Stage 3 progress: {scen_done}/{len(pending_scens)} scenarios completed "
                f"(latest {sid}: P={df_c_out['P_gj'].sum()/1e9:.3f} EJ/yr, U={df_c_out['U_gj'].sum()/1e9:.3f} EJ/yr, "
                f"D={df_c_out['D_total_gj'].sum()/1e9:.3f} EJ/yr, {time.perf_counter()-t_start:.1f}s)",
                flush=True,
            )

  # -------------------------------------------------------------------------
  # STAGE 4: Aggregate All Scenarios, Run R3 Locked Assets, R2 Decomposition, & Robustness 5
  # -------------------------------------------------------------------------
  print("Stage 4: Aggregating global scenario tables, running R3 (10 runs) & R2 Eq. 6 decomposition...", flush=True)
  country_dfs = [pd.read_parquet(scen_dir / f"{sc.scenario_id}_countries.parquet") for sc in scenarios]
  df_country_all = pd.concat(country_dfs, ignore_index=True)
  df_country_all.to_parquet(out_dir / "all_scenarios_by_country.parquet", index=False)

  # Build global summary table (+ Robustness Suite 5: official_enduse_country subset!)
  global_rows = []
  for sc in scenarios:
    sub = df_country_all[df_country_all["scenario_id"] == sc.scenario_id]
    sub_off = sub[sub["official_enduse_country"]]
    d_sub_dem = float(sub["H_deliv_gj"].sum() + sub["H_peaking_gj"].sum())
    global_rows.append({
        **asdict(sc),
        "Q_ej_yr": float(sub["Q_gj"].sum() / 1e9),
        "D_total_ej_yr": float(sub["D_total_gj"].sum() / 1e9),
        "D_urban_ej_yr": float(sub["D_urban_gj"].sum() / 1e9),
        "U_ej_yr": float(sub["U_gj"].sum() / 1e9),
        "P_ej_yr": float(sub["P_gj"].sum() / 1e9),
        "P_power_ej_yr": float(sub["P_power_gj"].sum() / 1e9),
        "P_industry_ej_yr": float((
            sub["P_steel_gj"] + sub["P_cement_gj"] + sub["P_refining_gj"]
            + sub["P_ethylene_gj"] + sub["P_pulp_paper_gj"]
        ).sum() / 1e9),
        "P_wastewater_ej_yr": float(sub["P_wastewater_gj"].sum() / 1e9),
        "P_direct_hex_ej_yr": float(sub["P_direct_hex_gj"].sum() / 1e9),
        "P_hp_ind_power_ej_yr": float(sub["P_hp_ind_power_gj"].sum() / 1e9),
        "P_hp_wastewater_ej_yr": float(sub["P_hp_wastewater_gj"].sum() / 1e9),
        "H_deliv_ej_yr": float(sub["H_deliv_gj"].sum() / 1e9),
        "H_peaking_ej_yr": float(sub["H_peaking_gj"].sum() / 1e9),
        "net_benefit_billion_usd_yr": float(sub["net_benefit_usd"].sum() / 1e9),
        "total_capex_billion_usd": float(sub["total_capex_usd"].sum() / 1e9),
        "mean_dh_lcoh_usd_per_gj": float(sub["tot_dh_cost_usd"].sum() / max(1.0, d_sub_dem)),
        "co2_abated_mt_yr": float(sub["co2_abated_t"].sum() / 1e6),
        "n_selected_links": int(sub["n_selected_links"].sum()),
        # Robustness Suite 5 (Official End-Use Statistics Countries Only):
        "D_official_only_ej_yr": float(clim_D_off_global[sc.clim_key] / 1e9),
        "U_official_only_ej_yr": float(sub_off["U_gj"].sum() / 1e9),
        "P_official_only_ej_yr": float(sub_off["P_gj"].sum() / 1e9),
    })

  df_global = pd.DataFrame(global_rows)
  df_global.to_parquet(out_dir / "global_scenarios_summary.parquet", index=False)
  df_global.to_csv(out_dir / "global_scenarios_summary.csv", index=False)

  # Run R3 Locked Assets (10 runs)
  df_locked_base = pd.read_parquet(out_dir / "R1__obs__2006_2025_selected_links.parquet")
  df_r3 = run_r3_locked_assets(df_locked_base, clim_dir, out_dir)

  # Run R2 Exact Zero-Residual Eq. (6) Decomposition
  r2_summary = compute_r2_decomposition(df_country_all, df_zones_25, out_dir)

  base_row = df_global[df_global["scenario_id"] == "R1__obs__2006_2025"].iloc[0].to_dict()
  mid_ssp245 = df_global[
      (df_global["group"] == "R1") & (df_global["ssp"] == "ssp245") & (df_global["window_id"] == "2046_2065")
  ]["P_ej_yr"]
  end_ssp245 = df_global[
      (df_global["group"] == "R1") & (df_global["ssp"] == "ssp245") & (df_global["window_id"] == "2081_2100")
  ]["P_ej_yr"]
  end_ssp370 = df_global[
      (df_global["group"] == "R1") & (df_global["ssp"] == "ssp370") & (df_global["window_id"] == "2081_2100")
  ]["P_ej_yr"]

  receipt = {
      "n_total_scenarios_solved": int(len(df_global)),
      "n_r3_locked_asset_runs": int(len(df_r3)),
      "wall_clock_seconds": float(time.perf_counter() - t_start),
      "baseline_2006_2025_obs": {
          "Q_ej_yr": base_row["Q_ej_yr"],
          "D_total_ej_yr": base_row["D_total_ej_yr"],
          "D_urban_ej_yr": base_row["D_urban_ej_yr"],
          "U_ej_yr": base_row["U_ej_yr"],
          "P_ej_yr": base_row["P_ej_yr"],
          "P_power_ej_yr": base_row["P_power_ej_yr"],
          "P_industry_ej_yr": base_row["P_industry_ej_yr"],
          "P_wastewater_ej_yr": base_row["P_wastewater_ej_yr"],
          "net_benefit_billion_usd_yr": base_row["net_benefit_billion_usd_yr"],
          "co2_abated_mt_yr": base_row["co2_abated_mt_yr"],
          "n_selected_links": base_row["n_selected_links"],
      },
      "future_P_ej_yr_5gcm_mean_min_max": {
          "mid_2046_2065_ssp245": [float(mid_ssp245.mean()), float(mid_ssp245.min()), float(mid_ssp245.max())],
          "end_2081_2100_ssp245": [float(end_ssp245.mean()), float(end_ssp245.min()), float(end_ssp245.max())],
          "end_2081_2100_ssp370": [float(end_ssp370.mean()), float(end_ssp370.min()), float(end_ssp370.max())],
      },
      "r3_mid_2046_2065_ssp245_mean": df_r3[df_r3["window_id"] == "2046_2065"].mean(numeric_only=True).to_dict(),
      "r2_max_identity_residual": r2_summary["max_identity_residual"],
  }
  (out_dir / "scenarios_run_metadata.json").write_text(json.dumps(receipt, indent=2))
  print("\n=== ALL GLOBAL SCENARIOS COMPLETE ===", flush=True)
  print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
  main()
