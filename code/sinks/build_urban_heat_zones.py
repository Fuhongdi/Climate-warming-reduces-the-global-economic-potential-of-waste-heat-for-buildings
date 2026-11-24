"""Delineate 100 m dense urban heating areas, evaluate distribution network costs and construct candidate connections."""
from __future__ import annotations

import os
for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "GDAL_CACHEMAX"):
  os.environ.setdefault(_k, "1" if "THREADS" in _k else "64")

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
import multiprocessing as mp
from pathlib import Path
import re
import subprocess

import numpy as np
import pandas as pd
import pyogrio
from pyproj import Transformer
import rasterio
from rasterio import features
from rasterio.transform import Affine
from rasterio.windows import Window
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from shapely import wkb
import xarray as xr

from code.common.parameters import (
    BAIT_THRESHOLD_CENTRAL_C,
    MAX_PIPE_LENGTH_KM,
    MIN_HEAT_DENSITY_GJ_HA_YR,
    MIN_ZONE_POWER_MW,
    PIPE_DETOUR_FACTOR,
    USD_PER_EUR_2025,
)
from code.common.paths import DATA_PROCESSED, DATA_RAW, GJ_PER_MW_DAY

NODATA_U32 = 4294967295
TILE_SIZE = 2048
N_WORKERS = 20
EARTH_RADIUS_KM = 6371.0088
MIN_ZONE_ANNUAL_GJ = MIN_ZONE_POWER_MW * 365.25 * GJ_PER_MW_DAY  # 31,557.6 GJ/yr
EUR_2015_TO_2025_HICP = 1.285  # Eurostat HICP / construction index 2015 -> 2025


def compute_persson_distribution_params(
    area_ha: float,
    vol_m3: float,
    ann_demand_gj: float,
) -> tuple[float, float, float]:
  """Return (dist_capex_eur2025_total, dist_fixed_om_eur2025_yr, lambda_dist_3gdh) per Persson et al. (2019).

  Equations follow Persson & Werner (2011) and Persson et al. (2019, Energy 176, 604-622):
    - Plot ratio e_z = (V_z / 3.0 m story height) / (1e4 * ha_z)
    - Effective width w_z = 61.8 * e_z^(-0.15) [m], clamped to [25, 110] m
    - Trench length L_z = (1e4 * ha_z) / w_z [m]
    - Linear heat density q_L = D_z / L_z [GJ/(m*yr)]
    - Average pipe outer media diameter d_a = clip(0.0486 * ln(q_L) + 0.0007, 0.040, 0.400) [m]
    - Construction cost C_1 + C_2 * d_a [EUR_2015/m] -> EUR_2025/m:
        (212, 4464) in dense urban cores (e_z >= 0.4), (151, 1378) in outer urban (e_z < 0.4),
        smoothly blended between e_z = 0.25 and 0.50.
    - Distribution heat loss: q_loss_L = K * 2*pi*d_a * dT * 8760 * 3.6e-6 [GJ/(m*yr)],
        with K = 0.5 W/(m2*K) and dT = 45 K for 70/40 C 3GDH (Persson 2019).
  """
  area_m2 = max(1e4, area_ha * 1e4)
  floor_m2 = vol_m3 / 3.0
  e_plot = max(0.02, floor_m2 / area_m2)
  w_eff = float(np.clip(61.8 * (e_plot ** -0.15), 25.0, 110.0))
  trench_m = area_m2 / w_eff
  q_lin = max(0.2, ann_demand_gj / trench_m)
  d_a = float(np.clip(0.0486 * math.log(q_lin) + 0.0007, 0.040, 0.400))

  w_core = float(np.clip((e_plot - 0.25) / 0.25, 0.0, 1.0))
  c1_2015 = w_core * 212.0 + (1.0 - w_core) * 151.0
  c2_2015 = w_core * 4464.0 + (1.0 - w_core) * 1378.0
  cost_per_m_eur25 = (c1_2015 + c2_2015 * d_a) * EUR_2015_TO_2025_HICP
  tot_capex_eur25 = trench_m * cost_per_m_eur25
  fom_eur25_yr = 0.01 * tot_capex_eur25  # 1%/yr distribution network O&M

  q_loss_lin = 0.5 * (2.0 * math.pi * d_a) * 45.0 * 8760.0 * 3.6e-6
  lambda_dist = float(np.clip(q_loss_lin / (q_lin + q_loss_lin), 0.03, 0.35))
  return tot_capex_eur25, fom_eur25_yr, lambda_dist


def compute_baseline_hdd14(out_path: Path) -> np.ndarray:
  """Compute 2006–2025 mean annual HDD14 (C*day/yr) for all 52,160 cells."""
  if out_path.exists():
    return np.load(out_path)
  bait_dir = DATA_PROCESSED / "bait/gswp3-era5__obsclim"
  acc = np.zeros(52160, dtype=np.float64)
  for yr in range(2006, 2026):
    ds = xr.open_dataset(bait_dir / f"year_{yr}.nc")
    b = ds["bait_c"].values.astype(np.float64)
    acc += np.maximum(BAIT_THRESHOLD_CENTRAL_C - b, 0.0).sum(axis=0)
    ds.close()
  mean_hdd = acc / 20.0
  np.save(out_path, mean_hdd)
  return mean_hdd


# Worker globals
_Z_LOOKUP: np.ndarray | None = None
_Z_HDD0: np.ndarray | None = None
_Z_PATHS: dict[str, Path] | None = None
_Z_TRANSFORM: Affine | None = None
_Z_WIDTH: int = 0
_Z_HEIGHT: int = 0


def _init_zone_worker(lookup: np.ndarray, hdd0: np.ndarray, paths: dict[str, Path], transform: Affine, width: int, height: int) -> None:
  global _Z_LOOKUP, _Z_HDD0, _Z_PATHS, _Z_TRANSFORM, _Z_WIDTH, _Z_HEIGHT
  _Z_LOOKUP = lookup
  _Z_HDD0 = hdd0
  _Z_PATHS = paths
  _Z_TRANSFORM = transform
  _Z_WIDTH = width
  _Z_HEIGHT = height


def _cluster_one_unit(task: tuple[int, str, str, str, str, float, float, bytes]) -> tuple[list[dict[str, object]], list[dict[str, object]], float]:
  """Cluster >=500 GJ/ha/yr 100 m pixels inside one GADM ADM_2 unit via exact sparse 8-connectivity."""
  assert _Z_LOOKUP is not None and _Z_HDD0 is not None and _Z_PATHS is not None and _Z_TRANSFORM is not None
  unit_id, iso3, region_id, name1, name2, k_c, w_c, geom_wkb = task
  geom = wkb.loads(geom_wkb)
  minx, miny, maxx, maxy = geom.bounds

  # Convert bounds to pixel indices
  # x = c + col * a (a = 100.0), y = f + row * e (e = -100.0)
  c0 = max(0, int(math.floor((minx - _Z_TRANSFORM.c) / _Z_TRANSFORM.a)))
  c1 = min(_Z_WIDTH, int(math.ceil((maxx - _Z_TRANSFORM.c) / _Z_TRANSFORM.a)))
  r0 = max(0, int(math.floor((maxy - _Z_TRANSFORM.f) / _Z_TRANSFORM.e)))
  r1 = min(_Z_HEIGHT, int(math.ceil((miny - _Z_TRANSFORM.f) / _Z_TRANSFORM.e)))
  if c1 <= c0 or r1 <= r0:
    return [], [], 0.0

  transformer = Transformer.from_crs("ESRI:54009", "EPSG:4326", always_xy=True)

  rows_dense = []
  cols_dense = []
  vols_dense = []
  pops_dense = []
  cells_dense = []
  dems_dense = []
  unit_total_dem_gj = 0.0

  has_pop = "pop" in _Z_PATHS and _Z_PATHS["pop"] is not None

  with rasterio.open(_Z_PATHS["tot"]) as s_tot, rasterio.open(_Z_PATHS["nres"]) as s_nres:
    s_pop = rasterio.open(_Z_PATHS["pop"]) if has_pop else None
    try:
      for rt in range(r0, r1, TILE_SIZE):
        rh = min(TILE_SIZE, r1 - rt)
        for ct in range(c0, c1, TILE_SIZE):
          cw = min(TILE_SIZE, c1 - ct)
          win = Window(ct, rt, cw, rh)
          tot_arr = s_tot.read(1, window=win)
          nres_arr = s_nres.read(1, window=win)
          t_clean = np.where(tot_arr == NODATA_U32, 0, tot_arr).astype(np.int64)
          n_clean = np.where(nres_arr == NODATA_U32, 0, nres_arr).astype(np.int64)
          v_arr = np.maximum(0, t_clean - n_clean)
          if not np.any(v_arr > 0):
            continue

          # Rasterize unit polygon onto this tile
          tx = _Z_TRANSFORM.c + ct * _Z_TRANSFORM.a
          ty = _Z_TRANSFORM.f + rt * _Z_TRANSFORM.e
          tile_tr = Affine(_Z_TRANSFORM.a, 0.0, tx, 0.0, _Z_TRANSFORM.e, ty)
          in_poly = features.rasterize([(geom, 1)], out_shape=(rh, cw), transform=tile_tr, fill=0, dtype="uint8", all_touched=False)
          active = (v_arr > 0) & (in_poly > 0)
          if not np.any(active):
            continue

          lr, lc = np.where(active)
          gr = rt + lr
          gc = ct + lc
          v_pts = v_arr[lr, lc].astype(np.float64)
          if s_pop is not None:
            p_raw = s_pop.read(1, window=win)[lr, lc]
            p_pts = np.where((p_raw < 0) | ~np.isfinite(p_raw), 0.0, p_raw).astype(np.float64)
          else:
            p_pts = np.zeros_like(v_pts)

          xs = _Z_TRANSFORM.c + (gc.astype(np.float64) + 0.5) * _Z_TRANSFORM.a
          ys = _Z_TRANSFORM.f + (gr.astype(np.float64) + 0.5) * _Z_TRANSFORM.e
          lons, lats = transformer.transform(xs, ys)
          valid_ll = np.isfinite(lons) & np.isfinite(lats)
          if not np.any(valid_ll):
            continue

          gr, gc, v_pts, p_pts, lons, lats = gr[valid_ll], gc[valid_ll], v_pts[valid_ll], p_pts[valid_ll], lons[valid_ll], lats[valid_ll]
          r05 = np.clip(np.floor((90.0 - lats) * 2.0).astype(np.int32), 0, 359)
          c05 = np.clip(np.floor((lons + 180.0) * 2.0).astype(np.int32), 0, 719)
          c_idx = _Z_LOOKUP[r05, c05]
          ok_c = c_idx >= 0
          if not np.any(ok_c):
            continue

          gr, gc, v_pts, p_pts, c_idx = gr[ok_c], gc[ok_c], v_pts[ok_c], p_pts[ok_c], c_idx[ok_c]
          # Baseline annual demand [GJ/ha/yr == GJ/pixel/yr]
          dem_gj = v_pts * (k_c * _Z_HDD0[c_idx] + 365.25 * w_c)
          unit_total_dem_gj += float(dem_gj.sum())

          dense = dem_gj >= MIN_HEAT_DENSITY_GJ_HA_YR
          if np.any(dense):
            rows_dense.append(gr[dense])
            cols_dense.append(gc[dense])
            vols_dense.append(v_pts[dense])
            pops_dense.append(p_pts[dense])
            cells_dense.append(c_idx[dense])
            dems_dense.append(dem_gj[dense])
    finally:
      if s_pop is not None:
        s_pop.close()

  if not rows_dense:
    return [], [], unit_total_dem_gj

  all_r = np.concatenate(rows_dense)
  all_c = np.concatenate(cols_dense)
  all_v = np.concatenate(vols_dense)
  all_p = np.concatenate(pops_dense)
  all_cell = np.concatenate(cells_dense)
  all_d = np.concatenate(dems_dense)

  if float(all_d.sum()) < MIN_ZONE_ANNUAL_GJ:
    return [], [], unit_total_dem_gj

  # Exact 8-connectivity: |dr| <= 1 and |dc| <= 1 <=> Euclidean pixel distance <= sqrt(2) = 1.41421 < 1.45
  coords_rc = np.column_stack([all_r.astype(np.float64), all_c.astype(np.float64)])
  tree = cKDTree(coords_rc)
  adj = tree.sparse_distance_matrix(tree, max_distance=1.45, output_type="coo_matrix")
  n_comp, labels = connected_components(adj, directed=False)

  # Sum demand per connected component first to filter >= 1 MW immediately
  comp_dem = np.bincount(labels, weights=all_d, minlength=n_comp)
  keep_comps = np.where(comp_dem >= MIN_ZONE_ANNUAL_GJ)[0]
  if len(keep_comps) == 0:
    return [], [], unit_total_dem_gj

  zones_out: list[dict[str, object]] = []
  weights_out: list[dict[str, object]] = []

  for local_k, comp_id in enumerate(keep_comps, start=1):
    m = labels == comp_id
    z_ha = float(np.sum(m))  # 1 pixel = 1.0000 ha in ESRI:54009!
    z_vol = float(all_v[m].sum())
    z_pop = float(all_p[m].sum())
    z_dem = float(all_d[m].sum())

    # Demand-weighted centroid in ESRI:54009 -> WGS84 (lon, lat)
    w_d = all_d[m]
    cx_54009 = float(np.average(_Z_TRANSFORM.c + (all_c[m] + 0.5) * _Z_TRANSFORM.a, weights=w_d))
    cy_54009 = float(np.average(_Z_TRANSFORM.f + (all_r[m] + 0.5) * _Z_TRANSFORM.e, weights=w_d))
    clon, clat = transformer.transform(cx_54009, cy_54009)

    dist_capex_eur25, dist_fom_eur25, lambda_dist = compute_persson_distribution_params(z_ha, z_vol, z_dem)
    zid = f"HZ_{iso3}_{unit_id:05d}_{local_k:03d}"

    zones_out.append({
        "zone_id": zid,
        "unit_id": int(unit_id),
        "iso3": iso3,
        "region_id": region_id,
        "adm1_name": name1,
        "adm2_name": name2,
        "lat": float(clat),
        "lon": float(clon),
        "area_ha": z_ha,
        "volume_m3": z_vol,
        "population": z_pop,
        "baseline_demand_gj_yr": z_dem,
        "baseline_mean_mw": z_dem / (365.25 * GJ_PER_MW_DAY),
        "mean_density_gj_ha_yr": z_dem / z_ha,
        "persson_dist_capex_eur2025": dist_capex_eur25,
        "persson_dist_fom_eur2025_yr": dist_fom_eur25,
        "lambda_dist_3gdh": lambda_dist,
    })

    # Aggregate exact 0.5-deg climate cell weights for this zone
    u_cells, inv = np.unique(all_cell[m], return_inverse=True)
    cell_vols = np.bincount(inv, weights=all_v[m], minlength=len(u_cells))
    for c_id, cv in zip(u_cells, cell_vols):
      weights_out.append({
          "zone_id": zid,
          "cell_idx": int(c_id),
          "volume_m3": float(cv),
      })

  return zones_out, weights_out, unit_total_dem_gj


def generate_candidate_links(df_zones: pd.DataFrame, df_sources: pd.DataFrame) -> pd.DataFrame:
  """Generate all same-country candidate links with ell_iz = 1.3 * d_Euclid <= 100 km."""
  max_euclid_km = MAX_PIPE_LENGTH_KM / PIPE_DETOUR_FACTOR  # 76.923 km
  max_chord = 2.0 * math.sin(max_euclid_km / (2.0 * EARTH_RADIUS_KM))

  def to_xyz(la: np.ndarray, lo: np.ndarray) -> np.ndarray:
    la_r, lo_r = np.deg2rad(la), np.deg2rad(lo)
    cl = np.cos(la_r)
    return np.column_stack([cl * np.cos(lo_r), cl * np.sin(lo_r), np.sin(la_r)])

  cand_blocks = []
  for iso3, z_grp in df_zones.groupby("iso3"):
    s_grp = df_sources[df_sources["iso3"] == iso3]
    if len(s_grp) == 0:
      continue
    z_grp = z_grp.reset_index(drop=True)
    s_grp = s_grp.reset_index(drop=True)
    s_xyz = to_xyz(s_grp["lat"].values, s_grp["lon"].values)
    z_xyz = to_xyz(z_grp["lat"].values, z_grp["lon"].values)
    z_tree = cKDTree(z_xyz)
    hits = z_tree.query_ball_point(s_xyz, r=max_chord)

    s_idx_list = []
    z_idx_list = []
    for si, z_list in enumerate(hits):
      if z_list:
        s_idx_list.extend([si] * len(z_list))
        z_idx_list.extend(z_list)
    if not s_idx_list:
      continue

    si_arr = np.asarray(s_idx_list, dtype=np.int32)
    zi_arr = np.asarray(z_idx_list, dtype=np.int32)
    chords = np.linalg.norm(s_xyz[si_arr] - z_xyz[zi_arr], axis=1)
    d_km = 2.0 * EARTH_RADIUS_KM * np.arcsin(np.clip(chords * 0.5, 0.0, 1.0))
    ell_km = PIPE_DETOUR_FACTOR * d_km
    keep = ell_km <= MAX_PIPE_LENGTH_KM + 1e-9
    si_arr, zi_arr, d_km, ell_km = si_arr[keep], zi_arr[keep], d_km[keep], ell_km[keep]

    cand_blocks.append(pd.DataFrame({
        "iso3": iso3,
        "stream_id": s_grp["stream_id"].values[si_arr],
        "facility_id": s_grp["facility_id"].values[si_arr],
        "sector": s_grp["sector"].values[si_arr],
        "subsector_detail": s_grp["subsector_detail"].values[si_arr],
        "is_fossil_power": s_grp["is_fossil_power"].values[si_arr],
        "upgrade_mode": s_grp["upgrade_mode"].values[si_arr],
        "zone_id": z_grp["zone_id"].values[zi_arr],
        "region_id": z_grp["region_id"].values[zi_arr],
        "euclid_km": d_km,
        "ell_km": np.maximum(0.5, ell_km),  # minimum 0.5 km connection tie-in length
    }))

  if not cand_blocks:
    return pd.DataFrame()
  return pd.concat(cand_blocks, ignore_index=True)


def run_validation_2_eu_peta(df_zones: pd.DataFrame, df_reg_cal: pd.DataFrame, out_dir: Path) -> dict[str, object]:
  """Validation 2: Compare European national & >=500 GJ/ha urban DH heat demand vs PETA / sEEnergies Table 21."""
  pdf_p = DATA_RAW / "seenergies_d51/zenodo_4785411/sEEnergies_WP5_D5.1_Revised_version.pdf"
  txt = subprocess.check_output(["pdftotext", "-layout", "-f", "56", "-l", "56", str(pdf_p), "-"], text=True)
  ms_to_iso3 = {
      "AT": "AUT", "BE": "BEL", "BG": "BGR", "CY": "CYP", "CZ": "CZE", "DE": "DEU",
      "DK": "DNK", "EE": "EST", "EL": "GRC", "ES": "ESP", "FI": "FIN", "FR": "FRA",
      "HR": "HRV", "HU": "HUN", "IE": "IRL", "IT": "ITA", "LT": "LTU", "LU": "LUX",
      "LV": "LVA", "MT": "MLT", "NL": "NLD", "PL": "POL", "PT": "PRT", "RO": "ROU",
      "SE": "SWE", "SI": "SVN", "SK": "SVK", "UK": "GBR",
  }
  peta_rows = []
  for ln in txt.splitlines():
    m = re.match(r"^\s*([A-Z]{2})\s+([\d,]+)\s+([\d,]+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)", ln)
    if m and m.group(1) in ms_to_iso3:
      peta_rows.append({
          "iso3": ms_to_iso3[m.group(1)],
          "peta_total_heat_pj": float(m.group(5)),
          "peta_dhe_urban_heat_pj": float(m.group(6)),
          "peta_dhe_share_pct": float(m.group(7)),
      })
  df_peta = pd.DataFrame(peta_rows)
  our_tot = df_reg_cal.groupby("iso3", as_index=False)["baseline_total_demand_gj_yr"].sum().assign(
      our_total_heat_pj=lambda d: d["baseline_total_demand_gj_yr"] / 1e6
  )
  our_urb = df_zones.groupby("iso3", as_index=False)["baseline_demand_gj_yr"].sum().assign(
      our_urban_zone_heat_pj=lambda d: d["baseline_demand_gj_yr"] / 1e6
  )
  cmp = df_peta.merge(our_tot[["iso3", "our_total_heat_pj"]], on="iso3", how="left").merge(
      our_urb[["iso3", "our_urban_zone_heat_pj"]], on="iso3", how="left"
  ).fillna(0.0)
  cmp["our_urban_share_pct"] = np.where(cmp["our_total_heat_pj"] > 0, 100.0 * cmp["our_urban_zone_heat_pj"] / cmp["our_total_heat_pj"], 0.0)

  val_dir = out_dir / "validation_2"
  val_dir.mkdir(parents=True, exist_ok=True)
  cmp.to_csv(val_dir / "eu28_urban_heat_vs_peta_hre4.csv", index=False)

  r_tot = float(np.corrcoef(cmp["peta_total_heat_pj"], cmp["our_total_heat_pj"])[0, 1])
  r_urb = float(np.corrcoef(cmp["peta_dhe_urban_heat_pj"], cmp["our_urban_zone_heat_pj"])[0, 1])
  summary = {
      "n_eu_countries_compared": int(len(cmp)),
      "peta_hre4_total_heat_pj": float(cmp["peta_total_heat_pj"].sum()),
      "our_baseline_total_heat_pj": float(cmp["our_total_heat_pj"].sum()),
      "total_heat_pearson_r": r_tot,
      "peta_dhe_urban_heat_pj": float(cmp["peta_dhe_urban_heat_pj"].sum()),
      "our_urban_zones_heat_pj": float(cmp["our_urban_zone_heat_pj"].sum()),
      "urban_zone_heat_pearson_r": r_urb,
  }
  (val_dir / "validation_2_summary.json").write_text(json.dumps(summary, indent=2))
  return summary


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--ghsl-epoch", type=int, choices=[2020, 2025], default=2025)
  args = parser.parse_args()

  sinks_dir = DATA_PROCESSED / "sinks_2025"
  out_dir = sinks_dir if args.ghsl_epoch == 2025 else (DATA_PROCESSED / "sinks_2020_robustness")
  out_dir.mkdir(parents=True, exist_ok=True)

  print("1. Loading baseline 2006-2025 HDD14 and calibrated (K_c, W_c)...", flush=True)
  hdd0 = compute_baseline_hdd14(sinks_dir / "baseline_cell_hdd14_2006_2025.npy")
  df_cal = pd.read_parquet(sinks_dir / "calibrated_region_parameters_2025.parquet")
  # Keep only valid regions with non-missing energy & price data
  df_cal_valid = df_cal[df_cal["valid_for_study"]].copy()
  kc_map = dict(zip(df_cal_valid["region_id"], df_cal_valid["k_c_gj_per_m3_hdd"]))
  wc_map = dict(zip(df_cal_valid["region_id"], df_cal_valid["w_c_gj_per_m3_day"]))

  print("2. Filtering GADM ADM_2 units with >= 1 MW potential building heat demand...", flush=True)
  df_uc = pd.read_parquet(sinks_dir / "ghsl_unit_cell_volumes.parquet")
  meta = pd.read_parquet(sinks_dir / "gadm_units_meta.parquet")
  v_col = "v25_m3" if args.ghsl_epoch == 2025 else "v20_m3"
  df_uc = df_uc.merge(meta[["unit_id", "iso3", "region_id", "NAME_1", "NAME_2"]], on="unit_id", how="left")
  df_uc = df_uc[df_uc["region_id"].isin(kc_map)].copy()
  df_uc["kc"] = df_uc["region_id"].map(kc_map)
  df_uc["wc"] = df_uc["region_id"].map(wc_map)
  df_uc["dem_gj"] = df_uc[v_col] * (df_uc["kc"] * hdd0[df_uc["cell_idx"].values] + 365.25 * df_uc["wc"])
  unit_dem = df_uc.groupby("unit_id", as_index=False)["dem_gj"].sum()
  qual_uids = set(unit_dem.loc[unit_dem["dem_gj"] >= MIN_ZONE_ANNUAL_GJ, "unit_id"].astype(int))
  print(f"   Qualifying GADM units >= 1 MW: {len(qual_uids)} / {len(meta)}", flush=True)

  print("3. Loading qualifying geometries into parent memory as WKB (zero worker SQLite contention)...", flush=True)
  gdf_all = pyogrio.read_dataframe(sinks_dir / "gadm_units_54009.gpkg", layer="units")
  gdf_q = gdf_all[gdf_all["unit_id"].isin(qual_uids)].copy()
  tasks = [
      (
          int(r.unit_id), str(r.iso3), str(r.region_id), str(r.NAME_1), str(r.NAME_2),
          float(kc_map[str(r.region_id)]), float(wc_map[str(r.region_id)]),
          wkb.dumps(r.geometry),
      )
      for r in gdf_q.itertuples(index=False)
  ]
  del gdf_all, gdf_q

  from code.sinks.aggregate_ghsl_to_climate_cells import build_climate_cell_lookup
  lookup = build_climate_cell_lookup()

  if args.ghsl_epoch == 2025:
    ghsl_dir = DATA_RAW / "sinks/2026-11-13/ghsl"
    paths = {
        "tot": ghsl_dir / "GHS_BUILT_V_E2025_GLOBE_R2023A_54009_100_V1_0.tif",
        "nres": ghsl_dir / "GHS_BUILT_V_NRES_E2025_GLOBE_R2023A_54009_100_V1_0.tif",
        "pop": ghsl_dir / "GHS_POP_E2025_GLOBE_R2023A_54009_100_V1_0.tif",
    }
  else:
    e20_dir = DATA_RAW / "supplements_2025/ghsl_e2020"
    paths = {
        "tot": e20_dir / "GHS_BUILT_V_E2020_GLOBE_R2023A_54009_100_V1_0.tif",
        "nres": e20_dir / "GHS_BUILT_V_NRES_E2020_GLOBE_R2023A_54009_100_V1_0.tif",
    }

  with rasterio.open(paths["tot"]) as src:
    transform = src.transform
    width, height = src.width, src.height

  print(f"4. Clustering 100 m urban heat zones across {len(tasks)} municipalities ({N_WORKERS} workers)...", flush=True)
  all_zones: list[dict[str, object]] = []
  all_weights: list[dict[str, object]] = []
  with ProcessPoolExecutor(
      max_workers=N_WORKERS,
      mp_context=mp.get_context("spawn"),
      initializer=_init_zone_worker,
      initargs=(lookup, hdd0, paths, transform, width, height),
  ) as ex:
    for idx, (z_list, w_list, _) in enumerate(ex.map(_cluster_one_unit, tasks, chunksize=8), start=1):
      if z_list:
        all_zones.extend(z_list)
        all_weights.extend(w_list)
      if idx % 1500 == 0 or idx == len(tasks):
        print(f"   Processed {idx}/{len(tasks)} municipalities -> {len(all_zones)} urban heat zones so far...", flush=True)

  df_zones = pd.DataFrame(all_zones)
  df_weights = pd.DataFrame(all_weights)
  df_zones.to_parquet(out_dir / "urban_heat_zones.parquet", index=False)
  df_weights.to_parquet(out_dir / "urban_zone_cell_weights.parquet", index=False)

  print("5. Generating same-country candidate pipelines (ell = 1.3 * d_Euclid <= 100 km)...", flush=True)
  df_sources = pd.read_parquet(DATA_PROCESSED / "sources_2025/sources_2025_streams.parquet")
  df_cands = generate_candidate_links(df_zones, df_sources)
  df_cands.to_parquet(out_dir / "candidates_2025.parquet", index=False)

  val2_summary = {}
  if args.ghsl_epoch == 2025:
    print("6. Running Validation 2 against European PETA / Heat Roadmap Europe 4...", flush=True)
    val2_summary = run_validation_2_eu_peta(df_zones, df_cal_valid, out_dir)
    print("   Validation 2 summary:", val2_summary, flush=True)

  tot_global_D_ej = float(df_cal_valid["baseline_total_demand_gj_yr"].sum() / 1e9)
  tot_urban_D_ej = float(df_zones["baseline_demand_gj_yr"].sum() / 1e9)
  receipt = {
      "ghsl_epoch": args.ghsl_epoch,
      "n_urban_heat_zones": int(len(df_zones)),
      "n_countries_with_zones": int(df_zones["iso3"].nunique()),
      "global_total_building_demand_D_ej_yr": tot_global_D_ej,
      "global_urban_zone_demand_D_ej_yr": tot_urban_D_ej,
      "urban_zone_share_of_global_D_pct": 100.0 * tot_urban_D_ej / tot_global_D_ej if tot_global_D_ej > 0 else 0.0,
      "n_candidate_links_le_100km": int(len(df_cands)),
      "n_unique_sources_in_candidates": int(df_cands["stream_id"].nunique()) if len(df_cands) > 0 else 0,
      "n_unique_zones_in_candidates": int(df_cands["zone_id"].nunique()) if len(df_cands) > 0 else 0,
      "validation_2": val2_summary,
  }
  (out_dir / "urban_zones_and_candidates_receipt.json").write_text(json.dumps(receipt, indent=2))
  print("\n=== DONE: URBAN HEATING AREAS AND CANDIDATE CONNECTIONS ===", flush=True)
  print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
  main()
