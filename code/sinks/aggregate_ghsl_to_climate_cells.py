"""Prepare municipal administrative units and aggregate 100 m GHSL building volumes to 0.5-degree climate grid cells."""
from __future__ import annotations

import os
for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "GDAL_CACHEMAX"):
  os.environ.setdefault(_k, "1" if "THREADS" in _k else "64")

from concurrent.futures import ProcessPoolExecutor
import json
import math
import multiprocessing as mp
from pathlib import Path
import zipfile

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
from pyproj import Transformer
import rasterio
from rasterio import features
from rasterio.transform import Affine
from rasterio.windows import Window
from scipy.spatial import cKDTree
from shapely import wkb
import xarray as xr

from code.common.paths import DATA_PROCESSED, DATA_RAW

NODATA_U32 = 4294967295
STRIP_HEIGHT = 1024   # 2 native 512-row tiles
COL_CHUNK = 45056     # 88 native 512-col tiles (~46M pixels -> <180 MB RAM per worker)
N_WORKERS = 24
MAX_COASTAL_SNAP_KM = 150.0
EARTH_RADIUS_KM = 6371.0088

US_DIVISIONS = {
    "Connecticut": "USA_DIV1", "Maine": "USA_DIV1", "Massachusetts": "USA_DIV1",
    "New Hampshire": "USA_DIV1", "Rhode Island": "USA_DIV1", "Vermont": "USA_DIV1",
    "New Jersey": "USA_DIV2", "New York": "USA_DIV2", "Pennsylvania": "USA_DIV2",
    "Illinois": "USA_DIV3", "Indiana": "USA_DIV3", "Michigan": "USA_DIV3",
    "Ohio": "USA_DIV3", "Wisconsin": "USA_DIV3",
    "Iowa": "USA_DIV4", "Kansas": "USA_DIV4", "Minnesota": "USA_DIV4",
    "Missouri": "USA_DIV4", "Nebraska": "USA_DIV4", "North Dakota": "USA_DIV4",
    "South Dakota": "USA_DIV4",
    "Delaware": "USA_DIV5", "District of Columbia": "USA_DIV5", "Florida": "USA_DIV5",
    "Georgia": "USA_DIV5", "Maryland": "USA_DIV5", "North Carolina": "USA_DIV5",
    "South Carolina": "USA_DIV5", "Virginia": "USA_DIV5", "West Virginia": "USA_DIV5",
    "Alabama": "USA_DIV6", "Kentucky": "USA_DIV6", "Mississippi": "USA_DIV6",
    "Tennessee": "USA_DIV6",
    "Arkansas": "USA_DIV7", "Louisiana": "USA_DIV7", "Oklahoma": "USA_DIV7",
    "Texas": "USA_DIV7",
    "Arizona": "USA_DIV8", "Colorado": "USA_DIV8", "Idaho": "USA_DIV8",
    "Montana": "USA_DIV8", "Nevada": "USA_DIV8", "New Mexico": "USA_DIV8",
    "Utah": "USA_DIV8", "Wyoming": "USA_DIV8",
    "Alaska": "USA_DIV9", "California": "USA_DIV9", "Hawaii": "USA_DIV9",
    "Oregon": "USA_DIV9", "Washington": "USA_DIV9",
}

CHN_NORTH_PROVINCES = {
    "Beijing", "Tianjin", "Hebei", "Shanxi", "Nei Mongol", "Inner Mongolia",
    "Liaoning", "Jilin", "Heilongjiang", "Shandong", "Henan",
    "Shaanxi", "Gansu", "Qinghai", "Ningxia Hui", "Ningxia",
    "Xinjiang Uygur", "Xinjiang",
}


def assign_region_id(iso3: str, name1: str) -> str:
  if iso3 == "USA":
    return US_DIVISIONS.get(str(name1).strip(), "USA_DIV5")
  if iso3 == "CHN":
    n1 = str(name1).strip()
    if any(n1.lower().startswith(p.lower()) for p in CHN_NORTH_PROVINCES):
      return "CHN_NORTH"
    return "CHN_REST"
  return iso3


def ensure_ghsl_e2020_unzipped() -> tuple[Path, Path]:
  e20_dir = DATA_RAW / "supplements_2025/ghsl_e2020"
  tot_tif = e20_dir / "GHS_BUILT_V_E2020_GLOBE_R2023A_54009_100_V1_0.tif"
  nres_tif = e20_dir / "GHS_BUILT_V_NRES_E2020_GLOBE_R2023A_54009_100_V1_0.tif"
  for zname, tpath in [
      ("GHS_BUILT_V_E2020_GLOBE_R2023A_54009_100_V1_0.zip", tot_tif),
      ("GHS_BUILT_V_NRES_E2020_GLOBE_R2023A_54009_100_V1_0.zip", nres_tif),
  ]:
    if not tpath.exists():
      print(f"   Unzipping {zname}...", flush=True)
      with zipfile.ZipFile(e20_dir / zname) as zf:
        for member in zf.namelist():
          if member.endswith(".tif"):
            with zf.open(member) as src, open(tpath, "wb") as dst:
              while True:
                buf = src.read(16 * 1024 * 1024)
                if not buf:
                  break
                dst.write(buf)
            break
  return tot_tif, nres_tif


def build_gadm_units(out_dir: Path) -> tuple[Path, pd.DataFrame]:
  gpkg_out = out_dir / "gadm_units_54009.gpkg"
  meta_out = out_dir / "gadm_units_meta.parquet"
  if gpkg_out.exists() and meta_out.exists():
    print("   Reusing existing gadm_units_54009.gpkg...", flush=True)
    return gpkg_out, pd.read_parquet(meta_out)

  raw_gpkg = DATA_RAW / "supplements_2025/gadm41/gadm_410-levels.gpkg"
  gdf2 = pyogrio.read_dataframe(raw_gpkg, layer="ADM_2", columns=["GID_0", "COUNTRY", "NAME_1", "GID_2", "NAME_2"])
  gdf1 = pyogrio.read_dataframe(raw_gpkg, layer="ADM_1", columns=["GID_0", "COUNTRY", "GID_1", "NAME_1"])
  gdf0 = pyogrio.read_dataframe(raw_gpkg, layer="ADM_0", columns=["GID_0", "COUNTRY"])

  have2 = set(gdf2["GID_0"].dropna().unique())
  sub1 = gdf1[~gdf1["GID_0"].isin(have2)].copy()
  sub1["GID_2"] = sub1["GID_1"]
  sub1["NAME_2"] = sub1["NAME_1"]
  sub1 = sub1[["GID_0", "COUNTRY", "NAME_1", "GID_2", "NAME_2", "geometry"]]

  have12 = have2 | set(sub1["GID_0"].dropna().unique())
  sub0 = gdf0[~gdf0["GID_0"].isin(have12)].copy()
  sub0["NAME_1"] = sub0["COUNTRY"]
  sub0["GID_2"] = sub0["GID_0"]
  sub0["NAME_2"] = sub0["COUNTRY"]
  sub0 = sub0[["GID_0", "COUNTRY", "NAME_1", "GID_2", "NAME_2", "geometry"]]

  gdf = pd.concat([gdf2, sub1, sub0], ignore_index=True)
  gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=gdf2.crs)
  gdf = gdf[gdf["geometry"].notna() & ~gdf["geometry"].is_empty].reset_index(drop=True)
  gdf["iso3"] = gdf["GID_0"].astype(str).str.strip()
  gdf["region_id"] = [assign_region_id(i, n1) for i, n1 in zip(gdf["iso3"], gdf["NAME_1"])]
  gdf["unit_id"] = np.arange(1, len(gdf) + 1, dtype=np.int32)

  gdf_54009 = gdf.to_crs("ESRI:54009")
  bnds = gdf_54009.bounds
  for c in ("minx", "miny", "maxx", "maxy"):
    gdf_54009[c] = bnds[c].values

  pyogrio.write_dataframe(gdf_54009, gpkg_out, layer="units")
  meta = pd.DataFrame(gdf_54009.drop(columns=["geometry"]))
  meta.to_parquet(meta_out, index=False)
  return gpkg_out, meta


def build_climate_cell_lookup() -> np.ndarray:
  ds = xr.open_dataset(DATA_PROCESSED / "climate_aligned/monthly_bias_offsets_2006_2025.nc")
  lats = ds["latitude"].values.astype(np.float64)
  lons = ds["longitude"].values.astype(np.float64)
  ds.close()

  lookup = np.full((360, 720), -1, dtype=np.int32)
  r_idx = np.clip(np.floor((90.0 - lats) * 2.0).astype(np.int32), 0, 359)
  c_idx = np.clip(np.floor((lons + 180.0) * 2.0).astype(np.int32), 0, 719)
  lookup[r_idx, c_idx] = np.arange(len(lats), dtype=np.int32)

  def to_xyz(la: np.ndarray, lo: np.ndarray) -> np.ndarray:
    la_r, lo_r = np.deg2rad(la), np.deg2rad(lo)
    cl = np.cos(la_r)
    return np.column_stack([cl * np.cos(lo_r), cl * np.sin(lo_r), np.sin(la_r)])

  tree = cKDTree(to_xyz(lats, lons))
  miss_r, miss_c = np.where(lookup < 0)
  miss_lat = 89.75 - 0.5 * miss_r
  miss_lon = -179.75 + 0.5 * miss_c
  max_chord = 2.0 * math.sin(MAX_COASTAL_SNAP_KM / (2.0 * EARTH_RADIUS_KM))
  dists, nn = tree.query(to_xyz(miss_lat, miss_lon), distance_upper_bound=max_chord)
  valid_snap = np.isfinite(dists) & (nn < len(lats))
  lookup[miss_r[valid_snap], miss_c[valid_snap]] = nn[valid_snap].astype(np.int32)
  return lookup


_W_LOOKUP: np.ndarray | None = None
_W_PATHS: dict[str, Path] | None = None
_W_TRANSFORM: Affine | None = None
_W_WIDTH: int = 0
_W_HEIGHT: int = 0


def _init_worker(lookup: np.ndarray, paths: dict[str, Path], transform: Affine, width: int, height: int) -> None:
  global _W_LOOKUP, _W_PATHS, _W_TRANSFORM, _W_WIDTH, _W_HEIGHT
  _W_LOOKUP = lookup
  _W_PATHS = paths
  _W_TRANSFORM = transform
  _W_WIDTH = width
  _W_HEIGHT = height


def _process_strip(job: tuple[int, list[tuple[int, float, float, bytes]]]) -> tuple[pd.DataFrame | None, dict[str, float]]:
  """Scan one 1024-row strip in 45,056-col chunks (<180 MB RAM per worker, zero GPKG lock)."""
  assert _W_LOOKUP is not None and _W_PATHS is not None and _W_TRANSFORM is not None
  row_start, poly_wkbs = job
  h = min(STRIP_HEIGHT, _W_HEIGHT - row_start)
  y_top = _W_TRANSFORM.f + row_start * _W_TRANSFORM.e
  transformer = Transformer.from_crs("ESRI:54009", "EPSG:4326", always_xy=True)

  # Decode strip geometries once for this strip
  decoded_polys = [(wkb.loads(wb), uid, minx, maxx) for uid, minx, maxx, wb in poly_wkbs]

  stats = {
      "raw_v25_m3": 0.0,
      "raw_nres25_m3": 0.0,
      "raw_pop25": 0.0,
      "raw_v20_m3": 0.0,
      "unassigned_gadm_v25_m3": 0.0,
      "unassigned_climate_v25_m3": 0.0,
  }
  chunk_dfs = []

  with rasterio.open(_W_PATHS["tot25"]) as s_t25, rasterio.open(_W_PATHS["nres25"]) as s_n25, \
       rasterio.open(_W_PATHS["pop25"]) as s_p25, rasterio.open(_W_PATHS["tot20"]) as s_t20, \
       rasterio.open(_W_PATHS["nres20"]) as s_n20:
    for c_start in range(0, _W_WIDTH, COL_CHUNK):
      w = min(COL_CHUNK, _W_WIDTH - c_start)
      win = Window(c_start, row_start, w, h)
      t25 = s_t25.read(1, window=win)
      t20 = s_t20.read(1, window=win)
      has_t25 = np.any((t25 > 0) & (t25 != NODATA_U32))
      has_t20 = np.any((t20 > 0) & (t20 != NODATA_U32))
      if not has_t25 and not has_t20:
        continue

      n25 = s_n25.read(1, window=win)
      n20 = s_n20.read(1, window=win)
      p25 = s_p25.read(1, window=win)

      t25_c = np.where(t25 == NODATA_U32, 0, t25).astype(np.int32)
      n25_c = np.where(n25 == NODATA_U32, 0, n25).astype(np.int32)
      t20_c = np.where(t20 == NODATA_U32, 0, t20).astype(np.int32)
      n20_c = np.where(n20 == NODATA_U32, 0, n20).astype(np.int32)
      p25_c = np.where((p25 < 0) | ~np.isfinite(p25), 0.0, p25).astype(np.float32)

      v25 = np.maximum(0, t25_c - n25_c)
      v20 = np.maximum(0, t20_c - n20_c)

      stats["raw_v25_m3"] += float(v25.sum(dtype=np.int64))
      stats["raw_nres25_m3"] += float(n25_c.sum(dtype=np.int64))
      stats["raw_pop25"] += float(p25_c.sum(dtype=np.float64))
      stats["raw_v20_m3"] += float(v20.sum(dtype=np.int64))

      active = (v25 > 0) | (v20 > 0) | (n25_c > 0)
      if not np.any(active):
        continue

      x_left = _W_TRANSFORM.c + c_start * _W_TRANSFORM.a
      x_right = x_left + w * _W_TRANSFORM.a
      sub_shapes = [(g, uid) for g, uid, mnx, mxx in decoded_polys if mxx >= x_left and mnx <= x_right]
      if not sub_shapes:
        stats["unassigned_gadm_v25_m3"] += float(v25.sum(dtype=np.int64))
        continue

      chunk_tr = Affine(_W_TRANSFORM.a, 0.0, x_left, 0.0, _W_TRANSFORM.e, y_top)
      uid_grid = features.rasterize(sub_shapes, out_shape=(h, w), transform=chunk_tr, fill=0, dtype="int32", all_touched=True)
      uid_exact = features.rasterize(sub_shapes, out_shape=(h, w), transform=chunk_tr, fill=0, dtype="int32", all_touched=False)
      mask_ex = uid_exact > 0
      uid_grid[mask_ex] = uid_exact[mask_ex]

      rr, cc = np.where(active)
      uids = uid_grid[rr, cc]
      v25_pts = v25[rr, cc].astype(np.float64)
      n25_pts = n25_c[rr, cc].astype(np.float64)
      v20_pts = v20[rr, cc].astype(np.float64)
      n20_pts = n20_c[rr, cc].astype(np.float64)
      p25_pts = p25_c[rr, cc].astype(np.float64)

      stats["unassigned_gadm_v25_m3"] += float(v25_pts[uids == 0].sum())

      xs = x_left + (cc.astype(np.float64) + 0.5) * _W_TRANSFORM.a
      ys = y_top + (rr.astype(np.float64) + 0.5) * _W_TRANSFORM.e
      lons, lats = transformer.transform(xs, ys)
      valid_ll = np.isfinite(lons) & np.isfinite(lats)

      c_idx_pts = np.full(len(rr), -1, dtype=np.int32)
      if np.any(valid_ll):
        r05 = np.clip(np.floor((90.0 - lats[valid_ll]) * 2.0).astype(np.int32), 0, 359)
        c05 = np.clip(np.floor((lons[valid_ll] + 180.0) * 2.0).astype(np.int32), 0, 719)
        c_idx_pts[valid_ll] = _W_LOOKUP[r05, c05]

      stats["unassigned_climate_v25_m3"] += float(v25_pts[(uids > 0) & (c_idx_pts < 0)].sum())

      keep = (uids > 0) & (c_idx_pts >= 0)
      if np.any(keep):
        dfc = pd.DataFrame({
            "unit_id": uids[keep],
            "cell_idx": c_idx_pts[keep],
            "v25_m3": v25_pts[keep],
            "nres25_m3": n25_pts[keep],
            "pop25": p25_pts[keep],
            "v20_m3": v20_pts[keep],
            "nres20_m3": n20_pts[keep],
            "built_ha_25": (v25_pts[keep] > 0).astype(np.int64),
        })
        chunk_dfs.append(dfc.groupby(["unit_id", "cell_idx"], as_index=False).sum())

  if not chunk_dfs:
    return None, stats
  agg = pd.concat(chunk_dfs, ignore_index=True).groupby(["unit_id", "cell_idx"], as_index=False).sum()
  return agg, stats


def main() -> None:
  out_dir = DATA_PROCESSED / "sinks_2025"
  out_dir.mkdir(parents=True, exist_ok=True)

  print("1. Ensuring GHSL E2020 GeoTIFFs are unzipped...", flush=True)
  tot20_tif, nres20_tif = ensure_ghsl_e2020_unzipped()

  print("2. Loading GADM 4.1 municipal unit layer in ESRI:54009...", flush=True)
  gpkg_path, meta_df = build_gadm_units(out_dir)

  print("3. Building 0.5-deg climate cell lookup table (52,160 land BAIT cells)...", flush=True)
  lookup = build_climate_cell_lookup()

  ghsl_dir = DATA_RAW / "sinks/2026-11-13/ghsl"
  paths = {
      "tot25": ghsl_dir / "GHS_BUILT_V_E2025_GLOBE_R2023A_54009_100_V1_0.tif",
      "nres25": ghsl_dir / "GHS_BUILT_V_NRES_E2025_GLOBE_R2023A_54009_100_V1_0.tif",
      "pop25": ghsl_dir / "GHS_POP_E2025_GLOBE_R2023A_54009_100_V1_0.tif",
      "tot20": tot20_tif,
      "nres20": nres20_tif,
  }
  with rasterio.open(paths["tot25"]) as src:
    transform = src.transform
    width, height = src.width, src.height

  print("4. Pre-indexing GADM WKB geometries by strip in parent process (zero worker GPKG locks)...", flush=True)
  gdf_u = pyogrio.read_dataframe(gpkg_path, layer="units", columns=["unit_id", "minx", "miny", "maxx", "maxy"])
  u_ids = gdf_u["unit_id"].values.astype(np.int32)
  u_minx = gdf_u["minx"].values.astype(np.float64)
  u_miny = gdf_u["miny"].values.astype(np.float64)
  u_maxx = gdf_u["maxx"].values.astype(np.float64)
  u_maxy = gdf_u["maxy"].values.astype(np.float64)
  u_wkbs = [wkb.dumps(g) for g in gdf_u.geometry]
  del gdf_u

  strip_starts = list(range(0, height, STRIP_HEIGHT))
  jobs = []
  for rs in strip_starts:
    h = min(STRIP_HEIGHT, height - rs)
    y_top = transform.f + rs * transform.e
    y_bot = transform.f + (rs + h) * transform.e
    s_miny, s_maxy = min(y_top, y_bot), max(y_top, y_bot)
    hit_idxs = np.where((u_maxy >= s_miny) & (u_miny <= s_maxy))[0]
    strip_polys = [(int(u_ids[i]), float(u_minx[i]), float(u_maxx[i]), u_wkbs[i]) for i in hit_idxs]
    jobs.append((rs, strip_polys))
  del u_wkbs

  print(f"5. Scanning {len(jobs)} strips ({height}x{width}) across {N_WORKERS} spawned workers (<4 GB total RAM)...", flush=True)
  dfs = []
  tot_stats = {
      "raw_v25_m3": 0.0, "raw_nres25_m3": 0.0, "raw_pop25": 0.0,
      "raw_v20_m3": 0.0, "unassigned_gadm_v25_m3": 0.0, "unassigned_climate_v25_m3": 0.0,
  }
  ctx = mp.get_context("spawn")
  with ProcessPoolExecutor(
      max_workers=N_WORKERS,
      mp_context=ctx,
      initializer=_init_worker,
      initargs=(lookup, paths, transform, width, height),
  ) as ex:
    for idx, (df_part, st) in enumerate(ex.map(_process_strip, jobs), start=1):
      if df_part is not None:
        dfs.append(df_part)
      for k, v in st.items():
        tot_stats[k] += v
      if idx % 25 == 0 or idx == len(jobs):
        print(f"   Completed {idx}/{len(jobs)} strips (cum v25={tot_stats['raw_v25_m3']/1e9:.2f} billion m3)...", flush=True)

  df_unit_cell = pd.concat(dfs, ignore_index=True).groupby(["unit_id", "cell_idx"], as_index=False).sum()
  df_unit_cell.to_parquet(out_dir / "ghsl_unit_cell_volumes.parquet", index=False)

  df_merged = df_unit_cell.merge(meta_df[["unit_id", "iso3", "region_id"]], on="unit_id", how="left")
  df_reg_cell = df_merged.groupby(["region_id", "iso3", "cell_idx"], as_index=False)[
      ["v25_m3", "nres25_m3", "pop25", "v20_m3", "nres20_m3", "built_ha_25"]
  ].sum()
  df_reg_cell.to_parquet(out_dir / "ghsl_region_cell_volumes.parquet", index=False)

  kept_v25 = float(df_reg_cell["v25_m3"].sum())
  receipt = {
      "n_gadm_units_total": int(len(meta_df)),
      "n_gadm_units_with_buildings": int(df_unit_cell["unit_id"].nunique()),
      "n_countries_with_buildings": int(df_reg_cell["iso3"].nunique()),
      "n_regions_with_buildings": int(df_reg_cell["region_id"].nunique()),
      "global_res_mixed_v25_billion_m3": kept_v25 / 1e9,
      "global_nres_v25_billion_m3": float(df_reg_cell["nres25_m3"].sum() / 1e9),
      "global_res_mixed_v20_billion_m3": float(df_reg_cell["v20_m3"].sum() / 1e9),
      "global_pop25_billion": float(df_reg_cell["pop25"].sum() / 1e9),
      "unassigned_gadm_share_pct": 100.0 * tot_stats["unassigned_gadm_v25_m3"] / tot_stats["raw_v25_m3"],
      "unassigned_climate_share_pct": 100.0 * tot_stats["unassigned_climate_v25_m3"] / tot_stats["raw_v25_m3"],
      "raw_totals": tot_stats,
  }
  (out_dir / "ghsl_aggregation_receipt.json").write_text(json.dumps(receipt, indent=2))
  print("\n=== DONE: GHSL 100m TO CLIMATE CELL AGGREGATION ===", flush=True)
  print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
  main()
