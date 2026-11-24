#!/usr/bin/env python3
"""Complete missing ISIMIP3b BAIT years (2001-2015), extract daily populated-cell tas_c, and compute 2006-2025 monthly bias alignment offsets."""
from __future__ import annotations

import os
for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
  os.environ[_k] = "1"

import concurrent.futures
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import netCDF4 as nc
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("WASTE_HEAT_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "code/climate/vendor/demand_ninja"))

from code.climate.bait_kernel import PARAMETERS, bait_daily_si
from demand_ninja.core import _bait as upstream_bait

BAIT_DIR = ROOT / "data/processed/bait"
TAS_DIR = ROOT / "data/processed/tas"
ALIGNED_DIR = ROOT / "data/processed/climate_aligned"
RAW_OBS = ROOT / "data/raw/gswp3_era5_daily/v20261020"
RAW_ISIMIP = ROOT / "data/raw/isimip3b_daily"
RAW_ISIMIP_GAP = ROOT / "data/raw/climate/isimip3b_gap_2061_2080"

MODELS = [
    "gfdl-esm4",
    "ipsl-cm6a-lr",
    "mpi-esm1-2-hr",
    "mri-esm2-0",
    "ukesm1-0-ll",
]
SSPS = ["ssp126", "ssp245", "ssp370"]
VARS = ["tas", "huss", "sfcwind", "rsds"]


def sha256_file(p: Path) -> str:
  h = hashlib.sha256()
  with p.open("rb") as f:
    for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
      h.update(b)
  return h.hexdigest()


def load_cells() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  df = pd.read_csv(BAIT_DIR / "cells.csv")
  cids = df["cell_id"].to_numpy(dtype=np.int32)
  lats = df["latitude"].to_numpy(dtype=np.float64)
  lons = df["longitude"].to_numpy(dtype=np.float64)
  if "row" in df.columns and "col" in df.columns:
    rows = df["row"].to_numpy(dtype=np.int64)
    cols = df["col"].to_numpy(dtype=np.int64)
  else:
    rows = np.rint((89.75 - lats) / 0.5).astype(np.int64)
    cols = np.rint((lons + 179.75) / 0.5).astype(np.int64)
  assert np.all((rows >= 0) & (rows < 360) & (cols >= 0) & (cols < 720))
  assert np.allclose(89.75 - 0.5 * rows, lats, atol=1e-5)
  assert np.allclose(-179.75 + 0.5 * cols, lons, atol=1e-5)
  return cids, lats, lons, rows, cols


def find_raw_nc(series: str, var: str, year: int) -> Path:
  if series == "gswp3-era5__obsclim":
    for p in sorted(RAW_OBS.glob(f"gswp3-era5_obsclim_{var}_global_daily_*.nc")):
      parts = p.stem.split("_")
      sy, ey = int(parts[-2]), int(parts[-1])
      if sy <= year <= ey:
        return p
    raise FileNotFoundError(f"Missing obsclim {var} for {year}")

  model, scen = series.split("__")
  search_roots = [RAW_ISIMIP / model / scen]
  if scen in SSPS and 2061 <= year <= 2080:
    search_roots.insert(0, RAW_ISIMIP_GAP / model / scen)
  for sroot in search_roots:
    if not sroot.exists():
      continue
    for p in sorted(sroot.rglob(f"{model}_*_{scen}_{var}_global_daily_*.nc")):
      parts = p.stem.split("_")
      sy, ey = int(parts[-2]), int(parts[-1])
      if sy <= year <= ey:
        return p
  raise FileNotFoundError(f"Missing {series} {var} for {year}")


def read_var_slice(
    nc_path: Path,
    var: str,
    start_date: str,
    end_date: str,
    rows: np.ndarray,
    cols: np.ndarray,
) -> tuple[pd.DatetimeIndex, np.ndarray]:
  with nc.Dataset(nc_path, "r") as ds:
    tvar = ds["time"]
    raw_dates = nc.num2date(tvar[:], tvar.units, getattr(tvar, "calendar", "standard"))
    dates = pd.DatetimeIndex([pd.Timestamp(d.year, d.month, d.day) for d in raw_dates])
    mask = (dates >= pd.Timestamp(start_date)) & (dates <= pd.Timestamp(end_date))
    idx = np.nonzero(mask)[0]
    assert len(idx) > 0, f"No dates in [{start_date}, {end_date}] in {nc_path}"
    t0, t1 = int(idx[0]), int(idx[-1]) + 1
    assert np.array_equal(idx, np.arange(t0, t1))
    # Sequential hyperslab read into RAM, then fast NumPy cell gather
    cube = np.asarray(ds[var][t0:t1, :, :], dtype=np.float64)
    vals = cube[:, rows, cols]
    return dates[t0:t1], vals


def write_annual_nc(
    dst_nc: Path,
    var_name: str,
    units: str,
    long_name: str,
    dates: pd.DatetimeIndex,
    cids: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    arr_f32: np.ndarray,
) -> dict:
  dst_nc.parent.mkdir(parents=True, exist_ok=True)
  assert arr_f32.shape == (len(dates), len(cids))
  assert np.isfinite(arr_f32).all(), f"Non-finite values in {dst_nc}"
  tmp_nc = dst_nc.with_suffix(".nc.tmp")
  with nc.Dataset(tmp_nc, "w") as ds_out:
    ds_out.createDimension("time", len(dates))
    ds_out.createDimension("cell", len(cids))
    t_var = ds_out.createVariable("time", "f8", ("time",))
    t_var.units = "days since 1900-01-01 00:00:00"
    t_var.calendar = "proleptic_gregorian"
    t_var[:] = nc.date2num(dates.to_pydatetime(), t_var.units, t_var.calendar)
    c_var = ds_out.createVariable("cell_id", "i4", ("cell",))
    c_var[:] = cids
    lat_var = ds_out.createVariable("latitude", "f8", ("cell",))
    lat_var.units = "degrees_north"
    lat_var[:] = lats
    lon_var = ds_out.createVariable("longitude", "f8", ("cell",))
    lon_var.units = "degrees_east"
    lon_var[:] = lons
    v_var = ds_out.createVariable(
        var_name,
        "f4",
        ("time", "cell"),
        zlib=True,
        complevel=4,
        chunksizes=(len(dates), min(1024, len(cids))),
        fill_value=np.float32(np.nan),
    )
    v_var.units = units
    v_var.long_name = long_name
    v_var[:, :] = arr_f32
  tmp_nc.replace(dst_nc)
  qc = {
      "status": "complete",
      "file": str(dst_nc.relative_to(ROOT)),
      "days": len(dates),
      "cells": len(cids),
      "bytes": dst_nc.stat().st_size,
      "sha256": sha256_file(dst_nc),
      "range": [float(arr_f32.min()), float(arr_f32.max())],
      "missing_values": 0,
  }
  dst_nc.with_suffix(".qc.json").write_text(json.dumps(qc, indent=2) + "\n", encoding="utf-8")
  return qc


def compute_one_bait_year(job: tuple[str, int]) -> str:
  series, year = job
  dst_nc = BAIT_DIR / series / f"year_{year}.nc"
  if dst_nc.exists() and dst_nc.stat().st_size > 10_000_000:
    return f"SKIP {series} {year}"

  cids, lats, lons, rows, cols = CELL_META
  model, scen = series.split("__")

  # Determine spin-up source (last 2 days of year-1)
  has_spinup = True
  if scen == "historical" and year == 2001:
    has_spinup = False
    prev_series = None
  elif scen in SSPS and year == 2015:
    prev_series = f"{model}__historical"
  else:
    prev_series = series

  var_arrays = {}
  dates_full = None
  for v in VARS:
    cur_p = find_raw_nc(series, v, year)
    d_cur, a_cur = read_var_slice(cur_p, v, f"{year}-01-01", f"{year}-12-31", rows, cols)
    if has_spinup and prev_series is not None:
      prev_p = find_raw_nc(prev_series, v, year - 1)
      d_prev, a_prev = read_var_slice(
          prev_p, v, f"{year - 1}-12-30", f"{year - 1}-12-31", rows, cols
      )
      d_cat = d_prev.append(d_cur)
      a_cat = np.concatenate([a_prev, a_cur], axis=0)
    else:
      d_cat, a_cat = d_cur, a_cur
    if dates_full is None:
      dates_full = d_cat
    var_arrays[v] = a_cat

  assert dates_full is not None
  res = bait_daily_si(
      var_arrays["tas"],
      var_arrays["huss"],
      var_arrays["sfcwind"],
      var_arrays["rsds"],
      dates_full,
      params=PARAMETERS,
  )
  bait_full = res["bait_c"]
  # Verify parity on cell 0 against upstream demand_ninja.core._bait
  w2m_factor = 4.87 / np.log(67.8 * 10.0 - 5.42)
  df_chk = pd.DataFrame(
      {
          "temperature": var_arrays["tas"][:, 0] - 273.15,
          "humidity": var_arrays["huss"][:, 0] * 1000.0,
          "wind_speed_2m": var_arrays["sfcwind"][:, 0] * w2m_factor,
          "radiation_global_horizontal": var_arrays["rsds"][:, 0],
      }
  )
  up_chk = upstream_bait(
      df_chk,
      smoothing=PARAMETERS.smoothing,
      solar_gains=PARAMETERS.solar_gains,
      wind_chill=PARAMETERS.wind_chill,
      humidity_discomfort=PARAMETERS.humidity_discomfort,
  ).to_numpy()
  max_err = float(np.max(np.abs(bait_full[:, 0] - up_chk)))
  assert max_err < 1e-9, f"BAIT parity error {max_err} in {series} {year}"

  offset = 2 if has_spinup else 0
  dates_yr = dates_full[offset:]
  bait_yr = bait_full[offset:, :].astype(np.float32)
  write_annual_nc(
      dst_nc,
      "bait_c",
      "degC",
      "Building-adjusted internal temperature, daily weather response only",
      dates_yr,
      cids,
      lats,
      lons,
      bait_yr,
  )
  return f"DONE {series} {year} (parity={max_err:.2e} C)"


def extract_tas_chunk(job: tuple[str, Path]) -> str:
  series, src_nc = job
  cids, lats, lons, rows, cols = CELL_META
  parts = src_nc.stem.split("_")
  sy, ey = int(parts[-2]), int(parts[-1])
  sy_clip = max(sy, 1946 if series == "gswp3-era5__obsclim" else 2001)
  if sy_clip > ey:
    return f"SKIP_RANGE {src_nc.name}"

  needed_years = [
      y
      for y in range(sy_clip, ey + 1)
      if not (TAS_DIR / series / f"year_{y}.nc").exists()
  ]
  if not needed_years:
    return f"SKIP_ALL {series} {sy_clip}_{ey}"

  with nc.Dataset(src_nc, "r") as ds:
    tvar = ds["time"]
    raw_dates = nc.num2date(tvar[:], tvar.units, getattr(tvar, "calendar", "standard"))
    dates = pd.DatetimeIndex([pd.Timestamp(d.year, d.month, d.day) for d in raw_dates])
    for yr in needed_years:
      mask = dates.year == yr
      idx = np.nonzero(mask)[0]
      t0, t1 = int(idx[0]), int(idx[-1]) + 1
      cube_c = np.asarray(ds["tas"][t0:t1, :, :], dtype=np.float32)[:, rows, cols] - np.float32(273.15)
      write_annual_nc(
          TAS_DIR / series / f"year_{yr}.nc",
          "tas_c",
          "degC",
          "Daily mean near-surface air temperature",
          dates[t0:t1],
          cids,
          lats,
          lons,
          cube_c,
      )
  return f"TAS_DONE {series} {needed_years[0]}..{needed_years[-1]}"


CELL_META = None


def compute_monthly_climatology(
    base_dir: Path, var_name: str, series_years: list[tuple[str, int]]
) -> np.ndarray:
  """Compute exact daily-weighted 20-year calendar-month mean (12, n_cells) in float64."""
  n_cells = len(CELL_META[0])
  sums = np.zeros((12, n_cells), dtype=np.float64)
  counts = np.zeros(12, dtype=np.int64)
  for ser, yr in series_years:
    p = base_dir / ser / f"year_{yr}.nc"
    with nc.Dataset(p, "r") as ds:
      tvar = ds["time"]
      raw_d = nc.num2date(tvar[:], tvar.units, getattr(tvar, "calendar", "proleptic_gregorian"))
      months = np.array([d.month for d in raw_d], dtype=np.int32)
      arr = np.asarray(ds[var_name][:, :], dtype=np.float64)
      for m in range(1, 13):
        sel = months == m
        sums[m - 1, :] += arr[sel, :].sum(axis=0)
        counts[m - 1] += int(sel.sum())
  assert np.all(counts > 0)
  return sums / counts[:, None]


def main() -> None:
  global CELL_META
  CELL_META = load_cells()
  cids, lats, lons, _, _ = CELL_META
  print(f"Loaded {len(cids)} populated 0.5-deg cells.", flush=True)

  # 1. Compute missing BAIT years: historical 2001-2014 (5 models) + SSP 2015 (15 runs)
  # First run historical 2001-2014 so 2014-12-30..31 raw files are verified, then SSP 2015
  hist_jobs = [(f"{m}__historical", yr) for m in MODELS for yr in range(2001, 2015)]
  ssp_2015_jobs = [(f"{m}__{s}", 2015) for m in MODELS for s in SSPS]
  all_bait_jobs = hist_jobs + ssp_2015_jobs
  print(f"1. Computing/verifying {len(all_bait_jobs)} BAIT model-years with 16 workers...", flush=True)
  with concurrent.futures.ProcessPoolExecutor(max_workers=16) as pool:
    for i, msg in enumerate(pool.map(compute_one_bait_year, all_bait_jobs), 1):
      if i % 10 == 0 or i == len(all_bait_jobs) or msg.startswith("DONE"):
        print(f"  [{i}/{len(all_bait_jobs)}] {msg}", flush=True)

  # 2. Extract daily tas_c on the 52,160 populated cells across obsclim (1946-2025), historical (2001-2014), and 15 SSPs (2015-2100)
  tas_jobs: list[tuple[str, Path]] = []
  for p in sorted(RAW_OBS.glob("gswp3-era5_obsclim_tas_global_daily_*.nc")):
    tas_jobs.append(("gswp3-era5__obsclim", p))
  for m in MODELS:
    for p in sorted((RAW_ISIMIP / m / "historical").rglob(f"{m}_*_historical_tas_global_daily_*.nc")):
      tas_jobs.append((f"{m}__historical", p))
    for s in SSPS:
      for sroot in [RAW_ISIMIP / m / s, RAW_ISIMIP_GAP / m / s]:
        if sroot.exists():
          for p in sorted(sroot.rglob(f"{m}_*_{s}_tas_global_daily_*.nc")):
            tas_jobs.append((f"{m}__{s}", p))

  print(f"2. Extracting populated-cell daily tas_c from {len(tas_jobs)} NetCDF chunks (24 workers)...", flush=True)
  with concurrent.futures.ProcessPoolExecutor(max_workers=24) as pool:
    for i, msg in enumerate(pool.map(extract_tas_chunk, tas_jobs), 1):
      if i % 20 == 0 or i == len(tas_jobs):
        print(f"  [{i}/{len(tas_jobs)}] {msg}", flush=True)

  # 3. Compute 2006-2025 Monthly Mean Bias Alignment Offsets (obsclim minus model)
  print("3. Computing 2006-2025 grid-cell monthly mean bias alignment offsets...", flush=True)
  ALIGNED_DIR.mkdir(parents=True, exist_ok=True)
  obs_2006_2025 = [("gswp3-era5__obsclim", yr) for yr in range(2006, 2026)]
  obs_bait_clim = compute_monthly_climatology(BAIT_DIR, "bait_c", obs_2006_2025)
  obs_tas_clim = compute_monthly_climatology(TAS_DIR, "tas_c", obs_2006_2025)

  run_names = [f"{m}__{s}" for m in MODELS for s in SSPS]
  bait_offsets = np.zeros((len(run_names), 12, len(cids)), dtype=np.float32)
  tas_offsets = np.zeros((len(run_names), 12, len(cids)), dtype=np.float32)

  summary_rows = []
  for r_idx, rname in enumerate(run_names):
    m, s = rname.split("__")
    mod_2006_2025 = [(f"{m}__historical", yr) for yr in range(2006, 2015)] + [
        (rname, yr) for yr in range(2015, 2026)
    ]
    mod_bait_clim = compute_monthly_climatology(BAIT_DIR, "bait_c", mod_2006_2025)
    mod_tas_clim = compute_monthly_climatology(TAS_DIR, "tas_c", mod_2006_2025)
    d_bait = obs_bait_clim - mod_bait_clim
    d_tas = obs_tas_clim - mod_tas_clim
    bait_offsets[r_idx] = d_bait.astype(np.float32)
    tas_offsets[r_idx] = d_tas.astype(np.float32)

    # Verify zero residual after alignment
    res_bait = float(np.max(np.abs((mod_bait_clim + bait_offsets[r_idx]) - obs_bait_clim)))
    res_tas = float(np.max(np.abs((mod_tas_clim + tas_offsets[r_idx]) - obs_tas_clim)))
    assert res_bait < 1e-5 and res_tas < 1e-5
    summary_rows.append({
        "run": rname,
        "mean_bait_offset_c": float(d_bait.mean()),
        "p05_bait_offset_c": float(np.percentile(d_bait, 5)),
        "p95_bait_offset_c": float(np.percentile(d_bait, 95)),
        "mean_tas_offset_c": float(d_tas.mean()),
        "max_alignment_residual_c": max(res_bait, res_tas),
    })
    print(
        f"  [{rname}] mean dBAIT={d_bait.mean():+.3f} C, mean dTAS={d_tas.mean():+.3f} C, residual={max(res_bait, res_tas):.2e} C",
        flush=True,
    )

  out_nc = ALIGNED_DIR / "monthly_bias_offsets_2006_2025.nc"
  tmp_nc = out_nc.with_suffix(".nc.tmp")
  with nc.Dataset(tmp_nc, "w") as ds:
    ds.createDimension("run", len(run_names))
    ds.createDimension("month", 12)
    ds.createDimension("cell", len(cids))
    r_var = ds.createVariable("run", str, ("run",))
    for i, rn in enumerate(run_names):
      r_var[i] = rn
    m_var = ds.createVariable("month", "i4", ("month",))
    m_var[:] = np.arange(1, 13, dtype=np.int32)
    c_var = ds.createVariable("cell_id", "i4", ("cell",))
    c_var[:] = cids
    lat_var = ds.createVariable("latitude", "f8", ("cell",))
    lat_var[:] = lats
    lon_var = ds.createVariable("longitude", "f8", ("cell",))
    lon_var[:] = lons

    ob_v = ds.createVariable("obs_bait_clim_2006_2025", "f4", ("month", "cell"), zlib=True, complevel=4)
    ob_v[:, :] = obs_bait_clim.astype(np.float32)
    ot_v = ds.createVariable("obs_tas_clim_2006_2025", "f4", ("month", "cell"), zlib=True, complevel=4)
    ot_v[:, :] = obs_tas_clim.astype(np.float32)

    bo_v = ds.createVariable("bait_offset_c", "f4", ("run", "month", "cell"), zlib=True, complevel=4)
    bo_v.long_name = "Additive monthly BAIT bias offset: obs(2006-2025) minus model(2006-2025)"
    bo_v[:, :, :] = bait_offsets

    to_v = ds.createVariable("tas_offset_c", "f4", ("run", "month", "cell"), zlib=True, complevel=4)
    to_v.long_name = "Additive monthly TAS bias offset: obs(2006-2025) minus model(2006-2025)"
    to_v[:, :, :] = tas_offsets
  tmp_nc.replace(out_nc)

  pd.DataFrame(summary_rows).to_csv(
      ALIGNED_DIR / "monthly_bias_alignment_summary_2006_2025.csv", index=False
  )
  (ALIGNED_DIR / "alignment_receipt.json").write_text(
      json.dumps(
          {
              "completed_utc": datetime.now(timezone.utc).isoformat(),
              "baseline_window": "2006-2025",
              "cells": len(cids),
              "runs": run_names,
              "nc_sha256": sha256_file(out_nc),
              "summary": summary_rows,
          },
          indent=2,
      )
      + "\n",
      encoding="utf-8",
  )
  print("ALL BAIT, TAS, AND 2006-2025 MONTHLY BIAS ALIGNMENT COMPLETE!", flush=True)


if __name__ == "__main__":
  main()
