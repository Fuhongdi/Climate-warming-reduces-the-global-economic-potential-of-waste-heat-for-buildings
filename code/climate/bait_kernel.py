"""Daily weather-response adapter for residential heat-demand calibration.

No output is an absolute heat demand. Implements Demand.ninja's published daily
BAIT kernel (before its hourly spline) using numpy for multiple climate cells.
Canonical upstream code is fetched separately and parity is tested when present.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class Parameters:
    heating_threshold_c: float = 14.0
    smoothing: float = 0.5
    solar_gains: float = 0.012
    wind_chill: float = -0.20
    humidity_discomfort: float = 0.05
    input_wind_height_m: float = 10.0
    wind_conversion: str = "fao56_short_grass_reference"
    dhw_equivalent_use_c: float = 45.0

PARAMETERS = Parameters()

def normalize_daily_dates(dates):
    """Use source calendar date; no UTC shifting of noon-labelled daily means."""
    idx = pd.DatetimeIndex(dates)
    if idx.tz is not None:
        raise ValueError("Daily labels must be timezone-naive source calendar dates")
    idx = idx.normalize()
    if not idx.is_unique or not idx.is_monotonic_increasing:
        raise ValueError("Duplicate or unsorted calendar dates")
    if len(idx) > 1 and not np.all(np.diff(idx.values) == np.timedelta64(1, 'D')):
        raise ValueError("Daily gaps must be separate windows; never smooth across a gap")
    return idx

def wind_2m_factor(height_m):
    """FAO56 Eq.47, short-grass reference; not observed global urban roughness."""
    if not np.isfinite(height_m) or height_m < 1.0:
        raise ValueError("Wind measurement height must be >=1 m")
    return 4.87 / np.log(67.8 * height_m - 5.42)

def bait_daily_si(tas_k, huss_kgkg, sfcwind_ms, rsds_wm2, dates, params=PARAMETERS):
    """Inputs share (time, ...cell axes); caller supplies >=2 spin-up days.

    Initial two rows repeat first-day lag exactly like upstream. They are tagged
    by the runner and excluded from analysis. Missing/invalid input is rejected.
    """
    idx = normalize_daily_dates(dates)
    arrays = [np.asarray(x, dtype=np.float64) for x in (tas_k, huss_kgkg, sfcwind_ms, rsds_wm2)]
    if not arrays or arrays[0].ndim < 1 or len(idx) != arrays[0].shape[0] or len(idx) < 3:
        raise ValueError("Require >=3 daily samples with matching time axis")
    if any(a.shape != arrays[0].shape or not np.isfinite(a).all() for a in arrays):
        raise ValueError("All four variables require identical shapes and finite values")
    t_k, q, w, solar = arrays
    if np.any((t_k <= 0) | (q < 0) | (q > 1) | (w < 0) | (solar < 0)):
        raise ValueError("Input violates physical range / SI unit contract")
    if params.wind_conversion != "fao56_short_grass_reference":
        raise ValueError("No unrecorded wind-height conversion allowed")
    t = t_k - 273.15
    h = q * 1000.0
    u2 = w * wind_2m_factor(params.input_wind_height_m)
    sun_reference = 100.0 + 7.0 * t
    wind_reference = 4.5 - 0.025 * t
    humidity_reference = np.exp(1.1 + 0.06 * t)
    apparent = t + (solar - sun_reference) * params.solar_gains + (u2 - wind_reference) * params.wind_chill
    apparent += (apparent - 16.0) * (h - humidity_reference) / 1000.0 * params.humidity_discomfort
    lag1 = np.concatenate((apparent[:1], apparent[:-1]), axis=0)
    lag2 = np.concatenate((apparent[:1], lag1[:-1]), axis=0)
    s = params.smoothing
    smoothed = (apparent + s * lag1 + s*s * lag2) / (1.0 + s + s*s)
    blend = 0.5 / (1.0 + np.exp(-(t - 19.0) * 10.0 / 8.0))
    bait = t * blend + smoothed * (1.0 - blend)
    return {"bait_c": bait, "hdd_k_day": np.maximum(params.heating_threshold_c - bait, 0.0), "wind_2m_reference_ms": u2}

def mains_temperature_c(annual_mean_c, monthly_range_c, day_of_year, latitude):
    """EnergyPlus v26.1 WaterMainsTempFromCorrelation, multiplier=1, offset=0.

    Fahrenheit regression, source hemisphere sign, and source 0 C lower bound.
    Inputs broadcast; day_of_year should be (time,1) for vector cells.
    """
    mean = np.asarray(annual_mean_c, dtype=np.float64)
    span = np.asarray(monthly_range_c, dtype=np.float64)
    lat = np.asarray(latitude, dtype=np.float64)
    day = np.asarray(day_of_year, dtype=np.float64)
    if not all(np.isfinite(x).all() for x in (mean, span, lat, day)) or np.any(span < 0):
        raise ValueError("Invalid water-mains climate inputs")
    if np.any(np.abs(lat) > 90) or np.any((day < 1) | (day > 366)):
        raise ValueError("Invalid latitude or day of year")
    mean_f = mean * 1.8 + 32.0
    span_f = span * 1.8
    ratio = 0.4 + 0.01 * (mean_f - 44.0)
    lag = 35.0 - (mean_f - 44.0)
    sign = np.where(lat >= 0, 1.0, -1.0)
    raw_f = mean_f + 6.0 + ratio * span_f / 2.0 * sign * np.sin(np.deg2rad(0.986 * (day - 15.0 - lag) - 90.0))
    result_c = (np.maximum(raw_f, 32.0) - 32.0) / 1.8
    return result_c

def mains_from_daily_air(tas_k, dates, latitudes):
    """Annual means and 12 calendar-month means from each full source year.

    Uses each year's weather, not a fixed historical climatology. This choice is
    an offline climate-response approximation, not a causal water-network model.
    """
    idx = normalize_daily_dates(dates)
    t = np.asarray(tas_k, dtype=np.float64) - 273.15
    if t.ndim == 1:
        t = t[:, None]
    if t.ndim != 2 or t.shape[0] != len(idx) or not np.isfinite(t).all():
        raise ValueError("Expected finite (daily_time, cell) temperatures")
    lat = np.broadcast_to(np.asarray(latitudes, dtype=float), (t.shape[1],))
    out = np.empty_like(t)
    for year in np.unique(idx.year):
        m = idx.year == year
        expected = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
        if not idx[m].equals(expected):
            raise ValueError(f"Water model requires complete calendar year {year}")
        a = t[m]
        monthly = np.stack([a[idx[m].month == mo].mean(axis=0) for mo in range(1,13)])
        mean = a.mean(axis=0)
        span = monthly.max(axis=0) - monthly.min(axis=0)
        out[m] = mains_temperature_c(mean, span, idx[m].dayofyear.to_numpy()[:,None], lat)
    return out

def provenance():
    return {
        "parameters": asdict(PARAMETERS),
        "wind_factor_10m_to_2m": wind_2m_factor(10.0),
        "wind_assumption": "FAO56 short-grass reference profile; not global measured urban roughness",
        "daily_model": "Demand.ninja daily BAIT stage; does not reconstruct hourly weather",
        "upstream": "https://github.com/renewables-ninja/demand-ninja",
        "wind_source": "https://www.fao.org/4/X0490E/x0490e07.htm#wind%20speed",
        "water_source": "https://raw.githubusercontent.com/NREL/EnergyPlus/v26.1.0/src/EnergyPlus/WeatherManager.cc",
        "absolute_heat_demand": False,
        "annual_renormalization": False,
        "calibration_status": "Country residential SH and DHW useful-heat calibration still required",
    }
