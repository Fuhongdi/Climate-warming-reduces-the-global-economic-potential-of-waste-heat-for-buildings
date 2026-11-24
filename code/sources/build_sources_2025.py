"""Compile the 2025 global facility-level waste heat source inventory across power, industry and wastewater."""
from __future__ import annotations

import glob
import json
import math
import re
from pathlib import Path
import unicodedata
import zipfile

import numpy as np
import openpyxl
import pandas as pd
from scipy.spatial import cKDTree

from code.common.parameters import (
    CRUDE_TONNES_PER_BBL,
    MIN_SOURCE_POWER_MW,
    POWER_TECH_SPECS,
    SEENERGIES_FACTORS,
    WASTEWATER_DELTA_T_K,
)
from code.common.paths import (
    DATA_PROCESSED,
    DATA_RAW,
    GJ_PER_MW_DAY,
    GJ_PER_MWH,
    WATER_VOLUMETRIC_HEAT_CAPACITY_MJ_M3_K,
)

DAYS_2025 = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype=np.float64)
MIN_ANNUAL_GJ = MIN_SOURCE_POWER_MW * 365.0 * GJ_PER_MW_DAY  # 31,536.0 GJ/yr
EARTH_RADIUS_KM = 6371.0088
MATCH_RADIUS_KM = 5.0


def latlon_to_xyz(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
  lat = np.deg2rad(np.asarray(lat_deg, dtype=np.float64))
  lon = np.deg2rad(np.asarray(lon_deg, dtype=np.float64))
  cos_lat = np.cos(lat)
  return np.column_stack([cos_lat * np.cos(lon), cos_lat * np.sin(lon), np.sin(lat)])


def chord_from_km(km: float) -> float:
  return 2.0 * math.sin(km / (2.0 * EARTH_RADIUS_KM))


def norm_name(s: object) -> str:
  if s is None or (isinstance(s, float) and math.isnan(s)):
    return ""
  nfkd = unicodedata.normalize("NFKD", str(s))
  return re.sub(r"[^a-z0-9]+", "", nfkd.encode("ascii", "ignore").decode("ascii").lower())


def parse_coordinates_col(series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
  lats = np.full(len(series), np.nan, dtype=np.float64)
  lons = np.full(len(series), np.nan, dtype=np.float64)
  for idx, val in enumerate(series):
    if isinstance(val, str) and "," in val:
      parts = val.split(",")
      try:
        lats[idx] = float(parts[0].strip())
        lons[idx] = float(parts[1].strip())
      except ValueError:
        pass
  return lats, lons


def load_cooling_inventories() -> tuple[cKDTree, pd.DataFrame, cKDTree, pd.DataFrame]:
  """Load US EIA-860 (2025) and Lohrmann et al. (2019 Nature Energy) cooling tables."""
  cool_dir = DATA_RAW / "supplements_2025/power_cooling"
  # 1. US EIA-860 2025
  with zipfile.ZipFile(cool_dir / "eia8602025.zip") as zf:
    df_pl = pd.read_excel(zf.open("2___Plant_Y2025.xlsx"), skiprows=1, usecols=["Plant Code", "Latitude", "Longitude"])
    df_cl = pd.read_excel(
        zf.open("6_2_EnviroEquip_Y2025.xlsx"),
        sheet_name="Cooling",
        skiprows=1,
        usecols=["Plant Code", "Cooling Status", "Cooling Type 1", "Cooling Type 2", "Percent Dry Cooling"],
    )
  df_cl = df_cl[df_cl["Cooling Status"].astype(str).str.upper().isin(["OP", "SB", "OA"])].copy()

  def eia_dry_frac(row: pd.Series) -> float:
    p = pd.to_numeric(row["Percent Dry Cooling"], errors="coerce")
    if pd.notna(p) and 0.0 <= p <= 100.0:
      return float(p) / 100.0
    t1 = str(row["Cooling Type 1"]).strip().upper()
    t2 = str(row["Cooling Type 2"]).strip().upper()
    if t1 in ("DC", "zen") or "DC" in (t1, t2):
      if t2 in ("ON", "RF", "RI", "RN", "RC", "OC") or t1 in ("ON", "RF", "RI", "RN", "RC", "OC", "HR", "HI"):
        return 0.5
      return 1.0
    if t1 in ("HR", "HI"):  # hybrid cooling system
      return 0.5
    return 0.0

  df_cl["dry_frac"] = df_cl.apply(eia_dry_frac, axis=1)
  eia_plant = df_cl.groupby("Plant Code", as_index=False)["dry_frac"].mean().merge(df_pl, on="Plant Code", how="inner")
  eia_plant["Latitude"] = pd.to_numeric(eia_plant["Latitude"], errors="coerce")
  eia_plant["Longitude"] = pd.to_numeric(eia_plant["Longitude"], errors="coerce")
  eia_plant = eia_plant.dropna(subset=["Latitude", "Longitude"]).reset_index(drop=True)
  eia_tree = cKDTree(latlon_to_xyz(eia_plant["Latitude"].values, eia_plant["Longitude"].values))

  # 2. Lohrmann et al. (2019, Nature Energy) global unit cooling inventory
  df_loh = pd.read_excel(cool_dir / "Lohrmann2019_MOESM2.xlsx", sheet_name="List of power plants")
  df_loh["Latitude"] = pd.to_numeric(df_loh["Latitude"], errors="coerce")
  df_loh["Longitude"] = pd.to_numeric(df_loh["Longitude"], errors="coerce")
  df_loh["cap_mw"] = pd.to_numeric(df_loh["Active Capacity [MW]"], errors="coerce").fillna(50.0).clip(lower=1.0)
  tech_str = df_loh["Technology"].astype(str).str.lower()
  # Keep steam Rankine & CCGT units for condenser cooling lookup (exclude pure OCGT / ICE 'Dry'/'Inlet')
  is_steam_or_ccgt = ~tech_str.str.contains("gas turbine|internal combustion", regex=True) | tech_str.str.contains("combined cycle|ccgt", regex=True)
  df_loh_steam = df_loh[is_steam_or_ccgt & df_loh["Latitude"].notna() & df_loh["Longitude"].notna()].copy().reset_index(drop=True)

  def loh_dry_frac(cs: object) -> float:
    s = str(cs).strip().lower()
    if s == "dry":
      return 1.0
    if "dry" in s:
      return 0.5
    return 0.0

  df_loh_steam["dry_frac"] = df_loh_steam["Cooling system"].map(loh_dry_frac)
  df_loh_steam["is_cogen"] = df_loh_steam["Technology"].astype(str).str.lower().str.contains("cogen")
  loh_tree = cKDTree(latlon_to_xyz(df_loh_steam["Latitude"].values, df_loh_steam["Longitude"].values))
  return eia_tree, eia_plant, loh_tree, df_loh_steam


def load_gem_power_units() -> tuple[cKDTree, pd.DataFrame]:
  """Load operating units from GEM Coal, Oil & Gas, and Bioenergy trackers."""
  records: list[dict[str, object]] = []

  # 1. GEM Coal
  df_c = pd.read_excel(
      DATA_RAW / "gem/coal_2026-07/Global-Coal-Plant-Tracker-July-2026.xlsx",
      sheet_name="Units",
      usecols=[
          "Country ISO 3", "Status", "Capacity (MW)", "Combustion technology",
          "CHP", "Captive", "Captive residential use", "Latitude", "Longitude",
      ],
  )
  df_c = df_c[df_c["Status"].astype(str).str.lower() == "operating"].copy()
  for _, r in df_c.iterrows():
    lat = pd.to_numeric(r["Latitude"], errors="coerce")
    lon = pd.to_numeric(r["Longitude"], errors="coerce")
    if pd.isna(lat) or pd.isna(lon):
      continue
    ct = str(r["Combustion technology"]).lower()
    if "ultra" in ct:
      spec_key = "coal_usc"
    elif "super" in ct:
      spec_key = "coal_sc"
    elif "igcc" in ct:
      spec_key = "coal_igcc"
    elif "cfb" in ct or "fluid" in ct:
      spec_key = "coal_cfb"
    else:
      spec_key = "coal_subc"
    res_use = str(r["Captive residential use"]).strip().lower()
    cap_ind = pd.notna(r["Captive"]) and str(r["Captive"]).strip().lower() not in ("", "nan", "none", "no")
    chp = str(r["CHP"]).strip().lower() == "yes" or res_use in ("heat", "both")
    is_captive_ind_only = cap_ind and (res_use not in ("heat", "both"))
    records.append({
        "iso3": str(r["Country ISO 3"]).strip(),
        "fuel_group": "coal",
        "cap_mw": float(pd.to_numeric(r["Capacity (MW)"], errors="coerce") or 100.0),
        "spec_key": spec_key,
        "is_chp": chp,
        "is_captive_ind_only": is_captive_ind_only,
        "lat": float(lat),
        "lon": float(lon),
    })

  # 2. GEM Oil & Gas (Gas & Oil Units + Sub-Threshold Units)
  og_path = DATA_RAW / "gem/oil_gas_2026-08/Global Oil and Gas Plant Tracker (GOGPT) - August 2026.xlsx"
  for sname in ("Gas & Oil Units", "Sub-Threshold Units"):
    df_og = pd.read_excel(
        og_path,
        sheet_name=sname,
        usecols=[
            "Fuel", "Capacity (MW)", "Status", "Turbine/Engine Technology",
            "CHP", "Captive industry use", "Captive non-industry use", "Latitude", "Longitude",
        ],
    )
    df_og = df_og[df_og["Status"].astype(str).str.lower() == "operating"].copy()
    for _, r in df_og.iterrows():
      lat = pd.to_numeric(r["Latitude"], errors="coerce")
      lon = pd.to_numeric(r["Longitude"], errors="coerce")
      if pd.isna(lat) or pd.isna(lon):
        continue
      fuel_s = str(r["Fuel"]).lower()
      is_oil = "oil" in fuel_s or "diesel" in fuel_s or "naphtha" in fuel_s or "petrol" in fuel_s
      tt = str(r["Turbine/Engine Technology"]).strip().lower()
      if "combined cycle" in tt or tt in ("iccc", "iscc"):
        spec_key = "gas_ccgt"
      elif "steam" in tt:
        spec_key = "oil_steam" if is_oil else "gas_steam"
      elif "internal combustion" in tt:
        spec_key = "oil_ocgt_ice" if is_oil else "gas_ice"
      elif "gas turbine" in tt:
        spec_key = "oil_ocgt_ice" if is_oil else "gas_ocgt"
      else:
        spec_key = "oil_steam" if is_oil else "gas_ccgt"
      non_ind = str(r["Captive non-industry use"]).strip().lower()
      cap_ind = pd.notna(r["Captive industry use"]) and str(r["Captive industry use"]).strip().lower() not in ("", "nan", "none", "no")
      chp = str(r["CHP"]).strip().lower() == "yes" or non_ind in ("heat", "both")
      is_captive_ind_only = cap_ind and (non_ind not in ("heat", "both"))
      records.append({
          "iso3": "",
          "fuel_group": "oil" if is_oil else "gas",
          "cap_mw": float(pd.to_numeric(r["Capacity (MW)"], errors="coerce") or 50.0),
          "spec_key": spec_key,
          "is_chp": chp,
          "is_captive_ind_only": is_captive_ind_only,
          "lat": float(lat),
          "lon": float(lon),
      })

  # 3. GEM Bioenergy
  df_b = pd.read_excel(
      DATA_RAW / "gem/bioenergy_2026-09_v4/Global Bioenergy Power Tracker (GBPT) - v4.xlsx",
      sheet_name="Data",
      usecols=["Capacity (MW)", "Status", "Latitude", "Longitude"],
  )
  df_b = df_b[df_b["Status"].astype(str).str.lower() == "operating"].copy()
  for _, r in df_b.iterrows():
    lat = pd.to_numeric(r["Latitude"], errors="coerce")
    lon = pd.to_numeric(r["Longitude"], errors="coerce")
    if pd.isna(lat) or pd.isna(lon):
      continue
    records.append({
        "iso3": "",
        "fuel_group": "biomass",
        "cap_mw": float(pd.to_numeric(r["Capacity (MW)"], errors="coerce") or 30.0),
        "spec_key": "biomass_steam",
        "is_chp": False,
        "is_captive_ind_only": False,
        "lat": float(lat),
        "lon": float(lon),
    })

  df_gem = pd.DataFrame(records)
  df_gem["cap_mw"] = df_gem["cap_mw"].clip(lower=1.0)
  gem_tree = cKDTree(latlon_to_xyz(df_gem["lat"].values, df_gem["lon"].values))
  return gem_tree, df_gem


def primary_fuel_from_ct(source_type: str) -> str:
  first = str(source_type).split(",")[0].strip().lower()
  if "coal" in first:
    return "coal"
  if "gas" in first:
    return "gas"
  if "oil" in first or "other_fossil" in first:
    return "oil"
  if "waste" in first:
    return "waste"
  if "biomass" in first:
    return "biomass"
  return "gas"


def build_combustion_power_streams(
    df_ct_pwr: pd.DataFrame,
    gem_tree: cKDTree,
    df_gem: pd.DataFrame,
    eia_tree: cKDTree,
    eia_plant: pd.DataFrame,
    loh_tree: cKDTree,
    df_loh_steam: pd.DataFrame,
) -> tuple[list[dict[str, object]], dict[str, float]]:
  """Build 2025 waste-heat streams for Climate TRACE combustion power plants."""
  df_ct_pwr = df_ct_pwr.copy()
  df_ct_pwr["month"] = pd.to_datetime(df_ct_pwr["start_time"]).dt.month
  piv = df_ct_pwr.pivot_table(index="source_id", columns="month", values="activity", aggfunc="sum", fill_value=0.0)
  for m in range(1, 13):
    if m not in piv.columns:
      piv[m] = 0.0
  piv = piv[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]

  meta = df_ct_pwr.drop_duplicates("source_id").set_index("source_id")
  rad_chord = chord_from_km(MATCH_RADIUS_KM)

  xyz_ct = latlon_to_xyz(meta["lat"].values, meta["lon"].values)
  gem_hits = gem_tree.query_ball_point(xyz_ct, r=rad_chord)
  eia_hits = eia_tree.query_ball_point(xyz_ct, r=rad_chord)
  loh_hits = loh_tree.query_ball_point(xyz_ct, r=rad_chord)

  audit = {
      "ct_power_plants_total": float(len(meta)),
      "ct_power_total_mwh": 0.0,
      "chp_direct_elec_mwh_equiv": 0.0,
      "wet_condensing_elec_mwh_equiv": 0.0,
      "excluded_captive_ind_chp_mwh_equiv": 0.0,
      "excluded_ocgt_ice_mwh": 0.0,
      "excluded_dry_cooling_mwh": 0.0,
      "sub_1mw_excluded_gj": 0.0,
      "raw_total_gj_before_1mw": 0.0,
  }

  streams: list[dict[str, object]] = []
  for idx_pos, (sid, m_row) in enumerate(meta.iterrows()):
    m_mwh = piv.loc[sid].values.astype(np.float64)
    ann_mwh = float(m_mwh.sum())
    if ann_mwh <= 0.0:
      continue
    audit["ct_power_total_mwh"] += ann_mwh

    iso3 = str(m_row["iso3_country"]).strip()
    lat = float(m_row["lat"])
    lon = float(m_row["lon"])
    cap_mw = float(pd.to_numeric(m_row["capacity"], errors="coerce") or 0.0)
    pfuel = primary_fuel_from_ct(str(m_row["source_type"]))
    is_fossil = pfuel in ("coal", "gas", "oil")

    # Determine dry-cooling fraction on steam/CCGT condensers at this site
    dry_frac = 0.0
    loh_cogen = False
    if iso3 == "USA" and eia_hits[idx_pos]:
      dry_frac = float(eia_plant.iloc[eia_hits[idx_pos]]["dry_frac"].mean())
    elif loh_hits[idx_pos]:
      sub_loh = df_loh_steam.iloc[loh_hits[idx_pos]]
      w = sub_loh["cap_mw"].values
      dry_frac = float(np.average(sub_loh["dry_frac"].values, weights=w))
      loh_cogen = bool(np.average(sub_loh["is_cogen"].values.astype(float), weights=w) >= 0.5)

    # Match GEM operating units within 5 km
    g_idxs = gem_hits[idx_pos]
    if g_idxs:
      sub_g = df_gem.iloc[g_idxs]
      # Prefer fuel-compatible GEM units if present at multi-fuel complexes
      compat = sub_g[sub_g["fuel_group"] == pfuel] if pfuel in ("coal", "gas", "oil", "biomass") else sub_g
      if len(compat) == 0:
        compat = sub_g
      weights = compat["cap_mw"].values
      tot_w = float(weights.sum())
      r_dir_mwh_per_mwh = 0.0
      r_hp_mwh_per_mwh = 0.0
      w_chp = 0.0
      w_cap_ind = 0.0
      w_wet = 0.0
      w_ocgt = 0.0
      w_dry = 0.0
      for (_, u), wu_np in zip(compat.iterrows(), weights):
        wu = float(wu_np)
        sp = POWER_TECH_SPECS[str(u["spec_key"])]
        u_chp = bool(u["is_chp"]) or (loh_cogen and pfuel in ("coal", "gas", "biomass", "waste"))
        u_cap_ind = bool(u["is_captive_ind_only"])
        if u_chp and sp.chp_cb is not None:
          if u_cap_ind:
            # Captive industrial-only CHP: extracted process steam is already used on-site by factory
            w_cap_ind += wu
            if sp.recoverable_cooling:
              tot_rej = (1.0 - sp.eta_lhv - sp.k_loss_lhv) / sp.eta_lhv
              d_part = min(1.0 / sp.chp_cb, tot_rej)
              rem_part = max(0.0, tot_rej - d_part)
              r_hp_mwh_per_mwh += (wu / tot_w) * rem_part * (1.0 - dry_frac)
          else:
            w_chp += wu
            if sp.recoverable_cooling:
              tot_rej = (1.0 - sp.eta_lhv - sp.k_loss_lhv) / sp.eta_lhv
              d_part = min(1.0 / sp.chp_cb, tot_rej)
              rem_part = max(0.0, tot_rej - d_part)
              r_dir_mwh_per_mwh += (wu / tot_w) * d_part
              r_hp_mwh_per_mwh += (wu / tot_w) * rem_part * (1.0 - dry_frac)
            else:
              # OCGT / ICE with public/DH CHP exhaust heat recovery boiler (DEA Sheet 04/06)
              d_part = min(1.0 / sp.chp_cb, max(0.0, (1.0 - sp.eta_lhv - 0.12) / sp.eta_lhv))
              r_dir_mwh_per_mwh += (wu / tot_w) * d_part
        else:
          if sp.recoverable_cooling:
            tot_rej = (1.0 - sp.eta_lhv - sp.k_loss_lhv) / sp.eta_lhv
            r_hp_mwh_per_mwh += (wu / tot_w) * tot_rej * (1.0 - dry_frac)
            w_wet += wu * (1.0 - dry_frac)
            w_dry += wu * dry_frac
          else:
            w_ocgt += wu
      audit["chp_direct_elec_mwh_equiv"] += float(ann_mwh * (w_chp / tot_w))
      audit["excluded_captive_ind_chp_mwh_equiv"] += float(ann_mwh * (w_cap_ind / tot_w))
      audit["wet_condensing_elec_mwh_equiv"] += float(ann_mwh * (w_wet / tot_w))
      audit["excluded_ocgt_ice_mwh"] += float(ann_mwh * (w_ocgt / tot_w))
      audit["excluded_dry_cooling_mwh"] += float(ann_mwh * (w_dry / tot_w))
    else:
      # Fallback when no GEM unit within 5 km
      cf = (ann_mwh / (cap_mw * 8760.0)) if cap_mw > 0 else 0.4
      if pfuel == "coal":
        sp_key = "coal_subc"
      elif pfuel == "waste":
        sp_key = "wte_steam"
      elif pfuel == "biomass":
        sp_key = "biomass_steam"
      elif pfuel == "oil":
        sp_key = "oil_ocgt_ice" if (cap_mw < 50.0 or cf < 0.15) else "oil_steam"
      else:
        sp_key = "gas_ocgt" if (cap_mw < 50.0 or cf < 0.15) else "gas_ccgt"
      sp = POWER_TECH_SPECS[sp_key]
      if loh_cogen and sp.chp_cb is not None:
        audit["chp_direct_elec_mwh_equiv"] += ann_mwh
        if sp.recoverable_cooling:
          tot_rej = (1.0 - sp.eta_lhv - sp.k_loss_lhv) / sp.eta_lhv
          r_dir_mwh_per_mwh = min(1.0 / sp.chp_cb, tot_rej)
          r_hp_mwh_per_mwh = max(0.0, tot_rej - r_dir_mwh_per_mwh) * (1.0 - dry_frac)
        else:
          r_dir_mwh_per_mwh = min(1.0 / sp.chp_cb, max(0.0, (1.0 - sp.eta_lhv - 0.12) / sp.eta_lhv))
          r_hp_mwh_per_mwh = 0.0
      elif sp.recoverable_cooling:
        tot_rej = (1.0 - sp.eta_lhv - sp.k_loss_lhv) / sp.eta_lhv
        r_dir_mwh_per_mwh = 0.0
        r_hp_mwh_per_mwh = tot_rej * (1.0 - dry_frac)
        audit["wet_condensing_elec_mwh_equiv"] += ann_mwh * (1.0 - dry_frac)
        audit["excluded_dry_cooling_mwh"] += ann_mwh * dry_frac
      else:
        r_dir_mwh_per_mwh = 0.0
        r_hp_mwh_per_mwh = 0.0
        audit["excluded_ocgt_ice_mwh"] += ann_mwh

    m_gj_dir = m_mwh * r_dir_mwh_per_mwh * GJ_PER_MWH
    m_gj_hp = m_mwh * r_hp_mwh_per_mwh * GJ_PER_MWH
    tot_fac_gj = float(m_gj_dir.sum() + m_gj_hp.sum())
    audit["raw_total_gj_before_1mw"] += tot_fac_gj
    if tot_fac_gj < MIN_ANNUAL_GJ:
      audit["sub_1mw_excluded_gj"] += tot_fac_gj
      continue

    fac_id = f"PWR_CT_{sid}"
    fac_name = str(m_row["source_name"])
    for suffix, up_mode, m_gj in (("CHP", "direct_hex", m_gj_dir), ("COND", "hp_ind_power", m_gj_hp)):
      ann_gj = float(m_gj.sum())
      if ann_gj <= 0.0:
        continue
      daily_gj = m_gj / DAYS_2025
      rec: dict[str, object] = {
          "stream_id": f"{fac_id}_{suffix}",
          "facility_id": fac_id,
          "facility_name": fac_name,
          "iso3": iso3,
          "lat": lat,
          "lon": lon,
          "sector": "power",
          "subsector_detail": f"power_{pfuel}",
          "is_fossil_power": is_fossil,
          "upgrade_mode": up_mode,
          "annual_activity": ann_mwh,
          "activity_unit": "MWh_net",
          "q3g_annual_gj": ann_gj,
          "q3g_mean_mw": ann_gj / (365.0 * GJ_PER_MW_DAY),
          "q4g_annual_gj": ann_gj,
          "q4g_mean_mw": ann_gj / (365.0 * GJ_PER_MW_DAY),
      }
      for m_i in range(12):
        rec[f"q3g_m{m_i+1:02d}_gj_d"] = float(daily_gj[m_i])
        rec[f"q4g_m{m_i+1:02d}_gj_d"] = float(daily_gj[m_i])
      streams.append(rec)

  return streams, audit


def build_nuclear_streams() -> tuple[list[dict[str, object]], dict[str, object]]:
  """Build 2025 nuclear waste-heat streams from IAEA PRIS opex2026 + GEM_GNPT coordinates."""
  # Map country name -> ISO3 via GEM_GNPT + explicit nuclear ISO3 dictionary
  NUC_ISO3 = {
      "argentina": "ARG", "armenia": "ARM", "bangladesh": "BGD", "belarus": "BLR",
      "belgium": "BEL", "brazil": "BRA", "bulgaria": "BGR", "canada": "CAN",
      "china": "CHN", "czechrepublic": "CZE", "czechia": "CZE", "egypt": "EGY",
      "finland": "FIN", "france": "FRA", "germany": "DEU", "hungary": "HUN",
      "india": "IND", "iran": "IRN", "iranislamicrepublicof": "IRN", "japan": "JPN",
      "korea": "KOR", "korearepublicof": "KOR", "southkorea": "KOR", "mexico": "MEX",
      "netherlands": "NLD", "pakistan": "PAK", "romania": "ROU", "russia": "RUS",
      "russianfederation": "RUS", "slovakia": "SVK", "slovenia": "SVN",
      "southafrica": "ZAF", "spain": "ESP", "sweden": "SWE", "switzerland": "CHE",
      "taiwan": "TWN", "taiwanchina": "TWN", "turkey": "TUR", "turkiye": "TUR",
      "ukraine": "UKR", "unitedarabemirates": "ARE", "unitedkingdom": "GBR",
      "unitedstates": "USA", "unitedstatesofamerica": "USA",
  }
  df_gn = pd.read_excel(
      DATA_RAW / "current_plan_2026-11-15/sources/gem_nuclear_coordinates_2025_09/GEM_GNPT.xlsx",
      sheet_name="Data",
  )
  df_gn["iso3"] = df_gn["Country/Area"].map(lambda x: NUC_ISO3.get(norm_name(x), ""))
  df_gn["Latitude"] = pd.to_numeric(df_gn["Latitude"], errors="coerce")
  df_gn["Longitude"] = pd.to_numeric(df_gn["Longitude"], errors="coerce")
  df_gn = df_gn.dropna(subset=["Latitude", "Longitude"]).copy()

  # Build lookup keys per ISO3
  gn_by_iso: dict[str, list[dict[str, object]]] = {}
  for _, r in df_gn.iterrows():
    iso = str(r["iso3"])
    proj = str(r["Project Name"]).replace("nuclear power plant", "").replace("Nuclear Power Plant", "").strip()
    unit = str(r["Unit Name"]).strip()
    u_suf = unit if unit not in ("--", "nan", "") else ""
    full1 = norm_name(f"{proj}{u_suf}")
    aliases = [
        norm_name(f"{al.strip()}{u_suf}")
        for al in re.split(r"[;,/]", str(r["Other Name(s)"]))
        if al.strip() and al.strip().lower() != "nan"
    ]
    gn_by_iso.setdefault(iso, []).append({
        "full1": full1,
        "aliases": aliases,
        "proj_norm": norm_name(proj),
        "unit_norm": norm_name(u_suf),
        "lat": float(r["Latitude"]),
        "lon": float(r["Longitude"]),
        "ref_mw": pd.to_numeric(r["Reference Net Capacity (MW)"], errors="coerce"),
        "th_mw": pd.to_numeric(r["Thermal Capacity (MWt)"], errors="coerce"),
    })

  months_map = {"Jan": 0, "Feb": 1, "Mar": 2, "Apr": 3, "May": 4, "Jun": 5,
                "Jul": 6, "Aug": 7, "Sep": 8, "Oct": 9, "Nov": 10, "Dec": 11}

  streams: list[dict[str, object]] = []
  audit: dict[str, object] = {
      "pris_active_reactors_2025": 0.0,
      "pris_total_net_mwh_2025": 0.0,
      "pris_coord_matched": 0.0,
      "pris_coord_unmatched": 0.0,
      "unmatched_reactors": [],
      "raw_total_gj_before_1mw": 0.0,
      "sub_1mw_excluded_gj": 0.0,
  }

  for pf in sorted(glob.glob(str(DATA_RAW / "current_plan_2026-11-15/sources/pris/opex2026/*.xlsx"))):
    cname_raw = Path(pf).stem.replace("2025_", "")
    iso3 = NUC_ISO3.get(norm_name(cname_raw), "")
    wb = openpyxl.load_workbook(pf, read_only=True, data_only=True)
    for sn in wb.sheetnames:
      if sn.lower().startswith("document"):
        continue
      ws = wb[sn]
      rows = list(ws.iter_rows(max_row=72, values_only=True))
      m_gwh = np.zeros(12, dtype=np.float64)
      design_net_mw = np.nan
      ref_net_mw = np.nan
      th_mw = np.nan
      cooling = "unknown"
      non_elec = "none"
      for r in rows:
        if len(r) > 12 and r[2] == "Design Net Capacity":
          design_net_mw = pd.to_numeric(r[12], errors="coerce")
        elif len(r) > 12 and r[2] == "Reference Unit Power (Net Capacity)":
          ref_net_mw = pd.to_numeric(r[12], errors="coerce")
        elif len(r) > 12 and r[2] == "Thermal Capacity":
          th_mw = pd.to_numeric(r[12], errors="coerce")
        for idx_c, val in enumerate(r):
          if val == "Primary means of condenser cooling":
            after = [x for x in r[idx_c + 1:] if x is not None and str(x).strip() != ""]
            if after:
              cooling = str(after[0]).strip()
          elif val == "Non-electrical applications":
            after = [x for x in r[idx_c + 1:] if x is not None and str(x).strip() != ""]
            if after:
              non_elec = str(after[0]).strip()
        if len(r) > 8 and r[2] == 2025 and str(r[4]).strip() in months_map:
          val_gwh = pd.to_numeric(r[8], errors="coerce")
          if pd.notna(val_gwh) and val_gwh > 0:
            m_gwh[months_map[str(r[4]).strip()]] += float(val_gwh)

      m_mwh = m_gwh * 1000.0
      ann_mwh = float(m_mwh.sum())
      if ann_mwh <= 0.0:
        continue
      audit["pris_active_reactors_2025"] += 1.0
      audit["pris_total_net_mwh_2025"] += ann_mwh

      # Match coordinates in GEM_GNPT (handle PRIS listing Taiwan reactors inside 2025_China.xlsx)
      r_iso3 = "TWN" if "taiwan" in sn.lower() else iso3
      sn_clean = re.sub(r"\([^)]*\)", "", sn).strip()
      sn_norm = norm_name(sn_clean)
      m_num = re.search(r"(\d+)$", sn_clean)
      u_num = m_num.group(1) if m_num else ""
      base_norm = norm_name(re.sub(r"[-_ ]*\d+$", "", sn_clean))
      cands = gn_by_iso.get(r_iso3, [])
      best = None
      for c in cands:
        if sn_norm == c["full1"] or sn_norm in c["aliases"]:  # type: ignore[operator]
          best = c
          break
      if best is None:
        for c in cands:
          p_n = str(c["proj_norm"])
          al_hit = any(base_norm and base_norm in al for al in c["aliases"])  # type: ignore[union-attr]
          if (base_norm and (base_norm in p_n or p_n in base_norm or al_hit)) and (not u_num or u_num == c["unit_norm"]):
            best = c
            break
      if best is None:
        for c in cands:
          p_n = str(c["proj_norm"])
          if base_norm and (base_norm[:5] in p_n or p_n[:5] in base_norm):
            best = c
            break
      if best is None and cands:
        best = cands[0]
        audit["pris_coord_unmatched"] = float(audit["pris_coord_unmatched"]) + 1.0
        audit["unmatched_reactors"].append(f"{r_iso3}:{sn.strip()}")  # type: ignore[union-attr]
      elif best is not None:
        audit["pris_coord_matched"] = float(audit["pris_coord_matched"]) + 1.0
      else:
        audit["pris_coord_unmatched"] = float(audit["pris_coord_unmatched"]) + 1.0
        audit["unmatched_reactors"].append(f"{r_iso3}:{sn.strip()}")  # type: ignore[union-attr]
        continue

      net_cap = ref_net_mw if pd.notna(ref_net_mw) and ref_net_mw > 0 else design_net_mw
      if (pd.isna(th_mw) or th_mw <= 0) and pd.notna(best["th_mw"]):
        th_mw = best["th_mw"]
      if pd.notna(net_cap) and pd.notna(th_mw) and th_mw > net_cap > 0:
        eta_u = float(np.clip(net_cap / th_mw, 0.25, 0.40))
      else:
        eta_u = POWER_TECH_SPECS["nuclear_lwr"].eta_lhv
      k_loss = POWER_TECH_SPECS["nuclear_lwr"].k_loss_lhv
      rej_per_mwh = max(0.0, (1.0 - eta_u - k_loss) / eta_u)

      m_gj = m_mwh * rej_per_mwh * GJ_PER_MWH
      ann_gj = float(m_gj.sum())
      audit["raw_total_gj_before_1mw"] += ann_gj
      if ann_gj < MIN_ANNUAL_GJ:
        audit["sub_1mw_excluded_gj"] += ann_gj
        continue

      daily_gj = m_gj / DAYS_2025
      fac_id = f"PWR_PRIS_{r_iso3}_{re.sub(r'[^A-Za-z0-9]+', '_', sn_clean)}"
      rec: dict[str, object] = {
          "stream_id": f"{fac_id}_COND",
          "facility_id": fac_id,
          "facility_name": f"{sn.strip()} ({cooling})",
          "iso3": r_iso3,
          "lat": float(best["lat"]),
          "lon": float(best["lon"]),
          "sector": "power",
          "subsector_detail": "power_nuclear",
          "is_fossil_power": False,
          "upgrade_mode": "hp_ind_power",
          "annual_activity": ann_mwh,
          "activity_unit": "MWh_net",
          "q3g_annual_gj": ann_gj,
          "q3g_mean_mw": ann_gj / (365.0 * GJ_PER_MW_DAY),
          "q4g_annual_gj": ann_gj,
          "q4g_mean_mw": ann_gj / (365.0 * GJ_PER_MW_DAY),
      }
      for m_i in range(12):
        rec[f"q3g_m{m_i+1:02d}_gj_d"] = float(daily_gj[m_i])
        rec[f"q4g_m{m_i+1:02d}_gj_d"] = float(daily_gj[m_i])
      streams.append(rec)
    wb.close()

  return streams, audit


def build_industrial_streams(df_ct_ind: pd.DataFrame) -> tuple[list[dict[str, object]], dict[str, dict[str, float]]]:
  """Build 2025 high-T (direct HEX) and low-T (scrubber+HP) streams for 5 industries."""
  # Load GEM Cement for clinker/cement ratio & grinding plant check
  df_cem_gem = pd.read_excel(
      DATA_RAW / "gem/cement_2026-07_V1/Plant-level data - Global Cement and Concrete Tracker - July 2026 - Standard Copy V1.xlsx",
      sheet_name="Final data",
      usecols=[
          "Coordinates", "Plant type", "Clinker production method",
          "Cement capacity (million metric tonnes per annum)",
          "Clinker capacity (million metric tonnes per annum)",
      ],
  )
  c_lat, c_lon = parse_coordinates_col(df_cem_gem["Coordinates"])
  df_cem_gem["lat"] = c_lat
  df_cem_gem["lon"] = c_lon
  df_cem_gem = df_cem_gem.dropna(subset=["lat", "lon"]).reset_index(drop=True)
  cem_tree = cKDTree(latlon_to_xyz(df_cem_gem["lat"].values, df_cem_gem["lon"].values))

  # Load GEM Steel for BF-BOF vs EAF capacity shares
  st_path = DATA_RAW / "gem/iron_steel_2026-09_V1/Plant-level data - Global Iron and Steel Tracker - September 2026 (V1).xlsx"
  df_st_loc = pd.read_excel(st_path, sheet_name="Plant data", usecols=["GEM plant ID", "Coordinates"])
  df_st_cap = pd.read_excel(
      st_path,
      sheet_name="Plant capacities and status",
      usecols=["GEM plant ID", "Status", "Nominal BOF steel capacity (ttpa)", "Nominal EAF steel capacity (ttpa)", "Nominal IF steel capacity (ttpa)"],
  )
  df_st_cap = df_st_cap[df_st_cap["Status"].astype(str).str.lower() == "operating"].copy()
  for col in ["Nominal BOF steel capacity (ttpa)", "Nominal EAF steel capacity (ttpa)", "Nominal IF steel capacity (ttpa)"]:
    df_st_cap[col] = pd.to_numeric(df_st_cap[col], errors="coerce").fillna(0.0)
  st_agg = df_st_cap.groupby("GEM plant ID", as_index=False)[
      ["Nominal BOF steel capacity (ttpa)", "Nominal EAF steel capacity (ttpa)", "Nominal IF steel capacity (ttpa)"]
  ].sum()
  df_st = df_st_loc.merge(st_agg, on="GEM plant ID", how="inner")
  s_lat, s_lon = parse_coordinates_col(df_st["Coordinates"])
  df_st["lat"], df_st["lon"] = s_lat, s_lon
  df_st = df_st.dropna(subset=["lat", "lon"]).reset_index(drop=True)
  st_tree = cKDTree(latlon_to_xyz(df_st["lat"].values, df_st["lon"].values))

  # Load SFI Pulp & Paper for pulp vs paper capacity split
  df_pp_sfi = pd.read_excel(
      DATA_RAW / "sfi_geoasset/pulp_paper_2024-10/SFI_ALD_Pulp_Paper_Database_October_2024__1_.xlsx",
      sheet_name="SFI_Pulp_Paper_Database",
      usecols=["latitude", "longitude", "plant_type", "products", "capacity_pulp (tonnes per year) ", "capacity_paper (tonnes per year)"],
  )
  df_pp_sfi["latitude"] = pd.to_numeric(df_pp_sfi["latitude"], errors="coerce")
  df_pp_sfi["longitude"] = pd.to_numeric(df_pp_sfi["longitude"], errors="coerce")
  df_pp_sfi = df_pp_sfi.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)
  pp_tree = cKDTree(latlon_to_xyz(df_pp_sfi["latitude"].values, df_pp_sfi["longitude"].values))

  rad_chord = chord_from_km(MATCH_RADIUS_KM)
  df_ct_ind = df_ct_ind.copy()
  df_ct_ind["month"] = pd.to_datetime(df_ct_ind["start_time"]).dt.month

  streams: list[dict[str, object]] = []
  ind_audit: dict[str, dict[str, float]] = {}

  sub_map = {
      "cement": "cement",
      "iron-and-steel": "steel",
      "oil-and-gas-refining": "refining",
      "petrochemical-steam-cracking": "ethylene",
      "pulp-and-paper": "pulp_paper",
  }

  for ct_sub, sec_name in sub_map.items():
    sub_df = df_ct_ind[df_ct_ind["subsector"] == ct_sub]
    piv = sub_df.pivot_table(index="source_id", columns="month", values="activity", aggfunc="sum", fill_value=0.0)
    for m in range(1, 13):
      if m not in piv.columns:
        piv[m] = 0.0
    piv = piv[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]
    meta = sub_df.drop_duplicates("source_id").set_index("source_id")
    # Mean clinker ratio per cement plant from CT other9
    if ct_sub == "cement":
      clk_series = sub_df.assign(o9=pd.to_numeric(sub_df["other9"], errors="coerce")).groupby("source_id")["o9"].mean()
    else:
      clk_series = pd.Series(dtype=float)

    xyz = latlon_to_xyz(meta["lat"].values, meta["lon"].values)
    sec_stats = {
        "n_facilities_raw": float(len(meta)),
        "annual_physical_activity": 0.0,
        "raw_total_gj_to_25c": 0.0,
        "sub_1mw_excluded_gj": 0.0,
        "kept_3gdh_direct_gj": 0.0,
        "kept_3gdh_hp_gj": 0.0,
        "kept_4gdh_direct_gj": 0.0,
        "kept_4gdh_hp_gj": 0.0,
    }

    for idx_pos, (sid, m_row) in enumerate(meta.iterrows()):
      m_act = piv.loc[sid].values.astype(np.float64)
      if m_act.sum() <= 0.0:
        continue
      iso3 = str(m_row["iso3_country"]).strip()
      lat = float(m_row["lat"])
      lon = float(m_row["lon"])
      stype = str(m_row["source_type"]).strip().lower()
      pt_xyz = xyz[idx_pos]

      if ct_sub == "cement":
        ratio = float(clk_series.get(sid, np.nan))
        if math.isnan(ratio) or ratio <= 0.0 or ratio > 1.0:
          ratio = 0.80
        hits = cem_tree.query_ball_point(pt_xyz, r=rad_chord)
        is_grinding = False
        if hits:
          sub_g = df_cem_gem.iloc[hits]
          if (sub_g["Plant type"].astype(str).str.lower() == "grinding").all():
            is_grinding = True
          c_cap = pd.to_numeric(sub_g["Cement capacity (million metric tonnes per annum)"], errors="coerce").sum()
          k_cap = pd.to_numeric(sub_g["Clinker capacity (million metric tonnes per annum)"], errors="coerce").sum()
          if c_cap > 0 and k_cap > 0 and math.isnan(float(clk_series.get(sid, np.nan))):
            ratio = float(np.clip(k_cap / c_cap, 0.40, 0.98))
        if is_grinding:
          continue
        m_phys = m_act * ratio  # tonnes of clinker
        act_unit = "t_clinker"
        fac = SEENERGIES_FACTORS["cement_wet" if "wet" in stype else "cement_dry"]
        q25, q55, q95 = fac.q_to_25c_gj_per_t, fac.q_to_55c_gj_per_t, fac.q_to_95c_gj_per_t
        detail = "cement_wet" if "wet" in stype else "cement_dry"

      elif ct_sub == "iron-and-steel":
        m_phys = m_act  # tonnes of crude steel
        act_unit = "t_steel"
        has_bof = "bof" in stype
        has_eaf = "eaf" in stype or "if" in stype
        if has_bof and not has_eaf:
          bof_share = 1.0
        elif has_eaf and not has_bof:
          bof_share = 0.0
        else:
          hits = st_tree.query_ball_point(pt_xyz, r=rad_chord)
          if hits:
            sub_s = df_st.iloc[hits]
            b_c = float(sub_s["Nominal BOF steel capacity (ttpa)"].sum())
            e_c = float(sub_s["Nominal EAF steel capacity (ttpa)"].sum() + sub_s["Nominal IF steel capacity (ttpa)"].sum())
            bof_share = b_c / (b_c + e_c) if (b_c + e_c) > 0 else 0.5
          else:
            bof_share = 0.5
        f_b = SEENERGIES_FACTORS["steel_bf_bof"]
        f_e = SEENERGIES_FACTORS["steel_eaf"]
        q25 = bof_share * f_b.q_to_25c_gj_per_t + (1.0 - bof_share) * f_e.q_to_25c_gj_per_t
        q55 = bof_share * f_b.q_to_55c_gj_per_t + (1.0 - bof_share) * f_e.q_to_55c_gj_per_t
        q95 = bof_share * f_b.q_to_95c_gj_per_t + (1.0 - bof_share) * f_e.q_to_95c_gj_per_t
        detail = "steel_bf_bof" if bof_share >= 0.5 else "steel_eaf"

      elif ct_sub == "oil-and-gas-refining":
        m_phys = m_act * CRUDE_TONNES_PER_BBL  # tonnes of crude oil
        act_unit = "t_crude"
        if "deep" in stype:
          detail = "refining_deep"
        elif "hydroskimming" in stype:
          detail = "refining_hydroskimming"
        else:
          detail = "refining_medium"
        fac = SEENERGIES_FACTORS[detail]
        q25, q55, q95 = fac.q_to_25c_gj_per_t, fac.q_to_55c_gj_per_t, fac.q_to_95c_gj_per_t

      elif ct_sub == "petrochemical-steam-cracking":
        m_phys = m_act  # tonnes of ethylene
        act_unit = "t_ethylene"
        detail = "ethylene_cracker"
        fac = SEENERGIES_FACTORS[detail]
        q25, q55, q95 = fac.q_to_25c_gj_per_t, fac.q_to_55c_gj_per_t, fac.q_to_95c_gj_per_t

      else:  # pulp-and-paper
        m_phys = m_act  # tonnes of pulp & paper
        act_unit = "t_pulp_paper"
        hits = pp_tree.query_ball_point(pt_xyz, r=rad_chord)
        pulp_share = 0.5
        if hits:
          sub_p = df_pp_sfi.iloc[hits]
          cp = pd.to_numeric(sub_p["capacity_pulp (tonnes per year) "], errors="coerce").fillna(0.0).sum()
          cpa = pd.to_numeric(sub_p["capacity_paper (tonnes per year)"], errors="coerce").fillna(0.0).sum()
          if (cp + cpa) > 0:
            pulp_share = float(cp / (cp + cpa))
        if "recycled" in stype and "kraft" not in stype and "chemical" not in stype:
          # Recycled fiber mill: 0 pulping excess heat, paper drying [0.40, 0.10, 0.10] on paper output
          paper_frac = max(1.0 - pulp_share, 0.85) if hits else 1.0
          fp = SEENERGIES_FACTORS["paper_board_packaging"]
          q25, q55, q95 = paper_frac * fp.q_to_25c_gj_per_t, paper_frac * fp.q_to_55c_gj_per_t, paper_frac * fp.q_to_95c_gj_per_t
          detail = "paper_recycled"
        elif "mechanical" in stype and "chemical" not in stype:
          fm = SEENERGIES_FACTORS["pulp_mechanical"]
          fp = SEENERGIES_FACTORS["paper_graphic_tissue"]
          q25 = pulp_share * fm.q_to_25c_gj_per_t + (1.0 - pulp_share) * fp.q_to_25c_gj_per_t
          q55 = pulp_share * fm.q_to_55c_gj_per_t + (1.0 - pulp_share) * fp.q_to_55c_gj_per_t
          q95 = pulp_share * fm.q_to_95c_gj_per_t + (1.0 - pulp_share) * fp.q_to_95c_gj_per_t
          detail = "pulp_mechanical"
        else:
          fc = SEENERGIES_FACTORS["pulp_chemical"]
          fp = SEENERGIES_FACTORS["paper_board_packaging"]
          q25 = pulp_share * fc.q_to_25c_gj_per_t + (1.0 - pulp_share) * fp.q_to_25c_gj_per_t
          q55 = pulp_share * fc.q_to_55c_gj_per_t + (1.0 - pulp_share) * fp.q_to_55c_gj_per_t
          q95 = pulp_share * fc.q_to_95c_gj_per_t + (1.0 - pulp_share) * fp.q_to_95c_gj_per_t
          detail = "pulp_chemical"

      ann_phys = float(m_phys.sum())
      sec_stats["annual_physical_activity"] += ann_phys
      m_gj_25 = m_phys * q25
      tot_gj_25 = float(m_gj_25.sum())
      sec_stats["raw_total_gj_to_25c"] += tot_gj_25
      if tot_gj_25 < MIN_ANNUAL_GJ:
        sec_stats["sub_1mw_excluded_gj"] += tot_gj_25
        continue

      # High-temperature direct HEX stream: >=95 C in 3GDH, >=55 C in 4GDH
      m_dir_3g = m_phys * q95
      m_dir_4g = m_phys * q55
      # Low-temperature scrubber + HP stream: 95->25 C in 3GDH, 55->25 C in 4GDH
      m_hp_3g = m_phys * max(0.0, q25 - q95)
      m_hp_4g = m_phys * max(0.0, q25 - q55)

      sec_stats["kept_3gdh_direct_gj"] += float(m_dir_3g.sum())
      sec_stats["kept_3gdh_hp_gj"] += float(m_hp_3g.sum())
      sec_stats["kept_4gdh_direct_gj"] += float(m_dir_4g.sum())
      sec_stats["kept_4gdh_hp_gj"] += float(m_hp_4g.sum())

      fac_id = f"IND_{sec_name.upper()}_{sid}"
      fac_name = str(m_row["source_name"])
      for suffix, up_mode, g3, g4 in (
          ("DIRECT", "direct_hex", m_dir_3g, m_dir_4g),
          ("HP", "hp_ind_power", m_hp_3g, m_hp_4g),
      ):
        a3, a4 = float(g3.sum()), float(g4.sum())
        if a3 <= 0.0 and a4 <= 0.0:
          continue
        d3, d4 = g3 / DAYS_2025, g4 / DAYS_2025
        rec = {
            "stream_id": f"{fac_id}_{suffix}",
            "facility_id": fac_id,
            "facility_name": fac_name,
            "iso3": iso3,
            "lat": lat,
            "lon": lon,
            "sector": sec_name,
            "subsector_detail": detail,
            "is_fossil_power": False,
            "upgrade_mode": up_mode,
            "annual_activity": ann_phys,
            "activity_unit": act_unit,
            "q3g_annual_gj": a3,
            "q3g_mean_mw": a3 / (365.0 * GJ_PER_MW_DAY),
            "q4g_annual_gj": a4,
            "q4g_mean_mw": a4 / (365.0 * GJ_PER_MW_DAY),
        }
        for m_i in range(12):
          rec[f"q3g_m{m_i+1:02d}_gj_d"] = float(d3[m_i])
          rec[f"q4g_m{m_i+1:02d}_gj_d"] = float(d4[m_i])
        streams.append(rec)

    ind_audit[sec_name] = sec_stats

  return streams, ind_audit


def build_wastewater_streams() -> tuple[list[dict[str, object]], dict[str, float]]:
  """Build 2025 municipal wastewater streams from HydroWASTE v1.0 (deltaT = 5 K)."""
  df_hw = pd.read_csv(
      DATA_RAW / "hydrowaste/2026-10-25/HydroWASTE_v10.csv",
      usecols=["WASTE_ID", "WWTP_NAME", "CNTRY_ISO", "LAT_WWTP", "LON_WWTP", "STATUS", "WASTE_DIS"],
  )
  active_statuses = {"Operational", "Construction Completed", "Not Reported"}
  df_hw = df_hw[df_hw["STATUS"].isin(active_statuses)].copy()
  df_hw["WASTE_DIS"] = pd.to_numeric(df_hw["WASTE_DIS"], errors="coerce").fillna(0.0)
  df_hw["LAT_WWTP"] = pd.to_numeric(df_hw["LAT_WWTP"], errors="coerce")
  df_hw["LON_WWTP"] = pd.to_numeric(df_hw["LON_WWTP"], errors="coerce")
  df_hw = df_hw[(df_hw["WASTE_DIS"] > 0.0) & df_hw["LAT_WWTP"].notna() & df_hw["LON_WWTP"].notna()].copy()

  # GJ/day = (4.186 MJ/(m3*K) * 5.0 K / 1000 MJ/GJ) * WASTE_DIS [m3/d] = 0.02093 * WASTE_DIS
  gj_per_m3 = WATER_VOLUMETRIC_HEAT_CAPACITY_MJ_M3_K * WASTEWATER_DELTA_T_K / 1000.0
  df_hw["daily_gj"] = df_hw["WASTE_DIS"] * gj_per_m3
  df_hw["annual_gj"] = df_hw["daily_gj"] * 365.0

  raw_tot_gj = float(df_hw["annual_gj"].sum())
  sub1_mask = df_hw["annual_gj"] < MIN_ANNUAL_GJ
  sub1_gj = float(df_hw.loc[sub1_mask, "annual_gj"].sum())
  kept = df_hw[~sub1_mask].copy()

  audit = {
      "active_wwtp_count": float(len(df_hw)),
      "kept_ge_1mw_wwtp_count": float(len(kept)),
      "raw_total_gj_before_1mw": raw_tot_gj,
      "sub_1mw_excluded_gj": sub1_gj,
      "kept_annual_gj": float(kept["annual_gj"].sum()),
  }

  # Vectorized dictionary construction for ~15k-20k >=1 MW WWTPs
  streams: list[dict[str, object]] = []
  for row in kept.itertuples(index=False):
    wid = int(row.WASTE_ID)
    dgj = float(row.daily_gj)
    agj = float(row.annual_gj)
    mw = agj / (365.0 * GJ_PER_MW_DAY)
    fac_id = f"WW_HW_{wid}"
    rec: dict[str, object] = {
        "stream_id": f"{fac_id}_HP",
        "facility_id": fac_id,
        "facility_name": str(row.WWTP_NAME) if pd.notna(row.WWTP_NAME) else fac_id,
        "iso3": str(row.CNTRY_ISO).strip(),
        "lat": float(row.LAT_WWTP),
        "lon": float(row.LON_WWTP),
        "sector": "wastewater",
        "subsector_detail": "municipal_wwtp",
        "is_fossil_power": False,
        "upgrade_mode": "hp_wastewater",
        "annual_activity": float(row.WASTE_DIS) * 365.0,
        "activity_unit": "m3_wastewater",
        "q3g_annual_gj": agj,
        "q3g_mean_mw": mw,
        "q4g_annual_gj": agj,
        "q4g_mean_mw": mw,
    }
    for m_i in range(12):
      rec[f"q3g_m{m_i+1:02d}_gj_d"] = dgj
      rec[f"q4g_m{m_i+1:02d}_gj_d"] = dgj
    streams.append(rec)

  return streams, audit


def main() -> None:
  out_dir = DATA_PROCESSED / "sources_2025"
  out_dir.mkdir(parents=True, exist_ok=True)

  print("1. Loading cooling inventories (EIA-860 2025 + Lohrmann 2019 Nature Energy)...", flush=True)
  eia_tree, eia_plant, loh_tree, df_loh_steam = load_cooling_inventories()
  print(f"   EIA-860 plants={len(eia_plant)}, Lohrmann steam/CCGT units={len(df_loh_steam)}", flush=True)

  print("2. Loading GEM operating power units (Coal, Oil & Gas, Bioenergy)...", flush=True)
  gem_tree, df_gem = load_gem_power_units()
  print(f"   GEM operating thermal units={len(df_gem)}", flush=True)

  print("3. Reading Climate TRACE 2025 monthly source dataset...", flush=True)
  zpath = DATA_RAW / "current_plan_2026-11-15/sources/climate_trace_bulk_2025/all_six_source_sectors_2025_monthly_co2_proxy.zip"
  with zipfile.ZipFile(zpath) as zf:
    df_ct = pd.read_csv(
        zf.open("DATA/emissions_co2_v5_12_0.csv"),
        usecols=[
            "source_id", "source_name", "source_type", "iso3_country",
            "sector", "subsector", "start_time", "lat", "lon",
            "activity", "activity_units", "capacity", "other9",
        ],
        low_memory=False,
    )
  df_ct = df_ct[df_ct["start_time"].astype(str).str.startswith("2025")].copy()
  print(f"   Climate TRACE 2025 monthly rows={len(df_ct)}", flush=True)

  print("4. Building combustion power waste-heat streams...", flush=True)
  pwr_streams, pwr_audit = build_combustion_power_streams(
      df_ct[df_ct["subsector"] == "electricity-generation"],
      gem_tree, df_gem, eia_tree, eia_plant, loh_tree, df_loh_steam,
  )
  print(f"   Combustion power streams={len(pwr_streams)}, audit={pwr_audit}", flush=True)

  print("5. Building IAEA PRIS 2025 nuclear waste-heat streams...", flush=True)
  nuc_streams, nuc_audit = build_nuclear_streams()
  print(f"   Nuclear streams={len(nuc_streams)}, audit={nuc_audit}", flush=True)

  print("6. Building five energy-intensive industrial streams (sEEnergies D5.1)...", flush=True)
  ind_streams, ind_audit = build_industrial_streams(df_ct[df_ct["subsector"] != "electricity-generation"])
  print(f"   Industrial streams={len(ind_streams)}", flush=True)

  print("7. Building HydroWASTE v1.0 municipal wastewater streams (deltaT = 5 K)...", flush=True)
  ww_streams, ww_audit = build_wastewater_streams()
  print(f"   Wastewater streams={len(ww_streams)}, audit={ww_audit}", flush=True)

  df_all = pd.DataFrame(pwr_streams + nuc_streams + ind_streams + ww_streams)
  parquet_path = out_dir / "sources_2025_streams.parquet"
  df_all.to_parquet(parquet_path, index=False)

  # Sector & upgrade-mode summary table (EJ/yr = 1e-9 GJ/yr)
  summary_rows = []
  for (sec, sub, up_mode, is_fos), g in df_all.groupby(["sector", "subsector_detail", "upgrade_mode", "is_fossil_power"]):
    summary_rows.append({
        "sector": sec,
        "subsector_detail": sub,
        "upgrade_mode": up_mode,
        "is_fossil_power": bool(is_fos),
        "n_streams": int(len(g)),
        "n_facilities": int(g["facility_id"].nunique()),
        "q3g_ej_yr": float(g["q3g_annual_gj"].sum() / 1e9),
        "q4g_ej_yr": float(g["q4g_annual_gj"].sum() / 1e9),
    })
  df_sum = pd.DataFrame(summary_rows).sort_values(["sector", "subsector_detail", "upgrade_mode"])
  df_sum.to_csv(out_dir / "sources_2025_sector_summary.csv", index=False)

  # Country summary table
  df_cntry = (
      df_all.groupby(["iso3", "sector"], as_index=False)[["q3g_annual_gj", "q4g_annual_gj"]]
      .sum()
      .assign(q3g_ej_yr=lambda d: d["q3g_annual_gj"] / 1e9, q4g_ej_yr=lambda d: d["q4g_annual_gj"] / 1e9)
  )
  df_cntry.to_csv(out_dir / "sources_2025_country_sector.csv", index=False)

  # Global totals & audit receipt
  tot_3g_ej = float(df_all["q3g_annual_gj"].sum() / 1e9)
  tot_4g_ej = float(df_all["q4g_annual_gj"].sum() / 1e9)
  raw_all_gj = (
      pwr_audit["raw_total_gj_before_1mw"]
      + nuc_audit["raw_total_gj_before_1mw"]
      + sum(v["raw_total_gj_to_25c"] for v in ind_audit.values())
      + ww_audit["raw_total_gj_before_1mw"]
  )
  sub1_all_gj = (
      pwr_audit["sub_1mw_excluded_gj"]
      + nuc_audit["sub_1mw_excluded_gj"]
      + sum(v["sub_1mw_excluded_gj"] for v in ind_audit.values())
      + ww_audit["sub_1mw_excluded_gj"]
  )
  receipt = {
      "total_streams": int(len(df_all)),
      "total_facilities": int(df_all["facility_id"].nunique()),
      "total_countries": int(df_all["iso3"].nunique()),
      "global_Q_3gdh_ej_yr": tot_3g_ej,
      "global_Q_4gdh_ej_yr": tot_4g_ej,
      "global_Q_nonfossil_3gdh_ej_yr": float(df_all.loc[~df_all["is_fossil_power"], "q3g_annual_gj"].sum() / 1e9),
      "sub_1mw_excluded_share_pct": 100.0 * sub1_all_gj / raw_all_gj if raw_all_gj > 0 else 0.0,
      "combustion_power_excluded_nonrecoverable_share_pct": 100.0 * (
          pwr_audit["excluded_ocgt_ice_mwh"] + pwr_audit["excluded_dry_cooling_mwh"]
      ) / pwr_audit["ct_power_total_mwh"],
      "combustion_power_audit": pwr_audit,
      "nuclear_audit": nuc_audit,
      "industrial_audit": ind_audit,
      "wastewater_audit": ww_audit,
  }
  (out_dir / "sources_2025_audit_receipt.json").write_text(json.dumps(receipt, indent=2))
  print("\n=== DONE: 2025 GLOBAL WASTE-HEAT SOURCES ===", flush=True)
  print(df_sum.to_string(index=False), flush=True)
  print(f"Global Q (3GDH) = {tot_3g_ej:.3f} EJ/yr | Global Q (4GDH) = {tot_4g_ej:.3f} EJ/yr", flush=True)
  print(f"Sub-1MW excluded = {receipt['sub_1mw_excluded_share_pct']:.3f}% | Power non-recoverable (OCGT/ICE + Dry ACC) = {receipt['combustion_power_excluded_nonrecoverable_share_pct']:.2f}% of CT MWh", flush=True)


if __name__ == "__main__":
  main()
