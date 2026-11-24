"""Validate national 2025 facility generation and industrial outputs against independent statistical benchmarks."""
from __future__ import annotations

import gzip
import json
import re
import subprocess
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pandas as pd

from code.common.paths import DATA_PROCESSED, DATA_RAW

EU27_ISO3 = {
    "AUT", "BEL", "BGR", "HRV", "CYP", "CZE", "DNK", "EST", "FIN",
    "FRA", "DEU", "GRC", "HUN", "IRL", "ITA", "LVA", "LTU", "LUX",
    "MLT", "NLD", "POL", "PRT", "ROU", "SVK", "SVN", "ESP", "SWE",
}
EU28_ISO3 = EU27_ISO3 | {"GBR"}

EUROSTAT_GEO_TO_ISO3 = {
    "AT": "AUT", "BE": "BEL", "BG": "BGR", "HR": "HRV", "CY": "CYP",
    "CZ": "CZE", "DK": "DNK", "EE": "EST", "FI": "FIN", "FR": "FRA",
    "DE": "DEU", "EL": "GRC", "HU": "HUN", "IE": "IRL", "IT": "ITA",
    "LV": "LVA", "LT": "LTU", "LU": "LUX", "MT": "MLT", "NL": "NLD",
    "PL": "POL", "PT": "PRT", "RO": "ROU", "SK": "SVK", "SI": "SVN",
    "ES": "ESP", "SE": "SWE", "UK": "GBR", "NO": "NOR", "IS": "ISL",
    "CH": "CHE", "RS": "SRB", "UA": "UKR", "MD": "MDA", "AL": "ALB",
    "BA": "BIH", "ME": "MNE", "MK": "MKD", "XK": "XKX", "GE": "GEO",
    "TR": "TUR",
}


def validate_power_and_nuclear(df_streams: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
  """Compare 2025 combustion & nuclear MWh against Ember 2026 yearly release."""
  emb = pd.read_csv(DATA_RAW / "supplements_2025/ember/release_generation_yearly_global.csv")
  emb25 = emb[(emb["Year"] == 2025) & (emb["Area type"] == "Country or economy") & emb["ISO 3 code"].notna()].copy()
  emb_world25 = emb[(emb["Year"] == 2025) & (emb["Area"] == "World")]

  comb_sources = {"Coal", "Gas", "Other Fossil", "Bioenergy"}
  world_comb_twh = float(emb_world25[emb_world25["Electricity source"].isin(comb_sources)]["Generation (TWh)"].sum())
  world_nuc_twh = float(emb_world25[emb_world25["Electricity source"] == "Nuclear"]["Generation (TWh)"].sum())
  emb_comb = (
      emb25[emb25["Electricity source"].isin(comb_sources)]
      .groupby("ISO 3 code", as_index=False)["Generation (TWh)"]
      .sum()
      .rename(columns={"ISO 3 code": "iso3", "Generation (TWh)": "ember_comb_twh_2025"})
  )
  emb_nuc = (
      emb25[emb25["Electricity source"] == "Nuclear"]
      .groupby("ISO 3 code", as_index=False)["Generation (TWh)"]
      .sum()
      .rename(columns={"ISO 3 code": "iso3", "Generation (TWh)": "ember_nuclear_gross_twh_2025"})
  )

  # Read raw Climate TRACE 2025 electricity MWh by country (before sub-1MW filter)
  zpath = DATA_RAW / "current_plan_2026-11-15/sources/climate_trace_bulk_2025/all_six_source_sectors_2025_monthly_co2_proxy.zip"
  with zipfile.ZipFile(zpath) as zf:
    ct = pd.read_csv(
        zf.open("DATA/emissions_co2_v5_12_0.csv"),
        usecols=["iso3_country", "subsector", "start_time", "activity"],
        low_memory=False,
    )
  ct_pwr = ct[(ct["subsector"] == "electricity-generation") & ct["start_time"].astype(str).str.startswith("2025")]
  ct_comb = (
      ct_pwr.groupby("iso3_country", as_index=False)["activity"]
      .sum()
      .assign(ct_comb_twh_2025=lambda d: d["activity"] / 1e6)
      .rename(columns={"iso3_country": "iso3"})[["iso3", "ct_comb_twh_2025"]]
  )

  pris_nuc = (
      df_streams[df_streams["subsector_detail"] == "power_nuclear"]
      .drop_duplicates("facility_id")
      .groupby("iso3", as_index=False)["annual_activity"]
      .sum()
      .assign(pris_nuclear_net_twh_2025=lambda d: d["annual_activity"] / 1e6)[["iso3", "pris_nuclear_net_twh_2025"]]
  )

  merged = emb_comb.merge(ct_comb, on="iso3", how="outer").merge(emb_nuc, on="iso3", how="outer").merge(pris_nuc, on="iso3", how="outer").fillna(0.0)
  comb_valid = merged[(merged["ember_comb_twh_2025"] > 1.0) & (merged["ct_comb_twh_2025"] > 0.0)]
  nuc_valid = merged[(merged["ember_nuclear_gross_twh_2025"] > 0.1) | (merged["pris_nuclear_net_twh_2025"] > 0.1)]

  r_comb = float(np.corrcoef(comb_valid["ember_comb_twh_2025"], comb_valid["ct_comb_twh_2025"])[0, 1])
  r_nuc = float(np.corrcoef(nuc_valid["ember_nuclear_gross_twh_2025"], nuc_valid["pris_nuclear_net_twh_2025"])[0, 1])

  stats = {
      "ember_world_combustion_twh_2025": world_comb_twh,
      "ct_global_combustion_twh_2025": float(merged["ct_comb_twh_2025"].sum()),
      "combustion_global_coverage_pct": float(100.0 * merged["ct_comb_twh_2025"].sum() / world_comb_twh),
      "combustion_country_pearson_r": r_comb,
      "combustion_n_countries_compared": int(len(comb_valid)),
      "ember_world_nuclear_gross_twh_2025": world_nuc_twh,
      "pris_global_nuclear_net_twh_2025": float(merged["pris_nuclear_net_twh_2025"].sum()),
      "pris_net_to_ember_gross_ratio_pct": float(100.0 * merged["pris_nuclear_net_twh_2025"].sum() / world_nuc_twh),
      "nuclear_country_pearson_r": r_nuc,
      "nuclear_n_countries_compared": int(len(nuc_valid)),
  }
  return merged.sort_values("ember_comb_twh_2025", ascending=False), stats


def validate_industrial_outputs() -> tuple[pd.DataFrame, dict[str, object]]:
  """Compare 2025 CT steel, cement, and pulp/paper output vs Worldsteel 2026, USGS 2026, and FAOSTAT."""
  zpath = DATA_RAW / "current_plan_2026-11-15/sources/climate_trace_bulk_2025/all_six_source_sectors_2025_monthly_co2_proxy.zip"
  with zipfile.ZipFile(zpath) as zf:
    ct = pd.read_csv(
        zf.open("DATA/emissions_co2_v5_12_0.csv"),
        usecols=["iso3_country", "subsector", "start_time", "activity"],
        low_memory=False,
    )
  ct25 = ct[ct["start_time"].astype(str).str.startswith("2025")].copy()

  # 1. Steel: parse World Steel in Figures 2026 PDF (p.4 global 2025 = 1,849 Mt; p.6 country process table)
  ws_pdf = DATA_RAW / "worldsteel/2026/World-Steel-in-Figures-2026.pdf"
  p35_txt = subprocess.check_output(["pdftotext", "-layout", "-f", "3", "-l", "5", str(ws_pdf), "-"], text=True)
  m_world = re.search(r"2025\s+1[\s,]*849", p35_txt)
  ws_world_mt = 1849.0 if m_world else np.nan

  p6_txt = subprocess.check_output(["pdftotext", "-layout", "-f", "6", "-l", "6", str(ws_pdf), "-"], text=True)
  ws_iso_map = {
      "Austria": "AUT", "Belgium": "BEL", "Bulgaria": "BGR", "Czechia": "CZE",
      "Finland": "FIN", "France": "FRA", "Germany": "DEU", "Greece": "GRC",
      "Italy": "ITA", "Luxembourg": "LUX", "Netherlands": "NLD", "Poland": "POL",
      "Portugal": "PRT", "Romania": "ROU", "Slovakia": "SVK", "Slovenia": "SVN",
      "Spain": "ESP", "Sweden": "SWE", "Türkiye": "TUR", "United Kingdom": "GBR",
      "Russia": "RUS", "Ukraine": "UKR", "Canada": "CAN", "Mexico": "MEX",
      "United States": "USA", "Argentina": "ARG", "Brazil": "BRA", "Chile": "CHL",
      "Egypt": "EGY", "South Africa": "ZAF", "Iran": "IRN", "Saudi Arabia": "SAU",
      "China": "CHN", "India": "IND", "Japan": "JPN", "South Korea": "KOR",
      "Taiwan, China": "TWN", "Vietnam": "VNM", "Australia": "AUS",
  }
  ws_rows = []
  for line in p6_txt.splitlines():
    # Left table on p.6 has: Country [e] <Million tonnes> <Oxygen %> <Electric %> ...
    left_col = line[:82]
    for cname, iso in ws_iso_map.items():
      m = re.match(rf"^\s*{re.escape(cname)}(?:\s+e|\s+\d)?\s+(\d[\d\s]*\.\d)", left_col)
      if m:
        val_mt = float(m.group(1).replace(" ", ""))
        ws_rows.append({"iso3": iso, "sector": "steel", "official_2025_mt": val_mt, "official_source": "Worldsteel_2026_p6"})
        break
  df_ws = pd.DataFrame(ws_rows).drop_duplicates("iso3")
  ct_steel = (
      ct25[ct25["subsector"] == "iron-and-steel"]
      .groupby("iso3_country", as_index=False)["activity"]
      .sum()
      .assign(ct_2025_mt=lambda d: d["activity"] / 1e6)
      .rename(columns={"iso3_country": "iso3"})[["iso3", "ct_2025_mt"]]
  )
  df_steel_cmp = df_ws.merge(ct_steel, on="iso3", how="left").fillna({"ct_2025_mt": 0.0})

  # 2. Cement: parse USGS MCS Feb 2026 p.2 World Production 2025 table (thousand metric tons -> Mt)
  usgs_pdf = DATA_RAW / "usgs_mcs/2026/mcs2026-cement.pdf"
  u2_txt = subprocess.check_output(["pdftotext", "-layout", "-f", "2", "-l", "2", str(usgs_pdf), "-"], text=True)
  usgs_iso_map = {
      "United States (includes Puerto Rico)": "USA",
      "Brazil": "BRA", "China": "CHN", "Egypt": "EGY", "India": "IND",
      "Indonesia": "IDN", "Iran": "IRN", "Japan": "JPN",
      "Korea, Republic of": "KOR", "Mexico": "MEX", "Russia": "RUS",
      "Saudi Arabia": "SAU", "Turkey": "TUR", "Vietnam": "VNM",
  }
  usgs_rows = []
  usgs_world_mt = 3800.0
  for line in u2_txt.splitlines():
    for cname, iso in usgs_iso_map.items():
      if line.strip().startswith(cname):
        nums = re.findall(r"[\d,]+", line[len(cname):])
        if len(nums) >= 2:
          prod_2025_mt = float(nums[1].replace(",", "")) / 1000.0
          usgs_rows.append({"iso3": iso, "sector": "cement", "official_2025_mt": prod_2025_mt, "official_source": "USGS_MCS_2026_p2"})
  df_usgs = pd.DataFrame(usgs_rows)
  ct_cem = (
      ct25[ct25["subsector"] == "cement"]
      .groupby("iso3_country", as_index=False)["activity"]
      .sum()
      .assign(ct_2025_mt=lambda d: d["activity"] / 1e6)
      .rename(columns={"iso3_country": "iso3"})[["iso3", "ct_2025_mt"]]
  )
  df_cem_cmp = df_usgs.merge(ct_cem, on="iso3", how="left").fillna({"ct_2025_mt": 0.0})

  # 3. Pulp & Paper: FAOSTAT Forestry latest year (Wood pulp + Paper and paperboard production)
  with zipfile.ZipFile(DATA_RAW / "faostat/forestry_2026-10-10/Forestry_E_All_Data_Normalized.zip") as zf:
    fao = pd.read_csv(
        zf.open("Forestry_E_All_Data_(Normalized).csv"),
        usecols=["Area Code (M49)", "Area", "Item", "Element", "Year", "Unit", "Value"],
        low_memory=False,
    )
  fao_prod = fao[
      (fao["Element"] == "Production")
      & (fao["Item"].isin(["Wood pulp", "Paper and paperboard"]))
      & (fao["Area Code (M49)"].astype(str).str.strip("'") == "001")
  ]
  latest_fao_yr = int(fao_prod["Year"].max())
  fao_world_mt = float(fao_prod[fao_prod["Year"] == latest_fao_yr]["Value"].sum() / 1e6)
  ct_pp_world_mt = float(ct25[ct25["subsector"] == "pulp-and-paper"]["activity"].sum() / 1e6)

  df_ind_cmp = pd.concat([df_steel_cmp, df_cem_cmp], ignore_index=True)
  r_steel = float(np.corrcoef(df_steel_cmp["official_2025_mt"], df_steel_cmp["ct_2025_mt"])[0, 1]) if len(df_steel_cmp) > 2 else np.nan
  r_cem = float(np.corrcoef(df_cem_cmp["official_2025_mt"], df_cem_cmp["ct_2025_mt"])[0, 1]) if len(df_cem_cmp) > 2 else np.nan

  ind_stats = {
      "steel_worldsteel_2025_mt": ws_world_mt,
      "steel_ct_global_2025_mt": float(ct_steel["ct_2025_mt"].sum()),
      "steel_global_coverage_pct": float(100.0 * ct_steel["ct_2025_mt"].sum() / ws_world_mt),
      "steel_country_pearson_r": r_steel,
      "steel_n_countries_compared": int(len(df_steel_cmp)),
      "cement_usgs_world_2025_mt_incl_grinding": usgs_world_mt,
      "cement_ct_integrated_kiln_2025_mt": float(ct_cem["ct_2025_mt"].sum()),
      "cement_integrated_to_total_cement_pct": float(100.0 * ct_cem["ct_2025_mt"].sum() / usgs_world_mt),
      "cement_major_producers_pearson_r": r_cem,
      "cement_n_countries_compared": int(len(df_cem_cmp)),
      "pulp_paper_faostat_latest_year": latest_fao_yr,
      "pulp_paper_faostat_world_pulp_plus_paper_mt": fao_world_mt,
      "pulp_paper_ct_2025_mt": ct_pp_world_mt,
  }
  return df_ind_cmp, ind_stats


def validate_eu28_seenergies(df_streams: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
  """Compare EU28 (EU27 + GBR) industrial excess heat vs sEEnergies D5.1 Table 24."""
  # sEEnergies D5.1 Table 24 (p.59), Max Internal Heat Recovery [PJ/a] for EU28:
  seenergies_t24 = {
      "ethylene": {"seenergies_sector": "Chemical industry (all basic chemicals)", "pj_25c": 105.9, "pj_55c": 27.0, "pj_95c": 15.6},
      "steel": {"seenergies_sector": "Iron and steel", "pj_25c": 101.2, "pj_55c": 68.9, "pj_95c": 53.8},
      "cement": {"seenergies_sector": "Non-metallic minerals (cement+glass+lime)", "pj_25c": 105.7, "pj_55c": 63.9, "pj_95c": 50.9},
      "pulp_paper": {"seenergies_sector": "Paper and printing", "pj_25c": 79.6, "pj_55c": 21.9, "pj_95c": 15.0},
      "refining": {"seenergies_sector": "Refineries", "pj_25c": 297.4, "pj_55c": 79.8, "pj_95c": 53.5},
  }
  eu_ind = df_streams[df_streams["iso3"].isin(EU28_ISO3) & df_streams["sector"].isin(seenergies_t24.keys())].copy()

  rows = []
  for sec, ref in seenergies_t24.items():
    sub = eu_ind[eu_ind["sector"] == sec]
    # In our stream representation:
    #   to 95 C = 3GDH direct_hex
    #   to 55 C = 4GDH direct_hex
    #   to 25 C = 3GDH direct_hex + 3GDH hp_ind_power (== 4GDH direct_hex + 4GDH hp_ind_power)
    pj_95 = float(sub.loc[sub["upgrade_mode"] == "direct_hex", "q3g_annual_gj"].sum() / 1e6)
    pj_55 = float(sub.loc[sub["upgrade_mode"] == "direct_hex", "q4g_annual_gj"].sum() / 1e6)
    pj_25 = float(sub["q3g_annual_gj"].sum() / 1e6)
    rows.append({
        "sector": sec,
        "seenergies_reference_sector": ref["seenergies_sector"],
        "our_2025_eu28_to_25c_pj": pj_25,
        "seenergies_t24_max_rec_25c_pj": ref["pj_25c"],
        "our_2025_eu28_to_55c_pj": pj_55,
        "seenergies_t24_max_rec_55c_pj": ref["pj_55c"],
        "our_2025_eu28_to_95c_pj": pj_95,
        "seenergies_t24_max_rec_95c_pj": ref["pj_95c"],
    })
  df_eu = pd.DataFrame(rows)

  # Also read Hotmaps EU28 site-level database (Excess_Heat_Total in GWh/yr -> PJ/yr = * 0.0036)
  hm = pd.read_csv(DATA_RAW / "hotmaps_industrial/zenodo_15834781/Industrial_Database.csv", sep=";", low_memory=False)
  hm["Excess_Heat_Total"] = pd.to_numeric(hm["Excess_Heat_Total"], errors="coerce").fillna(0.0)
  hm_total_pj = float(hm["Excess_Heat_Total"].sum() * 0.0036)

  eu_stats = {
      "our_2025_eu28_5sectors_to_25c_pj": float(df_eu["our_2025_eu28_to_25c_pj"].sum()),
      "seenergies_t24_eu28_max_rec_25c_pj": float(df_eu["seenergies_t24_max_rec_25c_pj"].sum()),
      "our_2025_eu28_5sectors_to_55c_pj": float(df_eu["our_2025_eu28_to_55c_pj"].sum()),
      "seenergies_t24_eu28_max_rec_55c_pj": float(df_eu["seenergies_t24_max_rec_55c_pj"].sum()),
      "our_2025_eu28_5sectors_to_95c_pj": float(df_eu["our_2025_eu28_to_95c_pj"].sum()),
      "seenergies_t24_eu28_max_rec_95c_pj": float(df_eu["seenergies_t24_max_rec_95c_pj"].sum()),
      "hotmaps_zenodo_15834781_highT_flue_pj": hm_total_pj,
  }
  return df_eu, eu_stats


def extract_existing_chp_dh_national() -> tuple[pd.DataFrame, dict[str, float]]:
  """Extract official national District Heat & CHP heat production/delivery (Eurostat + UNSD)."""
  # Map UN M49 numeric code -> ISO3 using pycountry or country-name lookup from DSD + FAOSTAT/Ember
  emb = pd.read_csv(
      DATA_RAW / "supplements_2025/ember/release_generation_yearly_global.csv",
      usecols=["Area", "ISO 3 code"],
  ).dropna().drop_duplicates()
  name_to_iso = {re.sub(r"[^a-z0-9]+", "", str(r["Area"]).lower()): str(r["ISO 3 code"]).strip() for _, r in emb.iterrows()}
  name_to_iso.update({
      "chinamainland": "CHN", "china": "CHN", "chinahongkongspecialadministrativeregion": "HKG",
      "chinamacaospecialadministrativeregion": "MAC", "russianfederation": "RUS",
      "unitedstatesofamerica": "USA", "unitedkingdomofgreatbritainandnorthernireland": "GBR",
      "republicofkorea": "KOR", "iranislamicrepublicof": "IRN", "turkiye": "TUR", "turkey": "TUR",
      "czechia": "CZE", "republicofmoldova": "MDA", "vietnam": "VNM",
      "democraticpeoplesrepublicofkorea": "PRK", "syrianarabrepublic": "SYR",
      "boliviaplurinationalstateof": "BOL", "venezuelabolivarianrepublicof": "VEN",
      "tanzaniaunitedrepublicof": "TZA", "laopeoplesdemocraticrepublic": "LAO",
      "netherlandskingdomofthe": "NLD",
  })

  dsd_tree = ET.parse(DATA_RAW / "current_plan_2026-11-15/sinks/unsd_energy_balance_dsd.xml")
  m49_to_iso: dict[str, str] = {}
  for cl in dsd_tree.findall(".//{http://www.sdmx.org/resources/sdmxml/schemas/v2_1/structure}Codelist"):
    if "AREA" in str(cl.attrib.get("id", "")):
      for c in cl.findall("{http://www.sdmx.org/resources/sdmxml/schemas/v2_1/structure}Code"):
        cid = str(c.attrib.get("id", "")).strip()
        nm_el = c.find("{http://www.sdmx.org/resources/sdmxml/schemas/v2_1/common}Name")
        nm = nm_el.text if nm_el is not None and nm_el.text else ""
        key = re.sub(r"[^a-z0-9]+", "", nm.lower())
        if key in name_to_iso:
          m49_to_iso[cid] = name_to_iso[key]

  # Parse UNSD Energy Balance Heat XML (COMMODITY=B08_HT, UNIT=HSO [Terajoules])
  # Key transactions:
  #   B11_088   = Electricity, Heat and CHP plants (total derived heat gross output)
  #   B12_08821 = Transformation in CHP plants - main activity producers
  #   B25_FEC   = Final Energy Consumption of derived heat
  #   B47_123   = Other sectors (Residential + Services + Agriculture/Other buildings)
  unsd_tree = ET.parse(DATA_RAW / "current_plan_2026-11-15/sinks/unsd_balance_heat_2020_2025.xml")
  ns_g = "{http://www.sdmx.org/resources/sdmxml/schemas/v2_1/data/generic}"
  unsd_rows = []
  for ser in unsd_tree.findall(f".//{ns_g}Series"):
    kv = {v.attrib["id"]: v.attrib["value"] for v in ser.findall(f"{ns_g}SeriesKey/{ns_g}Value")}
    if kv.get("COMMODITY") != "B08_HT" or kv.get("UNIT") != "HSO":
      continue
    tr = kv.get("TRANSACTION", "")
    if tr not in ("B11_088", "B12_08821", "B25_FEC", "B47_123"):
      continue
    area = kv.get("REF_AREA", "")
    iso3 = "WORLD_UNSD" if area == "001" else m49_to_iso.get(area, "")
    if not iso3:
      continue
    best_yr, best_val = -1, 0.0
    for obs in ser.findall(f"{ns_g}Obs"):
      ydim = obs.find(f"{ns_g}ObsDimension")
      oval = obs.find(f"{ns_g}ObsValue")
      if ydim is None or oval is None:
        continue
      yr = int(ydim.attrib["value"])
      umult = 0
      for av in obs.findall(f"{ns_g}Attributes/{ns_g}Value"):
        if av.attrib.get("id") == "UNIT_MULT":
          umult = int(av.attrib.get("value", "0"))
      val_tj = float(oval.attrib["value"]) * (10.0 ** umult)
      if yr >= best_yr and not np.isnan(val_tj):
        best_yr, best_val = yr, max(0.0, val_tj)
    if best_yr > 0:
      unsd_rows.append({"iso3": iso3, "transaction": tr, "year": best_yr, "tj": best_val})

  df_u = pd.DataFrame(unsd_rows)
  piv_u = df_u.pivot_table(index="iso3", columns="transaction", values="tj", aggfunc="max", fill_value=0.0).reset_index()
  yr_u = df_u.groupby("iso3", as_index=False)["year"].max().rename(columns={"year": "unsd_latest_year"})
  piv_u = piv_u.merge(yr_u, on="iso3", how="left")
  for col in ("B11_088", "B12_08821", "B25_FEC", "B47_123"):
    if col not in piv_u.columns:
      piv_u[col] = 0.0
  piv_u = piv_u.rename(columns={
      "B11_088": "dh_total_output_tj",
      "B12_08821": "chp_main_output_tj",
      "B25_FEC": "dh_final_consumption_tj",
      "B47_123": "dh_buildings_other_tj",
  })

  # Overlay Eurostat 2024 nrg_bal_c (siec=H8000, unit=TJ) for European countries where more recent
  euro_rows = []
  euro_codes = ("TO_EHG", "TO_EHG_MAPCHP", "TO_EHG_APCHP", "FC_E", "FC_OTH_HH_E", "FC_OTH_CP_E")
  with gzip.open(DATA_RAW / "eurostat_nrg_bal/2026-10-10/nrg_bal_c.tsv.gz", "rt") as f:
    header = f.readline().rstrip("\r\n").split("\t")
    yr_cols = [(i, int(c.strip())) for i, c in enumerate(header[1:], start=1) if c.strip().isdigit()]
    for line in f:
      if ",H8000,TJ," not in line:
        continue
      parts = line.rstrip("\r\n").split("\t")
      meta_p = parts[0].split(",")
      if len(meta_p) != 5:
        continue
      _, nrg_bal, siec, unit, geo = meta_p
      if geo not in EUROSTAT_GEO_TO_ISO3 or nrg_bal not in euro_codes:
        continue
      iso3 = EUROSTAT_GEO_TO_ISO3[geo]
      # Pick latest valid year (2024 -> 2023 -> 2022)
      for col_idx, yr in sorted(yr_cols, key=lambda x: -x[1]):
        if col_idx < len(parts):
          raw_v = re.sub(r"[^0-9.\-]", "", parts[col_idx])
          if raw_v and raw_v not in ("", "-", "."):
            euro_rows.append({"iso3": iso3, "nrg_bal": nrg_bal, "year": yr, "tj": max(0.0, float(raw_v))})
            break
  df_e = pd.DataFrame(euro_rows)
  piv_e = df_e.pivot_table(index="iso3", columns="nrg_bal", values="tj", aggfunc="max", fill_value=0.0).reset_index()
  yr_e = df_e.groupby("iso3", as_index=False)["year"].max().rename(columns={"year": "eurostat_latest_year"})
  piv_e = piv_e.merge(yr_e, on="iso3", how="left")
  for c in euro_codes:
    if c not in piv_e.columns:
      piv_e[c] = 0.0
  piv_e["euro_chp_tj"] = piv_e["TO_EHG_MAPCHP"] + piv_e["TO_EHG_APCHP"]
  piv_e["euro_bldg_tj"] = piv_e["FC_OTH_HH_E"] + piv_e["FC_OTH_CP_E"]
  merged_dh = piv_u.merge(
      piv_e[["iso3", "eurostat_latest_year", "TO_EHG", "euro_chp_tj", "FC_E", "euro_bldg_tj"]],
      on="iso3",
      how="outer",
  )

  # Prefer Eurostat 2024 where available, else UNSD
  has_e = merged_dh["eurostat_latest_year"].notna()
  merged_dh["ref_year"] = merged_dh["eurostat_latest_year"].combine_first(merged_dh["unsd_latest_year"]).fillna(0).astype(int)
  merged_dh["source_agency"] = np.where(has_e, "Eurostat_nrg_bal_c", "UNSD_EnergyBalance")
  merged_dh["gross_dh_output_ej_yr"] = np.where(has_e, merged_dh["TO_EHG"], merged_dh["dh_total_output_tj"]) / 1e6
  merged_dh["chp_heat_output_ej_yr"] = np.where(has_e, merged_dh["euro_chp_tj"], merged_dh["chp_main_output_tj"]) / 1e6
  merged_dh["final_dh_consumption_ej_yr"] = np.where(has_e, merged_dh["FC_E"], merged_dh["dh_final_consumption_tj"]) / 1e6
  merged_dh["buildings_dh_consumption_ej_yr"] = np.where(has_e, merged_dh["euro_bldg_tj"], merged_dh["dh_buildings_other_tj"]) / 1e6

  out_cols = [
      "iso3", "ref_year", "source_agency",
      "gross_dh_output_ej_yr", "chp_heat_output_ej_yr",
      "final_dh_consumption_ej_yr", "buildings_dh_consumption_ej_yr",
  ]
  df_cntry = merged_dh[merged_dh["iso3"] != "WORLD_UNSD"][out_cols].sort_values("gross_dh_output_ej_yr", ascending=False).reset_index(drop=True)
  dh_stats = {
      "global_sum_countries_gross_dh_output_ej_yr": float(df_cntry["gross_dh_output_ej_yr"].sum()),
      "global_sum_countries_chp_heat_output_ej_yr": float(df_cntry["chp_heat_output_ej_yr"].sum()),
      "global_sum_countries_buildings_dh_ej_yr": float(df_cntry["buildings_dh_consumption_ej_yr"].sum()),
      "chn_gross_dh_output_ej_yr": float(df_cntry.loc[df_cntry["iso3"] == "CHN", "gross_dh_output_ej_yr"].sum()),
      "rus_gross_dh_output_ej_yr": float(df_cntry.loc[df_cntry["iso3"] == "RUS", "gross_dh_output_ej_yr"].sum()),
      "eu27_gross_dh_output_ej_yr": float(df_cntry.loc[df_cntry["iso3"].isin(EU27_ISO3), "gross_dh_output_ej_yr"].sum()),
      "n_countries_with_dh": int((df_cntry["gross_dh_output_ej_yr"] > 0).sum()),
  }
  return df_cntry, dh_stats


def main() -> None:
  out_dir = DATA_PROCESSED / "sources_2025"
  val_dir = out_dir / "validation_1"
  val_dir.mkdir(parents=True, exist_ok=True)

  df_streams = pd.read_parquet(out_dir / "sources_2025_streams.parquet")

  print("1. Validating 2025 Power & Nuclear generation against Ember 2026...", flush=True)
  df_pwr_val, pwr_stats = validate_power_and_nuclear(df_streams)
  df_pwr_val.to_csv(val_dir / "power_nuclear_vs_ember2025.csv", index=False)
  print("  ", pwr_stats, flush=True)

  print("2. Validating 2025 Industrial outputs against Worldsteel 2026, USGS 2026 & FAOSTAT...", flush=True)
  df_ind_val, ind_stats = validate_industrial_outputs()
  df_ind_val.to_csv(val_dir / "industrial_output_vs_official2025.csv", index=False)
  print("  ", ind_stats, flush=True)

  print("3. Validating EU28 Industrial Excess Heat against sEEnergies D5.1 Table 24...", flush=True)
  df_eu_val, eu_stats = validate_eu28_seenergies(df_streams)
  df_eu_val.to_csv(val_dir / "eu28_excess_heat_vs_seenergies_t24.csv", index=False)
  print(df_eu_val.to_string(index=False), flush=True)

  print("4. Extracting official national District Heating & CHP baselines (Eurostat + UNSD)...", flush=True)
  df_dh, dh_stats = extract_existing_chp_dh_national()
  df_dh.to_csv(out_dir / "existing_chp_dh_national.csv", index=False)
  print("  ", dh_stats, flush=True)
  print("   Top 10 DH countries:\n", df_dh.head(10).to_string(index=False), flush=True)

  report = {
      "validation_1_power_and_nuclear": pwr_stats,
      "validation_1_industrial_physical_output": ind_stats,
      "validation_1_eu28_seenergies_d51": eu_stats,
      "existing_chp_dh_baseline_summary": dh_stats,
  }
  (val_dir / "validation_1_summary.json").write_text(json.dumps(report, indent=2))
  print("\n=== VALIDATION 1 COMPLETE ===", flush=True)


if __name__ == "__main__":
  main()
