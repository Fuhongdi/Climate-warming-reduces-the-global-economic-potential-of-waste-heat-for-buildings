"""Calibrate regional space-heating and water-heating demand coefficients, 2025 energy tariffs and grid carbon factors."""
from __future__ import annotations

import gzip
import json
import math
import re
import xml.etree.ElementTree as ET
import zipfile

import netCDF4 as nc
import numpy as np
import openpyxl
import pandas as pd

from code.common.parameters import (
    BAIT_THRESHOLD_CENTRAL_C,
    BAIT_THRESHOLD_HIGH_C,
    BAIT_THRESHOLD_LOW_C,
    USD_PER_EUR_2025,
)
from code.common.paths import DATA_PROCESSED, DATA_RAW

GJ_PER_KTOE = 41_868.0
GJ_PER_PJ = 1_000_000.0
GJ_PER_TJ = 1_000.0
GJ_PER_KBTU = 0.00105505585
GJ_LHV_PER_MMBTU_HHV = 1.05505585 * 0.901  # natural gas HHV -> LHV (0.9506 GJ_LHV / MMBtu_HHV)

# Stock-average final-to-useful thermal efficiencies (DEA & JRC-IDEES stock averages, LHV basis)
EFF_SH = {
    "gas": 0.88,
    "oil": 0.84,
    "coal": 0.70,
    "bio": 0.72,
    "heat": 0.98,
    "elec": 1.32,  # stock blend of direct electric resistance (1.0) & existing heat pumps (~2.8)
}
EFF_WH = {
    "gas": 0.84,
    "oil": 0.80,
    "coal": 0.65,
    "bio": 0.68,
    "heat": 0.98,
    "elec": 0.95,
}

EU27_MAP = {
    "AT": "AUT", "BE": "BEL", "BG": "BGR", "CY": "CYP", "CZ": "CZE", "DE": "DEU",
    "DK": "DNK", "EE": "EST", "EL": "GRC", "ES": "ESP", "FI": "FIN", "FR": "FRA",
    "HR": "HRV", "HU": "HUN", "IE": "IRL", "IT": "ITA", "LT": "LTU", "LU": "LUX",
    "LV": "LVA", "MT": "MLT", "NL": "NLD", "PL": "POL", "PT": "PRT", "RO": "ROU",
    "SE": "SWE", "SI": "SVN", "SK": "SVK",
}
EUROSTAT_GEO_MAP = {
    **EU27_MAP,
    "UK": "GBR", "NO": "NOR", "IS": "ISL", "CH": "CHE", "TR": "TUR", "UA": "UKR",
    "RS": "SRB", "MD": "MDA", "AL": "ALB", "BA": "BIH", "ME": "MNE", "MK": "MKD",
    "GE": "GEO", "XK": "XKX",
}
OECD_ISO3 = set(EU27_MAP.values()) | {
    "GBR", "NOR", "ISL", "CHE", "USA", "CAN", "JPN", "KOR", "AUS", "NZL",
    "ISR", "CHL", "COL", "CRI", "MEX", "TUR",
}


def norm_key(s: object) -> str:
  return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def compute_annual_hdd_cache() -> dict[tuple[int, float], np.ndarray]:
  """Compute cell-level annual HDD for years 2018..2025 at thresholds 12, 14, 16 C + 2006-2025 mean."""
  bait_dir = DATA_PROCESSED / "bait/gswp3-era5__obsclim"
  cache: dict[tuple[int, float], np.ndarray] = {}
  acc_06_25 = {th: np.zeros(52160, dtype=np.float64) for th in (12.0, 14.0, 16.0)}

  for yr in range(2006, 2026):
    with nc.Dataset(bait_dir / f"year_{yr}.nc", "r") as ds:
      b = np.asarray(ds["bait_c"][:, :], dtype=np.float64)
    for th in (12.0, 14.0, 16.0):
      hdd = np.maximum(th - b, 0.0).sum(axis=0)
      acc_06_25[th] += hdd
      if yr >= 2018:
        cache[(yr, th)] = hdd

  for th in (12.0, 14.0, 16.0):
    cache[(0, th)] = acc_06_25[th] / 20.0  # yr=0 denotes 2006-2025 20-yr baseline mean
  return cache


def build_country_name_and_m49_maps() -> tuple[dict[str, str], dict[str, str]]:
  emb = pd.read_csv(
      DATA_RAW / "supplements_2025/ember/release_generation_yearly_global.csv",
      usecols=["Area", "ISO 3 code"],
  ).dropna().drop_duplicates()
  name_to_iso = {norm_key(r["Area"]): str(r["ISO 3 code"]).strip() for _, r in emb.iterrows()}
  meta_g = pd.read_parquet(DATA_PROCESSED / "sinks_2025/gadm_units_meta.parquet")[["iso3", "COUNTRY"]].drop_duplicates()
  for _, r in meta_g.iterrows():
    name_to_iso.setdefault(norm_key(r["COUNTRY"]), str(r["iso3"]).strip())
  name_to_iso.update({
      "czechrepublic": "CZE", "czechia": "CZE", "korearepublicof": "KOR", "republicofkorea": "KOR",
      "korea": "KOR", "slovakrepublic": "SVK", "turkiye": "TUR", "turkey": "TUR",
      "unitedstates": "USA", "unitedstatesofamerica": "USA", "unitedkingdom": "GBR",
      "unitedkingdomofgreatbritainandnorthernireland": "GBR", "russianfederation": "RUS",
      "russia": "RUS", "chinamainland": "CHN", "peoplesrepublicofchina": "CHN",
      "iranislamicrepublicof": "IRN", "egyptarabrep": "EGY", "vietnam": "VNM",
      "republicofmoldova": "MDA", "netherlandskingdomofthe": "NLD",
      "boliviaplurinationalstateof": "BOL", "venezuelabolivarianrepublicof": "VEN",
      "tanzaniaunitedrepublicof": "TZA", "syrianarabrepublic": "SYR",
      "chinahongkongspecialadministrativeregion": "HKG", "hongkongchina": "HKG",
      "chinamacaospecialadministrativeregion": "MAC", "taiwanchina": "TWN", "chinesetaipei": "TWN",
      "kyrgyzrepublic": "KGZ", "kyrgyzstan": "KGZ", "ivorycoast": "CIV", "cotedivoire": "CIV",
  })

  m49_to_iso: dict[str, str] = {}
  for dsd_p in [
      DATA_RAW / "current_plan_2026-11-15/sinks/unsd_energy_balance_dsd.xml",
      DATA_RAW / "sinks/2026-11-13/unsd/energy_dsd.xml",
  ]:
    tree = ET.parse(dsd_p)
    for cl in tree.findall(".//{http://www.sdmx.org/resources/sdmxml/schemas/v2_1/structure}Codelist"):
      if "AREA" in str(cl.attrib.get("id", "")):
        for c in cl.findall("{http://www.sdmx.org/resources/sdmxml/schemas/v2_1/structure}Code"):
          cid = str(c.attrib.get("id", "")).strip()
          nm_el = c.find("{http://www.sdmx.org/resources/sdmxml/schemas/v2_1/common}Name")
          nm = nm_el.text if nm_el is not None and nm_el.text else ""
          if norm_key(nm) in name_to_iso:
            m49_to_iso[cid] = name_to_iso[norm_key(nm)]
            m49_to_iso[cid.lstrip("0") or "0"] = name_to_iso[norm_key(nm)]
  return name_to_iso, m49_to_iso


def load_tier1_us_uk_nz() -> dict[str, dict[str, object]]:
  """Load USA 9 Census Divisions (RECS 2020 + CBECS 2018), GBR (ECUK 2025), and NZL (EECA 2024)."""
  out: dict[str, dict[str, object]] = {}

  # 1. USA 9 Census Divisions
  recs = pd.read_csv(
      DATA_RAW / "sinks/2026-11-13/eia_recs2020/recs2020_public_v7.csv",
      usecols=[
          "DIVISION", "NWEIGHT",
          "BTUELSPH", "BTUNGSPH", "BTULPSPH", "BTUFOSPH",
          "BTUELWTH", "BTUNGWTH", "BTULPWTH", "BTUFOWTH",
      ],
  )
  div_names_recs = {
      "New England": "USA_DIV1", "Middle Atlantic": "USA_DIV2", "East North Central": "USA_DIV3",
      "West North Central": "USA_DIV4", "South Atlantic": "USA_DIV5", "East South Central": "USA_DIV6",
      "West South Central": "USA_DIV7", "Mountain North": "USA_DIV8", "Mountain South": "USA_DIV8",
      "Mountain": "USA_DIV8", "Pacific": "USA_DIV9",
      "1": "USA_DIV1", "2": "USA_DIV2", "3": "USA_DIV3", "4": "USA_DIV4", "5": "USA_DIV5",
      "6": "USA_DIV6", "7": "USA_DIV7", "8": "USA_DIV8", "9": "USA_DIV9",
  }
  recs["rid"] = recs["DIVISION"].astype(str).str.strip().map(div_names_recs)
  w_r = recs["NWEIGHT"].values
  recs["sh_useful_gj"] = w_r * GJ_PER_KBTU * (
      recs["BTUELSPH"].fillna(0) * EFF_SH["elec"]
      + (recs["BTUNGSPH"].fillna(0) + recs["BTULPSPH"].fillna(0)) * EFF_SH["gas"]
      + recs["BTUFOSPH"].fillna(0) * EFF_SH["oil"]
  )
  recs["wh_useful_gj"] = w_r * GJ_PER_KBTU * (
      recs["BTUELWTH"].fillna(0) * EFF_WH["elec"]
      + (recs["BTUNGWTH"].fillna(0) + recs["BTULPWTH"].fillna(0)) * EFF_WH["gas"]
      + recs["BTUFOWTH"].fillna(0) * EFF_WH["oil"]
  )
  recs["gas_final_gj"] = w_r * GJ_PER_KBTU * (recs["BTUNGSPH"].fillna(0) + recs["BTUNGWTH"].fillna(0))
  r_agg = recs.groupby("rid", as_index=False)[["sh_useful_gj", "wh_useful_gj", "gas_final_gj"]].sum()

  cbecs = pd.read_csv(
      DATA_RAW / "sinks/2026-11-13/eia_cbecs2018/cbecs2018_final_public.csv",
      usecols=[
          "CENDIV", "FINALWT",
          "ELHTBTU", "NGHTBTU", "FKHTBTU", "DHHTBTU",
          "ELWTBTU", "NGWTBTU", "FKWTBTU", "DHWTBTU",
      ],
  )
  cbecs["rid"] = cbecs["CENDIV"].astype(int).map(lambda x: f"USA_DIV{x}")
  w_c = cbecs["FINALWT"].values
  cbecs["sh_useful_gj"] = w_c * GJ_PER_KBTU * (
      cbecs["ELHTBTU"].fillna(0) * EFF_SH["elec"]
      + cbecs["NGHTBTU"].fillna(0) * EFF_SH["gas"]
      + cbecs["FKHTBTU"].fillna(0) * EFF_SH["oil"]
      + cbecs["DHHTBTU"].fillna(0) * EFF_SH["heat"]
  )
  cbecs["wh_useful_gj"] = w_c * GJ_PER_KBTU * (
      cbecs["ELWTBTU"].fillna(0) * EFF_WH["elec"]
      + cbecs["NGWTBTU"].fillna(0) * EFF_WH["gas"]
      + cbecs["FKWTBTU"].fillna(0) * EFF_WH["oil"]
      + cbecs["DHWTBTU"].fillna(0) * EFF_WH["heat"]
  )
  cbecs["gas_final_gj"] = w_c * GJ_PER_KBTU * (cbecs["NGHTBTU"].fillna(0) + cbecs["NGWTBTU"].fillna(0))
  c_agg = cbecs.groupby("rid", as_index=False)[["sh_useful_gj", "wh_useful_gj", "gas_final_gj"]].sum()

  us_m = r_agg.merge(c_agg, on="rid", suffixes=("_res", "_com"))
  for _, row in us_m.iterrows():
    out[str(row["rid"])] = {
        "iso3": "USA",
        "cal_year": 2020,
        "sh_useful_gj": float(row["sh_useful_gj_res"] + row["sh_useful_gj_com"]),
        "wh_useful_gj": float(row["wh_useful_gj_res"] + row["wh_useful_gj_com"]),
        "has_gas": True,
        "official_enduse": True,
        "cal_source": "EIA_RECS2020_CBECS2018",
    }

  # 2. GBR: ECUK 2026 Table U2 (Year = 2025, Domestic + Services, ktoe)
  wb_uk = openpyxl.load_workbook(
      DATA_RAW / "residential_sink_20261114/national/uk_ecuk2026/ECUK_2026_End_Use_tables.xlsx",
      read_only=True, data_only=True,
  )
  ws_uk = wb_uk["Table U2"]
  uk_sh, uk_wh = 0.0, 0.0
  for r in ws_uk.iter_rows(min_row=4, values_only=True):
    if r[0] == 2025 and r[1] in ("Domestic", "Services"):
      eu = str(r[2]).strip().lower()
      ng, oil, coal, el, ht, bio = [float(r[k] or 0.0) * GJ_PER_KTOE for k in range(3, 9)]
      if eu == "space heating":
        uk_sh += ng * EFF_SH["gas"] + oil * EFF_SH["oil"] + coal * EFF_SH["coal"] + el * EFF_SH["elec"] + ht * EFF_SH["heat"] + bio * EFF_SH["bio"]
      elif eu == "water heating":
        uk_wh += ng * EFF_WH["gas"] + oil * EFF_WH["oil"] + coal * EFF_WH["coal"] + el * EFF_WH["elec"] + ht * EFF_WH["heat"] + bio * EFF_WH["bio"]
  wb_uk.close()
  out["GBR"] = {
      "iso3": "GBR", "cal_year": 2025,
      "sh_useful_gj": uk_sh, "wh_useful_gj": uk_wh,
      "has_gas": True, "official_enduse": True, "cal_source": "UK_ECUK_2026_TableU2",
  }

  # 3. NZL: EECA EEUD 2024 (TJ)
  nz = pd.read_csv(DATA_RAW / "sinks/2026-11-13/eeca/EEUD-Data-2017-2024.csv")
  nz24 = nz[nz["PeriodEndDate"].astype(str).str.contains("2024") & nz["SectorGroup"].isin(["Residential", "Commercial"])]
  nz_sh, nz_wh, nz_gas = 0.0, 0.0, 0.0
  for _, r in nz24.iterrows():
    eu = str(r["EndUse"]).lower()
    fg = str(r["FuelGroup"]).lower()
    fl = str(r["Fuel"]).lower()
    gj = float(r["EnergyValue (Terajoules - TJ)"] or 0.0) * GJ_PER_TJ
    if "natural gas" in fl:
      nz_gas += gj
    fkey = "elec" if "elec" in fg else ("gas" if "gas" in fl else ("coal" if "coal" in fl else ("bio" if "wood" in fl or "bio" in fg else "oil")))
    if "space heat" in eu or ("low" in eu and "heat" in eu and "water" not in eu):
      nz_sh += gj * EFF_SH[fkey]
    elif "water heat" in eu:
      nz_wh += gj * EFF_WH[fkey]
  out["NZL"] = {
      "iso3": "NZL", "cal_year": 2024,
      "sh_useful_gj": nz_sh, "wh_useful_gj": nz_wh,
      "has_gas": nz_gas > 1e5, "official_enduse": True, "cal_source": "NZ_EECA_EEUD_2024",
  }
  return out


def load_tier2_eu27_jrc() -> dict[str, dict[str, object]]:
  """Load EU27 direct useful thermal energy (SH + WH/HW) from JRC-IDEES-2023 (year 2021)."""
  out: dict[str, dict[str, object]] = {}
  with zipfile.ZipFile(DATA_RAW / "residential_sink_20261114/national/jrc_idees/JRC-IDEES-2023.zip") as zf:
    names = set(zf.namelist())
    for geo2, iso3 in EU27_MAP.items():
      r_file = f"{geo2}/JRC-IDEES-2023_Residential_{geo2}.xlsx"
      s_file = f"{geo2}/JRC-IDEES-2023_Tertiary_{geo2}.xlsx"
      if r_file not in names or s_file not in names:
        continue
      sh_ktoe, wh_ktoe, gas_ktoe = 0.0, 0.0, 0.0
      for fpath, sh_name, code_sh, code_wh in [
          (r_file, "RES_hh_tes", f"TES.ktoe.{geo2}.Res.HH.Thermal.SH", f"TES.ktoe.{geo2}.Res.HH.Thermal.WH"),
          (s_file, "SER_hh_tes", f"TES.ktoe.{geo2}.Ser.Bld.Thermal.SH", f"TES.ktoe.{geo2}.Ser.Bld.Thermal.HW"),
      ]:
        wb = openpyxl.load_workbook(zf.open(fpath), read_only=True, data_only=True)
        ws = wb[sh_name]
        for row in ws.iter_rows(max_row=36, values_only=True):
          code_cells = [str(x).strip() for x in row if isinstance(x, str) and x.startswith("TES.ktoe.")]
          if not code_cells:
            continue
          code = code_cells[0]
          nums = [float(x) for x in row[1:] if isinstance(x, (int, float)) and not math.isnan(float(x))]
          if not nums:
            continue
          val_2021 = nums[-1]
          if code == code_sh:
            sh_ktoe += val_2021
          elif code == code_wh:
            wh_ktoe += val_2021
          if ".NG_Biogas" in code:
            gas_ktoe += val_2021
        wb.close()
      out[iso3] = {
          "iso3": iso3, "cal_year": 2021,
          "sh_useful_gj": sh_ktoe * GJ_PER_KTOE,
          "wh_useful_gj": wh_ktoe * GJ_PER_KTOE,
          "has_gas": gas_ktoe > 2.5,  # > ~0.1 PJ useful gas
          "official_enduse": True,
          "cal_source": "JRC_IDEES_2023_UsefulTES",
      }
  return out


def load_eurostat_hhq_donors() -> list[dict[str, object]]:
  """Extract exact household fuel-by-end-use shares (SH, WH) from Eurostat nrg_d_hhq (35+ countries)."""
  siec_map = {
      "G3000": "gas",
      "O4000": "oil",
      "SFF_P1000_S2000": "coal",
      "R5110-5150_W6000RI": "bio",
      "H8000": "heat",
      "E7000": "elec",
  }
  bals_req = {"FC_OTH_HH_E", "FC_OTH_HH_E_SH", "FC_OTH_HH_E_WH"}
  vals: dict[tuple[str, str, str], float] = {}
  with gzip.open(DATA_RAW / "supplements_2025/eurostat/nrg_d_hhq.tsv.gz", "rt") as f:
    hdr = f.readline().rstrip("\r\n").split("\t")
    yr_cols = sorted([(i, int(c.strip())) for i, c in enumerate(hdr[1:], start=1) if c.strip().isdigit()], key=lambda x: -x[1])
    for ln in f:
      if ",TJ," not in ln:
        continue
      parts = ln.rstrip("\r\n").split("\t")
      mp = parts[0].split(",")
      if len(mp) != 5:
        continue
      _, nb, sc, _, geo = mp
      if nb not in bals_req or sc not in siec_map or geo not in EUROSTAT_GEO_MAP:
        continue
      iso = EUROSTAT_GEO_MAP[geo]
      fk = siec_map[sc]
      for ci, _ in yr_cols:
        if ci < len(parts):
          rv = re.sub(r"[^0-9.\-]", "", parts[ci])
          if rv and rv not in ("", "-", "."):
            vals[(iso, nb, fk)] = max(0.0, float(rv))
            break
  donors = []
  isos = sorted({k[0] for k in vals})
  for iso in isos:
    sh_s, wh_s = {}, {}
    for fk in EFF_SH:
      tot = vals.get((iso, "FC_OTH_HH_E", fk), 0.0)
      if tot > 0:
        sh_s[fk] = min(1.0, vals.get((iso, "FC_OTH_HH_E_SH", fk), 0.0) / tot)
        wh_s[fk] = min(max(0.0, 1.0 - sh_s[fk]), vals.get((iso, "FC_OTH_HH_E_WH", fk), 0.0) / tot)
    if sh_s:
      donors.append({"iso3": iso, "res_sh": sh_s, "res_wh": wh_s})
  return donors


def load_tier3_iea_eei(name_to_iso: dict[str, str], existing: dict[str, dict[str, object]]) -> tuple[dict[str, dict[str, object]], list[dict[str, object]]]:
  """Load IEA EEI June 2026 Residential + Services end-use data & collect Eurostat fuel end-use donors."""
  f_iea = DATA_RAW / "current_plan_2026-11-15/sinks/iea_public/IEAEEIdatabase-HighlightsJune2026.xlsb"
  df_res = pd.read_excel(f_iea, engine="pyxlsb", sheet_name="Residential - Energy", skiprows=1)
  df_ser = pd.read_excel(f_iea, engine="pyxlsb", sheet_name="Services - Energy", skiprows=1)

  prod_map = {
      "Total final use (PJ)": "total",
      "Oil and oil products (PJ)": "oil",
      "Gas (PJ)": "gas",
      "Coal and coal products (PJ)": "coal",
      "Biofuels and waste (PJ)": "bio",
      "Heat (PJ)": "heat",
      "Electricity (PJ)": "elec",
  }
  yr_cols = sorted([c for c in df_res.columns if str(c).strip().isdigit() and 2018 <= int(str(c).strip()) <= 2025], key=lambda x: -int(str(x).strip()))

  def extract_country_fuel_enduse(df_sec: pd.DataFrame, tot_label: str) -> dict[str, tuple[int, dict[tuple[str, str], float]]]:
    res_map: dict[str, tuple[int, dict[tuple[str, str], float]]] = {}
    for cname, grp in df_sec.groupby("Country"):
      iso = name_to_iso.get(norm_key(cname), "")
      if not iso or len(iso) != 3:
        continue
      sub = grp[grp["Product"].isin(prod_map.keys())]
      for yr in yr_cols:
        vals = pd.to_numeric(sub[yr], errors="coerce")
        tot_sub = sub[sub["End use"] == tot_label]
        if pd.to_numeric(tot_sub[yr], errors="coerce").fillna(0).sum() > 0.05:
          d: dict[tuple[str, str], float] = {}
          for (_, r), v in zip(sub.iterrows(), vals):
            if pd.notna(v) and v >= 0:
              d[(str(r["End use"]).strip(), prod_map[str(r["Product"]).strip()])] = float(v) * GJ_PER_PJ
          res_map[iso] = (int(str(yr).strip()), d)
          break
    return res_map

  res_by_iso = extract_country_fuel_enduse(df_res, "Total residential")
  ser_by_iso = extract_country_fuel_enduse(df_ser, "Total services")
  donor_records = load_eurostat_hhq_donors()

  out: dict[str, dict[str, object]] = {}
  for iso, (yr_r, d_r) in res_by_iso.items():
    if iso in existing or iso in ("USA", "CHN"):
      continue
    r_sh_tot = d_r.get(("Space heating", "total"), 0.0)
    r_wh_tot = d_r.get(("Water heating", "total"), 0.0)
    if r_sh_tot <= 0.0 and r_wh_tot <= 0.0:
      continue

    # Compute country's weighted thermal conversion efficiency from its own residential fuel mix
    th_weights_sh = [
        (d_r.get(("Total residential", "gas"), 0.0), EFF_SH["gas"]),
        (d_r.get(("Total residential", "oil"), 0.0), EFF_SH["oil"]),
        (d_r.get(("Total residential", "coal"), 0.0), EFF_SH["coal"]),
        (d_r.get(("Total residential", "bio"), 0.0), EFF_SH["bio"]),
        (d_r.get(("Total residential", "heat"), 0.0), EFF_SH["heat"]),
        (0.25 * d_r.get(("Total residential", "elec"), 0.0), EFF_SH["elec"]),
    ]
    w_sum = sum(w for w, _ in th_weights_sh)
    eta_r_sh = (sum(w * e for w, e in th_weights_sh) / w_sum) if w_sum > 0 else 0.90
    eta_r_wh = min(0.95, eta_r_sh * 0.94)

    sh_u = r_sh_tot * eta_r_sh
    wh_u = r_wh_tot * eta_r_wh
    gas_tot = d_r.get(("Total residential", "gas"), 0.0)

    if iso in ser_by_iso:
      _, d_s = ser_by_iso[iso]
      gas_tot += d_s.get(("Total services", "gas"), 0.0)
      s_sh_tot = d_s.get(("Space heating", "total"), 0.0)
      s_wh_tot = d_s.get(("Water heating", "total"), 0.0)
      if s_sh_tot > 0 or s_wh_tot > 0:
        sh_u += s_sh_tot * eta_r_sh
        wh_u += s_wh_tot * eta_r_wh
      else:
        # Estimate Services SH & WH from country's reported Total services fuel breakdown
        r_tot_all = max(1.0, d_r.get(("Total residential", "total"), 1.0))
        r_sh_intensity = min(0.85, r_sh_tot / r_tot_all)
        for fk in ("gas", "oil", "coal", "bio"):
          tot_sf = d_s.get(("Total services", fk), 0.0)
          sh_u += tot_sf * min(0.75, 1.15 * r_sh_intensity) * EFF_SH[fk]
          wh_u += tot_sf * 0.14 * EFF_WH[fk]
        sh_u += d_s.get(("Total services", "heat"), 0.0) * (0.82 if r_sh_tot > 0 else 0.0) * EFF_SH["heat"]
        wh_u += d_s.get(("Total services", "heat"), 0.0) * 0.15 * EFF_WH["heat"]
        sh_u += d_s.get(("Total services", "elec"), 0.0) * (0.12 * r_sh_intensity) * EFF_SH["elec"]
        wh_u += d_s.get(("Total services", "elec"), 0.0) * 0.04 * EFF_WH["elec"]

    out[iso] = {
        "iso3": iso, "cal_year": yr_r,
        "sh_useful_gj": sh_u, "wh_useful_gj": wh_u,
        "has_gas": gas_tot > 1e5,
        "official_enduse": True,
        "cal_source": f"IEA_EEI_June2026_{yr_r}",
    }
  return out, donor_records


def load_unsd_buildings_fuels(m49_to_iso: dict[str, str]) -> dict[str, dict[str, object]]:
  """Parse UNSD Households (1235) + Services (1231) fuels (TJ -> GJ), stripping traditional solid biomass in non-OECD."""
  # Map UNSD 4-digit commodity code -> fuel group
  def classify_comm(code: str) -> str | None:
    if code in ("7000", "E7000"):
      return "elec"
    if code in ("8000", "H8000"):
      return "heat"
    if code in ("3000", "0350", "0360", "4630"):  # Natural gas, coke/gasworks gas, LPG
      return "gas_lpg" if code == "4630" else "gas"
    if code.startswith(("01", "02", "03", "11")):
      return "coal"
    if code.startswith("4"):  # Kerosenes, gasoil/diesel, fuel oil
      return "oil"
    if code.startswith("51"):  # Fuelwood, charcoal, animal/vegetal waste
      return "trad_bio"
    if code.startswith("52") or code.startswith("53"):  # Liquid biofuels, biogases
      return "bio"
    return None

  xml_path = DATA_RAW / "sinks/2026-11-13/unsd/households_services_2016_2024.xml"
  by_iso_yr: dict[tuple[str, int], dict[str, float]] = {}

  for _, elem in ET.iterparse(xml_path, events=("end",)):
    if elem.tag.endswith("Series"):
      area = elem.attrib.get("REF_AREA", "")
      iso = m49_to_iso.get(area, m49_to_iso.get(area.lstrip("0"), ""))
      tr = elem.attrib.get("TRANSACTION", "")
      comm = elem.attrib.get("COMMODITY", "")
      umeas = elem.attrib.get("UNIT_MEASURE", "")
      fgrp = classify_comm(comm)
      if iso and tr in ("1235", "1231") and fgrp is not None:
        for obs in elem:
          if obs.tag.endswith("Obs"):
            yr_s = obs.attrib.get("TIME_PERIOD", "0")
            val_s = obs.attrib.get("OBS_VALUE", "")
            if not yr_s.isdigit() or not val_s:
              continue
            yr = int(yr_s)
            if yr < 2019:
              continue
            val = float(val_s)
            umult = int(obs.attrib.get("UNIT_MULT", "0"))
            cf = float(obs.attrib.get("CONVERSION_FACTOR", "1.0") or 1.0)
            # Convert to GJ:
            if umeas == "TJ":
              gj = val * (10.0 ** umult) * GJ_PER_TJ
            elif umeas == "GWHR":
              gj = val * (10.0 ** umult) * 3600.0
            elif umeas in ("TN", "M3"):
              # In UNSD Energy Statistics: TN with UNIT_MULT=3 is thousand metric tons (kt);
              # CONVERSION_FACTOR is TJ per thousand metric tons (e.g. 47.3 TJ/kt for LPG, 29.3 TJ/kt for coal)!
              gj = val * (10.0 ** (umult - 3)) * cf * GJ_PER_TJ
            else:
              continue
            if gj > 0:
              d = by_iso_yr.setdefault((iso, yr), {})
              key = f"{tr}_{fgrp}"
              d[key] = d.get(key, 0.0) + gj
      elem.clear()

  # Also add derived heat B47_123 (Buildings/Other sectors) from existing_chp_dh_national.csv
  df_dh = pd.read_csv(DATA_PROCESSED / "sources_2025/existing_chp_dh_national.csv")
  dh_bldg_gj = dict(zip(df_dh["iso3"], df_dh["buildings_dh_consumption_ej_yr"] * 1e9))

  out: dict[str, dict[str, object]] = {}
  all_isos = sorted({k[0] for k in by_iso_yr} | set(dh_bldg_gj.keys()))
  for iso in all_isos:
    cand_yrs = sorted([y for (i, y) in by_iso_yr if i == iso and y <= 2025], reverse=True)
    best_yr = cand_yrs[0] if cand_yrs else 2023
    fuels = dict(by_iso_yr.get((iso, best_yr), {}))
    # Ensure derived heat is captured
    dh_gj = dh_bldg_gj.get(iso, 0.0)
    cur_ht = fuels.get("1235_heat", 0.0) + fuels.get("1231_heat", 0.0)
    if dh_gj > cur_ht:
      fuels["1235_heat"] = dh_gj * 0.70
      fuels["1231_heat"] = dh_gj * 0.30
    out[iso] = {"cal_year": best_yr, "fuels": fuels}
  return out


def compile_2025_prices_plis_and_ef(all_isos: list[str]) -> dict[str, dict[str, object]]:
  """Compile 2025 USD pre-tax & retail electricity/gas prices, WB PPP capital PLIs, & Ember 2025 grid EF."""
  # 1. World Bank US GDP deflator (to inflate 2019/2021 USD to 2025 USD) & PPP / FX for capital PLIs
  def load_wb_indicator(fname: str) -> dict[tuple[str, int], float]:
    with open(DATA_RAW / "supplements_2025/worldbank" / fname) as f:
      j = json.load(f)
    res = {}
    for x in j[1]:
      iso = str(x.get("countryiso3code") or "").strip()
      yr_s = str(x.get("date") or "")
      val = x.get("value")
      if len(iso) == 3 and yr_s.isdigit() and val is not None:
        res[(iso, int(yr_s))] = float(val)
    return res

  wb_defl = load_wb_indicator("NY.GDP.DEFL.ZS_2015_2025.json")
  wb_ppp = load_wb_indicator("PA.NUS.PPP_2015_2025.json")
  wb_fx = load_wb_indicator("PA.NUS.FCRF_2015_2025.json")

  us_defl_2025 = wb_defl.get(("USA", 2025), wb_defl.get(("USA", 2024), 128.0))
  us_defl_2019 = wb_defl.get(("USA", 2019), 104.0)
  us_defl_2021 = wb_defl.get(("USA", 2021), 110.0)
  infl_2019_to_2025 = us_defl_2025 / us_defl_2019
  infl_2021_to_2025 = us_defl_2025 / us_defl_2021

  EUROZONE_ISO3 = {
      "AUT", "BEL", "HRV", "CYP", "EST", "FIN", "FRA", "DEU", "GRC", "IRL",
      "ITA", "LVA", "LTU", "LUX", "MLT", "NLD", "PRT", "SVK", "SVN", "ESP",
  }

  # Price Level Index PLI_c = (PPP_c / FX_c) relative to DEU (DEA catalog European benchmark)
  def get_latest_pli(iso: str) -> float | None:
    for yr in range(2025, 2017, -1):
      p = wb_ppp.get((iso, yr))
      x = wb_fx.get((iso, yr))
      if x is None and iso in EUROZONE_ISO3:
        x = wb_fx.get(("EMU", yr), 0.8849689584)
      if p and x and p > 0 and x > 0:
        return p / x
    return None

  deu_pli = get_latest_pli("DEU") or 0.82

  # 2. Ember 2025 grid carbon intensity [tCO2/MWh == MtCO2e / TWh] & total generation [MWh]
  emb = pd.read_csv(DATA_RAW / "supplements_2025/ember/release_generation_yearly_global.csv")
  emb_c = emb[(emb["Area type"] == "Country or economy") & emb["ISO 3 code"].notna()].copy()
  emb_c["Generation (TWh)"] = pd.to_numeric(emb_c["Generation (TWh)"], errors="coerce").fillna(0.0)
  emb_c["Emissions (MtCO2e)"] = pd.to_numeric(emb_c["Emissions (MtCO2e)"], errors="coerce").fillna(0.0)
  emb_agg = (
      emb_c.groupby(["ISO 3 code", "Year"], as_index=False)[["Generation (TWh)", "Emissions (MtCO2e)"]]
      .sum()
      .sort_values("Year", ascending=False)
  )
  ef_2025_map: dict[str, float] = {}
  gen_mwh_map: dict[str, float] = {}
  for _, r in emb_agg.iterrows():
    iso = str(r["ISO 3 code"]).strip()
    gen_twh = float(r["Generation (TWh)"])
    emi_mt = float(r["Emissions (MtCO2e)"])
    if gen_twh > 0.0:
      if iso not in ef_2025_map:
        ef_2025_map[iso] = emi_mt / gen_twh  # MtCO2e / TWh == tCO2 / MWh
      if iso not in gen_mwh_map:
        gen_mwh_map[iso] = gen_twh * 1e6

  # 3. IMF Explicit Subsidies (Electricity IMF_FFS_ECGFTET & Natural Gas IMF_FFS_ECGFTEN, in constant 2021 USD -> 2025 USD)
  imf_w = pd.read_csv(DATA_RAW / "supplements_2025/imf_prices/WB_Data360_IMF_FFS_WIDEF.csv")
  imf_sub = imf_w[
      imf_w["INDICATOR"].isin(["IMF_FFS_ECGFTET", "IMF_FFS_ECGFTEN"])
      & (imf_w["UNIT_MEASURE"] == "USD_K_2021")
  ].copy()
  yr_cols_imf = sorted([c for c in imf_sub.columns if str(c).isdigit() and int(c) <= 2025], key=int, reverse=True)
  imf_el_sub_usd: dict[str, float] = {}
  imf_ng_sub_usd: dict[str, float] = {}
  for _, r in imf_sub.iterrows():
    iso = str(r["REF_AREA"]).strip()
    ind = str(r["INDICATOR"]).strip()
    umult = int(pd.to_numeric(r.get("UNIT_MULT", 0), errors="coerce") or 0)
    # USD_K_2021 in WB Data360 is US dollars at constant 2021 prices (UNIT_MULT=0 -> raw USD)
    for yc in yr_cols_imf:
      v = pd.to_numeric(r[yc], errors="coerce")
      if pd.notna(v) and v >= 0:
        usd_2025 = float(v) * (10.0 ** umult) * infl_2021_to_2025
        if ind == "IMF_FFS_ECGFTET":
          imf_el_sub_usd[iso] = usd_2025
        else:
          imf_ng_sub_usd[iso] = usd_2025
        break

  # 4. Eurostat 2025 electricity (nrg_pc_204/205, EUR/kWh -> USD/MWh) & gas (nrg_pc_202/203, EUR/GJ -> USD/GJ)
  def read_eurostat_price(fname: str, unit_req: str, scale_to_target: float, pref_band: str = "") -> dict[tuple[str, str], float]:
    out_p: dict[tuple[str, str], float] = {}
    with gzip.open(DATA_RAW / "supplements_2025/eurostat" / fname, "rt") as f:
      hdr = f.readline().rstrip("\r\n").split("\t")
      sem_cols = [(i, c.strip()) for i, c in enumerate(hdr[1:], start=1) if "2025" in c or "2024" in c or "2023" in c]
      sem_cols.sort(key=lambda x: x[1], reverse=True)
      vals_pref: dict[tuple[str, str], list[float]] = {}
      vals_all: dict[tuple[str, str], list[float]] = {}
      for ln in f:
        if f",{unit_req}," not in ln or ",EUR," not in ln:
          continue
        parts = ln.rstrip("\r\n").split("\t")
        mp = parts[0].split(",")
        if len(mp) != 7:
          continue
        _, _, band, unit, tax, curr, geo = mp
        if geo not in EUROSTAT_GEO_MAP or tax not in ("X_TAX", "I_TAX"):
          continue
        iso = EUROSTAT_GEO_MAP[geo]
        for ci, _ in sem_cols:
          if ci < len(parts):
            rv = re.sub(r"[^0-9.\-]", "", parts[ci])
            if rv and rv not in ("", "-", "."):
              fv = float(rv)
              if fv > 0:
                vals_all.setdefault((iso, tax), []).append(fv * scale_to_target)
                if pref_band and band == pref_band:
                  vals_pref.setdefault((iso, tax), []).append(fv * scale_to_target)
                break
      for k, vlist in vals_all.items():
        out_p[k] = float(np.median(vals_pref[k])) if k in vals_pref else float(np.median(vlist))
    return out_p

  eu_el_hh = read_eurostat_price("nrg_pc_204.tsv.gz", "KWH", 1000.0 * USD_PER_EUR_2025, "KWH2500-4999")
  eu_el_nh = read_eurostat_price("nrg_pc_205.tsv.gz", "KWH", 1000.0 * USD_PER_EUR_2025, "MWH500-1999")
  eu_ng_hh = read_eurostat_price("nrg_pc_202.tsv.gz", "GJ_GCV", 1.108 * USD_PER_EUR_2025, "GJ20-199")
  eu_ng_nh = read_eurostat_price("nrg_pc_203.tsv.gz", "GJ_GCV", 1.108 * USD_PER_EUR_2025, "GJ10000-99999")

  # 5. World Bank DB1619 Electricity Tariffs (185+ countries, US cents/kWh -> 2025 USD/MWh via local CPI/FX)
  wb_cpi = load_wb_indicator("FP.CPI.TOTL_2015_2025.json")
  with open(DATA_RAW / "supplements_2025/worldbank/IC.ELC.PRI.KH.DB1619.json") as f:
    j_db = json.load(f)
  wb_el_usd_mwh: dict[str, float] = {}
  tmp_db: dict[str, dict[int, list[float]]] = {}
  for x in j_db[1]:
    iso = str(x.get("countryiso3code") or "").strip()
    yr_s = str(x.get("date") or "")
    val = x.get("value")
    if len(iso) == 3 and yr_s.isdigit() and val is not None and float(val) > 0:
      tmp_db.setdefault(iso, {}).setdefault(int(yr_s), []).append(float(val))
  for iso, ydict in tmp_db.items():
    latest_y = max(ydict.keys())
    cents_kwh = float(np.mean(ydict[latest_y]))
    # Adjust from survey year USD to 2025 USD via local CPI change divided by FX change
    cpi_0 = wb_cpi.get((iso, latest_y))
    cpi_1 = wb_cpi.get((iso, 2025)) or wb_cpi.get((iso, 2024)) or wb_cpi.get((iso, 2023))
    fx_0 = wb_fx.get((iso, latest_y))
    fx_1 = wb_fx.get((iso, 2025)) or wb_fx.get((iso, 2024)) or wb_fx.get((iso, 2023))
    if cpi_0 and cpi_1 and fx_0 and fx_1 and cpi_0 > 0 and fx_0 > 0 and fx_1 > 0:
      adj = float(np.clip((cpi_1 / cpi_0) * (fx_0 / fx_1), 0.75, 1.30))
    else:
      adj = 1.10
    wb_el_usd_mwh[iso] = cents_kwh * 10.0 * adj

  # 6. World Bank Pink Sheet 2025 Regional Wholesale Natural Gas Benchmarks ($/MMBtu HHV -> 2025 USD/GJ LHV)
  wb_cmo = openpyxl.load_workbook(
      DATA_RAW / "supplements_2025/worldbank/CMO-Historical-Data-Monthly.xlsx",
      read_only=True, data_only=True,
  )
  ws_cmo = wb_cmo["Monthly Prices"]
  cmo_rows = list(ws_cmo.iter_rows(values_only=True))
  hdr_names = [str(x or "") for x in cmo_rows[4]]
  idx_us = next(i for i, s in enumerate(hdr_names) if "Natural gas, US" in s)
  idx_eu = next(i for i, s in enumerate(hdr_names) if "Natural gas, Europe" in s)
  idx_jp = next(i for i, s in enumerate(hdr_names) if "Liquefied natural gas, Japan" in s)
  us_25, eu_25, jp_25 = [], [], []
  for r in cmo_rows[6:]:
    if r[0] and str(r[0]).startswith("2025M"):
      if isinstance(r[idx_us], (int, float)): us_25.append(float(r[idx_us]))
      if isinstance(r[idx_eu], (int, float)): eu_25.append(float(r[idx_eu]))
      if isinstance(r[idx_jp], (int, float)): jp_25.append(float(r[idx_jp]))
  wb_cmo.close()
  hh_2025_gj = float(np.mean(us_25)) / GJ_LHV_PER_MMBTU_HHV
  ttf_2025_gj = float(np.mean(eu_25)) / GJ_LHV_PER_MMBTU_HHV
  jkm_2025_gj = float(np.mean(jp_25)) / GJ_LHV_PER_MMBTU_HHV
  # Median European city distribution + retail delivery margin from Eurostat X_TAX minus TTF wholesale:
  eu_dist_margin_gj = max(4.5, float(np.median([v for (_, t), v in eu_ng_hh.items() if t == "X_TAX"])) - ttf_2025_gj)

  GAS_EXPORTERS = {
      "RUS", "IRN", "QAT", "TKM", "UZB", "KAZ", "AZE", "DZA", "SAU", "ARE",
      "OMN", "BHR", "KWT", "IRQ", "LBY", "NGA", "EGY", "NOR", "CAN", "AUS",
      "MYS", "IDN", "BRN", "TTO", "BOL", "ARG", "VEN", "PER",
  }

  # US 2025 EIA exact prices from T09.08 & T09.10 verified above:
  # Residential 17.30 c/kWh, Commercial 13.41 c/kWh -> 60/40 building blend = 157.4 USD/MWh retail, 147.1 pre-tax
  # Residential gas $15.34/Mcf, Commercial $10.98/Mcf -> 60/40 blend = $13.60/Mcf = 13.76 USD/GJ_LHV retail, 12.86 pre-tax
  US_EL_RETAIL, US_EL_PRETAX = 157.44, 147.14
  US_NG_RETAIL, US_NG_PRETAX = 13.76, 12.86

  info: dict[str, dict[str, object]] = {}
  for iso in all_isos:
    # A. Electricity prices [2025 USD/MWh]
    p_el_pre, p_el_ret, p_src = np.nan, np.nan, "missing"
    if iso == "USA":
      p_el_pre, p_el_ret, p_src = US_EL_PRETAX, US_EL_RETAIL, "US_EIA_MER_2025"
    elif (iso, "X_TAX") in eu_el_hh or (iso, "I_TAX") in eu_el_hh:
      hh_x = eu_el_hh.get((iso, "X_TAX"), eu_el_nh.get((iso, "X_TAX"), np.nan))
      hh_i = eu_el_hh.get((iso, "I_TAX"), hh_x * 1.25 if pd.notna(hh_x) else np.nan)
      if pd.isna(hh_x) and pd.notna(hh_i):
        hh_x = hh_i / 1.25
      p_el_pre, p_el_ret, p_src = float(hh_x), float(hh_i), "Eurostat_nrg_pc_204_2025"
    elif iso in wb_el_usd_mwh:
      ret = wb_el_usd_mwh[iso]
      sub_mwh = imf_el_sub_usd.get(iso, 0.0) / max(1e5, gen_mwh_map.get(iso, 1e9))
      sub_mwh = min(sub_mwh, 120.0)  # cap noisy micro-grid division
      if sub_mwh > 2.0:
        # Subsidized tariff: retail is below supply cost; supply cost = retail + explicit subsidy
        p_el_ret = ret
        p_el_pre = ret + sub_mwh
      else:
        p_el_ret = ret
        p_el_pre = ret / 1.12  # strip standard ~12% VAT/sales tax
      p_src = "WorldBank_DB1619_IMF_SubsidyAdj_2025"

    # B. Natural gas prices [2025 USD/GJ LHV]
    if iso == "USA":
      p_ng_pre, p_ng_ret = US_NG_PRETAX, US_NG_RETAIL
    elif (iso, "X_TAX") in eu_ng_hh or (iso, "I_TAX") in eu_ng_hh:
      gx = eu_ng_hh.get((iso, "X_TAX"), eu_ng_nh.get((iso, "X_TAX"), ttf_2025_gj + eu_dist_margin_gj))
      gi = eu_ng_hh.get((iso, "I_TAX"), gx * 1.22)
      if pd.isna(gx) and pd.notna(gi):
        gx = gi / 1.22
      p_ng_pre, p_ng_ret = float(gx), float(gi)
    else:
      # Regional hub + local distribution margin (scaled by PLI; cap non-OECD peg distortions at 0.85)
      pli_raw = get_latest_pli(iso)
      max_pli = 1.50 if iso in OECD_ISO3 or iso in ("SGP", "HKG", "MAC", "ARE", "QAT", "ISR") else 0.85
      pli_rel = float(np.clip((pli_raw / deu_pli) if pli_raw is not None else 0.65, 0.35, max_pli))
      local_dist = eu_dist_margin_gj * pli_rel
      if iso in ("CAN", "MEX"):
        hub = hh_2025_gj
      elif iso in ("JPN", "KOR", "TWN", "CHN", "HKG", "SGP", "IND", "PAK", "BGD", "THA", "VNM"):
        hub = jkm_2025_gj * (0.60 if iso in GAS_EXPORTERS else 1.0)
      elif iso in GAS_EXPORTERS:
        hub = hh_2025_gj * 0.85
      else:
        hub = ttf_2025_gj * 0.85
      p_ng_pre = float(hub + local_dist)
      p_ng_ret = float(p_ng_pre * 1.10 if imf_ng_sub_usd.get(iso, 0.0) < 1e8 else max(2.0, p_ng_pre * 0.65))

    # C. Capital scaling factors (World Bank PPP / FX relative to DEU)
    pli_c = get_latest_pli(iso)
    max_cap_pli = 1.60 if iso in OECD_ISO3 or iso in ("SGP", "HKG", "MAC", "ARE", "QAT", "ISR") else 0.85
    pli_ratio = float(np.clip((pli_c / deu_pli) if pli_c is not None else np.nan, 0.30, max_cap_pli)) if pli_c is not None else np.nan
    cap_civ = pli_ratio if pd.notna(pli_ratio) else np.nan
    cap_eq = (0.50 + 0.50 * cap_civ) if pd.notna(cap_civ) else np.nan

    # D. Grid EF 2025 & 2050 (IEA WEO APS/NZE trajectory) & R4 2050 envelope retrofit multiplier
    ef25 = ef_2025_map.get(iso, np.nan)
    if pd.notna(ef25):
      if iso in OECD_ISO3:
        ef50 = min(ef25 * 0.10, 0.020)
      elif iso in ("CHN", "BRA", "CHL", "URY", "CRI"):
        ef50 = min(ef25 * 0.12, 0.035)
      else:
        ef50 = min(ef25 * 0.18, 0.060)
    else:
      ef50 = np.nan

    if iso in EU27_MAP.values() or iso in ("GBR", "NOR", "CHE", "ISL"):
      retro50 = 0.68  # -32% envelope K_c (EU renovation wave / van Heerden et al. 2025)
    elif iso == "CHN":
      retro50 = 0.72  # -28% stock average (IEA & Tsinghua 2024 China clean heating & efficiency)
    elif iso in OECD_ISO3:
      retro50 = 0.75  # -25% OECD buildings
    else:
      retro50 = 0.82  # -18% emerging economies

    info[iso] = {
        "p_elec_pretax_usd_per_mwh": p_el_pre,
        "p_elec_retail_usd_per_mwh": p_el_ret,
        "p_gas_pretax_usd_per_gj": p_ng_pre,
        "p_gas_retail_usd_per_gj": p_ng_ret,
        "price_source": p_src,
        "cap_scale_civil": cap_civ,
        "cap_scale_equip": cap_eq,
        "grid_ef_2025_tco2_per_mwh": ef25,
        "grid_ef_2050_tco2_per_mwh": ef50,
        "kc_retrofit_2050_mult": retro50,
    }
  return info


def main() -> None:
  sinks_dir = DATA_PROCESSED / "sinks_2025"
  print("1. Computing observed GSWP3-ERA5 annual HDD cache (2018..2025 & 2006-2025 baseline)...", flush=True)
  hdd_cache = compute_annual_hdd_cache()
  np.save(sinks_dir / "baseline_cell_hdd14_2006_2025.npy", hdd_cache[(0, 14.0)])

  df_rc = pd.read_parquet(sinks_dir / "ghsl_region_cell_volumes.parquet")
  name_to_iso, m49_to_iso = build_country_name_and_m49_maps()

  # Precompute regional V_25 and volume-weighted mean baseline HDD14
  hdd0_14 = hdd_cache[(0, 14.0)]
  df_rc["vhdd0_14"] = df_rc["v25_m3"] * hdd0_14[df_rc["cell_idx"].values]
  reg_base = df_rc.groupby(["region_id", "iso3"], as_index=False)[["v25_m3", "pop25", "vhdd0_14"]].sum()
  reg_base["mean_hdd14_0"] = reg_base["vhdd0_14"] / np.maximum(1.0, reg_base["v25_m3"])

  print("2. Loading Tier 1 (USA 9 divs, GBR, NZL), Tier 2 (EU27 JRC-IDEES), & Tier 3 (IEA EEI 61 countries)...", flush=True)
  t1 = load_tier1_us_uk_nz()
  t2 = load_tier2_eu27_jrc()
  known = {**t2, **t1}  # Tier 1 overrides where applicable (e.g. GBR 2025)
  t3, donor_list = load_tier3_iea_eei(name_to_iso, known)
  known.update(t3)
  print(f"   Official end-use calibrated regions (Tiers 1-3): {len(known)}", flush=True)

  print("3. Loading UNSD global Households + Services fuels & calibrating China (CHN_NORTH / CHN_REST) + Tier 5...", flush=True)
  unsd_map = load_unsd_buildings_fuels(m49_to_iso)

  # Attach baseline mean_hdd14 to donors for nearest-3 climate matching
  iso_hdd0 = reg_base.groupby("iso3")["mean_hdd14_0"].mean().to_dict()
  donors_valid = [d for d in donor_list if d["iso3"] in iso_hdd0 and iso_hdd0[d["iso3"]] > 20.0]

  # Tier 4: China subnational split (CHN_NORTH vs CHN_REST)
  if "CHN" in unsd_map:
    cn_yr = int(unsd_map["CHN"]["cal_year"])
    cn_f = unsd_map["CHN"]["fuels"]
    # Useful DH + coal + gas + oil + elec space & water heating across China
    dh_tot_u = (cn_f.get("1235_heat", 0.0) + cn_f.get("1231_heat", 0.0)) * EFF_SH["heat"]
    coal_tot = cn_f.get("1235_coal", 0.0) + cn_f.get("1231_coal", 0.0)
    gas_tot = cn_f.get("1235_gas", 0.0) + cn_f.get("1231_gas", 0.0)
    lpg_tot = cn_f.get("1235_gas_lpg", 0.0) + cn_f.get("1231_gas_lpg", 0.0)
    oil_tot = cn_f.get("1235_oil", 0.0) + cn_f.get("1231_oil", 0.0)
    elec_tot = cn_f.get("1235_elec", 0.0) + cn_f.get("1231_elec", 0.0)

    # Per IEA & Tsinghua (2024, Future of Heat Pumps in China):
    # Northern China accounts for 100% of statutory DH + ~82% of dispersed coal/gas space heating;
    # Southern/transitional China (Hot Summer Cold Winter) uses decentralized reversible AC/HP & gas/elec DHW.
    sh_north_u = (
        dh_tot_u * 0.94
        + coal_tot * 0.75 * EFF_SH["coal"]
        + gas_tot * 0.45 * EFF_SH["gas"]
        + elec_tot * 0.08 * EFF_SH["elec"]
    )
    sh_rest_u = (
        dh_tot_u * 0.02
        + coal_tot * 0.08 * EFF_SH["coal"]
        + gas_tot * 0.12 * EFF_SH["gas"]
        + elec_tot * 0.07 * EFF_SH["elec"]
    )
    wh_tot_u = (
        dh_tot_u * 0.04
        + gas_tot * 0.25 * EFF_WH["gas"]
        + lpg_tot * 0.35 * EFF_WH["gas"]
        + oil_tot * 0.20 * EFF_WH["oil"]
        + elec_tot * 0.12 * EFF_WH["elec"]
    )
    v_n = float(reg_base.loc[reg_base["region_id"] == "CHN_NORTH", "v25_m3"].sum())
    v_s = float(reg_base.loc[reg_base["region_id"] == "CHN_REST", "v25_m3"].sum())
    n_share_v = v_n / max(1.0, v_n + v_s)
    known["CHN_NORTH"] = {
        "iso3": "CHN", "cal_year": cn_yr,
        "sh_useful_gj": sh_north_u, "wh_useful_gj": wh_tot_u * n_share_v,
        "has_gas": True, "official_enduse": True, "cal_source": f"UNSD_{cn_yr}_IEA_Tsinghua2024_North",
    }
    known["CHN_REST"] = {
        "iso3": "CHN", "cal_year": cn_yr,
        "sh_useful_gj": sh_rest_u, "wh_useful_gj": wh_tot_u * (1.0 - n_share_v),
        "has_gas": True, "official_enduse": True, "cal_source": f"UNSD_{cn_yr}_IEA_Tsinghua2024_Rest",
    }

  # Tier 5: All remaining countries in UNSD via nearest-3 HDD14 donor end-use shares
  for _, r_reg in reg_base.iterrows():
    rid = str(r_reg["region_id"])
    iso = str(r_reg["iso3"])
    if rid in known or iso not in unsd_map:
      continue
    u_entry = unsd_map[iso]
    yr_u = int(u_entry["cal_year"])
    f_u = u_entry["fuels"]
    tot_comm_gj = sum(v for k, v in f_u.items() if not k.endswith("trad_bio"))
    if tot_comm_gj <= 1e3:
      continue

    hdd_c = float(r_reg["mean_hdd14_0"])
    # Nearest 3 donors in HDD14 space
    d_sorted = sorted(donors_valid, key=lambda d: abs(iso_hdd0[d["iso3"]] - hdd_c))[:3]

    def mean_share(sec_key: str, fk: str, default_v: float) -> float:
      vals = [d[sec_key][fk] for d in d_sorted if fk in d[sec_key]]
      return float(np.mean(vals)) if vals else default_v

    is_trop = hdd_c < 15.0
    sh_u, wh_u = 0.0, 0.0
    # Derived heat -> 85% SH, 15% WH (0% SH if tropical)
    ht_gj = f_u.get("1235_heat", 0.0) + f_u.get("1231_heat", 0.0)
    sh_u += (0.0 if is_trop else 0.85 * ht_gj) * EFF_SH["heat"]
    wh_u += (1.0 if is_trop else 0.15) * ht_gj * EFF_WH["heat"]

    for fk_u, fk_eff in [("gas", "gas"), ("gas_lpg", "gas"), ("oil", "oil"), ("coal", "coal"), ("bio", "bio"), ("elec", "elec")]:
      gj_tot = f_u.get(f"1235_{fk_u}", 0.0) + f_u.get(f"1231_{fk_u}", 0.0)
      if gj_tot <= 0:
        continue
      if fk_u == "gas_lpg":
        # LPG in developing countries is predominantly cooking + water heating, minor space heating
        s_sh = 0.0 if is_trop else min(0.20, 0.35 * mean_share("res_sh", "gas", 0.5))
        s_wh = 0.20
      elif fk_u == "elec":
        s_sh = 0.0 if is_trop else min(0.35, mean_share("res_sh", "elec", 0.08))
        s_wh = min(0.20, mean_share("res_wh", "elec", 0.10))
      else:
        s_sh = 0.0 if is_trop else mean_share("res_sh", fk_eff, 0.60)
        s_wh = mean_share("res_wh", fk_eff, 0.18)
      sh_u += gj_tot * s_sh * EFF_SH[fk_eff]
      wh_u += gj_tot * s_wh * EFF_WH[fk_eff]

    # Note: trad_bio (5110..5150) is intentionally excluded in non-OECD developing countries!
    if iso in OECD_ISO3:
      tb_gj = f_u.get("1235_trad_bio", 0.0) + f_u.get("1231_trad_bio", 0.0)
      sh_u += (0.0 if is_trop else 0.75 * tb_gj) * EFF_SH["bio"]
      wh_u += 0.10 * tb_gj * EFF_WH["bio"]

    gas_net = f_u.get("1235_gas", 0.0) + f_u.get("1231_gas", 0.0)
    known[rid] = {
        "iso3": iso, "cal_year": yr_u,
        "sh_useful_gj": sh_u, "wh_useful_gj": wh_u,
        "has_gas": gas_net > 1e5,
        "official_enduse": False,
        "cal_source": f"UNSD_{yr_u}_Nearest3ClimateDonor",
    }

  print("4. Compiling 2025 pre-tax & retail energy prices, WB PPP PLIs, & Ember 2025 grid carbon factors...", flush=True)
  price_info = compile_2025_prices_plis_and_ef(sorted(reg_base["iso3"].unique()))

  print("5. Solving exact (K_c, W_c) per region for 14 C (and 12 C / 16 C robustness thresholds)...", flush=True)
  rows = []
  for _, r_reg in reg_base.iterrows():
    rid = str(r_reg["region_id"])
    iso = str(r_reg["iso3"])
    v25 = float(r_reg["v25_m3"])
    pop25 = float(r_reg["pop25"])
    p_inf = price_info.get(iso, {})
    cal = known.get(rid)

    has_energy = cal is not None and (cal["sh_useful_gj"] + cal["wh_useful_gj"]) > 0 and v25 > 0
    has_prices = (
        pd.notna(p_inf.get("p_elec_pretax_usd_per_mwh", np.nan))
        and pd.notna(p_inf.get("cap_scale_civil", np.nan))
        and pd.notna(p_inf.get("grid_ef_2025_tco2_per_mwh", np.nan))
    )
    valid = bool(has_energy and has_prices)

    if has_energy and cal is not None:
      yr_c = min(2025, max(2018, int(cal["cal_year"])))
      sh_gj = float(cal["sh_useful_gj"])
      wh_gj = float(cal["wh_useful_gj"])
      sub_rc = df_rc[df_rc["region_id"] == rid]
      c_idxs = sub_rc["cell_idx"].values
      c_vols = sub_rc["v25_m3"].values

      vhdd_14 = float(np.sum(c_vols * hdd_cache[(yr_c, BAIT_THRESHOLD_CENTRAL_C)][c_idxs]))
      vhdd_12 = float(np.sum(c_vols * hdd_cache[(yr_c, BAIT_THRESHOLD_LOW_C)][c_idxs]))
      vhdd_16 = float(np.sum(c_vols * hdd_cache[(yr_c, BAIT_THRESHOLD_HIGH_C)][c_idxs]))
      vhdd0_14 = float(np.sum(c_vols * hdd_cache[(0, BAIT_THRESHOLD_CENTRAL_C)][c_idxs]))

      kc_14 = (sh_gj / vhdd_14) if (vhdd_14 > 0 and r_reg["mean_hdd14_0"] >= 5.0) else 0.0
      kc_12 = (sh_gj / vhdd_12) if (vhdd_12 > 0 and r_reg["mean_hdd14_0"] >= 5.0) else 0.0
      kc_16 = (sh_gj / vhdd_16) if (vhdd_16 > 0 and r_reg["mean_hdd14_0"] >= 5.0) else 0.0
      wc = wh_gj / (365.25 * v25)
      base_d_gj = kc_14 * vhdd0_14 + wc * 365.25 * v25
    else:
      yr_c = 0
      sh_gj, wh_gj, kc_14, kc_12, kc_16, wc, base_d_gj = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    rows.append({
        "region_id": rid,
        "iso3": iso,
        "valid_for_study": valid,
        "official_enduse": bool(cal["official_enduse"]) if cal else False,
        "cal_source": str(cal["cal_source"]) if cal else "excluded_missing_energy",
        "cal_year": yr_c,
        "v25_m3": v25,
        "pop25": pop25,
        "mean_hdd14_2006_2025": float(r_reg["mean_hdd14_0"]),
        "cal_sh_useful_gj": sh_gj,
        "cal_wh_useful_gj": wh_gj,
        "k_c_gj_per_m3_hdd": kc_14,
        "k_c_12c_gj_per_m3_hdd": kc_12,
        "k_c_16c_gj_per_m3_hdd": kc_16,
        "w_c_gj_per_m3_day": wc,
        "baseline_total_demand_gj_yr": base_d_gj if valid else 0.0,
        "has_gas": bool(cal["has_gas"]) if cal else False,
        **p_inf,
    })

  df_out = pd.DataFrame(rows).sort_values("baseline_total_demand_gj_yr", ascending=False).reset_index(drop=True)
  df_out.to_parquet(sinks_dir / "calibrated_region_parameters_2025.parquet", index=False)
  df_out.to_csv(sinks_dir / "calibrated_region_parameters_2025.csv", index=False)

  tot_v = float(df_out["v25_m3"].sum())
  kept_v = float(df_out.loc[df_out["valid_for_study"], "v25_m3"].sum())
  excl_v_pct = 100.0 * (tot_v - kept_v) / tot_v
  tot_D_ej = float(df_out.loc[df_out["valid_for_study"], "baseline_total_demand_gj_yr"].sum() / 1e9)
  off_D_ej = float(df_out.loc[df_out["valid_for_study"] & df_out["official_enduse"], "baseline_total_demand_gj_yr"].sum() / 1e9)

  receipt = {
      "n_regions_total": int(len(df_out)),
      "n_regions_valid": int(df_out["valid_for_study"].sum()),
      "n_countries_valid": int(df_out.loc[df_out["valid_for_study"], "iso3"].nunique()),
      "n_official_enduse_regions": int((df_out["valid_for_study"] & df_out["official_enduse"]).sum()),
      "included_building_volume_share_pct": 100.0 - excl_v_pct,
      "excluded_missing_data_volume_share_pct": excl_v_pct,
      "global_baseline_2006_2025_D_ej_yr": tot_D_ej,
      "official_enduse_subset_D_ej_yr": off_D_ej,
      "official_enduse_share_of_D_pct": 100.0 * off_D_ej / tot_D_ej if tot_D_ej > 0 else 0.0,
      "excluded_countries_top10_by_volume": (
          df_out[~df_out["valid_for_study"]][["iso3", "v25_m3", "cal_source", "price_source"]]
          .sort_values("v25_m3", ascending=False)
          .head(10)
          .to_dict("records")
      ),
  }
  (sinks_dir / "calibration_and_prices_receipt.json").write_text(json.dumps(receipt, indent=2))
  print("\n=== DONE: REGIONAL DEMAND CALIBRATION AND 2025 PRICES ===", flush=True)
  print(json.dumps(receipt, indent=2), flush=True)
  print("\nTop 15 Regions by Baseline Building Heat Demand D (EJ/yr):", flush=True)
  print(
      df_out[df_out["valid_for_study"]][
          ["region_id", "iso3", "cal_source", "baseline_total_demand_gj_yr", "p_elec_pretax_usd_per_mwh", "p_gas_pretax_usd_per_gj", "has_gas"]
      ]
      .assign(D_ej_yr=lambda d: d["baseline_total_demand_gj_yr"] / 1e9)
      .head(15)
      .to_string(index=False),
      flush=True,
  )


if __name__ == "__main__":
  main()
