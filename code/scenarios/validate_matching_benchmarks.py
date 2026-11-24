"""Compare regional demand and waste heat potentials against European and Chinese district heating benchmarks."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd

from code.common.paths import DATA_PROCESSED

# Exact published country-level values from Manz et al. (2024, Renewable Energy 224, 120111)
# TableAnnex 2 (2020 DH area demand in GWh) & TableAnnex 9 (3GDH matched sum of potentials in GWh for WWTP, WtE, Industrial EH)
MANZ_2024_EU25_GWH: dict[str, dict[str, float]] = {
    "AUT": {"dh_area_2020": 38763, "dh_cov_2020": 21319, "wwtp": 5641, "wte": 3820, "ind_eh": 499},
    "BEL": {"dh_area_2020": 26027, "dh_cov_2020": 3904, "wwtp": 1767, "wte": 2190, "ind_eh": 1204},
    "BGR": {"dh_area_2020": 10160, "dh_cov_2020": 6503, "wwtp": 1647, "wte": 0, "ind_eh": 198},
    "CZE": {"dh_area_2020": 51729, "dh_cov_2020": 22761, "wwtp": 6438, "wte": 869, "ind_eh": 412},
    "DEU": {"dh_area_2020": 375321, "dh_cov_2020": 120103, "wwtp": 43337, "wte": 33153, "ind_eh": 3692},
    "DNK": {"dh_area_2020": 35273, "dh_cov_2020": 31040, "wwtp": 4889, "wte": 2392, "ind_eh": 41},
    "EST": {"dh_area_2020": 9105, "dh_cov_2020": 5281, "wwtp": 839, "wte": 0, "ind_eh": 20},
    "GRC": {"dh_area_2020": 14346, "dh_cov_2020": 4160, "wwtp": 1275, "wte": 0, "ind_eh": 198},
    "ESP": {"dh_area_2020": 65881, "dh_cov_2020": 1976, "wwtp": 20887, "wte": 1875, "ind_eh": 1135},
    "FIN": {"dh_area_2020": 57870, "dh_cov_2020": 36458, "wwtp": 13, "wte": 306, "ind_eh": 853},
    "FRA": {"dh_area_2020": 203160, "dh_cov_2020": 30474, "wwtp": 25093, "wte": 17545, "ind_eh": 1742},
    "HRV": {"dh_area_2020": 7702, "dh_cov_2020": 2850, "wwtp": 1213, "wte": 0, "ind_eh": 95},
    "HUN": {"dh_area_2020": 32249, "dh_cov_2020": 10642, "wwtp": 4817, "wte": 465, "ind_eh": 245},
    "IRL": {"dh_area_2020": 1760, "dh_cov_2020": 0, "wwtp": 468, "wte": 469, "ind_eh": 110},
    "ITA": {"dh_area_2020": 178312, "dh_cov_2020": 30313, "wwtp": 25265, "wte": 7077, "ind_eh": 2038},
    "LTU": {"dh_area_2020": 10987, "dh_cov_2020": 8570, "wwtp": 1852, "wte": 0, "ind_eh": 58},
    "LUX": {"dh_area_2020": 5266, "dh_cov_2020": 1527, "wwtp": 363, "wte": 195, "ind_eh": 52},
    "LVA": {"dh_area_2020": 11195, "dh_cov_2020": 6381, "wwtp": 1170, "wte": 0, "ind_eh": 10},
    "NLD": {"dh_area_2020": 46759, "dh_cov_2020": 12157, "wwtp": 6320, "wte": 10369, "ind_eh": 832},
    "POL": {"dh_area_2020": 130002, "dh_cov_2020": 66301, "wwtp": 16118, "wte": 16, "ind_eh": 800},
    "PRT": {"dh_area_2020": 5437, "dh_cov_2020": 1794, "wwtp": 1886, "wte": 636, "ind_eh": 140},
    "ROU": {"dh_area_2020": 29236, "dh_cov_2020": 12571, "wwtp": 5228, "wte": 0, "ind_eh": 404},
    "SWE": {"dh_area_2020": 66416, "dh_cov_2020": 57118, "wwtp": 6409, "wte": 6724, "ind_eh": 1089},
    "SVN": {"dh_area_2020": 3192, "dh_cov_2020": 1660, "wwtp": 208, "wte": 12, "ind_eh": 44},
    "SVK": {"dh_area_2020": 13582, "dh_cov_2020": 10051, "wwtp": 2759, "wte": 47, "ind_eh": 280},
}

EU27_ISOS = set(MANZ_2024_EU25_GWH.keys()) | {"CYP", "MLT"}

NORTHERN_CHINA_ADM1 = {
    "Beijing", "Tianjin", "Hebei", "Shanxi", "Nei Mongol", "Inner Mongolia",
    "Liaoning", "Jilin", "Heilongjiang", "Shandong", "Henan",
    "Shaanxi", "Gansu", "Qinghai", "Ningxia", "Ningxia Hui", "Xinjiang", "Xinjiang Uygur",
}


def main() -> None:
  scen_dir = DATA_PROCESSED / "scenarios_2025"
  out_dir = scen_dir / "validation_3_benchmarks"
  out_dir.mkdir(parents=True, exist_ok=True)

  df_c = pd.read_parquet(scen_dir / "all_scenarios_by_country.parquet")
  df_links = pd.read_parquet(scen_dir / "R1__obs__2006_2025_selected_links.parquet")
  df_sources = pd.read_parquet(DATA_PROCESSED / "sources_2025/sources_2025_streams.parquet")
  df_zones = pd.read_parquet(DATA_PROCESSED / "sinks_2025/urban_heat_zones.parquet")
  df_dh_nat = pd.read_csv(DATA_PROCESSED / "sources_2025/existing_chp_dh_national.csv")

  # -------------------------------------------------------------------------
  # 1. EU-27 Comparison vs Manz et al. (2024, Renewable Energy 224, 120111)
  # -------------------------------------------------------------------------
  eu_base = df_c[(df_c["scenario_id"] == "R1__obs__2006_2025") & (df_c["iso3"].isin(EU27_ISOS))].set_index("iso3")
  eu_4g = df_c[(df_c["scenario_id"] == "R4__S4_4gdh_symmetric__obs__obsclim__2006_2025") & (df_c["iso3"].isin(EU27_ISOS))].set_index("iso3")
  eu_src = df_sources[df_sources["iso3"].isin(EU27_ISOS)]

  eu_rows = []
  for iso, m_vals in sorted(MANZ_2024_EU25_GWH.items()):
    if iso not in eu_base.index:
      continue
    r0 = eu_base.loc[iso]
    s_iso = eu_src[eu_src["iso3"] == iso]
    q_wwtp_pj = float(s_iso.loc[s_iso["sector"] == "wastewater", "q3g_annual_gj"].sum() / 1e6)
    q_ind_dir3g_pj = float(s_iso.loc[
        s_iso["sector"].isin(["steel", "cement", "refining", "ethylene", "pulp_paper"])
        & (s_iso["upgrade_mode"] == "direct_hex"),
        "q3g_annual_gj"
    ].sum() / 1e6)
    q_ind_tot_pj = float(s_iso.loc[
        s_iso["sector"].isin(["steel", "cement", "refining", "ethylene", "pulp_paper"]),
        "q3g_annual_gj"
    ].sum() / 1e6)

    # Convert Manz GWh to PJ (1 GWh = 0.0036 PJ); note Manz WWTP is after HP (SPF=2.7 -> net WH = *(1 - 1/2.7) = *0.6296)
    manz_dh_area_pj = m_vals["dh_area_2020"] * 0.0036
    manz_dh_cov_pj = m_vals["dh_cov_2020"] * 0.0036
    manz_ex_gross_pj = (m_vals["wwtp"] + m_vals["wte"] + m_vals["ind_eh"]) * 0.0036
    manz_ex_net_pj = (m_vals["wwtp"] * (1.0 - 1.0 / 2.7) + m_vals["wte"] + m_vals["ind_eh"]) * 0.0036

    eu_rows.append({
        "iso3": iso,
        "manz_2020_dh_area_demand_pj": manz_dh_area_pj,
        "our_urban_zone_demand_pj": float(r0["D_urban_gj"] / 1e6),
        "manz_2020_existing_dh_pj": manz_dh_cov_pj,
        "manz_matched_excess_net_nonpower_pj": manz_ex_net_pj,
        "manz_matched_excess_gross_nonpower_pj": manz_ex_gross_pj,
        "our_Q_wwtp_pj": q_wwtp_pj,
        "our_Q_ind_direct_95c_pj": q_ind_dir3g_pj,
        "our_Q_ind_total_25c_pj": q_ind_tot_pj,
        "our_U_total_pj": float(r0["U_gj"] / 1e6),
        "our_P_total_3gdh_pj": float(r0["P_gj"] / 1e6),
        "our_P_nonfossil_or_ind_wwtp_pj": float((
            r0["P_steel_gj"] + r0["P_cement_gj"] + r0["P_refining_gj"]
            + r0["P_ethylene_gj"] + r0["P_pulp_paper_gj"] + r0["P_wastewater_gj"]
        ) / 1e6),
        "our_P_total_4gdh_sym_pj": float(eu_4g.loc[iso, "P_gj"] / 1e6) if iso in eu_4g.index else 0.0,
    })

  df_eu = pd.DataFrame(eu_rows)
  df_eu.to_csv(out_dir / "validation_3_eu27_vs_manz2024.csv", index=False)

  r_eu_urb = float(np.corrcoef(df_eu["manz_2020_dh_area_demand_pj"], df_eu["our_urban_zone_demand_pj"])[0, 1])
  r_eu_pot = float(np.corrcoef(df_eu["manz_matched_excess_net_nonpower_pj"], df_eu["our_U_total_pj"])[0, 1])
  r_eu_p_dh = float(np.corrcoef(df_eu["manz_2020_existing_dh_pj"], df_eu["our_P_total_3gdh_pj"])[0, 1])

  # EU-27 aggregate Table 2 comparison:
  eu_q_wwtp_pj = float(eu_src.loc[eu_src["sector"] == "wastewater", "q3g_annual_gj"].sum() / 1e6)
  eu_q_ind95_pj = float(eu_src.loc[
      eu_src["sector"].isin(["steel", "cement", "refining", "ethylene", "pulp_paper"])
      & (eu_src["upgrade_mode"] == "direct_hex"), "q3g_annual_gj"
  ].sum() / 1e6)
  eu_q_ind55_pj = float(eu_src.loc[
      eu_src["sector"].isin(["steel", "cement", "refining", "ethylene", "pulp_paper"])
      & (eu_src["upgrade_mode"] == "direct_hex"), "q4g_annual_gj"
  ].sum() / 1e6)

  # -------------------------------------------------------------------------
  # 2. Northern China Comparison vs Liu et al. (2024, Nature Energy)
  # -------------------------------------------------------------------------
  chn_zones = df_zones[df_zones["iso3"] == "CHN"].copy()
  chn_zones["is_north"] = chn_zones["adm1_name"].isin(NORTHERN_CHINA_ADM1) | (chn_zones["lat"] >= 33.0)
  north_zids = set(chn_zones.loc[chn_zones["is_north"], "zone_id"])

  chn_links = df_links[df_links["iso3"] == "CHN"].copy()
  chn_north_links = chn_links[chn_links["zone_id"].isin(north_zids)].copy()

  n_chn_D_urb_ej = float(chn_zones.loc[chn_zones["is_north"], "baseline_demand_gj_yr"].sum() / 1e9)
  n_chn_P_tot_ej = float(chn_north_links["P_net_waste_heat_gj_yr"].sum() / 1e9)
  n_chn_H_deliv_ej = float(chn_north_links["H_deliv_gj_yr"].sum() / 1e9)
  n_chn_P_power_ej = float(chn_north_links.loc[chn_north_links["sector"] == "power", "P_net_waste_heat_gj_yr"].sum() / 1e9)
  n_chn_P_ind_ej = float(chn_north_links.loc[
      chn_north_links["sector"].isin(["steel", "cement", "refining", "ethylene", "pulp_paper"]),
      "P_net_waste_heat_gj_yr"
  ].sum() / 1e9)
  n_chn_P_wwtp_ej = float(chn_north_links.loc[chn_north_links["sector"] == "wastewater", "P_net_waste_heat_gj_yr"].sum() / 1e9)

  chn_dh_row = df_dh_nat[df_dh_nat["iso3"] == "CHN"]
  chn_official_bldg_dh_ej = (
      float(chn_dh_row["buildings_dh_consumption_ej_yr"].iloc[0])
      if "buildings_dh_consumption_ej_yr" in chn_dh_row.columns and len(chn_dh_row) > 0
      else np.nan
  )
  chn_official_gross_dh_ej = (
      float(chn_dh_row["gross_dh_output_ej_yr"].iloc[0])
      if "gross_dh_output_ej_yr" in chn_dh_row.columns and len(chn_dh_row) > 0
      else np.nan
  )

  summary = {
      "eu27_vs_manz_2024": {
          "manz_table_annex2_2020_dh_area_demand_pj": 1429.732 * 3.6,
          "our_eu27_urban_zone_demand_pj": float(eu_base["D_urban_gj"].sum() / 1e6),
          "country_urban_demand_pearson_r": r_eu_urb,
          "manz_table2_wwtp_technical_net_wh_pj": (373.35 - 155.56) * 3.6,
          "our_eu27_wwtp_technical_Q_pj": eu_q_wwtp_pj,
          "manz_table2_ind_direct_95c_2050_pj": 19.91 * 3.6,
          "manz_table2_ind_direct_55c_2050_pj": 32.48 * 3.6,
          "our_eu27_ind_direct_95c_2025_pj": eu_q_ind95_pj,
          "our_eu27_ind_direct_55c_2025_pj": eu_q_ind55_pj,
          "manz_2020_existing_dh_delivered_pj": 505.916 * 3.6,
          "our_eu27_U_upper_bound_pj": float(eu_base["U_gj"].sum() / 1e6),
          "our_eu27_P_economic_3gdh_pj": float(eu_base["P_gj"].sum() / 1e6),
          "our_eu27_P_economic_4gdh_symmetric_pj": float(eu_4g["P_gj"].sum() / 1e6),
          "country_existing_dh_vs_our_P_pearson_r": r_eu_p_dh,
          "country_manz_excess_vs_our_U_pearson_r": r_eu_pot,
      },
      "northern_china_vs_liu_2024": {
          "liu_2024_city_shares_pct": {
              "abundant_chp_power_demand_share_pct": 62.0,
              "abundant_industrial_wh_demand_share_pct": 12.0,
              "local_wwtp_msw_biomass_share_pct": "<10%",
          },
          "our_northern_china_urban_D_ej_yr": n_chn_D_urb_ej,
          "our_china_total_P_ej_yr": float(df_c.loc[(df_c["scenario_id"] == "R1__obs__2006_2025") & (df_c["iso3"] == "CHN"), "P_gj"].sum() / 1e9),
          "our_northern_china_P_net_wh_ej_yr": n_chn_P_tot_ej,
          "our_northern_china_H_delivered_ej_yr": n_chn_H_deliv_ej,
          "our_northern_china_P_by_sector_ej_yr": {
              "power": n_chn_P_power_ej,
              "industry_5sectors": n_chn_P_ind_ej,
              "wastewater": n_chn_P_wwtp_ej,
          },
          "our_northern_china_sector_shares_of_P_pct": {
              "power_pct": 100.0 * n_chn_P_power_ej / max(1e-9, n_chn_P_tot_ej),
              "industry_pct": 100.0 * n_chn_P_ind_ej / max(1e-9, n_chn_P_tot_ej),
              "wastewater_pct": 100.0 * n_chn_P_wwtp_ej / max(1e-9, n_chn_P_tot_ej),
          },
          "official_china_existing_building_dh_ej_yr": chn_official_bldg_dh_ej,
          "official_china_existing_gross_dh_ej_yr": chn_official_gross_dh_ej,
          "mean_selected_pipe_length_km": float(chn_north_links["ell_km"].mean()) if len(chn_north_links) > 0 else 0.0,
      },
  }
  (out_dir / "validation_3_summary.json").write_text(json.dumps(summary, indent=2))
  print("=== VALIDATION 3 (MANZ 2024 EU-27 & LIU 2024 NORTHERN CHINA) COMPLETE ===", flush=True)
  print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
  main()
