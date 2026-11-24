#!/usr/bin/env python3
"""Generate Supplementary Figures 1–8 and compile Supplementary Data 1 and Source Data Excel workbooks."""
from __future__ import annotations
import os

from decimal import Decimal, ROUND_HALF_UP
import json
import re
import sys
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.text import Text
import numpy as np
import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
import pandas as pd
from pyproj import Transformer

warnings.filterwarnings("ignore")

ROOT = Path(os.environ.get("WASTE_HEAT_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "code/figures"))

from code.common.parameters import (  # noqa: E402
    DISCOUNT_RATE_CENTRAL,
    HP_INDUSTRIAL_POWER_2025,
    HP_INDUSTRIAL_POWER_2050,
    HP_WASTEWATER_2025,
    HP_WASTEWATER_2050,
    INDIV_ASHP_2025,
    INDIV_ASHP_2050,
    INDIV_GAS_BOILER_2025,
    INDIV_GAS_BOILER_2050,
    PEAKING_ELEC_BOILER_2025,
    PEAKING_ELEC_BOILER_2050,
    PEAKING_GAS_BOILER_2025,
    PEAKING_GAS_BOILER_2050,
    PIPE_CAPEX_COEF_EUR_PER_KM,
    PIPE_CAPEX_EXPONENT,
    PIPE_LIFETIME_YR,
    POWER_TECH_SPECS,
    SEENERGIES_FACTORS,
    SOURCE_HEX_STATION_CAPEX_EUR_PER_MW,
    STATION_LIFETIME_YR,
    USD_PER_EUR_2025,
)
from figure_utils import MM, DOUBLE_COL, check_typography, load_world_basemap  # noqa: E402
from build_extended_figures import (  # noqa: E402
    COL_APRICOT,
    COL_BLDG_SLATE,
    COL_CORAL,
    COL_DEM_BLUE,
    COL_EMERALD,
    COL_SAPPHIRE,
    COL_SKY_CYAN,
    COL_SSP2,
    COL_TECH_SLATE,
    FS_COMPACT,
    FS_UNIFIED,
    GCM_MARKER,
    GCM_ORDER,
    MINUS,
    W_IN,
    W_MM,
    configure_matplotlib_style,
    export_figure,
    clean_ax,
    cname,
    fm,
    lock_spines_above_data,
    panel_tag,
)

DATA_PROC = ROOT / "data/processed"
SCEN_DIR = DATA_PROC / "scenarios_2025"
SI_DIR = ROOT / "figures/si"
SRC_DIR = ROOT / "figures/source_data"
MS_DIR = ROOT / "manuscript"

F5_POWER = "#355C96"
F5_STEEL = "#C84632"
F5_REFINING = "#EFC06E"
F5_CEMENT = "#88C6D8"
F5_CHEM = "#19384A"

GCM_LABELS = dict([(k, lbl) for k, lbl, _ in GCM_ORDER])
GCM_COLORS = {
    "gfdl-esm4": COL_SAPPHIRE,
    "ipsl-cm6a-lr": COL_SKY_CYAN,
    "mpi-esm1-2-hr": "#586373",
    "mri-esm2-0": COL_APRICOT,
    "ukesm1-0-ll": COL_CORAL,
}


def _robinson_base():
    gdf_bg = load_world_basemap(ROOT)
    gdf_bg = gdf_bg[gdf_bg["GID_0"] != "ATA"].copy().to_crs("+proj=robin")
    trans = Transformer.from_crs("EPSG:4326", "+proj=robin", always_xy=True)
    lats_b = np.linspace(-58, 78, 181)
    lons_b = np.linspace(-180, 180, 301)
    bx = np.concatenate([np.full_like(lats_b, -180), lons_b, np.full_like(lats_b, 180), lons_b[::-1]])
    by = np.concatenate([lats_b, np.full_like(lons_b, 78), lats_b[::-1], np.full_like(lons_b, -58)])
    px, py = trans.transform(bx, by)
    return gdf_bg, trans, px, py


# ==============================================================================
# SUPPLEMENTARY FIGURE 1: Global Facility Inventory & Temperature Grades
# ==============================================================================
def build_supp_fig_1() -> dict:
    configure_matplotlib_style()
    H_MM = 140.0
    fig = plt.figure(figsize=(W_IN, H_MM * MM))

    df_s = pd.read_parquet(DATA_PROC / "sources_2025/sources_2025_streams.parquet")
    gdf_rob, trans, px, py = _robinson_base()

    ax_m = fig.add_axes([0.025, 0.455, 0.950, 0.515])
    ax_b = fig.add_axes([0.145, 0.085, 0.355, 0.310])
    ax_c = fig.add_axes([0.635, 0.085, 0.340, 0.310])

    ax_m.fill(px, py, facecolor="#FFFFFF", edgecolor="#333333", linewidth=0.52, zorder=0)
    for glat in [-30, 0, 30, 60]:
        gx, gy = trans.transform(np.linspace(-180, 180, 181), np.full(181, glat))
        ax_m.plot(gx, gy, color="#D8E0E8", lw=0.30, zorder=0.4)
    gdf_rob.plot(ax=ax_m, facecolor="#F3F5F8", edgecolor="#8A96A3", linewidth=0.24, zorder=1)

    fac = df_s.groupby(["facility_id", "sector"], as_index=False).agg(
        lat=("lat", "first"), lon=("lon", "first"), q_gj=("q3g_annual_gj", "sum")
    )
    fac["cat"] = np.where(fac["sector"] == "power", "Power plants", np.where(fac["sector"] == "wastewater", "Wastewater", "Heavy industry"))
    for cat_name, col_hex, z_o, al in [
        ("Wastewater", COL_EMERALD, 2.2, 0.52),
        ("Power plants", COL_SAPPHIRE, 2.8, 0.62),
        ("Heavy industry", COL_APRICOT, 3.4, 0.78),
    ]:
        sub = fac[fac["cat"] == cat_name]
        xf, yf = trans.transform(sub["lon"].to_numpy(), sub["lat"].to_numpy())
        sf = np.clip(np.sqrt(sub["q_gj"].to_numpy() / 1e6) * 2.1, 0.85, 19.0)
        ax_m.scatter(xf, yf, s=sf, color=col_hex, edgecolors="none", alpha=al, zorder=z_o, rasterized=True, label=f"{cat_name} ($n$ = {len(sub):,})")

    ax_m.set_xlim(px.min() * 1.02, px.max() * 1.02)
    ax_m.set_ylim(py.min() * 1.02, py.max() * 1.02)
    ax_m.set_xticks([])
    ax_m.set_yticks([])
    ax_m.set_xlabel("")
    ax_m.set_ylabel("")
    ax_m.axis("off")
    leg_m = ax_m.legend(loc="lower left", bbox_to_anchor=(0.085, 0.075), frameon=False, fontsize=FS_COMPACT, markerscale=1.4, handletextpad=0.3, labelspacing=0.20)
    leg_m.set_zorder(6)

    clean_ax(ax_b, "x")
    ind_s = df_s[df_s["sector"] != "power"].copy()
    sec_map = [
        ("refining", "Oil refining"),
        ("cement", "Cement"),
        ("steel", "Iron and steel"),
        ("ethylene", "Petrochemicals"),
        ("pulp_paper", "Pulp and paper"),
        ("wastewater", "Wastewater"),
    ]
    ind_rows = []
    for s_code, s_lbl in sec_map:
        sub = ind_s[ind_s["sector"] == s_code]
        q_dir3 = float(sub.loc[sub["upgrade_mode"] == "direct_hex", "q3g_annual_gj"].sum()) / 1e6
        q_dir4 = float(sub.loc[sub["upgrade_mode"] == "direct_hex", "q4g_annual_gj"].sum()) / 1e6
        q_tot = float(sub["q3g_annual_gj"].sum()) / 1e6
        ind_rows.append({
            "label": s_lbl,
            "ge95_pj": q_dir3,
            "mid55_95_pj": max(0.0, q_dir4 - q_dir3),
            "lt55_pj": max(0.0, q_tot - q_dir4),
            "tot": q_tot,
        })
    df_ib = pd.DataFrame(ind_rows).sort_values("tot", ascending=True).reset_index(drop=True)
    yb = np.arange(len(df_ib))
    ax_b.barh(yb, df_ib["ge95_pj"], height=0.60, color=COL_SAPPHIRE, edgecolor="none", zorder=3, label="≥95 °C direct")
    ax_b.barh(yb, df_ib["mid55_95_pj"], left=df_ib["ge95_pj"], height=0.60, color=COL_EMERALD, edgecolor="none", zorder=3, label="55–95 °C low temp. direct")
    ax_b.barh(yb, df_ib["lt55_pj"], left=df_ib["ge95_pj"] + df_ib["mid55_95_pj"], height=0.60, color=COL_APRICOT, edgecolor="none", zorder=3, label="<55 °C electric heat pump")
    for yi, r in df_ib.iterrows():
        if r["ge95_pj"] > 20 and r["mid55_95_pj"] > 20:
            ax_b.plot([r["ge95_pj"], r["ge95_pj"]], [yi - 0.30, yi + 0.30], color="white", lw=0.45, zorder=3.5)
        if (r["ge95_pj"] + r["mid55_95_pj"]) > 20 and r["lt55_pj"] > 20:
            xdiv = r["ge95_pj"] + r["mid55_95_pj"]
            ax_b.plot([xdiv, xdiv], [yi - 0.30, yi + 0.30], color="white", lw=0.45, zorder=3.5)
    ax_b.set_yticks(yb)
    ax_b.set_yticklabels(df_ib["label"], fontsize=FS_COMPACT)
    ax_b.set_xlabel("Non-power recoverable waste heat (PJ yr$^{-1}$)")
    ax_b.legend(loc="lower right", bbox_to_anchor=(1.02, 0.00), frameon=False, fontsize=FS_COMPACT, labelspacing=0.18, handlelength=0.95)
    lock_spines_above_data(ax_b)

    clean_ax(ax_c, "x")
    pwr_s = df_s[df_s["sector"] == "power"]
    pwr_map = [
        ("power_coal", "Coal steam"),
        ("power_nuclear", "Nuclear reactor"),
        ("power_gas", "Gas combined/steam"),
        ("power_biomass", "Biomass steam"),
        ("power_oil", "Oil steam"),
        ("power_waste", "Waste steam"),
    ][::-1]
    yc = np.arange(len(pwr_map))
    d_ej = [float(pwr_s[(pwr_s["subsector_detail"] == k) & (pwr_s["upgrade_mode"] == "direct_hex")]["q3g_annual_gj"].sum()) / 1e9 for k, _ in pwr_map]
    h_ej = [float(pwr_s[(pwr_s["subsector_detail"] == k) & (pwr_s["upgrade_mode"] != "direct_hex")]["q3g_annual_gj"].sum()) / 1e9 for k, _ in pwr_map]
    ax_c.barh(yc, d_ej, height=0.60, color=COL_SAPPHIRE, edgecolor="none", zorder=3, label="Cogeneration ≥95 °C")
    ax_c.barh(yc, h_ej, left=d_ej, height=0.60, color=COL_SKY_CYAN, edgecolor="none", zorder=3, label="Wet condensers 30–40 °C")
    for yi, (dv, hv) in enumerate(zip(d_ej, h_ej)):
        if dv > 0.2 and hv > 0.2:
            ax_c.plot([dv, dv], [yi - 0.30, yi + 0.30], color="white", lw=0.45, zorder=3.5)
    ax_c.set_yticks(yc)
    ax_c.set_yticklabels([lbl for _, lbl in pwr_map], fontsize=FS_COMPACT)
    ax_c.set_xlabel("Power plant recoverable waste heat (EJ yr$^{-1}$)")
    ax_c.legend(loc="lower right", bbox_to_anchor=(1.02, 0.00), frameon=False, fontsize=FS_COMPACT, labelspacing=0.18, handlelength=0.95)
    lock_spines_above_data(ax_c)

    panel_tag(fig, 0.015, 0.982, "a")
    panel_tag(fig, 0.015, 0.422, "b")
    panel_tag(fig, 0.515, 0.422, "c")
    return export_figure(fig, "Supplementary_Fig1_Facility_Inventory", out_root=SI_DIR)


# ==============================================================================
# SUPPLEMENTARY FIGURE 2: Climate Monthly Bias Alignment & Urban Demand
# ==============================================================================
def build_supp_fig_2(df_glob: pd.DataFrame) -> dict:
    configure_matplotlib_style()
    H_MM = 96.0
    fig = plt.figure(figsize=(W_IN, H_MM * MM))

    df_al = pd.read_csv(DATA_PROC / "climate_aligned/monthly_bias_alignment_summary_2006_2025.csv")
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.05, 1.32], left=0.148, right=0.980, bottom=0.155, top=0.885, wspace=0.28)

    ax_a = fig.add_subplot(gs[0, 0])
    clean_ax(ax_a, "x")
    df_al["gcm"] = [r.split("__")[0] for r in df_al["run"]]
    g_al = df_al.groupby("gcm", as_index=False).agg(
        m=("mean_bait_offset_c", "mean"), p05=("p05_bait_offset_c", "mean"), p95=("p95_bait_offset_c", "mean")
    )
    order_m = [k for k, _, _ in GCM_ORDER][::-1]
    g_al = g_al.set_index("gcm").loc[order_m].reset_index()
    ya = np.arange(len(g_al))
    ax_a.axvline(0.0, color="#000000", lw=0.70, ls="--", zorder=2)
    for yi, row in g_al.iterrows():
        ax_a.plot([row["p05"], row["p95"]], [yi, yi], color="#8E9FB1", lw=1.35, zorder=2.5)
        ax_a.plot([row["p05"], row["p05"]], [yi - 0.12, yi + 0.12], color="#8E9FB1", lw=0.95, zorder=2.5)
        ax_a.plot([row["p95"], row["p95"]], [yi - 0.12, yi + 0.12], color="#8E9FB1", lw=0.95, zorder=2.5)
    ax_a.scatter(g_al["m"], ya, s=28, color=COL_APRICOT, edgecolors="white", linewidths=0.35, zorder=4, label="Raw offset (5th–95th %ile)")
    ax_a.scatter(np.zeros(len(g_al)), ya, s=24, marker="D", color=COL_SAPPHIRE, edgecolors="white", linewidths=0.35, zorder=5, label="Aligned (<10$^{-6}$ °C)")
    ax_a.set_yticks(ya)
    ax_a.set_yticklabels([GCM_LABELS[m] for m in g_al["gcm"]], fontsize=FS_UNIFIED)
    ax_a.set_xlabel("2006–2025 BAIT offset (°C)")
    ax_a.legend(loc="lower center", bbox_to_anchor=(0.48, 1.01), ncol=1, frameon=False, fontsize=FS_COMPACT, labelspacing=0.16)

    ax_b = fig.add_subplot(gs[0, 1])
    clean_ax(ax_b, "y")
    r1 = df_glob[df_glob["group"] == "R1"].copy()
    r1["mid_year"] = 0.5 * (r1["start_year"] + r1["end_year"])
    obs = r1[r1["ssp"] == "obsclim"].sort_values("mid_year")
    ax_b.axvline(2015.5, color="#333333", ls=":", lw=0.65, zorder=2)
    ax_b.plot(obs["mid_year"], obs["D_urban_ej_yr"], color="#000000", lw=1.65, marker="o", ms=3.5, zorder=5, label="Observed (GSWP3-ERA5)")
    for ssp_c, lbl_s, scol, mk in [
        ("ssp126", "SSP1-2.6", COL_SAPPHIRE, "s"),
        ("ssp245", "SSP2-4.5", COL_SSP2, "^"),
        ("ssp370", "SSP3-7.0", COL_CORAL, "D"),
    ]:
        sub = r1[r1["ssp"] == ssp_c]
        grp = sub.groupby("mid_year", as_index=False).agg(m=("D_urban_ej_yr", "mean"), lo=("D_urban_ej_yr", "min"), hi=("D_urban_ej_yr", "max"))
        ax_b.fill_between(grp["mid_year"], grp["lo"], grp["hi"], color=scol, alpha=0.16, lw=0, zorder=2)
        ax_b.plot(grp["mid_year"], grp["m"], color=scol, lw=1.55, marker=mk, ms=3.4, markeredgecolor="white", markeredgewidth=0.35, zorder=4, label=lbl_s)
    ax_b.set_xlabel("20 year climate window midpoint")
    ax_b.set_ylabel("Urban heating area demand (EJ yr$^{-1}$)")
    ax_b.legend(loc="lower left", frameon=False, fontsize=FS_COMPACT, labelspacing=0.18)

    panel_tag(fig, 0.015, 0.968, "a")
    panel_tag(fig, 0.495, 0.968, "b")
    return export_figure(fig, "Supplementary_Fig2_Climate_Alignment", out_root=SI_DIR)


# ==============================================================================
# SUPPLEMENTARY FIGURE 3: Energy Price Comparisons & Grid Decarbonization
# ==============================================================================
def build_supp_fig_3() -> dict:
    configure_matplotlib_style()
    H_MM = 100.0
    fig = plt.figure(figsize=(W_IN, H_MM * MM))

    df_rp = pd.read_parquet(DATA_PROC / "sinks_2025/calibrated_region_parameters_2025.parquet")
    rp = df_rp[(df_rp["valid_for_study"]) & (df_rp["baseline_total_demand_gj_yr"] > 1e6)].copy()

    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.15, 1.0], left=0.095, right=0.980, bottom=0.148, top=0.915, wspace=0.30)

    ax_a = fig.add_subplot(gs[0, 0])
    clean_ax(ax_a, "y")
    has_g = rp[rp["has_gas"]]
    no_g = rp[~rp["has_gas"]]
    s_g = np.clip(np.sqrt(has_g["baseline_total_demand_gj_yr"].to_numpy() / 1e6) * 1.35, 8.0, 92.0)
    s_ng = np.clip(np.sqrt(no_g["baseline_total_demand_gj_yr"].to_numpy() / 1e6) * 1.35, 8.0, 72.0)

    ax_a.scatter(has_g["p_gas_pretax_usd_per_gj"], has_g["p_elec_pretax_usd_per_mwh"] / 3.6, s=s_g, color=COL_SAPPHIRE, alpha=0.70, edgecolors="white", linewidths=0.35, zorder=3, label="Gas grid regions")
    ax_a.scatter(np.zeros(len(no_g)) + 1.2, no_g["p_elec_pretax_usd_per_mwh"] / 3.6, s=s_ng, marker="^", color=COL_APRICOT, alpha=0.90, edgecolors="#333333", linewidths=0.30, zorder=3.5, label="Non-gas regions")

    gx = np.linspace(0.5, 40.0, 120)
    ax_a.plot(gx, gx * (2.99 / 1.01), color="#000000", ls="--", lw=0.75, zorder=2, label="2025 parity (SCOP 2.99)")
    ax_a.plot(gx, gx * (4.29 / 1.02), color=COL_CORAL, ls=":", lw=0.90, zorder=2.2, label="2050 parity (SCOP 4.29)")

    for iso_code, txt_l, (xt_p, yt_p), ha_p in [
        ("USA", "United States", (7.8, 52.0), "right"),
        ("CAN", "Canada", (6.8, 40.0), "right"),
        ("RUS", "Russia", (5.2, 26.0), "right"),
        ("CHN", "China", (26.5, 37.0), "left"),
        ("POL", "Poland", (27.5, 49.0), "left"),
        ("DEU", "Germany", (22.0, 108.0), "right"),
        ("ITA", "Italy", (25.0, 92.0), "right"),
    ]:
        sub = rp[rp["iso3"] == iso_code].sort_values("baseline_total_demand_gj_yr", ascending=False).head(1)
        if not sub.empty:
            r0 = sub.iloc[0]
            xv = float(r0["p_gas_pretax_usd_per_gj"]) if bool(r0["has_gas"]) else 1.2
            yv = float(r0["p_elec_pretax_usd_per_mwh"]) / 3.6
            xe = xt_p + (0.45 if ha_p == "right" else -0.45)
            ax_a.plot([xv, xe], [yv, yt_p], color="#000000", lw=0.45, zorder=4.5)
            ax_a.text(xt_p, yt_p, txt_l, ha=ha_p, va="center", fontsize=FS_COMPACT, color="#000000", zorder=5)

    ax_a.set_xlim(0.0, 42.0)
    ax_a.set_ylim(0.0, 122.0)
    ax_a.set_xlabel("2025 untaxed natural gas price (USD GJ$^{-1}$)")
    ax_a.set_ylabel("2025 untaxed electricity price\n(USD GJ$^{-1}$)")
    ax_a.legend(loc="upper left", bbox_to_anchor=(0.00, 1.02), frameon=False, fontsize=FS_COMPACT, markerscale=0.58, labelspacing=0.18, handlelength=1.2)

    ax_b = fig.add_subplot(gs[0, 1])
    clean_ax(ax_b, "x")
    maj_isos = ["POL", "CHN", "KAZ", "USA", "KOR", "RUS", "ITA", "DEU", "NLD", "GBR", "CAN", "SWE"]
    ef_df = df_rp[df_rp["iso3"].isin(maj_isos)].groupby("iso3", as_index=False)[["grid_ef_2025_tco2_per_mwh", "grid_ef_2050_tco2_per_mwh"]].mean()
    ef_df = ef_df.sort_values("grid_ef_2025_tco2_per_mwh", ascending=True).reset_index(drop=True)
    yb = np.arange(len(ef_df))
    for yi, r_e in ef_df.iterrows():
        ax_b.plot([r_e["grid_ef_2050_tco2_per_mwh"] * 1000, r_e["grid_ef_2025_tco2_per_mwh"] * 1000], [yi, yi], color="#8E9FB1", lw=1.25, zorder=2)
    ax_b.scatter(ef_df["grid_ef_2025_tco2_per_mwh"] * 1000, yb, s=24, color=COL_SAPPHIRE, edgecolors="white", linewidths=0.35, zorder=4, label="2025 grid")
    ax_b.scatter(ef_df["grid_ef_2050_tco2_per_mwh"] * 1000, yb, s=24, marker="D", color=COL_EMERALD, edgecolors="white", linewidths=0.35, zorder=5, label="2050 Net Zero grid")
    ax_b.set_yticks(yb)
    ax_b.set_yticklabels(["UK" if c == "GBR" else cname(c) for c in ef_df["iso3"]], fontsize=FS_COMPACT)
    ax_b.set_xlabel("Grid carbon intensity (gCO$_2$ kWh$^{-1}$)")
    ax_b.legend(loc="lower right", frameon=False, fontsize=FS_COMPACT, labelspacing=0.18)

    panel_tag(fig, 0.015, 0.968, "a")
    panel_tag(fig, 0.535, 0.968, "b")
    return export_figure(fig, "Supplementary_Fig3_Competitor_HeatPump_Dynamics", out_root=SI_DIR)


# ==============================================================================
# SUPPLEMENTARY FIGURE 4: Individual CMIP6 Model Trajectories & Cleveland Plot
# ==============================================================================
def build_supp_fig_4(df_glob: pd.DataFrame) -> dict:
    configure_matplotlib_style()
    H_MM = 136.0
    fig = plt.figure(figsize=(W_IN, H_MM * MM))

    df_dec = pd.read_parquet(SCEN_DIR / "r2_decomposition/r2_eq6_decomposition_all.parquet")
    gs_root = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[0.88, 1.22], left=0.132, right=0.980, bottom=0.098, top=0.945, hspace=0.40)

    gs_top = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=gs_root[0, 0], wspace=0.15)
    r1 = df_glob[(df_glob["group"] == "R1") & (df_glob["ssp"] != "obsclim")].copy()
    r1["mid_year"] = 0.5 * (r1["start_year"] + r1["end_year"])

    for idx_m, (m_code, m_lbl, _) in enumerate(GCM_ORDER):
        ax = fig.add_subplot(gs_top[0, idx_m])
        clean_ax(ax, "y")
        sub_m = r1[r1["model"] == m_code]
        for ssp_c, lbl_s, scol in [("ssp126", "SSP1-2.6", COL_SAPPHIRE), ("ssp245", "SSP2-4.5", COL_SSP2), ("ssp370", "SSP3-7.0", COL_CORAL)]:
            s_df = sub_m[sub_m["ssp"] == ssp_c].sort_values("mid_year")
            ax.plot(s_df["mid_year"], s_df["P_ej_yr"], color=scol, lw=1.35, marker="o", ms=2.4, label=lbl_s if idx_m == 0 else None)
        ax.axhline(3.05, color="#333333", ls=":", lw=0.60, zorder=2)
        ax.set_ylim(1.50, 3.35)
        ax.set_xlim(2010, 2096)
        ax.set_xticks([2015, 2055, 2090])
        ax.set_xticklabels(["2015", "2055", "2090"], fontsize=FS_COMPACT)
        # Clean pure-black in-panel label (ZERO ax.set_title!)
        ax.text(0.50, 1.03, m_lbl, transform=ax.transAxes, ha="center", va="bottom", fontsize=FS_COMPACT, fontweight="bold")
        if idx_m == 0:
            ax.set_ylabel("Economic potential\n(EJ yr$^{-1}$)")
            ax.legend(loc="lower left", frameon=False, fontsize=FS_COMPACT, handlelength=1.0, labelspacing=0.15)
        else:
            ax.set_yticklabels([])

    # Panel b: Cleveland Multi-Model Dot & Range Plot (ZERO Clustered Bars!)
    ax_b = fig.add_subplot(gs_root[1, 0])
    clean_ax(ax_b, "x")
    sel_isos = ["GLOBAL", "CHN", "RUS", "DEU", "KAZ", "KOR", "POL", "ITA", "NLD", "GBR", "CZE", "DNK", "SWE"][::-1]
    sub_c = df_dec[(df_dec["ssp"] == "ssp245") & (df_dec["window_id"] == "2046_2065") & (df_dec["entity_id"].isin(sel_isos))]

    yb = np.arange(len(sel_isos))
    ax_b.axvline(0.0, color="#000000", lw=0.65, ls="--", zorder=18)
    for yi, iso in enumerate(sel_isos):
        rows_i = sub_c[sub_c["entity_id"] == iso]
        vals = rows_i["dP_rel_pct"].to_numpy(dtype=float)
        lo, hi, mv = float(vals.min()), float(vals.max()), float(vals.mean())
        ax_b.plot([lo, hi], [yi, yi], color="#8E9FB1", lw=1.20, zorder=2)
        ax_b.plot([mv, mv], [yi - 0.24, yi + 0.24], color="#000000", lw=1.35, zorder=5)
        for m_code, m_lbl, mk in GCM_ORDER:
            mv_i = float(rows_i[rows_i["model"] == m_code]["dP_rel_pct"].iloc[0])
            ax_b.scatter([mv_i], [yi], s=22, marker=mk, color=GCM_COLORS[m_code], edgecolors="white", linewidths=0.35, zorder=4)

    leg_b = [
        mpl.lines.Line2D([], [], color="none", marker=mk, markerfacecolor=GCM_COLORS[mc], markeredgecolor="white", ms=5.0, label=ml)
        for mc, ml, mk in GCM_ORDER
    ] + [mpl.lines.Line2D([], [], color="#000000", lw=1.35, label="5-model mean")]
    ax_b.legend(handles=leg_b, loc="lower center", bbox_to_anchor=(0.48, 1.01), ncol=6, frameon=False, fontsize=FS_COMPACT, columnspacing=0.65, handletextpad=0.25)

    ax_b.set_yticks(yb)
    ax_b.set_yticklabels(["Global" if c == "GLOBAL" else ("UK" if c == "GBR" else cname(c)) for c in sel_isos], fontsize=FS_COMPACT)
    ax_b.set_xlim(-82.0, 4.0)
    ax_b.set_xticks([-80, -60, -40, -20, 0])
    ax_b.set_xticklabels([f"{MINUS}80", f"{MINUS}60", f"{MINUS}40", f"{MINUS}20", "0"])
    ax_b.set_xlabel("2046–2065 SSP2-4.5 change (%)")

    panel_tag(fig, 0.015, 0.982, "a")
    panel_tag(fig, 0.015, 0.565, "b")
    return export_figure(fig, "Supplementary_Fig4_Individual_GCM_Trajectories", out_root=SI_DIR)


# ==============================================================================
# SUPPLEMENTARY FIGURE 5: Pipe Distance ECDF & Sublinear Capital Scaling
# ==============================================================================
def build_supp_fig_5() -> dict:
    configure_matplotlib_style()
    H_MM = 96.0
    fig = plt.figure(figsize=(W_IN, H_MM * MM))

    lk = pd.read_parquet(SCEN_DIR / "R1__obs__2006_2025_selected_links.parquet")
    cand = pd.read_parquet(DATA_PROC / "sinks_2025/candidates_2025.parquet", columns=["ell_km"])

    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.12, 1.0], left=0.095, right=0.980, bottom=0.155, top=0.915, wspace=0.28)

    ax_a = fig.add_subplot(gs[0, 0])
    clean_ax(ax_a, "y")
    c_bins = np.linspace(0.0, 100.0, 101)
    c_hist, _ = np.histogram(cand["ell_km"].to_numpy(), bins=c_bins)
    c_cdf = 100.0 * np.cumsum(c_hist) / max(1, int(c_hist.sum()))

    ord_l = np.argsort(lk["ell_km"].to_numpy())
    ell_s = lk["ell_km"].to_numpy()[ord_l]
    p_w = lk["P_net_waste_heat_gj_yr"].to_numpy()[ord_l]
    p_cdf = 100.0 * np.cumsum(p_w) / float(p_w.sum())
    n_cdf = 100.0 * np.arange(1, len(ell_s) + 1) / len(ell_s)

    ax_a.plot(c_bins[1:], c_cdf, color=COL_BLDG_SLATE, ls="--", lw=1.25, label="Candidate connections ($n$ = 2.49M)")
    ax_a.plot(ell_s, n_cdf, color=COL_SKY_CYAN, lw=1.50, label="Selected networks ($n$ = 5,327)")
    ax_a.plot(ell_s, p_cdf, color=COL_SAPPHIRE, lw=1.75, label="Economic potential $P_0$ share")
    ax_a.axvline(50.0, color="#000000", ls=":", lw=0.75)
    p50 = float(p_w[ell_s <= 50.0].sum() / p_w.sum() * 100.0)
    ax_a.text(52.0, 74.0, f"{p50:.1f}% of $P_0$ within 50 km\n(max = {ell_s.max():.1f} km)", fontsize=FS_COMPACT, va="center", ha="left", linespacing=1.12)

    ax_a.set_xlim(0.0, 100.0)
    ax_a.set_ylim(0.0, 103.0)
    ax_a.set_xlabel("Routed transmission pipe length (km)")
    ax_a.set_ylabel("Cumulative share (%)")
    ax_a.legend(loc="lower right", frameon=False, fontsize=FS_COMPACT, labelspacing=0.18)

    ax_b = fig.add_subplot(gs[0, 1])
    clean_ax(ax_b, "y")
    c_mw = np.logspace(0, 3, 200)
    spec_capex_k_usd = (PIPE_CAPEX_COEF_EUR_PER_KM * (c_mw ** PIPE_CAPEX_EXPONENT) / c_mw) * USD_PER_EUR_2025 / 1e3
    ax_b.plot(c_mw, spec_capex_k_usd, color=COL_SAPPHIRE, lw=1.70, label="70/40 °C conventional networks")
    ax_b.plot(c_mw, spec_capex_k_usd * 0.95, color=COL_EMERALD, ls="--", lw=1.45, label=f"55/25 °C low temp. ({MINUS}5%)")
    ax_b.set_xscale("log")
    ax_b.set_xticks([1, 10, 100, 1000])
    ax_b.set_xlabel("Design pipe capacity (MW)")
    ax_b.set_ylabel("Specific pipe capital\n(thousand USD MW$^{-1}$ km$^{-1}$)")
    ax_b.legend(loc="upper right", frameon=False, fontsize=FS_COMPACT, labelspacing=0.18)
    ax_b.text(14.0, 260.0, "Sublinear scale economy\n$I_{\\mathrm{p}}(C) \\propto C^{0.429}$", fontsize=FS_COMPACT, linespacing=1.15)

    panel_tag(fig, 0.015, 0.968, "a")
    panel_tag(fig, 0.535, 0.968, "b")
    return export_figure(fig, "Supplementary_Fig5_Distance_and_Pipe_Scale", out_root=SI_DIR)


# ==============================================================================
# SUPPLEMENTARY FIGURE 6: End-Century SSP3-7.0 Three-Term Decomposition
# ==============================================================================
def build_supp_fig_6() -> dict:
    configure_matplotlib_style()
    H_MM = 114.0
    fig = plt.figure(figsize=(W_IN, H_MM * MM))

    df_dec = pd.read_parquet(SCEN_DIR / "r2_decomposition/r2_eq6_decomposition_all.parquet")
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.12, 1.0], left=0.145, right=0.980, bottom=0.135, top=0.845, wspace=0.32)

    ax_a = fig.add_subplot(gs[0, 0])
    clean_ax(ax_a, "x")
    c_end = df_dec[(df_dec["ssp"] == "ssp370") & (df_dec["window_id"] == "2081_2100")]

    def _agg_e(sub: pd.DataFrame, lbl: str) -> dict:
        net_m = float(sub["dP_rel_pct"].mean())
        t1 = float(sub["term1_demand_shift_pct"].mean())
        t2 = float(sub["term2_seasonal_mismatch_pct"].mean())
        t3 = net_m - t1 - t2
        return {"lbl": lbl, "t1": t1, "t2": t2, "t3": t3, "net": net_m, "lo": float(sub["dP_rel_pct"].min()), "hi": float(sub["dP_rel_pct"].max())}

    isos_a = ["CHN", "RUS", "DEU", "KAZ", "KOR", "POL", "ITA", "NLD", "GBR", "CZE", "DNK", "SWE"]
    rows_a = [_agg_e(c_end[c_end["level"] == "global"], "Global")]
    for iso in isos_a:
        sub_i = c_end[(c_end["level"] == "country") & (c_end["entity_id"] == iso)]
        if len(sub_i):
            rows_a.append(_agg_e(sub_i, "UK" if iso == "GBR" else cname(iso)))
    rows_a = rows_a[::-1]

    ya = np.arange(len(rows_a))
    for yi, r in enumerate(rows_a):
        neg_l, pos_l = 0.0, 0.0
        for val, col in [(r["t3"], COL_CORAL), (r["t1"], COL_DEM_BLUE), (r["t2"], "#8CD4EC")]:
            if val < 0:
                ax_a.barh(yi, val, left=neg_l, height=0.62, color=col, edgecolor="none", zorder=3)
                if neg_l < -0.05:
                    ax_a.plot([neg_l, neg_l], [yi - 0.32, yi + 0.32], color="white", lw=0.45, zorder=3.5)
                neg_l += val
            elif val > 0:
                ax_a.barh(yi, val, left=pos_l, height=0.62, color=col, edgecolor="none", zorder=3)
                pos_l += val
        if r["lo"] < -88.0:
            ax_a.errorbar(r["net"], yi, xerr=[[0.0], [r["hi"] - r["net"]]], fmt="D", ms=3.0, color="#000000", ecolor="#000000", elinewidth=0.60, capsize=1.5, zorder=6)
            ax_a.annotate("", xy=(-88.0, yi), xytext=(r["net"], yi), arrowprops=dict(arrowstyle="->,head_width=0.14,head_length=0.22", color="#000000", lw=0.60), zorder=6)
        else:
            ax_a.errorbar(r["net"], yi, xerr=[[r["net"] - r["lo"]], [r["hi"] - r["net"]]], fmt="D", ms=3.0, color="#000000", ecolor="#000000", elinewidth=0.60, capsize=1.5, zorder=6)

    ax_a.axvline(0.0, color="#000000", lw=0.65, zorder=18)
    ax_a.axhline(len(rows_a) - 1.5, ls="--", lw=0.50, color="#444444", zorder=10)
    ax_a.set_yticks(ya)
    ax_a.set_yticklabels([r["lbl"] for r in rows_a], fontsize=FS_COMPACT)
    ax_a.set_xlim(-90.0, 16.0)
    ax_a.set_xticks([-80, -60, -40, -20, 0, 15])
    ax_a.set_xticklabels([f"{MINUS}80", f"{MINUS}60", f"{MINUS}40", f"{MINUS}20", "0", "+15"])
    ax_a.set_xlabel("2081–2100 SSP3-7.0 contribution (%)")
    lock_spines_above_data(ax_a)

    leg_top = [
        mpatches.Patch(facecolor=COL_DEM_BLUE, edgecolor="none", label="Heating demand"),
        mpatches.Patch(facecolor="#8CD4EC", edgecolor="none", label="Seasonal matching"),
        mpatches.Patch(facecolor=COL_CORAL, edgecolor="none", label="Economic viability"),
        mpl.lines.Line2D([], [], color="#000000", marker="D", lw=0.80, ms=3.2, label="Net change"),
    ]
    fig.legend(handles=leg_top, loc="upper center", bbox_to_anchor=(0.54, 0.980), ncol=4, frameon=False, fontsize=FS_COMPACT, columnspacing=1.0, handlelength=0.95)

    ax_b = fig.add_subplot(gs[0, 1])
    clean_ax(ax_b, "x")
    z_dir = SCEN_DIR / "zones"
    df_zm = pd.read_parquet(DATA_PROC / "sinks_2025/urban_heat_zones.parquet")[["zone_id", "lat", "baseline_mean_mw"]]
    z_rows = []
    for m, _, _ in GCM_ORDER:
        b0 = pd.read_parquet(z_dir / f"R1__{m}__ssp370__2006_2025_zones.parquet")[["zone_id", "D_gj", "U_gj", "P_gj"]].rename(columns={"D_gj": "D0", "U_gj": "U0", "P_gj": "P0"})
        f1 = pd.read_parquet(z_dir / f"R1__{m}__ssp370__2081_2100_zones.parquet")[["zone_id", "D_gj", "U_gj", "P_gj"]].rename(columns={"D_gj": "D1", "U_gj": "U1", "P_gj": "P1"})
        mg = b0.merge(f1, on="zone_id").merge(df_zm, on="zone_id", how="left")
        mg["model"] = m
        z_rows.append(mg)
    z_all = pd.concat(z_rows, ignore_index=True)

    def _strat_dec(mask_arr, name_s: str) -> dict:
        sub = z_all[mask_arr].groupby("model")[["D0", "D1", "U0", "U1", "P0", "P1"]].sum()
        dD = 100.0 * (sub["D1"] - sub["D0"]) / sub["D0"]
        dU = 100.0 * (sub["U1"] - sub["U0"]) / sub["U0"]
        dP = 100.0 * (sub["P1"] - sub["P0"]) / sub["P0"]
        pts = [(m_id, float(dP.loc[m_id])) for m_id in sub.index]
        return {
            "lbl": name_s, "t1": float(dD.mean()), "t2": float((dU - dD).mean()), "t3": float((dP - dU).mean()),
            "net": float(dP.mean()), "lo": float(dP.min()), "hi": float(dP.max()), "pts": pts,
        }

    al = z_all["lat"].abs()
    mw = z_all["baseline_mean_mw"]
    st_list = [
        _strat_dec(al >= 60.0, ">60°"),
        _strat_dec((al >= 50.0) & (al < 60.0), "50–60°"),
        _strat_dec((al >= 40.0) & (al < 50.0), "40–50°"),
        _strat_dec((al >= 30.0) & (al < 40.0), "30–40°"),
        _strat_dec(al < 30.0, "<30°"),
        _strat_dec(mw > 100.0, ">100 MW cities"),
        _strat_dec((mw >= 10.0) & (mw <= 100.0), "10–100 MW cities"),
        _strat_dec(mw < 10.0, "1–10 MW cities"),
    ][::-1]

    yb = np.arange(len(st_list))
    for yi, r in enumerate(st_list):
        neg_l, pos_l = 0.0, 0.0
        for val, col in [(r["t3"], COL_CORAL), (r["t1"], COL_DEM_BLUE), (r["t2"], "#8CD4EC")]:
            if val < 0:
                ax_b.barh(yi, val, left=neg_l, height=0.60, color=col, edgecolor="none", zorder=3)
                if neg_l < -0.05:
                    ax_b.plot([neg_l, neg_l], [yi - 0.30, yi + 0.30], color="white", lw=0.45, zorder=3.5)
                neg_l += val
            elif val > 0:
                ax_b.barh(yi, val, left=pos_l, height=0.60, color=col, edgecolor="none", zorder=3)
                pos_l += val
        ax_b.errorbar(r["net"], yi, xerr=[[r["net"] - r["lo"]], [r["hi"] - r["net"]]], fmt="D", ms=3.0, color="#000000", ecolor="#000000", elinewidth=0.60, capsize=1.5, zorder=6)
        for m_id, m_val in r["pts"]:
            ax_b.scatter([m_val], [yi], facecolors="none", edgecolors="#000000", marker=GCM_MARKER[m_id], s=9.0, linewidths=0.45, zorder=5)

    ax_b.axvline(0.0, color="#000000", lw=0.65, zorder=18)
    ax_b.axhline(2.5, color="#444444", ls="--", lw=0.50, zorder=10)
    ax_b.set_yticks(yb)
    ax_b.set_yticklabels([r["lbl"] for r in st_list], fontsize=FS_COMPACT)
    ax_b.set_xlim(-82.0, 16.0)
    ax_b.set_xticks([-80, -60, -40, -20, 0, 15])
    ax_b.set_xticklabels([f"{MINUS}80", f"{MINUS}60", f"{MINUS}40", f"{MINUS}20", "0", "+15"])
    ax_b.set_xlabel("Stratified 2081–2100 change (%)")
    lock_spines_above_data(ax_b)

    panel_tag(fig, 0.015, 0.895, "a")
    panel_tag(fig, 0.545, 0.895, "b")
    return export_figure(fig, "Supplementary_Fig6_EndCentury_Decomposition", out_root=SI_DIR)


# ==============================================================================
# SUPPLEMENTARY FIGURE 7: Model Lollipop, Margin Screen & Pit Storage Sensitivity
# ==============================================================================
def build_supp_fig_7() -> dict:
    configure_matplotlib_style()
    H_MM = 98.0
    fig = plt.figure(figsize=(W_IN, H_MM * MM))

    r3_dir = SCEN_DIR / "r3_locked_assets"
    df_r3 = pd.read_csv(r3_dir / "r3_locked_assets_summary.csv")
    link_dfs = [pd.read_parquet(p) for p in sorted(r3_dir.glob("R3__*__ssp245__2046_2065_links.parquet"))]
    df_lk = pd.concat(link_dfs, ignore_index=True).groupby(["stream_id", "zone_id"], as_index=False).agg(
        cap=("total_sunk_capex_usd", "first"),
        d0=("sub_D_0_gj_yr", "first"),
        n0=("net_benefit_0_usd_yr", "first"),
        nfut=("net_benefit_fut_usd_yr", "mean"),
        p0=("P_0_gj_yr", "first"),
    )
    df_lk["m0"] = df_lk["n0"] / np.maximum(1.0, df_lk["d0"])
    df_lk["str"] = df_lk["nfut"] < 0.0

    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1.06, 1.02, 1.02], left=0.148, right=0.980, bottom=0.165, top=0.905, wspace=0.36)

    # Panel a: Horizontal Lollipop Plot of Model-Specific Stranded Capital & Network Counts (ZERO Bars!)
    ax_a = fig.add_subplot(gs[0, 0])
    clean_ax(ax_a, "x")
    cnt_col = "loss_making_links" if "loss_making_links" in df_r3.columns else [c for c in df_r3.columns if "loss" in c and "capex" not in c][0]
    sub2055 = df_r3[df_r3["window_id"] == "2046_2065"].sort_values("loss_making_capex_billion_usd", ascending=False).reset_index(drop=True)
    ya = np.arange(len(sub2055))
    for yi, r in sub2055.iterrows():
        v_cap = float(r["loss_making_capex_billion_usd"])
        n_cnt = int(round(float(r[cnt_col])))
        ax_a.plot([0.0, v_cap], [yi, yi], color=COL_CORAL, lw=1.45, zorder=3)
        ax_a.scatter([v_cap], [yi], s=34, color=COL_CORAL, edgecolors="white", linewidths=0.40, zorder=4)
        ax_a.text(v_cap + 2.0, yi, f"{v_cap:.1f} ({n_cnt:,})", va="center", ha="left", fontsize=FS_COMPACT)

    ax_a.set_yticks(ya)
    ax_a.set_yticklabels([GCM_LABELS.get(str(m), str(m)) for m in sub2055["model"]], fontsize=FS_COMPACT)
    ax_a.set_xlim(0.0, 92.0)
    ax_a.set_xticks([0, 25, 50, 75])
    ax_a.set_xlabel("2046–2065 stranded capital\n(billion USD, networks)")

    # Panel b: Initial Cost-Saving Margin Screening Trade-off Curve
    ax_b = fig.add_subplot(gs[0, 1])
    clean_ax(ax_b, "y")
    cuts = np.linspace(0.0, 6.0, 61)
    tot_str_cap = float(df_lk.loc[df_lk["str"], "cap"].sum())
    tot_p0 = float(df_lk["p0"].sum())
    avoided_str = [100.0 * float(df_lk.loc[df_lk["str"] & (df_lk["m0"] < c), "cap"].sum()) / max(1.0, tot_str_cap) for c in cuts]
    kept_p0 = [100.0 * float(df_lk.loc[df_lk["m0"] >= c, "p0"].sum()) / max(1.0, tot_p0) for c in cuts]

    av_3 = 100.0 * float(df_lk.loc[df_lk["str"] & (df_lk["m0"] < 3.0), "cap"].sum()) / max(1.0, tot_str_cap)
    kp_3 = 100.0 * float(df_lk.loc[df_lk["m0"] >= 3.0, "p0"].sum()) / max(1.0, tot_p0)

    ax_b.plot(cuts, avoided_str, color=COL_CORAL, lw=1.65, label="Stranded capital avoided")
    ax_b.plot(cuts, kept_p0, color=COL_SAPPHIRE, lw=1.65, ls="--", label="Present $P_0$ retained")
    ax_b.axvline(3.0, color="#000000", ls=":", lw=0.75)
    ax_b.scatter([3.0, 3.0], [av_3, kp_3], s=18, color=[COL_CORAL, COL_SAPPHIRE], zorder=5)
    ax_b.text(3.18, av_3 + 1.5, f"{av_3:.1f}%", ha="left", va="bottom", fontsize=FS_COMPACT)
    ax_b.text(3.18, kp_3 + 2.0, f"{kp_3:.1f}%", ha="left", va="bottom", fontsize=FS_COMPACT)

    ax_b.set_xlim(0.0, 6.0)
    ax_b.set_ylim(0.0, 105.0)
    ax_b.set_xlabel("Minimum initial saving\nscreen (USD GJ$^{-1}$)")
    ax_b.set_ylabel("Portfolio share (%)")
    ax_b.legend(loc="lower right", bbox_to_anchor=(1.02, 0.02), frameon=False, fontsize=FS_COMPACT, labelspacing=0.16, handlelength=1.1)

    # Panel c: Seasonal Water Pit Storage Sensitivity Across Efficiency & Discount Rates (Non-Duplicating!)
    ax_c = fig.add_subplot(gs[0, 2])
    clean_ax(ax_c, "y")
    eta_grid = np.linspace(0.55, 0.85, 91)
    capex_usd_per_gj = 172.2 * USD_PER_EUR_2025
    for r_val, col_r, ls_r, lbl_r in [
        (0.10, COL_CORAL, "-.", "$r$ = 10%"),
        (0.07, COL_SAPPHIRE, "-", "$r$ = 7% (central)"),
        (0.03, "#5EC2E6", "--", "$r$ = 3%"),
    ]:
        crf = r_val * (1.0 + r_val) ** 35 / ((1.0 + r_val) ** 35 - 1.0)
        dc = ((crf + 0.01) * capex_usd_per_gj) / eta_grid
        ax_c.plot(eta_grid, dc, color=col_r, ls=ls_r, lw=1.60, label=lbl_r)

    crf7 = 0.07 * (1.07 ** 35) / ((1.07 ** 35) - 1.0)
    dc70 = ((crf7 + 0.01) * capex_usd_per_gj) / 0.70
    ax_c.scatter([0.70], [dc70], s=26, marker="D", color=COL_SAPPHIRE, edgecolors="white", linewidths=0.40, zorder=5)
    ax_c.text(0.71, dc70 + 1.2, f"{dc70:.1f} USD GJ$^{{-1}}$", ha="left", va="bottom", fontsize=FS_COMPACT)

    # Margin ceiling = 18.58 (ASHP benchmark) - 14.01 (network LCOH) = 4.57 USD GJ^-1
    ax_c.axhline(4.57, color="#000000", ls=":", lw=0.75)
    ax_c.text(0.56, 5.4, "Break-even margin (4.6)", ha="left", va="bottom", fontsize=FS_COMPACT)

    ax_c.set_xlim(0.55, 0.85)
    ax_c.set_ylim(0.0, 42.0)
    ax_c.set_xticks([0.55, 0.65, 0.75, 0.85])
    ax_c.set_xlabel("Seasonal storage efficiency $\\eta$")
    ax_c.set_ylabel("Incremental cost per\nstored GJ (USD GJ$^{-1}$)")
    ax_c.legend(loc="upper right", frameon=False, fontsize=FS_COMPACT, labelspacing=0.16, handlelength=1.2)

    panel_tag(fig, 0.015, 0.968, "a")
    panel_tag(fig, 0.425, 0.968, "b")
    panel_tag(fig, 0.712, 0.968, "c")
    return export_figure(fig, "Supplementary_Fig7_Stranding_Screening_and_PTES", out_root=SI_DIR)


# ==============================================================================
# SUPPLEMENTARY FIGURE 8: Mid-Century Non-Fossil Economic Potential
# ==============================================================================
def build_supp_fig_8(df_country: pd.DataFrame) -> dict:
    configure_matplotlib_style()
    H_MM = 138.0
    fig = plt.figure(figsize=(W_IN, H_MM * MM))

    lk0 = pd.read_parquet(SCEN_DIR / "R1__obs__2006_2025_selected_links.parquet")
    df_zm = pd.read_parquet(DATA_PROC / "sinks_2025/urban_heat_zones.parquet")[["zone_id", "lat", "lon"]]
    gdf_rob, trans, px, py = _robinson_base()

    ax_m = fig.add_axes([0.025, 0.465, 0.950, 0.505])
    ax_b = fig.add_axes([0.145, 0.085, 0.830, 0.330])

    ax_m.fill(px, py, facecolor="#FFFFFF", edgecolor="#333333", linewidth=0.52, zorder=0)
    for glat in [-30, 0, 30, 60]:
        gx, gy = trans.transform(np.linspace(-180, 180, 181), np.full(181, glat))
        ax_m.plot(gx, gy, color="#D8E0E8", lw=0.30, zorder=0.4)
    gdf_rob.plot(ax=ax_m, facecolor="#F3F5F8", edgecolor="#8A96A3", linewidth=0.24, zorder=1)

    nf_lk = lk0[~lk0["is_fossil_power"]].merge(df_zm, on="zone_id", how="left")
    nf_lk["grp"] = np.where(nf_lk["sector"] == "power", "Nuclear and bioenergy", "Heavy industry and wastewater")
    for g_lbl, col_h, zo in [("Heavy industry and wastewater", F5_STEEL, 3.2), ("Nuclear and bioenergy", F5_POWER, 3.6)]:
        sub = nf_lk[nf_lk["grp"] == g_lbl]
        xz, yz = trans.transform(sub["lon"].to_numpy(), sub["lat"].to_numpy())
        sz = np.clip(np.sqrt(sub["P_net_waste_heat_gj_yr"].to_numpy() / 1e6) * 9.5, 3.0, 40.0)
        ax_m.scatter(xz, yz, s=sz, color=col_h, edgecolors="white", linewidths=0.25, alpha=0.88, zorder=zo, label=f"{g_lbl} ($n$ = {len(sub):,})")

    ax_m.set_xlim(px.min() * 1.02, px.max() * 1.02)
    ax_m.set_ylim(py.min() * 1.02, py.max() * 1.02)
    ax_m.set_xticks([])
    ax_m.set_yticks([])
    ax_m.set_xlabel("")
    ax_m.set_ylabel("")
    ax_m.axis("off")
    leg_m = ax_m.legend(loc="lower left", bbox_to_anchor=(0.065, 0.075), frameon=True, facecolor="white", framealpha=0.90, edgecolor="none", fontsize=FS_COMPACT, markerscale=0.72, handletextpad=0.3, labelspacing=0.20)
    leg_m.set_zorder(6)

    clean_ax(ax_b, "x")
    s5_raw = df_country[(df_country["group"] == "R4") & (df_country["subgroup"] == "S5_combined_all") & (df_country["window_id"] == "2046_2065")].copy()
    s5_raw["chem_ww_gj"] = s5_raw["P_ethylene_gj"] + s5_raw["P_pulp_paper_gj"] + s5_raw["P_wastewater_gj"]
    s5_mean = s5_raw.groupby("iso3", as_index=False)[
        ["P_gj", "P_steel_gj", "P_refining_gj", "P_cement_gj", "P_power_gj", "chem_ww_gj"]
    ].mean()
    s5_minmax = s5_raw.groupby("iso3")["P_gj"].agg(["min", "max"]).reset_index()
    s5 = s5_mean.merge(s5_minmax, on="iso3").sort_values("P_gj", ascending=False).head(12).iloc[::-1].reset_index(drop=True)

    yb = np.arange(len(s5))
    left_b = np.zeros(len(s5))
    for col_k, lbl_k, hex_k in [
        ("P_steel_gj", "Iron and steel", F5_STEEL),
        ("P_refining_gj", "Oil refining", F5_REFINING),
        ("P_cement_gj", "Cement", F5_CEMENT),
        ("P_power_gj", "Nuclear and bioenergy", F5_POWER),
        ("chem_ww_gj", "Chemicals, pulp and wastewater", F5_CHEM),
    ]:
        v = s5[col_k].to_numpy() / 1e6
        ax_b.barh(yb, v, left=left_b, height=0.62, color=hex_k, edgecolor="none", zorder=3, label=lbl_k)
        for yi in range(len(s5)):
            if left_b[yi] > 0.2 and v[yi] > 0.2:
                ax_b.plot([left_b[yi], left_b[yi]], [yi - 0.31, yi + 0.31], color="white", lw=0.40, zorder=3.5)
        left_b += v

    tot_pj = s5["P_gj"].to_numpy() / 1e6
    lo_pj = s5["min"].to_numpy() / 1e6
    hi_pj = s5["max"].to_numpy() / 1e6
    ax_b.errorbar(tot_pj, yb, xerr=[tot_pj - lo_pj, hi_pj - tot_pj], fmt="none", ecolor="#000000", elinewidth=0.60, capsize=1.5, zorder=5)
    for yi, (mv, hv) in enumerate(zip(tot_pj, hi_pj)):
        ax_b.text(hv + 1.2, yi, f"{mv:.1f}", va="center", ha="left", fontsize=FS_COMPACT)

    ax_b.set_yticks(yb)
    ax_b.set_yticklabels(["UK" if c == "GBR" else cname(c) for c in s5["iso3"]], fontsize=FS_COMPACT)
    ax_b.set_xlim(0.0, 104.0)
    ax_b.set_xlabel("2046–2065 combined 2050 economic potential (PJ yr$^{-1}$)")
    ax_b.legend(loc="lower right", frameon=False, fontsize=FS_COMPACT, ncol=2, columnspacing=0.8, labelspacing=0.18, handlelength=0.95)
    lock_spines_above_data(ax_b)

    panel_tag(fig, 0.015, 0.982, "a")
    panel_tag(fig, 0.015, 0.440, "b")

    res = export_figure(fig, "Supplementary_Fig8_NonFossil_Supply", out_root=SI_DIR)
    # Also save legacy filename copy so any script looking for old filename still finds the updated clean figure
    for ext in ("png", "pdf"):
        src_p = SI_DIR / f"Supplementary_Fig8_NonFossil_Supply.{ext}"
        dst_p = SI_DIR / f"Supplementary_Fig8_ZeroFossil_Resilient_Core.{ext}"
        dst_p.write_bytes(src_p.read_bytes())
    return res


# ==============================================================================
# VERIFIED TABLE PAYLOAD EXPORTER (100% Clean Glossary & Zero Bad Punctuation!)
# ==============================================================================
def export_verified_tables_json(out_json: Path) -> dict:
    df_g = pd.read_parquet(SCEN_DIR / "global_scenarios_summary.parquet")
    df_c = pd.read_parquet(SCEN_DIR / "all_scenarios_by_country.parquet")
    df_dec = pd.read_parquet(SCEN_DIR / "r2_decomposition/r2_eq6_decomposition_all.parquet")
    df_r3_sum = pd.read_csv(SCEN_DIR / "r3_locked_assets/r3_locked_assets_summary.csv")
    df_rp = pd.read_parquet(DATA_PROC / "sinks_2025/calibrated_region_parameters_2025.parquet")
    df_dh = pd.read_csv(DATA_PROC / "sources_2025/existing_chp_dh_national.csv")
    val1_p = pd.read_csv(DATA_PROC / "sources_2025/validation_1/power_nuclear_vs_ember2025.csv")
    val1_i = pd.read_csv(DATA_PROC / "sources_2025/validation_1/industrial_output_vs_official2025.csv")
    rob_csv = pd.read_csv(SRC_DIR / "Extended_Data_Fig3_sensitivity_levels_source_data.csv")

    u_fx = USD_PER_EUR_2025

    ed_table_1 = [
        ["Direct heat exchange station (≥95 °C or ≥55 °C)", "Upgrade ratio κ = 1.00", f"{SOURCE_HEX_STATION_CAPEX_EUR_PER_MW * u_fx / 1e3:.1f}", f"{SOURCE_HEX_STATION_CAPEX_EUR_PER_MW * u_fx / 1e3:.1f}", "1.0% CAPEX yr⁻¹", f"{int(STATION_LIFETIME_YR)}", "DEA Gen. and DH"],
        ["Source electric heat pump (power and industry)", f"COP = {HP_INDUSTRIAL_POWER_2025.cop:.2f} → {HP_INDUSTRIAL_POWER_2050.cop:.2f}", f"{HP_INDUSTRIAL_POWER_2025.capex_eur_per_mw_h * u_fx / 1e3:.1f}", f"{HP_INDUSTRIAL_POWER_2050.capex_eur_per_mw_h * u_fx / 1e3:.1f}", f"{HP_INDUSTRIAL_POWER_2025.fixed_om_eur_per_mw_h_yr * u_fx / 1e3:.2f} kUSD MW⁻¹ + {HP_INDUSTRIAL_POWER_2025.var_om_eur_per_mwh_h * u_fx:.2f} USD MWh⁻¹", f"{int(HP_INDUSTRIAL_POWER_2025.lifetime_yr)}", "DEA Sheet 40 (10 MW)"],
        ["Source electric heat pump (wastewater, ΔT = 5 K)", f"COP = {HP_WASTEWATER_2025.cop:.2f} → {HP_WASTEWATER_2050.cop:.2f}", f"{HP_WASTEWATER_2025.capex_eur_per_mw_h * u_fx / 1e3:.1f}", f"{HP_WASTEWATER_2050.capex_eur_per_mw_h * u_fx / 1e3:.1f}", f"{HP_WASTEWATER_2025.fixed_om_eur_per_mw_h_yr * u_fx / 1e3:.2f} kUSD MW⁻¹ + {HP_WASTEWATER_2025.var_om_eur_per_mwh_h * u_fx:.2f} USD MWh⁻¹", f"{int(HP_WASTEWATER_2025.lifetime_yr)}", "DEA Sheet 40 (1 MW)"],
        ["Insulated transmission mains (70/40 °C)", "Loss 1.0 / 0.25 / 0.10% km⁻¹ and 1% pump power", "421.4 × C⁰·⁴²⁹ kUSD km⁻¹", "400.3 × C⁰·⁴²⁹ kUSD km⁻¹ (55 °C)", "113 USD MW⁻¹ km⁻¹ yr⁻¹", f"{int(PIPE_LIFETIME_YR)}", "DEA Energy Transport 3.1"],
        ["Urban distribution pipes and substation", "Plot ratio density curve and η = 1.00", "Area plot ratio + 83.0", "−5% pipes + 76.2", "0.39 → 0.34 kUSD MW⁻¹ yr⁻¹", "40 / 25", "Persson et al. (2019), DEA 203"],
        ["Centralized gas peaking boiler", f"η (LHV) = {PEAKING_GAS_BOILER_2025.efficiency_lhv:.2f} → {PEAKING_GAS_BOILER_2050.efficiency_lhv:.2f}", f"{PEAKING_GAS_BOILER_2025.capex_eur_per_mw_h * u_fx / 1e3:.1f}", f"{PEAKING_GAS_BOILER_2050.capex_eur_per_mw_h * u_fx / 1e3:.1f}", f"{PEAKING_GAS_BOILER_2025.fixed_om_eur_per_mw_h_yr * u_fx / 1e3:.2f} kUSD MW⁻¹ yr⁻¹", f"{int(PEAKING_GAS_BOILER_2025.lifetime_yr)}", "DEA Sheet 44"],
        ["Centralized electric peaking boiler", f"η = {PEAKING_ELEC_BOILER_2025.efficiency_lhv:.2f} → {PEAKING_ELEC_BOILER_2050.efficiency_lhv:.2f}", f"{PEAKING_ELEC_BOILER_2025.capex_eur_per_mw_h * u_fx / 1e3:.1f}", f"{PEAKING_ELEC_BOILER_2050.capex_eur_per_mw_h * u_fx / 1e3:.1f}", f"{PEAKING_ELEC_BOILER_2025.fixed_om_eur_per_mw_h_yr * u_fx / 1e3:.2f} kUSD MW⁻¹ yr⁻¹", "20 → 25", "DEA Sheet 41"],
        ["Building gas boiler", f"η (LHV) = {INDIV_GAS_BOILER_2025.efficiency_or_scop_ref:.2f} → {INDIV_GAS_BOILER_2050.efficiency_or_scop_ref:.2f}", f"{INDIV_GAS_BOILER_2025.capex_eur_per_mw_h * u_fx / 1e3:.1f}", f"{INDIV_GAS_BOILER_2050.capex_eur_per_mw_h * u_fx / 1e3:.1f}", f"{INDIV_GAS_BOILER_2025.fixed_om_eur_per_mw_h_yr * u_fx / 1e3:.2f} → {INDIV_GAS_BOILER_2050.fixed_om_eur_per_mw_h_yr * u_fx / 1e3:.2f} kUSD MW⁻¹ yr⁻¹", f"{int(INDIV_GAS_BOILER_2025.lifetime_yr)}", "DEA Sheet 202"],
        ["Building air source heat pump", f"SCOP = {INDIV_ASHP_2025.efficiency_or_scop_ref:.2f} → {INDIV_ASHP_2050.efficiency_or_scop_ref:.2f}", f"{INDIV_ASHP_2025.capex_eur_per_mw_h * u_fx / 1e3:.1f}", f"{INDIV_ASHP_2050.capex_eur_per_mw_h * u_fx / 1e3:.1f}", f"{INDIV_ASHP_2025.fixed_om_eur_per_mw_h_yr * u_fx / 1e3:.2f} → {INDIV_ASHP_2050.fixed_om_eur_per_mw_h_yr * u_fx / 1e3:.2f} kUSD MW⁻¹ yr⁻¹", f"{int(INDIV_ASHP_2025.lifetime_yr)}", "DEA Sheet 207, Brudermueller et al. (2025)"],
    ]

    p0_ref = float(df_g.loc[df_g["scenario_id"] == "R1__obs__2006_2025", "P_ej_yr"].iloc[0])

    def _fmt_g(sub_df: pd.DataFrame, name_s: str, boundary_s: str, p_base_val: float = p0_ref) -> list[str]:
        d_m = float(sub_df["D_total_ej_yr"].mean())
        u_m = float(sub_df["U_ej_yr"].mean())
        p_v = sub_df["P_ej_yr"].to_numpy(dtype=float)
        p_m, p_lo, p_hi = float(p_v.mean()), float(p_v.min()), float(p_v.max())
        dp_m = 100.0 * (p_m - p_base_val) / max(1e-9, p_base_val)
        nb_m = float(sub_df["net_benefit_billion_usd_yr"].mean())
        co2_m = float(sub_df["co2_abated_mt_yr"].mean())
        p_str = f"{p_m:.2f}" if len(p_v) == 1 else f"{p_m:.2f} ({p_lo:.2f}–{p_hi:.2f})"
        dp_str = f"{fm(dp_m, '.1f')}%" if dp_m <= 0 else f"+{dp_m:.1f}%"
        return [name_s, boundary_s, f"{d_m:.2f}", f"{u_m:.2f}", p_str, dp_str, f"{nb_m:.2f}", f"{co2_m:.1f}"]

    ed_table_2 = [
        _fmt_g(df_g[df_g["scenario_id"] == "R1__obs__1946_1965"], "Observed 1946–1965", "Fixed 2025 system, GSWP3-ERA5"),
        _fmt_g(df_g[df_g["scenario_id"] == "R1__obs__1966_1985"], "Observed 1966–1985", "Fixed 2025 system, GSWP3-ERA5"),
        _fmt_g(df_g[df_g["scenario_id"] == "R1__obs__1986_2005"], "Observed 1986–2005", "Fixed 2025 system, GSWP3-ERA5"),
        _fmt_g(df_g[df_g["scenario_id"] == "R1__obs__2006_2025"], "Present 2006–2025", "Fixed 2025 system, GSWP3-ERA5"),
        _fmt_g(df_g[(df_g["group"] == "R1") & (df_g["ssp"] == "ssp126") & (df_g["window_id"] == "2046_2065")], "SSP1-2.6 (2046–2065)", "Fixed 2025 system, 5 CMIP6 models"),
        _fmt_g(df_g[(df_g["group"] == "R1") & (df_g["ssp"] == "ssp126") & (df_g["window_id"] == "2081_2100")], "SSP1-2.6 (2081–2100)", "Fixed 2025 system, 5 CMIP6 models"),
        _fmt_g(df_g[(df_g["group"] == "R1") & (df_g["ssp"] == "ssp245") & (df_g["window_id"] == "2046_2065")], "SSP2-4.5 (2046–2065)", "Fixed 2025 system, 5 CMIP6 models"),
        _fmt_g(df_g[(df_g["group"] == "R1") & (df_g["ssp"] == "ssp245") & (df_g["window_id"] == "2081_2100")], "SSP2-4.5 (2081–2100)", "Fixed 2025 system, 5 CMIP6 models"),
        _fmt_g(df_g[(df_g["group"] == "R1") & (df_g["ssp"] == "ssp370") & (df_g["window_id"] == "2046_2065")], "SSP3-7.0 (2046–2065)", "Fixed 2025 system, 5 CMIP6 models"),
        _fmt_g(df_g[(df_g["group"] == "R1") & (df_g["ssp"] == "ssp370") & (df_g["window_id"] == "2081_2100")], "SSP3-7.0 (2081–2100)", "Fixed 2025 system, 5 CMIP6 models"),
        _fmt_g(df_g[(df_g["group"] == "GWL") & (df_g["subgroup"] == "GWL_1.5C")], "Global warming 1.5 °C (n = 14)", "20 year global warming level windows"),
        _fmt_g(df_g[(df_g["group"] == "GWL") & (df_g["subgroup"] == "GWL_2.0C")], "Global warming 2.0 °C (n = 12)", "20 year global warming level windows"),
        _fmt_g(df_g[(df_g["group"] == "GWL") & (df_g["subgroup"] == "GWL_3.0C")], "Global warming 3.0 °C (n = 7)", "20 year global warming level windows"),
        _fmt_g(df_g[(df_g["group"] == "R4") & (df_g["subgroup"] == "S3_4gdh_standalone") & (df_g["window_id"] == "2046_2065")], "Low temp. networks (vs 2025)", "55/25 °C networks + 2025 building heat pumps"),
        _fmt_g(df_g[(df_g["group"] == "R4") & (df_g["subgroup"] == "S4_4gdh_symmetric") & (df_g["window_id"] == "2046_2065")], "Low temp. networks (vs 2050)", "55/25 °C networks + 2050 building heat pumps"),
        _fmt_g(df_g[(df_g["group"] == "R4") & (df_g["subgroup"] == "S2_envelope_retrofit") & (df_g["window_id"] == "2046_2065")], "Building envelope retrofits", "2050 space heating conductance cuts"),
        _fmt_g(df_g[(df_g["group"] == "R4") & (df_g["subgroup"] == "S1_fossil_retire") & (df_g["window_id"] == "2046_2065")], "Fossil power retirement", "No coal, gas or oil power + 2050 grid"),
        _fmt_g(df_g[(df_g["group"] == "R4") & (df_g["subgroup"] == "S5_combined_all") & (df_g["window_id"] == "2046_2065")], "Combined 2050 transitions", "Retirement + retrofits + 55/25 °C (vs 2050)"),
    ]

    supp_table_1 = []
    pwr_names = {
        "coal_usc": "Ultra supercritical coal steam",
        "coal_sc": "Supercritical coal steam",
        "coal_subc": "Subcritical coal steam",
        "coal_cfb": "Circulating fluidized bed coal",
        "coal_igcc": "Integrated gasification combined cycle",
        "gas_ccgt": "Natural gas combined cycle",
        "gas_steam": "Natural gas steam turbine",
        "oil_steam": "Fuel oil steam turbine",
        "gas_ocgt": "Open cycle gas turbine",
        "gas_ice": "Gas internal combustion engine",
        "oil_ocgt_ice": "Oil open cycle or engine",
        "biomass_steam": "Biomass steam turbine",
        "wte_steam": "Municipal waste steam turbine",
        "nuclear_lwr": "Nuclear water reactor",
    }
    for k_p, sp_p in POWER_TECH_SPECS.items():
        supp_table_1.append([
            pwr_names.get(k_p, k_p),
            f"{sp_p.eta_lhv * 100:.1f}%",
            f"{sp_p.k_loss_lhv * 100:.1f}%",
            "NA" if sp_p.chp_cb is None else f"{sp_p.chp_cb:.2f}",
            "Yes (wet condenser or CHP)" if sp_p.recoverable_cooling else "Excluded unless CHP",
            f"{sp_p.reject_heat_per_mwh_e:.2f}",
        ])

    ind_names = {
        "cement_dry": ("Cement (dry kiln)", "t clinker"),
        "cement_wet": ("Cement (wet or semidry kiln)", "t clinker"),
        "steel_bf_bof": ("Integrated steel (blast furnace)", "t crude steel"),
        "steel_eaf": ("Electric arc furnace steel", "t crude steel"),
        "ethylene_cracker": ("Petrochemical steam cracking", "t ethylene"),
        "refining_hydroskimming": ("Oil refining (hydroskimming)", "t crude throughput"),
        "refining_medium": ("Oil refining (medium conversion)", "t crude throughput"),
        "refining_deep": ("Oil refining (deep conversion)", "t crude throughput"),
        "pulp_chemical": ("Chemical pulp mill", "t air dried pulp"),
        "pulp_mechanical": ("Mechanical pulp mill", "t air dried pulp"),
        "paper_board_packaging": ("Packaging and board mill", "t paper"),
        "paper_graphic_tissue": ("Graphic and tissue paper mill", "t paper"),
        "pulp_paper_integrated": ("Integrated pulp and paper mill", "t product"),
    }
    supp_table_2 = []
    for k_i, f_i in SEENERGIES_FACTORS.items():
        nm_i, unit_i = ind_names.get(k_i, (k_i, f_i.unit_basis))
        supp_table_2.append([
            nm_i, unit_i,
            f"{f_i.direct_3gdh_gj_per_t:.3f}",
            f"{f_i.hp_3gdh_gj_per_t:.3f}",
            f"{f_i.direct_4gdh_gj_per_t:.3f}",
            f"{f_i.hp_4gdh_gj_per_t:.3f}",
            f"{f_i.q_to_25c_gj_per_t:.3f}",
        ])

    gwl_rows = df_g[df_g["group"] == "GWL"].sort_values(["subgroup", "model", "ssp"])
    supp_table_3 = []
    for _, gr in gwl_rows.iterrows():
        lvl = gr["subgroup"].replace("GWL_", "").replace("C", " °C")
        supp_table_3.append([
            lvl,
            GCM_LABELS.get(str(gr["model"]), str(gr["model"])),
            str(gr["ssp"]).upper().replace("SSP126", "SSP1-2.6").replace("SSP245", "SSP2-4.5").replace("SSP370", "SSP3-7.0"),
            f"{int(gr['start_year'])}–{int(gr['end_year'])}",
            f"{float(gr['D_total_ej_yr']):.2f}",
            f"{float(gr['U_ej_yr']):.2f}",
            f"{float(gr['P_ej_yr']):.2f}",
        ])

    rp_top = df_rp[df_rp["valid_for_study"]].sort_values("baseline_total_demand_gj_yr", ascending=False).head(25)
    supp_table_4 = []
    for _, rr in rp_top.iterrows():
        sh_pct = 100.0 * float(rr["cal_sh_useful_gj"]) / max(1.0, float(rr["cal_sh_useful_gj"] + rr["cal_wh_useful_gj"]))
        supp_table_4.append([
            str(rr["region_id"]).replace("-", " "),
            cname(str(rr["iso3"])),
            f"{float(rr['baseline_total_demand_gj_yr']) / 1e6:.1f}",
            f"{sh_pct:.1f}%",
            "Yes" if bool(rr["has_gas"]) else "No",
            f"{float(rr['p_elec_pretax_usd_per_mwh']):.1f}",
            f"{float(rr['p_gas_pretax_usd_per_gj']):.2f}" if bool(rr["has_gas"]) else "NA",
            f"{float(rr['grid_ef_2025_tco2_per_mwh']) * 1000:.0f} → {float(rr['grid_ef_2050_tco2_per_mwh']) * 1000:.0f}",
        ])

    dh_map = df_dh.set_index("iso3")["buildings_dh_consumption_ej_yr"].to_dict() if "iso3" in df_dh.columns else {}
    v1_top = val1_p.sort_values("ember_comb_twh_2025", ascending=False).head(15)
    st_map = val1_i[val1_i["sector"] == "steel"].set_index("iso3")
    supp_table_5 = []
    for _, vr in v1_top.iterrows():
        iso = str(vr["iso3"])
        st_off = float(st_map.loc[iso, "official_2025_mt"]) if iso in st_map.index and pd.notna(st_map.loc[iso, "official_2025_mt"]) else 0.0
        st_inv = float(st_map.loc[iso, "ct_2025_mt"]) if iso in st_map.index and pd.notna(st_map.loc[iso, "ct_2025_mt"]) else 0.0
        dh_pj = float(dh_map.get(iso, 0.0)) * 1e3
        supp_table_5.append([
            cname(iso),
            f"{float(vr['ember_comb_twh_2025']):.1f} / {float(vr['ct_comb_twh_2025']):.1f}",
            f"{float(vr['ember_nuclear_gross_twh_2025']):.1f} / {float(vr['pris_nuclear_net_twh_2025']):.1f}",
            f"{st_off:.1f} / {st_inv:.1f}" if st_off > 0 else "NA",
            f"{dh_pj:.1f}" if dh_pj > 0 else "NA",
        ])

    c_base = df_c[df_c["scenario_id"] == "R1__obs__2006_2025"].sort_values(["P_gj", "U_gj"], ascending=False).head(30)
    supp_table_6 = []
    for _, cr in c_base.iterrows():
        lcoh_c = float(cr["tot_dh_cost_usd"]) / max(1.0, float(cr["H_deliv_gj"] + cr["H_peaking_gj"])) if float(cr["P_gj"]) > 0 else 0.0
        supp_table_6.append([
            cname(str(cr["iso3"])),
            f"{float(cr['Q_gj']) / 1e6:.1f}",
            f"{float(cr['D_total_gj']) / 1e6:.1f}",
            f"{float(cr['D_urban_gj']) / 1e6:.1f}",
            f"{float(cr['U_gj']) / 1e6:.1f}",
            f"{float(cr['P_gj']) / 1e6:.2f}",
            f"{int(cr['n_selected_links']):,}",
            f"{float(cr['total_capex_usd']) / 1e9:.2f}",
            f"{lcoh_c:.2f}" if float(cr["P_gj"]) > 0 else "NA",
            f"{float(cr['co2_abated_t']) / 1e6:.2f}",
        ])

    c_d_mid = df_dec[(df_dec["ssp"] == "ssp245") & (df_dec["window_id"] == "2046_2065")].groupby(["level", "entity_id"], as_index=False)[
        ["P0_pj", "P1_pj", "dP_rel_pct", "term1_demand_shift_pct", "term2_seasonal_mismatch_pct", "term3_cost_exit_pct"]
    ].mean()
    c_d_end = df_dec[(df_dec["ssp"] == "ssp370") & (df_dec["window_id"] == "2081_2100")].groupby(["level", "entity_id"], as_index=False)[
        ["dP_rel_pct"]
    ].mean().set_index("entity_id")
    top_dec = pd.concat([
        c_d_mid[c_d_mid["level"] == "global"],
        c_d_mid[c_d_mid["level"] == "country"].sort_values("P0_pj", ascending=False).head(18),
    ], ignore_index=True)

    def _sgn(val: float) -> str:
        return f"+{val:.1f}%" if val > 0 else f"{fm(val, '.1f')}%"

    supp_table_7 = []
    for _, dr in top_dec.iterrows():
        eid = str(dr["entity_id"])
        nm = "Global" if eid == "GLOBAL" else cname(eid)
        end_dp = float(c_d_end.loc[eid, "dP_rel_pct"]) if eid in c_d_end.index else 0.0
        supp_table_7.append([
            nm,
            f"{float(dr['P0_pj']):.1f}",
            f"{float(dr['P1_pj']):.1f}",
            _sgn(float(dr["term1_demand_shift_pct"])),
            _sgn(float(dr["term2_seasonal_mismatch_pct"])),
            _sgn(float(dr["term3_cost_exit_pct"])),
            _sgn(float(dr["dP_rel_pct"])),
            _sgn(end_dp),
        ])

    r3_files = sorted((SCEN_DIR / "r3_locked_assets").glob("R3__*__ssp245__2046_2065_links.parquet"))
    r3_dfs = [pd.read_parquet(f).assign(gcm_i=idx) for idx, f in enumerate(r3_files)]
    r3_all = pd.concat(r3_dfs, ignore_index=True)
    r3_all["loss_cap_b"] = np.where(r3_all["is_loss_making_fut"], r3_all["total_sunk_capex_usd"] / 1e9, 0.0)
    r3_all["loss_cnt"] = r3_all["is_loss_making_fut"].astype(int)
    c_tot_r3 = r3_dfs[0].groupby("iso3", as_index=False).agg(
        n_links=("zone_id", "count"), tot_cap_b=("total_sunk_capex_usd", lambda s: float(s.sum()) / 1e9)
    )
    c_gcm_r3 = r3_all.groupby(["gcm_i", "iso3"], as_index=False).agg(
        loss_cap_b=("loss_cap_b", "sum"), loss_cnt=("loss_cnt", "sum"), nb_fut_b=("net_benefit_fut_usd_yr", lambda s: float(s.sum()) / 1e9)
    )
    c_r3_m = c_gcm_r3.groupby("iso3", as_index=False).agg(
        m_cap=("loss_cap_b", "mean"), lo_cap=("loss_cap_b", "min"), hi_cap=("loss_cap_b", "max"),
        m_cnt=("loss_cnt", "mean"), nb_fut=("nb_fut_b", "mean"),
    ).merge(c_tot_r3, on="iso3").sort_values("tot_cap_b", ascending=False).head(18)
    supp_table_8 = []
    for _, r8 in c_r3_m.iterrows():
        pct_c = 100.0 * float(r8["m_cap"]) / max(1e-6, float(r8["tot_cap_b"]))
        nb_v = float(r8["nb_fut"])
        nb_s = f"{fm(nb_v, '.2f')}" if nb_v < 0 else f"{nb_v:.2f}"
        supp_table_8.append([
            cname(str(r8["iso3"])),
            f"{int(r8['n_links']):,}",
            f"{float(r8['tot_cap_b']):.2f}",
            f"{float(r8['m_cnt']):.0f}",
            f"{float(r8['m_cap']):.2f} ({float(r8['lo_cap']):.2f}–{float(r8['hi_cap']):.2f})",
            f"{pct_c:.1f}%",
            nb_s,
        ])

    t15_isos = c_base["iso3"].head(15).tolist()
    c_245 = df_c[(df_c["group"] == "R1") & (df_c["ssp"] == "ssp245") & (df_c["window_id"] == "2046_2065")].groupby("iso3")["P_gj"].mean() / 1e6
    c_s3 = df_c[(df_c["group"] == "R4") & (df_c["subgroup"] == "S3_4gdh_standalone") & (df_c["window_id"] == "2046_2065")].groupby("iso3")["P_gj"].mean() / 1e6
    c_s4 = df_c[(df_c["group"] == "R4") & (df_c["subgroup"] == "S4_4gdh_symmetric") & (df_c["window_id"] == "2046_2065")].groupby("iso3")["P_gj"].mean() / 1e6
    c_s2 = df_c[(df_c["group"] == "R4") & (df_c["subgroup"] == "S2_envelope_retrofit") & (df_c["window_id"] == "2046_2065")].groupby("iso3")["P_gj"].mean() / 1e6
    c_s1 = df_c[(df_c["group"] == "R4") & (df_c["subgroup"] == "S1_fossil_retire") & (df_c["window_id"] == "2046_2065")].groupby("iso3")["P_gj"].mean() / 1e6
    c_s5 = df_c[(df_c["group"] == "R4") & (df_c["subgroup"] == "S5_combined_all") & (df_c["window_id"] == "2046_2065")].groupby("iso3")["P_gj"].mean() / 1e6
    b_p0 = c_base.set_index("iso3")["P_gj"] / 1e6

    supp_table_9 = []
    for iso in t15_isos:
        supp_table_9.append([
            cname(str(iso)),
            f"{float(b_p0.get(iso, 0.0)):.1f}",
            f"{float(c_245.get(iso, 0.0)):.1f}",
            f"{float(c_s3.get(iso, 0.0)):.1f}",
            f"{float(c_s4.get(iso, 0.0)):.1f}",
            f"{float(c_s2.get(iso, 0.0)):.1f}",
            f"{float(c_s1.get(iso, 0.0)):.2f}",
            f"{float(c_s5.get(iso, 0.0)):.2f}",
        ])

    supp_table_10 = []
    for _, rr in rob_csv.iterrows():
        supp_table_10.append([
            str(rr["specification"]),
            f"{float(rr['P_2006_2025_ej_yr']):.2f}",
            f"{float(rr['P_2046_2065_mean_ej_yr']):.2f} ({float(rr['P_2046_2065_min_ej_yr']):.2f}–{float(rr['P_2046_2065_max_ej_yr']):.2f})",
            _sgn(float(rr["rel_change_mean_pct"])),
            f"{float(rr['nb_2006_2025_busd_yr']):.2f} → {float(rr['nb_2046_2065_mean_busd_yr']):.2f}",
            f"{float(rr['co2_2006_2025_mt_yr']):.1f} → {float(rr['co2_2046_2065_mean_mt_yr']):.1f}",
        ])

    payload = {
        "ed_table_1": ed_table_1,
        "ed_table_2": ed_table_2,
        "supp_table_1": supp_table_1,
        "supp_table_2": supp_table_2,
        "supp_table_3": supp_table_3,
        "supp_table_4": supp_table_4,
        "supp_table_5": supp_table_5,
        "supp_table_6": supp_table_6,
        "supp_table_7": supp_table_7,
        "supp_table_8": supp_table_8,
        "supp_table_9": supp_table_9,
        "supp_table_10": supp_table_10,
        "r3_global_summary": df_r3_sum.to_dict("records"),
    }


    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"[PASS] Verified & audited 12 table payloads -> {out_json}", flush=True)
    return payload


# ==============================================================================
# EXCEL WORKBOOKS (`Supplementary_Data_1.xlsx` & `Source_Data_Figures.xlsx`)
# ==============================================================================
def _style_sheet(ws, df: pd.DataFrame, title: str, add_sum_cols: list[int] | None = None) -> None:
    font_title = Font(name="Arial", size=11, bold=True, color="000000")
    font_hdr = Font(name="Arial", size=9.5, bold=True, color="FFFFFF")
    font_body = Font(name="Arial", size=9, bold=False, color="000000")
    font_tot = Font(name="Arial", size=9.5, bold=True, color="000000")
    fill_hdr = PatternFill(start_color="36558F", end_color="36558F", fill_type="solid")
    fill_tot = PatternFill(start_color="F2F6FA", end_color="F2F6FA", fill_type="solid")
    thin_bottom = Border(bottom=Side(style="thin", color="D8E0E8"))
    tot_border = Border(top=Side(style="thin", color="000000"), bottom=Side(style="double", color="000000"))

    ws.cell(row=1, column=1, value=title).font = font_title
    for c_idx, col_name in enumerate(df.columns, 1):
        cell = ws.cell(row=3, column=c_idx, value=str(col_name))
        cell.font = font_hdr
        cell.fill = fill_hdr
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

    n_rows = len(df)
    for r_idx, row_vals in enumerate(df.itertuples(index=False), 4):
        for c_idx, val in enumerate(row_vals, 1):
            if pd.isna(val):
                v_clean = None
            elif isinstance(val, (np.integer, int)):
                v_clean = int(val)
            elif isinstance(val, (np.floating, float)):
                v_clean = float(val)
            else:
                v_clean = str(val)
            c = ws.cell(row=r_idx, column=c_idx, value=v_clean)
            c.font = font_body
            c.border = thin_bottom
            if isinstance(v_clean, float):
                c.number_format = "0.00"
            elif isinstance(v_clean, int):
                c.number_format = "#,##0"

    if add_sum_cols and n_rows > 0:
        tot_row = 4 + n_rows
        lbl_c = ws.cell(row=tot_row, column=1, value="Total (Formula)")
        lbl_c.font = font_tot
        lbl_c.fill = fill_tot
        lbl_c.border = tot_border
        for c_idx in range(2, len(df.columns) + 1):
            col_let = get_column_letter(c_idx)
            tc = ws.cell(row=tot_row, column=c_idx)
            tc.font = font_tot
            tc.fill = fill_tot
            tc.border = tot_border
            if c_idx in add_sum_cols:
                tc.value = f"=SUM({col_let}4:{col_let}{3 + n_rows})"
                tc.number_format = "0.00"

    ws.freeze_panes = "A4"
    for c_idx, col_name in enumerate(df.columns, 1):
        col_let = get_column_letter(c_idx)
        ws.column_dimensions[col_let].width = max(14, min(32, len(str(col_name)) + 5))


def build_excel_workbooks() -> None:
    MS_DIR.mkdir(parents=True, exist_ok=True)

    wb1 = openpyxl.Workbook()
    ws_readme = wb1.active
    ws_readme.title = "README"
    readme_df = pd.DataFrame([
        ["Global_190_Scenarios", "Global summary across all 190 climate, global warming level, transition and sensitivity runs (1946–2100)"],
        ["Baseline_Countries_2025", "National 2006–2025 recoverable waste heat Q, building demand D, urban area demand, technical U0 and economic P0"],
        ["MidCentury_Decomposition", "National additive decomposition into heating demand, seasonal matching and economic viability (2046–2065 SSP2-4.5)"],
        ["GWL_Crossings", "Exact 20 year CMIP6 global warming level windows at 1.5, 2.0 and 3.0 °C above 1850–1900"],
        ["Calibrated_183_Regions", "Calibrated space heating and water heating parameters, energy prices and grid emission intensities across 183 regions"],
    ], columns=["Worksheet", "Contents and units"])
    _style_sheet(ws_readme, readme_df, "Supplementary Data 1 | Climate warming reduces the global economic potential of waste heat for buildings")

    df_g = pd.read_parquet(SCEN_DIR / "global_scenarios_summary.parquet")
    keep_g = [
        "scenario_id", "group", "subgroup", "model", "ssp", "window_id", "start_year", "end_year",
        "Q_ej_yr", "D_total_ej_yr", "D_urban_ej_yr", "U_ej_yr", "P_ej_yr",
        "P_power_ej_yr", "P_industry_ej_yr", "P_wastewater_ej_yr",
        "net_benefit_billion_usd_yr", "total_capex_billion_usd", "mean_dh_lcoh_usd_per_gj", "co2_abated_mt_yr", "n_selected_links",
    ]
    ws_g = wb1.create_sheet("Global_190_Scenarios")
    _style_sheet(ws_g, df_g[keep_g], "Global Scenario Summary (190 model runs, 1946–2100, constant 2025 USD)")

    df_c0 = pd.read_parquet(SCEN_DIR / "all_scenarios_by_country.parquet")
    c_base = df_c0[df_c0["scenario_id"] == "R1__obs__2006_2025"].copy()
    c_base["country_name"] = [cname(c) for c in c_base["iso3"]]
    for col_gj in ("Q_gj", "D_total_gj", "D_urban_gj", "U_gj", "P_gj"):
        c_base[col_gj.replace("_gj", "_pj_yr")] = c_base[col_gj] / 1e6
    c_base["capex_billion_usd"] = c_base["total_capex_usd"] / 1e9
    c_base["net_benefit_billion_usd_yr"] = c_base["net_benefit_usd"] / 1e9
    c_base["co2_abated_mt_yr"] = c_base["co2_abated_t"] / 1e6
    c_base_out = c_base[[
        "iso3", "country_name", "Q_pj_yr", "D_total_pj_yr", "D_urban_pj_yr", "U_pj_yr", "P_pj_yr",
        "n_selected_links", "capex_billion_usd", "net_benefit_billion_usd_yr", "co2_abated_mt_yr",
    ]].sort_values(["P_pj_yr", "U_pj_yr"], ascending=False)
    ws_cb = wb1.create_sheet("Baseline_Countries_2025")
    _style_sheet(ws_cb, c_base_out, "National Present Potentials (2006–2025 observed climate, 2025 system)", add_sum_cols=[3, 4, 5, 6, 7, 8, 9, 10, 11])

    df_dec = pd.read_parquet(SCEN_DIR / "r2_decomposition/r2_eq6_decomposition_all.parquet")
    dec_c = df_dec[(df_dec["level"] == "country") & (df_dec["ssp"] == "ssp245") & (df_dec["window_id"] == "2046_2065")].groupby("entity_id", as_index=False)[
        ["D0_pj", "D1_pj", "U0_pj", "U1_pj", "P0_pj", "P1_pj", "dP_rel_pct", "term1_demand_shift_pct", "term2_seasonal_mismatch_pct", "term3_cost_exit_pct"]
    ].mean().sort_values("P0_pj", ascending=False)
    dec_c.insert(1, "country_name", [cname(c) for c in dec_c["entity_id"]])
    dec_c = dec_c.rename(columns={
        "term1_demand_shift_pct": "heating_demand_pct",
        "term2_seasonal_mismatch_pct": "seasonal_matching_pct",
        "term3_cost_exit_pct": "economic_viability_pct",
    })
    ws_dc = wb1.create_sheet("MidCentury_Decomposition")
    _style_sheet(ws_dc, dec_c, "National Additive Decomposition (2046–2065 SSP2-4.5, five model mean)")

    gwl_df = df_g[df_g["group"] == "GWL"][["scenario_id", "subgroup", "model", "ssp", "start_year", "end_year", "D_total_ej_yr", "U_ej_yr", "P_ej_yr", "co2_abated_mt_yr"]].copy()
    ws_gwl = wb1.create_sheet("GWL_Crossings")
    _style_sheet(ws_gwl, gwl_df, "Exact 20 Year Global Warming Level Windows (1.5, 2.0 and 3.0 °C)")

    df_rp = pd.read_parquet(DATA_PROC / "sinks_2025/calibrated_region_parameters_2025.parquet")
    rp_out = df_rp[df_rp["valid_for_study"]][[
        "region_id", "iso3", "cal_source", "v25_m3", "mean_hdd14_2006_2025",
        "k_c_gj_per_m3_hdd", "w_c_gj_per_m3_day", "baseline_total_demand_gj_yr",
        "has_gas", "p_elec_pretax_usd_per_mwh", "p_elec_retail_usd_per_mwh",
        "p_gas_pretax_usd_per_gj", "p_gas_retail_usd_per_gj",
        "grid_ef_2025_tco2_per_mwh", "grid_ef_2050_tco2_per_mwh",
    ]].copy()
    ws_rp = wb1.create_sheet("Calibrated_183_Regions")
    _style_sheet(ws_rp, rp_out, "Calibrated Heating Parameters, Energy Tariffs and Grid Emission Intensities (183 regions)")

    p_sd1 = MS_DIR / "Supplementary_Data_1.xlsx"
    wb1.save(p_sd1)
    print(f"[PASS] Wrote {p_sd1}", flush=True)

    wb2 = openpyxl.Workbook()
    wb2.remove(wb2.active)
    fig_csvs = [
        ("Fig2_Trajectory", SRC_DIR / "Figure2_source_data.csv", "Source Data for Fig. 2 | Global 1946–2100 Economic Waste Heat Potential Trajectory"),
        ("Fig3_Decomposition", SRC_DIR / "Figure3_source_data.csv", "Source Data for Fig. 3 | Global, National and Stratified Additive Decomposition (2046–2065 SSP2-4.5)"),
        ("Fig4_PresentNetworks", SRC_DIR / "Figure4_source_data.csv", "Source Data for Fig. 4 | Stranded Pipe Capital, Cost Savings and Carbon Abatement Across Fixed Networks"),
        ("Fig5_Transitions", SRC_DIR / "Figure5_source_data.csv", "Source Data for Fig. 5 | Mid Century 2050 Energy Transitions and Sectoral Supply"),
        ("ED_Fig1d_MILP", SRC_DIR / "Extended_Data_Fig1d_milp_source_data.csv", "Source Data for Extended Data Fig. 1d | Exact HiGHS MILP vs Priority Queue Benchmark"),
        ("ED_Fig2_Dispatch", SRC_DIR / "Extended_Data_Fig2_dispatch_and_grades_source_data.csv", "Source Data for Extended Data Fig. 2 | Monthly Waste Heat Dispatch by Upgrade Technology"),
        ("ED_Fig3_Sensitivity", SRC_DIR / "Extended_Data_Fig3_sensitivity_levels_source_data.csv", "Source Data for Extended Data Fig. 3 | Ten Specification Sensitivity Levels"),
        ("ED_Fig4_Pathways", SRC_DIR / "Extended_Data_Fig4_pathway_decomposition_source_data.csv", "Source Data for Extended Data Fig. 4 | Global Decomposition Across SSP1-2.6, SSP2-4.5 and SSP3-7.0"),
    ]
    for sheet_nm, csv_p, title_s in fig_csvs:
        if csv_p.exists():
            ws = wb2.create_sheet(sheet_nm)
            _style_sheet(ws, pd.read_csv(csv_p), title_s)
    p_src = MS_DIR / "Source_Data_Figures.xlsx"
    wb2.save(p_src)
    print(f"[PASS] Wrote {p_src}", flush=True)


def main() -> None:
    df_glob = pd.read_parquet(SCEN_DIR / "global_scenarios_summary.parquet")
    df_country = pd.read_parquet(SCEN_DIR / "all_scenarios_by_country.parquet")

    audits = []
    audits.append(build_supp_fig_1())
    audits.append(build_supp_fig_2(df_glob))
    audits.append(build_supp_fig_3())
    audits.append(build_supp_fig_4(df_glob))
    audits.append(build_supp_fig_5())
    audits.append(build_supp_fig_6())
    audits.append(build_supp_fig_7())
    audits.append(build_supp_fig_8(df_country))

    print("\n=== SUPPLEMENTARY FIGURES 1–8 EXPORT SUMMARY ===")
    for a in audits:
        print(f"  {a['stem']}: {a['w_mm']:.2f}x{a['h_mm']:.2f} mm | {a['mb']:.2f} MB | {a['n_texts']} text objects | bbox touches = {len(a['overlaps'])}")

    export_verified_tables_json(SI_DIR / "si_tables_payload.json")
    build_excel_workbooks()
    tot_ov = sum(len(a["overlaps"]) for a in audits)
    assert tot_ov == 0, f"Total SI text overlaps = {tot_ov}"


if __name__ == "__main__":
    main()
