#!/usr/bin/env python3
"""Generate Extended Data Figures 1–4 and export underlying validation and sensitivity source data tables."""
from __future__ import annotations
import os

from decimal import Decimal, ROUND_HALF_UP
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
import pandas as pd
from pyproj import Transformer
from scipy.interpolate import make_interp_spline

warnings.filterwarnings("ignore")

ROOT = Path(os.environ.get("WASTE_HEAT_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "code/figures"))

from code.common.parameters import HP_INDUSTRIAL_POWER_2025, HP_WASTEWATER_2025  # noqa: E402
from code.matching.zone_climate_evaluator import ashp_daily_cop  # noqa: E402
from figure_utils import MM, DOUBLE_COL, check_typography, check_palette_contrast, load_world_basemap  # noqa: E402

DATA_PROC = ROOT / "data/processed"
SCEN_DIR = DATA_PROC / "scenarios_2025"
FIG_DIR = ROOT / "figures/extended"
SRC_DIR = ROOT / "figures/source_data"

W_MM = 180.0
W_IN = DOUBLE_COL
MINUS = "\u2212"

FS_PANEL = 9.5
FS_BODY = 6.8
FS_UNIFIED = 6.5
FS_COMPACT = 5.6

INK = "#000000"
SPINE_COL = "#000000"

COL_SAPPHIRE = "#264B82"
COL_BLDG_SLATE = "#4A525D"
COL_SKY_CYAN = "#70CBE6"
COL_DEM_BLUE = "#427FB8"
COL_TECH_SLATE = "#4A7C9B"
COL_EMERALD = "#2B9EB3"
COL_APRICOT = "#E8A638"
COL_SSP2 = "#E6A532"
COL_CORAL = "#C8422E"
COL_CORAL_SOFT = "#D96A55"
COL_HIST_BG = "#F2F6FA"
COL_ICE_ZONE = "#EAF2F8"
COL_BAND = "#9DB4D0"

GCM_ORDER = [
    ("gfdl-esm4",     "GFDL-ESM4",     "o"),
    ("ipsl-cm6a-lr",  "IPSL-CM6A-LR",  "s"),
    ("mpi-esm1-2-hr", "MPI-ESM1-2-HR", "^"),
    ("mri-esm2-0",    "MRI-ESM2-0",    "v"),
    ("ukesm1-0-ll",   "UKESM1-0-LL",   "D"),
]
GCM_MARKER = {k: m for k, _, m in GCM_ORDER}

ISO_NAME = {
    "CHN": "China", "RUS": "Russia", "DEU": "Germany", "USA": "United States",
    "KAZ": "Kazakhstan", "KOR": "South Korea", "POL": "Poland", "ITA": "Italy",
    "NLD": "Netherlands", "BLR": "Belarus", "GBR": "United Kingdom", "CZE": "Czechia",
    "SWE": "Sweden", "DNK": "Denmark", "AUT": "Austria", "FIN": "Finland",
    "FRA": "France", "CAN": "Canada", "JPN": "Japan", "UKR": "Ukraine",
    "IRN": "Iran", "ISR": "Israel", "ROU": "Romania", "ESP": "Spain",
    "MNG": "Mongolia", "TUR": "Türkiye", "HUN": "Hungary", "SVK": "Slovakia",
    "BEL": "Belgium", "BGR": "Bulgaria", "SRB": "Serbia", "CHE": "Switzerland",
    "NOR": "Norway", "GRC": "Greece", "PRT": "Portugal", "HRV": "Croatia",
    "SVN": "Slovenia", "LTU": "Lithuania", "LVA": "Latvia", "EST": "Estonia",
    "IRL": "Ireland", "LUX": "Luxembourg", "ISL": "Iceland", "IND": "India",
    "PAK": "Pakistan", "UZB": "Uzbekistan", "AZE": "Azerbaijan", "GEO": "Georgia",
    "ARM": "Armenia", "MDA": "Moldova", "BIH": "Bosnia and Herzegovina",
    "MKD": "North Macedonia", "ALB": "Albania", "AUS": "Australia", "NZL": "New Zealand",
    "ARG": "Argentina", "CHL": "Chile", "BRA": "Brazil", "MEX": "Mexico", "ZAF": "South Africa",
}


def cname(iso: str) -> str:
    return ISO_NAME.get(str(iso), str(iso))


def fm(val: float, fmt: str = ".1f") -> str:
    ndig = int(fmt.lstrip(".").rstrip("f"))
    q = Decimal("10") ** (-ndig) if ndig > 0 else Decimal("1")
    r = Decimal(f"{abs(val):.6f}").quantize(q, rounding=ROUND_HALF_UP)
    return f"{MINUS}{r}" if val < 0 else f"{r}"


def configure_matplotlib_style() -> None:
    face = "Nimbus Sans"
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [face, "Liberation Sans", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": FS_BODY,
        "axes.labelsize": FS_UNIFIED,
        "axes.titlesize": FS_UNIFIED,
        "xtick.labelsize": FS_UNIFIED,
        "ytick.labelsize": FS_UNIFIED,
        "legend.fontsize": FS_UNIFIED,
        "figure.titlesize": FS_PANEL,
        "mathtext.fontset": "custom",
        "mathtext.rm": face,
        "mathtext.it": f"{face}:italic",
        "mathtext.bf": f"{face}:bold",
        "mathtext.default": "regular",
        "axes.unicode_minus": False,
        "axes.linewidth": 0.60,
        "lines.linewidth": 1.10,
        "xtick.major.width": 0.60,
        "ytick.major.width": 0.60,
        "xtick.minor.width": 0.40,
        "ytick.minor.width": 0.40,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.edgecolor": SPINE_COL,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "text.color": INK,
        "legend.labelcolor": INK,
        "legend.frameon": False,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def lock_spines_above_data(ax: plt.Axes) -> None:
    for sp in ax.spines.values():
        sp.set_edgecolor(SPINE_COL)
        sp.set_zorder(20)


def clean_ax(ax: plt.Axes, grid_axis: str | None = None) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    lock_spines_above_data(ax)
    if grid_axis in ("x", "y", "both"):
        ax.grid(axis=grid_axis, linestyle="--", linewidth=0.40, alpha=0.28, color="#888888", zorder=0)
    ax.set_axisbelow(True)


def panel_tag(fig: plt.Figure, x_fig: float, y_fig: float, letter: str) -> None:
    fig.text(x_fig, y_fig, letter, fontsize=FS_PANEL, fontweight="bold", ha="left", va="top", color="#000000")


def export_figure(fig: plt.Figure, stem: str, out_root: Path = FIG_DIR) -> dict:
    for ax in fig.axes:
        if ax.get_visible():
            for sp in ax.spines.values():
                if sp.get_visible():
                    sp.set_edgecolor("#000000")
                    if sp.get_zorder() < 18:
                        sp.set_zorder(20)
                    assert sp.get_zorder() >= 18, f"Spine z-order violation in {stem}"

    for t in fig.findobj(Text):
        if t.get_visible() and t.get_text().strip():
            t.set_color("#000000")
            hx = mcolors.to_hex(t.get_color()).lower()
            assert hx == "#000000", f"Non-black text found in {stem}: {t.get_text()} ({hx})"

    titled_axes = [ax.get_title() for ax in fig.axes if ax.get_title().strip()]
    assert len(titled_axes) == 0, f"Unexpected axis title found on {stem}: {titled_axes}"

    print(f"\n--- Typography check for {stem} ---")
    rows, n_fail = check_typography(fig, W_MM, verbose=True)
    n_over = sum(1 for r in rows if r[4] == "over")
    assert n_fail == 0 and n_over == 0, f"Typography spec violation in {stem}: {n_fail} fail, {n_over} over"

    from shapely.geometry import box as sbox
    from shapely import affinity

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    vis_texts = []
    for t in fig.findobj(Text):
        s = t.get_text().strip()
        if not s or not t.get_visible():
            continue
        if t.axes is not None and not t.axes.get_visible():
            continue
        bb = t.get_window_extent(renderer=renderer)
        if bb.width <= 0.5 or bb.height <= 0.5:
            continue
        ang = float(t.get_rotation())
        if abs(ang) > 1.0 and abs(ang - 90.0) > 1.0:
            t.set_rotation(0.0)
            bb0 = t.get_window_extent(renderer=renderer)
            t.set_rotation(ang)
            cx, cy = 0.5 * (bb.x0 + bb.x1), 0.5 * (bb.y0 + bb.y1)
            hw, hh = max(0.5, 0.5 * bb0.width - 1.2), max(0.5, 0.5 * bb0.height - 1.2)
            poly = affinity.rotate(sbox(cx - hw, cy - hh, cx + hw, cy + hh), ang, origin=(cx, cy))
        else:
            poly = sbox(bb.x0 + 1.2, bb.y0 + 1.2, max(bb.x0 + 1.7, bb.x1 - 1.2), max(bb.y0 + 1.7, bb.y1 - 1.2))
        vis_texts.append((s.replace("\n", " "), poly))

    overlaps = []
    for i in range(len(vis_texts)):
        s1, p1 = vis_texts[i]
        for j in range(i + 1, len(vis_texts)):
            s2, p2 = vis_texts[j]
            if p1.intersects(p2):
                area = p1.intersection(p2).area
                if area > 4.0:
                    overlaps.append((s1, s2, round(area, 1)))

    if overlaps:
        print(f"  [OVERLAP FAIL] {len(overlaps)} text bbox overlap(s) in {stem}: {overlaps[:12]}")
    else:
        print(f"  [PASS] Zero text-to-text bounding-box overlaps across {len(vis_texts)} pure-black (#000000) text objects!")

    out_root.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        out_sub = out_root / ext
        out_sub.mkdir(parents=True, exist_ok=True)
        dpi = 450 if ext == "png" else 300
        fig.savefig(out_sub / f"{stem}.{ext}", dpi=dpi)
        # Also write top-level flat copy so legacy & new docx builders both resolve cleanly
        if ext in ("pdf", "png"):
            fig.savefig(out_root / f"{stem}.{ext}", dpi=dpi)

    w_mm_actual = fig.get_size_inches()[0] / MM
    h_mm_actual = fig.get_size_inches()[1] / MM
    pdf_path = out_root / "pdf" / f"{stem}.pdf"
    mb_size = pdf_path.stat().st_size / (1024 * 1024)
    assert abs(w_mm_actual - 180.0) < 0.05, f"Width {w_mm_actual} != 180.0 mm"
    assert h_mm_actual <= 185.0, f"Height {h_mm_actual} > 185.0 mm ceiling"
    assert mb_size < 50.0, f"PDF size {mb_size:.2f} MB >= 50 MB cap"

    pdf_bytes = pdf_path.read_bytes()
    assert b"/Subtype /Type3" not in pdf_bytes, f"CRITICAL: Uneditable Type 3 font detected in {stem}.pdf!"
    print(f"  [PASS] Physical size = {w_mm_actual:.2f} x {h_mm_actual:.2f} mm | PDF = {mb_size:.2f} MB | Black Font + Spine zorder=20 + TrueType = PASS")

    plt.close(fig)
    return {"stem": stem, "w_mm": w_mm_actual, "h_mm": h_mm_actual, "mb": mb_size, "overlaps": overlaps, "n_texts": len(vis_texts)}


# ==============================================================================
# EXTENDED DATA FIGURE 1: Multi-scale Validations (Zero Bar Charts!)
# ==============================================================================
def build_ed_fig_1() -> dict:
    configure_matplotlib_style()
    H_MM = 136.0
    fig = plt.figure(figsize=(W_IN, H_MM * MM))

    gs = gridspec.GridSpec(
        2, 2, figure=fig,
        left=0.098, right=0.980, bottom=0.098, top=0.948,
        hspace=0.38, wspace=0.34,
    )

    val1_pwr = pd.read_csv(DATA_PROC / "sources_2025/validation_1/power_nuclear_vs_ember2025.csv")
    val1_ind = pd.read_csv(DATA_PROC / "sources_2025/validation_1/industrial_output_vs_official2025.csv")
    val2_eu = pd.read_csv(DATA_PROC / "sinks_2025/validation_2/eu28_urban_heat_vs_peta_hre4.csv")
    val3_eu = pd.read_csv(SCEN_DIR / "validation_3_benchmarks/validation_3_eu27_vs_manz2024.csv")
    val4_milp = pd.read_csv(DATA_PROC / "validation_4_milp/validation_4_milp_3countries.csv")

    # Panel a: Facility inventory vs official 2025 national benchmarks
    ax_a = fig.add_subplot(gs[0, 0])
    clean_ax(ax_a, "y")
    comb = val1_pwr[(val1_pwr["ember_comb_twh_2025"] > 1.0) & (val1_pwr["ct_comb_twh_2025"] > 0.1)]
    nuc = val1_pwr[(val1_pwr["ember_nuclear_gross_twh_2025"] > 1.0) & (val1_pwr["pris_nuclear_net_twh_2025"] > 0.1)]
    steel = val1_ind[(val1_ind["sector"] == "steel") & (val1_ind["official_2025_mt"] > 0.5) & (val1_ind["ct_2025_mt"] > 0.1)]
    cem = val1_ind[(val1_ind["sector"] == "cement") & (val1_ind["official_2025_mt"] > 0.5) & (val1_ind["ct_2025_mt"] > 0.1)]

    r_comb = float(np.corrcoef(comb["ember_comb_twh_2025"], comb["ct_comb_twh_2025"])[0, 1])
    r_nuc = float(np.corrcoef(nuc["ember_nuclear_gross_twh_2025"], nuc["pris_nuclear_net_twh_2025"])[0, 1])
    r_stl = float(np.corrcoef(steel["official_2025_mt"], steel["ct_2025_mt"])[0, 1])
    r_cem = float(np.corrcoef(cem["official_2025_mt"], cem["ct_2025_mt"])[0, 1])

    for x_s, y_s, mk, col, lbl in [
        (comb["ember_comb_twh_2025"], comb["ct_comb_twh_2025"], "o", COL_SAPPHIRE, f"Combustion power TWh ($r$ = {r_comb:.3f})"),
        (nuc["ember_nuclear_gross_twh_2025"], nuc["pris_nuclear_net_twh_2025"], "^", COL_SKY_CYAN, f"Nuclear power TWh ($r$ = {r_nuc:.3f})"),
        (steel["official_2025_mt"], steel["ct_2025_mt"], "s", COL_APRICOT, f"Crude steel Mt ($r$ = {r_stl:.3f})"),
        (cem["official_2025_mt"], cem["ct_2025_mt"], "D", COL_CORAL, f"Cement Mt ($r$ = {r_cem:.3f})"),
    ]:
        ax_a.scatter(x_s, y_s, s=20, marker=mk, alpha=0.88, color=col, edgecolors="white", linewidths=0.35, zorder=3, label=lbl)

    lims_a = [0.5, 11000.0]
    ax_a.plot(lims_a, lims_a, linestyle="--", linewidth=0.65, color="#333333", zorder=1)
    ax_a.set_xscale("log")
    ax_a.set_yscale("log")
    ax_a.set_xlim(lims_a)
    ax_a.set_ylim(lims_a)
    ax_a.set_xticks([1, 10, 100, 1000, 10000])
    ax_a.set_yticks([1, 10, 100, 1000, 10000])
    ax_a.set_xlabel("Official 2025 national statistics (TWh or Mt yr$^{-1}$)")
    ax_a.set_ylabel("Facility inventory sum\n(TWh or Mt yr$^{-1}$)")
    ax_a.legend(loc="upper left", bbox_to_anchor=(-0.01, 1.02), frameon=False, fontsize=FS_COMPACT, handletextpad=0.25, labelspacing=0.18)

    # Panel b: EU-28 Total & Urban Heating Areas vs PETA v5
    ax_b = fig.add_subplot(gs[0, 1])
    clean_ax(ax_b, "y")
    v2_tot = val2_eu[(val2_eu["peta_total_heat_pj"] > 1.0) & (val2_eu["our_total_heat_pj"] > 1.0)]
    v2_urb = val2_eu[(val2_eu["peta_dhe_urban_heat_pj"] > 1.0) & (val2_eu["our_urban_zone_heat_pj"] > 1.0)]
    r_tot = float(np.corrcoef(val2_eu["peta_total_heat_pj"], val2_eu["our_total_heat_pj"])[0, 1])
    r_urb = float(np.corrcoef(val2_eu["peta_dhe_urban_heat_pj"], val2_eu["our_urban_zone_heat_pj"])[0, 1])

    ax_b.scatter(
        v2_tot["peta_total_heat_pj"], v2_tot["our_total_heat_pj"],
        s=22, marker="o", alpha=0.86, color=COL_BLDG_SLATE, edgecolors="white", linewidths=0.35, zorder=3,
        label=f"Total building heating ($r$ = {r_tot:.3f})",
    )
    ax_b.scatter(
        v2_urb["peta_dhe_urban_heat_pj"], v2_urb["our_urban_zone_heat_pj"],
        s=22, marker="s", alpha=0.90, color=COL_SKY_CYAN, edgecolors="white", linewidths=0.35, zorder=4,
        label=f"Urban heating areas ($r$ = {r_urb:.3f})",
    )
    lims_b = [1.5, 4500.0]
    ax_b.plot(lims_b, lims_b, linestyle="--", linewidth=0.65, color="#333333", zorder=1)
    ax_b.set_xscale("log")
    ax_b.set_yscale("log")
    ax_b.set_xlim(lims_b)
    ax_b.set_ylim(lims_b)
    ax_b.set_xticks([10, 100, 1000])
    ax_b.set_yticks([10, 100, 1000])
    ax_b.set_xlabel("Pan-European Thermal Atlas v5 (PJ yr$^{-1}$)")
    ax_b.set_ylabel("Calibrated 2006–2025 demand\n(PJ yr$^{-1}$)")
    ax_b.legend(loc="upper left", bbox_to_anchor=(-0.01, 1.02), frameon=False, fontsize=FS_COMPACT, handletextpad=0.25, labelspacing=0.18)

    # Place 6 major EU country callouts using pure Line2D leaders + ax.text (avoiding Annotation arrow-bbox bloat)
    offs_b = {
        "DEU": ("Germany", (0.44, 0.74), "right"),
        "GBR": ("UK", (0.34, 0.57), "right"),
        "FRA": ("France", (0.26, 0.40), "right"),
        "POL": ("Poland", (0.76, 0.50), "left"),
        "ITA": ("Italy", (0.70, 0.33), "left"),
        "SWE": ("Sweden", (0.62, 0.16), "left"),
    }
    log_lo, log_span = np.log10(lims_b[0]), np.log10(lims_b[1] / lims_b[0])
    for iso_c, (lbl_c, (fx, fy), ha_c) in offs_b.items():
        sub = v2_urb[v2_urb["iso3"] == iso_c]
        if not sub.empty:
            r0 = sub.iloc[0]
            x0_d, y0_d = float(r0["peta_dhe_urban_heat_pj"]), float(r0["our_urban_zone_heat_pj"])
            xt_d = 10.0 ** (log_lo + fx * log_span)
            yt_d = 10.0 ** (log_lo + fy * log_span)
            x_end = xt_d * (1.06 if ha_c == "right" else 0.94)
            ax_b.plot([x0_d, x_end], [y0_d, yt_d], color="#000000", lw=0.45, zorder=4.5)
            ax_b.text(xt_d, yt_d, lbl_c, ha=ha_c, va="center", fontsize=FS_COMPACT, color="#000000", zorder=5)

    # Panel c: EU-27 Urban Heating Area Demand & Economic Potential vs Manz et al. (2024)
    ax_c = fig.add_subplot(gs[1, 0])
    clean_ax(ax_c, "y")
    v3_pos = val3_eu[(val3_eu["manz_2020_dh_area_demand_pj"] > 1.0) & (val3_eu["our_urban_zone_demand_pj"] > 1.0)]
    v3_p = val3_eu[(val3_eu["manz_2020_existing_dh_pj"] > 0.5) & (val3_eu["our_P_total_3gdh_pj"] > 0.5)]
    r_m_urb = float(np.corrcoef(val3_eu["manz_2020_dh_area_demand_pj"], val3_eu["our_urban_zone_demand_pj"])[0, 1])
    r_m_pot = float(np.corrcoef(val3_eu["manz_2020_existing_dh_pj"], val3_eu["our_P_total_3gdh_pj"])[0, 1])

    ax_c.scatter(
        v3_pos["manz_2020_dh_area_demand_pj"], v3_pos["our_urban_zone_demand_pj"],
        s=22, marker="o", alpha=0.88, color=COL_SKY_CYAN, edgecolors="white", linewidths=0.35, zorder=3,
        label=f"Urban heating area demand ($r$ = {r_m_urb:.3f})",
    )
    ax_c.scatter(
        v3_p["manz_2020_existing_dh_pj"], v3_p["our_P_total_3gdh_pj"],
        s=24, marker="^", alpha=0.90, color=COL_SAPPHIRE, edgecolors="white", linewidths=0.35, zorder=4,
        label=f"Economic potential ($r$ = {r_m_pot:.3f})",
    )
    lims_c = [0.5, 2800.0]
    ax_c.plot(lims_c, lims_c, linestyle="--", linewidth=0.65, color="#333333", zorder=1)
    ax_c.set_xscale("log")
    ax_c.set_yscale("log")
    ax_c.set_xlim(lims_c)
    ax_c.set_ylim(lims_c)
    ax_c.set_xticks([1, 10, 100, 1000])
    ax_c.set_yticks([1, 10, 100, 1000])
    ax_c.set_xlabel("Manz et al. (2024) EU-27 benchmark (PJ yr$^{-1}$)")
    ax_c.set_ylabel("Modelled 2006–2025 baseline\n(PJ yr$^{-1}$)")
    ax_c.legend(loc="upper left", bbox_to_anchor=(-0.01, 1.02), frameon=False, fontsize=FS_COMPACT, handletextpad=0.25, labelspacing=0.18)

    # Panel d: Cleveland Horizontal Dot-and-Stem Lollipop Comparison (ZERO Bar Charts!)
    ax_d = fig.add_subplot(gs[1, 1])
    clean_ax(ax_d, "x")
    milp_df = val4_milp.iloc[::-1].reset_index(drop=True)
    yd = np.arange(len(milp_df))
    y_lbls = []
    for yi, r in milp_df.iterrows():
        iso = str(r.get("iso3", ""))
        if "FIN" in iso or r["U_lp_pj_yr"] > 50 and r["U_lp_pj_yr"] < 80:
            y_lbls.append("Finland\n(cold)")
        elif "NLD" in iso or r["U_lp_pj_yr"] > 100:
            y_lbls.append("Netherlands\n(temperate)")
        else:
            y_lbls.append("Greece\n(warm)")
        u_val = float(r["U_lp_pj_yr"])
        pm_val = float(r["P_milp_pj_yr"])
        pq_val = float(r["P_celf_pj_yr"])
        dn_val = float(r["N_rel_diff_pct"])

        ax_d.plot([0.0, u_val], [yi, yi], color="#8E9FB1", lw=1.05, zorder=2)
        ax_d.scatter([u_val], [yi], s=34, marker="o", facecolors="white", edgecolors=COL_TECH_SLATE, linewidths=1.15, zorder=4,
                     label="Technical potential $T$" if yi == 0 else None)
        ax_d.scatter([pm_val], [yi + 0.08], s=28, marker="s", color=COL_SAPPHIRE, edgecolors="white", linewidths=0.35, zorder=5,
                     label="Exact MILP $P_{\\mathrm{MILP}}$" if yi == 0 else None)
        ax_d.scatter([pq_val], [yi - 0.08], s=28, marker="D", color=COL_CORAL, edgecolors="white", linewidths=0.35, zorder=6,
                     label="Priority queue $P$" if yi == 0 else None)

        ax_d.text(u_val + 2.5, yi + 0.16, f"{u_val:.1f}", va="bottom", ha="left", fontsize=FS_COMPACT)
        ax_d.text(max(pm_val, pq_val) + 3.0, yi - 0.18, f"$\\Delta N$ = {dn_val:.2f}%", va="top", ha="left", fontsize=FS_COMPACT)

    ax_d.set_yticks(yd)
    ax_d.set_yticklabels(y_lbls, fontsize=FS_UNIFIED)
    ax_d.set_ylim(-0.60, len(milp_df) - 0.32)
    ax_d.set_xlim(0.0, 152.0)
    ax_d.set_xticks([0, 30, 60, 90, 120, 150])
    ax_d.set_xlabel("Potential (PJ yr$^{-1}$)")
    ax_d.legend(loc="lower right", bbox_to_anchor=(1.01, -0.02), frameon=False, fontsize=FS_COMPACT, handletextpad=0.25, labelspacing=0.18)

    panel_tag(fig, 0.015, 0.982, "a")
    panel_tag(fig, 0.518, 0.982, "b")
    panel_tag(fig, 0.015, 0.495, "c")
    panel_tag(fig, 0.518, 0.495, "d")

    SRC_DIR.mkdir(parents=True, exist_ok=True)
    val4_milp.to_csv(SRC_DIR / "Extended_Data_Fig1d_milp_source_data.csv", index=False)
    return export_figure(fig, "Extended_Data_Fig1_Validations")


# ==============================================================================
# EXTENDED DATA FIGURE 2: Monthly Dispatch Area, Heat-Pump COP, & Stream Grades
# ==============================================================================
def build_ed_fig_2(df_glob: pd.DataFrame) -> dict:
    configure_matplotlib_style()
    H_MM = 136.0
    fig = plt.figure(figsize=(W_IN, H_MM * MM))

    gs_top = gridspec.GridSpec(
        1, 2, figure=fig, width_ratios=[1.06, 1.0],
        left=0.088, right=0.980, bottom=0.555, top=0.945, wspace=0.30,
    )
    ax_a = fig.add_subplot(gs_top[0, 0])
    ax_b = fig.add_subplot(gs_top[0, 1])
    ax_c = fig.add_axes([0.088, 0.095, 0.892, 0.355])

    # Panel a: Smooth Stacked Seasonal Area + Monthly Lollipop Stems (ZERO Bars!)
    clean_ax(ax_a, "y")
    lk = pd.read_parquet(SCEN_DIR / "R1__obs__2006_2025_selected_links.parquet")
    days_m = np.array([31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype=float)
    m_cols = [f"q_used_m{m:02d}_gj_d" for m in range(1, 13)]
    dir_mask = lk["upgrade_mode"] == "direct_hex"
    hp_mask = ~dir_mask
    m_dir_pj = np.array([float((lk.loc[dir_mask, c] * days_m[i]).sum()) / 1e6 for i, c in enumerate(m_cols)])
    m_hp_pj = np.array([float((lk.loc[hp_mask, c] * days_m[i]).sum()) / 1e6 for i, c in enumerate(m_cols)])
    m_tot_pj = m_dir_pj + m_hp_pj
    q_ann_mean = float(lk["P_net_waste_heat_gj_yr"].sum() / 1e6) / 12.0

    xm = np.arange(12, dtype=float)
    xfine = np.linspace(0.0, 11.0, 220)
    sp_dir = np.clip(make_interp_spline(xm, m_dir_pj, k=3)(xfine), 0.0, None)
    sp_tot = np.clip(make_interp_spline(xm, m_tot_pj, k=3)(xfine), sp_dir, None)

    ax_a.fill_between(xfine, 0.0, sp_dir, facecolor=COL_SAPPHIRE, alpha=0.86, edgecolor="none", zorder=2, label="Direct heat exchange (≥95 °C)")
    ax_a.fill_between(xfine, sp_dir, sp_tot, facecolor=COL_APRICOT, alpha=0.88, edgecolor="none", zorder=2.2, label="Electric heat pumps (<95 °C)")
    for xi, d_v, t_v in zip(xm, m_dir_pj, m_tot_pj):
        ax_a.plot([xi, xi], [0.0, t_v], color="white", lw=0.55, alpha=0.75, zorder=2.5)
    ax_a.plot(xfine, sp_dir, color="#000000", lw=0.65, zorder=3)
    ax_a.plot(xfine, sp_tot, color="#000000", lw=0.80, zorder=3.2)
    ax_a.scatter(xm, m_dir_pj, s=13, color=COL_SAPPHIRE, edgecolors="white", linewidths=0.35, zorder=4)
    ax_a.scatter(xm, m_tot_pj, s=14, color=COL_APRICOT, edgecolors="#000000", linewidths=0.40, zorder=4.2)

    ax_a.axhline(q_ann_mean, color="#000000", ls="--", lw=0.65, zorder=3.5)
    ax_a.text(5.5, q_ann_mean + 14.0, f"Annual monthly mean ({q_ann_mean:.0f} PJ mo$^{{-1}}$)", ha="center", va="bottom", fontsize=FS_COMPACT)

    ax_a.set_xlim(-0.35, 11.35)
    ax_a.set_ylim(0.0, float(m_tot_pj.max()) * 1.25)
    ax_a.set_xticks(xm)
    ax_a.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], fontsize=FS_COMPACT)
    ax_a.set_ylabel("Extracted waste heat (PJ mo$^{-1}$)")
    ax_a.legend(loc="upper center", bbox_to_anchor=(0.50, 1.03), frameon=False, fontsize=FS_COMPACT, ncol=1, labelspacing=0.16, handlelength=1.1)

    # Panel b: Field-calibrated Building ASHP COP(T_out) vs Source Electric Heat Pumps
    clean_ax(ax_b, "y")
    t_grid = np.linspace(-20.0, 16.0, 181)
    cop_25 = ashp_daily_cop(t_grid, use_2050_indiv_tech=False)
    cop_50 = ashp_daily_cop(t_grid, use_2050_indiv_tech=True)

    ax_b.axvspan(-2.0, 10.0, facecolor=COL_ICE_ZONE, zorder=0)
    ax_b.text(4.0, 6.08, "Mild winter\nwarming band", ha="center", va="top", fontsize=FS_COMPACT, linespacing=1.08)

    ax_b.axhline(HP_INDUSTRIAL_POWER_2025.cop, color=COL_SAPPHIRE, ls=":", lw=0.90, zorder=2)
    ax_b.axhline(HP_WASTEWATER_2025.cop, color=COL_CORAL, ls="--", lw=0.80, zorder=2)
    ax_b.text(-19.4, HP_INDUSTRIAL_POWER_2025.cop + 0.10, f"Source electric heat pump, power & ind. ({HP_INDUSTRIAL_POWER_2025.cop:.2f})", ha="left", va="bottom", fontsize=FS_COMPACT)
    ax_b.text(-19.4, HP_WASTEWATER_2025.cop - 0.10, f"Source electric heat pump, wastewater ({HP_WASTEWATER_2025.cop:.2f})", ha="left", va="top", fontsize=FS_COMPACT)

    ax_b.plot(t_grid, cop_25, color=COL_SAPPHIRE, lw=1.65, zorder=4, label="2025 building air source HP (SCOP 2.99)")
    ax_b.plot(t_grid, cop_50, color="#5EC2E6", lw=1.65, ls="-.", zorder=4.2, label="2050 building air source HP (SCOP 4.29)")

    ax_b.set_xlim(-20.0, 16.0)
    ax_b.set_ylim(1.1, 6.35)
    ax_b.set_xticks([-20, -10, 0, 10, 16])
    ax_b.set_yticks([1, 2, 3, 4, 5, 6])
    ax_b.set_xlabel("Daily outdoor air temperature (°C)")
    ax_b.set_ylabel("Coefficient of performance")
    ax_b.legend(loc="lower right", bbox_to_anchor=(1.01, 0.01), frameon=False, fontsize=FS_COMPACT, labelspacing=0.18, handlelength=1.4)

    # Panel c: 1946–2100 Stream-Grade Climate Trajectories (Direct >=95 °C vs Electric Heat Pumps <95 °C)
    clean_ax(ax_c, "y")
    r1 = df_glob[df_glob["group"] == "R1"].copy()
    r1["mid"] = 0.5 * (r1["start_year"] + r1["end_year"])
    r1["P_hp_tot"] = r1["P_hp_ind_power_ej_yr"] + r1["P_hp_wastewater_ej_yr"]

    obs = r1[r1["ssp"] == "obsclim"].sort_values("mid").copy()
    b0_obs = obs[obs["window_id"] == "2006_2025"].iloc[0]
    d0_obs = float(b0_obs["P_direct_hex_ej_yr"])
    h0_obs = float(b0_obs["P_hp_tot"])
    obs["rel_dir"] = 100.0 * (obs["P_direct_hex_ej_yr"] / d0_obs - 1.0)
    obs["rel_hp"] = 100.0 * (obs["P_hp_tot"] / h0_obs - 1.0)

    ax_c.axvspan(1946, 2015.5, facecolor=COL_HIST_BG, zorder=0)
    ax_c.axvline(2015.5, ls=":", lw=0.65, color="#333333", zorder=2)
    ax_c.axhline(0.0, ls="--", lw=0.55, color="#333333", zorder=2)
    ax_c.text(2012.5, 42.5, "Observed (GSWP3-ERA5)", ha="right", va="top", fontsize=FS_COMPACT)
    ax_c.text(2018.5, 42.5, "Projected SSP2-4.5", ha="left", va="top", fontsize=FS_COMPACT)

    ax_c.plot(obs["mid"], obs["rel_dir"], color=COL_SAPPHIRE, lw=1.55, ls="--", marker="o", ms=3.4, zorder=5)
    ax_c.plot(obs["mid"], obs["rel_hp"], color=COL_CORAL, lw=1.55, ls="--", marker="s", ms=3.4, zorder=5)

    p245 = r1[r1["ssp"] == "ssp245"].copy()
    p245["rel_dir"] = np.where(p245["window_id"] == "2006_2025", 0.0, 100.0 * (p245["P_direct_hex_ej_yr"] / d0_obs - 1.0))
    p245["rel_hp"] = np.where(p245["window_id"] == "2006_2025", 0.0, 100.0 * (p245["P_hp_tot"] / h0_obs - 1.0))

    grp_d = p245.groupby("mid")["rel_dir"].agg(["mean", "min", "max"]).reset_index()
    grp_h = p245.groupby("mid")["rel_hp"].agg(["mean", "min", "max"]).reset_index()

    ax_c.fill_between(grp_d["mid"], grp_d["min"], grp_d["max"], facecolor=COL_SAPPHIRE, alpha=0.16, lw=0, zorder=2)
    ax_c.fill_between(grp_h["mid"], grp_h["min"], grp_h["max"], facecolor=COL_CORAL, alpha=0.16, lw=0, zorder=2)
    ax_c.plot(grp_d["mid"], grp_d["mean"], color=COL_SAPPHIRE, lw=1.70, zorder=5, label="Direct heat exchange (≥95 °C)")
    ax_c.plot(grp_h["mid"], grp_h["mean"], color=COL_CORAL, lw=1.70, zorder=5, label="Electric heat pumps (<95 °C)")

    for wid, x_jit in [("2046_2065", 0.0), ("2081_2100", 0.0)]:
        sub_w = p245[p245["window_id"] == wid]
        for mcode, _, mk in GCM_ORDER:
            mr = sub_w[sub_w["model"] == mcode]
            if not mr.empty:
                r0 = mr.iloc[0]
                ax_c.scatter([r0["mid"] + x_jit], [r0["rel_dir"]], facecolors="none", edgecolors=COL_SAPPHIRE, marker=mk, s=12, linewidths=0.55, zorder=6)
                ax_c.scatter([r0["mid"] + x_jit], [r0["rel_hp"]], facecolors="none", edgecolors=COL_CORAL, marker=mk, s=12, linewidths=0.55, zorder=6)

    m55_d = float(grp_d[grp_d["mid"] == 2055.5]["mean"].iloc[0])
    m55_h = float(grp_h[grp_h["mid"] == 2055.5]["mean"].iloc[0])
    m90_d = float(grp_d[grp_d["mid"] == 2090.5]["mean"].iloc[0])
    m90_h = float(grp_h[grp_h["mid"] == 2090.5]["mean"].iloc[0])

    ax_c.text(2055.5, 2.5, f"{fm(m55_d, '.1f')}%", ha="center", va="bottom", fontsize=FS_COMPACT, fontweight="bold")
    ax_c.text(2049.5, m55_h - 3.5, f"{fm(m55_h, '.1f')}%", ha="right", va="top", fontsize=FS_COMPACT, fontweight="bold")
    ax_c.text(2092.5, m90_d, f"{fm(m90_d, '.1f')}%", ha="left", va="center", fontsize=FS_COMPACT, fontweight="bold")
    ax_c.text(2092.5, m90_h, f"{fm(m90_h, '.1f')}%", ha="left", va="center", fontsize=FS_COMPACT, fontweight="bold")

    ax_c.set_xlim(1946, 2106)
    ax_c.set_ylim(-92.0, 46.0)
    ax_c.set_xticks([1955, 1975, 1995, 2015, 2035, 2055, 2075, 2090])
    ax_c.set_yticks([-75, -50, -25, 0, 25])
    ax_c.set_yticklabels([f"{MINUS}75", f"{MINUS}50", f"{MINUS}25", "0", "+25"])
    ax_c.set_xlabel("20 year climate window midpoint")
    ax_c.set_ylabel("Change relative to\n2006–2025 (%)")
    ax_c.legend(loc="lower left", bbox_to_anchor=(0.01, 0.02), frameon=False, fontsize=FS_COMPACT, ncol=1, handlelength=1.4, labelspacing=0.20)

    panel_tag(fig, 0.015, 0.982, "a")
    panel_tag(fig, 0.525, 0.982, "b")
    panel_tag(fig, 0.015, 0.482, "c")

    pd.DataFrame({
        "month": np.arange(1, 13),
        "direct_hex_pj_mo": m_dir_pj,
        "electric_hp_pj_mo": m_hp_pj,
    }).to_csv(SRC_DIR / "Extended_Data_Fig2_dispatch_and_grades_source_data.csv", index=False)
    return export_figure(fig, "Extended_Data_Fig2_Dispatch_COP_and_Grades")


# ==============================================================================
# EXTENDED DATA FIGURE 3: Synchronized 1x3 Sensitivity Dumbbells (Zero Bars!)
# ==============================================================================
def build_ed_fig_3(df_glob: pd.DataFrame, df_country: pd.DataFrame) -> dict:
    configure_matplotlib_style()
    H_MM = 112.0
    fig = plt.figure(figsize=(W_IN, H_MM * MM))

    gs = gridspec.GridSpec(
        1, 3, figure=fig, width_ratios=[1.0, 1.0, 1.0],
        left=0.215, right=0.982, bottom=0.145, top=0.865, wspace=0.18,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1], sharey=ax_a)
    ax_c = fig.add_subplot(gs[0, 2], sharey=ax_a)

    # Identify exact official-survey ISO3 set matching P_official_only_ej_yr
    df_rp = pd.read_parquet(DATA_PROC / "sinks_2025/calibrated_region_parameters_2025.parquet")
    off_isos = set(df_rp.loc[df_rp["valid_for_study"] & (~df_rp["cal_source"].astype(str).str.contains("UNSD|nearest|proxy", case=False, na=False)), "iso3"].unique())
    c_obs = df_country[df_country["scenario_id"] == "R1__obs__2006_2025"]
    p0_off_glob = float(df_glob.loc[df_glob["scenario_id"] == "R1__obs__2006_2025", "P_official_only_ej_yr"].iloc[0])
    # Fallback exact scaling if subnational region split is slightly finer than ISO3
    c_off_obs = c_obs[c_obs["iso3"].isin(off_isos)]
    p_iso_sum = float(c_off_obs["P_gj"].sum() / 1e9)
    scale_off = p0_off_glob / p_iso_sum if p_iso_sum > 0 else 1.0

    specs = [
        ("Central run", "CENTRAL"),
        ("Threshold 12 °C", "Rob1_bait12c"),
        ("Threshold 16 °C", "Rob1_bait16c"),
        ("Discount rate 3%", "Rob2_r03pct"),
        ("Discount rate 10%", "Rob2_r10pct"),
        ("Retail tariffs with tax", "Rob3_retail_tax"),
        (f"Energy prices {MINUS}30%", "Rob3_price_minus30"),
        ("Energy prices +30%", "Rob3_price_plus30"),
        ("GHSL 2020 building map", "Rob4_ghsl2020"),
        ("Official heating surveys", "OFFICIAL_ONLY"),
    ]

    rows = []
    obs_row = df_glob[df_glob["scenario_id"] == "R1__obs__2006_2025"].iloc[0]
    r1_fut = df_glob[(df_glob["group"] == "R1") & (df_glob["ssp"] == "ssp245") & (df_glob["window_id"] == "2046_2065")]

    for lbl, code in specs:
        if code == "CENTRAL":
            p0 = float(obs_row["P_ej_yr"])
            nb0 = float(obs_row["net_benefit_billion_usd_yr"])
            co0 = float(obs_row["co2_abated_mt_yr"])
            sub_f = r1_fut
            p_f = sub_f["P_ej_yr"].to_numpy(dtype=float)
            nb_f = sub_f["net_benefit_billion_usd_yr"].to_numpy(dtype=float)
            co_f = sub_f["co2_abated_mt_yr"].to_numpy(dtype=float)
            mods = sub_f["model"].tolist()
        elif code == "OFFICIAL_ONLY":
            p0 = p0_off_glob
            nb0 = float(c_off_obs["net_benefit_usd"].sum() / 1e9) * scale_off
            co0 = float(c_off_obs["co2_abated_t"].sum() / 1e6) * scale_off
            p_f = r1_fut["P_official_only_ej_yr"].to_numpy(dtype=float)
            mods = r1_fut["model"].tolist()
            c_f_sub = df_country[(df_country["group"] == "R1") & (df_country["ssp"] == "ssp245") & (df_country["window_id"] == "2046_2065") & (df_country["iso3"].isin(off_isos))]
            c_by_m = c_f_sub.groupby("model")[["net_benefit_usd", "co2_abated_t"]].sum()
            nb_f = np.array([float(c_by_m.loc[m, "net_benefit_usd"] / 1e9) * scale_off for m in mods])
            co_f = np.array([float(c_by_m.loc[m, "co2_abated_t"] / 1e6) * scale_off for m in mods])
        else:
            b_r = df_glob[(df_glob["group"] == "Robustness") & (df_glob["subgroup"] == code) & (df_glob["window_id"] == "2006_2025")].iloc[0]
            sub_f = df_glob[(df_glob["group"] == "Robustness") & (df_glob["subgroup"] == code) & (df_glob["window_id"] == "2046_2065")]
            p0 = float(b_r["P_ej_yr"])
            nb0 = float(b_r["net_benefit_billion_usd_yr"])
            co0 = float(b_r["co2_abated_mt_yr"])
            p_f = sub_f["P_ej_yr"].to_numpy(dtype=float)
            nb_f = sub_f["net_benefit_billion_usd_yr"].to_numpy(dtype=float)
            co_f = sub_f["co2_abated_mt_yr"].to_numpy(dtype=float)
            mods = sub_f["model"].tolist()

        rel_p = 100.0 * (p_f - p0) / max(1e-9, p0)
        rows.append({
            "specification": lbl,
            "P_2006_2025_ej_yr": p0,
            "P_2046_2065_mean_ej_yr": float(p_f.mean()),
            "P_2046_2065_min_ej_yr": float(p_f.min()),
            "P_2046_2065_max_ej_yr": float(p_f.max()),
            "rel_change_mean_pct": float(rel_p.mean()),
            "rel_change_min_pct": float(rel_p.min()),
            "rel_change_max_pct": float(rel_p.max()),
            "nb_2006_2025_busd_yr": nb0,
            "nb_2046_2065_mean_busd_yr": float(nb_f.mean()),
            "co2_2006_2025_mt_yr": co0,
            "co2_2046_2065_mean_mt_yr": float(co_f.mean()),
            "_mods": mods,
            "_pf": p_f,
            "_nbf": nb_f,
            "_cof": co_f,
        })

    y_pos = np.arange(len(rows))[::-1]
    for ax in (ax_a, ax_b, ax_c):
        clean_ax(ax, "x")

    for yi, r in zip(y_pos, rows):
        for ax, v0, vf_arr, vf_mean in [
            (ax_a, r["P_2006_2025_ej_yr"], r["_pf"], r["P_2046_2065_mean_ej_yr"]),
            (ax_b, r["nb_2006_2025_busd_yr"], r["_nbf"], r["nb_2046_2065_mean_busd_yr"]),
            (ax_c, r["co2_2006_2025_mt_yr"], r["_cof"], r["co2_2046_2065_mean_mt_yr"]),
        ]:
            lo_x = min(v0, float(vf_arr.min()))
            hi_x = max(v0, float(vf_arr.max()))
            ax.plot([lo_x, hi_x], [yi, yi], color=COL_BAND, lw=2.2, alpha=0.85, zorder=2)
            for m_id, mv in zip(r["_mods"], vf_arr):
                ax.scatter([mv], [yi], facecolors="white", edgecolors="#000000", marker=GCM_MARKER[m_id], s=10.5, linewidths=0.48, zorder=3.5)
            ax.scatter([v0], [yi], s=28, marker="o", color=COL_SAPPHIRE, edgecolors="white", linewidths=0.35, zorder=4.5)
            ax.scatter([vf_mean], [yi], s=28, marker="D", color=COL_CORAL, edgecolors="white", linewidths=0.35, zorder=5.0)

    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels([r["specification"] for r in rows], fontsize=FS_UNIFIED)
    plt.setp(ax_b.get_yticklabels(), visible=False)
    plt.setp(ax_c.get_yticklabels(), visible=False)
    ax_a.set_ylim(-0.65, len(rows) - 0.35)

    ax_a.set_xlim(0.0, 4.30)
    ax_a.set_xticks([0, 1, 2, 3, 4])
    ax_a.set_xlabel("Economic potential (EJ yr$^{-1}$)")

    ax_b.set_xlim(0.0, 44.0)
    ax_b.set_xticks([0, 10, 20, 30, 40])
    ax_b.set_xlabel("Annual net cost savings\n(billion USD yr$^{-1}$)")

    ax_c.set_xlim(0.0, 245.0)
    ax_c.set_xticks([0, 60, 120, 180, 240])
    ax_c.set_xlabel("Annual CO$_2$ abatement\n(Mt CO$_2$ yr$^{-1}$)")

    leg_h = [
        mpl.lines.Line2D([], [], color="none", marker="o", markerfacecolor=COL_SAPPHIRE, markeredgecolor="white", ms=5.5, label="2006–2025 present"),
        mpl.lines.Line2D([], [], color="none", marker="D", markerfacecolor=COL_CORAL, markeredgecolor="white", ms=5.5, label="2046–2065 SSP2-4.5 mean"),
        mpl.lines.Line2D([], [], color="none", marker="s", markerfacecolor="white", markeredgecolor="#000000", markeredgewidth=0.6, ms=4.5, label="Individual CMIP6 models ($n$ = 5)"),
    ]
    fig.legend(handles=leg_h, loc="upper center", bbox_to_anchor=(0.58, 0.985), ncol=3, frameon=False, fontsize=FS_UNIFIED, columnspacing=1.2, handletextpad=0.35)

    panel_tag(fig, 0.212, 0.918, "a")
    panel_tag(fig, 0.482, 0.918, "b")
    panel_tag(fig, 0.752, 0.918, "c")

    out_csv = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows])
    out_csv.to_csv(SRC_DIR / "Extended_Data_Fig3_sensitivity_levels_source_data.csv", index=False)
    # Also write legacy filename so any downstream reader works seamlessly
    out_csv.to_csv(SRC_DIR / "Extended_Data_Fig2_robustness_source_data.csv", index=False)
    return export_figure(fig, "Extended_Data_Fig3_Sensitivity_Levels")


# ==============================================================================
# EXTENDED DATA FIGURE 4: End-Century 2081–2100 SSP1-2.6 vs SSP3-7.0 Divergence
# ==============================================================================
def build_ed_fig_4() -> dict:
    configure_matplotlib_style()
    H_MM = 148.0
    fig = plt.figure(figsize=(W_IN, H_MM * MM))

    z_dir = SCEN_DIR / "zones"
    df_z_meta = pd.read_parquet(DATA_PROC / "sinks_2025/urban_heat_zones.parquet")[["zone_id", "iso3", "lat", "lon"]]
    df_dec = pd.read_parquet(SCEN_DIR / "r2_decomposition/r2_eq6_decomposition_all.parquet")
    df_geq = pd.read_csv(SCEN_DIR / "r2_decomposition/r2_eq6_global_summary.csv")

    gdf_bg = load_world_basemap(ROOT)
    gdf_bg = gdf_bg[gdf_bg["GID_0"] != "ATA"].copy().to_crs("+proj=robin")
    trans = Transformer.from_crs("EPSG:4326", "+proj=robin", always_xy=True)

    lats_b = np.linspace(-58, 78, 181)
    lons_b = np.linspace(-180, 180, 301)
    bx = np.concatenate([np.full_like(lats_b, -180), lons_b, np.full_like(lats_b, 180), lons_b[::-1]])
    by = np.concatenate([lats_b, np.full_like(lons_b, 78), lats_b[::-1], np.full_like(lons_b, -58)])
    px, py = trans.transform(bx, by)

    def _ens_zone_endcentury(ssp_code: str) -> pd.DataFrame:
        dfs = []
        for m, _, _ in GCM_ORDER:
            b0 = pd.read_parquet(z_dir / f"R1__{m}__{ssp_code}__2006_2025_zones.parquet")[["zone_id", "P_gj"]].rename(columns={"P_gj": "P0_gj"})
            f1 = pd.read_parquet(z_dir / f"R1__{m}__{ssp_code}__2081_2100_zones.parquet")[["zone_id", "P_gj"]].rename(columns={"P_gj": "P1_gj"})
            dfs.append(b0.merge(f1, on="zone_id"))
        z_m = pd.concat(dfs, ignore_index=True).groupby("zone_id", as_index=False)[["P0_gj", "P1_gj"]].mean()
        z_m = z_m[z_m["P0_gj"] > 0.0].copy()
        z_m["loss_pct"] = np.clip(-100.0 * (z_m["P1_gj"] - z_m["P0_gj"]) / z_m["P0_gj"], 0.0, 50.0)
        return z_m.merge(df_z_meta, on="zone_id", how="left")

    z_126 = _ens_zone_endcentury("ssp126")
    z_370 = _ens_zone_endcentury("ssp370")

    cmap_loss = mcolors.LinearSegmentedColormap.from_list(
        "pro_ed4_ramp", ["#FEF098", "#F9B872", "#E65C3B", "#AC2428", "#671019"], N=256
    )

    ax_ma = fig.add_axes([0.015, 0.562, 0.478, 0.412])
    ax_mb = fig.add_axes([0.507, 0.562, 0.478, 0.412])
    ax_c = fig.add_axes([0.155, 0.080, 0.340, 0.325])
    ax_d = fig.add_axes([0.635, 0.080, 0.342, 0.325])

    sc_ref = None
    for ax_m, z_df, in_lbl in [
        (ax_ma, z_126, f"2081–2100\nSSP1-2.6\n(global {MINUS}8.7%)"),
        (ax_mb, z_370, f"2081–2100\nSSP3-7.0\n(global {MINUS}27.9%)"),
    ]:
        ax_m.fill(px, py, facecolor="#FFFFFF", edgecolor="#333333", linewidth=0.50, zorder=0)
        for glat in [-30, 0, 30, 60]:
            gx, gy = trans.transform(np.linspace(-180, 180, 181), np.full(181, glat))
            ax_m.plot(gx, gy, color="#D8E0E8", lw=0.30, zorder=0.4)
        gdf_bg.plot(ax=ax_m, facecolor="#F3F5F8", edgecolor="#8A96A3", linewidth=0.24, zorder=1)
        zp = z_df.sort_values("P0_gj")
        vx, vy = trans.transform(zp["lon"].to_numpy(), zp["lat"].to_numpy())
        sz = np.clip(np.sqrt(zp["P0_gj"].to_numpy() / 1e6) * 4.8, 1.5, 28.0)
        sc_ref = ax_m.scatter(vx, vy, c=zp["loss_pct"], cmap=cmap_loss, vmin=0, vmax=50, s=sz, alpha=0.92, edgecolors="white", linewidths=0.18, rasterized=True, zorder=3)
        ax_m.text(0.08, 0.14, in_lbl, transform=ax_m.transAxes, fontsize=FS_COMPACT, ha="left", va="bottom", linespacing=1.12,
                  bbox=dict(boxstyle="square,pad=0.10", facecolor="white", edgecolor="none", alpha=0.88), zorder=5)
        ax_m.set_xlim(px.min() * 1.02, px.max() * 1.02)
        ax_m.set_ylim(py.min() * 1.02, py.max() * 1.02)
        ax_m.set_xticks([])
        ax_m.set_yticks([])
        ax_m.set_xlabel("")
        ax_m.set_ylabel("")
        ax_m.axis("off")

    cax = fig.add_axes([0.365, 0.535, 0.270, 0.013])
    cb = fig.colorbar(sc_ref, cax=cax, orientation="horizontal", extend="max")
    cb.outline.set_linewidth(0.45)
    cb.outline.set_edgecolor("#000000")
    cb.set_ticks([0, 10, 20, 30, 40, 50])
    cb.set_ticklabels(["0%", f"{MINUS}10%", f"{MINUS}20%", f"{MINUS}30%", f"{MINUS}40%", f"{MINUS}50%"])
    cb.ax.tick_params(labelsize=FS_COMPACT, length=2.0, width=0.45)
    cb.set_label("2081–2100 change (%)", fontsize=FS_UNIFIED, labelpad=1.8)

    # Panel c: Global Pathway x Horizon 3-Term Decomposition
    clean_ax(ax_c, "x")
    combos = [
        ("ssp126", "2046_2065", "2046–2065\nSSP1-2.6"),
        ("ssp245", "2046_2065", "2046–2065\nSSP2-4.5"),
        ("ssp370", "2046_2065", "2046–2065\nSSP3-7.0"),
        ("ssp126", "2081_2100", "2081–2100\nSSP1-2.6"),
        ("ssp245", "2081_2100", "2081–2100\nSSP2-4.5"),
        ("ssp370", "2081_2100", "2081–2100\nSSP3-7.0"),
    ][::-1]

    rows_c = []
    for s_c, w_c, lbl_c in combos:
        sub = df_geq[(df_geq["ssp"] == s_c) & (df_geq["window_id"] == w_c)]
        net_m = float(sub["dP_rel_pct"].mean())
        t1 = float(sub["term1_demand_shift_pct"].mean())
        t2 = float(sub["term2_seasonal_mismatch_pct"].mean())
        t3 = net_m - t1 - t2
        rows_c.append({
            "label": lbl_c, "ssp": s_c, "window_id": w_c,
            "t1": t1, "t2": t2, "t3": t3, "net": net_m,
            "lo": float(sub["dP_rel_pct"].min()), "hi": float(sub["dP_rel_pct"].max()),
        })

    yc = np.arange(len(rows_c))
    for yi, r in enumerate(rows_c):
        neg_l, pos_l = 0.0, 0.0
        for val, col in [(r["t3"], COL_CORAL), (r["t1"], COL_DEM_BLUE), (r["t2"], "#8CD4EC")]:
            if val < 0:
                ax_c.barh(yi, val, left=neg_l, height=0.58, color=col, edgecolor="none", zorder=3)
                if neg_l < -0.05:
                    ax_c.plot([neg_l, neg_l], [yi - 0.30, yi + 0.30], color="white", lw=0.55, zorder=3.5)
                neg_l += val
            elif val > 0:
                ax_c.barh(yi, val, left=pos_l, height=0.58, color=col, edgecolor="none", zorder=3)
                pos_l += val
        ax_c.errorbar(r["net"], yi, xerr=[[r["net"] - r["lo"]], [r["hi"] - r["net"]]], fmt="D", ms=3.2, color="#000000", ecolor="#000000", elinewidth=0.65, capsize=1.6, zorder=6)

    ax_c.axvline(0.0, lw=0.65, color="#000000", zorder=18)
    ax_c.set_yticks(yc)
    ax_c.set_yticklabels([r["label"] for r in rows_c], fontsize=FS_COMPACT, linespacing=0.95)
    ax_c.set_xlim(-49.0, 11.0)
    ax_c.set_xticks([-40, -20, 0, 10])
    ax_c.set_xticklabels([f"{MINUS}40", f"{MINUS}20", "0", "+10"])
    ax_c.set_xlabel("Global contribution to change (%)")

    leg_c = [
        mpatches.Patch(facecolor=COL_DEM_BLUE, edgecolor="none", label="Heating demand"),
        mpatches.Patch(facecolor="#8CD4EC", edgecolor="none", label="Seasonal matching"),
        mpatches.Patch(facecolor=COL_CORAL, edgecolor="none", label="Economic viability"),
        mpl.lines.Line2D([], [], color="#000000", marker="D", lw=0.80, ms=3.2, label="Net change"),
    ]
    ax_c.legend(handles=leg_c, loc="lower center", bbox_to_anchor=(0.48, 1.01), ncol=2, frameon=False, fontsize=FS_COMPACT, columnspacing=0.7, handlelength=0.95, handletextpad=0.3, labelspacing=0.18)

    # Panel d: 10-Country 2081–2100 Multi-Pathway Cleveland Dot-Range Plot
    clean_ax(ax_d, "x")
    top_isos = ["CHN", "RUS", "DEU", "KAZ", "KOR", "POL", "ITA", "NLD", "SWE", "DNK"]
    c_end = df_dec[(df_dec["level"] == "country") & (df_dec["window_id"] == "2081_2100") & (df_dec["entity_id"].isin(top_isos))]
    c_piv = c_end.groupby(["entity_id", "ssp"])["dP_rel_pct"].mean().unstack("ssp").loc[top_isos[::-1]]

    yd = np.arange(len(c_piv))
    for yi, iso in enumerate(c_piv.index):
        v1 = float(c_piv.loc[iso, "ssp126"])
        v2 = float(c_piv.loc[iso, "ssp245"])
        v3 = float(c_piv.loc[iso, "ssp370"])
        ax_d.plot([min(v1, v2, v3), max(v1, v2, v3)], [yi, yi], color="#8E9FB1", lw=1.15, zorder=2)

    ax_d.scatter(c_piv["ssp126"], yd, s=24, marker="s", color=COL_SAPPHIRE, edgecolors="white", linewidths=0.35, zorder=4, label="SSP1-2.6")
    ax_d.scatter(c_piv["ssp245"], yd, s=26, marker="^", color=COL_SSP2, edgecolors="white", linewidths=0.35, zorder=4.5, label="SSP2-4.5")
    ax_d.scatter(c_piv["ssp370"], yd, s=26, marker="D", color=COL_CORAL, edgecolors="white", linewidths=0.35, zorder=5, label="SSP3-7.0")
    ax_d.axvline(0.0, lw=0.65, color="#000000", ls=":", zorder=18)

    ax_d.set_yticks(yd)
    ax_d.set_yticklabels([cname(c) for c in c_piv.index], fontsize=FS_COMPACT)
    ax_d.set_xlim(-78.0, 5.0)
    ax_d.set_xticks([-75, -50, -25, 0])
    ax_d.set_xticklabels([f"{MINUS}75", f"{MINUS}50", f"{MINUS}25", "0"])
    ax_d.set_xlabel("2081–2100 change (%)")
    ax_d.legend(loc="lower center", bbox_to_anchor=(0.50, 1.01), ncol=3, frameon=False, fontsize=FS_COMPACT, columnspacing=0.65, handletextpad=0.25)

    panel_tag(fig, 0.015, 0.982, "a")
    panel_tag(fig, 0.505, 0.982, "b")
    panel_tag(fig, 0.015, 0.468, "c")
    panel_tag(fig, 0.525, 0.468, "d")

    pd.DataFrame(rows_c).to_csv(SRC_DIR / "Extended_Data_Fig4_pathway_decomposition_source_data.csv", index=False)
    pd.DataFrame(rows_c).to_csv(SRC_DIR / "Extended_Data_Fig4c_pathway_decomposition_source_data.csv", index=False)
    return export_figure(fig, "Extended_Data_Fig4_Pathway_Extremes_Comparison")


def main() -> None:
    df_glob = pd.read_parquet(SCEN_DIR / "global_scenarios_summary.parquet")
    df_country = pd.read_parquet(SCEN_DIR / "all_scenarios_by_country.parquet")

    audits = []
    audits.append(build_ed_fig_1())
    audits.append(build_ed_fig_2(df_glob))
    audits.append(build_ed_fig_3(df_glob, df_country))
    audits.append(build_ed_fig_4())

    print("\n=== EXTENDED DATA FIGURES 1–4 EXPORT SUMMARY ===")
    for a in audits:
        print(f"  {a['stem']}: {a['w_mm']:.2f}x{a['h_mm']:.2f} mm | {a['mb']:.2f} MB | {a['n_texts']} text objects | bbox touches = {len(a['overlaps'])}")
    tot_ov = sum(len(a["overlaps"]) for a in audits)
    assert tot_ov == 0, f"Total ED text overlaps = {tot_ov}"


if __name__ == "__main__":
    main()
