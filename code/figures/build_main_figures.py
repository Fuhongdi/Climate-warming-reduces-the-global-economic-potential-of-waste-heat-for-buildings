#!/usr/bin/env python3
"""Generate Main Figures 2–5 and export underlying panel source data tables."""
from __future__ import annotations
import math
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
from matplotlib.patches import ConnectionPatch, PathPatch
from matplotlib.path import Path as MplPath
import matplotlib.pyplot as plt
from matplotlib.text import Text
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy import stats

warnings.filterwarnings("ignore")

ROOT = Path(os.environ.get("WASTE_HEAT_ROOT", Path(__file__).resolve().parents[2]))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "code/figures"))
from figure_utils import MM, DOUBLE_COL, check_typography, check_palette_contrast, load_world_basemap  # noqa: E402

DATA_PROC = ROOT / "data/processed"
SCEN_DIR = DATA_PROC / "scenarios_2025"
FIG_DIR = ROOT / "figures/main"
SRC_DIR = ROOT / "figures/source_data"

W_MM = 180.0
W_IN = DOUBLE_COL  # 180.0 * MM = 7.086614 in
MINUS = "\u2212"

# ==============================================================================
# Unified Typography & Pure Black Font Across ALL Figures 2–5 
# ==============================================================================
FS_PANEL = 9.5     # Panel letters a, b, c, d (bold lowercase)
FS_BODY = 6.8      # Primary callout numbers
FS_UNIFIED = 6.5   # Axis labels == Tick labels == Legend text
FS_COMPACT = 5.6   # Compact insets & 5x3 small-multiple ticks (>= 5.0 pt floor)

INK = "#000000"    # 100% pure black text everywhere!
SPINE_COL = "#000000"

SEMANTIC_PALETTE = {
    "Power plants / SSP1-2.6": "#264B82",
    "Urban / Savings": "#70CBE6",
    "Wastewater / Seasonal": "#2B9EB3",
    "Heavy industry / SSP2-4.5": "#E8A638",
    "Viability loss / SSP3-7.0": "#C8422E",
    "Building / Sensitivity": "#4A525D",
}

COL_SAPPHIRE = "#264B82"
COL_BLDG_SLATE = "#4A525D"
COL_SKY_CYAN = "#70CBE6"
COL_DEM_BLUE = "#427FB8"
COL_SEASONAL = "#8CD4EC"
COL_TECH_SLATE = "#4A7C9B"
COL_EMERALD = "#2B9EB3"
COL_APRICOT = "#E8A638"
COL_SSP2 = "#E6A532"
COL_CORAL = "#C8422E"
COL_CORAL_SOFT = "#D96A55"
COL_MIST_DH = "#DAE2EC"
COL_RIBBON_FADE = "#E2E8F0"
COL_HIST_BG = "#F2F6FA"
COL_ICE_ZONE = "#EAF2F8"
COL_BLUSH_ZONE = "#FBE9E5"
COL_ZERO_DOT = "#8E9FB1"

# Refined Harmonious Nature Energy 5-Sector Palette for Figure 5
F5_POWER = "#355C96"     # Royal cobalt
F5_STEEL = "#C84632"     # Terracotta vermilion
F5_REFINING = "#EFC06E"  # Warm ochre sand
F5_CEMENT = "#88C6D8"    # Glacier mist cyan
F5_CHEM = "#19384A"      # Deep petrol slate
F5_BAND = "#9DB4D0"      # Soft cerulean-slate range band

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
    "ISR": "Israel", "ROU": "Romania", "ESP": "Spain", "MNG": "Mongolia",
    "IRN": "Iran",
}


def cname(iso: str) -> str:
    return ISO_NAME.get(str(iso), str(iso))


def fm(val: float, fmt: str = ".1f") -> str:
    ndig = int(fmt.lstrip(".").rstrip("f"))
    q = Decimal("10") ** (-ndig) if ndig > 0 else Decimal("1")
    r = Decimal(f"{abs(val):.6f}").quantize(q, rounding=ROUND_HALF_UP)
    return f"{MINUS}{r}" if val < 0 else f"{r}"


def configure_matplotlib_style() -> None:
    """Configure unified Matplotlib typography and line weights with 100% pure black text."""
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
    """Ensure axis spines always render strictly ABOVE any bar or patch (`zorder = 20`)."""
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


def sankey_ribbon(
    ax: plt.Axes,
    x0: float, y0_bot: float, y0_top: float,
    x1: float, y1_bot: float, y1_top: float,
    color: str, alpha: float = 0.88, ec: str = "white", lw: float = 0.45, zorder: float = 2.0
) -> None:
    """Draw a smooth cubic-Bezier horizontal Sankey ribbon with hairline white edges."""
    xm = 0.5 * (x0 + x1)
    verts = [
        (x0, y0_bot),
        (xm, y0_bot), (xm, y1_bot), (x1, y1_bot),
        (x1, y1_top),
        (xm, y1_top), (xm, y0_top), (x0, y0_top),
        (x0, y0_bot),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]
    patch = PathPatch(MplPath(verts, codes), facecolor=color, alpha=alpha, edgecolor=ec, linewidth=lw, zorder=zorder)
    ax.add_patch(patch)


def export_figure(fig: plt.Figure, stem: str) -> dict:
    """Run objective publication checks (spine z-order, 100% black text, check_typography, text collisions, PDF spec)."""
    # 0. Enforce & verify spine z-order >= 18 and 100% pure black text (#000000) everywhere!
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

    # 1. Check zero ax.set_title() on all axes
    titled_axes = [ax.get_title() for ax in fig.axes if ax.get_title().strip()]
    assert len(titled_axes) == 0, f"Unexpected axis title found on {stem}: {titled_axes}"

    # 2. Run Typography check at 180.0 mm
    print(f"\n--- Typography check for {stem} ---")
    rows, n_fail = check_typography(fig, W_MM, verbose=True)
    n_over = sum(1 for r in rows if r[4] == "over")
    assert n_fail == 0 and n_over == 0, f"Typography spec violation in {stem}: {n_fail} fail, {n_over} over"

    # 3. Force canvas draw & check true oriented polygon bounding boxes of all visible Text objects
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
        print(f"  [WARN] {len(overlaps)} text bbox touch(es) in {stem}: {overlaps[:8]}")
    else:
        print(f"  [PASS] Zero text-to-text bounding-box overlaps across {len(vis_texts)} pure-black (#000000) text objects!")

    for ext in ("pdf", "svg", "png"):
        out_sub = FIG_DIR / ext
        out_sub.mkdir(parents=True, exist_ok=True)
        dpi = 450 if ext == "png" else 300
        fig.savefig(out_sub / f"{stem}.{ext}", dpi=dpi)

    w_mm_actual = fig.get_size_inches()[0] / MM
    h_mm_actual = fig.get_size_inches()[1] / MM
    pdf_path = FIG_DIR / "pdf" / f"{stem}.pdf"
    mb_size = pdf_path.stat().st_size / (1024 * 1024)
    assert abs(w_mm_actual - 180.0) < 0.05, f"Width {w_mm_actual} != 180.0 mm"
    assert h_mm_actual <= 185.0, f"Height {h_mm_actual} > 185.0 mm ceiling"
    assert mb_size < 50.0, f"PDF size {mb_size:.2f} MB >= 50 MB cap"

    pdf_bytes = pdf_path.read_bytes()
    has_type3 = b"/Subtype /Type3" in pdf_bytes
    assert not has_type3, f"CRITICAL: Uneditable Type 3 font detected in {stem}.pdf!"
    print(f"  [PASS] Physical size = {w_mm_actual:.2f} x {h_mm_actual:.2f} mm | PDF = {mb_size:.2f} MB | Black Font + Spine zorder=20 + TrueType = PASS")

    plt.close(fig)
    return {"stem": stem, "w_mm": w_mm_actual, "h_mm": h_mm_actual, "mb": mb_size, "overlaps": overlaps, "n_texts": len(vis_texts)}


# ==============================================================================
# FIGURE 2 (Results 1 — Clean Spacious Sankey Cascade, ZERO top stage titles!)
# ==============================================================================
def build_figure_2(df_glob: pd.DataFrame, df_streams: pd.DataFrame) -> dict:
    configure_matplotlib_style()
    H_MM = 148.0
    fig = plt.figure(figsize=(W_IN, H_MM * MM))

    ax_a = fig.add_axes([0.018, 0.585, 0.964, 0.388])
    ax_b = fig.add_axes([0.078, 0.088, 0.482, 0.448])
    ax_c = fig.add_axes([0.596, 0.088, 0.334, 0.448], sharey=ax_b)

    obs0 = df_glob.loc[df_glob["scenario_id"] == "R1__obs__2006_2025"].iloc[0]
    Q_pow = float(df_streams.loc[df_streams["sector"] == "power", "q3g_annual_gj"].sum() / 1e9)
    Q_ind = float(df_streams.loc[~df_streams["sector"].isin(["power", "wastewater"]), "q3g_annual_gj"].sum() / 1e9)
    Q_ww = float(df_streams.loc[df_streams["sector"] == "wastewater", "q3g_annual_gj"].sum() / 1e9)
    Q_tot = Q_pow + Q_ind + Q_ww
    D_tot = float(obs0["D_total_ej_yr"])
    D_urb = float(obs0["D_urban_ej_yr"])
    U0 = float(obs0["U_ej_yr"])
    P0 = float(obs0["P_ej_yr"])
    P_pow = float(obs0["P_power_ej_yr"])
    DH_CUR = 7.57

    # --- Panel a: Spacious Horizontal Sankey Cascade + Nested District Heating Gauge (ZERO top titles!) ---
    ax_a.set_xlim(0, 100)
    ax_a.set_ylim(0, 100)
    ax_a.set_xticks([])
    ax_a.set_yticks([])
    ax_a.axis("off")

    # Left Supply Stack (x = 15.0..17.0, y = 34.0..86.5) with generous spacing between labels
    x_s0, x_s1 = 15.2, 17.2
    y_ww0, h_ww = 33.0, 3.8
    y_ww1 = y_ww0 + h_ww
    y_in0, h_ind = y_ww1 + 1.8, 5.2
    y_in1 = y_in0 + h_ind
    y_pw0, h_pow = y_in1 + 1.8, 41.0
    y_pw1 = y_pw0 + h_pow

    ax_a.add_patch(mpatches.Rectangle((x_s0, y_ww0), x_s1 - x_s0, h_ww, facecolor=COL_EMERALD, edgecolor="none", zorder=4))
    ax_a.add_patch(mpatches.Rectangle((x_s0, y_in0), x_s1 - x_s0, h_ind, facecolor=COL_APRICOT, edgecolor="none", zorder=4))
    ax_a.add_patch(mpatches.Rectangle((x_s0, y_pw0), x_s1 - x_s0, h_pow, facecolor=COL_SAPPHIRE, edgecolor="none", zorder=4))

    ax_a.text(0.5 * (x_s0 + x_s1), y_pw1 + 2.2, f"{Q_tot:.2f} EJ yr$^{{-1}}$\nRecoverable waste heat", ha="center", va="bottom", fontsize=FS_COMPACT, fontweight="bold", linespacing=1.08)
    ax_a.text(x_s0 - 1.0, 0.5 * (y_pw0 + y_pw1), f"Power plants ({Q_pow:.2f})", ha="right", va="center", fontsize=FS_COMPACT)
    ax_a.text(x_s0 - 1.0, 0.5 * (y_in0 + y_in1) + 0.5, f"Heavy industry ({Q_ind:.2f})", ha="right", va="center", fontsize=FS_COMPACT)
    ax_a.text(x_s0 - 1.0, 0.5 * (y_ww0 + y_ww1) - 1.0, f"Wastewater ({Q_ww:.2f})", ha="right", va="center", fontsize=FS_COMPACT)

    # Upper peel-off ribbon ends cleanly at x = 35.0 (well left of Technical potential at x = 47.5!)
    x_t0, x_t1 = 47.0, 49.0
    h_tech = 10.5
    y_tech0, y_tech1 = 38.5, 38.5 + h_tech
    h_peel1 = h_pow - 8.5

    sankey_ribbon(ax_a, x_s1, y_pw0 + 8.5, y_pw1, 34.8, 63.0, 63.0 + h_peel1, COL_RIBBON_FADE, alpha=0.85, zorder=1.5)
    ax_a.text(27.2, 77.5, f"{Q_tot - U0:.2f} EJ yr$^{{-1}}$\nBeyond 100 km or\noutside winter demand", ha="center", va="center", fontsize=FS_COMPACT, linespacing=1.10)

    # Ribbons into Technical potential (10.64 EJ yr^-1)
    sankey_ribbon(ax_a, x_s1, y_ww0, y_ww0 + 0.7, x_t0, y_tech0, y_tech0 + 0.7, COL_EMERALD, alpha=0.84, zorder=2.5)
    sankey_ribbon(ax_a, x_s1, y_in0, y_in0 + 1.5, x_t0, y_tech0 + 0.7, y_tech0 + 2.2, COL_APRICOT, alpha=0.88, zorder=2.6)
    sankey_ribbon(ax_a, x_s1, y_pw0, y_pw0 + 8.5, x_t0, y_tech0 + 2.2, y_tech1, COL_SAPPHIRE, alpha=0.90, zorder=2.7)

    ax_a.add_patch(mpatches.Rectangle((x_t0, y_tech0), x_t1 - x_t0, h_tech, facecolor=COL_TECH_SLATE, edgecolor="none", zorder=4))
    ax_a.text(0.5 * (x_t0 + x_t1), y_tech1 + 2.4, f"Technical potential\n{U0:.2f} EJ yr$^{{-1}}$", ha="center", va="bottom", fontsize=FS_COMPACT, fontweight="bold", linespacing=1.08)

    # Separate Demand Strip cleanly at bottom-left (x = 17.5..34.0, NO crossing lines!)
    y_d0 = 5.5
    h_dtot = 14.8
    h_durb = h_dtot * (D_urb / D_tot)
    ax_a.add_patch(mpatches.Rectangle((17.5, y_d0), 1.6, h_dtot, facecolor=COL_BLDG_SLATE, edgecolor="none", zorder=4))
    sankey_ribbon(ax_a, 19.1, y_d0 + h_durb, y_d0 + h_dtot, 30.0, y_d0 + h_durb + 2.5, y_d0 + h_dtot + 2.5, "#E5EAF0", alpha=0.80, zorder=1.5)
    sankey_ribbon(ax_a, 19.1, y_d0, y_d0 + h_durb, 33.5, 7.5, 7.5 + h_durb, COL_SKY_CYAN, alpha=0.86, zorder=2.2)
    ax_a.add_patch(mpatches.Rectangle((33.5, 7.5), 1.6, h_durb, facecolor=COL_SKY_CYAN, edgecolor="none", zorder=4))
    ax_a.text(16.6, y_d0 + 0.5 * h_dtot, f"Building heating\n({D_tot:.2f})", ha="right", va="center", fontsize=FS_COMPACT, linespacing=1.08)
    ax_a.text(35.8, 7.5 + 0.5 * h_durb, f"Urban heating\nareas ({D_urb:.2f})", ha="left", va="center", fontsize=FS_COMPACT, linespacing=1.08)

    # Economic screening split: 7.59 EJ peels downward cleanly to x=57.5; 3.05 EJ arrives at x=69.5..71.5
    x_e0, x_e1 = 69.5, 71.5
    h_econ = 5.6
    y_econ0, y_econ1 = 43.4, 43.4 + h_econ
    h_unecon = h_tech - 3.6
    sankey_ribbon(ax_a, x_t1, y_tech0, y_tech0 + h_unecon, 57.5, 17.5, 17.5 + h_unecon, COL_CORAL_SOFT, alpha=0.82, zorder=2.3)
    ax_a.text(56.5, 15.2, f"{U0 - P0:.2f} EJ yr$^{{-1}}$\nCostlier than boilers\n& heat pumps", ha="center", va="top", fontsize=FS_COMPACT, linespacing=1.08)

    h_epow = h_econ * (P_pow / P0)
    h_eind = h_econ - h_epow
    sankey_ribbon(ax_a, x_t1, y_tech1 - 3.6, y_tech1 - 3.2, x_e0, y_econ0, y_econ0 + h_eind, COL_APRICOT, alpha=0.92, zorder=3.0)
    sankey_ribbon(ax_a, x_t1, y_tech1 - 3.2, y_tech1, x_e0, y_econ0 + h_eind, y_econ1, COL_SAPPHIRE, alpha=0.94, zorder=3.1)
    ax_a.add_patch(mpatches.Rectangle((x_e0, y_econ0), x_e1 - x_e0, h_eind, facecolor=COL_APRICOT, edgecolor="none", zorder=4))
    ax_a.add_patch(mpatches.Rectangle((x_e0, y_econ0 + h_eind), x_e1 - x_e0, h_epow, facecolor=COL_SAPPHIRE, edgecolor="none", zorder=4))

    # Place Economic potential title ABOVE node and share breakdown ABOVE-LEFT / BELOW-LEFT strictly clear of zoom connectors!
    ax_a.text(0.5 * (x_e0 + x_e1), y_econ1 + 2.4, f"Economic potential\n{P0:.2f} EJ yr$^{{-1}}$ (28.7%)", ha="center", va="bottom", fontsize=FS_COMPACT, fontweight="bold", linespacing=1.08)
    ax_a.text(x_e1, y_econ0 - 3.2, "92.8% power plants\n7.1% heavy industry\n<1% electric heat pumps", ha="right", va="top", fontsize=FS_COMPACT, linespacing=1.10)

    # Far-right Proportional District Heating Gauge (x = 79.5..98.2, y = 43.0..53.5)
    gx0, gx1 = 79.5, 98.2
    gw = gx1 - gx0
    gy0, gh = 43.0, 10.8
    gw_econ = gw * (P0 / DH_CUR)
    gw_pow = gw_econ * (P_pow / P0)

    ax_a.plot([x_e1 + 0.3, gx0 - 0.3], [y_econ1, gy0 + gh], ls="--", lw=0.50, color="#444444", zorder=2)
    ax_a.plot([x_e1 + 0.3, gx0 - 0.3], [y_econ0, gy0], ls="--", lw=0.50, color="#444444", zorder=2)

    ax_a.add_patch(mpatches.Rectangle((gx0, gy0), gw, gh, facecolor=COL_MIST_DH, edgecolor="#000000", linewidth=0.55, zorder=3))
    ax_a.add_patch(mpatches.Rectangle((gx0, gy0 + 1.8), gw_pow, gh - 3.6, facecolor=COL_SAPPHIRE, edgecolor="none", zorder=4))
    ax_a.add_patch(mpatches.Rectangle((gx0 + gw_pow, gy0 + 1.8), gw_econ - gw_pow, gh - 3.6, facecolor=COL_APRICOT, edgecolor="none", zorder=4))
    ax_a.plot([gx0 + gw_econ, gx0 + gw_econ], [gy0 - 1.4, gy0 + gh + 1.4], color="#000000", lw=0.85, zorder=5)

    ax_a.text(gx0 + gw_econ + 1.0, gy0 + gh + 2.4, f"{P0:.2f} EJ yr$^{{-1}}$ (40.3%)", ha="center", va="bottom", fontsize=FS_COMPACT, fontweight="bold")
    ax_a.text(gx0, gy0 - 2.2, "0%", ha="left", va="top", fontsize=FS_COMPACT)
    ax_a.text(gx1, gy0 - 2.2, "100%", ha="right", va="top", fontsize=FS_COMPACT)
    ax_a.text(0.5 * (gx0 + gx1), gy0 - 10.2, f"Current building\ndistrict heating ({DH_CUR:.2f} EJ yr$^{{-1}}$)", ha="center", va="top", fontsize=FS_COMPACT, linespacing=1.10)

    # --- Panel b: 1946–2100 Observed | Projected (100% pure black labels!) ---
    clean_ax(ax_b, "y")
    r1 = df_glob[df_glob["group"] == "R1"].copy()
    r1["mid"] = 0.5 * (r1["start_year"] + r1["end_year"])
    obs = r1[r1["ssp"] == "obsclim"].sort_values("mid")

    ax_b.axvspan(1946, 2015.5, facecolor=COL_HIST_BG, zorder=0)
    ax_b.axvline(2015.5, ls=":", lw=0.65, color="#333333", zorder=2)
    ax_b.axhline(P0, ls="--", lw=0.55, color="#333333", zorder=2)
    ax_b.text(2012.5, 3.63, "Observed", ha="right", va="top", fontsize=FS_COMPACT)
    ax_b.text(2018.5, 3.63, "Projected", ha="left", va="top", fontsize=FS_COMPACT)

    ax_b.plot(obs["mid"], obs["P_ej_yr"], color="#000000", lw=1.65, marker="o", ms=3.6, zorder=6)
    obs_pts = list(zip(obs["mid"], obs["P_ej_yr"]))
    ax_b.text(obs_pts[0][0], obs_pts[0][1] + 0.05, "3.42", ha="center", va="bottom", fontsize=FS_COMPACT)
    ax_b.text(obs_pts[1][0], obs_pts[1][1] + 0.05, "3.42", ha="center", va="bottom", fontsize=FS_COMPACT)
    ax_b.text(obs_pts[2][0] + 1.5, obs_pts[2][1] + 0.05, "3.24", ha="center", va="bottom", fontsize=FS_COMPACT)
    ax_b.text(obs_pts[3][0] + 2.2, obs_pts[3][1] + 0.05, "3.05", ha="left", va="bottom", fontsize=FS_COMPACT)

    base_mod = r1[r1["window_id"] == "2006_2025"].set_index(["model", "ssp"])["P_ej_yr"].to_dict()
    ssp_cfg = [("ssp126", "SSP1-2.6", COL_SAPPHIRE), ("ssp245", "SSP2-4.5", COL_SSP2), ("ssp370", "SSP3-7.0", COL_CORAL)]
    src_rows = []
    for _, orow in obs.iterrows():
        src_rows.append({"series": "Observed", "mid_year": orow["mid"], "P_mean_ej": orow["P_ej_yr"], "P_min_ej": orow["P_ej_yr"], "P_max_ej": orow["P_ej_yr"]})

    for scode, slbl, scol in ssp_cfg:
        sub = r1[r1["ssp"] == scode].copy()
        sub["P_plot"] = np.where(sub["window_id"] == "2006_2025", P0, sub["P_ej_yr"])
        for mcode, _, mk in GCM_ORDER:
            msub = sub[sub["model"] == mcode].sort_values("mid")
            ax_b.plot(msub["mid"], msub["P_plot"], color=scol, lw=0.45, alpha=0.38, zorder=3)
            end_r = msub[msub["window_id"] == "2081_2100"]
            if len(end_r):
                ax_b.scatter(end_r["mid"], end_r["P_plot"], color="none", edgecolors=scol, marker=mk, s=13, linewidths=0.65, zorder=6)
        grp = sub.groupby("mid")["P_plot"].agg(["mean", "min", "max"]).reset_index()
        ax_b.fill_between(grp["mid"], grp["min"], grp["max"], color=scol, alpha=0.16, lw=0, zorder=2)
        ax_b.plot(grp["mid"], grp["mean"], color=scol, lw=1.65, zorder=5)
        end_m = float(grp.iloc[-1]["mean"])
        ax_b.scatter([2090.5], [end_m], color=scol, marker="o", s=16, zorder=7)
        ax_b.text(2093.5, end_m, f"{end_m:.2f}", ha="left", va="center", fontsize=FS_COMPACT, fontweight="bold")
        if scode == "ssp370":
            min_end = float(grp.iloc[-1]["min"])
            ax_b.text(2093.5, min_end, f"{min_end:.2f}", ha="left", va="center", fontsize=FS_COMPACT)
        for _, gr in grp.iterrows():
            src_rows.append({"series": slbl, "mid_year": gr["mid"], "P_mean_ej": gr["mean"], "P_min_ej": gr["min"], "P_max_ej": gr["max"]})

    ax_b.set_xlim(1946, 2110)
    ax_b.set_ylim(1.50, 3.68)
    ax_b.set_xticks([1950, 1975, 2000, 2025, 2050, 2075, 2100])
    ax_b.set_yticks([1.5, 2.0, 2.5, 3.0, 3.5])
    ax_b.set_xlabel("Year")
    ax_b.set_ylabel("Global economic potential (EJ yr$^{-1}$)")

    s_h = [mpl.lines.Line2D([], [], color=sc, lw=1.6, label=sl) for _, sl, sc in ssp_cfg]
    ax_b.legend(handles=s_h, loc="upper right", bbox_to_anchor=(0.98, 0.99), frameon=False, handlelength=1.3, labelspacing=0.22)

    ax_in = ax_b.inset_axes([0.135, 0.11, 0.335, 0.34])
    clean_ax(ax_in, None)
    ax_in.axhline(0, ls="--", lw=0.45, color="#333333")
    for wi, wid in enumerate(["2046_2065", "2081_2100"]):
        for si, (scode, _, scol) in enumerate(ssp_cfg):
            xp = wi * 1.08 + (si - 1) * 0.24
            sub = r1[(r1["ssp"] == scode) & (r1["window_id"] == wid)]
            dps = [100.0 * (row["P_ej_yr"] / base_mod[(row["model"], scode)] - 1.0) for _, row in sub.iterrows()]
            mv, lo, hi = float(np.mean(dps)), float(np.min(dps)), float(np.max(dps))
            ax_in.errorbar(xp, mv, yerr=[[mv - lo], [hi - mv]], fmt="s", ms=2.8, color=scol, ecolor=scol, elinewidth=0.75, capsize=1.8, zorder=4)
            for (_, row), dpv in zip(sub.iterrows(), dps):
                ax_in.scatter(xp, dpv, facecolors="none", edgecolors=scol, marker=GCM_MARKER[row["model"]], s=7, linewidths=0.45, zorder=3)
    ax_in.set_xlim(-0.50, 1.58)
    ax_in.set_ylim(-52, 4)
    ax_in.set_xticks([0, 1.08])
    ax_in.set_xticklabels(["2046–2065", "2081–2100"], fontsize=FS_COMPACT)
    ax_in.set_yticks([0, -20, -40])
    ax_in.set_yticklabels(["0", f"{MINUS}20", f"{MINUS}40"], fontsize=FS_COMPACT)
    ax_in.set_ylabel("Change (%)", fontsize=FS_COMPACT, labelpad=1.2)

    # --- Panel c: Raincloud Half-Violin + White IQR Box + 33 Points + OLS ---
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["left"].set_visible(True)
    ax_c.spines["right"].set_visible(True)
    lock_spines_above_data(ax_c)
    ax_c.tick_params(axis="y", left=True, labelleft=False)
    ax_c.grid(axis="y", linestyle="--", linewidth=0.40, alpha=0.28, color="#888888", zorder=0)
    ax_c.set_axisbelow(True)
    ax_c.axhline(P0, ls="--", lw=0.55, color="#333333", zorder=1)

    gwl = df_glob[df_glob["group"] == "GWL"].copy()
    gwl["x_c"] = gwl["subgroup"].map({"GWL_1.5C": 1.5, "GWL_2.0C": 2.0, "GWL_3.0C": 3.0})
    gwl["P_al"] = gwl["P_ej_yr"].astype(float)
    gwl["dp_pct"] = 100.0 * (gwl["P_al"] / P0 - 1.0)

    rng = np.random.default_rng(11)
    sc_col = {"ssp126": COL_SAPPHIRE, "ssp245": COL_SSP2, "ssp370": COL_CORAL}

    for xv in [1.5, 2.0, 3.0]:
        sub = gwl[gwl["x_c"] == xv]
        vals = sub["P_al"].to_numpy()
        yk = np.linspace(vals.min() - 0.08, vals.max() + 0.08, 80)
        kde = stats.gaussian_kde(vals, bw_method=0.55)
        dens = kde(yk)
        dens = dens / dens.max() * 0.20
        ax_c.fill_betweenx(yk, xv - 0.05 - dens, xv - 0.05, facecolor=COL_SAPPHIRE, alpha=0.68, edgecolor="#000000", linewidth=0.50, zorder=3)

        bp = ax_c.boxplot([vals], positions=[xv], widths=0.075, patch_artist=True, showfliers=False, zorder=4)
        for b in bp["boxes"]:
            b.set(facecolor="white", edgecolor="#000000", linewidth=0.60)
        for m in bp["medians"]:
            m.set(color="#000000", linewidth=1.0)
        for w in bp["whiskers"] + bp["caps"]:
            w.set(color="#000000", linewidth=0.55)

        jit = rng.uniform(0.08, 0.21, size=len(sub))
        for jx, (_, r) in zip(jit, sub.iterrows()):
            ax_c.scatter(xv + jx, r["P_al"], color=sc_col[r["ssp"]], marker=GCM_MARKER[r["model"]], s=14, edgecolors="white", linewidths=0.25, zorder=5)

    m_h = [mpl.lines.Line2D([], [], color="none", markeredgecolor="#000000", marker=mk, ms=3.5, markeredgewidth=0.65, label=ml) for _, ml, mk in GCM_ORDER]
    ax_c.legend(handles=m_h, loc="upper right", bbox_to_anchor=(0.99, 0.99), frameon=False, fontsize=FS_COMPACT, handlelength=0.8, labelspacing=0.18)

    res = stats.linregress(gwl["x_c"], gwl["P_al"])
    res_pct = stats.linregress(gwl["x_c"], gwl["dp_pct"])
    xx = np.linspace(1.38, 3.18, 60)
    yy = res.intercept + res.slope * xx
    n_g = len(gwl)
    se = res.stderr * np.sqrt(1 / n_g + (xx - gwl["x_c"].mean()) ** 2 / ((gwl["x_c"] - gwl["x_c"].mean()) ** 2).sum())
    t_q = stats.t.ppf(0.975, n_g - 2)
    ax_c.fill_between(xx, yy - t_q * se, yy + t_q * se, color="#D8DEE6", alpha=0.65, lw=0, zorder=2)
    ax_c.plot(xx, yy, color="#000000", lw=1.15, zorder=4.5)

    ax_c.text(0.04, 0.03, "$n$ = 33", transform=ax_c.transAxes, ha="left", va="bottom", fontsize=FS_COMPACT)
    ax_c.text(0.97, 0.03, f"Slope = {fm(res_pct.slope, '.1f')}% °C$^{{-1}}$, $R^2$ = {res_pct.rvalue**2:.2f}", transform=ax_c.transAxes, ha="right", va="bottom", fontsize=FS_COMPACT)

    ax_c.set_xlim(1.15, 3.38)
    ax_c.set_xticks([1.5, 2.0, 3.0])
    ax_c.set_xticklabels(["1.5", "2.0", "3.0"])
    ax_c.set_xlabel("Global warming level (°C)")

    ax_cr = ax_c.twinx()
    ax_cr.spines["top"].set_visible(False)
    ax_cr.spines["left"].set_visible(False)
    lock_spines_above_data(ax_cr)
    ylo, yhi = ax_b.get_ylim()
    ax_cr.set_ylim(100 * (ylo / P0 - 1), 100 * (yhi / P0 - 1))
    ax_cr.set_yticks([15, 0, -15, -30, -45])
    ax_cr.set_yticklabels(["+15", "0", f"{MINUS}15", f"{MINUS}30", f"{MINUS}45"])
    ax_cr.set_ylabel("Change relative to 2006–2025 (%)")

    panel_tag(fig, 0.015, 0.985, "a")
    panel_tag(fig, 0.015, 0.562, "b")
    panel_tag(fig, 0.575, 0.562, "c")

    SRC_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(src_rows).to_csv(SRC_DIR / "Figure2_source_data.csv", index=False)
    return export_figure(fig, "Figure2_Global_Erosion_Trajectory")


# ==============================================================================
# FIGURE 3 (Results 2): Robinson World Map + Unlabelled Inset & 3-Term Mechanism
# ==============================================================================
def build_figure_3(df_country: pd.DataFrame) -> dict:
    configure_matplotlib_style()
    H_MM = 180.0
    fig = plt.figure(figsize=(W_IN, H_MM * MM))

    ax_map = fig.add_axes([0.025, 0.482, 0.955, 0.492])
    ax_b = fig.add_axes([0.172, 0.058, 0.328, 0.338])
    ax_c = fig.add_axes([0.635, 0.058, 0.338, 0.338])

    z_all = pd.read_parquet(SCEN_DIR / "zones/R1__obs__2006_2025_zones.parquet")
    z_uhz = pd.read_parquet(DATA_PROC / "sinks_2025/urban_heat_zones.parquet")[["zone_id", "lat", "lon"]]
    if "lat" not in z_all.columns:
        z_all = z_all.merge(z_uhz, on="zone_id", how="left")
    u_col = "U_gj" if "U_gj" in z_all.columns else [c for c in z_all.columns if c.startswith("U")][0]
    p_col = "P_gj" if "P_gj" in z_all.columns else [c for c in z_all.columns if c.startswith("P")][0]
    z_zero = z_all[(z_all[u_col] > 0) & (z_all[p_col] <= 0)].copy()

    df_z = pd.read_parquet(SCEN_DIR / "r2_decomposition/r2_zone_midcentury_ssp245_ensmean.parquet")
    gdf_bg = load_world_basemap(ROOT).to_crs("+proj=robin")
    trans = Transformer.from_crs("EPSG:4326", "+proj=robin", always_xy=True)

    lats_b = np.linspace(-90, 90, 181)
    lons_b = np.linspace(-180, 180, 361)
    bx = np.concatenate([np.full_like(lats_b, -180), lons_b, np.full_like(lats_b, 180), lons_b[::-1]])
    by = np.concatenate([lats_b, np.full_like(lons_b, 90), lats_b[::-1], np.full_like(lons_b, -90)])
    px, py = trans.transform(bx, by)
    ax_map.fill(px, py, facecolor="#FFFFFF", edgecolor="#333333", linewidth=0.58, zorder=0)

    for glat in [-60, -30, 0, 30, 60]:
        gx, gy = trans.transform(np.linspace(-180, 180, 181), np.full(181, glat))
        ax_map.plot(gx, gy, color="#D8E0E8", lw=0.35, zorder=0.4)
        lx, ly = trans.transform(-180, glat)
        lbl_g = f"{abs(glat)}°{'N' if glat > 0 else ('S' if glat < 0 else '')}" if glat != 0 else "0°"
        ax_map.text(lx - 2.2e5, ly, lbl_g, ha="right", va="center", fontsize=FS_COMPACT)
    for glon in [-120, -60, 0, 60, 120]:
        gx, gy = trans.transform(np.full(121, glon), np.linspace(-80, 80, 121))
        ax_map.plot(gx, gy, color="#D8E0E8", lw=0.35, zorder=0.4)

    gdf_bg.plot(ax=ax_map, facecolor="#F3F5F8", edgecolor="#8A96A3", linewidth=0.28, zorder=1)

    zx0, zy0 = trans.transform(z_zero["lon"].to_numpy(), z_zero["lat"].to_numpy())
    ax_map.scatter(zx0, zy0, s=0.45, color=COL_ZERO_DOT, alpha=0.58, linewidths=0, rasterized=True, zorder=2)

    zp = df_z.sort_values("P0_gj").copy()
    vx, vy = trans.transform(zp["lon"].to_numpy(), zp["lat"].to_numpy())
    cmap_loss = mcolors.LinearSegmentedColormap.from_list(
        "pro_fig3_ramp", ["#FEF098", "#F9B872", "#E65C3B", "#AC2428", "#671019"], N=256
    )
    vals_loss = np.clip(-zp["dP_rel_pct"].to_numpy(), 0.0, 45.0)
    sz_v = np.clip(np.sqrt(zp["P0_gj"].to_numpy() / 1e6) * 8.2, 1.8, 56.0)
    sc = ax_map.scatter(vx, vy, c=vals_loss, cmap=cmap_loss, vmin=0, vmax=45, s=sz_v, alpha=0.92, edgecolors="white", linewidths=0.32, rasterized=True, zorder=3)

    callouts = [
        ("United States\n(2.08 EJ technical,\n0.001 EJ economic)", -98, 37, -142, 26),
        ("Sweden\n(\u221243.3%)", 16, 61, 5, 78),
        ("Denmark\n(\u221214.8%)", 10, 56, -25, 57),
        ("Germany\n(0.22 EJ, \u221210.8%)", 10, 50, -26, 40),
        ("Italy\n(\u221220.4%)", 12.5, 43, 18, 24),
        ("Russia\n(0.89 EJ, \u22129.2%)", 62, 56, 86, 70),
        ("China\n(1.33 EJ, \u221212.0%)", 117, 37, 155, 22),
    ]
    for txt_c, lon0, lat0, lon1, lat1 in callouts:
        x0, y0 = trans.transform(lon0, lat0)
        x1, y1 = trans.transform(lon1, lat1)
        ax_map.plot([x0, x1], [y0, y1], color="#000000", lw=0.50, zorder=5)
        ax_map.text(
            x1, y1, txt_c, fontsize=FS_COMPACT, ha="center", va="center", linespacing=1.12,
            bbox=dict(boxstyle="square,pad=0.10", facecolor="white", edgecolor="none", alpha=0.84), zorder=6
        )

    ax_map.set_xlim(px.min() * 1.03, px.max() * 1.03)
    ax_map.set_ylim(py.min() * 1.02, py.max() * 1.03)
    ax_map.set_xticks([])
    ax_map.set_yticks([])
    ax_map.set_xlabel("")
    ax_map.set_ylabel("")
    ax_map.axis("off")

    # South Pacific Paired-Bar Inset — ZERO inline bar numbers + bottom scale + pure black text!
    ax_pac = fig.add_axes([0.168, 0.586, 0.132, 0.132])
    ax_pac.patch.set_alpha(0.88)
    ax_pac.patch.set_facecolor("white")
    clean_ax(ax_pac, None)

    b0 = df_country[df_country["scenario_id"] == "R1__obs__2006_2025"].copy()
    b0["U_ej"] = b0["U_gj"] / 1e9
    b0["P_ej"] = b0["P_gj"] / 1e9
    pac_isos = ["USA", "CHN", "RUS", "DEU", "GBR", "KOR", "CAN", "POL"]
    pac_df = b0.set_index("iso3").loc[pac_isos].iloc[::-1].reset_index()
    yp = np.arange(len(pac_df))
    ax_pac.barh(yp + 0.18, pac_df["U_ej"], height=0.34, color="#BDCAD8", edgecolor="none", zorder=3, label="Technical")
    ax_pac.barh(yp - 0.18, pac_df["P_ej"], height=0.34, color=COL_SAPPHIRE, edgecolor="none", zorder=3, label="Economic")
    ax_pac.set_yticks(yp)
    ax_pac.set_yticklabels(["UK" if i == "GBR" else cname(i) for i in pac_df["iso3"]], fontsize=FS_COMPACT)
    ax_pac.tick_params(axis="both", length=1.8, labelsize=FS_COMPACT)
    ax_pac.set_xlim(0, 2.30)
    ax_pac.set_xticks([0, 1, 2])
    ax_pac.text(0.46, 1.24, "Technical vs Economic\npotential (EJ yr$^{-1}$)", transform=ax_pac.transAxes, ha="center", va="bottom", fontsize=FS_COMPACT, linespacing=1.10)
    ax_pac.legend(loc="lower center", bbox_to_anchor=(0.46, 1.00), ncol=2, frameon=False, fontsize=FS_COMPACT, handlelength=0.8, columnspacing=0.6, handletextpad=0.3)
    lock_spines_above_data(ax_pac)

    # Use real Line2D markers for map legend so 100% of Text objects are pure #000000!
    map_leg_h = [
        mpl.lines.Line2D([], [], color="none", marker="o", markerfacecolor=COL_ZERO_DOT, markeredgecolor="none", ms=4.5, label="52,324 urban areas with zero economic potential"),
        mpl.lines.Line2D([], [], color="none", marker="o", markerfacecolor="white", markeredgecolor="#000000", markeredgewidth=0.65, ms=4.8, label="5,304 viable urban heating areas, 3.05 EJ yr$^{-1}$"),
    ]
    fig.legend(handles=map_leg_h, loc="center left", bbox_to_anchor=(0.422, 0.542), frameon=False, fontsize=FS_COMPACT, handletextpad=0.35, labelspacing=0.35)

    cax = fig.add_axes([0.380, 0.484, 0.255, 0.013])
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal", extend="max")
    cb.outline.set_linewidth(0.45)
    cb.outline.set_edgecolor("#000000")
    cb.set_ticks([0, 15, 25, 35, 45])
    cb.set_ticklabels(["0%", f"{MINUS}15%", f"{MINUS}25%", f"{MINUS}35%", f"{MINUS}45%"])
    cb.ax.tick_params(labelsize=FS_COMPACT, length=2.2, width=0.45)
    cb.set_label("2046–2065 SSP2-4.5 change (%)", fontsize=FS_UNIFIED, labelpad=2.0)

    leg_bc = [
        mpatches.Patch(facecolor=COL_DEM_BLUE, edgecolor="none", label="Heating demand"),
        mpatches.Patch(facecolor=COL_SEASONAL, edgecolor="none", label="Seasonal matching"),
        mpatches.Patch(facecolor=COL_CORAL, edgecolor="none", label="Economic viability"),
        mpl.lines.Line2D([], [], color="#000000", marker="D", lw=0.85, ms=3.5, label="Net change"),
    ]
    fig.legend(handles=leg_bc, loc="upper center", bbox_to_anchor=(0.52, 0.448), ncol=4, frameon=False, handlelength=1.0, columnspacing=1.2, handletextpad=0.4)

    df_dec = pd.read_parquet(SCEN_DIR / "r2_decomposition/r2_eq6_decomposition_all.parquet")
    df_geq = pd.read_csv(SCEN_DIR / "r2_decomposition/r2_eq6_global_summary.csv")

    def agg_dec(sub: pd.DataFrame, lbl: str) -> dict:
        net_v = float(sub["dP_rel_pct"].mean())
        t1 = float(sub["term1_demand_shift_pct"].mean())
        t2 = float(sub["term2_seasonal_mismatch_pct"].mean())
        t3 = net_v - t1 - t2
        return {
            "lbl": lbl, "t1": t1, "t2": t2, "t3": t3, "net": net_v,
            "lo": float(sub["dP_rel_pct"].min()), "hi": float(sub["dP_rel_pct"].max()),
        }

    g_mid = agg_dec(df_geq[(df_geq["ssp"] == "ssp245") & (df_geq["window_id"] == "2046_2065")], "Global (2046–2065 SSP2-4.5)")
    g_end = agg_dec(df_geq[(df_geq["ssp"] == "ssp370") & (df_geq["window_id"] == "2081_2100")], "Global (2081–2100 SSP3-7.0)")

    isos_b = ["RUS", "POL", "NLD", "GBR", "DEU", "CHN", "KAZ", "BLR", "CZE", "KOR", "DNK", "ITA", "SWE"]
    c_sub = df_dec[(df_dec["level"] == "country") & (df_dec["ssp"] == "ssp245") & (df_dec["window_id"] == "2046_2065")]
    c_rows = [agg_dec(c_sub[c_sub["entity_id"] == iso], cname(iso)) for iso in isos_b if len(c_sub[c_sub["entity_id"] == iso])]
    c_rows = sorted(c_rows, key=lambda d: d["net"], reverse=True)
    rows_b = ([g_mid, g_end] + c_rows)[::-1]

    clean_ax(ax_b, "x")
    yb = np.arange(len(rows_b))
    highlight_b = {"Global (2046–2065 SSP2-4.5)", "Global (2081–2100 SSP3-7.0)", "Russia", "Germany", "China", "Denmark", "Italy", "Sweden"}
    for yi, r in enumerate(rows_b):
        neg_l, pos_l = 0.0, 0.0
        for val, col in [(r["t3"], COL_CORAL), (r["t1"], COL_DEM_BLUE), (r["t2"], COL_SEASONAL)]:
            if val < 0:
                ax_b.barh(yi, val, left=neg_l, height=0.66, color=col, edgecolor="none", zorder=3)
                if neg_l < -0.05:
                    ax_b.plot([neg_l, neg_l], [yi - 0.34, yi + 0.34], color="white", lw=0.55, zorder=3.5)
                neg_l += val
            elif val > 0:
                ax_b.barh(yi, val, left=pos_l, height=0.66, color=col, edgecolor="none", zorder=3)
                if pos_l > 0.05:
                    ax_b.plot([pos_l, pos_l], [yi - 0.34, yi + 0.34], color="white", lw=0.55, zorder=3.5)
                pos_l += val
        if r["lo"] < -59.0:
            ax_b.errorbar(r["net"], yi, xerr=[[0.0], [r["hi"] - r["net"]]], fmt="D", ms=3.2, color="#000000", ecolor="#000000", elinewidth=0.65, capsize=1.6, zorder=6)
            ax_b.annotate("", xy=(-59.2, yi), xytext=(r["net"], yi), arrowprops=dict(arrowstyle="->,head_width=0.14,head_length=0.22", color="#000000", lw=0.65), zorder=6)
            ax_b.text(pos_l + 0.8, yi, f"{fm(r['net'], '.1f')}%", va="center", ha="left", fontsize=FS_COMPACT)
        else:
            ax_b.errorbar(r["net"], yi, xerr=[[r["net"] - r["lo"]], [r["hi"] - r["net"]]], fmt="D", ms=3.2, color="#000000", ecolor="#000000", elinewidth=0.65, capsize=1.6, zorder=6)
            if r["lbl"] in highlight_b:
                ax_b.text(min(neg_l, r["lo"]) - 1.0, yi, f"{fm(r['net'], '.1f')}%", va="center", ha="right", fontsize=FS_COMPACT)

    ax_b.axvline(0, lw=0.65, color="#000000", zorder=18)
    ax_b.axhline(len(c_rows) - 0.5, ls="--", lw=0.50, color="#444444", zorder=10)
    ax_b.set_yticks(yb)
    ax_b.set_yticklabels([r["lbl"] for r in rows_b], fontsize=FS_COMPACT)
    ax_b.set_xlim(-60, 14)
    ax_b.set_ylim(-0.65, len(rows_b) - 0.35)
    ax_b.set_xticks([-50, -30, -10, 0, 10])
    ax_b.set_xticklabels([f"{MINUS}50", f"{MINUS}30", f"{MINUS}10", "0", "+10"])
    ax_b.set_xlabel("Contribution to potential change (%)")
    lock_spines_above_data(ax_b)

    clean_ax(ax_c, "x")
    df_lat = pd.read_csv(SCEN_DIR / "r2_decomposition/r2_stratified_by_abs_lat_band.csv")
    df_sz = pd.read_csv(SCEN_DIR / "r2_decomposition/r2_stratified_by_size_tier.csv")
    df_lat = df_lat[df_lat["P0_pj"] > 0.5].copy()

    z_meta = df_z[["zone_id", "P0_gj", "lat", "baseline_mean_mw"]].copy()
    z_meta["lat_lbl"] = pd.cut(z_meta["lat"].abs(), bins=[0, 30, 40, 50, 60, 90], labels=["<30°", "30–40°", "40–50°", "50–60°", ">60°"], include_lowest=True)
    z_meta["sz_lbl"] = pd.cut(z_meta["baseline_mean_mw"], bins=[0, 10, 100, 1e9], labels=["1–10 MW areas", "10–100 MW areas", ">100 MW areas"], include_lowest=True)
    gcm_lat_pts: dict[str, list[tuple[str, float]]] = {}
    gcm_sz_pts: dict[str, list[tuple[str, float]]] = {}
    for mcode, _, _ in GCM_ORDER:
        zfp = SCEN_DIR / f"zones/R1__{mcode}__ssp245__2046_2065_zones.parquet"
        if zfp.exists():
            zm = pd.read_parquet(zfp)[["zone_id", p_col]].merge(z_meta, on="zone_id", how="inner")
            for band, grp_b in zm.groupby("lat_lbl", observed=True):
                dp_b = 100.0 * (grp_b[p_col].sum() - grp_b["P0_gj"].sum()) / max(1.0, grp_b["P0_gj"].sum())
                gcm_lat_pts.setdefault(str(band), []).append((mcode, float(dp_b)))
            for stier, grp_s in zm.groupby("sz_lbl", observed=True):
                dp_s = 100.0 * (grp_s[p_col].sum() - grp_s["P0_gj"].sum()) / max(1.0, grp_s["P0_gj"].sum())
                gcm_sz_pts.setdefault(str(stier), []).append((mcode, float(dp_s)))

    lat_rows = []
    for _, r in df_lat.iterrows():
        raw_l = str(r["abs_lat_band"])
        lbl = raw_l.replace(" deg", "°").replace("-", "–")
        pts = gcm_lat_pts.get(lbl, [])
        vals_p = [v for _, v in pts]
        lat_rows.append({
            "lbl": lbl, "t1": float(r["term1_demand_shift_pct"]), "t2": float(r["term2_seasonal_mismatch_pct"]), "t3": float(r["term3_cost_exit_pct"]),
            "net": float(r["dP_rel_pct"]),
            "lo": float(min(vals_p)) if vals_p else float(r["dP_rel_pct"]),
            "hi": float(max(vals_p)) if vals_p else float(r["dP_rel_pct"]),
            "pts": pts,
        })
    lat_order = [">60°", "50–60°", "40–50°", "30–40°", "<30°"]
    lat_rows = sorted(lat_rows, key=lambda d: lat_order.index(d["lbl"]) if d["lbl"] in lat_order else 99)

    sz_rows = []
    for _, r in df_sz.iterrows():
        raw_s = str(r["size_tier"])
        lbl = raw_s.replace("Small (1-10 MW)", "1–10 MW areas").replace("Medium (10-100 MW)", "10–100 MW areas").replace("Large (>100 MW)", ">100 MW areas")
        pts = gcm_sz_pts.get(lbl, [])
        vals_p = [v for _, v in pts]
        sz_rows.append({
            "lbl": lbl, "t1": float(r["term1_demand_shift_pct"]), "t2": float(r["term2_seasonal_mismatch_pct"]), "t3": float(r["term3_cost_exit_pct"]),
            "net": float(r["dP_rel_pct"]),
            "lo": float(min(vals_p)) if vals_p else float(r["dP_rel_pct"]),
            "hi": float(max(vals_p)) if vals_p else float(r["dP_rel_pct"]),
            "pts": pts,
        })
    sz_order = [">100 MW areas", "10–100 MW areas", "1–10 MW areas"]
    sz_rows = sorted(sz_rows, key=lambda d: sz_order.index(d["lbl"]) if d["lbl"] in sz_order else 99)

    rows_c = (lat_rows + sz_rows)[::-1]
    yc = np.arange(len(rows_c))
    for yi, r in enumerate(rows_c):
        neg_l, pos_l = 0.0, 0.0
        for val, col in [(r["t3"], COL_CORAL), (r["t1"], COL_DEM_BLUE), (r["t2"], COL_SEASONAL)]:
            if val < 0:
                ax_c.barh(yi, val, left=neg_l, height=0.58, color=col, edgecolor="none", zorder=3)
                if neg_l < -0.05:
                    ax_c.plot([neg_l, neg_l], [yi - 0.30, yi + 0.30], color="white", lw=0.55, zorder=3.5)
                neg_l += val
            elif val > 0:
                ax_c.barh(yi, val, left=pos_l, height=0.58, color=col, edgecolor="none", zorder=3)
                if pos_l > 0.05:
                    ax_c.plot([pos_l, pos_l], [yi - 0.30, yi + 0.30], color="white", lw=0.55, zorder=3.5)
                pos_l += val
        ax_c.errorbar(r["net"], yi, xerr=[[r["net"] - r["lo"]], [r["hi"] - r["net"]]], fmt="D", ms=3.2, color="#000000", ecolor="#000000", elinewidth=0.65, capsize=1.6, zorder=6)
        for m_id, m_val in r["pts"]:
            ax_c.scatter(m_val, yi, facecolors="none", edgecolors="#000000", marker=GCM_MARKER[m_id], s=9.5, linewidths=0.48, zorder=5)
        ax_c.text(min(neg_l, r["lo"]) - 1.0, yi, f"{fm(r['net'], '.1f')}%", va="center", ha="right", fontsize=FS_COMPACT)

    ax_c.axvline(0, lw=0.65, color="#000000", zorder=18)
    ax_c.axhline(len(sz_rows) - 0.5, ls="--", lw=0.50, color="#444444", zorder=10)
    ax_c.set_yticks(yc)
    ax_c.set_yticklabels([r["lbl"] for r in rows_c], fontsize=FS_COMPACT)
    ax_c.set_xlim(-52, 8)
    ax_c.set_ylim(-0.62, len(rows_c) - 0.38)
    ax_c.set_xticks([-40, -30, -20, -10, 0, 8])
    ax_c.set_xticklabels([f"{MINUS}40", f"{MINUS}30", f"{MINUS}20", f"{MINUS}10", "0", "+8"])
    ax_c.set_xlabel("Contribution to potential change (%)")
    lock_spines_above_data(ax_c)

    panel_tag(fig, 0.015, 0.982, "a")
    panel_tag(fig, 0.015, 0.415, "b")
    panel_tag(fig, 0.518, 0.415, "c")

    pd.DataFrame([{k: v for k, v in r.items() if k not in ("sub", "pts")} for r in (rows_b + rows_c)]).to_csv(SRC_DIR / "Figure3_source_data.csv", index=False)
    return export_figure(fig, "Figure3_Spatial_Concentration_and_Mechanism")


# ==============================================================================
# FIGURE 4 (Results 3): 5,327-Network Semantic Quadrant & Stranded Capital/Carbon
# ==============================================================================
def build_figure_4() -> dict:
    configure_matplotlib_style()
    H_MM = 178.0
    fig = plt.figure(figsize=(W_IN, H_MM * MM))

    r3_dir = SCEN_DIR / "r3_locked_assets"
    df_r3_sum = pd.read_csv(r3_dir / "r3_locked_assets_summary.csv")
    near_files = sorted(r3_dir.glob("R3__*__ssp245__2026_2045_links.parquet"))
    mid_files = sorted(r3_dir.glob("R3__*__ssp245__2046_2065_links.parquet"))

    dfs_mid = []
    for fp in mid_files:
        m_id = fp.name.split("__")[1]
        d = pd.read_parquet(fp)
        d["model"] = m_id
        dfs_mid.append(d)
    df_mid_all = pd.concat(dfs_mid, ignore_index=True)

    df_lk = df_mid_all.groupby(["iso3", "stream_id", "zone_id"], as_index=False).agg(
        cap=("total_sunk_capex_usd", "first"),
        d0=("sub_D_0_gj_yr", "first"),
        dfut=("sub_D_fut_gj_yr", "mean"),
        lcoh0=("lcoh_0_usd_per_gj", "first"),
        lcoh_fut=("lcoh_fut_usd_per_gj", "mean"),
        nb0=("net_benefit_0_usd_yr", "first"),
        nbfut=("net_benefit_fut_usd_yr", "mean"),
    )
    df_lk["m0"] = df_lk["nb0"] / np.maximum(1.0, df_lk["d0"])
    df_lk["dd_pct"] = 100.0 * (df_lk["dfut"] - df_lk["d0"]) / np.maximum(1.0, df_lk["d0"])
    df_lk["stranded"] = df_lk["nbfut"] < 0

    df_near_all = pd.concat([pd.read_parquet(f) for f in near_files], ignore_index=True)
    df_lk_near = df_near_all.groupby(["stream_id", "zone_id"], as_index=False)[["lcoh_fut_usd_per_gj", "sub_D_fut_gj_yr"]].mean()

    ax_a = fig.add_axes([0.076, 0.458, 0.432, 0.452])
    ax_a_top = fig.add_axes([0.076, 0.914, 0.432, 0.054], sharex=ax_a)
    ax_a_rt = fig.add_axes([0.512, 0.458, 0.045, 0.452], sharey=ax_a)

    ax_b1 = fig.add_axes([0.692, 0.458, 0.120, 0.448])
    ax_b2 = fig.add_axes([0.836, 0.458, 0.122, 0.448], sharey=ax_b1)

    ax_c1 = fig.add_axes([0.076, 0.070, 0.182, 0.285])
    ax_c2 = fig.add_axes([0.345, 0.070, 0.136, 0.285])
    ax_d = fig.add_axes([0.588, 0.070, 0.378, 0.285])

    for sp in ax_a.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.60)
        sp.set_edgecolor("#000000")
        sp.set_zorder(20)

    ax_a.axhspan(3.0, 15.2, facecolor=COL_ICE_ZONE, zorder=0)
    ax_a.axhspan(0.0, 3.0, facecolor=COL_BLUSH_ZONE, zorder=0)
    ax_a.axhline(3.0, ls="--", lw=0.65, color="#000000", zorder=4)
    ax_a.text(-39.2, 3.25, "Threshold", fontsize=FS_COMPACT, va="bottom", ha="left",
              bbox=dict(boxstyle="square,pad=0.18", facecolor=COL_ICE_ZONE, edgecolor="none", alpha=0.96), zorder=5)
    ax_a.text(-39.2, 0.30, "Initial saving <3.0 USD GJ$^{-1}$ holds\n92.2% of exposed capital (34.8B USD)", fontsize=FS_COMPACT, va="bottom", ha="left", linespacing=1.14,
              bbox=dict(boxstyle="square,pad=0.06", facecolor=COL_BLUSH_ZONE, edgecolor="none", alpha=0.88), zorder=5)

    for mask, col, zo in [(~df_lk["stranded"], COL_SAPPHIRE, 2.8), (df_lk["stranded"], COL_CORAL, 3.2)]:
        sub = df_lk[mask].sort_values("cap", ascending=False)
        sz = np.clip(np.sqrt(sub["cap"].to_numpy() / 1e6) * 1.45, 2.2, 32.0)
        ax_a.scatter(
            np.clip(sub["dd_pct"], -39.5, -0.2), np.clip(sub["m0"], 0.05, 14.9),
            s=sz, color=col, alpha=0.72, edgecolors="white", linewidths=0.28, rasterized=True, zorder=zo
        )

    ax_a.add_patch(mpatches.Rectangle((-39.5, 4.45), 17.6, 3.25, facecolor=COL_ICE_ZONE, edgecolor="none", alpha=0.92, zorder=4.4))
    ax_a.text(0.04, 0.49, "$n$ = 5,327 networks (402.3B USD)", transform=ax_a.transAxes, fontsize=FS_COMPACT, va="top", ha="left", zorder=5)
    leg_a = [
        mpl.lines.Line2D([], [], color="none", marker="o", markerfacecolor=COL_SAPPHIRE, markeredgecolor="white", ms=4.8, label="Remains economic"),
        mpl.lines.Line2D([], [], color="none", marker="o", markerfacecolor=COL_CORAL, markeredgecolor="white", ms=4.8, label="Turns uneconomic\n(14.7%, 782 networks)"),
    ]
    lg_obj = ax_a.legend(handles=leg_a, loc="upper left", bbox_to_anchor=(0.02, 0.43), frameon=False, fontsize=FS_COMPACT, handletextpad=0.3, labelspacing=0.25)
    lg_obj.set_zorder(5.2)

    ax_a.set_xlim(-40, 0)
    ax_a.set_ylim(0, 15.2)
    ax_a.set_xticks([-40, -30, -20, -10, 0])
    ax_a.set_xticklabels([f"{MINUS}40", f"{MINUS}30", f"{MINUS}20", f"{MINUS}10", "0"])
    ax_a.set_yticks([0, 3, 6, 9, 12, 15])
    ax_a.set_xlabel("2046–2065 heating demand change (%)")
    ax_a.set_ylabel("Initial cost saving over building alternatives (USD GJ$^{-1}$)")

    xg = np.linspace(-40, 0, 200)
    yg = np.linspace(0, 15.2, 200)
    for mask, col, al in [(~df_lk["stranded"], COL_SAPPHIRE, 0.48), (df_lk["stranded"], COL_CORAL, 0.55)]:
        sub = df_lk[mask]
        kx = stats.gaussian_kde(np.clip(sub["dd_pct"], -39.5, -0.2), weights=sub["cap"], bw_method=0.28)(xg)
        kx = kx / kx.max()
        ax_a_top.fill_between(xg, 0, kx, facecolor=col, alpha=al, edgecolor=col, lw=0.80)

        ky = stats.gaussian_kde(np.clip(sub["m0"], 0.05, 14.9), weights=sub["cap"], bw_method=0.25)(yg)
        ky = ky / ky.max()
        ax_a_rt.fill_betweenx(yg, 0, ky, facecolor=col, alpha=al, edgecolor=col, lw=0.80)

    ax_a_top.set_ylim(0, 1.08)
    ax_a_top.axis("off")
    ax_a_rt.set_xlim(0, 1.08)
    ax_a_rt.axis("off")

    # ECDF inset with 100% pure black text legend!
    ax_cdf = ax_a.inset_axes([0.045, 0.625, 0.445, 0.335])
    ax_cdf.set_facecolor("white")
    clean_ax(ax_cdf, None)
    cdf_handles = []
    for vals, wts, col, ls, lbl_c in [
        (df_lk["lcoh0"].to_numpy(), df_lk["d0"].to_numpy(), COL_SAPPHIRE, "-", "Present (13.25)"),
        (df_lk_near["lcoh_fut_usd_per_gj"].to_numpy(), df_lk_near["sub_D_fut_gj_yr"].to_numpy(), COL_SSP2, "--", "2026–2045 (13.61)"),
        (df_lk["lcoh_fut"].to_numpy(), df_lk["dfut"].to_numpy(), COL_CORAL, "-", "2046–2065 (14.01)"),
    ]:
        ix = np.argsort(vals)
        vs = np.clip(vals[ix], 5.5, 24.0)
        ws = 100.0 * np.cumsum(wts[ix]) / wts.sum()
        ax_cdf.plot(vs, ws, color=col, ls=ls, lw=1.25)
        cdf_handles.append(mpl.lines.Line2D([], [], color=col, ls=ls, lw=1.20, label=lbl_c))
    ax_cdf.legend(handles=cdf_handles, loc="upper left", bbox_to_anchor=(-0.02, 1.04), frameon=False, fontsize=FS_COMPACT, handlelength=0.95, handletextpad=0.3, labelspacing=0.14)
    ax_cdf.set_xlim(5.5, 24.0)
    ax_cdf.set_ylim(0, 102)
    ax_cdf.set_xticks([10, 15, 20])
    ax_cdf.set_yticks([])
    ax_cdf.tick_params(axis="x", labelsize=FS_COMPACT, length=2.0)
    ax_cdf.set_xlabel("Unit cost (USD GJ$^{-1}$)", fontsize=FS_COMPACT, labelpad=1.2)
    lock_spines_above_data(ax_cdf)

    df_mid_all["str_cap"] = np.where(df_mid_all["is_loss_making_fut"], df_mid_all["total_sunk_capex_usd"], 0.0)
    cm = df_mid_all.groupby(["model", "iso3"], as_index=False).agg(tot=("total_sunk_capex_usd", "sum"), str_c=("str_cap", "sum"))
    cm["tot_b"] = cm["tot"] / 1e9
    cm["str_b"] = cm["str_c"] / 1e9
    cm["pct"] = 100.0 * cm["str_b"] / np.maximum(1e-9, cm["tot_b"])
    c_agg = cm.groupby("iso3", as_index=False).agg(
        str_m=("str_b", "mean"), str_lo=("str_b", "min"), str_hi=("str_b", "max"),
        pct_m=("pct", "mean"), pct_lo=("pct", "min"), pct_hi=("pct", "max"),
    )
    sel_isos = ["CHN", "RUS", "DEU", "POL", "KAZ", "CZE", "NLD", "SWE", "ISR", "CAN", "JPN", "ITA", "DNK"]
    c_plot = c_agg.set_index("iso3").loc[sel_isos].iloc[::-1].reset_index()
    yb = np.arange(len(c_plot))

    for axb in (ax_b1, ax_b2):
        axb.spines["right"].set_visible(False)
        axb.spines["top"].set_visible(True)
        axb.spines["top"].set_linewidth(0.55)
        axb.spines["bottom"].set_visible(True)
        lock_spines_above_data(axb)
        axb.grid(axis="x", linestyle="--", linewidth=0.40, alpha=0.28, color="#888888", zorder=0)
        axb.set_axisbelow(True)

    ax_b1.barh(yb, c_plot["str_m"], height=0.64, color=COL_CORAL, edgecolor="none", zorder=3)
    ax_b1.errorbar(c_plot["str_m"], yb, xerr=[c_plot["str_m"] - c_plot["str_lo"], c_plot["str_hi"] - c_plot["str_m"]], fmt="none", ecolor="#000000", elinewidth=0.60, capsize=1.5, zorder=4)
    for yi, r in c_plot.iterrows():
        if r["str_m"] >= 1.5:
            ax_b1.text(r["str_hi"] + 0.6, yi, f"{r['str_m']:.1f}", va="center", ha="left", fontsize=FS_COMPACT)

    ax_b2.barh(yb, c_plot["pct_m"], height=0.64, color=COL_APRICOT, edgecolor="none", zorder=3)
    ax_b2.errorbar(c_plot["pct_m"], yb, xerr=[c_plot["pct_m"] - c_plot["pct_lo"], c_plot["pct_hi"] - c_plot["pct_m"]], fmt="none", ecolor="#000000", elinewidth=0.60, capsize=1.5, zorder=4)
    for yi, r in c_plot.iterrows():
        ax_b2.text(r["pct_hi"] + 2.0, yi, f"{r['pct_m']:.0f}%", va="center", ha="left", fontsize=FS_COMPACT)

    ax_b1.axhline(5.5, ls="--", lw=0.50, color="#333333", zorder=5)
    ax_b2.axhline(5.5, ls="--", lw=0.50, color="#333333", zorder=5)
    ax_b1.set_yticks(yb)
    ax_b1.set_yticklabels([cname(i) for i in c_plot["iso3"]], fontsize=FS_COMPACT)
    plt.setp(ax_b2.get_yticklabels(), visible=False)
    ax_b2.tick_params(axis="y", length=1.8)

    ax_b1.set_xlim(0, 27.5)
    ax_b1.set_xticks([0, 10, 20])
    ax_b1.text(0.5, 1.03, "Exposed capital\n(billion USD)", transform=ax_b1.transAxes, ha="center", va="bottom", fontsize=FS_COMPACT, linespacing=1.10)

    ax_b2.set_xlim(0, 108)
    ax_b2.set_xticks([0, 40, 80])
    ax_b2.text(0.5, 1.03, "Exposed investment\nshare (%)", transform=ax_b2.transAxes, ha="center", va="bottom", fontsize=FS_COMPACT, linespacing=1.10)

    fig.text(0.980, 0.77, "Large network\nportfolios", rotation=90, va="center", ha="center", fontsize=FS_COMPACT)
    fig.text(0.980, 0.55, "Warm & electrified\nmarkets", rotation=90, va="center", ha="center", fontsize=FS_COMPACT)

    clean_ax(ax_c1, "y")
    clean_ax(ax_c2, "y")

    nb0_tot = float(df_lk["nb0"].sum() / 1e9)
    sw_near = df_r3_sum[df_r3_sum["window_id"] == "2026_2045"]
    sw_mid = df_r3_sum[df_r3_sum["window_id"] == "2046_2065"]
    sav_m = [nb0_tot, float(sw_near["net_benefit_fut_billion_usd_yr"].mean()), float(sw_mid["net_benefit_fut_billion_usd_yr"].mean())]

    ax_c1.bar([0, 1, 2], sav_m, width=0.58, color=[COL_SAPPHIRE, COL_SKY_CYAN, COL_SKY_CYAN], edgecolor="none", zorder=3)
    ax_c1.text(0, sav_m[0] + 0.45, f"{sav_m[0]:.2f}", ha="center", va="bottom", fontsize=FS_COMPACT)
    ax_c1.text(1, float(sw_near["net_benefit_fut_billion_usd_yr"].max()) + 0.45, f"{sav_m[1]:.2f}", ha="center", va="bottom", fontsize=FS_COMPACT)
    ax_c1.text(2, float(sw_mid["net_benefit_fut_billion_usd_yr"].max()) + 0.45, f"{sav_m[2]:.2f}", ha="center", va="bottom", fontsize=FS_COMPACT)
    for xi, sw in [(1, sw_near), (2, sw_mid)]:
        for _, r in sw.iterrows():
            ax_c1.scatter(xi, r["net_benefit_fut_billion_usd_yr"], facecolors="none", edgecolors="#000000", marker=GCM_MARKER[r["model"]], s=9, linewidths=0.5, zorder=4)

    ax_c1.plot([0, 0, 2, 2], [20.1, 21.3, 21.3, 18.0], color="#000000", lw=0.65)
    ax_c1.annotate("", xy=(2, 17.5), xytext=(2, 18.3), arrowprops=dict(arrowstyle="-|>,head_width=0.18,head_length=0.25", color="#000000", lw=0.65))
    ax_c1.text(1.0, 21.1, f"{MINUS}23.7%", ha="center", va="top", fontsize=FS_COMPACT)

    ax_c1.set_xlim(-0.55, 2.55)
    ax_c1.set_ylim(0, 22.8)
    ax_c1.set_xticks([0, 1, 2])
    ax_c1.set_xticklabels(["Present", "2026–\n2045", "2046–\n2065"], fontsize=FS_COMPACT)
    ax_c1.set_yticks([0, 5, 10, 15, 20])
    ax_c1.set_ylabel("Annual net cost savings\n(billion USD yr$^{-1}$)")
    lock_spines_above_data(ax_c1)

    for xi, sw in enumerate([sw_near, sw_mid]):
        fm_v = float(sw["foregone_co2_abatement_mt_yr"].mean())
        sm_v = float(sw["stranded_reversion_co2_mt_yr"].mean())
        ax_c2.bar(xi, fm_v, width=0.58, color=COL_SKY_CYAN, edgecolor="none", zorder=3)
        ax_c2.bar(xi, sm_v, bottom=fm_v, width=0.58, color=COL_CORAL_SOFT, edgecolor="none", zorder=3)
        ax_c2.plot([xi - 0.30, xi + 0.30], [fm_v, fm_v], color="white", lw=0.60, zorder=3.5)
        for _, r in sw.iterrows():
            ax_c2.scatter(xi, r["foregone_co2_abatement_mt_yr"] + r["stranded_reversion_co2_mt_yr"], facecolors="none", edgecolors="#000000", marker=GCM_MARKER[r["model"]], s=9.5, linewidths=0.5, zorder=5)
        if xi == 1:
            ax_c2.text(xi, fm_v + sm_v + 0.6, f"{fm_v + sm_v:.1f}", ha="center", va="bottom", fontsize=FS_COMPACT, fontweight="bold")
            ax_c2.text(xi, fm_v * 0.48, f"{fm_v:.1f}", ha="center", va="center", fontsize=FS_COMPACT)
            ax_c2.text(xi, fm_v + sm_v * 0.48, f"{sm_v:.1f}", ha="center", va="center", fontsize=FS_COMPACT)

    leg_c2 = [
        mpatches.Patch(facecolor=COL_SKY_CYAN, edgecolor="none", label="Lower delivery"),
        mpatches.Patch(facecolor=COL_CORAL_SOFT, edgecolor="none", label="Closure risk"),
    ]
    ax_c2.legend(handles=leg_c2, loc="upper left", bbox_to_anchor=(-0.04, 1.03), frameon=False, fontsize=FS_COMPACT, handlelength=0.85, labelspacing=0.18)

    ax_c2.set_xlim(-0.52, 1.52)
    ax_c2.set_ylim(0, 39.5)
    ax_c2.set_xticks([0, 1])
    ax_c2.set_xticklabels(["2026–\n2045", "2046–\n2065"], fontsize=FS_COMPACT)
    ax_c2.set_yticks([0, 10, 20, 30])
    ax_c2.set_ylabel("Lost & at-risk CO$_2$\n(Mt CO$_2$ yr$^{-1}$)")
    lock_spines_above_data(ax_c2)

    clean_ax(ax_d, "y")
    from code.common.parameters import (
        PEAKING_ELEC_BOILER_2025,
        PEAKING_GAS_BOILER_2025,
        PIPE_PUMP_ELEC_MWH_PER_MWH_TH,
        PIPE_VAR_OM_EUR_PER_MWH_KM,
        USD_PER_EUR_2025,
    )
    from code.common.paths import GJ_PER_MW_DAY, GJ_PER_MWH
    from code.matching.solver import TechConfig, crf, get_upgrade_params

    r0_idx = pd.read_parquet(SCEN_DIR / "R1__obs__2006_2025_selected_links.parquet").set_index(["stream_id", "zone_id"])
    rp_idx = pd.read_parquet(DATA_PROC / "sinks_2025/calibrated_region_parameters_2025.parquet").set_index("region_id")
    streams_idx = pd.read_parquet(DATA_PROC / "sources_2025/sources_2025_streams.parquet").set_index("stream_id")

    cfg_ptes = TechConfig(discount_rate=0.07, is_4gdh=False, use_2050_dh_tech=False)
    eta_ptes = 0.70
    c_ptes_cap = (crf(0.07, 35.0) + 0.01) * (172.2 * USD_PER_EUR_2025) / eta_ptes

    s_frac = np.linspace(0.0, 0.35, 36)
    s_pct = s_frac * 100.0
    curves_un, curves_all = [], []
    bench_un_vals, bench_all_vals = [], []
    rescued_counts, un_counts = [], []
    ptes_rows = []

    for fp in mid_files:
        m_id = fp.name.split("__")[1]
        df_m = pd.read_parquet(fp)
        stream_rem_gj = streams_idx["q3g_annual_gj"].astype(float).to_dict()

        n_m = len(df_m)
        h_deliv_fut = np.zeros(n_m)
        h_surp_chg_max = np.zeros(n_m)
        c_src_var = np.zeros(n_m)
        c_pk_var = np.zeros(n_m)

        for idx_r, row_m in enumerate(df_m.itertuples(index=False)):
            sid, zid = str(row_m.stream_id), str(row_m.zone_id)
            r_base = r0_idx.loc[(sid, zid)]
            up_mode = str(r_base["upgrade_mode"])
            ell_km = float(r_base["ell_km"])
            eta_tot = float(r_base["eta_tot"])
            kappa, cop, _, _, eq_vom_eur = get_upgrade_params(up_mode, cfg_ptes)

            p_el = float(rp_idx.loc[row_m.region_id, "p_elec_pretax_usd_per_mwh"])
            p_ng = float(rp_idx.loc[row_m.region_id, "p_gas_pretax_usd_per_gj"])
            has_gas = bool(rp_idx.loc[row_m.region_id, "has_gas"])

            hd = float(row_m.P_fut_gj_yr) * kappa
            h_deliv_fut[idx_r] = hd

            q_used_raw = hd / max(1e-9, kappa * eta_tot)
            q_avail_raw = max(0.0, stream_rem_gj.get(sid, 0.0) - q_used_raw)
            stream_rem_gj[sid] = q_avail_raw

            c_pipe_ann_gj = float(row_m.c_mw) * GJ_PER_MW_DAY * 365.25
            h_surp_chg_max[idx_r] = min(max(0.0, c_pipe_ann_gj - hd), q_avail_raw * kappa * eta_tot)

            mwh_src_per_gj = (1.0 / max(1e-9, eta_tot)) / GJ_PER_MWH
            hp_mwh = (mwh_src_per_gj / cop) if math.isfinite(cop) else 0.0
            pump_mwh = PIPE_PUMP_ELEC_MWH_PER_MWH_TH * mwh_src_per_gj
            vom_usd = (PIPE_VAR_OM_EUR_PER_MWH_KM * ell_km + eq_vom_eur) * mwh_src_per_gj * USD_PER_EUR_2025
            c_src_var[idx_r] = vom_usd + (hp_mwh + pump_mwh) * p_el

            if has_gas:
                sp = PEAKING_GAS_BOILER_2025
                c_pk_var[idx_r] = p_ng / sp.efficiency_lhv + (
                    sp.aux_elec_mwh_per_mwh_h * p_el + sp.var_om_non_elec_eur_per_mwh_h * USD_PER_EUR_2025
                ) / GJ_PER_MWH
            else:
                sp = PEAKING_ELEC_BOILER_2025
                c_pk_var[idx_r] = (p_el / sp.efficiency_lhv + sp.var_om_non_elec_eur_per_mwh_h * USD_PER_EUR_2025) / GJ_PER_MWH

        sub_d = np.maximum(1.0, df_m["sub_D_fut_gj_yr"].values.astype(float))
        lcoh0_fut = df_m["lcoh_fut_usd_per_gj"].values.astype(float)
        comp_lcoh = lcoh0_fut + df_m["net_benefit_fut_usd_yr"].values.astype(float) / sub_d
        un_mask = df_m["is_loss_making_fut"].values.astype(bool)

        pk_max_share = np.maximum(0.0, sub_d - h_deliv_fut) / sub_d
        surp_chg_max_share = h_surp_chg_max / sub_d
        surp_dis_max_share = eta_ptes * surp_chg_max_share

        c_u_s, c_a_s = [], []
        best_un = lcoh0_fut[un_mask].copy()
        for s_v in s_frac:
            s_disp = np.minimum(s_v, np.minimum(pk_max_share, surp_dis_max_share))
            rem_chg = np.maximum(0.0, surp_chg_max_share - s_disp / eta_ptes)
            s_sub = np.minimum(s_v - s_disp, rem_chg / (1.0 / eta_ptes - 1.0))
            nl = (
                lcoh0_fut
                + s_disp * (c_ptes_cap + c_src_var / eta_ptes - c_pk_var)
                + s_sub * (c_ptes_cap + (1.0 / eta_ptes - 1.0) * c_src_var)
            )
            if s_v > 0.0:
                best_un = np.minimum(best_un, nl[un_mask])
            c_a_s.append(float(np.average(nl, weights=sub_d)))
            c_u_s.append(float(np.average(nl[un_mask], weights=sub_d[un_mask])))

        rescued_counts.append(int(np.sum(best_un < comp_lcoh[un_mask])))
        un_counts.append(int(np.sum(un_mask)))
        bench_un_vals.append(float(np.average(comp_lcoh[un_mask], weights=sub_d[un_mask])))
        bench_all_vals.append(float(np.average(comp_lcoh, weights=sub_d)))
        curves_un.append(c_u_s)
        curves_all.append(c_a_s)

    arr_un = np.asarray(curves_un)
    arr_all = np.asarray(curves_all)
    un_mean, un_min, un_max = arr_un.mean(axis=0), arr_un.min(axis=0), arr_un.max(axis=0)
    all_mean, all_min, all_max = arr_all.mean(axis=0), arr_all.min(axis=0), arr_all.max(axis=0)
    comp_bench_un = float(np.mean(bench_un_vals))
    comp_bench_all = float(np.mean(bench_all_vals))
    unrescued_pct = 100.0 * (1.0 - float(np.sum(rescued_counts)) / float(np.sum(un_counts)))

    for idx_s, sp_val in enumerate(s_pct):
        ptes_rows.append({
            "storage_share_pct": float(round(sp_val, 2)),
            "uneconomic_lcoh_mean_usd_per_gj": float(un_mean[idx_s]),
            "uneconomic_lcoh_min_usd_per_gj": float(un_min[idx_s]),
            "uneconomic_lcoh_max_usd_per_gj": float(un_max[idx_s]),
            "uneconomic_competing_benchmark_usd_per_gj": comp_bench_un,
            "all_networks_lcoh_mean_usd_per_gj": float(all_mean[idx_s]),
            "all_networks_lcoh_min_usd_per_gj": float(all_min[idx_s]),
            "all_networks_lcoh_max_usd_per_gj": float(all_max[idx_s]),
            "all_networks_competing_benchmark_usd_per_gj": comp_bench_all,
            "uneconomic_unrescued_share_pct": unrescued_pct,
        })
    pd.DataFrame(ptes_rows).to_csv(SRC_DIR / "Figure4d_ptes_source_data.csv", index=False)

    ax_d.axhspan(comp_bench_un, 34.0, facecolor=COL_BLUSH_ZONE, zorder=0)
    ax_d.axhline(comp_bench_un, ls=":", lw=0.75, color=COL_CORAL, zorder=2)
    ax_d.axhline(comp_bench_all, ls="--", lw=0.65, color="#000000", zorder=2)

    ax_d.fill_between(s_pct, un_min, un_max, facecolor=COL_CORAL, alpha=0.17, edgecolor="none", zorder=2.6)
    ax_d.plot(s_pct, un_mean, color=COL_CORAL, lw=1.70, zorder=4.2)
    ax_d.scatter([0], [un_mean[0]], color=COL_CORAL, marker="o", s=16, zorder=5)

    ax_d.fill_between(s_pct, all_min, all_max, facecolor=COL_SAPPHIRE, alpha=0.16, edgecolor="none", zorder=2.5)
    ax_d.plot(s_pct, all_mean, color=COL_SAPPHIRE, lw=1.70, zorder=4.0)
    ax_d.scatter([0], [all_mean[0]], color=COL_SAPPHIRE, marker="D", s=16, zorder=5)

    leg_d = [
        mpl.lines.Line2D([], [], color=COL_CORAL, lw=1.70, marker="o", ms=3.6, label="Uneconomic networks (n = 782)"),
        mpl.lines.Line2D([], [], color=COL_SAPPHIRE, lw=1.70, marker="D", ms=3.4, label="All fixed networks (n = 5,327)"),
    ]
    ax_d.legend(handles=leg_d, loc="upper left", bbox_to_anchor=(-0.01, 1.02), frameon=False, fontsize=FS_COMPACT, handlelength=1.15, handletextpad=0.4, labelspacing=0.22)

    ax_d.text(1.2, 26.85, f"{un_mean[0]:.2f} USD GJ$^{{-1}}$ at 0%\n({unrescued_pct:.1f}% remain uneconomic)", fontsize=FS_COMPACT, va="bottom", ha="left", linespacing=1.12)
    ax_d.text(1.2, comp_bench_un - 0.24, f"Uneconomic benchmark ({comp_bench_un:.1f})", fontsize=FS_COMPACT, va="top", ha="left")
    ax_d.text(1.2, comp_bench_all + 0.20, f"All-network benchmark ({comp_bench_all:.2f})", fontsize=FS_COMPACT, va="bottom", ha="left")
    ax_d.text(1.3, 13.48, f"{all_mean[0]:.2f} USD GJ$^{{-1}}$", fontsize=FS_COMPACT, va="top", ha="left")
    ax_d.text(16.2, 15.65, "+24.2 USD GJ$^{-1}$ per stored GJ\n(30% loss, net of boiler offset)", fontsize=FS_COMPACT, va="top", ha="left", linespacing=1.12)

    ax_d.set_xlim(0, 35)
    ax_d.set_ylim(12, 34)
    ax_d.set_xticks([0, 5, 10, 15, 20, 25, 30, 35])
    ax_d.set_yticks([12, 16, 20, 24, 28, 32])
    ax_d.set_xlabel("Share of annual heat routed via seasonal pit storage (%)")
    ax_d.set_ylabel("Delivered heat cost\n(USD GJ$^{-1}$)")
    lock_spines_above_data(ax_d)

    panel_tag(fig, 0.015, 0.980, "a")
    panel_tag(fig, 0.592, 0.980, "b")
    panel_tag(fig, 0.015, 0.395, "c")
    panel_tag(fig, 0.540, 0.395, "d")

    df_r3_sum.to_csv(SRC_DIR / "Figure4_source_data.csv", index=False)
    return export_figure(fig, "Figure4_Pipe_Capital_Stranding_and_Carbon")


# ==============================================================================
# FIGURE 5 (Results 4 — Restored Stacked-Bar Architecture + Refreshed Palette
# + ZERO Connecting Polylines + Spine zorder=20 + 100% Pure Black Font!)
# ==============================================================================
def build_figure_5(df_glob: pd.DataFrame, df_country: pd.DataFrame) -> dict:
    configure_matplotlib_style()
    H_MM = 182.0  # Strictly <= 185.0 mm ceiling!
    fig = plt.figure(figsize=(W_IN, H_MM * MM))

    ax_a1 = fig.add_axes([0.064, 0.635, 0.330, 0.318])
    ax_a2 = fig.add_axes([0.432, 0.635, 0.152, 0.242])
    ax_b = fig.add_axes([0.758, 0.602, 0.160, 0.348])
    gs_c = gridspec.GridSpec(3, 5, figure=fig, left=0.064, right=0.982, bottom=0.058, top=0.484, wspace=0.35, hspace=0.58)

    def glob_subsectors(scen_filter: pd.Series) -> tuple[dict[str, float], pd.Series]:
        sub = df_country[scen_filter].copy()
        sub["chem_ww"] = sub["P_ethylene_gj"] + sub["P_pulp_paper_gj"] + sub["P_wastewater_gj"]
        by_m = sub.groupby("model")[["P_power_gj", "P_steel_gj", "P_refining_gj", "P_cement_gj", "chem_ww", "P_gj"]].sum() / 1e9
        return by_m.mean().to_dict(), by_m["P_gj"]

    steps_a = [
        ("Present", df_country["scenario_id"] == "R1__obs__2006_2025"),
        ("2046–2065\nwarming", (df_country["group"] == "R1") & (df_country["ssp"] == "ssp245") & (df_country["window_id"] == "2046_2065")),
        ("55/25 °C\n(vs 2025)", (df_country["group"] == "R4") & (df_country["subgroup"] == "S3_4gdh_standalone") & (df_country["window_id"] == "2046_2065")),
        ("55/25 °C\n(vs 2050)", (df_country["group"] == "R4") & (df_country["subgroup"] == "S4_4gdh_symmetric") & (df_country["window_id"] == "2046_2065")),
        ("Envelope\nretrofits", (df_country["group"] == "R4") & (df_country["subgroup"] == "S2_envelope_retrofit") & (df_country["window_id"] == "2046_2065")),
        ("Fossil\nretirement", (df_country["group"] == "R4") & (df_country["subgroup"] == "S1_fossil_retire") & (df_country["window_id"] == "2046_2065")),
        ("Combined\n2050", (df_country["group"] == "R4") & (df_country["subgroup"] == "S5_combined_all") & (df_country["window_id"] == "2046_2065")),
    ]

    sec_layers = [
        ("P_power_gj", "Power plants / Nuclear & bioenergy", F5_POWER),
        ("P_steel_gj", "Iron & steel", F5_STEEL),
        ("P_refining_gj", "Oil refining", F5_REFINING),
        ("P_cement_gj", "Cement", F5_CEMENT),
        ("chem_ww", "Chemicals, pulp & wastewater", F5_CHEM),
    ]

    clean_ax(ax_a1, "y")
    bw = 0.60
    src_f5 = []
    for xi, (slbl, sfilt) in enumerate(steps_a):
        means, mod_tot = glob_subsectors(sfilt)
        bot = 0.0
        for col_k, _, hex_k in sec_layers:
            val = float(means[col_k])
            if val > 0:
                ax_a1.bar(xi, val, bottom=bot, width=bw, color=hex_k, edgecolor="none", zorder=3)
                if bot > 1e-4:
                    ax_a1.plot([xi - 0.5 * bw, xi + 0.5 * bw], [bot, bot], color="white", lw=0.45, zorder=3.5)
                bot += val
        if len(mod_tot) > 1:
            for m_id, tv in mod_tot.items():
                ax_a1.scatter(xi, tv, facecolors="none", edgecolors="#000000", marker=GCM_MARKER[str(m_id)], s=8.5, linewidths=0.48, zorder=5)
        y_lbl = max(bot, float(mod_tot.max())) + 0.09
        if xi == 5:
            ax_a1.text(xi, y_lbl, f"0.32\n({MINUS}88.1%)", ha="center", va="bottom", fontsize=FS_COMPACT, fontweight="bold", linespacing=1.05, zorder=6)
        elif xi == 6:
            ax_a1.text(xi, 0.50, f"{bot:.2f}", ha="center", va="bottom", fontsize=FS_COMPACT, fontweight="bold")
        else:
            ax_a1.text(xi, y_lbl, f"{bot:.2f}", ha="center", va="bottom", fontsize=FS_COMPACT, fontweight="bold")
        src_f5.append({"step": slbl.replace("\n", " "), "P_mean_ej": bot, "P_min_ej": float(mod_tot.min()), "P_max_ej": float(mod_tot.max())})

    ZOOM_F5 = 0.45
    zx0, zx1 = 6 - 0.40, 6 + 0.40
    ax_a1.add_patch(mpatches.Rectangle((zx0, 0), zx1 - zx0, ZOOM_F5, fill=False, edgecolor="#000000", lw=0.55, zorder=4))

    ax_a1.set_xlim(-0.55, 6.55)
    ax_a1.set_ylim(0, 3.74)
    ax_a1.set_xticks(range(7))
    ax_a1.set_xticklabels([s[0] for s in steps_a], fontsize=FS_COMPACT, rotation=30, ha="right", linespacing=0.92)
    ax_a1.set_ylabel("Economic potential (EJ yr$^{-1}$)")
    lock_spines_above_data(ax_a1)

    leg_f5 = [mpatches.Patch(facecolor=hx, edgecolor="none", label=lb) for _, lb, hx in sec_layers]
    fig.legend(handles=leg_f5, loc="upper left", bbox_to_anchor=(0.328, 0.980), frameon=False, fontsize=FS_COMPACT, handlelength=0.95, handleheight=0.80, labelspacing=0.18)

    ax_a2.spines["top"].set_visible(True)
    ax_a2.spines["top"].set_linewidth(0.50)
    ax_a2.spines["left"].set_visible(True)
    ax_a2.spines["left"].set_linewidth(0.50)
    ax_a2.yaxis.tick_right()
    ax_a2.yaxis.set_label_position("right")
    ax_a2.grid(axis="y", linestyle="--", linewidth=0.40, alpha=0.28, color="#888888", zorder=0)
    ax_a2.set_axisbelow(True)

    s5_sub = df_country[(df_country["group"] == "R4") & (df_country["subgroup"] == "S5_combined_all") & (df_country["window_id"] == "2046_2065")].copy()
    s5_sub["chem_ww"] = s5_sub["P_ethylene_gj"] + s5_sub["P_pulp_paper_gj"] + s5_sub["P_wastewater_gj"]
    s5_mod = s5_sub.groupby("model")[["P_steel_gj", "P_power_gj", "P_refining_gj", "P_cement_gj", "chem_ww"]].sum() / 1e6
    z_specs = [
        ("Steel", "P_steel_gj", F5_STEEL),
        ("Nuclear", "P_power_gj", F5_POWER),
        ("Refining", "P_refining_gj", F5_REFINING),
        ("Cement", "P_cement_gj", F5_CEMENT),
        ("Chem.", "chem_ww", F5_CHEM),
    ]
    tot_s5 = float(s5_mod.sum(axis=1).mean())
    for xi2, (zlbl, zcol, zhex) in enumerate(z_specs):
        vals = s5_mod[zcol].to_numpy()
        mv = float(vals.mean())
        ax_a2.bar(xi2, mv, width=0.58, color=zhex, edgecolor="none", zorder=3)
        for m_id, vm in s5_mod[zcol].items():
            ax_a2.scatter(xi2, vm, facecolors="none", edgecolors="#000000", marker=GCM_MARKER[str(m_id)], s=8, linewidths=0.45, zorder=4)
        y_off = 8.5 if xi2 == 1 else (1.8 if xi2 == 2 else (9.5 if xi2 == 3 else 2.8))
        ax_a2.text(xi2, max(mv, float(vals.max())) + y_off, f"{100*mv/tot_s5:.1f}%", ha="center", va="bottom", fontsize=FS_COMPACT)

    ax_a2.text(0.58, 0.95, f"{tot_s5:.1f} PJ\nfossil-free\ncore", transform=ax_a2.transAxes, ha="center", va="top", fontsize=FS_COMPACT, linespacing=1.10)
    ax_a2.set_xlim(-0.72, 4.65)
    ax_a2.set_ylim(0, 125)
    ax_a2.set_xticks(range(5))
    ax_a2.set_xticklabels([z[0] for z in z_specs], fontsize=FS_COMPACT, rotation=28, ha="right", linespacing=0.92)
    ax_a2.set_yticks([0, 40, 80, 120])
    ax_a2.set_ylabel("Combined 2050 potential (PJ yr$^{-1}$)")
    lock_spines_above_data(ax_a2)

    for yl, yr in [(ZOOM_F5, 125.0), (0.0, 0.0)]:
        fig.add_artist(ConnectionPatch(
            xyA=(zx1, yl), coordsA=ax_a1.transData,
            xyB=(-0.72, yr), coordsB=ax_a2.transData,
            linestyle="--", linewidth=0.50, color="#333333"
        ))

    # --- Panel b: Shaded Range-Band Sensitivity Forest Plot + Right Tabular P0 Column ---
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["left"].set_visible(False)
    ax_b.spines["right"].set_visible(False)
    lock_spines_above_data(ax_b)
    ax_b.tick_params(axis="y", length=0)
    ax_b.grid(axis="x", linestyle="--", linewidth=0.40, alpha=0.28, color="#888888", zorder=0)
    ax_b.set_axisbelow(True)

    base = df_glob[df_glob["scenario_id"] == "R1__obs__2006_2025"].iloc[0]
    clim = df_glob[(df_glob["group"] == "R1") & (df_glob["ssp"] == "ssp245") & (df_glob["window_id"] == "2046_2065")]
    p0_v = float(base["P_ej_yr"])

    rob_specs = [
        ("Central run", "CENTRAL"),
        ("Threshold 12 °C", "Rob1_bait12c"),
        ("Threshold 16 °C", "Rob1_bait16c"),
        ("Discount rate 3%", "Rob2_r03pct"),
        ("Discount rate 10%", "Rob2_r10pct"),
        ("Retail + tax", "Rob3_retail_tax"),
        ("Prices \u221230%", "Rob3_price_minus30"),
        ("Prices +30%", "Rob3_price_plus30"),
        ("GHSL 2020 map", "Rob4_ghsl2020"),
        ("Survey subset", "OFFICIAL_ONLY"),
    ][::-1]

    yb_r = np.arange(len(rob_specs))
    ax_b.axvline(0.0, lw=0.65, color="#000000", zorder=18)
    ax_b.axvline(-11.29, ls="--", lw=0.60, color="#000000", zorder=2)
    ax_b.text(-11.29, len(rob_specs) - 0.35, f"{MINUS}11.3%", ha="center", va="bottom", fontsize=FS_COMPACT)
    ax_b.text(8.5, len(rob_specs) - 0.35, "$P_0$\n(EJ yr$^{-1}$)", ha="right", va="bottom", fontsize=FS_COMPACT, linespacing=1.05)

    for yi, (rlbl, rcode) in enumerate(rob_specs):
        if rcode == "CENTRAL":
            bv, sub, col_p = p0_v, clim, "P_ej_yr"
        elif rcode == "OFFICIAL_ONLY":
            bv, sub, col_p = float(base["P_official_only_ej_yr"]), clim, "P_official_only_ej_yr"
        else:
            sr = df_glob[(df_glob["group"] == "Robustness") & (df_glob["subgroup"] == rcode)]
            bv = float(sr.loc[sr["window_id"] == "2006_2025", "P_ej_yr"].iloc[0])
            sub = sr[sr["window_id"] == "2046_2065"]
            col_p = "P_ej_yr"
        dps = [100.0 * (float(r[col_p]) - bv) / bv for _, r in sub.iterrows()]
        lo, hi, mv = float(min(dps)), float(max(dps)), float(np.mean(dps))
        ax_b.barh(yi, hi - lo, left=lo, height=0.60, color=F5_BAND, alpha=0.84, edgecolor="none", zorder=1)
        ax_b.plot([mv, mv], [yi - 0.28, yi + 0.28], color="#000000", lw=1.25, zorder=4)
        for (_, r), dpv in zip(sub.iterrows(), dps):
            ax_b.scatter(dpv, yi, facecolors="none", edgecolors="#000000", marker=GCM_MARKER[r["model"]], s=8.5, linewidths=0.48, zorder=3)
        ax_b.text(8.5, yi, f"{bv:.2f}", ha="right", va="center", fontsize=FS_COMPACT)

    ax_b.set_yticks(yb_r)
    ax_b.set_yticklabels([x[0] for x in rob_specs], fontsize=FS_COMPACT)
    ax_b.set_xlim(-34, 9.5)
    ax_b.set_ylim(-0.6, len(rob_specs) - 0.3)
    ax_b.set_xticks([-30, -20, -10, 0])
    ax_b.set_xticklabels([f"{MINUS}30", f"{MINUS}20", f"{MINUS}10", "0"])
    ax_b.set_xlabel("2046–2065 change (%)")

    # --- Panel c: Monumental Full-Width 5x3 = 15-Country Stacked-Bar Matrix (ZERO polylines, pure black headers!) ---
    isos_15 = ["CHN", "RUS", "DEU", "KAZ", "KOR", "POL", "ITA", "NLD", "BLR", "GBR", "CZE", "SWE", "DNK", "AUT", "FIN"]
    for idx, iso in enumerate(isos_15):
        r_i, c_i = divmod(idx, 5)
        ax_m = fig.add_subplot(gs_c[r_i, c_i])
        clean_ax(ax_m, "y")
        c_df = df_country[df_country["iso3"] == iso].copy()
        c_df["chem_ww"] = c_df["P_ethylene_gj"] + c_df["P_pulp_paper_gj"] + c_df["P_wastewater_gj"]

        st_filters = [
            c_df["scenario_id"] == "R1__obs__2006_2025",
            (c_df["group"] == "R1") & (c_df["ssp"] == "ssp245") & (c_df["window_id"] == "2046_2065"),
            (c_df["group"] == "R4") & (c_df["subgroup"] == "S3_4gdh_standalone") & (c_df["window_id"] == "2046_2065"),
            (c_df["group"] == "R4") & (c_df["subgroup"] == "S4_4gdh_symmetric") & (c_df["window_id"] == "2046_2065"),
            (c_df["group"] == "R4") & (c_df["subgroup"] == "S5_combined_all") & (c_df["window_id"] == "2046_2065"),
        ]
        xs = np.arange(5)
        tots_m, los_m, his_m = [], [], []
        bw_m = 0.62
        for xi_m, sf in enumerate(st_filters):
            sub_m = c_df[sf]
            bot_m = 0.0
            for col_k, _, hex_k in sec_layers:
                vm = float(sub_m[col_k].mean()) / 1e6
                if vm > 0:
                    ax_m.bar(xi_m, vm, bottom=bot_m, width=bw_m, color=hex_k, edgecolor="none", zorder=3)
                    if bot_m > 1e-4:
                        ax_m.plot([xi_m - 0.5 * bw_m, xi_m + 0.5 * bw_m], [bot_m, bot_m], color="white", lw=0.35, zorder=3.5)
                    bot_m += vm
            p_arr = sub_m["P_gj"].to_numpy() / 1e6
            tots_m.append(bot_m)
            los_m.append(float(p_arr.min()))
            his_m.append(float(p_arr.max()))

        # NO connecting polyline! Only 5-GCM min-max error bars on projected states 1..4
        for xi_m in range(1, 5):
            ax_m.errorbar(xi_m, tots_m[xi_m], yerr=[[tots_m[xi_m] - los_m[xi_m]], [his_m[xi_m] - tots_m[xi_m]]], fmt="none", ecolor="#000000", elinewidth=0.55, capsize=1.4, zorder=5)

        peak_m = max(his_m)
        step_candidates = [4, 5, 8, 10, 12, 15, 20, 25, 30, 40, 50, 100, 200, 400, 600, 800]
        t_step = next((s for s in step_candidates if 2 * s >= peak_m * 0.90), 800)
        ymax_m = max(peak_m * 1.24, 2 * t_step * 1.22)
        ax_m.set_ylim(0, ymax_m)
        ax_m.set_yticks([0, t_step, 2 * t_step])

        short_c = "UK" if iso == "GBR" else cname(iso)
        ax_m.text(0.02, 1.04, short_c, transform=ax_m.transAxes, fontsize=FS_COMPACT, fontweight="bold", color="#000000", va="bottom", ha="left")
        ax_m.text(0.98, 1.04, f"{tots_m[-1]:.1f}", transform=ax_m.transAxes, fontsize=FS_COMPACT, color="#000000", va="bottom", ha="right")

        ax_m.set_xlim(-0.55, 4.55)
        ax_m.set_xticks(xs)
        if r_i == 2:
            ax_m.set_xticklabels(["2025", "Warm", "55°C(25)", "55°C(50)", "Zero-\nfossil"], fontsize=FS_COMPACT, rotation=34, ha="right", linespacing=0.92)
        else:
            ax_m.set_xticklabels([])
        ax_m.tick_params(axis="y", labelsize=FS_COMPACT, length=2.0)
        if c_i == 0 and r_i == 1:
            ax_m.set_ylabel("Economic potential (PJ yr$^{-1}$)")
        lock_spines_above_data(ax_m)

    panel_tag(fig, 0.015, 0.982, "a")
    panel_tag(fig, 0.652, 0.982, "b")
    panel_tag(fig, 0.015, 0.538, "c")

    pd.DataFrame(src_f5).to_csv(SRC_DIR / "Figure5_source_data.csv", index=False)
    return export_figure(fig, "Figure5_System_Transitions_and_Sensitivity")


def main() -> None:
    print("=== Palette Contrast Summary ===")
    check_palette_contrast(SEMANTIC_PALETTE, name="Main Figures 2–5 Unified Palette")

    df_glob = pd.read_parquet(SCEN_DIR / "global_scenarios_summary.parquet")
    df_country = pd.read_parquet(SCEN_DIR / "all_scenarios_by_country.parquet")
    df_streams = pd.read_parquet(DATA_PROC / "sources_2025/sources_2025_streams.parquet")

    audits = []
    audits.append(build_figure_2(df_glob, df_streams))
    audits.append(build_figure_3(df_country))
    audits.append(build_figure_4())
    audits.append(build_figure_5(df_glob, df_country))

    print("\n=== MAIN FIGURES 2–5 EXPORT SUMMARY ===")
    for a in audits:
        print(f"  {a['stem']}: {a['w_mm']:.2f}x{a['h_mm']:.2f} mm | {a['mb']:.2f} MB | {a['n_texts']} text objects | bbox touches = {len(a['overlaps'])}")


if __name__ == "__main__":
    main()
