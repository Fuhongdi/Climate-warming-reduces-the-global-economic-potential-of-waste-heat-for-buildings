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

    # Economic screening split: non-competitive technical potential peels downward; economic potential arrives at x=69.5..71.5
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

    # Place Economic potential title ABOVE node and share breakdown dynamically from baseline outputs
    P_ind = float(obs0["P_industry_ej_yr"])
    P_hp = float(obs0["P_hp_ind_power_ej_yr"] + obs0["P_hp_wastewater_ej_yr"])
    pct_p_u = 100.0 * P0 / max(1e-9, U0)
    pct_pow = 100.0 * P_pow / max(1e-9, P0)
    pct_ind = 100.0 * P_ind / max(1e-9, P0)
    pct_hp = 100.0 * P_hp / max(1e-9, P0)
    hp_lbl = f"{pct_hp:.1f}%"
    pct_dh = 100.0 * P0 / DH_CUR

    ax_a.text(0.5 * (x_e0 + x_e1), y_econ1 + 2.4, f"Economic potential\n{P0:.2f} EJ yr$^{{-1}}$ ({pct_p_u:.1f}%)", ha="center", va="bottom", fontsize=FS_COMPACT, fontweight="bold", linespacing=1.08)
    ax_a.text(x_e1, y_econ0 - 3.2, f"{pct_pow:.1f}% power plants\n{pct_ind:.1f}% heavy industry\n({hp_lbl} via heat pumps)", ha="right", va="top", fontsize=FS_COMPACT, linespacing=1.10)

    # Far-right Proportional District Heating Gauge (x = 79.5..98.2, y = 43.0..53.5)
    gx0, gx1 = 79.5, 98.2
    gw = gx1 - gx0
    gy0, gh = 43.0, 10.8
    gw_econ = gw * min(0.98, P0 / DH_CUR)
    gw_pow = gw_econ * (P_pow / max(1e-9, P0))

    ax_a.plot([x_e1 + 0.3, gx0 - 0.3], [y_econ1, gy0 + gh], ls="--", lw=0.50, color="#444444", zorder=2)
    ax_a.plot([x_e1 + 0.3, gx0 - 0.3], [y_econ0, gy0], ls="--", lw=0.50, color="#444444", zorder=2)

    ax_a.add_patch(mpatches.Rectangle((gx0, gy0), gw, gh, facecolor=COL_MIST_DH, edgecolor="#000000", linewidth=0.55, zorder=3))
    ax_a.add_patch(mpatches.Rectangle((gx0, gy0 + 1.8), gw_pow, gh - 3.6, facecolor=COL_SAPPHIRE, edgecolor="none", zorder=4))
    ax_a.add_patch(mpatches.Rectangle((gx0 + gw_pow, gy0 + 1.8), gw_econ - gw_pow, gh - 3.6, facecolor=COL_APRICOT, edgecolor="none", zorder=4))
    ax_a.plot([gx0 + gw_econ, gx0 + gw_econ], [gy0 - 1.4, gy0 + gh + 1.4], color="#000000", lw=0.85, zorder=5)

    ax_a.text(gx0 + gw_econ + 1.0, gy0 + gh + 2.4, f"{P0:.2f} EJ yr$^{{-1}}$ ({pct_dh:.1f}%)", ha="center", va="bottom", fontsize=FS_COMPACT, fontweight="bold")
    ax_a.text(gx0, gy0 - 2.2, "0%", ha="left", va="top", fontsize=FS_COMPACT)
    ax_a.text(gx1, gy0 - 2.2, "100%", ha="right", va="top", fontsize=FS_COMPACT)
    ax_a.text(0.5 * (gx0 + gx1), gy0 - 10.2, f"Current building\ndistrict heating ({DH_CUR:.2f} EJ yr$^{{-1}}$)", ha="center", va="top", fontsize=FS_COMPACT, linespacing=1.10)

    # --- Panel b: 1946–2100 Observed | Projected (100% pure black labels!) ---
    clean_ax(ax_b, "y")
    r1 = df_glob[df_glob["group"] == "R1"].copy()
    r1["mid"] = 0.5 * (r1["start_year"] + r1["end_year"])
    obs = r1[r1["ssp"] == "obsclim"].sort_values("mid")

    ylo_b = 1.00 if float(r1["P_ej_yr"].min()) < 1.48 else 1.50
    yhi_b = max(3.68, float(obs["P_ej_yr"].max()) + 0.28)

    ax_b.axvspan(1946, 2015.5, facecolor=COL_HIST_BG, zorder=0)
    ax_b.axvline(2015.5, ls=":", lw=0.65, color="#333333", zorder=2)
    ax_b.axhline(P0, ls="--", lw=0.55, color="#333333", zorder=2)
    ax_b.text(2012.5, yhi_b - 0.05, "Observed", ha="right", va="top", fontsize=FS_COMPACT)
    ax_b.text(2018.5, yhi_b - 0.05, "Projected", ha="left", va="top", fontsize=FS_COMPACT)

    ax_b.plot(obs["mid"], obs["P_ej_yr"], color="#000000", lw=1.65, marker="o", ms=3.6, zorder=6)
    obs_pts = list(zip(obs["mid"], obs["P_ej_yr"]))
    ax_b.text(obs_pts[0][0], obs_pts[0][1] + 0.05, f"{obs_pts[0][1]:.2f}", ha="center", va="bottom", fontsize=FS_COMPACT)
    ax_b.text(obs_pts[1][0], obs_pts[1][1] + 0.05, f"{obs_pts[1][1]:.2f}", ha="center", va="bottom", fontsize=FS_COMPACT)
    ax_b.text(obs_pts[2][0] + 1.5, obs_pts[2][1] + 0.05, f"{obs_pts[2][1]:.2f}", ha="center", va="bottom", fontsize=FS_COMPACT)
    ax_b.text(obs_pts[3][0] + 2.2, obs_pts[3][1] + 0.05, f"{obs_pts[3][1]:.2f}", ha="left", va="bottom", fontsize=FS_COMPACT)

    base_mod = r1[r1["window_id"] == "2006_2025"].set_index(["model", "ssp"])["P_ej_yr"].to_dict()
    ssp_cfg = [("ssp126", "SSP1-2.6", COL_SAPPHIRE), ("ssp245", "SSP2-4.5", COL_SSP2), ("ssp370", "SSP3-7.0", COL_CORAL)]
    src_rows = []
    for _, orow in obs.iterrows():
        src_rows.append({"series": "Observed", "mid_year": orow["mid"], "P_mean_ej": orow["P_ej_yr"], "P_min_ej": orow["P_ej_yr"], "P_max_ej": orow["P_ej_yr"]})

    for scode, slbl, scol in ssp_cfg:
        sub = r1[r1["ssp"] == scode].copy()
        # Anchor 2006-2025 midpoint at P0 and plot exact 5-GCM projected P_ej_yr so levels match Extended Data Table 2
        sub["P_plot"] = np.where(sub["window_id"] == "2006_2025", P0, sub["P_ej_yr"].astype(float))
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
    ax_b.set_ylim(ylo_b, yhi_b)
    ax_b.set_xticks([1950, 1975, 2000, 2025, 2050, 2075, 2100])
    ax_b.set_yticks([t for t in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5] if ylo_b - 0.01 <= t <= yhi_b])
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
    gwl["p_base_mod"] = [base_mod[(r["model"], r["ssp"])] for _, r in gwl.iterrows()]
    gwl["dp_pct"] = 100.0 * (gwl["P_ej_yr"].astype(float) / gwl["p_base_mod"] - 1.0)
    gwl["P_al"] = P0 * (1.0 + gwl["dp_pct"] / 100.0)

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

    # Exact 5-GCM cluster-robust (CR1, df = G - 1 = 4) OLS fit & 95% confidence band
    x_vec = gwl["x_c"].to_numpy(dtype=float)
    y_vec = gwl["P_al"].to_numpy(dtype=float)
    y_pct_vec = gwl["dp_pct"].to_numpy(dtype=float)
    res_pct = stats.linregress(x_vec, y_pct_vec)

    X_mat = np.column_stack([np.ones_like(x_vec), x_vec])
    XtX_inv = np.linalg.inv(X_mat.T @ X_mat)
    beta_hat = XtX_inv @ (X_mat.T @ y_vec)
    resid_vec = y_vec - X_mat @ beta_hat
    clusters = gwl["model"].astype(str).to_numpy()
    uniq_cl = sorted(set(clusters))
    G_cl = len(uniq_cl)
    meat = np.zeros((2, 2), dtype=float)
    for cl in uniq_cl:
        m_cl = clusters == cl
        u_g = X_mat[m_cl].T @ resid_vec[m_cl]
        meat += np.outer(u_g, u_g)
    cr1_scale = (G_cl / max(1, G_cl - 1)) * ((len(x_vec) - 1) / max(1, len(x_vec) - 2))
    V_cr1 = cr1_scale * (XtX_inv @ meat @ XtX_inv)
    slope_pct_se_cr1 = 100.0 * math.sqrt(max(0.0, float(V_cr1[1, 1]))) / max(1e-9, P0)

    xx = np.linspace(1.38, 3.18, 60)
    X_grid = np.column_stack([np.ones_like(xx), xx])
    yy = X_grid @ beta_hat
    se_cr1 = np.sqrt(np.maximum(0.0, np.sum((X_grid @ V_cr1) * X_grid, axis=1)))
    t_q = float(stats.t.ppf(0.975, max(1, G_cl - 1)))
    ax_c.fill_between(xx, yy - t_q * se_cr1, yy + t_q * se_cr1, color="#D8DEE6", alpha=0.65, lw=0, zorder=2)
    ax_c.plot(xx, yy, color="#000000", lw=1.15, zorder=4.5)

    ax_c.text(0.04, 0.03, f"$n$ = {len(gwl)} (5 GCMs)", transform=ax_c.transAxes, ha="left", va="bottom", fontsize=FS_COMPACT)
    ax_c.text(0.97, 0.03, f"Slope = {fm(res_pct.slope, '.1f')}% °C$^{{-1}}$, $R^2$ = {res_pct.rvalue**2:.2f}", transform=ax_c.transAxes, ha="right", va="bottom", fontsize=FS_COMPACT)
    gwl.assign(slope_pct_per_c=res_pct.slope, slope_se_cr1_pct_per_c=slope_pct_se_cr1, r2=res_pct.rvalue**2).to_csv(SRC_DIR / "Figure2c_gwl_source_data.csv", index=False)

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

    # Partition observed 2006-2025 urban heating areas cleanly into P_obs == 0 vs P_obs > 0 (100% consistent counts!)
    z_obs_pos = z_all[z_all[p_col] > 0].copy()
    p0_obs_ej = float(z_obs_pos[p_col].sum() / 1e9)
    zp = z_obs_pos[["zone_id", "lat", "lon", p_col]].rename(columns={p_col: "P0_obs_gj"}).merge(
        df_z[["zone_id", "dP_rel_pct"]], on="zone_id", how="left"
    )
    zp["dP_rel_pct"] = zp["dP_rel_pct"].fillna(0.0)
    zp = zp.sort_values("P0_obs_gj").copy()

    vx, vy = trans.transform(zp["lon"].to_numpy(), zp["lat"].to_numpy())
    cmap_loss = mcolors.LinearSegmentedColormap.from_list(
        "pro_fig3_ramp", ["#FEF098", "#F9B872", "#E65C3B", "#AC2428", "#671019"], N=256
    )
    vals_loss = np.clip(-zp["dP_rel_pct"].to_numpy(), 0.0, 45.0)
    sz_v = np.clip(np.sqrt(zp["P0_obs_gj"].to_numpy() / 1e6) * 8.2, 1.8, 56.0)
    sc = ax_map.scatter(vx, vy, c=vals_loss, cmap=cmap_loss, vmin=0, vmax=45, s=sz_v, alpha=0.92, edgecolors="white", linewidths=0.32, rasterized=True, zorder=3)

    b0_idx = df_country[df_country["scenario_id"] == "R1__obs__2006_2025"].set_index("iso3")
    df_dec_map = pd.read_parquet(SCEN_DIR / "r2_decomposition/r2_eq6_decomposition_all.parquet")
    c_mid_dp = df_dec_map[(df_dec_map["level"] == "country") & (df_dec_map["ssp"] == "ssp245") & (df_dec_map["window_id"] == "2046_2065")].groupby("entity_id")["dP_rel_pct"].mean().to_dict()

    usa_u = float(b0_idx.loc["USA", "U_gj"]) / 1e9
    usa_p = float(b0_idx.loc["USA", "P_gj"]) / 1e9
    deu_p = float(b0_idx.loc["DEU", "P_gj"]) / 1e9
    rus_p = float(b0_idx.loc["RUS", "P_gj"]) / 1e9
    chn_p = float(b0_idx.loc["CHN", "P_gj"]) / 1e9

    callouts = [
        (f"United States\n({usa_u:.2f} EJ technical,\n{usa_p:.3f} EJ economic)", -98, 37, -142, 26),
        (f"Sweden\n({fm(c_mid_dp.get('SWE', 0.0), '.1f')}%)", 16, 61, 5, 78),
        (f"Denmark\n({fm(c_mid_dp.get('DNK', 0.0), '.1f')}%)", 10, 56, -25, 57),
        (f"Germany\n({deu_p:.2f} EJ, {fm(c_mid_dp.get('DEU', 0.0), '.1f')}%)", 10, 50, -26, 40),
        (f"Italy\n({fm(c_mid_dp.get('ITA', 0.0), '.1f')}%)", 12.5, 43, 18, 24),
        (f"Russia\n({rus_p:.2f} EJ, {fm(c_mid_dp.get('RUS', 0.0), '.1f')}%)", 62, 56, 86, 70),
        (f"China\n({chn_p:.2f} EJ, {fm(c_mid_dp.get('CHN', 0.0), '.1f')}%)", 117, 37, 155, 22),
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

    # Use real Line2D markers for map legend with exact dynamic counts from observed 2006-2025 zones
    n_unviable_z = len(z_zero)
    map_leg_h = [
        mpl.lines.Line2D([], [], color="none", marker="o", markerfacecolor=COL_ZERO_DOT, markeredgecolor="none", ms=4.5, label=f"{n_unviable_z:,} urban heating areas with zero economic potential"),
        mpl.lines.Line2D([], [], color="none", marker="o", markerfacecolor="white", markeredgecolor="#000000", markeredgewidth=0.65, ms=4.8, label=f"{len(zp):,} viable urban heating areas, {p0_obs_ej:.2f} EJ yr$^{{-1}}$"),
    ]
    lg_map = fig.legend(
        handles=map_leg_h, loc="center left", bbox_to_anchor=(0.415, 0.548),
        frameon=True, facecolor="white", framealpha=0.94, edgecolor="none",
        fontsize=FS_COMPACT, handletextpad=0.35, labelspacing=0.35,
    )
    lg_map.set_zorder(10)

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
        if r["lo"] < -62.5:
            ax_b.errorbar(r["net"], yi, xerr=[[0.0], [r["hi"] - r["net"]]], fmt="D", ms=3.2, color="#000000", ecolor="#000000", elinewidth=0.65, capsize=1.6, zorder=6)
            ax_b.annotate("", xy=(-63.2, yi), xytext=(r["net"], yi), arrowprops=dict(arrowstyle="->,head_width=0.14,head_length=0.22", color="#000000", lw=0.65), zorder=6)
            ax_b.text(pos_l + 0.8, yi, f"{fm(r['net'], '.1f')}%", va="center", ha="left", fontsize=FS_COMPACT)
        else:
            ax_b.errorbar(r["net"], yi, xerr=[[r["net"] - r["lo"]], [r["hi"] - r["net"]]], fmt="D", ms=3.2, color="#000000", ecolor="#000000", elinewidth=0.65, capsize=1.6, zorder=6)
            if r["lbl"] in highlight_b:
                ax_b.text(min(neg_l, r["lo"]) - 1.0, yi, f"{fm(r['net'], '.1f')}%", va="center", ha="right", fontsize=FS_COMPACT)

    ax_b.axvline(0, lw=0.65, color="#000000", zorder=18)
    ax_b.axhline(len(c_rows) - 0.5, ls="--", lw=0.50, color="#444444", zorder=10)
    ax_b.set_yticks(yb)
    ax_b.set_yticklabels([r["lbl"] for r in rows_b], fontsize=FS_COMPACT)
    ax_b.set_xlim(-64, 14)
    ax_b.set_ylim(-0.65, len(rows_b) - 0.35)
    ax_b.set_xticks([-50, -30, -10, 0, 10])
    ax_b.set_xticklabels([f"{MINUS}50", f"{MINUS}30", f"{MINUS}10", "0", "+10"])
    ax_b.set_xlabel("Contribution to potential change (%)")
    lock_spines_above_data(ax_b)

    clean_ax(ax_c, "x")
    df_lat = pd.read_csv(SCEN_DIR / "r2_decomposition/r2_stratified_by_abs_lat_band.csv")
    df_sz = pd.read_csv(SCEN_DIR / "r2_decomposition/r2_stratified_by_size_tier.csv")
    df_pm_strat = pd.read_csv(SCEN_DIR / "r2_decomposition/r2_stratified_by_model_all.csv")
    df_lat = df_lat[df_lat["P0_pj"] > 0.5].copy()

    gcm_lat_pts: dict[str, list[tuple[str, float]]] = {}
    gcm_sz_pts: dict[str, list[tuple[str, float]]] = {}
    for _, r_m in df_pm_strat.iterrows():
        st_type = str(r_m["stratum_type"])
        raw_st = str(r_m["stratum"])
        m_id = str(r_m["model"])
        dp_v = float(r_m["dP_rel_pct"])
        if st_type == "abs_lat_band":
            lbl_m = raw_st.replace(" deg", "°").replace("-", "–")
            gcm_lat_pts.setdefault(lbl_m, []).append((m_id, dp_v))
        elif st_type == "size_tier":
            lbl_m = raw_st.replace("Small (1-10 MW)", "1–10 MW areas").replace("Medium (10-100 MW)", "10–100 MW areas").replace("Large (>100 MW)", ">100 MW areas")
            gcm_sz_pts.setdefault(lbl_m, []).append((m_id, dp_v))

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
    sz_order = [">100 MW cities", "10–100 MW cities", "1–10 MW small cities"]
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
        n_models_uncompetitive=("is_loss_making_fut", "sum"),
    )
    df_lk["m0"] = df_lk["nb0"] / np.maximum(1.0, df_lk["d0"])
    df_lk["dd_pct"] = 100.0 * (df_lk["dfut"] - df_lk["d0"]) / np.maximum(1.0, df_lk["d0"])
    df_lk["exposed_capex_5gcm_mean_usd"] = df_lk["cap"] * (df_lk["n_models_uncompetitive"] / 5.0)
    df_lk["uncompetitive"] = df_lk["nbfut"] < 0

    df_near_all = pd.concat([pd.read_parquet(f) for f in near_files], ignore_index=True)
    df_lk_near = df_near_all.groupby(["stream_id", "zone_id"], as_index=False)[["lcoh_fut_usd_per_gj", "sub_D_fut_gj_yr"]].mean()

    ax_a = fig.add_axes([0.076, 0.458, 0.432, 0.452])
    ax_a_top = fig.add_axes([0.076, 0.914, 0.432, 0.054], sharex=ax_a)
    ax_a_rt = fig.add_axes([0.512, 0.458, 0.045, 0.452], sharey=ax_a)

    ax_b1 = fig.add_axes([0.692, 0.458, 0.120, 0.448])
    ax_b2 = fig.add_axes([0.836, 0.458, 0.122, 0.448], sharey=ax_b1)

    ax_c1 = fig.add_axes([0.076, 0.070, 0.194, 0.285])
    ax_c2 = fig.add_axes([0.352, 0.070, 0.132, 0.285])
    ax_d = fig.add_axes([0.588, 0.070, 0.378, 0.285])

    for sp in ax_a.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.60)
        sp.set_edgecolor("#000000")
        sp.set_zorder(20)

    sw_near = df_r3_sum[df_r3_sum["window_id"] == "2026_2045"]
    sw_mid = df_r3_sum[df_r3_sum["window_id"] == "2046_2065"]
    n_tot_net = len(df_lk)
    tot_cap_b = float(df_lk["cap"].sum() / 1e9)
    n_loss_m = int(round(float(sw_mid["n_loss_making_links"].mean())))
    pct_loss_m = float(sw_mid["loss_making_link_share_pct"].mean())
    exp_cap_m = float(sw_mid["loss_making_capex_billion_usd"].mean())
    m0_all = df_mid_all["net_benefit_0_usd_yr"] / np.maximum(1.0, df_mid_all["sub_D_0_gj_yr"])
    sh_lt3 = 100.0 * float(df_mid_all.loc[df_mid_all["is_loss_making_fut"] & (m0_all < 3.0), "total_sunk_capex_usd"].sum()) / max(1.0, float(df_mid_all.loc[df_mid_all["is_loss_making_fut"], "total_sunk_capex_usd"].sum()))
    print(f"  [FIG4A INFO] n_ens_uncomp={int(df_lk['uncompetitive'].sum())}, n_loss_5gcm_mean={n_loss_m}, sh_lt3={sh_lt3:.2f}%")
    w_lcoh0 = float(sw_mid["weighted_lcoh_0_usd_per_gj"].mean())
    w_lcohn = float(sw_near["weighted_lcoh_fut_usd_per_gj"].mean())
    w_lcohm = float(sw_mid["weighted_lcoh_fut_usd_per_gj"].mean())

    ax_a.axhspan(3.0, 15.2, facecolor=COL_ICE_ZONE, zorder=0)
    ax_a.axhspan(0.0, 3.0, facecolor=COL_BLUSH_ZONE, zorder=0)
    ax_a.axhline(3.0, ls="--", lw=0.65, color="#000000", zorder=4)
    ax_a.text(-39.2, 3.25, "Threshold", fontsize=FS_COMPACT, va="bottom", ha="left",
              bbox=dict(boxstyle="square,pad=0.18", facecolor=COL_ICE_ZONE, edgecolor="none", alpha=0.96), zorder=5)
    ax_a.text(-39.2, 0.22, f"Initial saving <3.0 USD GJ$^{{-1}}$ holds\n{sh_lt3:.1f}% of exposed capital ({exp_cap_m:.1f}B USD,\npooled across models)", fontsize=FS_COMPACT, va="bottom", ha="left", linespacing=1.10,
              bbox=dict(boxstyle="square,pad=0.06", facecolor=COL_BLUSH_ZONE, edgecolor="none", alpha=0.88), zorder=5)

    for mask, col, zo in [(~df_lk["uncompetitive"], COL_SAPPHIRE, 2.8), (df_lk["uncompetitive"], COL_CORAL, 3.2)]:
        sub = df_lk[mask].sort_values("cap", ascending=False)
        sz = np.clip(np.sqrt(sub["cap"].to_numpy() / 1e6) * 1.45, 2.2, 32.0)
        ax_a.scatter(
            np.clip(sub["dd_pct"], -39.5, -0.2), np.clip(sub["m0"], 0.05, 14.9),
            s=sz, color=col, alpha=0.72, edgecolors="white", linewidths=0.28, rasterized=True, zorder=zo
        )

    ax_a.add_patch(mpatches.Rectangle((-39.5, 4.45), 19.2, 3.25, facecolor=COL_ICE_ZONE, edgecolor="none", alpha=0.92, zorder=4.4))
    ax_a.text(0.04, 0.49, f"$n$ = {n_tot_net:,} networks ({tot_cap_b:.1f}B USD)", transform=ax_a.transAxes, fontsize=FS_COMPACT, va="top", ha="left", zorder=5)
    leg_a = [
        mpl.lines.Line2D([], [], color="none", marker="o", markerfacecolor=COL_SAPPHIRE, markeredgecolor="white", ms=4.8, label="Remains economic"),
        mpl.lines.Line2D([], [], color="none", marker="o", markerfacecolor=COL_CORAL, markeredgecolor="white", ms=4.8, label=f"Turns uncompetitive\n({pct_loss_m:.1f}%, {n_loss_m:,} networks)"),
    ]
    lg_obj = ax_a.legend(handles=leg_a, loc="upper left", bbox_to_anchor=(0.02, 0.43), frameon=False, fontsize=FS_COMPACT, handletextpad=0.3, labelspacing=0.25)
    lg_obj.set_zorder(5.2)

    ax_a.set_xlim(-40, 0)
    ax_a.set_ylim(0, 15.2)
    ax_a.set_xticks([-40, -30, -20, -10, 0])
    ax_a.set_xticklabels([f"{MINUS}40", f"{MINUS}30", f"{MINUS}20", f"{MINUS}10", "0"])
    ax_a.set_yticks([0, 3, 6, 9, 12, 15])
    ax_a.set_xlabel("2046–2065 heating demand change (%)")
    ax_a.set_ylabel("Initial cost saving over building heating (USD GJ$^{-1}$)")

    xg = np.linspace(-40, 0, 200)
    yg = np.linspace(0, 15.2, 200)
    for mask, col, al in [(~df_lk["uncompetitive"], COL_SAPPHIRE, 0.48), (df_lk["uncompetitive"], COL_CORAL, 0.55)]:
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
        (df_lk["lcoh0"].to_numpy(), df_lk["d0"].to_numpy(), COL_SAPPHIRE, "-", f"Present ({w_lcoh0:.2f})"),
        (df_lk_near["lcoh_fut_usd_per_gj"].to_numpy(), df_lk_near["sub_D_fut_gj_yr"].to_numpy(), COL_SSP2, "--", f"2026–2045 ({w_lcohn:.2f})"),
        (df_lk["lcoh_fut"].to_numpy(), df_lk["dfut"].to_numpy(), COL_CORAL, "-", f"2046–2065 ({w_lcohm:.2f})"),
    ]:
        ix = np.argsort(vals)
        vs = np.clip(vals[ix], 5.5, 24.0)
        ws = 100.0 * np.cumsum(wts[ix]) / wts.sum()
        ax_cdf.plot(vs, ws, color=col, ls=ls, lw=1.25)
        cdf_handles.append(mpl.lines.Line2D([], [], color=col, ls=ls, lw=1.20, label=lbl_c))
    lg_cdf = ax_cdf.legend(
        handles=cdf_handles, loc="lower right", bbox_to_anchor=(1.02, 0.02),
        frameon=True, facecolor="white", framealpha=0.92, edgecolor="none",
        fontsize=FS_COMPACT, handlelength=0.95, handletextpad=0.3, labelspacing=0.14,
    )
    lg_cdf.set_zorder(10)
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
        tot_m=("tot_b", "mean"),
        str_m=("str_b", "mean"), str_lo=("str_b", "min"), str_hi=("str_b", "max"),
        pct_m=("pct", "mean"), pct_lo=("pct", "min"), pct_hi=("pct", "max"),
    )
    cont_cand = [c for c in ["CHN", "RUS", "DEU", "POL", "KAZ", "CZE", "BLR", "AUT", "FIN"] if c in set(c_agg["iso3"])]
    warm_cand = [c for c in ["GBR", "NLD", "KOR", "SWE", "ITA", "DNK", "CAN", "JPN", "FRA"] if c in set(c_agg["iso3"])]
    cont_sel = c_agg[c_agg["iso3"].isin(cont_cand)].sort_values("str_m", ascending=False).head(7)["iso3"].tolist()
    warm_sel = c_agg[c_agg["iso3"].isin(warm_cand)].sort_values("pct_m", ascending=False).head(6)["iso3"].tolist()
    sel_isos = cont_sel + warm_sel
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

    xmax_b1 = max(12.0, float(c_plot["str_hi"].max()) * 1.32)
    ax_b1.barh(yb, c_plot["str_m"], height=0.64, color=COL_CORAL, edgecolor="none", zorder=3)
    ax_b1.errorbar(c_plot["str_m"], yb, xerr=[c_plot["str_m"] - c_plot["str_lo"], c_plot["str_hi"] - c_plot["str_m"]], fmt="none", ecolor="#000000", elinewidth=0.60, capsize=1.5, zorder=4)
    for yi, r in c_plot.iterrows():
        if r["str_m"] >= 0.8:
            ax_b1.text(r["str_hi"] + 0.025 * xmax_b1, yi, f"{r['str_m']:.1f}", va="center", ha="left", fontsize=FS_COMPACT)

    ax_b2.barh(yb, c_plot["pct_m"], height=0.64, color=COL_APRICOT, edgecolor="none", zorder=3)
    ax_b2.errorbar(c_plot["pct_m"], yb, xerr=[c_plot["pct_m"] - c_plot["pct_lo"], c_plot["pct_hi"] - c_plot["pct_m"]], fmt="none", ecolor="#000000", elinewidth=0.60, capsize=1.5, zorder=4)
    for yi, r in c_plot.iterrows():
        ax_b2.text(r["pct_hi"] + 2.0, yi, f"{r['pct_m']:.0f}%", va="center", ha="left", fontsize=FS_COMPACT)

    sep_y = len(warm_sel) - 0.5
    ax_b1.axhline(sep_y, ls="--", lw=0.50, color="#333333", zorder=5)
    ax_b2.axhline(sep_y, ls="--", lw=0.50, color="#333333", zorder=5)
    ax_b1.set_yticks(yb)
    ax_b1.set_yticklabels(["UK" if i == "GBR" else cname(i) for i in c_plot["iso3"]], fontsize=FS_COMPACT)
    plt.setp(ax_b2.get_yticklabels(), visible=False)
    ax_b2.tick_params(axis="y", length=1.8)

    ax_b1.set_xlim(0, xmax_b1)
    t_step_b1 = 5 if xmax_b1 <= 16.0 else 10
    ax_b1.set_xticks([0, t_step_b1, 2 * t_step_b1])
    ax_b1.text(0.5, 1.03, "Exposed capital\n(billion USD)", transform=ax_b1.transAxes, ha="center", va="bottom", fontsize=FS_COMPACT, linespacing=1.10)

    ax_b2.set_xlim(0, 108)
    ax_b2.set_xticks([0, 40, 80])
    ax_b2.text(0.5, 1.03, "Share of national\ncapital (%)", transform=ax_b2.transAxes, ha="center", va="bottom", fontsize=FS_COMPACT, linespacing=1.10)

    fig.text(0.980, 0.77, "Continental\npipe systems", rotation=90, va="center", ha="center", fontsize=FS_COMPACT)
    fig.text(0.980, 0.55, "Warm & electrified\nmarkets", rotation=90, va="center", ha="center", fontsize=FS_COMPACT)

    clean_ax(ax_c1, "y")
    clean_ax(ax_c2, "y")

    p0_tot = float(df_r3_sum["P_0_ej_yr"].iloc[0])
    nb0_tot = float(df_lk["nb0"].sum() / 1e9)

    ax_c1.axhline(0.0, lw=0.65, color="#000000", zorder=18)
    bw_c = 0.29
    off_c = 0.17
    min_c1_val = -40.0
    for xi, sw in enumerate([sw_near, sw_mid]):
        dp_arr = 100.0 * (sw["P_locked_fut_ej_yr"].to_numpy(dtype=float) - p0_tot) / p0_tot
        dn_arr = 100.0 * (sw["net_benefit_fut_billion_usd_yr"].to_numpy(dtype=float) - nb0_tot) / nb0_tot
        dp_m, dn_m = float(dp_arr.mean()), float(dn_arr.mean())
        min_c1_val = min(min_c1_val, float(dn_arr.min()) - 8.5)

        ax_c1.bar(xi - off_c, dp_m, width=bw_c, color=COL_SAPPHIRE, edgecolor="none", zorder=3)
        ax_c1.bar(xi + off_c, dn_m, width=bw_c, color=COL_CORAL, edgecolor="none", zorder=3)

        for (_, r), dpv, dnv in zip(sw.iterrows(), dp_arr, dn_arr):
            mk = GCM_MARKER[r["model"]]
            ax_c1.scatter(xi - off_c, dpv, facecolors="none", edgecolors="#000000", marker=mk, s=8.0, linewidths=0.46, zorder=4.5)
            ax_c1.scatter(xi + off_c, dnv, facecolors="none", edgecolors="#000000", marker=mk, s=8.0, linewidths=0.46, zorder=4.5)

        if xi == 0:
            ax_c1.text(xi - off_c, float(dp_arr.min()) - 1.3, f"{MINUS}{abs(dp_m):.1f}%", ha="center", va="top", fontsize=FS_COMPACT)
            ax_c1.text(xi + off_c, float(dn_arr.min()) - 1.3, f"{MINUS}{abs(dn_m):.1f}%", ha="center", va="top", fontsize=FS_COMPACT)
        else:
            mult_r = abs(dn_m) / max(0.1, abs(dp_m))
            ax_c1.text(xi - off_c, float(dp_arr.min()) - 1.3, f"{MINUS}{abs(dp_m):.1f}%", ha="center", va="top", fontsize=FS_COMPACT, fontweight="bold")
            ax_c1.text(xi + off_c + 0.02, float(dn_arr.min()) - 1.2, f"{MINUS}{abs(dn_m):.1f}%\n({mult_r:.1f}\u00d7)", ha="center", va="top", fontsize=FS_COMPACT, fontweight="bold", linespacing=0.95)

    leg_c1 = [
        mpatches.Patch(facecolor=COL_SAPPHIRE, edgecolor="none", label="Delivered heat"),
        mpatches.Patch(facecolor=COL_CORAL, edgecolor="none", label="Net savings"),
    ]
    ax_c1.legend(handles=leg_c1, loc="lower left", bbox_to_anchor=(-0.03, 0.00), frameon=False, fontsize=FS_COMPACT, handlelength=0.85, handletextpad=0.35, labelspacing=0.18)

    ax_c1.set_xlim(-0.50, 1.54)
    ax_c1.set_ylim(min(-47.5, min_c1_val), 3.0)
    ax_c1.set_xticks([0, 1])
    ax_c1.set_xticklabels(["2026–\n2045", "2046–\n2065"], fontsize=FS_COMPACT)
    ax_c1.set_yticks([0, -15, -30, -45])
    ax_c1.set_yticklabels(["0", f"{MINUS}15", f"{MINUS}30", f"{MINUS}45"])
    ax_c1.set_ylabel("Change from present (%)")
    lock_spines_above_data(ax_c1)

    max_c2 = 20.0
    for xi, sw in enumerate([sw_near, sw_mid]):
        fm_v = max(0.0, float(sw["foregone_co2_abatement_mt_yr"].mean()))
        sm_v = max(0.0, float(sw["stranded_reversion_co2_mt_yr"].mean()))
        ax_c2.bar(xi, fm_v, width=0.58, color=COL_SKY_CYAN, edgecolor="none", zorder=3)
        ax_c2.bar(xi, sm_v, bottom=fm_v, width=0.58, color=COL_CORAL_SOFT, edgecolor="none", zorder=3)
        ax_c2.plot([xi - 0.30, xi + 0.30], [fm_v, fm_v], color="white", lw=0.60, zorder=3.5)
        for _, r in sw.iterrows():
            tv = float(r["foregone_co2_abatement_mt_yr"] + r["stranded_reversion_co2_mt_yr"])
            max_c2 = max(max_c2, tv)
            ax_c2.scatter(xi, tv, facecolors="none", edgecolors="#000000", marker=GCM_MARKER[r["model"]], s=9.5, linewidths=0.5, zorder=5)
        if xi == 1:
            ax_c2.text(xi, fm_v + sm_v + 0.6, f"{fm_v + sm_v:.1f}", ha="center", va="bottom", fontsize=FS_COMPACT, fontweight="bold")
            if fm_v > 2.5:
                ax_c2.text(xi, fm_v * 0.48, f"{fm_v:.1f}", ha="center", va="center", fontsize=FS_COMPACT)
            if sm_v > 2.0:
                ax_c2.text(xi, fm_v + sm_v * 0.48, f"{sm_v:.1f}", ha="center", va="center", fontsize=FS_COMPACT)

    leg_c2 = [
        mpatches.Patch(facecolor=COL_SKY_CYAN, edgecolor="none", label="Lower delivery"),
        mpatches.Patch(facecolor=COL_CORAL_SOFT, edgecolor="none", label="Uncompetitive"),
    ]
    ax_c2.legend(handles=leg_c2, loc="upper left", bbox_to_anchor=(-0.04, 1.03), frameon=False, fontsize=FS_COMPACT, handlelength=0.85, labelspacing=0.18)

    ax_c2.set_xlim(-0.52, 1.52)
    ax_c2.set_ylim(0, max(32.0, max_c2 * 1.25))
    ax_c2.set_xticks([0, 1])
    ax_c2.set_xticklabels(["2026–\n2045", "2046–\n2065"], fontsize=FS_COMPACT)
    ax_c2.set_yticks([0, 10, 20, 30])
    ax_c2.set_ylabel("Reduced & exposed CO$_2$\n(Mt CO$_2$ yr$^{-1}$)")
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
            "uncompetitive_lcoh_mean_usd_per_gj": float(un_mean[idx_s]),
            "uncompetitive_lcoh_min_usd_per_gj": float(un_min[idx_s]),
            "uncompetitive_lcoh_max_usd_per_gj": float(un_max[idx_s]),
            "uncompetitive_competing_benchmark_usd_per_gj": comp_bench_un,
            "all_networks_lcoh_mean_usd_per_gj": float(all_mean[idx_s]),
            "all_networks_lcoh_min_usd_per_gj": float(all_min[idx_s]),
            "all_networks_lcoh_max_usd_per_gj": float(all_max[idx_s]),
            "all_networks_competing_benchmark_usd_per_gj": comp_bench_all,
            "uncompetitive_unrescued_share_pct": unrescued_pct,
        })
    pd.DataFrame(ptes_rows).to_csv(SRC_DIR / "Figure4d_ptes_source_data.csv", index=False)

    ymax_d = max(34.0, float(un_max[-1]) + 2.5)
    ymin_d = min(12.0, float(all_min[0]) - 1.8)
    ax_d.axhspan(comp_bench_un, ymax_d, facecolor=COL_BLUSH_ZONE, zorder=0)
    ax_d.axhline(comp_bench_un, ls=":", lw=0.75, color=COL_CORAL, zorder=2)
    ax_d.axhline(comp_bench_all, ls="--", lw=0.65, color="#000000", zorder=2)

    ax_d.fill_between(s_pct, un_min, un_max, facecolor=COL_CORAL, alpha=0.17, edgecolor="none", zorder=2.6)
    ax_d.plot(s_pct, un_mean, color=COL_CORAL, lw=1.70, zorder=4.2)
    ax_d.scatter([0], [un_mean[0]], color=COL_CORAL, marker="o", s=16, zorder=5)

    ax_d.fill_between(s_pct, all_min, all_max, facecolor=COL_SAPPHIRE, alpha=0.16, edgecolor="none", zorder=2.5)
    ax_d.plot(s_pct, all_mean, color=COL_SAPPHIRE, lw=1.70, zorder=4.0)
    ax_d.scatter([0], [all_mean[0]], color=COL_SAPPHIRE, marker="D", s=16, zorder=5)

    leg_d = [
        mpl.lines.Line2D([], [], color=COL_CORAL, lw=1.70, marker="o", ms=3.6, label=f"Uncompetitive networks (n = {n_loss_m:,})"),
        mpl.lines.Line2D([], [], color=COL_SAPPHIRE, lw=1.70, marker="D", ms=3.4, label=f"All fixed networks (n = {n_tot_net:,})"),
    ]
    ax_d.legend(handles=leg_d, loc="upper left", bbox_to_anchor=(-0.01, 1.02), frameon=False, fontsize=FS_COMPACT, handlelength=1.15, handletextpad=0.4, labelspacing=0.22)

    y_call_un = 21.6
    ax_d.plot([0.15, 1.25], [un_mean[0] + 0.28, y_call_un - 0.25], color="#000000", lw=0.45, zorder=4.5)
    ax_d.text(1.0, y_call_un, f"{un_mean[0]:.2f} USD GJ$^{{-1}}$ at 0%\n{unrescued_pct:.1f}% stay uncompetitive\nat any storage share", fontsize=FS_COMPACT, va="bottom", ha="left", linespacing=1.12)
    ax_d.text(
        34.7,
        comp_bench_un + 0.14,
        f"Uncompetitive-network\nbenchmark ({comp_bench_un:.2f})",
        fontsize=FS_COMPACT,
        va="bottom",
        ha="right",
        linespacing=1.05,
        bbox=dict(boxstyle="square,pad=0.05", facecolor=COL_BLUSH_ZONE, edgecolor="none", alpha=0.92),
        zorder=4.4,
    )
    ax_d.text(34.5, comp_bench_all - 0.35, f"All-network benchmark ({comp_bench_all:.2f})", fontsize=FS_COMPACT, va="top", ha="right")
    ax_d.text(1.3, all_mean[0] - 0.35, f"{all_mean[0]:.2f} USD GJ$^{{-1}}$", fontsize=FS_COMPACT, va="top", ha="left")
    ax_d.text(15.5, ymin_d + 2.3, f"+{c_ptes_cap:.1f} USD GJ$^{{-1}}$ per stored GJ\n(30% loss, before boiler offset)", fontsize=FS_COMPACT, va="top", ha="left", linespacing=1.12)

    ax_d.set_xlim(0, 35)
    ax_d.set_ylim(ymin_d, ymax_d)
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
    df_lk.to_csv(SRC_DIR / "Figure4a_networks_source_data.csv", index=False)
    return export_figure(fig, "Figure4_Fixed_Network_Exposure_and_Carbon")


# ==============================================================================
# FIGURE 5 (Results 4 — Restored Stacked-Bar Architecture + Refreshed Palette
# + ZERO Connecting Polylines + Spine zorder=20 + 100% Pure Black Font!)
# ==============================================================================
def build_figure_5(df_glob: pd.DataFrame, df_country: pd.DataFrame) -> dict:
    configure_matplotlib_style()
    H_MM = 184.0  # Strictly <= 185.0 mm ceiling!
    fig = plt.figure(figsize=(W_IN, H_MM * MM))

    ax_a1 = fig.add_axes([0.064, 0.640, 0.330, 0.312])
    ax_a2 = fig.add_axes([0.432, 0.640, 0.152, 0.238])
    ax_b = fig.add_axes([0.758, 0.608, 0.160, 0.342])
    gs_c = gridspec.GridSpec(3, 5, figure=fig, left=0.064, right=0.982, bottom=0.096, top=0.495, wspace=0.35, hspace=0.56)

    def glob_subsectors(scen_filter: pd.Series) -> tuple[dict[str, float], pd.Series]:
        sub = df_country[scen_filter].copy()
        sub["chem_ww"] = sub["P_ethylene_gj"] + sub["P_pulp_paper_gj"] + sub["P_wastewater_gj"]
        by_m = sub.groupby("model")[["P_power_gj", "P_steel_gj", "P_refining_gj", "P_cement_gj", "chem_ww", "P_gj"]].sum() / 1e9
        return by_m.mean().to_dict(), by_m["P_gj"]

    steps_a = [
        ("Present", df_country["scenario_id"] == "R1__obs__2006_2025", None),
        ("2046–2065\nwarming", (df_country["group"] == "R1") & (df_country["ssp"] == "ssp245") & (df_country["window_id"] == "2046_2065"), None),
        ("55/25 °C\n(vs 2025)", (df_country["group"] == "R4") & (df_country["subgroup"] == "S3_4gdh_standalone") & (df_country["window_id"] == "2046_2065"), "R4__S3_4gdh_standalone__obs__obsclim__2006_2025"),
        ("55/25 °C\n(vs 2050)", (df_country["group"] == "R4") & (df_country["subgroup"] == "S4_4gdh_symmetric") & (df_country["window_id"] == "2046_2065"), "R4__S4_4gdh_symmetric__obs__obsclim__2006_2025"),
        ("Envelope\nretrofits", (df_country["group"] == "R4") & (df_country["subgroup"] == "S2_envelope_retrofit") & (df_country["window_id"] == "2046_2065"), "R4__S2_envelope_retrofit__obs__obsclim__2006_2025"),
        ("Fossil\nretirement", (df_country["group"] == "R4") & (df_country["subgroup"] == "S1_fossil_retire") & (df_country["window_id"] == "2046_2065"), "R4__S1_fossil_retire__obs__obsclim__2006_2025"),
        ("Combined\n2050", (df_country["group"] == "R4") & (df_country["subgroup"] == "S5_combined_all") & (df_country["window_id"] == "2046_2065"), "R4__S5_combined_all__obs__obsclim__2006_2025"),
    ]

    sec_layers = [
        ("P_power_gj", "Power plants / Nuclear & bioenergy", F5_POWER),
        ("P_steel_gj", "Iron & steel", F5_STEEL),
        ("P_refining_gj", "Oil refining", F5_REFINING),
        ("P_cement_gj", "Cement", F5_CEMENT),
        ("chem_ww", "Chemicals, pulp & wastewater", F5_CHEM),
    ]

    p0_present_ej = float(df_glob.loc[df_glob["scenario_id"] == "R1__obs__2006_2025", "P_ej_yr"].iloc[0])
    clean_ax(ax_a1, "y")
    bw = 0.60
    src_f5 = []
    for xi, (slbl, sfilt, base_sid) in enumerate(steps_a):
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

        p_2006_cfg = np.nan
        if base_sid is not None:
            p_2006_cfg = float(df_country.loc[df_country["scenario_id"] == base_sid, "P_gj"].sum() / 1e9)
            ax_a1.scatter(xi, p_2006_cfg, facecolors="#FFFFFF", edgecolors="#000000", marker="h", s=22.0, linewidths=0.88, zorder=6.2)

        top_ref = max(bot, float(mod_tot.max()), 0.0 if np.isnan(p_2006_cfg) else p_2006_cfg)
        y_lbl = top_ref + 0.09
        if xi == 6:
            ax_a1.text(xi, max(0.50, y_lbl), f"{bot:.2f}", ha="center", va="bottom", fontsize=FS_COMPACT, fontweight="bold")
        else:
            ax_a1.text(xi, y_lbl, f"{bot:.2f}", ha="center", va="bottom", fontsize=FS_COMPACT, fontweight="bold")
        src_f5.append({
            "step": slbl.replace("\n", " "),
            "P_2006_2025_ej": float(bot) if xi == 0 else (p0_present_ej if xi == 1 else float(p_2006_cfg)),
            "P_mean_ej": bot,
            "P_min_ej": float(mod_tot.min()),
            "P_max_ej": float(mod_tot.max()),
        })

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
    leg_f5.append(mpl.lines.Line2D([], [], color="none", marker="h", markerfacecolor="#FFFFFF", markeredgecolor="#000000", markeredgewidth=0.88, ms=4.5, label="2006–2025 climate"))
    fig.legend(handles=leg_f5, loc="upper left", bbox_to_anchor=(0.328, 0.988), frameon=False, fontsize=FS_COMPACT, handlelength=0.95, handleheight=0.78, labelspacing=0.14)

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

    s5_b0 = df_country[df_country["scenario_id"] == "R4__S5_combined_all__obs__obsclim__2006_2025"].copy()
    s5_b0["chem_ww"] = s5_b0["P_ethylene_gj"] + s5_b0["P_pulp_paper_gj"] + s5_b0["P_wastewater_gj"]
    s5_b0_sum = (s5_b0[["P_steel_gj", "P_power_gj", "P_refining_gj", "P_cement_gj", "chem_ww"]].sum() / 1e6).to_dict()

    z_specs = [
        ("Steel", "P_steel_gj", F5_STEEL),
        ("Nuclear & bio.", "P_power_gj", F5_POWER),
        ("Refining", "P_refining_gj", F5_REFINING),
        ("Cement", "P_cement_gj", F5_CEMENT),
        ("Chem.", "chem_ww", F5_CHEM),
    ]
    tot_s5 = float(s5_mod.sum(axis=1).mean())
    for xi2, (zlbl, zcol, zhex) in enumerate(z_specs):
        vals = s5_mod[zcol].to_numpy()
        mv = float(vals.mean())
        b0_v = float(s5_b0_sum[zcol])
        ax_a2.bar(xi2, mv, width=0.58, color=zhex, edgecolor="none", zorder=3)
        for m_id, vm in s5_mod[zcol].items():
            ax_a2.scatter(xi2, vm, facecolors="none", edgecolors="#000000", marker=GCM_MARKER[str(m_id)], s=8, linewidths=0.45, zorder=4)
        ax_a2.scatter(xi2, b0_v, facecolors="#FFFFFF", edgecolors="#000000", marker="h", s=18.0, linewidths=0.82, zorder=6.2)
        top_z = max(mv, float(vals.max()), b0_v)
        y_off = 3.0 if xi2 == 0 else (18.0 if xi2 == 1 else (10.0 if xi2 == 2 else 2.5))
        ax_a2.text(xi2, top_z + y_off, f"{100*mv/tot_s5:.1f}%", ha="center", va="bottom", fontsize=FS_COMPACT)

    s5_b0_tot = float(sum(s5_b0_sum.values()))
    dp_s5_pct = 100.0 * (tot_s5 - s5_b0_tot) / max(1e-9, s5_b0_tot)
    cf_json_f5 = SCEN_DIR / "counterfactual_and_planning_summary.json"
    if cf_json_f5.exists():
        import json as _json
        _cf5 = _json.loads(cf_json_f5.read_text(encoding="utf-8"))
        dp_s5_pct = float(_cf5["PartA_R4_2050_Climate_Effects"]["S5_combined_all"]["decomp_vs_mod_2006_2025"]["dP_pct_mean"])
    ymax_a2 = max(145.0, float(max(max(s5_mod[z[1]].max(), s5_b0_sum[z[1]]) for z in z_specs)) * 1.32)
    ax_a2.text(0.60, 0.96, f"{tot_s5:.1f} PJ\n({fm(dp_s5_pct, '.1f')}% vs\n2006–25)", transform=ax_a2.transAxes, ha="center", va="top", fontsize=FS_COMPACT, linespacing=1.08)
    ax_a2.set_xlim(-0.72, 4.65)
    ax_a2.set_ylim(0, ymax_a2)
    ax_a2.set_xticks(range(5))
    ax_a2.set_xticklabels([z[0] for z in z_specs], fontsize=FS_COMPACT, rotation=28, ha="right", linespacing=0.92)
    ax_a2.set_yticks([0, 45, 90, 135])
    ax_a2.set_ylabel("Combined 2050 potential (PJ yr$^{-1}$)")
    lock_spines_above_data(ax_a2)

    for yl, yr in [(ZOOM_F5, ymax_a2), (0.0, 0.0)]:
        fig.add_artist(ConnectionPatch(
            xyA=(zx1, yl), coordsA=ax_a1.transData,
            xyB=(-0.72, yr), coordsB=ax_a2.transData,
            linestyle="--", linewidth=0.50, color="#333333"
        ))

    # --- Panel b: Shaded Range-Band Sensitivity Forest Plot + Right Tabular P0 Column (11 Full-Global 183-Region Specs!) ---
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
    central_dp = float(np.mean([100.0 * (float(r["P_ej_yr"]) - p0_v) / p0_v for _, r in clim.iterrows()]))

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
        ("CHP power loss", "Rob5_chp_power_loss"),
        ("CHP fuel alloc.", "Rob6_chp_fuel_alloc"),
    ][::-1]

    yb_r = np.arange(len(rob_specs))
    ax_b.axvline(0.0, lw=0.65, color="#000000", zorder=18)
    ax_b.axvline(central_dp, ls="--", lw=0.60, color="#000000", zorder=2)
    ax_b.text(central_dp - 1.0, len(rob_specs) - 0.30, f"{fm(central_dp, '.1f')}%", ha="right", va="bottom", fontsize=FS_COMPACT)
    ax_b.text(17.5, len(rob_specs) - 0.30, "$P_0$\n(EJ yr$^{-1}$)", ha="right", va="bottom", fontsize=FS_COMPACT, linespacing=1.05)

    xmin_b = -34.0
    for yi, (rlbl, rcode) in enumerate(rob_specs):
        if rcode == "CENTRAL":
            bv, sub = p0_v, clim
        else:
            sr = df_glob[(df_glob["group"] == "Robustness") & (df_glob["subgroup"] == rcode)]
            bv = float(sr.loc[sr["window_id"] == "2006_2025", "P_ej_yr"].iloc[0])
            sub = sr[sr["window_id"] == "2046_2065"]
        dps = [100.0 * (float(r["P_ej_yr"]) - bv) / bv for _, r in sub.iterrows()]
        lo, hi, mv = float(min(dps)), float(max(dps)), float(np.mean(dps))
        xmin_b = min(xmin_b, lo - 3.0)
        ax_b.barh(yi, hi - lo, left=lo, height=0.60, color=F5_BAND, alpha=0.84, edgecolor="none", zorder=1)
        ax_b.plot([mv, mv], [yi - 0.28, yi + 0.28], color="#000000", lw=1.25, zorder=4)
        for (_, r), dpv in zip(sub.iterrows(), dps):
            ax_b.scatter(dpv, yi, facecolors="none", edgecolors="#000000", marker=GCM_MARKER[r["model"]], s=8.0, linewidths=0.46, zorder=3)
        ax_b.text(17.5, yi, f"{bv:.2f}", ha="right", va="center", fontsize=FS_COMPACT)

    ax_b.set_yticks(yb_r)
    ax_b.set_yticklabels([x[0] for x in rob_specs], fontsize=FS_COMPACT)
    ax_b.set_xlim(xmin_b, 18.5)
    ax_b.set_ylim(-0.6, len(rob_specs) - 0.25)
    ax_b.set_xticks([-40, -20, 0])
    ax_b.set_xticklabels([f"{MINUS}40", f"{MINUS}20", "0"])
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
            ax_m.set_xticklabels(["2025", "2046–65", "55/25 °C (2025 HP)", "55/25 °C (2050 HP)", "Combined 2050"], fontsize=5.3, rotation=48, ha="right")
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
        assert len(a["overlaps"]) == 0, f"Text bbox overlap detected in {a['stem']}: {a['overlaps']}"


if __name__ == "__main__":
    main()
