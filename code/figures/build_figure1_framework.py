#!/usr/bin/env python3
"""Generate Figure 1 (study modelling framework and experimental design) in vector PDF, SVG and PNG."""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib as mpl
from matplotlib.patches import Polygon as MplPolygon, Rectangle as MplRectangle
import matplotlib.pyplot as plt

ROOT = Path(os.environ["WASTE_HEAT_ROOT"]) if "WASTE_HEAT_ROOT" in os.environ else Path(__file__).resolve().parents[2]
MAIN_FIG_DIR = ROOT / "figures/main"

CANVAS_W = 1376
CANVAS_H = 768

STYLE_CFG = {
    "line_gap_px": 21.5,
    "outline_inset": 0.75,
    "arrow_len": 7.2,
    "arrow_half_w": 3.6,
    "rec_x": 477,
    "rec_w": 206,
    "roles": {
        "data_input":  {"fill": "#B9D4EE", "stroke": "#244D85"},
        "parameter":   {"fill": "#F2F3F5", "stroke": "#5C6672"},
        "calculation": {"fill": "#FCE9AE", "stroke": "#B88628"},
        "output":      {"fill": "#EE9B82", "stroke": "#A63A2B"},
    },
}


def build_nodes_spec(cfg: dict) -> list[dict]:
    rx, rw = cfg["rec_x"], cfg["rec_w"]
    return [
        {"id": "node_i",   "role": "data_input",  "x": 31,   "y": 109, "w": 201, "h": 84, "lines": ["(i) Global facility", "inventory (2025)"]},
        {"id": "node_ii",  "role": "data_input",  "x": 31,   "y": 290, "w": 201, "h": 84, "lines": ["(ii) Gridded 100 m", "building volumes"]},
        {"id": "node_iii", "role": "data_input",  "x": 31,   "y": 468, "w": 201, "h": 84, "lines": ["(iii) Daily weather &", "CMIP6 climates"]},
        {"id": "node_1",   "role": "calculation", "x": 257,  "y": 108, "w": 206, "h": 86, "lines": ["(1) Quantify plant", "waste heat streams"]},
        {"id": "node_3",   "role": "calculation", "x": 257,  "y": 289, "w": 206, "h": 86, "lines": ["(3) Delineate dense", "urban heating areas"]},
        {"id": "node_2",   "role": "calculation", "x": 257,  "y": 467, "w": 206, "h": 86, "lines": ["(2) Simulate daily", "building heat demand"]},
        {"id": "node_iv",  "role": "parameter",   "x": 258,  "y": 658, "w": 205, "h": 85, "lines": ["(iv) Heating thresholds", "& national surveys"]},
        {"id": "out_recoverable", "role": "output", "x": rx, "y": 130, "w": rw,  "h": 43, "lines": ["Recoverable waste heat"]},
        {"id": "node_4",   "role": "calculation", "x": 477,  "y": 289, "w": 206, "h": 86, "lines": ["(4) Dispatch daily heat", "within 100 km"]},
        {"id": "node_5",   "role": "calculation", "x": 477,  "y": 467, "w": 206, "h": 86, "lines": ["(5) Screen network vs", "building-level heating"]},
        {"id": "node_v",   "role": "parameter",   "x": 477,  "y": 658, "w": 206, "h": 85, "lines": ["(v) Equipment costs,", "tariffs & daily COP"]},
        {"id": "out_technical", "role": "output", "x": 698,  "y": 305, "w": 202, "h": 53, "lines": ["Technical potential"]},
        {"id": "out_economic",  "role": "output", "x": 698,  "y": 483, "w": 202, "h": 53, "lines": ["Economic potential"]},
        {"id": "node_vi",  "role": "parameter",   "x": 698,  "y": 658, "w": 202, "h": 85, "lines": ["(vi) 2050 network, heat", "pump & grid updates"]},
        {"id": "node_6",   "role": "calculation", "x": 922,  "y": 108, "w": 207, "h": 86, "lines": ["(6) Replay 1946\u20132100", "climate trajectories"]},
        {"id": "node_7",   "role": "calculation", "x": 922,  "y": 289, "w": 207, "h": 86, "lines": ["(7) Decompose regional", "potential changes"]},
        {"id": "node_8",   "role": "calculation", "x": 922,  "y": 467, "w": 207, "h": 86, "lines": ["(8) Lock 40 yr networks", "for present climate"]},
        {"id": "node_9",   "role": "calculation", "x": 922,  "y": 658, "w": 207, "h": 85, "lines": ["(9) Apply 2050 system", "transitions & 9 tests"]},
        {"id": "out_r1",   "role": "output",      "x": 1146, "y": 108, "w": 201, "h": 86, "lines": ["Historical & projected", "global potential"]},
        {"id": "out_r2",   "role": "output",      "x": 1146, "y": 289, "w": 201, "h": 86, "lines": ["Demand, matching &", "viability drivers"]},
        {"id": "out_r3",   "role": "output",      "x": 1146, "y": 467, "w": 201, "h": 86, "lines": ["Exposed network capital", "& lost CO\u2082 abatement"]},
        {"id": "out_r4",   "role": "output",      "x": 1146, "y": 658, "w": 201, "h": 85, "lines": ["2050 potential without", "fossil power"]},
    ]


def build_legend_spec() -> list[dict]:
    return [
        {"role": "data_input",  "sx": 31,  "sy": 621, "sw": 92, "sh": 33, "label": "Data inputs",  "ly": 671},
        {"role": "calculation", "sx": 138, "sy": 621, "sw": 94, "sh": 33, "label": "Calculations", "ly": 671},
        {"role": "parameter",   "sx": 31,  "sy": 683, "sw": 92, "sh": 35, "label": "Parameters",   "ly": 735},
        {"role": "output",      "sx": 138, "sy": 683, "sw": 94, "sh": 35, "label": "Outputs",      "ly": 735},
    ]


def build_edges_spec(nodes_by_id: dict[str, dict]) -> list[dict]:
    def r_mid(nid: str):
        n = nodes_by_id[nid]
        return (float(n["x"] + n["w"]), float(n["y"] + n["h"] / 2.0))

    def l_mid(nid: str):
        n = nodes_by_id[nid]
        return (float(n["x"]), float(n["y"] + n["h"] / 2.0))

    def t_mid(nid: str):
        n = nodes_by_id[nid]
        return (float(n["x"] + n["w"] / 2.0), float(n["y"]))

    def b_mid(nid: str):
        n = nodes_by_id[nid]
        return (float(n["x"] + n["w"] / 2.0), float(n["y"] + n["h"]))

    spine_x = 910.0
    econ_y = r_mid("out_economic")[1]
    vi_9_y = 710.0
    return [
        {"pts": [r_mid("node_i"), (l_mid("node_1")[0], r_mid("node_i")[1])], "arrow": "right"},
        {"pts": [(r_mid("node_1")[0], l_mid("out_recoverable")[1]), l_mid("out_recoverable")], "arrow": "right"},
        {"pts": [b_mid("out_recoverable"), t_mid("node_4")], "arrow": "down"},
        {"pts": [r_mid("node_ii"), (l_mid("node_3")[0], r_mid("node_ii")[1])], "arrow": "right"},
        {"pts": [r_mid("node_iii"), (l_mid("node_2")[0], r_mid("node_iii")[1])], "arrow": "right"},
        {"pts": [(243.0, r_mid("node_iii")[1]), (243.0, 757.0), (b_mid("node_v")[0], 757.0), b_mid("node_v")], "arrow": "up"},
        {"pts": [(t_mid("node_2")[0], t_mid("node_iv")[1]), b_mid("node_2")], "arrow": "up"},
        {"pts": [t_mid("node_2"), b_mid("node_3")], "arrow": "up"},
        {"pts": [r_mid("node_3"), l_mid("node_4")], "arrow": "right"},
        {"pts": [(r_mid("node_4")[0], l_mid("out_technical")[1]), l_mid("out_technical")], "arrow": "right"},
        {"pts": [b_mid("node_4"), t_mid("node_5")], "arrow": "down"},
        {"pts": [t_mid("node_v"), b_mid("node_5")], "arrow": "up"},
        {"pts": [(r_mid("node_5")[0], econ_y), l_mid("out_economic")], "arrow": "right"},
        {"pts": [r_mid("out_economic"), (spine_x, econ_y), (spine_x, l_mid("node_6")[1]), l_mid("node_6")], "arrow": "right"},
        {"pts": [(spine_x, l_mid("node_7")[1]), l_mid("node_7")], "arrow": "right"},
        {"pts": [(spine_x, econ_y), (l_mid("node_8")[0], econ_y)], "arrow": "right"},
        {"pts": [(spine_x, econ_y), (spine_x, 616.0), (t_mid("node_9")[0], 616.0), t_mid("node_9")], "arrow": "down"},
        {"pts": [(r_mid("node_vi")[0], vi_9_y), (l_mid("node_9")[0], vi_9_y)], "arrow": "right"},
        {"pts": [r_mid("node_6"), l_mid("out_r1")], "arrow": "right"},
        {"pts": [r_mid("node_7"), l_mid("out_r2")], "arrow": "right"},
        {"pts": [r_mid("node_8"), l_mid("out_r3")], "arrow": "right"},
        {"pts": [r_mid("node_9"), l_mid("out_r4")], "arrow": "right"},
    ]


def arrow_triangle_pts(tip: tuple[float, float], direction: str, alen: float, ahalf: float):
    tx, ty = tip
    if direction == "right":
        return [(tx, ty), (tx - alen, ty - ahalf), (tx - alen, ty + ahalf)], (tx - alen * 0.6, ty)
    if direction == "left":
        return [(tx, ty), (tx + alen, ty - ahalf), (tx + alen, ty + ahalf)], (tx + alen * 0.6, ty)
    if direction == "down":
        return [(tx, ty), (tx - ahalf, ty - alen), (tx + ahalf, ty - alen)], (tx, ty - alen * 0.6)
    return [(tx, ty), (tx - ahalf, ty + alen), (tx + ahalf, ty + alen)], (tx, ty + alen * 0.6)


def build_figure_1(out_dir: Path = MAIN_FIG_DIR) -> dict:
    w_mm = 180.0
    h_mm = w_mm * (CANVAS_H / CANVAS_W)
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Nimbus Sans", "Liberation Sans", "Arial", "Helvetica"],
        "mathtext.default": "regular",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })

    fig = plt.figure(figsize=(w_mm / 25.4, h_mm / 25.4), dpi=450, facecolor="#FFFFFF")
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0.0, float(CANVAS_W))
    ax.set_ylim(float(CANVAS_H), 0.0)
    ax.set_aspect("equal")
    ax.axis("off")

    nodes = build_nodes_spec(STYLE_CFG)
    nodes_by_id = {n["id"]: n for n in nodes}
    legends = build_legend_spec()
    edges = build_edges_spec(nodes_by_id)

    wcol = "#2C3138"
    for ed in edges:
        pts = list(ed["pts"])
        tri, shaft_end = arrow_triangle_pts(pts[-1], ed["arrow"], STYLE_CFG["arrow_len"], STYLE_CFG["arrow_half_w"])
        shaft_pts = pts[:-1] + [shaft_end]
        ax.plot([p[0] for p in shaft_pts], [p[1] for p in shaft_pts], color=wcol, linewidth=0.62, solid_capstyle="butt", zorder=2)
        ax.add_patch(MplPolygon(tri, closed=True, facecolor=wcol, edgecolor="none", zorder=3))

    ins = STYLE_CFG["outline_inset"]
    lgap = STYLE_CFG["line_gap_px"]
    for n in nodes:
        rs = STYLE_CFG["roles"][n["role"]]
        x, y, w, h = float(n["x"]), float(n["y"]), float(n["w"]), float(n["h"])
        ax.add_patch(MplRectangle((x, y), w, h, facecolor=rs["fill"], edgecolor="none", zorder=4))
        ax.add_patch(MplRectangle((x + ins, y + ins), w - 2 * ins, h - 2 * ins, facecolor="none", edgecolor=rs["stroke"], linewidth=0.65, zorder=6))
        cx, cy = x + w / 2.0, y + h / 2.0
        lines = n["lines"]
        ys_txt = [cy + 0.5] if len(lines) == 1 else [cy - lgap / 2.0 + 0.5, cy + lgap / 2.0 + 0.5]
        for ln, ty in zip(lines, ys_txt):
            ax.text(cx, ty, ln.replace("CO\u2082", "CO$_2$"), ha="center", va="center", fontsize=6.45, color="#000000", zorder=5)

    for lg in legends:
        rs = STYLE_CFG["roles"][lg["role"]]
        sx, sy, sw, sh = float(lg["sx"]), float(lg["sy"]), float(lg["sw"]), float(lg["sh"])
        ax.add_patch(MplRectangle((sx, sy), sw, sh, facecolor=rs["fill"], edgecolor="none", zorder=4))
        ax.add_patch(MplRectangle((sx + ins, sy + ins), sw - 2 * ins, sh - 2 * ins, facecolor="none", edgecolor=rs["stroke"], linewidth=0.65, zorder=6))
        ax.text(sx + sw / 2.0, float(lg["ly"]) - 5.5, lg["label"], ha="center", va="center", fontsize=6.30, color="#000000", zorder=5)

    for sub in ("pdf", "svg", "png"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    out_pdf = out_dir / "pdf/Figure1_Study_Framework.pdf"
    out_svg = out_dir / "svg/Figure1_Study_Framework.svg"
    out_png = out_dir / "png/Figure1_Study_Framework.png"
    fig.savefig(out_pdf, format="pdf", facecolor="#FFFFFF")
    fig.savefig(out_svg, format="svg", facecolor="#FFFFFF")
    fig.savefig(out_png, format="png", dpi=450, facecolor="#FFFFFF")
    plt.close(fig)
    print(f"[OK] Exported Figure 1 -> {out_pdf}")
    return {"pdf": str(out_pdf), "svg": str(out_svg), "png": str(out_png)}


if __name__ == "__main__":
    build_figure_1()
