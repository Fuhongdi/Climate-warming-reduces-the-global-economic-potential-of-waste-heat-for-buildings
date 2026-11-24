"""Typography, colour-contrast and basemap utilities for 180 mm two-column journal figures."""
from __future__ import annotations

from pathlib import Path
import matplotlib.colors as mcolors
from matplotlib.text import Text

MM: float = 1.0 / 25.4
SINGLE_COL: float = 88.0 * MM
DOUBLE_COL: float = 180.0 * MM

MIN_PT: float = 5.0
MAX_BODY_PT: float = 10.5
MAX_PANEL_PT: float = 12.0


def load_world_basemap(root_dir: Path):
    """Load cached simplified GADM v4.1 ADM_0 basemap, or generate it on first run from user-downloaded gadm_410-levels.gpkg."""
    import geopandas as gpd

    cache_pq = Path(root_dir) / "data/processed/scenarios_2025/_world_coastline_simplified.parquet"
    if cache_pq.exists():
        return gpd.read_parquet(cache_pq)

    for gp in (
        Path(root_dir) / "data/raw/supplements_2025/gadm41/gadm_410-levels.gpkg",
        Path(root_dir) / "data/raw/gadm_410-levels.gpkg",
    ):
        if gp.exists():
            gdf0 = gpd.read_file(gp, layer="ADM_0")
            gdf0["geometry"] = gdf0["geometry"].simplify(0.20, preserve_topology=True)
            cache_pq.parent.mkdir(parents=True, exist_ok=True)
            out_gdf = gdf0[["GID_0", "geometry"]]
            out_gdf.to_parquet(cache_pq, index=False)
            return out_gdf

    raise FileNotFoundError(
        "GADM v4.1 basemap not found (not redistributed per GADM licence). "
        "Download gadm_410-levels.gpkg from https://gadm.org/ (or run `python -m code.climate.fetch_supplementary_inputs`) "
        "and place it at `data/raw/supplements_2025/gadm41/gadm_410-levels.gpkg`."
    )


def _is_panel_tag(text: str) -> bool:
    s = text.strip()
    return len(s) == 1 and s.isalpha() and s.islower()


def check_typography(fig, placed_mm: float = 180.0, verbose: bool = True):
    """Verify rendered point sizes of all visible Text artists at the target column width."""
    drawn_mm = fig.get_size_inches()[0] / MM
    scale = float(placed_mm) / drawn_mm

    buckets: dict[float, list] = {}
    for t in fig.findobj(Text):
        s = t.get_text()
        if not s or not s.strip():
            continue
        pt = float(t.get_fontsize())
        entry = buckets.setdefault(round(pt, 2), [0, s.strip()])
        entry[0] += 1

    rows = []
    n_fail = 0
    for drawn_pt in sorted(buckets, reverse=True):
        cnt, sample = buckets[drawn_pt]
        final_pt = drawn_pt * scale
        ceil = MAX_PANEL_PT if _is_panel_tag(sample) else MAX_BODY_PT
        if final_pt < MIN_PT or final_pt > MAX_PANEL_PT:
            status = "FAIL"
            n_fail += 1
        elif final_pt > ceil:
            status = "over"
        else:
            status = "ok"
        rows.append((final_pt, drawn_pt, cnt, sample, status))

    if verbose:
        print(f"Figure width {placed_mm:.1f} mm (scale {scale:.3f}): {len(rows)} font-size tiers, {n_fail} out-of-range")
    return rows, n_fail


def _srgb_to_lstar(hex_color: str) -> float:
    r, g, b = mcolors.to_rgb(hex_color)
    def _lin(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    y = 0.2126 * _lin(r) + 0.7152 * _lin(g) + 0.0722 * _lin(b)
    f = y ** (1.0 / 3.0) if y > 0.008856 else (7.787 * y + 16.0 / 116.0)
    return 116.0 * f - 16.0


def check_palette_contrast(colors, labels=None, min_gap: float = 8.0, ordered: bool = False, name: str = ""):
    """Compute perceptual lightness (CIE L*) across palette entries."""
    vals = [_srgb_to_lstar(c) for c in colors]
    names = list(labels) if labels is not None else [str(c) for c in colors]
    close_pairs = []
    for i in range(len(vals)):
        j_range = [i + 1] if ordered and i + 1 < len(vals) else range(i + 1, len(vals))
        for j in j_range:
            gap = abs(vals[i] - vals[j])
            if gap < min_gap:
                close_pairs.append((names[i], names[j], round(gap, 2)))
    if name:
        print(f"Palette {name}: L* range [{min(vals):.1f}, {max(vals):.1f}]")
    return close_pairs
