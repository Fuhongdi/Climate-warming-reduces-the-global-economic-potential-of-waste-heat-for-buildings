"""Canonical 20-year climate windows, ISIMIP3b ensemble definitions, and calendar helpers."""
from __future__ import annotations

from dataclasses import dataclass

OBSERVED_DATASET = "gswp3-era5__obsclim"

ISIMIP3B_MODELS: tuple[str, ...] = (
    "gfdl-esm4",
    "ipsl-cm6a-lr",
    "mpi-esm1-2-hr",
    "mri-esm2-0",
    "ukesm1-0-ll",
)

SSP_SCENARIOS: tuple[str, ...] = (
    "ssp126",
    "ssp245",
    "ssp370",
)


@dataclass(frozen=True)
class ClimateWindow:
  """Twenty-year climate evaluation window (inclusive start and end years)."""

  window_id: str
  start_year: int
  end_year: int
  kind: str  # "observed" or "projected"
  label_en: str

  @property
  def years(self) -> tuple[int, ...]:
    return tuple(range(self.start_year, self.end_year + 1))


BASELINE_WINDOW = ClimateWindow(
    window_id="2006_2025",
    start_year=2006,
    end_year=2025,
    kind="observed",
    label_en="2006–2025 (Baseline)",
)

OBSERVED_WINDOWS: tuple[ClimateWindow, ...] = (
    ClimateWindow("1946_1965", 1946, 1965, "observed", "1946–1965"),
    ClimateWindow("1966_1985", 1966, 1985, "observed", "1966–1985"),
    ClimateWindow("1986_2005", 1986, 2005, "observed", "1986–2005"),
    BASELINE_WINDOW,
)

PROJECTED_WINDOWS: tuple[ClimateWindow, ...] = (
    ClimateWindow("2026_2045", 2026, 2045, "projected", "2026–2045 (Near-term)"),
    ClimateWindow("2046_2065", 2046, 2065, "projected", "2046–2065 (Mid-century)"),
    ClimateWindow("2066_2085", 2066, 2085, "projected", "2066–2085"),
    ClimateWindow("2081_2100", 2081, 2100, "projected", "2081–2100 (End-of-century)"),
)

ALL_EIGHT_WINDOWS: tuple[ClimateWindow, ...] = OBSERVED_WINDOWS + PROJECTED_WINDOWS
