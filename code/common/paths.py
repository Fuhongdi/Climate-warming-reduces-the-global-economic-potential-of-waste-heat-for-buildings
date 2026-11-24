"""Canonical paths, physical constants, and capital recovery factor (CRF) helpers."""
from __future__ import annotations
import os

from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("WASTE_HEAT_ROOT", Path(__file__).resolve().parents[2]))

DATA_RAW = PROJECT_ROOT / "data/raw"
DATA_INTERIM = PROJECT_ROOT / "data/interim"
DATA_PROCESSED = PROJECT_ROOT / "data/processed"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
WORK_DIR = PROJECT_ROOT / "work"
LOGS_DIR = PROJECT_ROOT / "logs"

# Physical & unit conversion constants
GJ_PER_MWH = 3.6
MWH_PER_GJ = 1.0 / 3.6
SECONDS_PER_DAY = 86_400.0
GJ_PER_MW_DAY = 86.4  # 1 MW * 86,400 s = 86.4 GJ/day
WATER_VOLUMETRIC_HEAT_CAPACITY_MJ_M3_K = 4.186  # rho * c_p ~ 1000 kg/m3 * 4.186 kJ/(kg K) = 4.186 MJ/(m3 K)


def crf(discount_rate: float, lifetime_years: float) -> float:
  """Capital recovery factor a(r, L) = r*(1+r)^L / ((1+r)^L - 1)."""
  if lifetime_years <= 0:
    raise ValueError(f"lifetime_years must be > 0, got {lifetime_years}")
  if abs(discount_rate) < 1e-12:
    return 1.0 / lifetime_years
  factor = (1.0 + discount_rate) ** lifetime_years
  return discount_rate * factor / (factor - 1.0)
