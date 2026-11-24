"""Automated verification of core physical and economic equations in Methods."""
from pathlib import Path
import math
import pandas as pd
from code.common.paths import PROJECT_ROOT


def test_project_root_resolves():
    assert PROJECT_ROOT.exists()
    assert (PROJECT_ROOT / "code").is_dir()


def test_capital_recovery_factor_and_pit_storage_cost():
    # Methods Eq. (4) & Seasonal thermal storage (24.2 USD GJ^-1 at r=7%, L=35 yr, m=1%, eta=0.70, I=194.6 USD GJ^-1)
    r, L = 0.07, 35
    crf = r * (1 + r) ** L / ((1 + r) ** L - 1)
    assert abs(crf - 0.077234) < 1e-5
    c_ptes = (crf + 0.01) * (172.2 * 1.13) / 0.70
    assert round(c_ptes, 1) == 24.2


def test_additive_decomposition_identity():
    # Methods Eqs. (9)-(12): g_D + g_M + g_E == (P1 - P0) / P0 identically
    dec_path = PROJECT_ROOT / "data/processed/scenarios_2025/r2_decomposition/r2_eq6_decomposition_all.parquet"
    if dec_path.exists():
        df = pd.read_parquet(dec_path)
        lhs = df["term1_demand_shift_pct"] + df["term2_seasonal_mismatch_pct"] + df["term3_cost_exit_pct"]
        diff = (lhs - df["dP_rel_pct"]).abs().max()
        assert diff < 1e-9
