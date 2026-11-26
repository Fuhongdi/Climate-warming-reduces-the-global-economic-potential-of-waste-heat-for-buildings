"""Automated verification of core physical and economic equations in Methods."""
from pathlib import Path
import math
import numpy as np
import pandas as pd
import pytest
from code.common.paths import PROJECT_ROOT
from code.matching.solver import TechConfig, solve_country_celf, solve_country_upper_bound_lp


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


def test_360_state_shared_source_conservation():
    # Verify 3 zones competing for 1 shared source never exceed 360-state physical LP upper bound U
    q_init = {"S1": np.full(12, 1000.0, dtype=np.float64)}
    q_prof = np.linspace(100.0, 2200.0, 30)  # spiky winter cold-day demand
    dmq = np.vstack([q_prof if m in (0, 1, 11) else q_prof * 0.15 for m in range(12)])
    zone_dmq = {f"Z{k}": dmq.copy() for k in range(3)}
    ann_gj = float(np.sum(dmq.mean(axis=1) * np.array([31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])))
    zone_meta = {
        f"Z{k}": {
            "ann_gj": ann_gj,
            "peak_gj_d": 2200.0,
            "lambda_dist": 0.08,
            "c_ind_usd_per_gj": 28.0,
            "cf_co2_t_per_gj": 0.056,
            "c_net_usd_per_gj": 6.0,
            "p_elec_usd_per_mwh": 120.0,
            "p_gas_usd_per_gj": 14.0,
            "has_gas": 1.0,
            "grid_ef_tco2_per_mwh": 0.35,
            "cap_scale_civil": 1.0,
            "cap_scale_equip": 1.0,
        }
        for k in range(3)
    }
    df_cands = pd.DataFrame([
        {"stream_id": "S1", "zone_id": f"Z{k}", "ell_km": 3.0 + k, "upgrade_mode": "direct_hex", "sector": "power", "subsector_detail": "power_coal"}
        for k in range(3)
    ])
    cfg = TechConfig()
    sel, p_gj = solve_country_celf(df_cands, q_init, zone_dmq, zone_meta, cfg)
    u_gj, _ = solve_country_upper_bound_lp(df_cands, q_init, zone_dmq, zone_meta, cfg)
    assert len(sel) >= 1
    assert p_gj <= u_gj + 1e-6


def test_price_non_inversion():
    rp_path = PROJECT_ROOT / "data/processed/sinks_2025/calibrated_region_parameters_2025.parquet"
    if not rp_path.exists():
        pytest.skip("Processed dataset not unpacked in repository root")
    df = pd.read_parquet(rp_path)
    dv = df[df["valid_for_study"]]
    assert (dv["p_elec_pretax_usd_per_mwh"] <= dv["p_elec_retail_usd_per_mwh"] + 1e-9).all()
    assert (dv["p_gas_pretax_usd_per_gj"] <= dv["p_gas_retail_usd_per_gj"] + 1e-9).all()


def test_additive_decomposition_identity():
    # Methods Eqs. (9)-(12): g_D + g_M + g_E == (P1 - P0) / P0 identically
    dec_path = PROJECT_ROOT / "data/processed/scenarios_2025/r2_decomposition/r2_eq6_decomposition_all.parquet"
    if not dec_path.exists():
        pytest.skip("Processed dataset not unpacked in repository root")
    df = pd.read_parquet(dec_path)
    lhs = df["term1_demand_shift_pct"] + df["term2_seasonal_mismatch_pct"] + df["term3_cost_exit_pct"]
    diff = (lhs - df["dP_rel_pct"]).abs().max()
    assert diff < 1e-9
