"""Authoritative physical and techno-economic parameters (2025 & 2050 baselines).

Every number in this module is traced to a primary public engineering source:
  1. Danish Energy Agency (DEA), Technology Data for Generation of Electricity and
     District Heating (Aug 2026 update, native 2025 EUR).
  2. Danish Energy Agency (DEA), Technology Data for Individual Heating Plants
     (Apr 2026 update, native 2025 EUR).
  3. Danish Energy Agency (DEA), Technology Data for Transport of Energy
     (media 7681/7680, Sheet 3.1–3.3 & Feb 2025 DH Distribution Brief, native 2025 EUR).
  4. Fleiter, Manz et al., sEEnergies D5.1: Excess heat potentials of industrial sites in
     Europe (Zenodo 10.5281/zenodo.4785411, conservative 'Full internal heat recovery').
  5. US DOE / NETL, Cost and Performance Baseline for Fossil Energy Plants Vol. 1 Rev. 4b.
  6. Persson et al. (2019, Energy 176, 604–622) urban district heating distribution model.
  7. Staffell et al. (2023, Nature Energy 8, 1328–1344; 2024 correction) BAIT model.
  8. World Bank WDI PA.NUS.FCRF (2025 official annual average EUR/USD exchange rate).
"""
from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# 1. Currency, Discount Rates, and Spatial/Physical Thresholds
# ---------------------------------------------------------------------------
# World Bank PA.NUS.FCRF (EMU/DEU 2025 official annual average): 0.8849689584 EUR per USD
EUR_PER_USD_2025: float = 0.88496895841236
USD_PER_EUR_2025: float = 1.0 / EUR_PER_USD_2025  # 1.1299831372 USD per EUR

DISCOUNT_RATE_CENTRAL: float = 0.07
DISCOUNT_RATE_LOW: float = 0.03
DISCOUNT_RATE_HIGH: float = 0.10

ASSET_LOCKED_LIFETIME_YR: int = 40  # R3 locked evaluation horizon: 2026-2045 + 2046-2065

BAIT_THRESHOLD_CENTRAL_C: float = 14.0
BAIT_THRESHOLD_LOW_C: float = 12.0
BAIT_THRESHOLD_HIGH_C: float = 16.0

MIN_HEAT_DENSITY_GJ_HA_YR: float = 500.0  # Persson et al. (2019) urban DH threshold
MIN_SOURCE_POWER_MW: float = 1.0  # >= 1 MW annual mean thermal power (31,536 GJ/yr)
MIN_ZONE_POWER_MW: float = 1.0  # >= 1 MW annual mean thermal demand (31,536 GJ/yr)

PIPE_DETOUR_FACTOR: float = 1.3
MAX_PIPE_LENGTH_KM: float = 100.0

WASTEWATER_DELTA_T_K: float = 5.0  # DEA Sheet 40 (13 -> 8 C) & Manz et al. (2024)
CRUDE_TONNES_PER_BBL: float = 0.1364  # IEA / BP standard crude oil conversion

# IPCC 2006 Vol. 2 Ch. 2 stationary combustion CO2 emission factors (LHV basis, tCO2/GJ)
NATURAL_GAS_EF_TCO2_PER_GJ_LHV: float = 0.0561


# ---------------------------------------------------------------------------
# 2. DEA Sheet 3.1 District Heating Transmission & 4GDH Multipliers (2025 EUR)
# ---------------------------------------------------------------------------
PIPE_LIFETIME_YR: float = 45.0
STATION_LIFETIME_YR: float = 25.0

# Continuous DEA Note D investment curve matching Sheet 3.1 2025 brackets (EUR_2025 / km):
# Specific cost = 330 * 1.130078 * C_MW^(-0.571) EUR_2025/MW/m => Total = 372_925.74 * C_MW^0.429 EUR_2025/km
PIPE_CAPEX_COEF_EUR_PER_KM: float = 372_925.74
PIPE_CAPEX_EXPONENT: float = 0.429
PIPE_FIXED_OM_EUR_PER_MW_KM_YR: float = 100.0
PIPE_VAR_OM_EUR_PER_MWH_KM: float = 0.116629
PIPE_PUMP_ELEC_MWH_PER_MWH_TH: float = 0.01  # 1.0% pumping electricity of transmitted heat

SOURCE_HEX_STATION_CAPEX_EUR_PER_MW: float = 140_161.82
SOURCE_PUMP_STATION_CAPEX_EUR_PER_MW: float = 127_473.20
SOURCE_STATION_FIXED_OM_FRAC_YR: float = 0.01

# DEA Feb 2025 District Heating Distribution Brief & Note R424 (55/25 C 4GDH vs 70/40 C 3GDH)
LTDH_LOSS_MULTIPLIER: float = 0.775  # -22.5% network heat loss (midpoint of 20-25%)
LTDH_NETWORK_CAPEX_MULTIPLIER: float = 0.95  # -5.0% pipe & distribution CAPEX


def pipe_loss_per_km(capacity_mw: float, is_4gdh: bool = False) -> float:
  """DEA Sheet 3.1 per-km transmission thermal loss fraction l_km(C)."""
  if capacity_mw <= 20.0:
    base = 0.010
  elif capacity_mw <= 100.0:
    base = 0.0025
  else:
    base = 0.0010
  return base * (LTDH_LOSS_MULTIPLIER if is_4gdh else 1.0)


# ---------------------------------------------------------------------------
# 3. DEA Sheet 40 Large Waste-Heat Heat Pumps & Sheet 44/41 Peaking Boilers
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SourceHeatPumpSpec:
  cop: float
  capex_eur_per_mw_h: float
  fixed_om_eur_per_mw_h_yr: float
  var_om_eur_per_mwh_h: float
  lifetime_yr: float = 25.0

  @property
  def kappa(self) -> float:
    """Upgrade ratio kappa = COP / (COP - 1) = delivered heat / extracted waste heat."""
    return self.cop / (self.cop - 1.0)


# DEA Sheet 40: Comp. hp, waste heat 10 MW (power condenser & industrial scrubber ~25-30 C)
HP_INDUSTRIAL_POWER_2025 = SourceHeatPumpSpec(
    cop=4.7,
    capex_eur_per_mw_h=950_000.0,
    fixed_om_eur_per_mw_h_yr=2_126.745436,
    var_om_eur_per_mwh_h=3.07,
    lifetime_yr=25.0,
)
HP_INDUSTRIAL_POWER_2050 = SourceHeatPumpSpec(
    cop=4.8843137254901965,
    capex_eur_per_mw_h=893_855.0,
    fixed_om_eur_per_mw_h_yr=2_001.054781,
    var_om_eur_per_mwh_h=2.888563,
    lifetime_yr=25.0,
)

# DEA Sheet 40: Comp. hp, waste heat 1 MW (wastewater 13 -> 8 C, deltaT = 5 K)
HP_WASTEWATER_2025 = SourceHeatPumpSpec(
    cop=3.1,
    capex_eur_per_mw_h=1_130_000.0,
    fixed_om_eur_per_mw_h_yr=2_126.745436,
    var_om_eur_per_mwh_h=3.07,
    lifetime_yr=25.0,
)
HP_WASTEWATER_2050 = SourceHeatPumpSpec(
    cop=3.2441860465116283,
    capex_eur_per_mw_h=1_063_217.0,
    fixed_om_eur_per_mw_h_yr=2_001.054781,
    var_om_eur_per_mwh_h=2.888563,
    lifetime_yr=25.0,
)


@dataclass(frozen=True)
class PeakingBoilerSpec:
  fuel: str  # "gas" or "electricity"
  efficiency_lhv: float
  aux_elec_mwh_per_mwh_h: float
  capex_eur_per_mw_h: float
  fixed_om_eur_per_mw_h_yr: float
  var_om_non_elec_eur_per_mwh_h: float
  lifetime_yr: float


# DEA Sheet 44: Natural Gas DH Only (condensing economizer)
PEAKING_GAS_BOILER_2025 = PeakingBoilerSpec(
    fuel="gas",
    efficiency_lhv=1.03,
    aux_elec_mwh_per_mwh_h=0.0015,
    capex_eur_per_mw_h=63_802.363,
    fixed_om_eur_per_mw_h_yr=2_126.745436,
    var_om_non_elec_eur_per_mwh_h=1.063373,
    lifetime_yr=25.0,
)
PEAKING_GAS_BOILER_2050 = PeakingBoilerSpec(
    fuel="gas",
    efficiency_lhv=1.04,
    aux_elec_mwh_per_mwh_h=0.0010,
    capex_eur_per_mw_h=53_168.636,
    fixed_om_eur_per_mw_h_yr=1_807.733621,
    var_om_non_elec_eur_per_mwh_h=0.957035,
    lifetime_yr=25.0,
)

# DEA Sheet 41: Electric boiler, large (countries without residential gas network)
PEAKING_ELEC_BOILER_2025 = PeakingBoilerSpec(
    fuel="electricity",
    efficiency_lhv=0.99,
    aux_elec_mwh_per_mwh_h=0.0,
    capex_eur_per_mw_h=70_000.0,
    fixed_om_eur_per_mw_h_yr=1_137.81,
    var_om_non_elec_eur_per_mwh_h=0.84,
    lifetime_yr=20.0,
)
PEAKING_ELEC_BOILER_2050 = PeakingBoilerSpec(
    fuel="electricity",
    efficiency_lhv=0.99,
    aux_elec_mwh_per_mwh_h=0.0,
    capex_eur_per_mw_h=60_000.0,
    fixed_om_eur_per_mw_h_yr=1_020.0,
    var_om_non_elec_eur_per_mwh_h=0.54,
    lifetime_yr=25.0,
)


# ---------------------------------------------------------------------------
# 4. DEA Building Substation & Individual Heating Specs (320 kW_h Ex. Apart)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BuildingHeatingSpec:
  efficiency_or_scop_ref: float
  capex_eur_per_mw_h: float
  fixed_om_eur_per_mw_h_yr: float
  aux_elec_mwh_per_mw_h_yr: float
  lifetime_yr: float


# DEA Sheet 203: DH indirect substation, existing apartment (320 kW_h)
DH_SUBSTATION_2025 = BuildingHeatingSpec(
    efficiency_or_scop_ref=1.00,
    capex_eur_per_mw_h=73_468.75,
    fixed_om_eur_per_mw_h_yr=345.00,
    aux_elec_mwh_per_mw_h_yr=1.71875,
    lifetime_yr=25.0,
)
DH_SUBSTATION_2050 = BuildingHeatingSpec(
    efficiency_or_scop_ref=1.00,
    capex_eur_per_mw_h=67_412.50,
    fixed_om_eur_per_mw_h_yr=303.875,
    aux_elec_mwh_per_mw_h_yr=1.25,
    lifetime_yr=25.0,
)

# DEA Sheet 202: Condensing gas boiler, existing apartment (320 kW_h)
INDIV_GAS_BOILER_2025 = BuildingHeatingSpec(
    efficiency_or_scop_ref=1.01,
    capex_eur_per_mw_h=103_290.625,
    fixed_om_eur_per_mw_h_yr=2_211.40625,
    aux_elec_mwh_per_mw_h_yr=2.03125,
    lifetime_yr=25.0,
)
INDIV_GAS_BOILER_2050 = BuildingHeatingSpec(
    efficiency_or_scop_ref=1.02,
    capex_eur_per_mw_h=91_968.75,
    fixed_om_eur_per_mw_h_yr=1_950.96875,
    aux_elec_mwh_per_mw_h_yr=1.5625,
    lifetime_yr=25.0,
)

# DEA Sheet 207: Air-to-water heat pump, existing apartment (320 kW_h, radiator)
INDIV_ASHP_2025 = BuildingHeatingSpec(
    efficiency_or_scop_ref=2.98889,
    capex_eur_per_mw_h=976_096.875,
    fixed_om_eur_per_mw_h_yr=11_250.0,
    aux_elec_mwh_per_mw_h_yr=1.71875,
    lifetime_yr=20.0,
)
INDIV_ASHP_2050 = BuildingHeatingSpec(
    efficiency_or_scop_ref=4.29333,
    capex_eur_per_mw_h=765_868.75,
    fixed_om_eur_per_mw_h_yr=9_924.96875,
    aux_elec_mwh_per_mw_h_yr=1.25,
    lifetime_yr=20.0,
)


# ---------------------------------------------------------------------------
# 5. sEEnergies D5.1 Industrial Excess Heat Factors (GJ / tonne of activity)
#    Conservative scenario ('Full internal use of excess heat'), cumulative
#    recoverable flue-gas heat cooled down to (25 C, 55 C, 95 C).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class IndustrialHeatFactor:
  unit_basis: str
  q_to_25c_gj_per_t: float
  q_to_55c_gj_per_t: float
  q_to_95c_gj_per_t: float

  @property
  def direct_3gdh_gj_per_t(self) -> float:
    return self.q_to_95c_gj_per_t

  @property
  def hp_3gdh_gj_per_t(self) -> float:
    return max(0.0, self.q_to_25c_gj_per_t - self.q_to_95c_gj_per_t)

  @property
  def direct_4gdh_gj_per_t(self) -> float:
    return self.q_to_55c_gj_per_t

  @property
  def hp_4gdh_gj_per_t(self) -> float:
    return max(0.0, self.q_to_25c_gj_per_t - self.q_to_55c_gj_per_t)


SEENERGIES_FACTORS: dict[str, IndustrialHeatFactor] = {
    # Cement (D5.1 Table 14, per tonne of clinker)
    "cement_dry": IndustrialHeatFactor("t_clinker", 0.51, 0.40, 0.30),
    "cement_wet": IndustrialHeatFactor("t_clinker", 1.09, 0.80, 0.70),
    # Iron & Steel (D5.1 Tables 8-11, per tonne of crude steel; integrated BF-BOF = 0.40*coke + BF + BOF)
    "steel_bf_bof": IndustrialHeatFactor("t_steel", 1.00, 0.66, 0.55),
    "steel_eaf": IndustrialHeatFactor("t_steel", 0.030, 0.015, 0.015),
    # Petrochemical steam cracking (D5.1 Table 15, per tonne of ethylene)
    "ethylene_cracker": IndustrialHeatFactor("t_ethylene", 6.00, 1.50, 0.90),
    # Oil refining (D5.1 Table 15, per tonne of crude throughput; 24%, 6%, 4% of specific fuel)
    "refining_hydroskimming": IndustrialHeatFactor("t_crude", 0.336, 0.084, 0.056),
    "refining_medium": IndustrialHeatFactor("t_crude", 0.432, 0.108, 0.072),
    "refining_deep": IndustrialHeatFactor("t_crude", 0.480, 0.120, 0.080),
    # Pulp & Paper (D5.1 Table 16, per tonne of air-dried pulp or paper)
    "pulp_chemical": IndustrialHeatFactor("t_pulp", 1.10, 0.30, 0.20),
    "pulp_mechanical": IndustrialHeatFactor("t_pulp", 0.10, 0.00, 0.00),
    "paper_board_packaging": IndustrialHeatFactor("t_paper", 0.40, 0.10, 0.10),
    "paper_graphic_tissue": IndustrialHeatFactor("t_paper", 0.50, 0.10, 0.10),
    "pulp_paper_integrated": IndustrialHeatFactor("t_pulp_paper", 0.75, 0.20, 0.15),
}


# ---------------------------------------------------------------------------
# 6. Power Plant Thermodynamic Parameters (DEA Aug 2026 & NETL Rev. 4b, LHV)
#    Q_power = E_net * (1 - eta_u - k_u) / eta_u
#    For CHP extraction plants: direct extraction share up to DEA Cb limit
#    Q_chp_direct_max = E_net / Cb (capped by total Q_power).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PowerTechSpec:
  eta_lhv: float  # Net electrical efficiency (LHV basis)
  k_loss_lhv: float  # Boiler stack + radiation + generator non-condenser loss fraction of fuel LHV
  chp_cb: float | None  # DEA back-pressure P_e / Q_dh ratio when operated as CHP (None if non-CHP)
  recoverable_cooling: bool  # False for open-cycle GT, reciprocating ICE, or air-cooled ACC

  @property
  def reject_heat_per_mwh_e(self) -> float:
    if not self.recoverable_cooling:
      return 0.0
    return max(0.0, (1.0 - self.eta_lhv - self.k_loss_lhv) / self.eta_lhv)


POWER_TECH_SPECS: dict[str, PowerTechSpec] = {
    # Coal steam turbines (DEA Sheet 01 & NETL Rev.4b B11A/B12A converted HHV->LHV by *1.05)
    "coal_usc": PowerTechSpec(eta_lhv=0.450, k_loss_lhv=0.100, chp_cb=0.84, recoverable_cooling=True),
    "coal_sc": PowerTechSpec(eta_lhv=0.422, k_loss_lhv=0.105, chp_cb=0.78, recoverable_cooling=True),
    "coal_subc": PowerTechSpec(eta_lhv=0.385, k_loss_lhv=0.110, chp_cb=0.75, recoverable_cooling=True),
    "coal_cfb": PowerTechSpec(eta_lhv=0.380, k_loss_lhv=0.110, chp_cb=0.75, recoverable_cooling=True),
    "coal_igcc": PowerTechSpec(eta_lhv=0.420, k_loss_lhv=0.120, chp_cb=0.84, recoverable_cooling=True),
    # Natural gas & oil turbines (DEA Sheet 05, 51, 52 & NETL Rev.4b B31A/B32A)
    # Note: CCGT reject heat is steam bottoming condenser (~1/3 of fuel input -> ~0.55 MWh_th/MWh_e)
    "gas_ccgt": PowerTechSpec(eta_lhv=0.550, k_loss_lhv=0.150, chp_cb=1.70, recoverable_cooling=True),
    "gas_steam": PowerTechSpec(eta_lhv=0.390, k_loss_lhv=0.110, chp_cb=0.75, recoverable_cooling=True),
    "oil_steam": PowerTechSpec(eta_lhv=0.380, k_loss_lhv=0.110, chp_cb=0.75, recoverable_cooling=True),
    # Open-cycle gas turbines & internal combustion engines: exhaust to air, no steam condenser loop
    # (unless explicitly tagged as CHP with exhaust heat recovery boiler, handled via chp_cb!)
    "gas_ocgt": PowerTechSpec(eta_lhv=0.390, k_loss_lhv=0.610, chp_cb=0.95, recoverable_cooling=False),
    "gas_ice": PowerTechSpec(eta_lhv=0.450, k_loss_lhv=0.550, chp_cb=0.95, recoverable_cooling=False),
    "oil_ocgt_ice": PowerTechSpec(eta_lhv=0.400, k_loss_lhv=0.600, chp_cb=0.95, recoverable_cooling=False),
    # Bioenergy & Waste-to-Energy (DEA Sheet 08 & 09a/09b)
    "biomass_steam": PowerTechSpec(eta_lhv=0.320, k_loss_lhv=0.120, chp_cb=0.44, recoverable_cooling=True),
    "wte_steam": PowerTechSpec(eta_lhv=0.235, k_loss_lhv=0.140, chp_cb=0.30, recoverable_cooling=True),
    # Nuclear default when reactor-specific Thermal Capacity MWt is missing (PRIS/GEM gives exact eta_u)
    "nuclear_lwr": PowerTechSpec(eta_lhv=0.330, k_loss_lhv=0.020, chp_cb=None, recoverable_cooling=True),
}
