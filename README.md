# Climate warming undermines waste heat for urban heating decarbonization

**Authors:** Hongdi Fu & Haoran Zhang (Peking University)  
**Version:** 1.0.0

This repository provides the modelling, optimization, climate-experiment and figure-generation code for the study *"Climate warming undermines waste heat for urban heating decarbonization"*.

---

## Overview

The modelling framework links 27,481 geolocated power, heavy industrial and municipal wastewater facilities with 64,653 dense urban heating areas at 100 m resolution across daily observed (GSWP3-ERA5, 1946–2025) and projected (five ISIMIP3b CMIP6 climate models under SSP1-2.6, SSP2-4.5 and SSP3-7.0, 2026–2100) weather sequences.

```text
.
├── code/
│   ├── common/       # Physical/economic parameters (constant 2025 USD), climate windows & paths
│   ├── sources/      # 2025 global facility waste heat inventory & statistical validation
│   ├── sinks/        # GHSL R2023A 100 m urban heating areas, demand calibration & tariffs
│   ├── climate/      # Daily BAIT simulation (1946–2100) & monthly additive bias alignment
│   ├── matching/     # National HiGHS LP technical potential, priority-queue screening & MILP benchmark
│   ├── scenarios/    # Global historical, SSP projection, warming-level, transition & sensitivity runner
│   └── figures/      # Figure 1–5, Extended Data Fig. 1–4, Supplementary Fig. 1–8 & Excel builders
├── tests/            # Unit tests for core equations and decomposition identities
├── requirements.txt  # Python scientific and HiGHS solver dependencies
└── LICENSE           # MIT License
```

---

## Installation

Python 3.11+ (tested with Python 3.13) and the open-source [HiGHS](https://highs.dev/) optimization solver (`highspy >= 1.8`) are required.

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

---

## Data Setup & Reproduction

1. Download the companion processed data archive (`Zenodo_Data_v1.0.0.zip`) from Zenodo and extract `data/` and `figures/source_data/` into the repository root (or point `WASTE_HEAT_ROOT` to the extracted directory). For global map basemaps (Fig. 3a, Extended Data Fig. 4a,b and Supplementary Figs. 1a and 8a) and upstream municipal aggregation, download GADM v4.1 (`gadm_410-levels.gpkg`, not redistributed per GADM licence terms) directly from https://gadm.org/ (or via `python -m code.climate.fetch_supplementary_inputs`) to `data/raw/supplements_2025/gadm41/gadm_410-levels.gpkg`; `load_world_basemap()` generates the simplified basemap cache automatically on first run.
2. Run the automated verification tests:
   ```bash
   pytest
   ```
3. Reproduce all manuscript figures, Extended Data figures, Supplementary figures and Excel workbooks directly from the processed datasets:
   ```bash
   python code/figures/build_figure1_framework.py
   python code/figures/build_main_figures.py
   python code/figures/build_extended_figures.py
   python code/figures/build_si_figures_and_tables.py
   ```
4. To rerun the upstream data preparation and all 190 global scenario simulations from primary inputs:
   ```bash
   python -m code.sources.build_sources_2025
   python -m code.sinks.calibrate_demand_and_prices
   python -m code.sinks.build_urban_heat_zones
   python -m code.climate.complete_bait_and_alignment
   python -m code.scenarios.run_global_scenarios
   ```
