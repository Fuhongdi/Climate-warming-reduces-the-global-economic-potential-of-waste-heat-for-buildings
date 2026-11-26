#!/usr/bin/env python3
"""Download and verify public supplementary input datasets (energy prices, grid factors, administrative boundaries)."""
from __future__ import annotations
import os

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import zipfile

ROOT = Path(os.environ.get("WASTE_HEAT_ROOT", Path(__file__).resolve().parents[2]))
OUT_RAW = ROOT / "data/raw/supplements_2025"
OUT_RAW.mkdir(parents=True, exist_ok=True)


def check_disk(min_gib: float = 20.0) -> float:
  free_gib = shutil.disk_usage(ROOT).free / (1024**3)
  print(f"[Disk free space]: {free_gib:.1f} GiB", flush=True)
  if free_gib < min_gib:
    raise RuntimeError(f"Insufficient disk free space: {free_gib:.1f} GiB (< {min_gib:.1f} GiB)")
  return free_gib


def sha256_file(p: Path) -> str:
  h = hashlib.sha256()
  with p.open("rb") as f:
    for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
      h.update(b)
  return h.hexdigest()


def curl_download(
    url: str,
    dest: Path,
    timeout_s: int = 1800,
    min_bytes: int = 200,
    referer: str | None = None,
) -> dict:
  dest.parent.mkdir(parents=True, exist_ok=True)
  if dest.exists() and dest.stat().st_size >= max(min_bytes, 500):
    sz = dest.stat().st_size
    print(f"  [SKIP existing] {dest.name} ({sz / 1e6:.2f} MB)", flush=True)
    return {
        "url": url,
        "path": str(dest.relative_to(ROOT)),
        "bytes": sz,
        "sha256": sha256_file(dest),
        "reused": True,
    }

  tmp = dest.with_name(dest.name + ".part")
  proxy_url = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY") or os.environ.get("https_proxy") or os.environ.get("http_proxy")
  modes = (["proxy", "direct"] if proxy_url else ["direct"])
  for attempt in range(1, 7):
    mode = modes[(attempt - 1) % len(modes)]
    cmd = ["curl", "--disable"]
    if mode == "proxy" and proxy_url:
      cmd += ["--proxy", proxy_url]
    cmd += [
        "-L",
        "--fail",
        "--silent",
        "--show-error",
        "--connect-timeout",
        "30",
        "--speed-limit",
        "65536",
        "--speed-time",
        "60",
        "--max-time",
        str(timeout_s),
        "-C",
        "-",
        "-H",
        "Accept: */*",
        "-A",
        "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0",
    ]
    if referer:
      cmd += ["-e", referer]
    cmd += ["-o", str(tmp), url]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0 and tmp.exists() and tmp.stat().st_size >= min_bytes:
      tmp.replace(dest)
      sz = dest.stat().st_size
      digest = sha256_file(dest)
      print(f"  [OK via {mode} attempt {attempt}] {dest.name}: {sz / 1e6:.2f} MB", flush=True)
      return {
          "url": url,
          "path": str(dest.relative_to(ROOT)),
          "bytes": sz,
          "sha256": digest,
          "mode": mode,
          "utc": datetime.now(timezone.utc).isoformat(),
      }
    cur_sz = tmp.stat().st_size / 1e6 if tmp.exists() else 0.0
    print(
        f"  [{mode} attempt {attempt}/6 for {dest.name} (part={cur_sz:.1f} MB)]: rc={res.returncode} {res.stderr.strip()[:160]}",
        flush=True,
    )
    # Stop immediately on hard 404/403
    if res.returncode == 22 and any(code in res.stderr for code in [" 404", " 403", " 400"]):
      if tmp.exists() and tmp.stat().st_size < 1024:
        tmp.unlink()
      break
  raise RuntimeError(f"Failed to download {url}")


def main() -> None:
  check_disk()
  manifest: list[dict] = []
  errors: list[dict] = []

  def safe_fetch(section: str, url: str, dest: Path, **kwargs) -> dict | None:
    try:
      rec = curl_download(url, dest, **kwargs)
      manifest.append(rec)
      return rec
    except Exception as e:
      print(f"  [ERROR in {section}]: {e}", flush=True)
      errors.append({"section": section, "url": url, "error": str(e)})
      return None

  # 1. DEA Technology Data for Transport of Energy
  print("1. Verifying DEA Energy Transport files...", flush=True)
  for existing_p in sorted((OUT_RAW / "dea_transport").glob("*")):
    if existing_p.is_file() and existing_p.suffix in (".xlsx", ".pdf"):
      manifest.append({
          "url": "https://ens.dk/en/analyses-and-statistics/technology-data-transport-energy",
          "path": str(existing_p.relative_to(ROOT)),
          "bytes": existing_p.stat().st_size,
          "sha256": sha256_file(existing_p),
          "reused": True,
      })

  # 2. Eurostat household end-use & 2025 energy prices & FX/HICP
  print("2. Verifying Eurostat household end-use & 2025 energy prices...", flush=True)
  euro_dir = OUT_RAW / "eurostat"
  for code in [
      "nrg_d_hhq",
      "nrg_pc_204",
      "nrg_pc_205",
      "nrg_pc_202",
      "nrg_pc_203",
      "ert_bil_eur_a",
      "prc_hicp_aind",
  ]:
    tsv_url = f"https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/{code}?format=TSV&compressed=true"
    safe_fetch("eurostat", tsv_url, euro_dir / f"{code}.tsv.gz", timeout_s=300)

  # 3. IMF Fossil Fuel Subsidies & EIA Prices
  print("3. Verifying IMF Fossil Fuel Subsidies & US EIA 2025 Energy Prices...", flush=True)
  imf_dir = OUT_RAW / "imf_prices"
  safe_fetch(
      "imf_arcgis_csv",
      "https://opendata.arcgis.com/api/v3/datasets/d48cfd2124954fb0900cef95f2db2724_0/downloads/data?format=csv&spatialRefId=4326&where=1%3D1",
      imf_dir / "imf_fossil_fuel_subsidies_climatedata.csv",
      timeout_s=300,
      min_bytes=1000,
  )
  safe_fetch(
      "imf_wb_data360",
      "https://data360files.worldbank.org/data360-data/data/IMF_FFS/IMF_FFS_WIDEF.csv",
      imf_dir / "WB_Data360_IMF_FFS_WIDEF.csv",
      timeout_s=300,
      min_bytes=1000,
  )
  for tbl in ["T09.08", "T09.10"]:
    safe_fetch(
        "eia_prices",
        f"https://www.eia.gov/totalenergy/data/browser/xls.php?tbl={tbl}&freq=m",
        OUT_RAW / "eia_prices" / f"EIA_{tbl}_monthly.xlsx",
        timeout_s=180,
    )

  # 4. Ember Yearly Electricity Data (2025 release)
  print("4. Verifying Ember Yearly Electricity Data CSV...", flush=True)
  safe_fetch(
      "ember",
      "https://files.ember-energy.org/public-downloads/generation/outputs/release_generation_yearly_global.csv",
      OUT_RAW / "ember/release_generation_yearly_global.csv",
      timeout_s=600,
      min_bytes=1_000_000,
  )

  # 5. World Bank WDI PPP, Exchange Rates, CPI/Deflator, Electricity Tariff & Pink Sheet
  print("5. Verifying World Bank WDI PPP, exchange rates, Pink Sheet & electricity tariff...", flush=True)
  wb_dir = OUT_RAW / "worldbank"
  for ind in [
      "PA.NUS.PPP",
      "PA.NUS.PRVT.PP",
      "PA.NUS.FCRF",
      "FP.CPI.TOTL",
      "NY.GDP.DEFL.ZS",
  ]:
    wb_url = f"https://api.worldbank.org/v2/country/all/indicator/{ind}?format=json&per_page=20000&date=2015:2025"
    safe_fetch("worldbank_wdi", wb_url, wb_dir / f"{ind}_2015_2025.json", timeout_s=180)
  safe_fetch(
      "worldbank_el_tariff",
      "https://api.worldbank.org/v2/country/all/indicator/IC.ELC.PRI.KH.DB1619?format=json&per_page=20000",
      wb_dir / "IC.ELC.PRI.KH.DB1619.json",
      timeout_s=180,
  )
  if (wb_dir / "CMO-Historical-Data-Monthly.xlsx").exists():
    p_ps = wb_dir / "CMO-Historical-Data-Monthly.xlsx"
    manifest.append({
        "url": "https://www.worldbank.org/en/research/commodity-markets",
        "path": str(p_ps.relative_to(ROOT)),
        "bytes": p_ps.stat().st_size,
        "sha256": sha256_file(p_ps),
        "reused": True,
    })

  # 6. CMIP6 / IPCC AR6 Global Warming Levels
  print("6. Verifying CMIP6 Global Warming Level tables...", flush=True)
  safe_fetch(
      "cmip6_gwl",
      "https://codeload.github.com/mathause/cmip_warming_levels/tar.gz/refs/heads/main",
      OUT_RAW / "cmip6_warming_levels/cmip_warming_levels_main.tar.gz",
      timeout_s=300,
  )

  # 7. GADM 4.1 Global Administrative Boundaries (ADM_0 / ADM_1 / ADM_2)
  print("7. Fetching GADM 4.1 global levels GeoPackage zip (~2.5 GB)...", flush=True)
  gadm_zip = OUT_RAW / "gadm41/gadm_410-levels.zip"
  gadm_gpkg = OUT_RAW / "gadm41/gadm_410-levels.gpkg"
  safe_fetch(
      "gadm41",
      "https://geodata.ucdavis.edu/gadm/gadm4.1/gadm_410-levels.zip",
      gadm_zip,
      timeout_s=1800,
      min_bytes=1_000_000_000,
  )
  if gadm_zip.exists() and not gadm_gpkg.exists():
    with zipfile.ZipFile(gadm_zip, "r") as zf:
      zf.extractall(gadm_zip.parent)

  # 8. GHSL E2020 100 m Built-up Volume Rasters (for Robustness Check 4)
  print("8. Fetching GHSL E2020 100 m total & non-residential volume rasters (Robustness 4)...", flush=True)
  ghsl20_dir = OUT_RAW / "ghsl_e2020"
  for fn, u in [
      (
          "GHS_BUILT_V_E2020_GLOBE_R2023A_54009_100_V1_0.zip",
          "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_BUILT_V_GLOBE_R2023A/"
          "GHS_BUILT_V_E2020_GLOBE_R2023A_54009_100/V1-0/GHS_BUILT_V_E2020_GLOBE_R2023A_54009_100_V1_0.zip",
      ),
      (
          "GHS_BUILT_V_NRES_E2020_GLOBE_R2023A_54009_100_V1_0.zip",
          "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_BUILT_V_GLOBE_R2023A/"
          "GHS_BUILT_V_NRES_E2020_GLOBE_R2023A_54009_100/V1-0/GHS_BUILT_V_NRES_E2020_GLOBE_R2023A_54009_100_V1_0.zip",
      ),
  ]:
    safe_fetch("ghsl_e2020", u, ghsl20_dir / fn, timeout_s=1800, min_bytes=50_000_000)

  # 9. Hotmaps EU 100 m Heat Density Repository (Validation 2)
  print("9. Fetching Hotmaps EU heat_tot_curr_density repository (Validation 2)...", flush=True)
  hm_dir = OUT_RAW / "hotmaps_heat_density"
  for hm_fn, hm_u in [
      ("datapackage.json", "https://gitlab.com/hotmaps/heat/heat_tot_curr_density/-/raw/master/datapackage.json"),
      ("README.md", "https://gitlab.com/hotmaps/heat/heat_tot_curr_density/-/raw/master/README.md"),
      ("heat_tot_curr_density.tif", "https://gitlab.com/hotmaps/heat/heat_tot_curr_density/-/raw/master/data/heat_tot_curr_density.tif"),
  ]:
    safe_fetch("hotmaps", hm_u, hm_dir / hm_fn, timeout_s=900)

  manifest_path = OUT_RAW / "MANIFEST_supplements_2025.json"
  manifest_path.write_text(
      json.dumps(
          {
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "count": len(manifest),
              "error_count": len(errors),
              "files": manifest,
              "errors": errors,
          },
          ensure_ascii=False,
          indent=2,
      )
      + "\n",
      encoding="utf-8",
  )
  print(
      f"SUPPLEMENTARY INPUTS VERIFIED: {len(manifest)} files verified, {len(errors)} warnings.",
      flush=True,
  )


if __name__ == "__main__":
  main()
