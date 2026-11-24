#!/usr/bin/env python3
"""Resolve and download the 40 ISIMIP3b historical (2001-2014) daily NetCDF files with SHA-512 verification."""
from __future__ import annotations

import datetime
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys

PROJECT = Path(os.environ.get("WASTE_HEAT_ROOT", Path(__file__).resolve().parents[2]))
WORK = PROJECT / "work/isimip3b_historical_2001_2014"
STORAGE = PROJECT / "data/raw/isimip3b_daily"

MODELS = [
    "gfdl-esm4",
    "ipsl-cm6a-lr",
    "mpi-esm1-2-hr",
    "mri-esm2-0",
    "ukesm1-0-ll",
]
VARIABLES = ["tas", "huss", "sfcwind", "rsds"]
CHUNKS = [(2001, 2010), (2011, 2014)]


def utc() -> str:
  return datetime.datetime.now(datetime.timezone.utc).isoformat()


def fetch_json_via_curl(url: str) -> dict:
  for mode in ["proxy", "direct"]:
    cmd = ["curl", "--disable"]
    if mode == "proxy":
      cmd += ["--proxy", "http://127.0.0.1:7897", "--noproxy", ""]
    else:
      cmd += ["--noproxy", "*"]
    cmd += [
        "-L",
        "--fail",
        "--silent",
        "--show-error",
        "--connect-timeout",
        "30",
        "--max-time",
        "60",
        "-A",
        "NatureEnergyPipeline/1.0",
        url,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0 and res.stdout.strip():
      return json.loads(res.stdout)
  raise RuntimeError(f"Failed API query: {url}")


def build_manifest() -> tuple[Path, str]:
  WORK.mkdir(parents=True, exist_ok=True)
  meta_dir = WORK / "publisher_metadata"
  meta_dir.mkdir(exist_ok=True)
  manifest_path = WORK / "download_manifest.json"
  sha_path = WORK / "download_manifest.sha256"

  if manifest_path.exists() and sha_path.exists():
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if digest == sha_path.read_text().strip():
      data = json.loads(manifest_path.read_text())
      if len(data.get("files", [])) == 40:
        print(
            f"[Reusing verified manifest]: 40 files, {data['total_bytes'] / 1e9:.2f} GB",
            flush=True,
        )
        return manifest_path, digest

  print("Resolving 40 ISIMIP3b historical (2001-2014) files via data.isimip.org API...", flush=True)
  rows: list[dict] = []
  for model in MODELS:
    api_url = (
        "https://data.isimip.org/api/v1/files/"
        f"?simulation_round=ISIMIP3b&product=InputData&climate_scenario=historical"
        f"&climate_forcing={model}&bias_adjustment=w5e5&time_step=daily&region=global&page_size=200"
    )
    cache_p = meta_dir / f"{model}_historical.json"
    if cache_p.exists():
      payload = json.loads(cache_p.read_text())
    else:
      payload = fetch_json_via_curl(api_url)
      # Follow pagination if any
      results = list(payload.get("results", []))
      nxt = payload.get("next")
      while nxt:
        page = fetch_json_via_curl(nxt)
        results.extend(page.get("results", []))
        nxt = page.get("next")
      payload = {"results": results}
      cache_p.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    results = payload["results"]
    for var in VARIABLES:
      for sy, ey in CHUNKS:
        matches = [
            r
            for r in results
            if r.get("specifiers", {}).get("climate_variable") == var
            and r.get("specifiers", {}).get("start_year") == sy
            and r.get("specifiers", {}).get("end_year") == ey
        ]
        if len(matches) != 1:
          raise RuntimeError(f"Expected 1 match for {model} {var} {sy}_{ey}, got {len(matches)}")
        item = matches[0]
        assert item["checksum_type"] == "sha512"
        assert re.fullmatch(r"[0-9a-f]{128}", item["checksum"])
        ver = "v" + str(item["version"]).lstrip("v")
        rel_target = f"{model}/historical/{ver}/{item['name']}"
        rows.append({
            "id": item["id"],
            "name": item["name"],
            "file_url": item["file_url"],
            "size": item["size"],
            "checksum": item["checksum"],
            "checksum_type": "sha512",
            "version": item["version"],
            "model": model,
            "scenario": "historical",
            "variable": var,
            "start_year": sy,
            "end_year": ey,
            "target": rel_target,
            "group": "historical_2001_2014",
        })

  assert len(rows) == 40, f"Expected 40 files, got {len(rows)}"
  assert len({r["id"] for r in rows}) == 40
  rows.sort(key=lambda r: (r["model"], r["start_year"], r["variable"]))
  total_bytes = sum(r["size"] for r in rows)
  manifest = {
      "created_utc": utc(),
      "project_root": str(PROJECT),
      "physical_storage": str(STORAGE),
      "description": "ISIMIP3b historical (2001-2014) daily meteorological files (5 GCMs x 4 vars x 2 chunks)",
      "total_files": 40,
      "total_bytes": total_bytes,
      "files": rows,
  }
  manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
  digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
  sha_path.write_text(digest + "\n", encoding="utf-8")
  print(f"Resolved 40 files ({total_bytes / 1e9:.2f} GB), sha256={digest}", flush=True)
  return manifest_path, digest


def main() -> int:
  manifest_path, digest = build_manifest()
  spec = importlib.util.spec_from_file_location(
      "hist_transfer_engine", PROJECT / "code/climate/download_engine.py"
  )
  assert spec and spec.loader
  engine = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(engine)

  engine.ROOT = STORAGE
  engine.WORK = WORK
  engine.MANIFEST = manifest_path
  engine.MANIFEST_SHA256 = digest
  engine.MAX_WORKERS = 8
  engine.RESERVE = 100 * 1024**3

  def run_curl(row: dict, partial: Path):
    attempt = engine.RETRIES[row["id"]]["attempts"]
    use_proxy = attempt % 3 != 0
    network = (
        ["--proxy", "http://127.0.0.1:7897", "--noproxy", ""]
        if use_proxy
        else ["--noproxy", "*"]
    )
    with engine.GUARD:
      engine.ACTIVE[row["id"]]["route"] = "http_proxy" if use_proxy else "direct"
    command = [
        "curl",
        "--disable",
        *network,
        "--proto",
        "=https",
        "--proto-redir",
        "=https",
        "--location",
        "--fail",
        "--silent",
        "--show-error",
        "--connect-timeout",
        "30",
        "--speed-limit",
        "131072",
        "--speed-time",
        "180",
        "--retry",
        "0",
        "--continue-at",
        "-",
        "--output",
        str(partial),
        "--write-out",
        "%{http_code}",
        row["file_url"],
    ]
    with (WORK / "transfer_logs" / (row["id"] + ".log")).open("a") as log:
      with engine.GUARD:
        if engine.STOP.is_set():
          return None, None
        process = subprocess.Popen(
            command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=log
        )
        engine.PROCESSES[row["id"]] = process
      try:
        output, _ = process.communicate()
        raw = output.decode("ascii", errors="replace").strip()
        return process.returncode, int(raw) if raw.isdigit() else None
      finally:
        with engine.GUARD:
          engine.PROCESSES.pop(row["id"], None)

  engine.run_curl = run_curl

  STORAGE.mkdir(parents=True, exist_ok=True)
  lock = (WORK / "download.lock").open("w")
  try:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
  except BlockingIOError:
    print("Existing historical downloader already holds the lock", flush=True)
    return 0

  engine.atomic(WORK / "pid.json", {"pid": os.getpid(), "started_utc": engine.utc()})
  for name in ["verified", "transfer_logs"]:
    (WORK / name).mkdir(exist_ok=True)

  manifest = json.loads(manifest_path.read_text())
  engine.ROWS = manifest["files"]
  assert len(engine.ROWS) == 40

  saved = WORK / "retry_state.json"
  if saved.exists():
    prior = json.loads(saved.read_text())
    if prior.get("manifest_sha256") == digest:
      engine.RETRIES.update(prior["records"])

  for row in engine.ROWS:
    target = engine.target(row)
    part = engine.partial(target)
    engine.INITIAL += min(engine.safe_size(target) or engine.safe_size(part), row["size"])

  remaining = sum(row["size"] for row in engine.ROWS) - engine.INITIAL
  if shutil.disk_usage(STORAGE).free < remaining + engine.RESERVE:
    engine.event("insufficient_space_at_start", remaining=remaining)
    engine.snapshot("blocked_disk_space")
    return 2

  signal.signal(signal.SIGTERM, engine.stop_handler)
  signal.signal(signal.SIGINT, engine.stop_handler)
  engine.event(
      "started",
      files=len(engine.ROWS),
      bytes=sum(row["size"] for row in engine.ROWS),
      workers=engine.MAX_WORKERS,
  )
  engine.run_queue(engine.ROWS)
  state = (
      "stopped"
      if engine.STOP.is_set()
      else "completed"
      if len(engine.VERIFIED) == len(engine.ROWS)
      else "needs_attention"
  )
  engine.snapshot(state)
  engine.event(state, verified_files=len(engine.VERIFIED), failed_files=len(engine.FAILED))
  return 0 if state == "completed" else 1


if __name__ == "__main__":
  sys.exit(main())
