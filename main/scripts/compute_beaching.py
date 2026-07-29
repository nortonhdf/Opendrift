"""
Aggregate ensemble NC files into coastal beaching (stranding) statistics.

A particle is 'stranded' when its status becomes 1 (OpenDrift deactivates it on
contact with the coastline). Its stranding location is the last valid position;
the time-to-strand is the elapsed time at that step.

For each (field, season) pair this computes:
  - strand_grid : per-cell stranding probability (stranded particles whose
                  landing falls in the cell, divided by all particles released)
  - stranded_fraction : overall fraction of released particles that beach
  - time-to-strand percentiles (p10 / p50 / p90, hours)
  - beaching centroid (probability-weighted landing point)

Grid matches compute_risk_grids.py (0.1° over the Campos Basin domain).

Usage (from repo root, opendrift env active):
    python main/scripts/compute_beaching.py

Outputs:
    main/outputs/beaching/<field>_<season>_beaching.npz
    main/outputs/beaching/manifest.json
"""

import json
import sys
import numpy as np
import xarray as xr
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Windows consoles default to cp1252, which can't encode the → status glyph.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from main.fields_config import CAMPOS_FIELDS
from main.status_utils import STRANDED, code_of, last_valid_index

ENSEMBLE_DIR = ROOT / "main" / "outputs" / "ensemble"
OUT_DIR      = ROOT / "main" / "outputs" / "beaching"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Grid domain/resolution and season keys come from the single source of
# truth (audit finding #11 — these were re-declared in 4 files).
from main.domain_config import (
    GRID_LAT_MAX, GRID_LAT_MIN, GRID_LON_MAX, GRID_LON_MIN, GRID_RES, SEASONS,
)

LON_MIN, LON_MAX = GRID_LON_MIN, GRID_LON_MAX
LAT_MIN, LAT_MAX = GRID_LAT_MIN, GRID_LAT_MAX


def make_grid() -> tuple[np.ndarray, np.ndarray]:
    lons = np.arange(LON_MIN, LON_MAX, GRID_RES)
    lats = np.arange(LAT_MIN, LAT_MAX, GRID_RES)
    return lons, lats


def stranding_events(lon, lat, status, times, stranded_code):
    """Return (slon, slat, shours) for particles that strand in one run.

    ``stranded_code`` is the per-file integer meaning 'stranded' (decoded from
    the file's flag_meanings — it varies between files; audit finding #1).
    Pass None when the file recorded no stranding at all.
    A particle counts as stranded when its status at its last valid position
    is the stranded code — including a stranding exactly at the final output
    step (a particle that survives shows status 'active' there instead).
    """
    t_hours = (times - times[0]) / np.timedelta64(1, "h")

    if stranded_code is None:
        return np.array([]), np.array([]), np.array([])

    last = last_valid_index(lon)
    idx = np.arange(lon.shape[0])
    ok = last >= 0
    stranded = ok & (status[idx, np.maximum(last, 0)] == stranded_code)
    li = last[stranded]
    return (lon[stranded, li],
            lat[stranded, li],
            np.asarray(t_hours[li], dtype=float))


def compute_beaching(nc_paths, lons, lats) -> dict:
    n_lon, n_lat = len(lons), len(lats)
    strand_count = np.zeros((n_lat, n_lon), dtype=np.float32)

    total_particles = 0
    total_stranded  = 0
    all_hours = []
    valid_runs = 0

    for path in nc_paths:
        try:
            ds     = xr.open_dataset(path)
            lon    = ds["lon"].values
            lat    = ds["lat"].values
            status = ds["status"].values
            times  = ds["time"].values
            stranded_code = code_of(ds["status"], STRANDED)  # varies per file!
            ds.close()
        except Exception as e:
            print(f"    [WARN] skipping {Path(path).name}: {e}")
            continue

        total_particles += lon.shape[0]
        slon, slat, shours = stranding_events(lon, lat, status, times, stranded_code)
        total_stranded += len(slon)
        all_hours.extend(shours)

        if len(slon):
            ci = np.floor((slon - LON_MIN) / GRID_RES).astype(int)
            ri = np.floor((slat - LAT_MIN) / GRID_RES).astype(int)
            m = (ci >= 0) & (ci < n_lon) & (ri >= 0) & (ri < n_lat)
            np.add.at(strand_count, (ri[m], ci[m]), 1)
        valid_runs += 1

    strand_grid = (strand_count / total_particles).astype(np.float32) \
        if total_particles else strand_count
    all_hours = np.array(all_hours, dtype=np.float32)

    if all_hours.size:
        p10, p50, p90 = np.percentile(all_hours, [10, 50, 90])
    else:
        p10 = p50 = p90 = np.nan

    return {
        "strand_grid":       strand_grid,
        "stranded_fraction": (total_stranded / total_particles) if total_particles else 0.0,
        "n_members":         valid_runs,
        "n_particles_total": total_particles,
        "n_stranded":        total_stranded,
        "hours_p10":         float(p10),
        "hours_p50":         float(p50),
        "hours_p90":         float(p90),
    }


def main() -> None:
    manifest_path = ENSEMBLE_DIR / "manifest.json"
    if not manifest_path.exists():
        print("[ERROR] No ensemble manifest — run run_ensemble.py first.")
        sys.exit(1)

    ensemble = json.loads(manifest_path.read_text())
    lons, lats = make_grid()
    beaching_manifest = {}

    for field in CAMPOS_FIELDS:
        field_key = field.lower().replace(" ", "_")
        for season in SEASONS:
            nc_paths = [
                str(ROOT / v["nc"])
                for v in ensemble.values()
                if v["field"] == field and v["season"] == season
            ]
            if not nc_paths:
                print(f"  [SKIP] {field_key}_{season} — no ensemble members")
                continue

            print(f"  {field_key}_{season}  ({len(nc_paths)} members)…", flush=True)
            r = compute_beaching(nc_paths, lons, lats)

            out_path = OUT_DIR / f"{field_key}_{season}_beaching.npz"
            np.savez_compressed(
                out_path,
                strand_grid       = r["strand_grid"],
                lons              = lons,
                lats              = lats,
                stranded_fraction = np.array(r["stranded_fraction"], dtype=np.float32),
                n_members         = np.array(r["n_members"]),
                n_particles_total = np.array(r["n_particles_total"]),
                n_stranded        = np.array(r["n_stranded"]),
                hours_p10         = np.array(r["hours_p10"], dtype=np.float32),
                hours_p50         = np.array(r["hours_p50"], dtype=np.float32),
                hours_p90         = np.array(r["hours_p90"], dtype=np.float32),
            )

            beaching_manifest[f"{field_key}_{season}"] = {
                "field":             field,
                "season":            season,
                "stranded_fraction": r["stranded_fraction"],
                "n_members":         r["n_members"],
                "npz":               str(out_path.relative_to(ROOT)),
            }
            print(f"    → {out_path.name}  "
                  f"(stranded {r['stranded_fraction']*100:.0f}%, "
                  f"median {r['hours_p50']:.0f}h)")

    (OUT_DIR / "manifest.json").write_text(json.dumps(beaching_manifest, indent=2))
    print(f"\n[OK] {len(beaching_manifest)} beaching grids → {OUT_DIR}")


if __name__ == "__main__":
    main()
