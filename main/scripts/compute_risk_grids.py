"""
Aggregate ensemble NC files into probability risk grids.

For each (field, season) pair, computes two probability layers:
  - prob_any   : P(cell visited at any point in 120h window)  — exposure risk
  - prob_final : P(cell occupied at end of simulation)        — persistence risk

Grid: 0.1° resolution over the full Campos Basin domain.

Usage (from repo root, opendrift env active):
    python main/scripts/compute_risk_grids.py

Outputs:
    main/outputs/risk_grids/<field>_<season>_risk.npz
    main/outputs/risk_grids/manifest.json
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
from main.status_utils import ACTIVE, code_of

ENSEMBLE_DIR = ROOT / "main" / "outputs" / "ensemble"
OUT_DIR      = ROOT / "main" / "outputs" / "risk_grids"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LON_MIN, LON_MAX = -43.0, -38.5
LAT_MIN, LAT_MAX = -25.0, -21.0
GRID_RES = 0.1

SEASONS = ["jan", "apr", "jul", "oct"]


def make_grid() -> tuple[np.ndarray, np.ndarray]:
    lons = np.arange(LON_MIN, LON_MAX, GRID_RES)
    lats = np.arange(LAT_MIN, LAT_MAX, GRID_RES)
    return lons, lats


def particles_to_grid(
    lon_arr: np.ndarray, lat_arr: np.ndarray,
    n_lon: int, n_lat: int,
) -> np.ndarray:
    """Binary presence grid (n_lat, n_lon): 1 where any particle is present."""
    grid = np.zeros((n_lat, n_lon), dtype=np.float32)
    valid = ~(np.isnan(lon_arr) | np.isnan(lat_arr))
    if not valid.any():
        return grid
    lon_v = lon_arr[valid]
    lat_v = lat_arr[valid]
    ci = np.floor((lon_v - LON_MIN) / GRID_RES).astype(int)
    ri = np.floor((lat_v - LAT_MIN) / GRID_RES).astype(int)
    mask = (ci >= 0) & (ci < n_lon) & (ri >= 0) & (ri < n_lat)
    np.add.at(grid, (ri[mask], ci[mask]), 1)
    return (grid > 0).astype(np.float32)


def compute_risk(nc_paths: list, lons: np.ndarray, lats: np.ndarray) -> dict:
    n_lon, n_lat = len(lons), len(lats)
    sum_final = np.zeros((n_lat, n_lon), dtype=np.float32)
    sum_any   = np.zeros((n_lat, n_lon), dtype=np.float32)
    valid_runs = 0

    for path in nc_paths:
        try:
            ds     = xr.open_dataset(path)
            lond   = ds["lon"].values    # (n_particles, n_time)
            latd   = ds["lat"].values
            status = ds["status"].values
            # 'active' is 0 by convention, but decode it from the file's
            # flag_meanings anyway — codes are per-file (audit finding #1).
            active_code = code_of(ds["status"], ACTIVE)
            ds.close()
        except Exception as e:
            print(f"    [WARN] skipping {Path(path).name}: {e}")
            continue
        if active_code is None:
            active_code = 0

        # Final timestep — active particles only
        active_f = status[:, -1] == active_code
        sum_final += particles_to_grid(lond[active_f, -1], latd[active_f, -1], n_lon, n_lat)

        # Any timestep — union of all cells ever visited by active particles
        n_t   = lond.shape[1]
        g_any = np.zeros((n_lat, n_lon), dtype=np.float32)
        for t in range(n_t):
            active_t = status[:, t] == active_code
            g_t = particles_to_grid(lond[active_t, t], latd[active_t, t], n_lon, n_lat)
            np.maximum(g_any, g_t, out=g_any)
        sum_any += g_any
        valid_runs += 1

    if valid_runs == 0:
        return {"prob_final": sum_final, "prob_any": sum_any, "n_members": 0}
    return {
        "prob_final": sum_final / valid_runs,
        "prob_any":   sum_any   / valid_runs,
        "n_members":  valid_runs,
    }


def main() -> None:
    manifest_path = ENSEMBLE_DIR / "manifest.json"
    if not manifest_path.exists():
        print("[ERROR] No ensemble manifest — run run_ensemble.py first.")
        sys.exit(1)

    ensemble = json.loads(manifest_path.read_text())
    lons, lats = make_grid()
    risk_manifest = {}

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
            result = compute_risk(nc_paths, lons, lats)

            out_path = OUT_DIR / f"{field_key}_{season}_risk.npz"
            np.savez_compressed(
                out_path,
                prob_final = result["prob_final"],
                prob_any   = result["prob_any"],
                lons       = lons,
                lats       = lats,
                n_members  = np.array(result["n_members"]),
            )

            risk_manifest[f"{field_key}_{season}"] = {
                "field":     field,
                "season":    season,
                "n_members": result["n_members"],
                "npz":       str(out_path.relative_to(ROOT)),
            }
            print(f"    → {out_path.name}  "
                  f"(max prob_any={result['prob_any'].max():.2f})")

    (OUT_DIR / "manifest.json").write_text(json.dumps(risk_manifest, indent=2))
    print(f"\n[OK] {len(risk_manifest)} risk grids → {OUT_DIR}")


if __name__ == "__main__":
    main()
