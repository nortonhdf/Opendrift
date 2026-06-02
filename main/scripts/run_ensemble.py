"""
Run ensemble of simulations for each (field, season) pair.

10 start dates spread evenly through each month capture seasonal variability
in currents and wind without needing to perturb physics parameters.

Usage (from repo root, opendrift env active):
    python main/scripts/run_ensemble.py

Outputs:
    main/outputs/ensemble/<field>_<season>_m<N>.nc
    main/outputs/ensemble/manifest.json

Safe to interrupt and resume — already-done members are skipped.
"""

import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Windows consoles default to cp1252, which can't encode the ✓/✗ status glyphs.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from main.fields_config import CAMPOS_FIELDS
from main.run_open_oil import run_simulation

OUT_DIR  = ROOT / "main" / "outputs" / "ensemble"
MANIFEST = OUT_DIR / "manifest.json"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_MEMBERS   = 10
N_PARTICLES = 200
DURATION_H  = 120

# 10 start dates per season, evenly spaced through the month
_DAYS = [1, 4, 7, 10, 13, 16, 19, 22, 25, 28]
SEASON_DATES = {
    "jan": [datetime(2025,  1, d) for d in _DAYS],
    "apr": [datetime(2025,  4, d) for d in _DAYS],
    "jul": [datetime(2025,  7, d) for d in _DAYS],
    "oct": [datetime(2025, 10, d) for d in _DAYS],
}


def load_manifest() -> dict:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text())
    return {}


def save_manifest(m: dict) -> None:
    MANIFEST.write_text(json.dumps(m, indent=2))


def main() -> None:
    manifest = load_manifest()

    tasks = [
        (field, season, midx, start_dt)
        for field in CAMPOS_FIELDS
        for season, dates in SEASON_DATES.items()
        for midx, start_dt in enumerate(dates)
    ]
    total = len(tasks)

    print(f"\n{'='*62}")
    print(f"Ensemble: {total} runs  "
          f"({len(CAMPOS_FIELDS)} fields × 4 seasons × {N_MEMBERS} members)")
    print(f"Particles per run: {N_PARTICLES}   Duration: {DURATION_H}h")
    print(f"Output: {OUT_DIR}")
    print(f"{'='*62}\n")

    done = skipped = failed = 0

    for i, (field, season, midx, start_dt) in enumerate(tasks, 1):
        key    = f"{field.lower().replace(' ', '_')}_{season}_m{midx:02d}"
        nc_out = OUT_DIR / f"{key}.nc"
        prefix = f"[{i:03d}/{total}] {key}"

        if nc_out.exists() and key in manifest:
            print(f"{prefix}  ✓ skip")
            skipped += 1
            continue

        print(f"{prefix}  running…", flush=True)
        cfg = CAMPOS_FIELDS[field]

        try:
            run_simulation(
                seed_lon      = cfg["lon"],
                seed_lat      = cfg["lat"],
                n_particles   = N_PARTICLES,
                start_time    = start_dt,
                duration_hours= DURATION_H,
                oil_type      = cfg["oil_type"],
                use_wind      = True,
                use_waves     = False,
                outfile       = str(nc_out),
                figfile       = str(OUT_DIR / f"{key}.png"),
                loglevel      = 40,
            )
            manifest[key] = {
                "field":  field,
                "season": season,
                "member": midx,
                "start":  start_dt.isoformat(),
                "nc":     str(nc_out.relative_to(ROOT)),
            }
            save_manifest(manifest)
            print(f"{prefix}  ✓ done")
            done += 1

        except Exception:
            print(f"{prefix}  ✗ FAILED")
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*62}")
    print(f"Done: {done}   Skipped: {skipped}   Failed: {failed}")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()
