"""
Pre-compute all 48 scenarios: 6 fields × 4 seasons × 2 wind states.

Usage (from repo root, opendrift env active):
    python main/scripts/precompute_scenarios.py

Outputs:
    main/outputs/scenarios/<field>_<season>_<wind>.nc   (one per scenario)
    main/outputs/scenarios/manifest.json                (index of ready scenarios)

Skips scenarios already on disk — safe to interrupt and resume.
"""

import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Windows consoles default to cp1252, which can't encode the ✓/✗ status glyphs.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from main.domain_config import SEASONS as SEASON_KEYS, season_date
from main.fields_config import CAMPOS_FIELDS
from main.run_open_oil import run_simulation

OUT_DIR = ROOT / "main" / "outputs" / "scenarios"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST = OUT_DIR / "manifest.json"

SEASONS = {s: season_date(s) for s in SEASON_KEYS}

WIND_STATES = {"wind_on": True, "wind_off": False}

DURATION_H  = 120
N_PARTICLES = 500   # lighter than full 1000 for faster batch runs


def load_manifest() -> dict:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text())
    return {}


def save_manifest(m: dict) -> None:
    MANIFEST.write_text(json.dumps(m, indent=2))


def scenario_key(field: str, season: str, wind: str) -> str:
    return f"{field.lower().replace(' ', '_')}_{season}_{wind}"


def main() -> None:
    manifest = load_manifest()

    scenarios = [
        (field, season, wind)
        for field  in CAMPOS_FIELDS
        for season in SEASONS
        for wind   in WIND_STATES
    ]
    total = len(scenarios)

    print(f"\n{'='*60}")
    print(f"Pre-computing {total} scenarios  ({N_PARTICLES} particles each)")
    print(f"Output: {OUT_DIR}")
    print(f"{'='*60}\n")

    done = skipped = failed = 0

    for i, (field, season, wind_label) in enumerate(scenarios, 1):
        key    = scenario_key(field, season, wind_label)
        nc_out = OUT_DIR / f"{key}.nc"

        prefix = f"[{i:02d}/{total}] {key}"

        if nc_out.exists() and key in manifest:
            print(f"{prefix}  ✓ already done — skipping")
            skipped += 1
            continue

        print(f"{prefix}  running …", flush=True)
        cfg = CAMPOS_FIELDS[field]

        try:
            run_simulation(
                seed_lon      = cfg["lon"],
                seed_lat      = cfg["lat"],
                n_particles   = N_PARTICLES,
                start_time    = SEASONS[season],
                duration_hours= DURATION_H,
                oil_type      = cfg["oil_type"],
                use_wind      = WIND_STATES[wind_label],
                use_waves     = False,
                outfile       = str(nc_out),
                figfile       = str(OUT_DIR / f"{key}.png"),
                loglevel      = 40,
            )
            manifest[key] = {
                "field":    field,
                "season":   season,
                "wind":     wind_label,
                "nc":       str(nc_out.relative_to(ROOT)),
                "computed": datetime.now(timezone.utc).isoformat(),
            }
            save_manifest(manifest)
            print(f"{prefix}  ✓ done")
            done += 1

        except Exception:
            print(f"{prefix}  ✗ FAILED")
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*60}")
    print(f"Done: {done}   Skipped: {skipped}   Failed: {failed}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
