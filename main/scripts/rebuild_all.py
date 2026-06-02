"""
One-shot rebuild of every precomputed product with the *current* model code
(correct ADIOS oil per field, oil-budget sidecars, latest physics).

Stages, in dependency order:
  1. scenarios  — 48 runs   (~47 min)   precompute_scenarios   → outputs/scenarios/
  2. ensemble   — 240 runs  (~3 h)      run_ensemble           → outputs/ensemble/
  3. risk       — 24 grids  (~min)      compute_risk_grids     → outputs/risk_grids/
  4. beaching   — 24 grids  (~min)      compute_beaching       → outputs/beaching/

Usage (from the repo root, with the `opendrift` env active):

    python main/scripts/rebuild_all.py                 # show the plan, change nothing
    python main/scripts/rebuild_all.py --fresh         # delete manifests + rebuild ALL (~3.5–4 h)
    python main/scripts/rebuild_all.py --fresh --only scenarios
    python main/scripts/rebuild_all.py --fresh --only scenarios,beaching
    python main/scripts/rebuild_all.py --resume        # continue an interrupted rebuild

--fresh   wipes the scenarios/ensemble manifests so every run is recomputed and the
          existing NetCDFs are overwritten in place.
--resume  keeps the manifests; scenarios/ensemble skip members already on disk, so an
          interrupted rebuild picks up where it stopped.
risk/beaching always recompute fully from the current ensemble (no manifest skip).

Safe to interrupt at any point (Ctrl-C) and re-run with --resume.
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Windows consoles default to cp1252, which can't encode the ✓/✗/→ glyphs.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUTPUTS = ROOT / "main" / "outputs"

STAGES = ["scenarios", "ensemble", "risk", "beaching"]

ESTIMATES = {
    "scenarios": "~47 min  (48 runs)",
    "ensemble":  "~3 h     (240 runs)",
    "risk":      "~minutes (24 grids)",
    "beaching":  "~minutes (24 grids)",
}

# Manifests cleared by --fresh to force a full recompute of that stage.
MANIFESTS = {
    "scenarios": OUTPUTS / "scenarios" / "manifest.json",
    "ensemble":  OUTPUTS / "ensemble" / "manifest.json",
}


def banner(msg: str) -> None:
    print("\n" + "=" * 64)
    print(msg)
    print("=" * 64, flush=True)


def run_stage(name: str, fresh: bool) -> None:
    banner(f"STAGE: {name}   ({ESTIMATES[name]})")
    t0 = time.time()

    if fresh and name in MANIFESTS and MANIFESTS[name].exists():
        MANIFESTS[name].unlink()
        print(f"  [fresh] removed {MANIFESTS[name].relative_to(ROOT)} — full recompute")

    if name == "scenarios":
        from main.scripts import precompute_scenarios
        precompute_scenarios.main()
    elif name == "ensemble":
        from main.scripts import run_ensemble
        run_ensemble.main()
    elif name == "risk":
        from main.scripts import compute_risk_grids
        compute_risk_grids.main()
    elif name == "beaching":
        from main.scripts import compute_beaching
        compute_beaching.main()

    print(f"  [{name}] done in {(time.time() - t0) / 60:.1f} min", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Rebuild all precomputed Campos products.")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--fresh", action="store_true",
                      help="Delete scenario/ensemble manifests and recompute everything.")
    mode.add_argument("--resume", action="store_true",
                      help="Keep manifests; continue an interrupted rebuild.")
    p.add_argument("--only", default="",
                   help="Comma-separated subset of stages: " + ",".join(STAGES))
    args = p.parse_args()

    stages = [s.strip() for s in args.only.split(",") if s.strip()] or STAGES
    bad = [s for s in stages if s not in STAGES]
    if bad:
        p.error(f"unknown stage(s): {bad}. Choose from {STAGES}")

    # No mode flag → just print the plan and exit (nothing is changed).
    if not (args.fresh or args.resume):
        banner("REBUILD PLAN (dry run — nothing changed)")
        print("Stages that would run, in order:")
        for s in stages:
            print(f"  • {s:10s} {ESTIMATES[s]}")
        print("\nRe-run with:")
        print("  --fresh    to delete manifests and recompute everything")
        print("  --resume   to continue an interrupted rebuild")
        print("Add --only scenarios[,ensemble,...] to limit the stages.")
        return

    fresh = args.fresh
    banner(f"REBUILD START  ({'FRESH' if fresh else 'RESUME'})  stages: {', '.join(stages)}")
    t0 = time.time()
    for s in stages:
        run_stage(s, fresh)
    banner(f"REBUILD COMPLETE in {(time.time() - t0) / 60:.1f} min")
    print("Launch the app with:  python -m streamlit run main/app.py")


if __name__ == "__main__":
    main()
