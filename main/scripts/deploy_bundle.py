"""Assemble the data the app actually needs to run, and nothing else.

The repository carries ~2.6 GB of inputs and outputs because the author chose
to version every product (PERGUNTAS_ABERTAS #9). Almost none of it is needed
at RUN time, and that distinction is what a deployment decision turns on:

  ~410 MB of ``*_raw*.nc``   inputs to prep, never opened by the app
  ~1.5 GB of run archives    training168_*, ensemble/, holdout_* — evidence
                             behind the published numbers, not app data
  ~111 MB of extra years     only needed to forecast a date in those years

What is left is small. Each tab is listed separately because the expensive
one is optional: only "Custom Run" needs the wind field, and only because it
runs a live simulation.

Usage (repo root, opendrift env):
    python main/scripts/deploy_bundle.py --dry-run
    python main/scripts/deploy_bundle.py --out ../campos-deploy
    python main/scripts/deploy_bundle.py --out DIR --tabs scenarios risk forecast
    python main/scripts/deploy_bundle.py --out DIR --years 2024 2025
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

IN = Path("main/inputs")
OUT = Path("main/outputs")

# What each tab opens at run time. Paths are repo-relative; directories are
# copied whole, minus the patterns in EXCLUDE.
TAB_FILES = {
    "scenarios": [OUT / "scenarios"],
    "risk": [OUT / "risk_grids"],
    "beaching": [OUT / "beaching"],
    "custom": [IN / "currents.nc", IN / "wind_cf.nc", IN / "waves_cf.nc"],
    "forecast": [OUT / "ml" / "forecast_product.joblib",
                 OUT / "ml" / "footprint_plume.joblib"],
}
# The forecast tab reads antecedent statistics from the currents field of the
# release year, so one currents file per year it should be able to answer for.
YEAR_FILES = {2025: IN / "currents.nc"}
for _y in (2022, 2023, 2024):
    YEAR_FILES[_y] = IN / f"currents_{_y}.nc"

EXCLUDE = ("*.png", "*_budget.npz")


def _size(p: Path) -> int:
    if p.is_dir():
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    return p.stat().st_size if p.exists() else 0


def plan(tabs, years) -> list:
    """(source, bytes, present) for everything the chosen tabs need."""
    wanted: list = []
    for t in tabs:
        wanted += TAB_FILES[t]
    if "forecast" in tabs:
        wanted += [YEAR_FILES[y] for y in years if y in YEAR_FILES]
    seen, out = set(), []
    for rel in wanted:
        if str(rel) in seen:
            continue
        seen.add(str(rel))
        src = ROOT / rel
        out.append((rel, _size(src), src.exists()))
    return out


def copy_bundle(items, dest: Path) -> int:
    total = 0
    for rel, size, present in items:
        if not present:
            continue
        src, dst = ROOT / rel, dest / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns(*EXCLUDE))
            total += _size(dst)
        else:
            shutil.copy2(src, dst)
            total += size
    return total


def main() -> None:
    ap = argparse.ArgumentParser(description="Minimal data bundle for the app.")
    ap.add_argument("--out", type=Path, help="destination directory")
    ap.add_argument("--tabs", nargs="*", default=list(TAB_FILES),
                    choices=list(TAB_FILES))
    ap.add_argument("--years", type=int, nargs="*",
                    default=sorted(YEAR_FILES), metavar="ANO",
                    help="years the forecast tab must answer for")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    items = plan(args.tabs, args.years)
    print(f"{'caminho':52s}{'MB':>10s}  estado")
    total = 0
    for rel, size, present in items:
        total += size if present else 0
        print(f"{str(rel):52s}{size / 1e6:10.1f}  "
              + ("ok" if present else "AUSENTE"))
    print(f"{'TOTAL':52s}{total / 1e6:10.1f} MB")
    missing = [str(r) for r, _, p in items if not p]
    if missing:
        print("\n[AVISO] ausentes: " + ", ".join(missing))

    if args.dry_run or not args.out:
        if not args.out:
            print("\n(sem --out: apenas o plano; nada foi copiado)")
        return
    copied = copy_bundle(items, args.out)
    print(f"\n[OK] {copied / 1e6:.1f} MB -> {args.out.resolve()}")
    print("Faltam o código (git clone) e o env (environment.yml).")


if __name__ == "__main__":
    main()
