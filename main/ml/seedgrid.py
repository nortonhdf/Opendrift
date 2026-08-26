"""Release points sampled over the domain, not just the six fields.

Limitation #1 of the whole project (`ESTADO_ATUAL.md` §6): every claim about
forecasting at "a location never seen in training" rests on leave-one-field-out
across six neighbouring sites of one basin. Six locations cannot tell apart
"the model transfers in space" from "the six sites are alike enough". This
module removes that ceiling by generating the archive from a sample of
locations, so leave-one-LOCATION-out has tens of held-out sites instead of
five.

Three choices that shape what the experiment can conclude:

1. **Only location varies.** Oil API is fixed at REF_API and the spill volume
   at the project reference, so a difference between locations cannot be an
   oil-type effect wearing a map. Sampling those too is a later experiment,
   not this one.

2. **The land mask is the one the simulation sees** — a cell is water iff the
   CMEMS currents field is finite there. Any other mask could place a seed
   where the forcing has no data, which fails for a reason unrelated to
   geography. Seeds also keep COAST_MARGIN_CELLS of water around them, so a
   run does not begin by stranding.

3. **The sampling box is asymmetric on purpose.** Drift in this domain runs
   broadly south-westward, so points are kept further from the south and west
   edges than from the north and east. Selecting the region BEFORE running is
   what keeps this honest: dropping runs afterwards because they left the box
   would quietly delete exactly the fast south-westward scenarios.

Bathymetry is not available for arbitrary points, so ``water_depth_m`` is
NaN in this archive. HistGradientBoosting consumes NaN natively, and the
feature is worth nothing measurable — it does not reach the top ten
permutation importances of the published model, below even ``api`` at
0.10 km — so its absence here costs nothing. A model trained on the six
fields, where the depth is known, must still not be evaluated here without
saying so.

**Coastal locations are kept, on purpose.** COAST_MARGIN_CELLS only keeps a
seed off land; it does not keep it far from land, so a point ~20 km offshore
lands in the sample and beaches within days. The pilot found exactly one such
location out of eight (half its runs stranded; the other seven never
stranded). Raising the margin until they disappear would be selecting on the
outcome — the same mistake the domain-exit margin avoids — and it would also
delete the one regime six offshore production fields can never teach. They
stay, and the evaluation should report coastal and offshore separately.

One consequence to keep in mind when reading targets from a beached run:
OpenDrift pads deactivated elements with NaN, so the centroid at D+n is the
centroid of what is STILL DRIFTING, not of the oil including what is already
on the beach. That convention was invisible while beaching was near zero
(six fields); here it is not.

Usage (repo root, opendrift env):
    python -m main.ml.seedgrid plan               # show the sample, run nothing
    python -m main.ml.seedgrid generate 2025      # generate (resumable)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from main.domain_config import SEASON_MONTHS  # noqa: E402
from main.fields_config import oil_type_for_api  # noqa: E402
from main.ml.multiyear import DURATION_H, N_PARTICLES, forcing_paths  # noqa: E402
from main.run_open_oil import run_simulation  # noqa: E402

GRID_DIR_TMPL = "training168_grid_{year}"

# Sampling region inside the forcing box (lon -45..-36 / lat -27..-19).
#
# The margins are MEASURED, not guessed. Over the 720 archived 168-h runs the
# furthest any particle travelled from its release was 373 km west, 384 km
# south, 128 km east and 161 km north — the asymmetry of the Brazil Current,
# in numbers. Converted to degrees at this latitude and rounded up, that is
# the margin below, so a D+7 run started anywhere in this region cannot reach
# the edge of the forcing even in the worst case observed.
#
# A first attempt used a guessed margin (2 deg west, 1.5 deg south). The
# pilot caught it: one run in 32 lost 39 % of its particles over the boundary
# — a regression against the "0 domain exits" bar the regeneration set.
#
# The cost is honest and worth stating: this region is roughly the Campos
# Basin and its surroundings, so the archive strengthens the spatial claim at
# BASIN scale. Reaching arbitrary geography needs a wider forcing download,
# which is a different job.
SAMPLE_LON = (-41.2, -37.2)
SAMPLE_LAT = (-23.5, -20.4)

COAST_MARGIN_CELLS = 2      # ~18 km of water around the seed at 1/12 deg
REF_API = 28.0              # fixed: this experiment isolates LOCATION
N_LOCATIONS = 40
POOL = 1000          # fixed LHS pool: makes sample_locations(n) a prefix
START_DAYS = [5, 15, 25]
SEED = 42


def water_mask(year: int):
    """(lons, lats, mask) from the currents file — mask is True over water."""
    cur, _ = forcing_paths(year)
    ds = xr.open_dataset(cur)
    lons = np.asarray(ds["longitude"].values, float)
    lats = np.asarray(ds["latitude"].values, float)
    u = np.asarray(ds["x_sea_water_velocity"].isel(time=0).values, float)
    ds.close()
    return lons, lats, np.isfinite(np.squeeze(u))


def _is_open_water(lons, lats, mask, lon: float, lat: float,
                   margin: int = COAST_MARGIN_CELLS) -> bool:
    i = int(np.abs(lats - lat).argmin())
    j = int(np.abs(lons - lon).argmin())
    if not (margin <= i < len(lats) - margin and margin <= j < len(lons) - margin):
        return False
    block = mask[i - margin:i + margin + 1, j - margin:j + margin + 1]
    return bool(block.all())


def sample_locations(n: int = N_LOCATIONS, year: int = 2025,
                     seed: int = SEED) -> list:
    """Latin-hypercube sample of open-water release points, deterministic.

    LHS spreads the sample over the region instead of clumping the way plain
    uniform draws do — with only a few dozen locations that matters, because
    a clump would understate how far the model has to transfer.

    **The pool size is fixed, not n * something.** An LHS of m points
    stratifies the region into m slices, so drawing 160 points and drawing
    800 gives two unrelated samples — ``sample_locations(8)`` would NOT be a
    prefix of ``sample_locations(40)``. That property is load-bearing here:
    it is what lets a pilot batch become part of the full archive instead of
    being thrown away, and what stops a later "add ten more locations" from
    invalidating every run already generated. Draw the same pool always and
    take the first n that land in open water.
    """
    from scipy.stats import qmc

    lons, lats, mask = water_mask(year)
    draws = qmc.LatinHypercube(d=2, seed=seed).random(POOL)
    out = []
    for u1, u2 in draws:
        lon = SAMPLE_LON[0] + u1 * (SAMPLE_LON[1] - SAMPLE_LON[0])
        lat = SAMPLE_LAT[0] + u2 * (SAMPLE_LAT[1] - SAMPLE_LAT[0])
        if _is_open_water(lons, lats, mask, lon, lat):
            out.append((round(float(lon), 4), round(float(lat), 4)))
        if len(out) == n:
            break
    if len(out) < n:
        raise SystemExit(
            f"Só {len(out)} de {n} pontos caíram em água aberta num pool de "
            f"{POOL} — aumente POOL ou reduza COAST_MARGIN_CELLS.")
    return out


def tasks(locations, seasons=None, days=None) -> list:
    seasons = list(seasons or SEASON_MONTHS)
    days = list(days or START_DAYS)
    return [(k, lon, lat, s, d)
            for k, (lon, lat) in enumerate(locations)
            for s in seasons for d in days]


def generate(year: int, n_locations: int = N_LOCATIONS, seasons=None,
             days=None) -> dict:
    """Run the archive, resumable — an existing NetCDF is never recomputed."""
    locations = sample_locations(n_locations, year)
    out_dir = ROOT / "main" / "outputs" / GRID_DIR_TMPL.format(year=year)
    out_dir.mkdir(parents=True, exist_ok=True)
    mpath = out_dir / "manifest.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {}
    cur, wnd = forcing_paths(year)

    todo = tasks(locations, seasons, days)
    done = skipped = failed = 0
    for i, (k, lon, lat, season, day) in enumerate(todo, 1):
        key = f"grid{k:03d}_{season}_d{day:02d}"
        nc_out = out_dir / f"{key}.nc"
        if nc_out.exists() and key in manifest:
            skipped += 1
            continue
        print(f"[{i:04d}/{len(todo)}] {year} {key} "
              f"({lat:.2f}, {lon:.2f})…", flush=True)
        try:
            run_simulation(
                seed_lon=lon, seed_lat=lat, n_particles=N_PARTICLES,
                start_time=datetime(year, SEASON_MONTHS[season], day),
                duration_hours=DURATION_H,
                oil_type=oil_type_for_api(REF_API),
                use_wind=True, use_waves=False,
                currents_file=str(cur), wind_file=str(wnd), waves_file=None,
                outfile=str(nc_out), figfile=None, loglevel=40,
            )
            manifest[key] = {
                "field": f"grid{k:03d}", "season": season, "day": day,
                "year": year, "lon": lon, "lat": lat, "api": REF_API,
                "water_depth_m": None,
                "nc": str(nc_out.relative_to(ROOT)),
            }
            mpath.write_text(json.dumps(manifest, indent=2))
            done += 1
        except Exception as e:
            print(f"  FAILED {key}: {type(e).__name__}: {e}", flush=True)
            failed += 1
    print(f"\n{year} -> feitos: {done}  pulados: {skipped}  falhas: {failed}")
    return {"done": done, "skipped": skipped, "failed": failed,
            "manifest": str(mpath), "n_locations": len(locations)}


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="Seed-location grid archive.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_plan = sub.add_parser("plan", help="show the sample, run nothing")
    p_plan.add_argument("--year", type=int, default=2025)
    p_plan.add_argument("-n", type=int, default=N_LOCATIONS)
    p_gen = sub.add_parser("generate", help="generate the archive (resumable)")
    p_gen.add_argument("year", type=int)
    p_gen.add_argument("-n", type=int, default=N_LOCATIONS)
    p_gen.add_argument("--days", type=int, nargs="*", default=None)
    args = ap.parse_args()

    if args.cmd == "plan":
        locs = sample_locations(args.n, args.year)
        t = tasks(locs)
        print(f"{len(locs)} locais x {len(SEASON_MONTHS)} estações x "
              f"{len(START_DAYS)} dias = {len(t)} runs")
        print(f"região amostrada: lon {SAMPLE_LON}, lat {SAMPLE_LAT}")
        for k, (lon, lat) in enumerate(locs):
            print(f"  grid{k:03d}  {lat:8.3f}, {lon:9.3f}")
    else:
        generate(args.year, args.n, days=args.days)


if __name__ == "__main__":
    main()
