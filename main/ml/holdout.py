"""Blind evaluation of the transport surrogate on the frozen 2024 hold-out.

2024 forcing was downloaded and frozen BEFORE any model existed and never
enters training. This module:

  prep      — build CF/NetCDF3 forcing for 2024 (currents+SST, wind)
  generate  — run the OpenDrift ground truth: 6 fields x 4 months x 3 start
              days (5/15/25) = 72 runs, 200 particles, 120 h (resumable)
  evaluate  — roll the surrogate (and the passive-advection baseline) from
              the release point in 6-h steps to 120 h, against the OpenDrift
              patch-centroid track: Liu-Weisberg SS, final centroid error,
              final-patch IoU on the analysis grid

Usage (repo root, opendrift env):
    python -m main.ml.holdout prep
    python -m main.ml.holdout generate
    python -m main.ml.holdout evaluate
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from main.domain_config import (  # noqa: E402
    GRID_LAT_MIN, GRID_LON_MIN, GRID_RES, SEASON_MONTHS,
)
from main.fields_config import CAMPOS_FIELDS  # noqa: E402
from main.ml.baselines import predict_advection  # noqa: E402
from main.ml.dataset import (  # noqa: E402
    DT_HOURS, KM_PER_DEG, ForcingSampler, patch_state,
)
from main.ml.metrics import haversine_km, iou, liu_weisberg_ss  # noqa: E402

YEAR = 2024
START_DAYS = [5, 15, 25]
N_PARTICLES = 200
DURATION_H = 120
N_STEPS = int(DURATION_H / DT_HOURS)          # 20 rollout steps of 6 h

INPUTS = ROOT / "main" / "inputs"
CUR_2024 = INPUTS / "currents_2024.nc"
WND_2024 = INPUTS / "wind_cf_2024.nc"
OUT_DIR = ROOT / "main" / "outputs" / "holdout_2024"
MANIFEST = OUT_DIR / "manifest.json"
REPORT = ROOT / "main" / "outputs" / "ml" / "holdout_2024_report.json"
MODEL = ROOT / "main" / "outputs" / "ml" / "surrogate_hgb.joblib"


# ── prep ──────────────────────────────────────────────────────────────────────

def prep() -> None:
    from main.scripts.prep_cmems_currents import prep as prep_cur
    from main.scripts.prep_era5_wind import prep as prep_wnd

    ds_cur = xr.open_dataset(INPUTS / "currents_raw_2024.nc")
    ds_sst = xr.open_dataset(INPUTS / "sst_raw_2024.nc")
    out = prep_cur(ds_cur, ds_sst).load()
    out.to_netcdf(CUR_2024, format="NETCDF3_64BIT",
                  encoding={c: {"_FillValue": None} for c in out.coords})
    print("OK ->", CUR_2024.name, list(out.data_vars))

    wnd = prep_wnd(xr.open_dataset(INPUTS / "wind_raw_2024.nc")).load()
    wnd.to_netcdf(WND_2024, format="NETCDF3_64BIT",
                  encoding={c: {"_FillValue": None} for c in wnd.coords})
    print("OK ->", WND_2024.name, list(wnd.data_vars))


# ── generate (OpenDrift ground truth) ────────────────────────────────────────

def generate() -> None:
    from main.run_open_oil import run_simulation

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}

    tasks = [(f, s, d) for f in CAMPOS_FIELDS
             for s in SEASON_MONTHS for d in START_DAYS]
    done = failed = skipped = 0
    for i, (field, season, day) in enumerate(tasks, 1):
        key = f"{field.lower().replace(' ', '_')}_{season}_d{day:02d}"
        nc_out = OUT_DIR / f"{key}.nc"
        if nc_out.exists() and key in manifest:
            skipped += 1
            continue
        start = datetime(YEAR, SEASON_MONTHS[season], day)
        print(f"[{i:02d}/{len(tasks)}] {key} running…", flush=True)
        cfg = CAMPOS_FIELDS[field]
        try:
            run_simulation(
                seed_lon=cfg["lon"], seed_lat=cfg["lat"],
                n_particles=N_PARTICLES, start_time=start,
                duration_hours=DURATION_H, oil_type=cfg["oil_type"],
                use_wind=True, use_waves=False,
                currents_file=str(CUR_2024), wind_file=str(WND_2024),
                waves_file=None,
                outfile=str(nc_out), figfile=str(OUT_DIR / f"{key}.png"),
                loglevel=40,
            )
            manifest[key] = {"field": field, "season": season, "day": day,
                             "start": start.isoformat(),
                             "nc": str(nc_out.relative_to(ROOT))}
            MANIFEST.write_text(json.dumps(manifest, indent=2))
            done += 1
        except Exception as e:
            print(f"  FAILED {key}: {type(e).__name__}: {e}", flush=True)
            failed += 1
    print(f"\nDone: {done}  Skipped: {skipped}  Failed: {failed}")


# ── rollout + evaluation ─────────────────────────────────────────────────────

def _features(lon_c, lat_c, spread, age_h, env) -> np.ndarray:
    return np.array([[lon_c, lat_c, spread, age_h,
                      env["u_cur"], env["v_cur"],
                      env["u_wind"], env["v_wind"], env["sst"]]], np.float32)


def rollout(predict_fn, sampler: ForcingSampler, lon0: float, lat0: float,
            t0: np.datetime64, n_steps: int = N_STEPS):
    """Iterate 6-h patch predictions from the release point.

    predict_fn(features (1,9)) -> (1,3) [dx_km, dy_km, dspread_km].
    Returns (lons, lats, spreads) arrays of length n_steps+1.
    """
    lon_c, lat_c, spread = float(lon0), float(lat0), 0.0
    lons, lats, spreads = [lon_c], [lat_c], [spread]
    when = t0
    for k in range(n_steps):
        env = sampler.at(lon_c, lat_c, when)
        out = np.asarray(predict_fn(
            _features(lon_c, lat_c, spread, k * DT_HOURS, env)))[0]
        lon_c += float(out[0]) / (KM_PER_DEG * np.cos(np.radians(lat_c)))
        lat_c += float(out[1]) / KM_PER_DEG
        spread = max(0.0, spread + float(out[2]))
        when = when + np.timedelta64(int(DT_HOURS * 3600), "s")
        lons.append(lon_c)
        lats.append(lat_c)
        spreads.append(spread)
    return np.array(lons), np.array(lats), np.array(spreads)


def _truth_track(nc_path: Path, stride: int):
    ds = xr.open_dataset(nc_path)
    lon = np.asarray(ds["lon"].values, float)
    lat = np.asarray(ds["lat"].values, float)
    times = ds["time"].values
    ds.close()
    idx = list(range(0, lon.shape[1], stride))
    track = [patch_state(lon[:, t], lat[:, t]) for t in idx]
    tl = np.array([t[0] for t in track])
    tb = np.array([t[1] for t in track])
    sp = np.array([t[2] for t in track])
    return tl, tb, sp, lon[:, idx[-1]], lat[:, idx[-1]], times[0]


def _occupancy(lon, lat) -> np.ndarray:
    """Boolean occupancy of final particle positions on the analysis grid."""
    ok = np.isfinite(lon) & np.isfinite(lat)
    ci = np.floor((lon[ok] - GRID_LON_MIN) / GRID_RES).astype(int)
    ri = np.floor((lat[ok] - GRID_LAT_MIN) / GRID_RES).astype(int)
    g = np.zeros((int(8 / GRID_RES), int(9 / GRID_RES)), bool)
    m = (ci >= 0) & (ci < g.shape[1]) & (ri >= 0) & (ri < g.shape[0])
    g[ri[m], ci[m]] = True
    return g


def _disc(lon_c, lat_c, radius_km) -> np.ndarray:
    """Gaussian-patch proxy: cells within one RMS radius of the centroid."""
    ny, nx = int(8 / GRID_RES), int(9 / GRID_RES)
    lons = GRID_LON_MIN + (np.arange(nx) + 0.5) * GRID_RES
    lats = GRID_LAT_MIN + (np.arange(ny) + 0.5) * GRID_RES
    lg, tg = np.meshgrid(lons, lats)
    d = haversine_km(lg, tg, lon_c, lat_c)
    return d <= max(radius_km, GRID_RES * KM_PER_DEG / 2)


def evaluate() -> dict:
    import joblib
    models = joblib.load(MODEL)

    def surrogate_fn(F):
        return np.column_stack([m.predict(F) for m in models])

    def advection_fn(F):
        return predict_advection(F, DT_HOURS)

    candidates = [("surrogate", surrogate_fn), ("advection", advection_fn)]

    model_res_path = MODEL.with_name("surrogate_hgb_residual.joblib")
    if model_res_path.exists():
        models_res = joblib.load(model_res_path)

        def residual_fn(F):
            corr = np.column_stack([m.predict(F) for m in models_res])
            return predict_advection(F, DT_HOURS) + corr

        candidates.append(("adv+corr", residual_fn))

    sampler = ForcingSampler(CUR_2024, WND_2024)
    manifest = json.loads(MANIFEST.read_text())

    # output stride: DT_HOURS over the file's own output step
    rows = []
    for key, entry in sorted(manifest.items()):
        ds = xr.open_dataset(ROOT / entry["nc"])
        step_h = (ds["time"].values[1] - ds["time"].values[0]) / np.timedelta64(1, "h")
        ds.close()
        stride = int(round(DT_HOURS / step_h))
        tl, tb, tsp, flon, flat, t0 = _truth_track(ROOT / entry["nc"], stride)
        n_steps = len(tl) - 1
        truth_occ = _occupancy(flon, flat)

        res = {"key": key}
        for name, fn in candidates:
            ml, mb, msp = rollout(fn, sampler, tl[0], tb[0], t0, n_steps)
            res[name] = {
                "lw_ss": liu_weisberg_ss(tl, tb, ml, mb),
                "final_err_km": float(haversine_km(tl[-1], tb[-1], ml[-1], mb[-1])),
                "iou": iou(truth_occ, _disc(ml[-1], mb[-1], msp[-1])),
            }
        rows.append(res)
        print(f"  {key:28s} SS surr={res['surrogate']['lw_ss']:.2f} "
              f"adv={res['advection']['lw_ss']:.2f} | "
              f"err120h surr={res['surrogate']['final_err_km']:5.1f} km "
              f"adv={res['advection']['final_err_km']:5.1f} km", flush=True)
    sampler.close()

    summary = {}
    for name, _ in candidates:
        summary[name] = {
            "lw_ss_median": float(np.median([r[name]["lw_ss"] for r in rows])),
            "lw_ss_mean": float(np.mean([r[name]["lw_ss"] for r in rows])),
            "final_err_km_median": float(np.median([r[name]["final_err_km"] for r in rows])),
            "iou_mean": float(np.mean([r[name]["iou"] for r in rows])),
        }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps({"runs": rows, "summary": summary}, indent=2))

    print("\n===== HOLDOUT 2024 (cego) =====")
    print(f"{'modelo':10s} {'LW-SS med':>9s} {'err 120h med':>13s} {'IoU medio':>10s}")
    for name, s in summary.items():
        print(f"{name:10s} {s['lw_ss_median']:9.2f} "
              f"{s['final_err_km_median']:10.1f} km {s['iou_mean']:10.2f}")
    print(f"\n[OK] relatorio -> {REPORT.relative_to(ROOT)}")
    return summary


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    cmd = sys.argv[1] if len(sys.argv) > 1 else "evaluate"
    {"prep": prep, "generate": generate, "evaluate": evaluate}[cmd]()


if __name__ == "__main__":
    main()
