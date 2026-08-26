"""Scenario-level dataset: initial conditions -> slick projection at D+n.

This is the project's actual target (author, 2026-08-07): given only what is
known AT RELEASE TIME — where, which oil, which season, and the ocean state
over the preceding days/weeks/months — project the slick over HORIZONS_D
(D+1..D+7 with the 168-h archives; D+14 would need longer runs).

Why this differs from the step-wise surrogate in dataset.py: there the model
was handed the true forcing at every step, so it could only compete with
numerical integration — and integration wins (docs/auditoria/CAMADA_IA.md
§5c). Here NO future forcing is available, so numerical advection is not
even applicable and the honest baselines become climatology, antecedent
persistence and historical analogues.

CAUSALITY RULE: every feature is computed from data strictly BEFORE the
release instant. Lookback windows that fall outside the available forcing
file yield NaN, which HistGradientBoosting consumes natively — never
silently zero-filled.

Usage (repo root, opendrift env):
    python -m main.ml.scenario            # train years -> scenario_dataset.npz
    python -m main.ml.scenario --holdout  # 2024 -> scenario_dataset_2024.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from main.fields_config import CAMPOS_FIELDS  # noqa: E402
from main.ml.dataset import KM_PER_DEG, patch_state  # noqa: E402
from main.ml.multiyear import TRAIN_YEARS, forcing_paths  # noqa: E402

ML_OUT = ROOT / "main" / "outputs" / "ml"

# Antecedent windows: "últimos dias / semanas / meses"
LOOKBACKS_D = [3, 7, 30, 90]
# Forecast horizons in days. 168-h archives support D+7, the working scope
# (author decision 2026-08-07); D+14 deferred.
HORIZONS_D = [1, 2, 3, 5, 7]

TARGET_NAMES = [f"{q}_d{h}" for h in HORIZONS_D
                for q in ("dx_km", "dy_km", "dist_km", "spread_km")]


def feature_names() -> list[str]:
    names = ["lon", "lat", "api", "water_depth_m",
             "month_sin", "month_cos", "doy_sin", "doy_cos"]
    for w in LOOKBACKS_D:
        names += [f"u_mean_{w}d", f"v_mean_{w}d", f"speed_mean_{w}d",
                  f"speed_std_{w}d", f"dir_steadiness_{w}d", f"sst_mean_{w}d",
                  f"coverage_{w}d"]
    return names


def feature_row(lon: float, lat: float, api: float, water_depth_m: float,
                release, sampler) -> np.ndarray:
    """The full feature vector for one release, in ``feature_names()`` order.

    Used by the dataset builder AND by the live predictor behind the app
    (main/ml/predict.py). It exists so the two cannot drift: a feature
    assembled one way at training time and another way at inference time is
    the quietest way to break a deployed model, and the audit already caught
    the same class of bug once with constants re-declared in four files
    (finding #11 -> domain_config.py).
    """
    rel = np.datetime64(release)
    year = int(str(rel)[:4])
    doy = (rel - np.datetime64(f"{year}-01-01")) / np.timedelta64(1, "D")
    month = int(str(rel)[5:7])
    feats = [lon, lat, api, water_depth_m,
             np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12),
             np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)]
    feats += sampler.features(year, lon, lat, rel)
    return np.asarray(feats, np.float32)


class AntecedentSampler:
    """Ocean-state statistics over the N days preceding a release.

    Caches the full-year point time series per (year, field) — 6 fields x a
    handful of years, so the whole archive costs a few dozen extractions.
    """

    def __init__(self, years):
        self._cur = {}
        for y in years:
            cur_path, _ = forcing_paths(y)
            self._cur[y] = xr.open_dataset(cur_path)
        self._cache: dict = {}

    def _series(self, year: int, lon: float, lat: float):
        key = (year, round(lon, 4), round(lat, 4))
        if key not in self._cache:
            ds = self._cur[year].sel(longitude=lon, latitude=lat,
                                     method="nearest")
            u = np.asarray(ds["x_sea_water_velocity"].values, float)
            v = np.asarray(ds["y_sea_water_velocity"].values, float)
            sst = (np.asarray(ds["sea_water_temperature"].values, float)
                   if "sea_water_temperature" in ds else np.full(len(u), np.nan))
            self._cache[key] = (ds["time"].values, u, v, sst)
        return self._cache[key]

    def features(self, year: int, lon: float, lat: float,
                 release: np.datetime64) -> list[float]:
        times, u, v, sst = self._series(year, lon, lat)
        out: list[float] = []
        for w in LOOKBACKS_D:
            start = release - np.timedelta64(w, "D")
            m = (times >= start) & (times < release)   # strictly before release
            n_avail = int(m.sum())
            coverage = n_avail / w                     # daily fields
            if n_avail < max(2, w // 4):               # too thin to trust
                out += [np.nan] * 6 + [coverage]
                continue
            uu, vv = u[m], v[m]
            spd = np.sqrt(uu ** 2 + vv ** 2)
            # Steadiness: |mean vector| / mean speed — 1 = unidirectional
            # flow, ~0 = reversing/eddying. This is what "condições de
            # correntes das últimas semanas" really means for transport.
            mean_speed = float(np.nanmean(spd))
            steadiness = (float(np.hypot(np.nanmean(uu), np.nanmean(vv)))
                          / mean_speed if mean_speed > 0 else np.nan)
            out += [float(np.nanmean(uu)), float(np.nanmean(vv)),
                    mean_speed, float(np.nanstd(spd)), steadiness,
                    float(np.nanmean(sst[m])), coverage]
        return out

    def close(self):
        for ds in self._cur.values():
            ds.close()


def targets_from_run(nc_path: Path) -> tuple[list[float], float, float]:
    """Slick descriptors at each horizon, relative to the release point."""
    ds = xr.open_dataset(nc_path)
    lon = np.asarray(ds["lon"].values, float)
    lat = np.asarray(ds["lat"].values, float)
    times = ds["time"].values
    ds.close()

    lon0, lat0, _ = patch_state(lon[:, 0], lat[:, 0])
    step_h = (times[1] - times[0]) / np.timedelta64(1, "h")
    out: list[float] = []
    for h in HORIZONS_D:
        idx = int(round(h * 24 / step_h))
        if idx >= lon.shape[1]:
            out += [np.nan] * 4
            continue
        lc, tc, spread = patch_state(lon[:, idx], lat[:, idx])
        dx = (lc - lon0) * KM_PER_DEG * np.cos(np.radians(lat0))
        dy = (tc - lat0) * KM_PER_DEG
        out += [dx, dy, float(np.hypot(dx, dy)), spread]
    return out, lon0, lat0


def _archives(holdout: bool = False, grid_years=None) -> list:
    """168-h archives (D+7 scope), one balanced set per year.

    ``grid_years`` switches to the seed-location archives of main.ml.seedgrid
    instead of the six-field ones — same file layout, many more locations.
    """
    from main.ml.multiyear import TRAIN_DIR_TMPL
    from main.ml.seedgrid import GRID_DIR_TMPL

    tmpl = GRID_DIR_TMPL if grid_years else TRAIN_DIR_TMPL
    years = list(grid_years) if grid_years else ([2024] if holdout
                                                 else TRAIN_YEARS)
    out = []
    for y in years:
        m = (ROOT / "main" / "outputs" / tmpl.format(year=y) / "manifest.json")
        if m.exists():
            out.append((y, m))
    if not out:
        how = ("python -m main.ml.seedgrid generate <ano>" if grid_years
               else "python -m main.ml.multiyear generate <ano>")
        raise SystemExit(f"Nenhum arquivo de 168 h encontrado. Gere com: {how}")
    return out


def release_config(entry: dict) -> dict:
    """Release parameters for one manifest entry.

    Six-field archives name a field and look the rest up in fields_config;
    seed-grid archives carry lon/lat/api/depth in the manifest because there
    is no registry entry to look up. Manifest values win, so a future archive
    can override anything without touching this builder.
    """
    cfg = dict(CAMPOS_FIELDS.get(entry["field"], {}))
    for k in ("api", "water_depth_m", "lon", "lat"):
        if k in entry and entry[k] is not None:
            cfg[k] = entry[k]
        elif k in entry:                      # explicit null = unknown
            cfg[k] = float("nan")
    if "api" not in cfg:
        raise KeyError(f"manifesto sem api e campo desconhecido: {entry}")
    cfg.setdefault("water_depth_m", float("nan"))
    return cfg


def build(holdout: bool = False, grid_years=None) -> dict:
    years = (list(grid_years) if grid_years
             else ([2024] if holdout else TRAIN_YEARS))
    sampler = AntecedentSampler(years)
    X, Y, blocks, meta = [], [], [], []

    for year, mpath in _archives(holdout, grid_years):
        man = json.loads(mpath.read_text())
        for key, entry in sorted(man.items()):
            field = entry["field"]
            cfg = release_config(entry)
            nc = ROOT / entry["nc"]
            targs, lon0, lat0 = targets_from_run(nc)
            if not np.isfinite(targs[:4]).all():      # need at least D+1
                continue

            ds = xr.open_dataset(nc)
            release = ds["time"].values[0]
            ds.close()

            X.append(feature_row(lon0, lat0, cfg["api"], cfg["water_depth_m"],
                                 release, sampler))
            Y.append(targs)
            blocks.append(f"{field}_{entry['season']}")
            meta.append((year, field, entry["season"], key))
        print(f"  {mpath.parent.name}: acumulado {len(X)} cenários", flush=True)
    sampler.close()

    name = ("scenario_dataset_grid.npz" if grid_years
            else "scenario_dataset_2024.npz" if holdout
            else "scenario_dataset.npz")
    out = ML_OUT / name
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        X=np.asarray(X, np.float32), Y=np.asarray(Y, np.float32),
        block=np.asarray(blocks),
        year=np.asarray([m[0] for m in meta], np.int32),
        field=np.asarray([m[1] for m in meta]),
        season=np.asarray([m[2] for m in meta]),
        run_key=np.asarray([m[3] for m in meta]),
        feature_names=np.asarray(feature_names()),
        target_names=np.asarray(TARGET_NAMES),
        horizons_d=np.asarray(HORIZONS_D, np.int32),
    )
    nan_frac = float(np.isnan(np.asarray(X, np.float32)).mean())
    print(f"[OK] {len(X)} cenários, {len(feature_names())} features "
          f"({nan_frac:.1%} NaN por falta de janela), "
          f"{len(TARGET_NAMES)} alvos -> {out.name}")
    return {"n": len(X), "out": str(out)}


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    p = argparse.ArgumentParser(description="Build the scenario-level dataset.")
    p.add_argument("--holdout", action="store_true",
                   help="Build from the frozen 2024 archive instead.")
    p.add_argument("--grid", type=int, nargs="*", metavar="ANO",
                   help="Build from the seed-location archives of these years.")
    args = p.parse_args()
    build(holdout=args.holdout, grid_years=args.grid)


if __name__ == "__main__":
    main()
