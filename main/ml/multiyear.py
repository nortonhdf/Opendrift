"""Multi-year training pipeline for the transport surrogate (v3).

The 2024 blind test showed a single-year-trained correction does not
transfer across years (docs/auditoria/CAMADA_IA.md §5b). This module makes
extra training years reproducible end-to-end:

  download YEAR   — CMEMS currents+SST (anfc; falls back to the GLORYS 'my'
                    reanalysis when the analysis product lacks coverage, and
                    RECORDS which source was used) + ERA5 wind, year-suffixed
  prep YEAR       — CF/NetCDF3 forcing: currents_YYYY.nc, wind_cf_YYYY.nc
  generate YEAR   — training trajectories: 6 fields x 4 months x 10 start
                    days = 240 runs, 200 particles, 120 h (resumable)
  dataset         — multi-year patch-transition dataset with per-sample year
  train-eval      — leave-one-year-out rollout validation (train on the
                    other years, roll out on the held year vs advection),
                    then fit on ALL training years and save the candidate
                    for the final blind 2024 evaluation (holdout.evaluate)

2024 stays frozen: it is NEVER downloaded/used here for training.
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
    FORCING_LAT_MAX, FORCING_LAT_MIN, FORCING_LON_MAX, FORCING_LON_MIN,
    SEASON_MONTHS,
)
from main.fields_config import CAMPOS_FIELDS  # noqa: E402
from main.ml.baselines import predict_advection  # noqa: E402
from main.ml.dataset import (  # noqa: E402
    DT_HOURS, FEATURE_NAMES, TARGET_NAMES, ForcingSampler, samples_from_run,
)
from main.ml.holdout import _truth_track, rollout  # noqa: E402
from main.ml.metrics import haversine_km, liu_weisberg_ss  # noqa: E402

INPUTS = ROOT / "main" / "inputs"
ML_OUT = ROOT / "main" / "outputs" / "ml"
TRAIN_YEARS = [2022, 2023, 2025]
START_DAYS = [1, 4, 7, 10, 13, 16, 19, 22, 25, 28]
N_PARTICLES = 200
DURATION_H = 120
SEED = 42

CMEMS_ANFC = {"cur": "cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m",
              "sst": "cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m"}
CMEMS_MY = "cmems_mod_glo_phy_my_0.083deg_P1D-m"     # GLORYS reanalysis


def forcing_paths(year: int) -> tuple[Path, Path]:
    """CF forcing for a year; 2025 keeps the unsuffixed production names."""
    if year == 2025:
        return INPUTS / "currents.nc", INPUTS / "wind_cf.nc"
    return INPUTS / f"currents_{year}.nc", INPUTS / f"wind_cf_{year}.nc"


class ForcingRegistry:
    """Dispatch sampling to the right year's forcing by timestamp."""

    def __init__(self, years):
        self._samplers = {}
        for y in years:
            cur, wnd = forcing_paths(y)
            self._samplers[y] = ForcingSampler(cur, wnd)

    def at(self, lon, lat, when: np.datetime64) -> dict:
        year = int(str(np.datetime64(when, "Y")))
        return self._samplers[year].at(lon, lat, when)

    def close(self):
        for s in self._samplers.values():
            s.close()


# ── download / prep ──────────────────────────────────────────────────────────

def download(year: int) -> None:
    assert year != 2024, "2024 is the frozen hold-out — never download here"
    import copernicusmarine

    sources = {}
    for tag, variables, anfc_id in [("currents", ["uo", "vo"], CMEMS_ANFC["cur"]),
                                    ("sst", ["thetao"], CMEMS_ANFC["sst"])]:
        fname = f"{tag}_raw_{year}.nc" if tag != "currents" else f"currents_raw_{year}.nc"
        fname = f"sst_raw_{year}.nc" if tag == "sst" else fname
        target = INPUTS / fname
        if target.exists():
            target.unlink()
        used = anfc_id
        try:
            copernicusmarine.subset(
                dataset_id=anfc_id, variables=variables,
                minimum_longitude=FORCING_LON_MIN, maximum_longitude=FORCING_LON_MAX,
                minimum_latitude=FORCING_LAT_MIN, maximum_latitude=FORCING_LAT_MAX,
                start_datetime=f"{year}-01-01T00:00:00",
                end_datetime=f"{year}-12-31T23:00:00",
                minimum_depth=0, maximum_depth=1,
                output_filename=fname, output_directory=str(INPUTS),
            )
        except Exception as e:
            print(f"[INFO] {anfc_id} sem cobertura para {year} "
                  f"({type(e).__name__}) — usando reanálise GLORYS.")
            used = CMEMS_MY
            copernicusmarine.subset(
                dataset_id=CMEMS_MY, variables=variables,
                minimum_longitude=FORCING_LON_MIN, maximum_longitude=FORCING_LON_MAX,
                minimum_latitude=FORCING_LAT_MIN, maximum_latitude=FORCING_LAT_MAX,
                start_datetime=f"{year}-01-01T00:00:00",
                end_datetime=f"{year}-12-31T23:00:00",
                minimum_depth=0, maximum_depth=1,
                output_filename=fname, output_directory=str(INPUTS),
            )
        sources[tag] = used
        print(f"OK: {fname}  (fonte: {used})")

    (INPUTS / f"forcing_source_{year}.json").write_text(
        json.dumps(sources, indent=2))

    import cdsapi
    out = INPUTS / f"wind_raw_{year}.nc"
    cdsapi.Client().retrieve(
        "reanalysis-era5-single-levels",
        {"product_type": "reanalysis", "format": "netcdf",
         "variable": ["10m_u_component_of_wind", "10m_v_component_of_wind"],
         "year": str(year),
         "month": [f"{m:02d}" for m in range(1, 13)],
         "day": [f"{d:02d}" for d in range(1, 32)],
         "time": [f"{h:02d}:00" for h in range(24)],
         "area": [FORCING_LAT_MAX, FORCING_LON_MIN,
                  FORCING_LAT_MIN, FORCING_LON_MAX]},
        str(out))
    print(f"OK: {out.name}")


def prep(year: int) -> None:
    from main.scripts.prep_cmems_currents import prep as prep_cur
    from main.scripts.prep_era5_wind import prep as prep_wnd

    cur_out, wnd_out = forcing_paths(year)
    ds_cur = xr.open_dataset(INPUTS / f"currents_raw_{year}.nc")
    ds_sst = xr.open_dataset(INPUTS / f"sst_raw_{year}.nc")
    out = prep_cur(ds_cur, ds_sst).load()
    out.to_netcdf(cur_out, format="NETCDF3_64BIT",
                  encoding={c: {"_FillValue": None} for c in out.coords})
    print("OK ->", cur_out.name, list(out.data_vars))

    wnd = prep_wnd(xr.open_dataset(INPUTS / f"wind_raw_{year}.nc")).load()
    wnd.to_netcdf(wnd_out, format="NETCDF3_64BIT",
                  encoding={c: {"_FillValue": None} for c in wnd.coords})
    print("OK ->", wnd_out.name, list(wnd.data_vars))


# ── training trajectories ────────────────────────────────────────────────────

def generate(year: int) -> None:
    from main.run_open_oil import run_simulation

    cur, wnd = forcing_paths(year)
    out_dir = ROOT / "main" / "outputs" / f"training_{year}"
    out_dir.mkdir(parents=True, exist_ok=True)
    mpath = out_dir / "manifest.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {}

    tasks = [(f, s, d) for f in CAMPOS_FIELDS
             for s in SEASON_MONTHS for d in START_DAYS]
    done = failed = skipped = 0
    for i, (field, season, day) in enumerate(tasks, 1):
        key = f"{field.lower().replace(' ', '_')}_{season}_d{day:02d}"
        nc_out = out_dir / f"{key}.nc"
        if nc_out.exists() and key in manifest:
            skipped += 1
            continue
        print(f"[{i:03d}/{len(tasks)}] {year} {key} running…", flush=True)
        cfg = CAMPOS_FIELDS[field]
        try:
            run_simulation(
                seed_lon=cfg["lon"], seed_lat=cfg["lat"],
                n_particles=N_PARTICLES,
                start_time=datetime(year, SEASON_MONTHS[season], day),
                duration_hours=DURATION_H, oil_type=cfg["oil_type"],
                use_wind=True, use_waves=False,
                currents_file=str(cur), wind_file=str(wnd), waves_file=None,
                outfile=str(nc_out), figfile=str(out_dir / f"{key}.png"),
                loglevel=40,
            )
            manifest[key] = {"field": field, "season": season, "day": day,
                             "year": year,
                             "nc": str(nc_out.relative_to(ROOT))}
            mpath.write_text(json.dumps(manifest, indent=2))
            done += 1
        except Exception as e:
            print(f"  FAILED {key}: {type(e).__name__}: {e}", flush=True)
            failed += 1
    print(f"\n{year} -> Done: {done}  Skipped: {skipped}  Failed: {failed}")


# ── multi-year dataset ───────────────────────────────────────────────────────

def _year_manifests() -> list[tuple[int, Path]]:
    out = [(2025, ROOT / "main/outputs/scenarios/manifest.json"),
           (2025, ROOT / "main/outputs/ensemble/manifest.json")]
    for y in TRAIN_YEARS:
        if y == 2025:
            continue
        m = ROOT / "main" / "outputs" / f"training_{y}" / "manifest.json"
        if m.exists():
            out.append((y, m))
    return out


def build_dataset() -> None:
    registry = ForcingRegistry(TRAIN_YEARS)
    X, Y, blocks, years = [], [], [], []
    n_runs = 0
    for year, mpath in _year_manifests():
        man = json.loads(mpath.read_text())
        for key, entry in sorted(man.items()):
            block = f"{entry['field']}_{entry['season']}"
            rows = samples_from_run(ROOT / entry["nc"], block, registry)
            for feats, targs in rows:
                X.append(feats)
                Y.append(targs)
                blocks.append(block)
                years.append(year)
            n_runs += 1
            if n_runs % 100 == 0:
                print(f"  {n_runs} runs, {len(X)} amostras…", flush=True)
    registry.close()

    out = ML_OUT / "patch_dataset_multi.npz"
    np.savez_compressed(
        out,
        X=np.asarray(X, np.float32), Y=np.asarray(Y, np.float32),
        block=np.asarray(blocks), year=np.asarray(years, np.int32),
        feature_names=np.asarray(FEATURE_NAMES),
        target_names=np.asarray(TARGET_NAMES),
        dt_hours=np.float32(DT_HOURS),
    )
    print(f"[OK] {len(X)} amostras de {n_runs} runs "
          f"(anos: {sorted(set(years))}) -> {out.name}")


# ── leave-one-year-out rollout validation + final candidate ──────────────────

def _rollout_eval_year(fn, year: int, manifest_path: Path,
                       registry: ForcingRegistry, max_runs: int = 72) -> dict:
    man = json.loads(manifest_path.read_text())
    keys = sorted(man)[:max_runs]
    ss, err = [], []
    for key in keys:
        entry = man[key]
        ds = xr.open_dataset(ROOT / entry["nc"])
        step_h = (ds["time"].values[1] - ds["time"].values[0]) / np.timedelta64(1, "h")
        ds.close()
        stride = int(round(DT_HOURS / step_h))
        tl, tb, _, _, _, t0 = _truth_track(ROOT / entry["nc"], stride)
        ml, mb, _ = rollout(fn, registry, tl[0], tb[0], t0, len(tl) - 1)
        ss.append(liu_weisberg_ss(tl, tb, ml, mb))
        err.append(float(haversine_km(tl[-1], tb[-1], ml[-1], mb[-1])))
    return {"lw_ss_median": float(np.median(ss)),
            "final_err_km_median": float(np.median(err)),
            "n_runs": len(keys)}


def train_eval() -> None:
    import joblib

    from main.ml.train import HGB_PARAMS, fit_model, predict

    d = np.load(ML_OUT / "patch_dataset_multi.npz")
    X, Y, years = d["X"], d["Y"], d["year"]
    adv = predict_advection(X, float(d["dt_hours"]))
    R = Y - adv                                  # residual targets

    registry = ForcingRegistry(TRAIN_YEARS)
    eval_manifests = {
        2022: ROOT / "main/outputs/training_2022/manifest.json",
        2023: ROOT / "main/outputs/training_2023/manifest.json",
        2025: ROOT / "main/outputs/ensemble/manifest.json",
    }

    def make_fn(models):
        def fn(F):
            corr = np.column_stack([m.predict(F) for m in models])
            return predict_advection(F, DT_HOURS) + corr
        return fn

    def adv_fn(F):
        return predict_advection(F, DT_HOURS)

    print("=== Leave-one-year-out (rollout 120 h no ano deixado de fora) ===")
    results = {}
    for held in TRAIN_YEARS:
        tr = years != held
        models = fit_model(X[tr], R[tr])
        m_model = _rollout_eval_year(make_fn(models), held,
                                     eval_manifests[held], registry)
        m_adv = _rollout_eval_year(adv_fn, held, eval_manifests[held], registry)
        results[held] = {"adv+corr": m_model, "advection": m_adv}
        print(f"  {held}: adv+corr SS={m_model['lw_ss_median']:.2f} "
              f"err={m_model['final_err_km_median']:.1f} km | "
              f"advection SS={m_adv['lw_ss_median']:.2f} "
              f"err={m_adv['final_err_km_median']:.1f} km "
              f"({m_model['n_runs']} runs)", flush=True)
    registry.close()

    print("\nFit final (todos os anos de treino) -> candidato para o cego 2024")
    models = fit_model(X, R)
    joblib.dump(models, ML_OUT / "surrogate_hgb_residual.joblib")
    (ML_OUT / "surrogate_hgb_residual.json").write_text(json.dumps({
        "mode": "residual_over_advection_multiyear",
        "train_years": TRAIN_YEARS,
        "n_samples": int(len(X)),
        "seed": SEED,
        "params": HGB_PARAMS,
        "loyo_rollout": {str(k): v for k, v in results.items()},
    }, indent=2))
    print("[OK] surrogate_hgb_residual.joblib atualizado (multi-ano). "
          "Rode: python -m main.ml.holdout evaluate")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    cmd = sys.argv[1]
    if cmd in ("download", "prep", "generate"):
        {"download": download, "prep": prep, "generate": generate}[cmd](int(sys.argv[2]))
    elif cmd == "dataset":
        build_dataset()
    elif cmd == "train-eval":
        train_eval()
    else:
        raise SystemExit(f"comando desconhecido: {cmd}")


if __name__ == "__main__":
    main()
