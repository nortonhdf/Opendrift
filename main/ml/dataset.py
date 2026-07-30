"""Build patch-transition samples for the transport surrogate (target a).

One sample = the state of the particle patch at time t plus the local
forcing, and the patch's displacement/spread-change over the next DT_HOURS:

  features : lon_c, lat_c, spread_km, age_h, u_cur, v_cur, u_wind, v_wind, sst
  targets  : dx_km, dy_km (centroid displacement), dspread_km
  block    : "<field>_<season>" — the ONLY valid split unit (members of the
             same field x month share lagged forcing; splitting by sample or
             particle inflates every metric).

Usage (repo root, opendrift env):
    python -m main.ml.dataset                 # scenarios + ensemble manifests
    python -m main.ml.dataset --scenarios-only

Output: main/outputs/ml/patch_dataset.npz  (float32 arrays + block strings)
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

from main.status_utils import status_map  # noqa: E402

DT_HOURS = 6.0            # transition horizon (12 output steps of 1800 s)
OUT_DIR = ROOT / "main" / "outputs" / "ml"

FEATURE_NAMES = ["lon_c", "lat_c", "spread_km", "age_h",
                 "u_cur", "v_cur", "u_wind", "v_wind", "sst"]
TARGET_NAMES = ["dx_km", "dy_km", "dspread_km"]

KM_PER_DEG = 111.32


def patch_state(lon: np.ndarray, lat: np.ndarray) -> tuple[float, float, float]:
    """Centroid (deg) and RMS spread (km) of the active particles at one step."""
    ok = np.isfinite(lon) & np.isfinite(lat)
    if ok.sum() == 0:
        return np.nan, np.nan, np.nan
    lo, la = lon[ok], lat[ok]
    lon_c, lat_c = float(lo.mean()), float(la.mean())
    dx = (lo - lon_c) * KM_PER_DEG * np.cos(np.radians(lat_c))
    dy = (la - lat_c) * KM_PER_DEG
    return lon_c, lat_c, float(np.sqrt(np.mean(dx ** 2 + dy ** 2)))


class ForcingSampler:
    """Nearest-neighbour sampler over the CF forcing files (fast, adequate at
    1/12 deg daily currents and 0.25 deg hourly wind)."""

    def __init__(self, currents_nc: Path, wind_nc: Path):
        self.cur = xr.open_dataset(currents_nc)
        self.wnd = xr.open_dataset(wind_nc)

    def at(self, lon: float, lat: float, when: np.datetime64) -> dict:
        c = self.cur.sel(longitude=lon, latitude=lat, time=when, method="nearest")
        w = self.wnd.sel(longitude=lon, latitude=lat, time=when, method="nearest")
        return {
            "u_cur": float(c["x_sea_water_velocity"].values),
            "v_cur": float(c["y_sea_water_velocity"].values),
            "sst": float(c["sea_water_temperature"].values)
            if "sea_water_temperature" in c else np.nan,
            "u_wind": float(w["x_wind"].values),
            "v_wind": float(w["y_wind"].values),
        }

    def close(self):
        self.cur.close()
        self.wnd.close()


def samples_from_run(nc_path: Path, block: str, sampler: ForcingSampler,
                     dt_hours: float = DT_HOURS):
    """Yield (features, targets) rows for one trajectory file."""
    ds = xr.open_dataset(nc_path)
    lon = np.asarray(ds["lon"].values, float)
    lat = np.asarray(ds["lat"].values, float)
    times = ds["time"].values
    ds.close()

    step_h = (times[1] - times[0]) / np.timedelta64(1, "h")
    stride = int(round(dt_hours / step_h))
    if stride < 1:
        raise ValueError(f"dt_hours={dt_hours} below output step {step_h} h")

    n_t = lon.shape[1]
    rows = []
    for t0 in range(0, n_t - stride, stride):
        lon0, lat0, sp0 = patch_state(lon[:, t0], lat[:, t0])
        lon1, lat1, sp1 = patch_state(lon[:, t0 + stride], lat[:, t0 + stride])
        if not np.isfinite([lon0, lon1, sp0, sp1]).all():
            continue
        env = sampler.at(lon0, lat0, times[t0])
        feats = [lon0, lat0, sp0, t0 * step_h,
                 env["u_cur"], env["v_cur"], env["u_wind"], env["v_wind"],
                 env["sst"]]
        dx = (lon1 - lon0) * KM_PER_DEG * np.cos(np.radians(lat0))
        dy = (lat1 - lat0) * KM_PER_DEG
        rows.append((feats, [dx, dy, sp1 - sp0]))
    return rows


def build(manifests: list[Path], out: Path) -> dict:
    sampler = ForcingSampler(ROOT / "main/inputs/currents.nc",
                             ROOT / "main/inputs/wind_cf.nc")
    X, Y, blocks, sources = [], [], [], []
    n_runs = 0
    for mpath in manifests:
        man = json.loads(mpath.read_text())
        for key, entry in sorted(man.items()):
            block = f"{entry['field']}_{entry['season']}"
            rows = samples_from_run(ROOT / entry["nc"], block, sampler)
            for feats, targs in rows:
                X.append(feats)
                Y.append(targs)
                blocks.append(block)
                sources.append(key)
            n_runs += 1
            print(f"  [{n_runs:4d}] {key:32s} -> {len(rows):3d} amostras "
                  f"(total {len(X)})", flush=True)
    sampler.close()

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        X=np.asarray(X, np.float32),
        Y=np.asarray(Y, np.float32),
        block=np.asarray(blocks),
        source=np.asarray(sources),
        feature_names=np.asarray(FEATURE_NAMES),
        target_names=np.asarray(TARGET_NAMES),
        dt_hours=np.float32(DT_HOURS),
    )
    return {"n_samples": len(X), "n_runs": n_runs,
            "n_blocks": len(set(blocks)), "out": str(out)}


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    p = argparse.ArgumentParser(description="Build patch-transition dataset.")
    p.add_argument("--scenarios-only", action="store_true",
                   help="Use only outputs/scenarios (skip the ensemble).")
    args = p.parse_args()

    manifests = [ROOT / "main/outputs/scenarios/manifest.json"]
    if not args.scenarios_only:
        manifests.append(ROOT / "main/outputs/ensemble/manifest.json")

    print(f"Dataset de transicoes de patch  (DT={DT_HOURS:.0f} h)")
    info = build(manifests, OUT_DIR / "patch_dataset.npz")
    print(f"\n[OK] {info['n_samples']} amostras de {info['n_runs']} runs "
          f"em {info['n_blocks']} blocos -> {info['out']}")


if __name__ == "__main__":
    main()
