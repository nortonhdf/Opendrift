"""Footprint dataset: which grid cells the slick actually oils, by D+n.

Agenda v6 item 1. The scenario layer (scenario.py / forecast.py) predicts
where the CENTROID of the slick goes; what the app has to draw — and what a
responder actually asks for — is the set of cells the oil touches.

Two design decisions, both forced by measurement rather than taste:

1. **Swept area, not snapshot.** Measured on the 168-h archive, the
   instantaneous patch is tiny: RMS spread 0.35 km at D+1 and 1.25 km at
   D+7, bounding box ~6 km. On the 0.1 deg grid of the app (11.1 km) that is
   1-2 cells — predicting it would be the centroid problem again, wearing a
   grid. The CUMULATIVE footprint (every cell visited up to D+n) has real
   shape: 7 cells at D+1, 40 at D+7, and it is exactly the quantity the risk
   tab already shows as `prob_any`. Both are stored here; `swept` is the
   target, `snap` is kept so the degeneracy stays measurable.

2. **Release-relative frame, isotropic cells.** Offsets in km from the
   release point, cells of GRID_RES degrees of latitude (11.132 km) so the
   frame matches the resolution of the app while staying square. Translation
   invariance is the whole point: a model that reads absolute lon/lat cannot
   transfer to a location it never saw, which is the deployment case
   (docs/auditoria/CAMADA_IA.md 5e, Resultado 2).

Extent is +-500 km around the release; the measured maximum displacement at
D+7 is ~360 km. Particles beyond the extent are COUNTED and reported, never
silently dropped.

Usage (repo root, opendrift env):
    python -m main.ml.footprint            # train years -> footprint_dataset.npz
    python -m main.ml.footprint --holdout  # 2024      -> footprint_dataset_2024.npz
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

from main.domain_config import GRID_RES  # noqa: E402
from main.ml.dataset import KM_PER_DEG  # noqa: E402
from main.ml.multiyear import TRAIN_DIR_TMPL, TRAIN_YEARS  # noqa: E402
from main.ml.scenario import HORIZONS_D  # noqa: E402
from main.status_utils import ACTIVE, code_of  # noqa: E402

ML_OUT = ROOT / "main" / "outputs" / "ml"

# Square cell of GRID_RES degrees of latitude: same scale as the app grid.
CELL_KM = GRID_RES * KM_PER_DEG          # 11.132 km
HALF_CELLS = 45                          # -> +-500.9 km around the release
N_SIDE = 2 * HALF_CELLS                  # 90 x 90 = 8100 candidate cells

KINDS = ("swept", "snap")


# ── relative grid geometry ───────────────────────────────────────────────────

def cell_of(dx_km, dy_km) -> tuple[np.ndarray, np.ndarray]:
    """Cell id for km offsets from the release point, plus an in-extent mask.

    Ids are row-major over the (dy, dx) grid: ``id = iy * N_SIDE + ix``.
    Points outside the extent get id -1 and False; callers must account for
    them rather than dropping them silently.
    """
    dx = np.asarray(dx_km, float)
    dy = np.asarray(dy_km, float)
    with np.errstate(invalid="ignore"):
        ix = np.floor(np.nan_to_num(dx, nan=1e9) / CELL_KM).astype(np.int64) + HALF_CELLS
        iy = np.floor(np.nan_to_num(dy, nan=1e9) / CELL_KM).astype(np.int64) + HALF_CELLS
    inside = ((ix >= 0) & (ix < N_SIDE) & (iy >= 0) & (iy < N_SIDE)
              & np.isfinite(dx) & np.isfinite(dy))
    ids = np.where(inside, iy * N_SIDE + ix, -1)
    return ids, inside


def cell_offsets_km(ids) -> tuple[np.ndarray, np.ndarray]:
    """Cell-centre offsets (dx_km, dy_km) for cell ids."""
    ids = np.asarray(ids, np.int64)
    ix = ids % N_SIDE
    iy = ids // N_SIDE
    return ((ix - HALF_CELLS + 0.5) * CELL_KM,
            (iy - HALF_CELLS + 0.5) * CELL_KM)


def cells_to_lonlat(ids, lon0: float, lat0: float):
    """Map cell ids back to geographic centres, for drawing on the app map."""
    dx, dy = cell_offsets_km(ids)
    return (lon0 + dx / (KM_PER_DEG * np.cos(np.radians(lat0))),
            lat0 + dy / KM_PER_DEG)


# ── target extraction ────────────────────────────────────────────────────────

def masks_from_run(nc_path: Path, horizons=HORIZONS_D) -> dict:
    """Swept and snapshot cell sets at each horizon, in the release frame.

    Only ACTIVE particles contribute (a stranded element stops adding cells
    at its stranding step, which is the physically right behaviour), and the
    status code is decoded per file — codes vary between runs (audit
    finding #1).
    """
    ds = xr.open_dataset(nc_path)
    lon = np.asarray(ds["lon"].values, float)
    lat = np.asarray(ds["lat"].values, float)
    status = np.asarray(ds["status"].values)
    active_code = code_of(ds["status"], ACTIVE)
    times = ds["time"].values
    ds.close()
    if active_code is None:
        active_code = 0

    lon0 = float(np.nanmean(lon[:, 0]))
    lat0 = float(np.nanmean(lat[:, 0]))
    step_h = (times[1] - times[0]) / np.timedelta64(1, "h")

    live = (status == active_code) & np.isfinite(lon) & np.isfinite(lat)
    dx = (lon - lon0) * KM_PER_DEG * np.cos(np.radians(lat0))
    dy = (lat - lat0) * KM_PER_DEG
    ids, inside = cell_of(dx, dy)
    ok = live & inside

    out = {"lon0": lon0, "lat0": lat0,
           "n_out_of_extent": int((live & ~inside).sum()),
           "swept": {}, "snap": {}}
    for h in horizons:
        idx = int(round(h * 24 / step_h))
        if idx >= lon.shape[1]:                     # run shorter than horizon
            out["swept"][h] = None
            out["snap"][h] = None
            continue
        out["swept"][h] = np.unique(ids[:, :idx + 1][ok[:, :idx + 1]])
        out["snap"][h] = np.unique(ids[:, idx][ok[:, idx]])
    return out


# ── packing / loading ────────────────────────────────────────────────────────

def _pack(seqs: list) -> tuple[np.ndarray, np.ndarray]:
    """CSR-style packing: concatenated ids + row pointers."""
    lens = [0 if s is None else len(s) for s in seqs]
    ptr = np.zeros(len(seqs) + 1, np.int64)
    ptr[1:] = np.cumsum(lens)
    parts = [s for s in seqs if s is not None and len(s)]
    flat = np.concatenate(parts) if parts else np.zeros(0, np.int64)
    return flat.astype(np.int32), ptr


class FootprintSet:
    """Read-side view of a footprint dataset (sparse cell ids per scenario)."""

    def __init__(self, path: Path):
        d = np.load(path, allow_pickle=True)
        self.uid = d["uid"]
        self.field = d["field"]
        self.season = d["season"]
        self.year = d["year"]
        self.lon0 = d["lon0"]
        self.lat0 = d["lat0"]
        self.horizons = [int(h) for h in d["horizons_d"]]
        self.cell_km = float(d["cell_km"])
        self.n_side = int(d["n_side"])
        self._d = {(k, h): (d[f"{k}_cells_d{h}"], d[f"{k}_ptr_d{h}"])
                   for k in KINDS for h in self.horizons}

    def __len__(self) -> int:
        return len(self.uid)

    def cells(self, i: int, h: int, kind: str = "swept") -> np.ndarray:
        flat, ptr = self._d[(kind, h)]
        return flat[ptr[i]:ptr[i + 1]].astype(np.int64)

    def sizes(self, h: int, kind: str = "swept") -> np.ndarray:
        _, ptr = self._d[(kind, h)]
        return np.diff(ptr)

    def valid(self, h: int, kind: str = "swept") -> np.ndarray:
        """Scenarios whose run actually reached this horizon.

        A live run always oils at least the release cell, so an empty row can
        only mean the run ended before D+h — the counterpart of the NaN
        targets scenario.py writes in the same situation.
        """
        return self.sizes(h, kind) > 0


def _archives(holdout: bool = False, grid_years=None) -> list:
    """Same archives the scenario builder reads, six-field or seed-grid."""
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


def build(holdout: bool = False, grid_years=None) -> dict:
    swept: dict = {h: [] for h in HORIZONS_D}
    snap: dict = {h: [] for h in HORIZONS_D}
    uid, field, season, year, lon0, lat0 = [], [], [], [], [], []
    n_out_total = 0

    for y, mpath in _archives(holdout, grid_years):
        man = json.loads(mpath.read_text())
        for key, entry in sorted(man.items()):
            m = masks_from_run(ROOT / entry["nc"])
            if m["swept"][HORIZONS_D[0]] is None:       # need at least D+1
                continue
            for h in HORIZONS_D:
                swept[h].append(m["swept"][h])
                snap[h].append(m["snap"][h])
            uid.append(f"{y}:{key}")
            field.append(entry["field"])
            season.append(entry["season"])
            year.append(y)
            lon0.append(m["lon0"])
            lat0.append(m["lat0"])
            n_out_total += m["n_out_of_extent"]
        print(f"  {mpath.parent.name}: acumulado {len(uid)} cenários", flush=True)

    arrays = {}
    for k, src in (("swept", swept), ("snap", snap)):
        for h in HORIZONS_D:
            flat, ptr = _pack(src[h])
            arrays[f"{k}_cells_d{h}"] = flat
            arrays[f"{k}_ptr_d{h}"] = ptr

    name = ("footprint_dataset_grid.npz" if grid_years
            else "footprint_dataset_2024.npz" if holdout
            else "footprint_dataset.npz")
    out = ML_OUT / name
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        uid=np.asarray(uid), field=np.asarray(field),
        season=np.asarray(season), year=np.asarray(year, np.int32),
        lon0=np.asarray(lon0, np.float64), lat0=np.asarray(lat0, np.float64),
        horizons_d=np.asarray(HORIZONS_D, np.int32),
        cell_km=np.float64(CELL_KM), n_side=np.int32(N_SIDE),
        half_cells=np.int32(HALF_CELLS),
        n_out_of_extent=np.int64(n_out_total),
        **arrays,
    )
    # A run that ends before a horizon contributes an EMPTY row there, and the
    # pointer array is what marks it — the same convention scenario.py uses
    # when it writes NaN targets. Counted and reported, never averaged in.
    med, missing = {}, {}
    for h in HORIZONS_D:
        lens = [len(s) for s in swept[h] if s is not None]
        med[h] = float(np.median(lens)) if lens else float("nan")
        missing[h] = int(sum(1 for s in swept[h] if s is None))
    print(f"[OK] {len(uid)} cenários -> {out.name}")
    print("     células varridas (mediana): "
          + "  ".join(f"D+{h}={med[h]:.0f}" for h in HORIZONS_D))
    if any(missing.values()):
        print("     runs curtos demais para o horizonte: "
              + "  ".join(f"D+{h}={missing[h]}" for h in HORIZONS_D
                          if missing[h]))
    print(f"     partículas fora do quadro de ±{HALF_CELLS * CELL_KM:.0f} km: "
          f"{n_out_total}")
    return {"n": len(uid), "out": str(out), "median_cells": med,
            "missing_by_horizon": missing, "n_out_of_extent": n_out_total}


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    p = argparse.ArgumentParser(description="Build the footprint dataset.")
    p.add_argument("--holdout", action="store_true",
                   help="Build from the frozen 2024 archive instead.")
    p.add_argument("--grid", type=int, nargs="*", metavar="ANO",
                   help="Build from the seed-location archives of these years.")
    args = p.parse_args()
    build(holdout=args.holdout, grid_years=args.grid)


if __name__ == "__main__":
    main()
