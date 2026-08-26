"""Why the anisotropic plume lost, in one fold: three hypotheses, one table.

The plume draws an empirical 2-D shape around the path the v4 centroid model
predicts. At a location it never saw, it lost to the isotropic corridor
(CAMADA_IA.md 5f). Three explanations were tested; this script reproduces the
evidence for the one that survived.

  H1  bins finer than a grid cell        -> rejected in the full evaluation:
                                            coarsening left LOFO IoU at D+1
                                            unchanged (0.464 -> 0.469)
  H2  the predicted bearing is wrong,    -> rejected AND contradicted: the
      and an anisotropic shape               plume loses most where the
      misaligns worse than a round one       bearing is best (below)
  H3  normalising the axes by the        -> survives: dropping it recovers
      PREDICTED displacement injects         most of the gap (below)
      the model's own error into them

H3 is what the module now implements. The verdict it does NOT change: even
with the coordinate fixed, anisotropy ties the corridor and never beats it,
which is why the corridor is the shipped shape.

Usage (repo root, opendrift env):
    python main/scripts/_plume_frames.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import main.ml.footprint_forecast as ff  # noqa: E402
from main.ml.forecast import tcol  # noqa: E402
from main.ml.scenario import HORIZONS_D  # noqa: E402

HELD = "Marlim"
CAL_YEAR = 2025

# The superseded frame, kept here so the comparison stays reproducible.
OLD_A = np.arange(-0.6, 2.201, 0.05)
OLD_C = np.arange(-1.0, 1.001, 0.05)


def old_frame(path, cdx, cdy):
    """Along/across the chord, DIVIDED by the predicted displacement (v1)."""
    end = np.asarray(path, float)[-1]
    L = float(np.hypot(end[0], end[1]))
    if not np.isfinite(L) or L < 1e-6:
        return None
    ex, ey = end[0] / L, end[1] / L
    return (cdx * ex + cdy * ey) / L, (-cdx * ey + cdy * ex) / L


def old_fit(paths, T, ok, cdx, cdy):
    num = np.zeros((len(OLD_A) - 1, len(OLD_C) - 1))
    den = np.zeros_like(num)
    for r, p in enumerate(paths):
        if not ok[r]:
            continue
        fr = old_frame(p, cdx, cdy)
        if fr is None:
            continue
        a, c = fr
        t = T[r].astype(float)
        m = ((a >= OLD_A[0]) & (a < OLD_A[-1])
             & (c >= OLD_C[0]) & (c < OLD_C[-1]))
        num += np.histogram2d(a[m], c[m], bins=[OLD_A, OLD_C],
                              weights=t[m])[0]
        den += np.histogram2d(a[m], c[m], bins=[OLD_A, OLD_C])[0]
    base = float(num.sum() / max(den.sum(), 1))
    return np.where(den > 0, num / np.maximum(den, 1), base), base


def old_predict(kernel, base, paths, cdx, cdy, fallback):
    out = np.empty((len(paths), len(cdx)), np.float32)
    for r, p in enumerate(paths):
        fr = old_frame(p, cdx, cdy)
        if fr is None:
            out[r] = fallback[r]
            continue
        a, c = fr
        ia = np.clip(np.searchsorted(OLD_A, a, side="right") - 1, 0,
                     len(OLD_A) - 2)
        ic = np.clip(np.searchsorted(OLD_C, c, side="right") - 1, 0,
                     len(OLD_C) - 2)
        m = ((a >= OLD_A[0]) & (a < OLD_A[-1])
             & (c >= OLD_C[0]) & (c < OLD_C[-1]))
        out[r] = np.where(m, kernel[ia, ic], base)
    return out


def main() -> None:
    ctx = ff.load_pair(ff.FP_TRAIN, ff.SC_TRAIN)
    rows = np.arange(len(ctx["uid"]))
    te = rows[ctx["field"] == HELD]
    tr = rows[ctx["field"] != HELD]
    fit = tr[ctx["year"][tr] != CAL_YEAR]
    cal = tr[ctx["year"][tr] == CAL_YEAR]
    cmods = ff.centroid_models(ctx, fit)

    print(f"Fold: {HELD} retido, ajuste {sorted(set(ctx['year'][fit].tolist()))}, "
          f"calibracao {CAL_YEAR}\n")
    print("H3 — a normalizacao pelo deslocamento previsto e o defeito")
    print(f"{'h':>3}{'modelo':>22s}{'IoU':>9s}{'area 80% (km2)':>17s}")

    store = {}
    for h in HORIZONS_D:
        radius = ff.candidate_radius(ctx, fit, h)
        cand, cdx, cdy = ff.candidates(radius)
        T_cal = ff.truth_matrix(ctx, cal, h, cand)
        T_te = ff.truth_matrix(ctx, te, h, cand)
        ok_cal, ok_te = ff.valid_rows(ctx, cal, h), ff.valid_rows(ctx, te, h)
        P_cal = ff.derive_dist(ff.predict_models(cmods, ctx["X"][cal]))
        P_te = ff.derive_dist(ff.predict_models(cmods, ctx["X"][te]))
        paths_cal = ff.paths_from_centroid(P_cal, h)
        paths_te = ff.paths_from_centroid(P_te, h)

        d_cal = [ff.dist_to_path(p, cdx, cdy) for p in paths_cal]
        iso = ff.fit_distance_calibration(
            [d for d, o in zip(d_cal, ok_cal) if o], T_cal[ok_cal])
        pc_cal = np.array([iso.predict(d) for d in d_cal], np.float32)
        pc_te = np.array([iso.predict(ff.dist_to_path(p, cdx, cdy))
                          for p in paths_te], np.float32)

        ko, base = old_fit(paths_cal, T_cal, ok_cal, cdx, cdy)
        po_cal = old_predict(ko, base, paths_cal, cdx, cdy, pc_cal)
        po_te = old_predict(ko, base, paths_te, cdx, cdy, pc_te)

        kk = ff.fit_plume_kernel(paths_cal, T_cal, ok_cal, cdx, cdy)
        pk_cal = ff.predict_plume(kk, paths_cal, cdx, cdy, pc_cal)
        pk_te = ff.predict_plume(kk, paths_te, cdx, cdy, pc_te)

        for name, cp, tp in (("corredor", pc_cal, pc_te),
                             ("pluma normalizada", po_cal, po_te),
                             ("pluma em km", pk_cal, pk_te)):
            thr = ff.pick_threshold(cp[ok_cal], T_cal[ok_cal])
            iou = float(ff.iou_rows(tp[ok_te], T_te[ok_te], thr).mean())
            area = float(np.nanmedian(
                ff.capture_area_rows(tp[ok_te], T_te[ok_te])))
            print(f"{h:>3}{name:>22s}{iou:9.3f}{area:17.0f}", flush=True)
            if name != "pluma normalizada":
                store.setdefault(h, {})[name] = (tp, thr)
        print()

    # H2 — does the plume only lose when the predicted bearing is wrong?
    h = HORIZONS_D[0]
    hi = HORIZONS_D.index(h)
    radius = ff.candidate_radius(ctx, fit, h)
    cand, cdx, cdy = ff.candidates(radius)
    T_te = ff.truth_matrix(ctx, te, h, cand)
    ok_te = ff.valid_rows(ctx, te, h)
    Y = ctx["Y"][te][ok_te]
    P = ff.derive_dist(ff.predict_models(cmods, ctx["X"][te][ok_te]))
    at = np.arctan2(Y[:, tcol(hi, "dy_km")], Y[:, tcol(hi, "dx_km")])
    ap = np.arctan2(P[:, tcol(hi, "dy_km")], P[:, tcol(hi, "dx_km")])
    err = np.degrees(np.abs(np.arctan2(np.sin(at - ap), np.cos(at - ap))))

    tp_k, thr_k = store[h]["pluma em km"]
    tp_c, thr_c = store[h]["corredor"]
    diff = (ff.iou_rows(tp_k[ok_te], T_te[ok_te], thr_k)
            - ff.iou_rows(tp_c[ok_te], T_te[ok_te], thr_c))
    print(f"H2 — pluma menos corredor por erro de rumo (D+{h})")
    print(f"{'erro angular':>16s}{'n':>6s}{'pluma-corredor':>17s}")
    for a, b in ((0, 15), (15, 30), (30, 60), (60, 180)):
        m = (err >= a) & (err < b)
        if m.sum() >= 3:
            print(f"{f'{a}-{b} graus':>16s}{m.sum():6d}{diff[m].mean():+17.3f}")
    print(f"\ncorrelacao (erro angular, pluma-corredor): "
          f"r = {np.corrcoef(err, diff)[0, 1]:+.3f}")
    print("Se a pluma perdesse por desalinhamento, perderia MAIS com rumo "
          "ruim. Nao e o caso.")


if __name__ == "__main__":
    main()
