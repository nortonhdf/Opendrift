"""Footprint forecasting: which cells get oiled, with a calibrated probability.

The scenario layer answers "where will the centroid be" (forecast.py). This
one answers the question the app actually draws and a responder actually
asks: **which cells does the oil touch by D+n, and how sure are we?**

Inputs are still release-time only (position, oil, season, antecedent ocean
state) — no future forcing. Target is the SWEPT footprint in the
release-relative frame (footprint.py explains why swept and not snapshot).

Because a single deterministic ribbon is not a usable product at these lead
times (the centroid alone is already ~80 km off at D+7, seven cells wide),
every model here outputs a PROBABILITY per cell. That makes the honest
metrics probabilistic:

  Brier / BSS   : calibration + sharpness against the individual outcome
  IoU@threshold : the deterministic reading, at an operating point chosen on
                  calibration data, never on the test set
  capture area  : km^2 that must be searched, taking the most likely cells
                  first, to cover 80 % of the cells that actually got oiled.
                  This is the operational currency of the whole layer.

Competitors, in increasing order of ambition:

  climatology : per-season cell frequency in the relative frame
  persistence : corridor along the antecedent 7-day mean current
  analogue    : k-NN over scenario features, footprint frequency of the k
  centroid    : corridor around the path predicted by the v4 centroid model
                — the control that matters: does a footprint model add
                anything over the centroid model we already have?
  plume       : same predicted path, but with the shape of the oiled region
                measured rather than assumed (2-D empirical kernel)
  occupancy   : direct per-cell classifier (scenario features + cell offset)

Protocol: fit on 2022+2023, calibrate the corridor/plume shape AND every
IoU operating point on the held-out year 2025, evaluate on the frozen blind
2024 archive and on leave-one-field-out. Scenarios of the same year share the
ocean state, so the calibration split is by YEAR. Models without a two-stage
structure are fitted on fit+cal so that every row of the table consumes the
same 720 scenarios — see run_fold for why that asymmetry is the safe one.

Usage (repo root, opendrift env):
    python -m main.ml.footprint_forecast
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from main.ml.baselines import KM_PER_H_PER_MS  # noqa: E402
from main.ml.footprint import (  # noqa: E402
    CELL_KM, FootprintSet, cell_offsets_km, cells_to_lonlat,
)
from main.ml.forecast import derive_dist, fit_hgb, predict_models, tcol  # noqa: E402
from main.ml.scenario import HORIZONS_D  # noqa: E402

ML_OUT = ROOT / "main" / "outputs" / "ml"
FP_TRAIN = ML_OUT / "footprint_dataset.npz"
FP_BLIND = ML_OUT / "footprint_dataset_2024.npz"
SC_TRAIN = ML_OUT / "scenario_dataset.npz"
SC_BLIND = ML_OUT / "scenario_dataset_2024.npz"
REPORT = ML_OUT / "footprint_report.json"
SCORES = ML_OUT / "footprint_scores.npz"
PRODUCT = ML_OUT / "footprint_plume.joblib"
RELIABILITY = ML_OUT / "footprint_reliability.json"
SEED = 42

# Per-cell classifier. Smaller than the regressors of forecast.py because the
# row count is ~500x larger (one row per candidate cell, not per scenario).
OCC = dict(max_iter=200, learning_rate=0.1, min_samples_leaf=20,
           l2_regularization=1.0, random_state=SEED)
N_NEG = 500              # negatives sampled per scenario, inverse-weighted
K_ANALOGUE = 25          # neighbours whose footprints are pooled
CAPTURE = 0.8            # fraction of the true footprint the area must cover
THRESHOLDS = np.round(np.linspace(0.02, 0.9, 45), 4)
CAND_MARGIN_KM = 3 * CELL_KM

MODELS = ["climatology", "persistence", "analogue", "centroid", "plume",
          "occupancy"]

# Which shape the exported product draws by default. Set from the
# leave-one-field-out evaluation (a new location is the deployment case),
# not from the blind year, where a per-field climatology has already seen
# the site. Both shapes ship inside the artefact either way.
PRODUCT_MODEL = "centroid"


# ── joining the two datasets ─────────────────────────────────────────────────

def load_pair(fp_path: Path, sc_path: Path):
    """Footprint masks + scenario features for the SAME runs, in one order.

    The two builders skip runs independently (a run too short for D+1), so
    the join is on the run identity, never on position.
    """
    fp = FootprintSet(fp_path)
    d = np.load(sc_path, allow_pickle=True)
    sc_uid = np.array([f"{y}:{k}" for y, k in zip(d["year"], d["run_key"])])
    pos = {u: i for i, u in enumerate(sc_uid)}
    keep_fp, keep_sc = [], []
    for i, u in enumerate(fp.uid):
        j = pos.get(str(u))
        if j is not None:
            keep_fp.append(i)
            keep_sc.append(j)
    if not keep_fp:
        raise SystemExit(f"Nenhum run em comum entre {fp_path.name} e "
                         f"{sc_path.name} — reconstrua os dois datasets.")
    keep_fp = np.asarray(keep_fp)
    keep_sc = np.asarray(keep_sc)
    return {
        "fp": fp, "row": keep_fp,
        "X": d["X"][keep_sc], "Y": d["Y"][keep_sc],
        "feature_names": d["feature_names"],
        "field": fp.field[keep_fp], "season": fp.season[keep_fp],
        "year": fp.year[keep_fp], "uid": fp.uid[keep_fp],
        "lon0": fp.lon0[keep_fp], "lat0": fp.lat0[keep_fp],
    }


# ── candidate cells and truth ────────────────────────────────────────────────

def cells_of(ctx, i: int, h: float) -> np.ndarray:
    """Swept cells of joined row i — the join reindexes, so never index fp directly."""
    return ctx["fp"].cells(int(ctx["row"][i]), h)


def candidate_radius(ctx, rows, h: float) -> float:
    """How far from the release any training footprint ever reached, + margin.

    Measured on TRAINING rows only: the candidate set must not be shaped by
    the test outcomes it will be scored against.
    """
    rmax = 0.0
    for i in rows:
        cells = cells_of(ctx, i, h)
        if len(cells) == 0:
            continue
        dx, dy = cell_offsets_km(cells)
        rmax = max(rmax, float(np.hypot(dx, dy).max()))
    return rmax + CAND_MARGIN_KM


def candidates(radius_km: float):
    """Cell ids within radius of the release, with their centre offsets."""
    n_side = int(np.ceil(radius_km / CELL_KM)) * 2 + 2
    from main.ml.footprint import HALF_CELLS, N_SIDE
    half = min(n_side // 2, HALF_CELLS)
    ix = np.arange(HALF_CELLS - half, HALF_CELLS + half)
    gx, gy = np.meshgrid(ix, ix, indexing="xy")
    ids = (gy * N_SIDE + gx).ravel()
    dx, dy = cell_offsets_km(ids)
    keep = np.hypot(dx, dy) <= radius_km
    return ids[keep], dx[keep], dy[keep]


def truth_matrix(ctx, rows, h: float, cand_ids: np.ndarray) -> np.ndarray:
    """(n_rows, n_cand) boolean: did this scenario oil this cell by D+h?"""
    order = np.argsort(cand_ids)
    sorted_ids = cand_ids[order]
    T = np.zeros((len(rows), len(cand_ids)), bool)
    for r, i in enumerate(rows):
        cells = cells_of(ctx, i, h)
        if len(cells) == 0:
            continue
        # Cells outside the candidate radius exist (a footprint may reach
        # further than any training one did); searchsorted would map them to
        # a neighbour, so the equality check is what discards them.
        pos = np.clip(np.searchsorted(sorted_ids, cells), 0,
                      len(sorted_ids) - 1)
        hit = sorted_ids[pos] == cells
        T[r, order[pos[hit]]] = True
    return T


def valid_rows(ctx, rows, h: float) -> np.ndarray:
    """Rows whose run actually reached D+h (empty mask = run ended earlier)."""
    return np.array([len(cells_of(ctx, i, h)) > 0 for i in rows])


# ── geometry: corridors around a predicted path ──────────────────────────────

def dist_to_path(path_xy: np.ndarray, cx: np.ndarray, cy: np.ndarray):
    """Distance (km) from each candidate cell centre to a predicted polyline.

    The footprint is swept along the whole path, not just its endpoint, so
    the corridor baselines must measure distance to the PATH — using the
    endpoint alone would predict a blob where the truth is a ribbon.
    """
    d = np.full(cx.shape, np.inf)
    pts = np.atleast_2d(path_xy)
    if len(pts) == 1:
        return np.hypot(cx - pts[0, 0], cy - pts[0, 1])
    for (x1, y1), (x2, y2) in zip(pts[:-1], pts[1:]):
        vx, vy = x2 - x1, y2 - y1
        L2 = vx * vx + vy * vy
        t = (((cx - x1) * vx + (cy - y1) * vy) / L2 if L2 > 0
             else np.zeros_like(cx))
        t = np.clip(t, 0.0, 1.0)
        d = np.minimum(d, np.hypot(cx - (x1 + t * vx), cy - (y1 + t * vy)))
    return d


def paths_persistence(X, feat_names, h: float) -> list:
    """Straight path along the antecedent 7-day mean current (the null)."""
    names = list(feat_names)
    u = X[:, names.index("u_mean_7d")]
    v = X[:, names.index("v_mean_7d")]
    out = []
    for k in range(len(X)):
        end = np.array([u[k] * KM_PER_H_PER_MS * h * 24,
                        v[k] * KM_PER_H_PER_MS * h * 24])
        end = np.nan_to_num(end)            # missing window -> stays put
        out.append(np.vstack([[0.0, 0.0], end]))
    return out


def paths_from_centroid(P: np.ndarray, h: float) -> list:
    """Polyline through the v4 centroid predictions up to D+h."""
    steps = [hh for hh in HORIZONS_D if hh <= h]
    out = []
    for k in range(len(P)):
        pts = [[0.0, 0.0]]
        for hh in steps:
            hi = HORIZONS_D.index(hh)
            pts.append([P[k, tcol(hi, "dx_km")], P[k, tcol(hi, "dy_km")]])
        out.append(np.asarray(pts, float))
    return out


def path_frame(path_xy: np.ndarray, cdx: np.ndarray, cdy: np.ndarray):
    """Cell coordinates along and across the predicted path, both in km.

    ``along`` is arc length measured on the predicted polyline; ``cross`` is
    the signed offset from it. Beyond the ends the first and last segment
    directions are extended, so a footprint that overshoots the predicted
    endpoint lands past the end of the axis instead of piling onto the last
    bin. Returns None when the model predicts no displacement at all, in
    which case there is no frame and the caller falls back.

    **Why km and not fractions of the predicted displacement.** The first
    version of this kernel divided both coordinates by the predicted
    displacement L. That was measured and rejected (CAMADA_IA.md 5h): L is
    itself a model output with error, so dividing by it injects the model's
    own uncertainty into the axis — when L is under-predicted the true
    footprint runs off to twice the normalised length, and pooling such cases
    smears probability along the whole axis. The corridor never paid that
    price because it measures distance to the drawn path in km, and the path
    already carries L. Dropping the normalisation recovered most of the gap.
    """
    pts = np.asarray(path_xy, float)
    if len(pts) < 2 or not np.isfinite(pts).all():
        return None
    if float(np.hypot(*(pts[-1] - pts[0]))) < 1e-6:
        return None
    best_d = np.full(np.shape(cdx), np.inf)
    best_s = np.zeros_like(best_d)
    best_c = np.zeros_like(best_d)
    acc = 0.0
    for i, ((x1, y1), (x2, y2)) in enumerate(zip(pts[:-1], pts[1:])):
        vx, vy = x2 - x1, y2 - y1
        seg = float(np.hypot(vx, vy))
        if seg < 1e-9:
            continue
        t = ((cdx - x1) * vx + (cdy - y1) * vy) / (seg * seg)
        lo = -np.inf if i == 0 else 0.0
        hi = np.inf if i == len(pts) - 2 else 1.0
        tc = np.clip(t, lo, hi)
        px, py = x1 + tc * vx, y1 + tc * vy
        d = np.hypot(cdx - px, cdy - py)
        take = d < best_d
        best_d = np.where(take, d, best_d)
        best_s = np.where(take, acc + tc * seg, best_s)
        best_c = np.where(take, ((cdx - x1) * vy - (cdy - y1) * vx) / seg,
                          best_c)
        acc += seg
    return best_s, best_c, acc


def plume_bins(radius_km: float):
    """Kernel bin edges, one grid cell wide, covering the candidate disc.

    One cell is the finest resolution the target can express, and the
    candidate radius is the furthest any cell can sit from the release, so
    these edges always contain the frame.
    """
    r = float(radius_km) + CELL_KM
    edges = np.arange(-r, r + CELL_KM, CELL_KM)
    return edges, edges.copy()


def fit_plume_kernel(paths: list, T: np.ndarray, ok: np.ndarray,
                     cdx, cdy) -> dict:
    """Empirical P(oiled | along, cross) around the predicted displacement.

    This is the corridor idea with the shape LEARNED instead of assumed: the
    isotonic corridor can only make probability fall with distance from the
    path, whereas the real uncertainty is anisotropic — longer along the
    track (timing) than across it (direction). Measured verdict: it does not
    pay. Even with the coordinate defect fixed it ties the corridor and never
    beats it (CAMADA_IA.md 5h), so it stays here as the control that answers
    "does anisotropy earn its place", not as the product.
    """
    edges_a, edges_c = plume_bins(float(np.hypot(cdx, cdy).max()))
    num = np.zeros((len(edges_a) - 1, len(edges_c) - 1))
    den = np.zeros_like(num)
    out_hit = out_tot = 0.0
    for r, p in enumerate(paths):
        if not ok[r]:
            continue
        fr = path_frame(p, cdx, cdy)
        if fr is None:
            continue
        a, c, _ = fr
        t = T[r].astype(float)
        inside = ((a >= edges_a[0]) & (a < edges_a[-1])
                  & (c >= edges_c[0]) & (c < edges_c[-1]))
        num += np.histogram2d(a[inside], c[inside], bins=[edges_a, edges_c],
                              weights=t[inside])[0]
        den += np.histogram2d(a[inside], c[inside],
                              bins=[edges_a, edges_c])[0]
        out_hit += float(t[~inside].sum())
        out_tot += float((~inside).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        P = np.where(den > 0, num / np.maximum(den, 1), np.nan)
    base = float(num.sum() / max(den.sum(), 1))
    return {"P": np.nan_to_num(P, nan=base), "base": base,
            "outside": float(out_hit / out_tot) if out_tot else 0.0,
            "edges_a": edges_a, "edges_c": edges_c}


def predict_plume(kernel: dict, paths: list, cdx, cdy,
                  fallback: np.ndarray) -> np.ndarray:
    """Look the kernel up for each scenario; fall back where the frame fails."""
    edges_a, edges_c = kernel["edges_a"], kernel["edges_c"]
    out = np.empty((len(paths), len(cdx)), np.float32)
    for r, p in enumerate(paths):
        fr = path_frame(p, cdx, cdy)
        if fr is None:
            out[r] = fallback[r]
            continue
        a, c, _ = fr
        ia = np.clip(np.searchsorted(edges_a, a, side="right") - 1,
                     -1, len(edges_a) - 2)
        ic = np.clip(np.searchsorted(edges_c, c, side="right") - 1,
                     -1, len(edges_c) - 2)
        inside = ((a >= edges_a[0]) & (a < edges_a[-1])
                  & (c >= edges_c[0]) & (c < edges_c[-1]))
        vals = kernel["P"][np.clip(ia, 0, None), np.clip(ic, 0, None)]
        out[r] = np.where(inside, vals, kernel["outside"])
    return out


def fit_distance_calibration(dists: list, truths: np.ndarray) -> IsotonicRegression:
    """Monotone distance -> P(oiled), fitted on calibration scenarios.

    A corridor is a geometry, not a probability. Isotonic regression turns
    "how far is this cell from the predicted path" into a calibrated
    likelihood, without assuming a shape for the decay.
    """
    d = np.concatenate(dists)
    y = truths.ravel().astype(float)
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=False,
                             out_of_bounds="clip")
    iso.fit(d, y)
    return iso


# ── models ───────────────────────────────────────────────────────────────────

def _uv_idx(feat_names) -> tuple:
    names = list(feat_names)
    return names.index("u_mean_7d"), names.index("v_mean_7d")


_CENTROID_CACHE: dict = {}


def centroid_models(ctx, rows_fit: np.ndarray):
    """The v4 centroid regressors for one training split.

    One fit serves every horizon (the model is multi-target), so it is cached
    per split — otherwise the same 20 regressors would be refitted five times
    per fold for nothing.
    """
    key = (id(ctx), rows_fit.tobytes())
    if key not in _CENTROID_CACHE:
        _CENTROID_CACHE[key] = fit_hgb(ctx["X"][rows_fit], ctx["Y"][rows_fit])
    return _CENTROID_CACHE[key]


GEOM_NAMES = ["cell_dx_km", "cell_dy_km", "cell_r_km", "cell_sin", "cell_cos",
              "cell_down_km", "cell_cross_km"]


def occupancy_features(X_row: np.ndarray, cdx: np.ndarray, cdy: np.ndarray,
                       iu: int, iv: int):
    """One row per candidate cell: scenario features + where the cell is.

    Offsets are release-relative (km and polar), so the rule the model learns
    transfers to a release point it never saw. The last two columns give the
    same offset rotated into the frame of the antecedent 7-day current —
    downstream and cross-stream distance. Trees split on axes: without this
    rotation they would have to rediscover a 2-D interaction between (u, v)
    and (dx, dy) from scratch, which is the same mistake the patch surrogate
    made when it was asked to learn integration error (CAMADA_IA.md 5c).
    A missing antecedent window leaves these NaN, which HGB consumes.
    """
    r = np.hypot(cdx, cdy)
    ang = np.arctan2(cdy, cdx)
    u, v = float(X_row[iu]), float(X_row[iv])
    speed = np.hypot(u, v)
    if speed > 0:
        down = (cdx * u + cdy * v) / speed
        cross = (cdx * -v + cdy * u) / speed
    else:
        down = np.full_like(cdx, np.nan)
        cross = np.full_like(cdx, np.nan)
    geom = np.column_stack([cdx, cdy, r, np.sin(ang), np.cos(ang), down, cross])
    return np.hstack([np.repeat(X_row[None, :], len(cdx), axis=0), geom])



def sample_cells(pos: np.ndarray, n_cand: int, rng, n_neg: int = N_NEG):
    """Keep every positive cell, sample negatives, weight them back up.

    Returns (selected cell indices, labels, weights). The weights are what
    keep the fitted probabilities on the natural scale: a negative that stood
    for ``n_pool / n_neg`` unsampled negatives carries that much weight, so
    the model still sees the true ~1 % base rate. Brier and the capture area
    are read directly off those probabilities, so this cannot be skipped.
    """
    neg_pool = np.setdiff1d(np.arange(n_cand), pos)
    k = min(n_neg, len(neg_pool))
    neg = rng.choice(neg_pool, size=k, replace=False) if k else neg_pool[:0]
    w_neg = len(neg_pool) / k if k else 1.0
    sel = np.concatenate([pos, neg]).astype(int)
    labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    weights = np.concatenate([np.ones(len(pos)), np.full(len(neg), w_neg)])
    return sel, labels, weights


def fit_occupancy(ctx, rows, h: float, cand_ids, cdx, cdy, rng):
    """Per-cell classifier with inverse-probability weighted negatives.

    Positives are ~1 % of the candidate set, so training on every cell would
    be 4 000 rows per scenario of almost-all zeros. We keep all positives and
    a uniform sample of N_NEG negatives, then weight the negatives by the
    inverse sampling rate — the fitted probabilities stay on the natural
    scale, which is what Brier and the capture area are read on.
    """
    X = ctx["X"]
    iu, iv = _uv_idx(ctx["feature_names"])
    T = truth_matrix(ctx, rows, h, cand_ids)
    ok = valid_rows(ctx, rows, h)
    n_cand = len(cand_ids)
    feats, labels, weights = [], [], []
    for r, i in enumerate(rows):
        if not ok[r]:
            continue
        sel, lab, w = sample_cells(np.flatnonzero(T[r]), n_cand, rng)
        feats.append(occupancy_features(X[r], cdx[sel], cdy[sel], iu, iv))
        labels.append(lab)
        weights.append(w)
    Xtr = np.vstack(feats)
    ytr = np.concatenate(labels)
    wtr = np.concatenate(weights)
    clf = HistGradientBoostingClassifier(**OCC)
    clf.fit(Xtr, ytr, sample_weight=wtr)
    return clf


def predict_occupancy(clf, ctx, rows, cdx, cdy) -> np.ndarray:
    X = ctx["X"]
    iu, iv = _uv_idx(ctx["feature_names"])
    out = np.empty((len(rows), len(cdx)), np.float32)
    for r in range(len(rows)):
        out[r] = clf.predict_proba(
            occupancy_features(X[r], cdx, cdy, iu, iv))[:, 1]
    return out


def predict_frequency(T_ref: np.ndarray, groups_ref, groups_te,
                      ok_ref: np.ndarray) -> np.ndarray:
    """Cell-frequency climatology within a group (season), global fallback."""
    table = {}
    for g in np.unique(groups_ref):
        m = (groups_ref == g) & ok_ref
        if m.any():
            table[g] = T_ref[m].mean(axis=0)
    glob = T_ref[ok_ref].mean(axis=0)
    return np.array([table.get(g, glob) for g in groups_te], np.float32)


def predict_analogue(X_ref, T_ref, ok_ref, X_te, k=K_ANALOGUE) -> np.ndarray:
    """Footprint frequency over the k most similar historical scenarios."""
    ref = np.flatnonzero(ok_ref)
    mu = np.nanmean(X_ref[ref], axis=0)
    sd = np.nanstd(X_ref[ref], axis=0)
    sd[sd == 0] = 1.0
    A = (X_ref[ref] - mu) / sd
    out = np.empty((len(X_te), T_ref.shape[1]), np.float32)
    for i, b in enumerate((X_te - mu) / sd):
        d = np.nansum((A - b) ** 2, axis=1)
        nn = ref[np.argsort(d)[:k]]
        out[i] = T_ref[nn].mean(axis=0)
    return out


# ── scoring ──────────────────────────────────────────────────────────────────

def brier_rows(P: np.ndarray, T: np.ndarray) -> np.ndarray:
    return ((P - T.astype(np.float32)) ** 2).mean(axis=1)


def iou_rows(P: np.ndarray, T: np.ndarray, thr: float) -> np.ndarray:
    pred = P >= thr
    inter = (pred & T).sum(axis=1)
    union = (pred | T).sum(axis=1)
    return np.where(union > 0, inter / np.maximum(union, 1), 1.0)


def capture_area_rows(P: np.ndarray, T: np.ndarray,
                      target: float = CAPTURE, rng=None) -> np.ndarray:
    """km^2 to search, most-likely cells first, to cover `target` of the truth.

    NaN when the scenario has no oiled cell at this horizon. This is the
    number an operator can act on: an area, not a score.

    ``rng`` breaks ties at random. It matters because cell ids run row-major
    from the south-west corner, which is the direction the Brazil Current
    carries oil — leaving ties to index order would quietly rank the likely
    side first for whichever model has the coarsest probabilities. Default
    (None) keeps index order; see the sensitivity check in the report.
    """
    n = T.sum(axis=1)
    out = np.full(len(P), np.nan)
    if rng is not None:
        # float64 first: in float32 a 1e-9 nudge rounds away and ties survive.
        P = P.astype(np.float64) + rng.uniform(0, 1e-9, size=P.shape)
    order = np.argsort(-P, axis=1, kind="stable")
    hits = np.take_along_axis(T, order, axis=1).cumsum(axis=1)
    need = np.ceil(target * n).astype(int)
    for r in range(len(P)):
        if n[r] == 0:
            continue
        idx = np.searchsorted(hits[r], need[r], side="left")
        out[r] = (idx + 1) * CELL_KM * CELL_KM
    return out


def pick_threshold(P: np.ndarray, T: np.ndarray) -> float:
    """Operating point that maximises mean IoU on CALIBRATION data."""
    best, best_iou = THRESHOLDS[0], -1.0
    for t in THRESHOLDS:
        v = float(np.mean(iou_rows(P, T, t)))
        if v > best_iou:
            best, best_iou = float(t), v
    return best


def summarize(P, T, thr: float, ref_brier=None) -> dict:
    b = brier_rows(P, T)
    i = iou_rows(P, T, thr)
    a = capture_area_rows(P, T)
    out = {"brier": float(np.mean(b)), "iou": float(np.mean(i)),
           "threshold": thr, "capture_area_km2": float(np.nanmedian(a)),
           "n": int(len(P))}
    if ref_brier is not None:
        out["bss_vs_climatology"] = float(1 - np.mean(b) / np.mean(ref_brier))
    return out


# ── one evaluation pass ──────────────────────────────────────────────────────

def run_fold(ctx_tr, rows_fit, rows_cal, ctx_te, rows_te, h: float,
             rng, return_preds: bool = False) -> dict:
    """Fit on rows_fit, calibrate on rows_cal, score on rows_te (any context).

    On who gets to see what — the asymmetry matters and is deliberate. The
    corridor and plume models NEED the split: their shape is only honest when
    measured on paths a centroid model produced for scenarios it never
    trained on. The models that have no such two-stage structure
    (climatology, analogue, occupancy) are therefore fitted on fit+cal, so
    every model in the table consumes the same scenarios. Their operating
    point is then chosen on data they were fitted on, which flatters them,
    not the plume — and the headline metric (Brier) uses no threshold at all.

    Returns per-model per-scenario arrays so the caller can pool folds and
    run paired tests on the same scenarios.
    """
    radius = candidate_radius(ctx_tr, rows_fit, h)
    cand_ids, cdx, cdy = candidates(radius)
    rows_all = np.concatenate([rows_fit, rows_cal])

    T_fit = truth_matrix(ctx_tr, rows_fit, h, cand_ids)
    T_all = truth_matrix(ctx_tr, rows_all, h, cand_ids)
    T_cal = truth_matrix(ctx_tr, rows_cal, h, cand_ids)
    T_te = truth_matrix(ctx_te, rows_te, h, cand_ids)
    ok_all = valid_rows(ctx_tr, rows_all, h)
    ok_cal = valid_rows(ctx_tr, rows_cal, h)
    ok_te = valid_rows(ctx_te, rows_te, h)

    X_fit, X_cal = ctx_tr["X"][rows_fit], ctx_tr["X"][rows_cal]
    X_all, X_te = ctx_tr["X"][rows_all], ctx_te["X"][rows_te]
    s_all, s_cal = ctx_tr["season"][rows_all], ctx_tr["season"][rows_cal]
    s_te = ctx_te["season"][rows_te]
    names = ctx_tr["feature_names"]

    preds_cal, preds_te = {}, {}

    # climatology — cell frequency by season, in the release frame
    preds_cal["climatology"] = predict_frequency(T_all, s_all, s_cal, ok_all)
    preds_te["climatology"] = predict_frequency(T_all, s_all, s_te, ok_all)

    # analogue — k-NN footprint frequency
    preds_cal["analogue"] = predict_analogue(X_all, T_all, ok_all, X_cal)
    preds_te["analogue"] = predict_analogue(X_all, T_all, ok_all, X_te)

    # persistence corridor — isotonic on the calibration year
    d_cal = [dist_to_path(p, cdx, cdy)
             for p in paths_persistence(X_cal, names, h)]
    iso = fit_distance_calibration([d for d, o in zip(d_cal, ok_cal) if o],
                                   T_cal[ok_cal])
    preds_cal["persistence"] = np.array([iso.predict(d) for d in d_cal],
                                        np.float32)
    preds_te["persistence"] = np.array(
        [iso.predict(dist_to_path(p, cdx, cdy))
         for p in paths_persistence(X_te, names, h)], np.float32)

    # centroid corridor — the v4 model, then isotonic on the same held-out year
    cmods = centroid_models(ctx_tr, rows_fit)
    P_cal_c = derive_dist(predict_models(cmods, X_cal))
    P_te_c = derive_dist(predict_models(cmods, X_te))
    paths_cal_c = paths_from_centroid(P_cal_c, h)
    paths_te_c = paths_from_centroid(P_te_c, h)
    dc_cal = [dist_to_path(p, cdx, cdy) for p in paths_cal_c]
    iso_c = fit_distance_calibration(
        [d for d, o in zip(dc_cal, ok_cal) if o], T_cal[ok_cal])
    preds_cal["centroid"] = np.array([iso_c.predict(d) for d in dc_cal],
                                     np.float32)
    preds_te["centroid"] = np.array(
        [iso_c.predict(dist_to_path(p, cdx, cdy)) for p in paths_te_c],
        np.float32)

    # plume — same v4 path, but the shape of the oiled region is measured on
    # the calibration year instead of assumed to fall off with distance
    kernel = fit_plume_kernel(paths_cal_c, T_cal, ok_cal, cdx, cdy)
    preds_cal["plume"] = predict_plume(kernel, paths_cal_c, cdx, cdy,
                                       preds_cal["climatology"])
    preds_te["plume"] = predict_plume(kernel, paths_te_c, cdx, cdy,
                                      preds_te["climatology"])

    # occupancy — the direct per-cell model
    clf = fit_occupancy(ctx_tr, rows_all, h, cand_ids, cdx, cdy, rng)
    preds_cal["occupancy"] = predict_occupancy(clf, ctx_tr, rows_cal, cdx, cdy)
    preds_te["occupancy"] = predict_occupancy(clf, ctx_te, rows_te, cdx, cdy)

    out = {"radius_km": radius, "n_cand": int(len(cand_ids)),
           "per_model": {}, "ok_te": ok_te}
    for name in MODELS:
        thr = pick_threshold(preds_cal[name][ok_cal], T_cal[ok_cal])
        out["per_model"][name] = {
            "threshold": thr,
            "brier": brier_rows(preds_te[name][ok_te], T_te[ok_te]),
            "iou": iou_rows(preds_te[name][ok_te], T_te[ok_te], thr),
            "area": capture_area_rows(preds_te[name][ok_te], T_te[ok_te]),
        }
    if return_preds:
        out["preds_te"] = {n: preds_te[n][ok_te] for n in MODELS}
        out["T_te"] = T_te[ok_te]
    return out


def pooled_report(pool: dict, ref: str = "climatology") -> dict:
    """Aggregate per-scenario arrays into the reported numbers."""
    out = {}
    ref_b = np.concatenate(pool[ref]["brier"])
    for name in MODELS:
        b = np.concatenate(pool[name]["brier"])
        i = np.concatenate(pool[name]["iou"])
        a = np.concatenate(pool[name]["area"])
        out[name] = {
            "brier": float(np.mean(b)),
            "bss_vs_climatology": float(1 - np.mean(b) / np.mean(ref_b)),
            "iou": float(np.mean(i)),
            "capture_area_km2": float(np.nanmedian(a)),
            "threshold": float(np.median(pool[name]["threshold"])),
            "n": int(len(b)),
        }
    return out


def paired(pool: dict, a: str, b: str, key: str) -> dict:
    x = np.concatenate(pool[a][key])
    y = np.concatenate(pool[b][key])
    ok = np.isfinite(x) & np.isfinite(y)
    if not ok.any() or np.allclose(x[ok], y[ok]):
        return {"p": 1.0, "median_delta": 0.0, "wins": 0, "n": int(ok.sum())}
    better = x[ok] > y[ok] if key == "iou" else x[ok] < y[ok]
    return {"p": float(stats.wilcoxon(x[ok], y[ok]).pvalue),
            "median_delta": float(np.median(x[ok] - y[ok])),
            "wins": int(better.sum()), "n": int(ok.sum())}


def paired_block(pool: dict, ref: str) -> dict:
    """Every model against one reference, on all three metrics.

    The metrics genuinely disagree here — a probability field can be no
    better calibrated (Brier) while ranking cells much better (capture area)
    — so a claim on one of them has to be tested on its own, not inherited.
    """
    return {name: {k: paired(pool, name, ref, k)
                   for k in ("iou", "brier", "area")}
            for name in MODELS if name != ref}


def _new_pool() -> dict:
    return {m: {"brier": [], "iou": [], "area": [], "threshold": []}
            for m in MODELS}


def _accumulate(pool: dict, fold: dict) -> None:
    for name in MODELS:
        r = fold["per_model"][name]
        pool[name]["brier"].append(r["brier"])
        pool[name]["iou"].append(r["iou"])
        pool[name]["area"].append(r["area"])
        pool[name]["threshold"].append(r["threshold"])


# ── evaluations ──────────────────────────────────────────────────────────────

def evaluate_blind(ctx, ctx_h, cal_year: int, rng, scores: dict) -> dict:
    rows = np.arange(len(ctx["uid"]))
    fit = rows[ctx["year"] != cal_year]
    cal = rows[ctx["year"] == cal_year]
    rows_h = np.arange(len(ctx_h["uid"]))
    res = {}
    print(f"\n=== Cego 2024 ({len(rows_h)} cenários) — ajuste em "
          f"{sorted(set(ctx['year'][fit].tolist()))}, calibração em {cal_year} ===")
    for h in HORIZONS_D:
        t0 = time.time()
        fold = run_fold(ctx, fit, cal, ctx_h, rows_h, h, rng)
        pool = _new_pool()
        _accumulate(pool, fold)
        rep = pooled_report(pool)
        rep["_meta"] = {"radius_km": fold["radius_km"],
                        "n_candidate_cells": fold["n_cand"]}
        rep["_paired_vs_climatology"] = paired_block(pool, "climatology")
        rep["_paired_vs_centroid"] = paired_block(pool, "centroid")
        scores["blind"][h] = pool
        res[f"D+{h}"] = rep
        print(f"  D+{h} ({fold['n_cand']} células candidatas, "
              f"{time.time() - t0:.0f}s)")
        print(f"    {'modelo':12s}{'Brier':>10s}{'BSS':>8s}{'IoU':>8s}"
              f"{'área 80% (km²)':>16s}")
        for name in MODELS:
            r = rep[name]
            print(f"    {name:12s}{r['brier']:10.5f}"
                  f"{r['bss_vs_climatology']:8.2f}{r['iou']:8.3f}"
                  f"{r['capture_area_km2']:16.0f}")
    return res


def evaluate_lofo(ctx, cal_year: int, rng, scores: dict) -> dict:
    """Leave-one-FIELD-out: forecast a footprint where nothing was ever seen.

    Same logic as forecast.py — this is the deployment case, and the only
    setting where a per-field climatology cannot quietly do the work. Note
    what it does and does not hold out: space, not time. The blind 2024
    evaluation is the one that holds out the ocean state.
    """
    rows = np.arange(len(ctx["uid"]))
    fields = ctx["field"]
    res = {}
    print("\n=== Leave-one-FIELD-out (local nunca visto) ===")
    for h in HORIZONS_D:
        t0 = time.time()
        pool = _new_pool()
        by_field = {}
        for held in sorted(set(fields.tolist())):
            te = rows[fields == held]
            tr = rows[fields != held]
            fit = tr[ctx["year"][tr] != cal_year]
            cal = tr[ctx["year"][tr] == cal_year]
            fold = run_fold(ctx, fit, cal, ctx, te, h, rng)
            _accumulate(pool, fold)
            one = _new_pool()
            _accumulate(one, fold)
            by_field[held] = pooled_report(one)
        rep = pooled_report(pool)
        rep["_by_field"] = by_field
        rep["_paired_vs_climatology"] = paired_block(pool, "climatology")
        rep["_paired_vs_centroid"] = paired_block(pool, "centroid")
        scores["lofo"][h] = pool
        res[f"D+{h}"] = rep
        print(f"  D+{h} ({time.time() - t0:.0f}s)")
        print(f"    {'modelo':12s}{'Brier':>10s}{'BSS':>8s}{'IoU':>8s}"
              f"{'área 80% (km²)':>16s}")
        for name in MODELS:
            r = rep[name]
            print(f"    {name:12s}{r['brier']:10.5f}"
                  f"{r['bss_vs_climatology']:8.2f}{r['iou']:8.3f}"
                  f"{r['capture_area_km2']:16.0f}")
        for name in ("plume", "occupancy"):
            pv = rep["_paired_vs_climatology"][name]
            print(f"    {name} vs climatologia: IoU Δ={pv['iou']['median_delta']:+.3f} "
                  f"p={pv['iou']['p']:.1e} | área Δ={pv['area']['median_delta']:+.0f} km² "
                  f"p={pv['area']['p']:.1e} ({pv['area']['wins']}/{pv['area']['n']})")
    return res


def dump_scores(scores: dict, path: Path) -> None:
    """Per-scenario metric arrays, so a follow-up analysis needs no refit."""
    flat = {}
    for split, per_h in scores.items():
        for h, pool in per_h.items():
            for name in MODELS:
                for key in ("brier", "iou", "area"):
                    flat[f"{split}_d{h}_{name}_{key}"] = np.concatenate(
                        pool[name][key])
    np.savez_compressed(path, **flat)


# ── the deployable product ───────────────────────────────────────────────────

def export_product(ctx, cal_year: int, out: Path = PRODUCT) -> tuple:
    """Persist the shapes, calibrated against the SHIPPED centroid models.

    The centroid models come from the v4 product (main.ml.forecast --export),
    they are not refitted here. That is the whole point: the corridor and the
    plume are calibrated on paths drawn by the very models the app will use,
    so the band around a predicted point belongs to that point. Refitting
    would give the app a track its corridor was never measured against, and
    would duplicate 30 MB of identical regressors in git.

    Not "the best possible fit on all data" either: the shape is only honest
    when the paths it was measured on came from a model that never saw those
    scenarios, which is why the v4 product is fitted on 2022+2023 and
    everything here is calibrated on 2025.
    """
    import joblib

    fc = load_forecast_product()
    if fc["calibration_year"] != cal_year:
        raise SystemExit(
            f"Produto v4 calibrado em {fc['calibration_year']}, mas este "
            f"dataset calibra em {cal_year} — regere os dois.")

    rows = np.arange(len(ctx["uid"]))
    fit = rows[ctx["year"] != cal_year]
    cal = rows[ctx["year"] == cal_year]
    P_cal = derive_dist(predict_models(fc["point_models"], ctx["X"][cal]))
    rows_all = np.concatenate([fit, cal])
    seasons_all, seasons_cal = ctx["season"][rows_all], ctx["season"][cal]

    payload = {
        "feature_names": [str(f) for f in ctx["feature_names"]],
        "horizons_d": HORIZONS_D, "cell_km": CELL_KM,
        "kernels": {},   # each kernel carries the bin edges it was fitted at
        "fit_years": sorted(set(ctx["year"][fit].tolist())),
        "calibration_year": int(cal_year),
        "default_model": PRODUCT_MODEL,
        "centroid_from": "forecast_product.joblib",
    }
    for h in HORIZONS_D:
        radius = candidate_radius(ctx, fit, h)
        cand_ids, cdx, cdy = candidates(radius)
        T_all = truth_matrix(ctx, rows_all, h, cand_ids)
        T_cal = truth_matrix(ctx, cal, h, cand_ids)
        ok_all, ok_cal = valid_rows(ctx, rows_all, h), valid_rows(ctx, cal, h)
        paths = paths_from_centroid(P_cal, h)
        kernel = fit_plume_kernel(paths, T_cal, ok_cal, cdx, cdy)
        d_cal = [dist_to_path(p, cdx, cdy) for p in paths]
        iso = fit_distance_calibration([d for d, o in zip(d_cal, ok_cal) if o],
                                       T_cal[ok_cal])
        clim = {str(s): predict_frequency(T_all, seasons_all,
                                          np.array([s]), ok_all)[0]
                for s in np.unique(seasons_all)}
        fb = predict_frequency(T_all, seasons_all, seasons_cal, ok_all)
        # Both shapes travel with the product; the evaluation picks the
        # default, so swapping it is a documented decision and not an edit.
        thr = {
            "plume": pick_threshold(
                predict_plume(kernel, paths, cdx, cdy, fb)[ok_cal],
                T_cal[ok_cal]),
            "centroid": pick_threshold(
                np.array([iso.predict(d) for d in d_cal], np.float32)[ok_cal],
                T_cal[ok_cal]),
        }
        payload["kernels"][h] = {
            "P": kernel["P"], "outside": kernel["outside"],
            "base": kernel["base"], "radius_km": radius,
            "cand_ids": cand_ids.astype(np.int32), "isotonic": iso,
            "climatology": clim, "threshold": thr,
        }
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, out)
    print(f"[OK] formas -> {out.relative_to(ROOT)} "
          f"(centróide vem de {payload['centroid_from']})")
    return payload, fc


def load_forecast_product():
    """The v4 layer this one draws its tracks from (fit it if absent)."""
    import joblib

    from main.ml import forecast

    if not forecast.PRODUCT.exists():
        print(f"[..] {forecast.PRODUCT.name} não existe; ajustando agora "
              f"(python -m main.ml.forecast --export)")
        return forecast.export_product()
    return joblib.load(forecast.PRODUCT)


def predict_footprint(payload, fc_payload, x_row: np.ndarray, h: int,
                      lon0: float, lat0: float, season: str,
                      model: str = None) -> dict:
    """Oiling probability per cell for ONE release — what the app draws.

    ``payload`` holds the shapes, ``fc_payload`` the v4 models that place
    them (main.ml.forecast). ``x_row`` is a scenario feature vector in the
    order of ``payload["feature_names"]`` (build it with main.ml.scenario).
    Returns geographic cell centres, their probability, and the evaluated
    operating point, so the caller can outline a deterministic footprint if
    it wants one.
    """
    k = payload["kernels"][h]
    name = model or payload.get("default_model", "centroid")
    cand_ids = k["cand_ids"].astype(np.int64)
    cdx, cdy = cell_offsets_km(cand_ids)
    P_c = derive_dist(predict_models(fc_payload["point_models"],
                                     np.asarray(x_row, np.float32)[None, :]))
    path = paths_from_centroid(P_c, h)[0]
    if name == "centroid":
        prob = np.asarray(k["isotonic"].predict(dist_to_path(path, cdx, cdy)),
                          np.float32)
    else:
        clim = k["climatology"].get(str(season),
                                    next(iter(k["climatology"].values())))
        kernel = {"P": k["P"], "outside": k["outside"], "base": k["base"],
                  "edges_a": k["edges_a"], "edges_c": k["edges_c"]}
        prob = predict_plume(kernel, [path], cdx, cdy, clim[None, :])[0]
    thr = k["threshold"]
    lon, lat = cells_to_lonlat(cand_ids, lon0, lat0)
    return {"lon": lon, "lat": lat, "prob": prob, "model": name,
            "threshold": thr[name] if isinstance(thr, dict) else thr,
            "path_dx_km": path[:, 0], "path_dy_km": path[:, 1]}


RELIABILITY_BINS = np.array([0.0, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70,
                             0.90, 1.0001])


def evaluate_reliability(ctx_h, payload, fc_payload) -> dict:
    """Does a cell drawn at 30 % get oiled 30 % of the time? (blind year)

    A probability map is only usable if its numbers mean what they say, and
    the scenario layer already learned that lesson the hard way: raw quantile
    boosters promised 80 % coverage and delivered 35-49 % (CAMADA_IA.md 5e,
    Resultado 3). This runs THROUGH the exported product, so it also checks
    the artefact the app will load, not just the code path used in training.
    """
    rows = np.arange(len(ctx_h["uid"]))
    res = {}
    print(f"\n=== Confiabilidade do produto ({payload['default_model']}) "
          f"no cego 2024, via artefato exportado ===")
    for h in HORIZONS_D:
        k = payload["kernels"][h]
        cand = k["cand_ids"].astype(np.int64)
        T = truth_matrix(ctx_h, rows, h, cand)
        ok = valid_rows(ctx_h, rows, h)
        P = np.vstack([
            predict_footprint(payload, fc_payload, ctx_h["X"][i], h,
                              float(ctx_h["lon0"][i]), float(ctx_h["lat0"][i]),
                              str(ctx_h["season"][i]))["prob"]
            for i in rows[ok]])
        y = T[ok].ravel()
        p = P.ravel()
        idx = np.clip(np.digitize(p, RELIABILITY_BINS) - 1, 0,
                      len(RELIABILITY_BINS) - 2)
        table = []
        for b in range(len(RELIABILITY_BINS) - 1):
            m = idx == b
            if not m.any():
                continue
            table.append({
                "bin": [float(RELIABILITY_BINS[b]),
                        float(RELIABILITY_BINS[b + 1])],
                "n_cells": int(m.sum()),
                "predicted": float(p[m].mean()),
                "observed": float(y[m].mean()),
            })
        res[f"D+{h}"] = table
        print(f"  D+{h}: " + "  ".join(
            f"{t['predicted']:.2f}->{t['observed']:.2f}" for t in table))
    print("  (previsto -> observado por faixa; iguais = calibrado)")
    return res


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="Footprint forecasting evaluation.")
    ap.add_argument("--skip-lofo", action="store_true",
                    help="Blind evaluation only (faster smoke run).")
    ap.add_argument("--export", action="store_true",
                    help="Fit and persist the product for the app, no evaluation.")
    ap.add_argument("--reliability", action="store_true",
                    help="Export the product, then check its calibration on 2024.")
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    ctx = load_pair(FP_TRAIN, SC_TRAIN)
    ctx_h = load_pair(FP_BLIND, SC_BLIND)
    cal_year = int(max(set(ctx["year"].tolist())))
    print(f"Footprint — {len(ctx['uid'])} cenários de treino "
          f"{sorted(set(ctx['year'].tolist()))}, {len(ctx_h['uid'])} no cego, "
          f"células de {CELL_KM:.1f} km, horizontes {HORIZONS_D}")
    med = {h: float(np.median(ctx["fp"].sizes(h)[ctx["row"]])) for h in HORIZONS_D}
    print("Alvo (área varrida, mediana de células): "
          + "  ".join(f"D+{h}={med[h]:.0f}" for h in HORIZONS_D))

    if args.export or args.reliability:
        payload, fc_payload = export_product(ctx, cal_year)
        if args.reliability:
            rel = evaluate_reliability(ctx_h, payload, fc_payload)
            RELIABILITY.write_text(json.dumps(
                {"blind_2024": rel, "model": payload["default_model"],
                 "bins": RELIABILITY_BINS.tolist()}, indent=2))
            print(f"[OK] confiabilidade -> {RELIABILITY.relative_to(ROOT)}")
        return

    scores = {"blind": {}, "lofo": {}}
    blind = evaluate_blind(ctx, ctx_h, cal_year, rng, scores)
    lofo = {} if args.skip_lofo else evaluate_lofo(ctx, cal_year, rng, scores)

    REPORT.write_text(json.dumps({
        "blind_2024": blind, "lofo_new_location": lofo,
        "horizons_d": HORIZONS_D, "cell_km": CELL_KM,
        "capture_target": CAPTURE, "calibration_year": cal_year,
        "n_train": int(len(ctx["uid"])), "n_blind": int(len(ctx_h["uid"])),
        "median_swept_cells": med,
    }, indent=2))
    dump_scores(scores, SCORES)
    print(f"\n[OK] relatório -> {REPORT.relative_to(ROOT)}")
    print(f"[OK] métricas por cenário -> {SCORES.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
