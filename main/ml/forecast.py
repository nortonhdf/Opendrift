"""Scenario-level slick forecasting: baselines, model, blind evaluation.

Problem (author's target): at release time we know only WHERE, WHICH OIL,
WHICH SEASON and the ocean state over the preceding days/weeks/months. No
future forcing. Project the slick at D+1..D+7 (the working scope since the
168-h archives; D+14 would need longer runs).

Because future currents are unavailable, numerical advection cannot be run —
the honest competitors are:

  climatology : the average outcome for that field & season in other years
  persistence : extrapolate the antecedent mean current over the horizon
  analogue    : copy the outcome of the most similar historical scenario
  HGB         : gradient-boosted trees over all scenario features

Validation is leave-one-YEAR-out (scenarios of the same year share ocean
state) plus the frozen blind 2024 set. Every comparison is paired-tested.

Usage (repo root, opendrift env):
    python -m main.ml.forecast
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from main.ml.baselines import KM_PER_H_PER_MS  # noqa: E402
from main.ml.scenario import HORIZONS_D  # noqa: E402

ML_OUT = ROOT / "main" / "outputs" / "ml"
DATASET = ML_OUT / "scenario_dataset.npz"
HOLDOUT = ML_OUT / "scenario_dataset_2024.npz"
REPORT = ML_OUT / "forecast_report.json"
SEED = 42

# loss='absolute_error' fits the CONDITIONAL MEDIAN, matching the median
# position error we report. The squared-error default fits the mean and is
# pulled by the long tail of eddy-dominated scenarios.
HGB = dict(max_iter=400, learning_rate=0.05, min_samples_leaf=8,
           l2_regularization=1.0, loss="absolute_error", random_state=SEED)
# Uncertainty envelope: same trees, pinball loss at the tails.
QUANTILES = (0.1, 0.9)

# Target layout: 4 quantities per horizon (dx, dy, dist, spread)
Q = ["dx_km", "dy_km", "dist_km", "spread_km"]


def tcol(h_idx: int, q: str) -> int:
    return h_idx * len(Q) + Q.index(q)


# ── baselines ────────────────────────────────────────────────────────────────

def fit_climatology(Y_tr, blocks_tr):
    """Mean outcome per (field, season) block; global mean as fallback."""
    table = {b: Y_tr[blocks_tr == b].mean(axis=0)
             for b in np.unique(blocks_tr)}
    return table, Y_tr.mean(axis=0)


def predict_climatology(model, blocks_te):
    table, glob = model
    return np.array([table.get(b, glob) for b in blocks_te])


def predict_persistence(X, feat_names) -> np.ndarray:
    """Slick drifts along the antecedent 7-day mean current.

    The physical null hypothesis for 'condições de correntes dos últimos
    dias': whatever the ocean has been doing recently, it keeps doing.
    Spread is taken from climatological growth (not modelled here) so only
    dx/dy/dist are predicted; spread falls back to NaN and is scored apart.
    """
    iu = list(feat_names).index("u_mean_7d")
    iv = list(feat_names).index("v_mean_7d")
    u, v = X[:, iu], X[:, iv]
    out = np.full((len(X), len(HORIZONS_D) * len(Q)), np.nan, np.float32)
    for hi, h in enumerate(HORIZONS_D):
        hours = h * 24
        dx = u * KM_PER_H_PER_MS * hours
        dy = v * KM_PER_H_PER_MS * hours
        out[:, tcol(hi, "dx_km")] = dx
        out[:, tcol(hi, "dy_km")] = dy
        out[:, tcol(hi, "dist_km")] = np.hypot(dx, dy)
    return out


def predict_analogue(X_tr, Y_tr, X_te) -> np.ndarray:
    """1-NN over z-scored features, NaN-tolerant (compare on shared columns)."""
    mu = np.nanmean(X_tr, axis=0)
    sd = np.nanstd(X_tr, axis=0)
    sd[sd == 0] = 1.0
    A = (X_tr - mu) / sd
    B = (X_te - mu) / sd
    out = np.empty((len(B), Y_tr.shape[1]), np.float32)
    for i, b in enumerate(B):
        d = np.nansum((A - b) ** 2, axis=1)
        out[i] = Y_tr[int(np.argmin(d))]
    return out


def make_ridge():
    """Linear baseline: does the trees' nonlinearity actually earn its place?

    Ridge cannot consume NaN, so lookback windows that fall outside the year
    are median-imputed here (the trees instead learn a split direction for
    missingness). Standardised inputs, L2 penalty chosen by internal CV.
    """
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        RidgeCV(alphas=np.logspace(-3, 3, 13)),
    )


def fit_ridge(X, Y):
    models = []
    for k in range(Y.shape[1]):
        ok = np.isfinite(Y[:, k])
        m = make_ridge()
        m.fit(X[ok], Y[ok, k])
        models.append(m)
    return models


def fit_hgb(X, Y, **over):
    params = {**HGB, **over}
    models = []
    for k in range(Y.shape[1]):
        ok = np.isfinite(Y[:, k])
        m = HistGradientBoostingRegressor(**params)
        m.fit(X[ok], Y[ok, k])
        models.append(m)
    return models


def predict_models(models, X):
    return np.column_stack([m.predict(X) for m in models]).astype(np.float32)


# Backwards-compatible alias used across the module/tests.
predict_hgb = predict_models


def derive_dist(P: np.ndarray) -> np.ndarray:
    """Force dist_km = |(dx, dy)| instead of predicting it independently.

    Fitting one booster per target let the model output a distance that
    contradicted its own displacement vector. Deriving it removes that
    inconsistency for free and cannot hurt: the vector carries the
    information the scalar was estimating.
    """
    P = P.copy()
    for hi in range(len(HORIZONS_D)):
        P[:, tcol(hi, "dist_km")] = np.hypot(P[:, tcol(hi, "dx_km")],
                                             P[:, tcol(hi, "dy_km")])
    return P


def fit_quantile_envelope(X, Y, **over):
    """P10/P90 boosters for the travelled distance at each horizon.

    A predicted slick position without a confidence envelope is not
    operationally usable — responders need "how far could it plausibly be",
    not just a point estimate.
    """
    out = {}
    for hi, h in enumerate(HORIZONS_D):
        k = tcol(hi, "dist_km")
        ok = np.isfinite(Y[:, k])
        out[h] = {}
        for q in QUANTILES:
            m = HistGradientBoostingRegressor(
                **{**HGB, "loss": "quantile", "quantile": q, **over})
            m.fit(X[ok], Y[ok, k])
            out[h][q] = m
    return out


def conformal_correction(models_q, X_cal, Y_cal, alpha=0.2) -> dict:
    """Split-conformal widening (CQR) so the band covers what it promises.

    Quantile boosters are fitted, not calibrated: measured on the blind year
    their raw P10-P90 covered ~45% against a nominal 80% (the trees fit the
    training quantiles too tightly to survive a new year). CQR (Romano,
    Patterson & Candès 2019) restores a marginal guarantee by widening the
    band by the empirical (1-alpha) quantile of the conformity score
    max(lo - y, y - up), measured on data the quantile models never saw.

    Calibration must be a HELD-OUT YEAR, not a random split: scenarios of the
    same year share the ocean state, so a random split leaks and returns an
    optimistically small correction.

    A negative correction is legitimate — it means the raw band over-covered
    and conformal is tightening it.
    """
    out = {}
    for hi, h in enumerate(HORIZONS_D):
        k = tcol(hi, "dist_km")
        ok = np.isfinite(Y_cal[:, k])
        lo = models_q[h][QUANTILES[0]].predict(X_cal[ok])
        up = models_q[h][QUANTILES[1]].predict(X_cal[ok])
        y = Y_cal[ok, k]
        scores = np.maximum(lo - y, y - up)
        n = len(scores)
        level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
        out[h] = float(np.quantile(scores, level, method="higher"))
    return out


def envelope_coverage(models_q, X, Y, corrections=None) -> dict:
    """Empirical coverage of the P10-P90 band — should sit near 80%."""
    res = {}
    for hi, h in enumerate(HORIZONS_D):
        k = tcol(hi, "dist_km")
        ok = np.isfinite(Y[:, k])
        c = corrections.get(h, 0.0) if corrections else 0.0
        lo = models_q[h][QUANTILES[0]].predict(X[ok]) - c
        hi_ = models_q[h][QUANTILES[1]].predict(X[ok]) + c
        inside = (Y[ok, k] >= lo) & (Y[ok, k] <= hi_)
        res[f"D+{h}"] = {"coverage": float(inside.mean()),
                         "nominal": QUANTILES[1] - QUANTILES[0],
                         "median_width_km": float(np.median(hi_ - lo)),
                         "conformal_km": float(c)}
    return res


# ── scoring ──────────────────────────────────────────────────────────────────

def score(Y_true, Y_pred) -> dict:
    """Per-horizon centroid-position error and spread MAE (km)."""
    out = {}
    for hi, h in enumerate(HORIZONS_D):
        dx_t, dy_t = Y_true[:, tcol(hi, "dx_km")], Y_true[:, tcol(hi, "dy_km")]
        dx_p, dy_p = Y_pred[:, tcol(hi, "dx_km")], Y_pred[:, tcol(hi, "dy_km")]
        err = np.hypot(dx_t - dx_p, dy_t - dy_p)
        sp_t = Y_true[:, tcol(hi, "spread_km")]
        sp_p = Y_pred[:, tcol(hi, "spread_km")]
        out[f"D+{h}"] = {
            "pos_err_km_median": float(np.nanmedian(err)),
            "pos_err_km_mean": float(np.nanmean(err)),
            "spread_mae_km": float(np.nanmean(np.abs(sp_t - sp_p)))
            if np.isfinite(sp_p).any() else None,
            "n": int(np.isfinite(err).sum()),
        }
    return out


def pos_errors(Y_true, Y_pred, hi: int) -> np.ndarray:
    return np.hypot(Y_true[:, tcol(hi, "dx_km")] - Y_pred[:, tcol(hi, "dx_km")],
                    Y_true[:, tcol(hi, "dy_km")] - Y_pred[:, tcol(hi, "dy_km")])


# ── evaluation ───────────────────────────────────────────────────────────────

MODELS = ["climatology", "persistence", "analogue", "ridge", "HGB"]


def all_predictions(X_tr, Y_tr, b_tr, X_te, feat_names):
    clim = fit_climatology(Y_tr, b_tr)
    return {
        "climatology": lambda b_te: predict_climatology(clim, b_te),
        "persistence": lambda b_te: predict_persistence(X_te, feat_names),
        "analogue": lambda b_te: predict_analogue(X_tr, Y_tr, X_te),
        "ridge": lambda b_te, m=fit_ridge(X_tr, Y_tr): derive_dist(
            predict_models(m, X_te)),
        "HGB": lambda b_te, m=fit_hgb(X_tr, Y_tr): derive_dist(
            predict_models(m, X_te)),
    }


def leave_one_year_out(X, Y, blocks, years, feat_names) -> dict:
    res = {}
    for held in sorted(set(years.tolist())):
        te = years == held
        preds = all_predictions(X[~te], Y[~te], blocks[~te], X[te], feat_names)
        res[str(held)] = {name: score(Y[te], fn(blocks[te]))
                          for name, fn in preds.items()}
        for h in (HORIZONS_D[0], HORIZONS_D[-1]):
            line = "  ".join(
                f"{n}={res[str(held)][n][f'D+{h}']['pos_err_km_median']:6.1f}"
                for n in MODELS)
            print(f"  {held}  D+{h} (km): {line}", flush=True)
    return res


def leave_one_field_out(X, Y, blocks, fields, feat_names) -> dict:
    """Can we forecast a spill at a location the model never saw?

    This is the deployment question: the end goal is an arbitrary release
    point, where a per-field climatology cannot exist. Training on 5 fields
    and testing on the 6th is the honest proxy. Climatology here degrades to
    the season-mean over OTHER fields, exactly as it would in practice.

    The linear control runs here too. This is the project's headline claim,
    so "the trees beat climatology" is not enough — if RidgeCV on the same
    features gets the same gain, the honest statement is that the antecedent
    features carry the signal, not that gradient boosting was required.
    """
    res = {}
    pooled = {h: {"hgb": [], "clim": [], "ridge": []} for h in HORIZONS_D}
    print("\n=== Leave-one-FIELD-out (local nunca visto) ===")
    print(f"{'campo':12s}" + "".join(f"{'D+'+str(h):>16s}" for h in HORIZONS_D))
    for held in sorted(set(fields.tolist())):
        te = fields == held
        tr = ~te
        # Season-only climatology: the field is unknown, so pool by season.
        seasons_tr = np.array([b.split("_")[-1] for b in blocks[tr]])
        seasons_te = np.array([b.split("_")[-1] for b in blocks[te]])
        table = {s: Y[tr][seasons_tr == s].mean(axis=0)
                 for s in np.unique(seasons_tr)}
        glob = Y[tr].mean(axis=0)
        P_clim = np.array([table.get(s, glob) for s in seasons_te])
        P_hgb = derive_dist(predict_models(fit_hgb(X[tr], Y[tr]), X[te]))
        P_ridge = derive_dist(predict_models(fit_ridge(X[tr], Y[tr]), X[te]))

        s_clim, s_hgb = score(Y[te], P_clim), score(Y[te], P_hgb)
        s_ridge = score(Y[te], P_ridge)
        res[held] = {"climatology_season": s_clim, "ridge": s_ridge,
                     "HGB": s_hgb}
        for hi, h in enumerate(HORIZONS_D):
            pooled[h]["hgb"].append(pos_errors(Y[te], P_hgb, hi))
            pooled[h]["clim"].append(pos_errors(Y[te], P_clim, hi))
            pooled[h]["ridge"].append(pos_errors(Y[te], P_ridge, hi))
        cells = "".join(
            f"{s_hgb[f'D+{h}']['pos_err_km_median']:7.1f}/"
            f"{s_clim[f'D+{h}']['pos_err_km_median']:<8.1f}"
            for h in HORIZONS_D)
        print(f"{held:12s}{cells}")
    print("  (HGB / climatologia-sazonal, erro mediano em km)")

    print("\n  Teste pareado (todos os cenários de todos os campos retidos):")
    res["paired"] = {}
    for h in HORIZONS_D:
        a = np.concatenate(pooled[h]["hgb"])
        b = np.concatenate(pooled[h]["clim"])
        c = np.concatenate(pooled[h]["ridge"])
        ok = np.isfinite(a) & np.isfinite(b) & np.isfinite(c)
        p = float(stats.wilcoxon(a[ok], b[ok]).pvalue)
        p_ridge_clim = float(stats.wilcoxon(c[ok], b[ok]).pvalue)
        p_hgb_ridge = float(stats.wilcoxon(a[ok], c[ok]).pvalue)
        gain = 100 * (1 - np.median(a[ok]) / np.median(b[ok]))
        gain_ridge = 100 * (1 - np.median(c[ok]) / np.median(b[ok]))
        res["paired"][f"D+{h}"] = {
            "p": p, "n": int(ok.sum()),
            "median_hgb_km": float(np.median(a[ok])),
            "median_clim_km": float(np.median(b[ok])),
            "median_ridge_km": float(np.median(c[ok])),
            "gain_pct": float(gain),
            "gain_pct_ridge": float(gain_ridge),
            "hgb_wins": int((a[ok] < b[ok]).sum()),
            "p_ridge_vs_clim": p_ridge_clim,
            "p_hgb_vs_ridge": p_hgb_ridge,
            "hgb_wins_vs_ridge": int((a[ok] < c[ok]).sum()),
        }
        print(f"    D+{h}: HGB {np.median(a[ok]):5.1f} | ridge "
              f"{np.median(c[ok]):5.1f} | clim {np.median(b[ok]):5.1f} km   "
              f"HGB vs clim {gain:+.0f}% p={p:.1e} | "
              f"ridge vs clim {gain_ridge:+.0f}% p={p_ridge_clim:.1e} | "
              f"HGB vs ridge p={p_hgb_ridge:.1e} "
              f"({int((a[ok] < c[ok]).sum())}/{int(ok.sum())})")
    return res


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    d = np.load(DATASET, allow_pickle=True)
    X, Y = d["X"], d["Y"]
    blocks, years = d["block"], d["year"]
    feat_names = d["feature_names"]

    print(f"Previsão em nível de cenário — {len(X)} cenários de treino, "
          f"anos {sorted(set(years.tolist()))}, horizontes {HORIZONS_D}\n")

    print("=== Leave-one-year-out ===")
    loyo = leave_one_year_out(X, Y, blocks, years, feat_names)

    h = np.load(HOLDOUT, allow_pickle=True)
    Xh, Yh, bh = h["X"], h["Y"], h["block"]
    print(f"\n=== Cego 2024 ({len(Xh)} cenários, modelo treinado em "
          f"{'+'.join(str(y) for y in sorted(set(years.tolist())))}) ===")
    preds = all_predictions(X, Y, blocks, Xh, feat_names)
    blind = {}
    pred_arrays = {}
    for name, fn in preds.items():
        P = fn(bh)
        pred_arrays[name] = P
        blind[name] = score(Yh, P)

    hdr = f"{'modelo':12s}" + "".join(f"{'D+'+str(x):>10s}" for x in HORIZONS_D)
    print(hdr)
    for name in MODELS:
        row = "".join(f"{blind[name][f'D+{x}']['pos_err_km_median']:10.1f}"
                      for x in HORIZONS_D)
        print(f"{name:12s}{row}")
    print("(erro mediano de posição do centróide, km)")

    # Paired tests at EVERY horizon — predictability from initial conditions
    # decays with lead time, so a single-horizon test hides the whole story.
    tests = {}
    print("\nTeste pareado de Wilcoxon (HGB vs baseline), por horizonte:")
    for hi, h in enumerate(HORIZONS_D):
        e_hgb = pos_errors(Yh, pred_arrays["HGB"], hi)
        tests[f"D+{h}"] = {}
        parts = []
        for name in [m for m in MODELS if m != "HGB"]:
            e_b = pos_errors(Yh, pred_arrays[name], hi)
            ok = np.isfinite(e_hgb) & np.isfinite(e_b)
            p = float(stats.wilcoxon(e_hgb[ok], e_b[ok]).pvalue)
            delta = float(np.median(e_hgb[ok] - e_b[ok]))
            tests[f"D+{h}"][name] = {
                "p": p, "median_delta_km": delta,
                "hgb_wins": int((e_hgb[ok] < e_b[ok]).sum()),
                "n": int(ok.sum()),
            }
            mark = "*" if p < 0.05 else " "
            parts.append(f"{name[:5]}:{delta:+6.1f}km p={p:.3f}{mark}")
        print(f"  D+{h}: " + "  ".join(parts))
    print("  (* = significativo a 5%; Δ negativo = HGB melhor)")

    # Longest horizon: the hardest cell of the table, and the one the trend
    # analysis below is read at. Pinned explicitly rather than inherited from
    # the loop variable above, so the scope can change without silent drift.
    hi_last = len(HORIZONS_D) - 1
    h_last = HORIZONS_D[hi_last]

    # Spread (patch size) — the quantity advection never modelled
    print(f"\nEspalhamento da mancha (MAE km) em D+{h_last}:")
    for name in ["climatology", "analogue", "ridge", "HGB"]:
        v = blind[name][f"D+{h_last}"]["spread_mae_km"]
        print(f"  {name:12s} {v:.2f}" if v is not None else f"  {name:12s}  n/a")

    # Uncertainty envelope (P10-P90) on the travelled distance.
    # The quantile boosters are fitted on the older training years and
    # CALIBRATED on the most recent one — a held-out year, because scenarios
    # of the same year share the ocean state. Raw and conformal coverage are
    # both reported on the blind year: the gap between them is the honest
    # measure of how far uncalibrated quantile trees are from their promise.
    cal_year = max(set(years.tolist()))
    fit_mask = years != cal_year
    print(f"\n=== Envelope P10–P90 (distância) — ajuste em "
          f"{sorted(set(years[fit_mask].tolist()))}, calibração conformal em "
          f"{cal_year}, medição no cego 2024 ===")
    q_models = fit_quantile_envelope(X[fit_mask], Y[fit_mask])
    corr = conformal_correction(q_models, X[~fit_mask], Y[~fit_mask])
    cov_raw = envelope_coverage(q_models, Xh, Yh)
    cov = envelope_coverage(q_models, Xh, Yh, corrections=corr)
    for h in HORIZONS_D:
        r, c = cov_raw[f"D+{h}"], cov[f"D+{h}"]
        print(f"  D+{h}: cru {r['coverage']:.0%} ({r['median_width_km']:.0f} km)"
              f"  ->  conformal {c['coverage']:.0%} "
              f"({c['median_width_km']:.0f} km, +{corr[h]:.0f} km/lado)"
              f"   nominal {c['nominal']:.0%}")

    # Trend analysis: what actually drives the outcome?
    print(f"\n=== Tendências: importância por permutação (D+{h_last}, distância) ===")
    k = tcol(hi_last, "dist_km")
    ok = np.isfinite(Y[:, k])
    m = HistGradientBoostingRegressor(**HGB).fit(X[ok], Y[ok, k])
    # Importance is measured on the blind set, so drop any hold-out scenario
    # whose target is missing (a run that ended before the horizon).
    okh = np.isfinite(Yh[:, k])
    imp = permutation_importance(m, Xh[okh], Yh[okh, k], n_repeats=10,
                                 random_state=SEED, scoring="neg_mean_absolute_error")
    order = np.argsort(imp.importances_mean)[::-1][:10]
    for i in order:
        print(f"  {str(feat_names[i]):22s} {imp.importances_mean[i]:7.2f} km "
              f"± {imp.importances_std[i]:.2f}")

    lofo = leave_one_field_out(X, Y, blocks, d["field"], feat_names)

    REPORT.write_text(json.dumps({
        "loyo": loyo, "lofo_new_location": lofo,
        "blind_2024": blind, "paired_tests_by_horizon": tests,
        "uncertainty_envelope": cov,
        "uncertainty_envelope_raw": cov_raw,
        "uncertainty_calibration": {"fit_years": sorted(
            set(years[fit_mask].tolist())), "calibration_year": int(cal_year)},
        "horizons_d": HORIZONS_D,
        f"top_features_d{h_last}_dist": [
            {"feature": str(feat_names[i]),
             "importance_km": float(imp.importances_mean[i])} for i in order],
    }, indent=2))
    print(f"\n[OK] relatório -> {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
