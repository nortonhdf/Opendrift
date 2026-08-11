"""Methodological guarantees of the forecasting layer.

These pin the properties that make the D+n claims defensible: physical
self-consistency of the outputs, a linear control for the trees, and
calibrated uncertainty.
"""

import numpy as np
import pytest

from main.ml.forecast import (
    HORIZONS_D, Q, conformal_correction, derive_dist, envelope_coverage,
    fit_hgb, fit_quantile_envelope, fit_ridge, predict_models, tcol,
)


def _synthetic(n=400, seed=0):
    """dx driven linearly by u, dy by v, plus a nonlinear speed term."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 36)).astype(np.float32)
    u, v = X[:, 8], X[:, 9]
    Y = np.zeros((n, len(HORIZONS_D) * len(Q)), np.float32)
    for hi, h in enumerate(HORIZONS_D):
        Y[:, tcol(hi, "dx_km")] = 30 * h * u
        Y[:, tcol(hi, "dy_km")] = 30 * h * v
        Y[:, tcol(hi, "dist_km")] = np.hypot(Y[:, tcol(hi, "dx_km")],
                                             Y[:, tcol(hi, "dy_km")])
        Y[:, tcol(hi, "spread_km")] = 2.0 * h + rng.normal(scale=0.1, size=n)
    return X, Y


def test_derive_dist_enforces_vector_consistency():
    X, Y = _synthetic(n=50)
    P = np.zeros_like(Y)
    for hi in range(len(HORIZONS_D)):
        P[:, tcol(hi, "dx_km")] = 3.0
        P[:, tcol(hi, "dy_km")] = 4.0
        P[:, tcol(hi, "dist_km")] = 999.0        # deliberately inconsistent
    D = derive_dist(P)
    for hi in range(len(HORIZONS_D)):
        assert np.allclose(D[:, tcol(hi, "dist_km")], 5.0)


def test_derive_dist_does_not_touch_other_targets():
    X, Y = _synthetic(n=20)
    D = derive_dist(Y)
    for hi in range(len(HORIZONS_D)):
        for q in ("dx_km", "dy_km", "spread_km"):
            assert np.allclose(D[:, tcol(hi, q)], Y[:, tcol(hi, q)])


def test_ridge_recovers_a_linear_signal():
    """The linear control must be strong when the truth IS linear —
    otherwise it is not a fair yardstick for the trees."""
    X, Y = _synthetic()
    models = fit_ridge(X[:300], Y[:300])
    P = predict_models(models, X[300:])
    hi = 0
    err = np.hypot(Y[300:, tcol(hi, "dx_km")] - P[:, tcol(hi, "dx_km")],
                   Y[300:, tcol(hi, "dy_km")] - P[:, tcol(hi, "dy_km")])
    scale = np.hypot(Y[300:, tcol(hi, "dx_km")], Y[300:, tcol(hi, "dy_km")])
    assert np.median(err) < 0.05 * np.median(scale)


def test_ridge_tolerates_missing_lookback_windows():
    X, Y = _synthetic(n=200)
    X[:50, 30:] = np.nan               # 90-day windows unavailable
    models = fit_ridge(X, Y)           # imputer must absorb it
    P = predict_models(models, X)
    assert np.isfinite(P).all()


def test_hgb_consumes_nan_natively():
    X, Y = _synthetic(n=200)
    X[:50, 30:] = np.nan
    P = predict_models(fit_hgb(X, Y, max_iter=30), X)
    assert np.isfinite(P).all()


def test_quantile_envelope_is_ordered_and_covers():
    X, Y = _synthetic(n=300)
    qm = fit_quantile_envelope(X[:200], Y[:200])
    cov = envelope_coverage(qm, X[200:], Y[200:])
    for h in HORIZONS_D:
        assert cov[f"D+{h}"]["median_width_km"] > 0      # P90 above P10
        # Nominal 80%; allow generous slack on 100 held-out points.
        assert 0.4 <= cov[f"D+{h}"]["coverage"] <= 1.0


def _dist_only(n, noise, seed):
    """Scenarios whose travelled distance is a signal plus tunable noise."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 36)).astype(np.float32)
    Y = np.zeros((n, len(HORIZONS_D) * len(Q)), np.float32)
    for hi, h in enumerate(HORIZONS_D):
        d = 40.0 * h + 20.0 * X[:, 8] + noise * rng.normal(size=n)
        Y[:, tcol(hi, "dist_km")] = d
    return X, Y


def test_conformal_calibration_restores_coverage_under_shift():
    """The failure that motivated CQR: quantile trees fitted on one regime
    under-cover badly on a noisier one. Calibration must repair it."""
    X_fit, Y_fit = _dist_only(400, noise=2.0, seed=1)
    X_cal, Y_cal = _dist_only(300, noise=20.0, seed=2)
    X_te, Y_te = _dist_only(300, noise=20.0, seed=3)

    qm = fit_quantile_envelope(X_fit, Y_fit, max_iter=60)
    raw = envelope_coverage(qm, X_te, Y_te)
    corr = conformal_correction(qm, X_cal, Y_cal, alpha=0.2)
    cal = envelope_coverage(qm, X_te, Y_te, corrections=corr)

    for h in HORIZONS_D:
        assert raw[f"D+{h}"]["coverage"] < 0.6          # the broken state
        assert cal[f"D+{h}"]["coverage"] >= 0.72        # ~80% nominal
        assert cal[f"D+{h}"]["median_width_km"] > raw[f"D+{h}"]["median_width_km"]


def test_conformal_correction_may_tighten_an_overwide_band():
    """A negative correction is correct behaviour, not a bug to clamp away:
    a band that over-covers should be narrowed, or the envelope is useless."""
    X_fit, Y_fit = _dist_only(400, noise=25.0, seed=4)
    X_cal, Y_cal = _dist_only(300, noise=1.0, seed=5)
    corr = conformal_correction(fit_quantile_envelope(X_fit, Y_fit, max_iter=60),
                                X_cal, Y_cal, alpha=0.2)
    assert min(corr.values()) < 0


def test_hgb_fits_conditional_median_not_mean():
    """loss='absolute_error' must resist a heavy-tailed contamination that
    would drag a squared-error fit."""
    rng = np.random.default_rng(3)
    n = 600
    X = rng.normal(size=(n, 36)).astype(np.float32)
    base = 10.0 * X[:, 8]
    y = base.copy()
    y[:30] += 5000.0                                   # rare huge outliers
    Y = np.zeros((n, len(HORIZONS_D) * len(Q)), np.float32)
    for hi in range(len(HORIZONS_D)):
        Y[:, tcol(hi, "dx_km")] = y
    mae_model = fit_hgb(X, Y, max_iter=60)
    mse_model = fit_hgb(X, Y, max_iter=60, loss="squared_error")
    k = tcol(0, "dx_km")
    clean = np.arange(30, n)
    e_mae = np.median(np.abs(predict_models(mae_model, X)[clean, k] - base[clean]))
    e_mse = np.median(np.abs(predict_models(mse_model, X)[clean, k] - base[clean]))
    assert e_mae < e_mse
