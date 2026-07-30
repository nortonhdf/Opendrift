"""Surrogate training sanity: learns a learnable signal, reproducibly."""

import numpy as np
import pytest

from main.ml.train import fit_model, lobo_evaluate, predict


def _linear_dataset(n=300, seed=0):
    """Targets are a clean linear function of two features + small noise."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 9)).astype(np.float32)
    Y = np.column_stack([
        3.0 * X[:, 4] + 0.5 * X[:, 6],       # dx ~ current + wind
        3.0 * X[:, 5] + 0.5 * X[:, 7],       # dy
        0.1 * X[:, 2],                        # dspread ~ spread
    ]).astype(np.float32) + rng.normal(scale=0.05, size=(n, 3)).astype(np.float32)
    blocks = np.array(["A", "B", "C"])[np.arange(n) % 3]
    return X, Y, blocks


def test_model_learns_linear_signal():
    X, Y, _ = _linear_dataset()
    models = fit_model(X[:250], Y[:250])
    pred = predict(models, X[250:])
    resid = np.abs(pred - Y[250:]).mean()
    baseline = np.abs(Y[250:] - Y[:250].mean(axis=0)).mean()
    assert resid < 0.5 * baseline          # clearly better than predicting the mean


def test_training_is_reproducible():
    X, Y, _ = _linear_dataset()
    p1 = predict(fit_model(X, Y), X[:10])
    p2 = predict(fit_model(X, Y), X[:10])
    np.testing.assert_allclose(p1, p2)     # fixed random_state


def test_lobo_evaluate_returns_metrics():
    X, Y, blocks = _linear_dataset(n=90)
    m = lobo_evaluate(X, Y, blocks)
    assert set(m) >= {"mean_disp_err_km", "mae_dspread"}
    assert m["mean_disp_err_km"] > 0
