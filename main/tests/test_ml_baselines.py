"""Baseline predictors: analytic behaviour and block-split discipline."""

import numpy as np
import pytest

from main.ml.baselines import (
    leave_one_block_out, mae_km, predict_advection, predict_nearest,
    predict_persistence,
)


def test_persistence_predicts_zero():
    X = np.random.default_rng(0).normal(size=(5, 9)).astype(np.float32)
    assert np.all(predict_persistence(X) == 0)


def test_advection_arithmetic():
    # 0.5 m/s current east + 10 m/s wind east (3%) over 6 h
    X = np.zeros((1, 9), np.float32)
    X[0, 4] = 0.5      # u_cur
    X[0, 6] = 10.0     # u_wind
    p = predict_advection(X, dt_hours=6.0)
    expected = (0.5 + 0.03 * 10.0) * 3.6 * 6.0
    assert p[0, 0] == pytest.approx(expected, rel=1e-6)
    assert p[0, 1] == 0.0 and p[0, 2] == 0.0


def test_nearest_recovers_exact_match():
    rng = np.random.default_rng(1)
    X_tr = rng.normal(size=(20, 9)).astype(np.float32)
    Y_tr = rng.normal(size=(20, 3)).astype(np.float32)
    # Query with training rows themselves -> must return their own targets
    p = predict_nearest(X_tr, Y_tr, X_tr[:5])
    np.testing.assert_allclose(p, Y_tr[:5])


def test_mae_km_zero_for_perfect():
    Y = np.array([[1.0, -2.0, 0.5]], np.float32)
    m = mae_km(Y, Y)
    assert m["mean_disp_err_km"] == 0.0 and m["mae_dspread"] == 0.0


def test_leave_one_block_out_runs_and_separates_blocks():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(40, 9)).astype(np.float32)
    Y = rng.normal(size=(40, 3)).astype(np.float32)
    blocks = np.array(["A"] * 20 + ["B"] * 20)
    res = leave_one_block_out(X, Y, blocks, dt_hours=6.0)
    assert set(res) == {"persistence", "advection", "nearest"}
    for m in res.values():
        assert m["mean_disp_err_km"] > 0
