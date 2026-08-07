"""Scenario-level dataset and forecasting baselines.

The load-bearing property here is CAUSALITY: antecedent features must be
computed from forcing strictly before the release instant, otherwise the
whole D+n forecasting claim is leakage.
"""

import numpy as np
import pytest
import xarray as xr

from main.ml import scenario
from main.ml.forecast import (
    Q, fit_climatology, predict_climatology, predict_persistence, score, tcol,
)
from main.ml.scenario import HORIZONS_D, LOOKBACKS_D, feature_names


def _currents(tmp_path, jump_day=200):
    """Year of daily currents: u=0.1 before the jump day, u=9.9 after."""
    times = np.arange(np.datetime64("2025-01-01"), np.datetime64("2026-01-01"),
                      np.timedelta64(1, "D"))
    u = np.where(np.arange(len(times)) < jump_day, 0.1, 9.9)
    shape = (len(times), 2, 2)
    ds = xr.Dataset(
        {"x_sea_water_velocity": (("time", "latitude", "longitude"),
                                  np.broadcast_to(u[:, None, None], shape).copy()),
         "y_sea_water_velocity": (("time", "latitude", "longitude"),
                                  np.zeros(shape)),
         "sea_water_temperature": (("time", "latitude", "longitude"),
                                   np.full(shape, 24.0))},
        coords={"time": times, "latitude": [-23.0, -22.9],
                "longitude": [-41.0, -40.9]},
    )
    p = tmp_path / "cur.nc"
    ds.to_netcdf(p)
    return p


def test_antecedent_features_are_strictly_causal(tmp_path, monkeypatch):
    p = _currents(tmp_path, jump_day=200)
    monkeypatch.setattr(scenario, "forcing_paths", lambda y: (p, p))
    s = scenario.AntecedentSampler([2025])

    # Release the day before the jump: no post-jump value may leak in.
    before = s.features(2025, -41.0, -23.0, np.datetime64("2025-07-19"))
    names = feature_names()[8:]          # antecedent block only
    for w in LOOKBACKS_D:
        i = names.index(f"u_mean_{w}d")
        assert before[i] == pytest.approx(0.1, abs=1e-6), f"vazou futuro em {w}d"

    # Well after the jump, the same windows must show the new regime.
    after = s.features(2025, -41.0, -23.0, np.datetime64("2025-10-01"))
    i3 = names.index("u_mean_3d")
    assert after[i3] == pytest.approx(9.9, abs=1e-6)
    s.close()


def test_short_window_at_year_start_is_nan_not_zero(tmp_path, monkeypatch):
    p = _currents(tmp_path)
    monkeypatch.setattr(scenario, "forcing_paths", lambda y: (p, p))
    s = scenario.AntecedentSampler([2025])
    # 2 Jan: a 90-day lookback has almost no data — must be NaN, never 0.
    f = s.features(2025, -41.0, -23.0, np.datetime64("2025-01-02"))
    names = feature_names()[8:]
    assert np.isnan(f[names.index("u_mean_90d")])
    assert f[names.index("coverage_90d")] < 0.05
    s.close()


def test_targets_measure_displacement_from_release(tmp_path):
    """Patch drifting exactly 0.1 deg east per day -> known dx at each D+n."""
    n_t = 241                                   # 120 h at 30-min output
    times = np.array([np.datetime64("2025-01-01") +
                      np.timedelta64(30 * i, "m") for i in range(n_t)])
    days = np.arange(n_t) * 0.5 / 24.0
    lon = np.tile(-41.0 + 0.1 * days, (5, 1))
    lat = np.full((5, n_t), -23.0)
    ds = xr.Dataset(
        {"lon": (("trajectory", "time"), lon),
         "lat": (("trajectory", "time"), lat),
         "status": (("trajectory", "time"), np.zeros((5, n_t), int))},
        coords={"time": times},
    )
    p = tmp_path / "run.nc"
    ds.to_netcdf(p)

    targs, lon0, lat0 = scenario.targets_from_run(p)
    assert lon0 == pytest.approx(-41.0)
    for hi, h in enumerate(HORIZONS_D):
        dx = targs[tcol(hi, "dx_km")]
        expected = 0.1 * h * scenario.KM_PER_DEG * np.cos(np.radians(-23.0))
        assert dx == pytest.approx(expected, rel=0.01), f"D+{h}"
        assert targs[tcol(hi, "dy_km")] == pytest.approx(0.0, abs=1e-6)


def test_climatology_predicts_block_means():
    Y = np.array([[10.0] * 16, [20.0] * 16, [100.0] * 16], np.float32)
    blocks = np.array(["A_jan", "A_jan", "B_jul"])
    model = fit_climatology(Y, blocks)
    p = predict_climatology(model, np.array(["A_jan", "B_jul", "C_oct"]))
    assert p[0][0] == pytest.approx(15.0)      # mean of A_jan
    assert p[1][0] == pytest.approx(100.0)
    assert p[2][0] == pytest.approx(Y.mean(axis=0)[0])   # unseen -> global


def test_persistence_extrapolates_antecedent_current():
    names = feature_names()
    X = np.zeros((1, len(names)), np.float32)
    X[0, names.index("u_mean_7d")] = 0.5       # m/s eastward
    P = predict_persistence(X, np.array(names))
    for hi, h in enumerate(HORIZONS_D):
        assert P[0, tcol(hi, "dx_km")] == pytest.approx(0.5 * 3.6 * 24 * h, rel=1e-5)
        assert P[0, tcol(hi, "dy_km")] == pytest.approx(0.0)


def test_score_is_zero_for_perfect_prediction():
    Y = np.tile(np.arange(len(HORIZONS_D) * len(Q), dtype=np.float32), (4, 1))
    s = score(Y, Y)
    for h in HORIZONS_D:
        assert s[f"D+{h}"]["pos_err_km_median"] == pytest.approx(0.0)
