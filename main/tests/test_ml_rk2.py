"""Midpoint (RK2) advection rollout: exactness and sampling behaviour."""

import numpy as np
import pytest

from main.ml.dataset import KM_PER_DEG
from main.ml.holdout import rollout_rk2


class _UniformSampler:
    """Spatially uniform, steady current — RK2 must match Euler exactly."""

    def __init__(self):
        self.calls = 0

    def at(self, lon, lat, when):
        self.calls += 1
        return {"u_cur": 0.5, "v_cur": 0.0, "u_wind": 0.0, "v_wind": 0.0,
                "sst": 24.0}


class _ShearSampler:
    """Current grows eastward with longitude — midpoint must see more than
    the start point, so RK2 and Euler must differ."""

    def at(self, lon, lat, when):
        return {"u_cur": 0.5 + 0.5 * (lon + 41.0), "v_cur": 0.0,
                "u_wind": 0.0, "v_wind": 0.0, "sst": 24.0}


def test_rk2_matches_analytic_uniform_flow():
    s = _UniformSampler()
    lons, lats, spreads = rollout_rk2(s, -41.0, -23.0,
                                      np.datetime64("2024-01-05"), n_steps=2)
    # 0.5 m/s for 6 h = 10.8 km per step, purely zonal
    expected_km = 0.5 * 3.6 * 6.0
    dlon = expected_km / (KM_PER_DEG * np.cos(np.radians(-23.0)))
    assert lons[-1] == pytest.approx(-41.0 + 2 * dlon, rel=1e-3)
    assert lats[-1] == pytest.approx(-23.0)
    assert np.all(spreads == 0)
    assert s.calls == 4          # two samples (start + midpoint) per step


def test_rk2_differs_from_euler_under_shear():
    from main.ml.holdout import _drift_km

    s = _ShearSampler()
    lons_rk2, _, _ = rollout_rk2(s, -41.0, -23.0,
                                 np.datetime64("2024-01-05"), n_steps=3)
    # Plain Euler for comparison
    lon, lat = -41.0, -23.0
    for _ in range(3):
        dx, _dy = _drift_km(s.at(lon, lat, None), 6.0)
        lon += dx / (KM_PER_DEG * np.cos(np.radians(lat)))
    assert lons_rk2[-1] != pytest.approx(lon, rel=1e-6)
