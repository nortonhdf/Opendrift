"""Rollout arithmetic for the hold-out evaluation (fake model + sampler)."""

import numpy as np
import pytest

from main.ml.dataset import KM_PER_DEG
from main.ml.holdout import _disc, _occupancy, rollout


class _StillSampler:
    def at(self, lon, lat, when):
        return {"u_cur": 0.0, "v_cur": 0.0, "u_wind": 0.0, "v_wind": 0.0,
                "sst": 24.0}


def test_rollout_constant_eastward_prediction():
    # Fake model: always predict 11.132 km east, 0 north, +1 km spread.
    def fn(F):
        return np.array([[11.132, 0.0, 1.0]], np.float32)

    lons, lats, spreads = rollout(fn, _StillSampler(), -41.0, -23.0,
                                  np.datetime64("2024-01-05"), n_steps=4)
    assert len(lons) == 5
    # 11.132 km at lat -23 = 0.1 deg / cos(lat) ... dx_deg = km/(111.32*cos)
    dlon = 11.132 / (KM_PER_DEG * np.cos(np.radians(-23.0)))
    assert lons[-1] == pytest.approx(-41.0 + 4 * dlon, rel=1e-3)
    assert lats[-1] == pytest.approx(-23.0)
    assert spreads[-1] == pytest.approx(4.0)


def test_rollout_spread_never_negative():
    def fn(F):
        return np.array([[0.0, 0.0, -5.0]], np.float32)

    _, _, spreads = rollout(fn, _StillSampler(), -41.0, -23.0,
                            np.datetime64("2024-01-05"), n_steps=3)
    assert np.all(spreads >= 0)


def test_occupancy_and_disc_overlap():
    lon = np.array([-41.0, -41.0, -40.95])
    lat = np.array([-23.0, -22.95, -23.0])
    occ = _occupancy(lon, lat)
    assert occ.sum() >= 1
    disc = _disc(-41.0, -23.0, radius_km=12.0)
    assert disc.sum() >= 4
    # The disc centred on the particles must overlap their occupancy.
    assert (occ & disc).sum() >= 1
