"""Patch-transition dataset construction on synthetic trajectories."""

from datetime import datetime, timedelta

import numpy as np
import pytest
import xarray as xr

from main.ml.dataset import KM_PER_DEG, ForcingSampler, patch_state, samples_from_run


def test_patch_state_centroid_and_spread():
    lon = np.array([-41.0, -41.0, -40.9, np.nan])
    lat = np.array([-23.0, -22.9, -23.0, np.nan])
    lon_c, lat_c, spread = patch_state(lon, lat)
    assert lon_c == pytest.approx(np.nanmean(lon[:3]))
    assert lat_c == pytest.approx(np.nanmean(lat[:3]))
    assert 0 < spread < 15          # points are a few km apart


def test_patch_state_all_nan():
    out = patch_state(np.array([np.nan]), np.array([np.nan]))
    assert all(np.isnan(v) for v in out)


class _FakeSampler:
    def at(self, lon, lat, when):
        return {"u_cur": 0.1, "v_cur": 0.0, "u_wind": 5.0, "v_wind": 0.0,
                "sst": 24.0}


def _run_file(tmp_path, n_t=25, step_h=0.5):
    """Patch drifting east at a constant 0.1 deg / 6 h."""
    n = 4
    times = np.array([np.datetime64(datetime(2025, 1, 1)) +
                      np.timedelta64(int(i * step_h * 3600), "s")
                      for i in range(n_t)])
    base = np.linspace(0, 0.1 * (n_t - 1) / 12.0, n_t)   # deg east over time
    lon = np.tile(base, (n, 1)) - 41.0
    lat = np.full((n, n_t), -23.0)
    ds = xr.Dataset(
        {"lon": (("trajectory", "time"), lon),
         "lat": (("trajectory", "time"), lat),
         "status": (("trajectory", "time"), np.zeros((n, n_t), int))},
        coords={"time": times},
    )
    p = tmp_path / "run.nc"
    ds.to_netcdf(p)
    return p


def test_samples_from_run_targets_match_motion(tmp_path):
    p = _run_file(tmp_path)
    rows = samples_from_run(p, "TestField_jan", _FakeSampler())
    assert len(rows) == 2                       # 25 steps of 0.5 h -> 2 windows of 6 h
    feats, targs = rows[0]
    assert len(feats) == 9 and len(targs) == 3
    dx_expected = 0.1 * KM_PER_DEG * np.cos(np.radians(-23.0))
    assert targs[0] == pytest.approx(dx_expected, rel=0.01)   # eastward km
    assert targs[1] == pytest.approx(0.0, abs=1e-6)           # no meridional motion
    assert targs[2] == pytest.approx(0.0, abs=1e-6)           # rigid patch
    assert feats[3] == 0.0                                    # age at release
