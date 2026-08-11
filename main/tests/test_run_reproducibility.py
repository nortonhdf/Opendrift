"""Runs are reproducible, and that is a stated property — not an accident.

OpenDrift seeds numpy's global RNG in its constructor (basemodel __init__,
default seed=0), and the declared uncertainties
(drift:current_uncertainty 0.05, drift:wind_uncertainty 0.5) draw from it at
every step. So the archives ARE re-derivable from this repo — provided the
seed keeps flowing through. These tests pin that it does, and that the
parameter is not silently ignored (an earlier attempt called
np.random.seed() before constructing OpenOil, which the constructor then
overwrote — the seed looked wired up and did nothing).
"""

import numpy as np
import xarray as xr

from main.run_open_oil import run_simulation


def _lon_track(tmp_path, name, **kw):
    res = run_simulation(
        n_particles=20,
        duration_hours=2,
        use_wind=False,
        use_waves=False,
        smoke_test=True,
        outfile=str(tmp_path / f"{name}.nc"),
        figfile=str(tmp_path / f"{name}.png"),
        loglevel=50,
        **kw,
    )
    ds = xr.open_dataset(res["netcdf"])
    lon = np.asarray(ds["lon"].values, float).copy()
    ds.close()
    return lon


def test_default_run_is_reproducible(tmp_path):
    """The archived generation used the default — it must stay repeatable."""
    assert np.allclose(_lon_track(tmp_path, "a"), _lon_track(tmp_path, "b"),
                       equal_nan=True)


def test_same_seed_reproduces_the_run(tmp_path):
    a = _lon_track(tmp_path, "c", random_seed=123)
    b = _lon_track(tmp_path, "d", random_seed=123)
    assert np.allclose(a, b, equal_nan=True)


def test_different_seeds_give_different_runs(tmp_path):
    """Guards against the seed being accepted and quietly dropped."""
    a = _lon_track(tmp_path, "e", random_seed=1)
    b = _lon_track(tmp_path, "f", random_seed=2)
    assert not np.allclose(a, b, equal_nan=True)
