"""Regression tests for the beaching/risk aggregation (audit finding #1).

Synthetic OpenDrift-like NetCDF files exercise the case that broke the real
products: a file whose code 1 means `missing_data`, which the old hard-coded
``STRANDED = 1`` counted as beaching.
"""

from datetime import datetime, timedelta

import numpy as np
import xarray as xr

from main.scripts.compute_beaching import compute_beaching, stranding_events
from main.scripts.compute_risk_grids import make_grid, particles_to_grid
from main.status_utils import STRANDED, code_of

# particles_to_grid bins against the module-level domain constants, so tests
# must use the same shared grid (domain_config), not a hand-rolled one.
LONS, LATS = make_grid()


def _write_run(path, lon, lat, status, meanings):
    """Write a minimal OpenDrift-like trajectory file."""
    n, nt = lon.shape
    times = np.array(
        [np.datetime64(datetime(2025, 1, 1) + timedelta(hours=h)) for h in range(nt)]
    )
    ds = xr.Dataset(
        {
            "lon": (("trajectory", "time"), lon),
            "lat": (("trajectory", "time"), lat),
            "status": (("trajectory", "time"), status),
        },
        coords={"time": times},
    )
    ds["status"].attrs["flag_values"] = np.arange(len(meanings.split()))
    ds["status"].attrs["flag_meanings"] = meanings
    ds.to_netcdf(path)
    return path


nan = np.nan


def _missing_data_run(path):
    """3 particles: survivor, left-domain (code 1 = missing_data), survivor."""
    lon = np.array([[-41.0, -41.1, -41.2],
                    [-41.0, -42.9, nan],
                    [-41.0, -41.0, -41.0]])
    lat = np.array([[-23.0, -23.0, -23.0],
                    [-23.0, -24.9, nan],
                    [-23.1, -23.1, -23.1]])
    status = np.array([[0, 0, 0],
                       [0, 1, 1],
                       [0, 0, 0]])
    return _write_run(path, lon, lat, status, "active missing_data")


def _stranded_run(path):
    """3 particles: beaches mid-run, beaches at the FINAL step, survivor.

    Here stranded is code 2 (missing_data occurred first in this run).
    """
    lon = np.array([[-41.0, -41.5, nan],       # strands at step 1
                    [-41.0, -41.2, -41.4],     # strands exactly at last step
                    [-41.0, -40.9, -40.8]])    # survives
    lat = np.array([[-22.0, -22.0, nan],
                    [-22.1, -22.1, -22.1],
                    [-22.2, -22.2, -22.2]])
    status = np.array([[0, 2, 2],
                       [0, 0, 2],
                       [0, 0, 0]])
    return _write_run(path, lon, lat, status, "active missing_data stranded")


def test_missing_data_is_not_beaching(tmp_path):
    p = _missing_data_run(tmp_path / "md.nc")
    ds = xr.open_dataset(p)
    scode = code_of(ds["status"], STRANDED)
    assert scode is None                       # this file has no stranding
    slon, slat, shours = stranding_events(
        ds["lon"].values, ds["lat"].values, ds["status"].values,
        ds["time"].values, scode)
    ds.close()
    assert len(slon) == 0                      # old code counted 1 here


def test_stranding_decoded_per_file_and_counts_final_step(tmp_path):
    p = _stranded_run(tmp_path / "st.nc")
    ds = xr.open_dataset(p)
    scode = code_of(ds["status"], STRANDED)
    assert scode == 2                          # NOT the hard-coded 1
    slon, slat, shours = stranding_events(
        ds["lon"].values, ds["lat"].values, ds["status"].values,
        ds["time"].values, scode)
    ds.close()
    # Both stranding events count — including the one at the final step,
    # which the old `last < nt-1` guard silently dropped.
    assert len(slon) == 2
    assert sorted(shours.tolist()) == [1.0, 2.0]
    np.testing.assert_allclose(sorted(slon.tolist()), [-41.5, -41.4])


def test_compute_beaching_end_to_end(tmp_path):
    p1 = _missing_data_run(tmp_path / "md.nc")
    p2 = _stranded_run(tmp_path / "st.nc")
    lons, lats = LONS, LATS
    r = compute_beaching([str(p1), str(p2)], lons, lats)
    assert r["n_particles_total"] == 6
    assert r["n_stranded"] == 2                # old code reported 3 (1 fake)
    assert abs(r["stranded_fraction"] - 2 / 6) < 1e-9
    assert r["strand_grid"].sum() > 0
    # Grid mass sits where the strandings happened, not at the domain edge.
    ri, ci = np.where(r["strand_grid"] > 0)
    assert all(lons[c] > -42.0 for c in ci)


def test_particles_to_grid_analytic():
    lons, lats = LONS, LATS
    # Two particles in the same cell + one NaN -> single cell flagged once.
    g = particles_to_grid(np.array([-41.01, -41.05, np.nan]),
                          np.array([-23.01, -23.05, np.nan]),
                          len(lons), len(lats))
    assert g.sum() == 1.0
    ri, ci = np.where(g == 1.0)
    assert lons[ci[0]] <= -41.01 < lons[ci[0]] + 0.1
    assert lats[ri[0]] <= -23.01 < lats[ri[0]] + 0.1
