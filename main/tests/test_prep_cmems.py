"""prep_cmems_currents must surface-select, CF-rename and merge SST."""

import numpy as np
import pytest
import xarray as xr

from main.scripts.prep_cmems_currents import prep


def _ds(var_names, with_depth=True):
    shape = (2, 2, 3, 3)  # time, depth, lat, lon
    coords = {
        "time": np.array(["2025-01-01", "2025-01-02"], dtype="datetime64[ns]"),
        "depth": [0.49, 1.5],
        "latitude": [-23.0, -22.9, -22.8],
        "longitude": [-41.0, -40.9, -40.8],
    }
    dims = ("time", "depth", "latitude", "longitude")
    if not with_depth:
        shape = (2, 3, 3)
        dims = ("time", "latitude", "longitude")
        coords.pop("depth")
    data = {v: (dims, np.full(shape, i, dtype=np.float32))
            for i, v in enumerate(var_names)}
    return xr.Dataset(data, coords=coords)


def test_prep_renames_and_merges_sst():
    out = prep(_ds(["uo", "vo"]), _ds(["thetao"]))
    assert "x_sea_water_velocity" in out.data_vars
    assert "y_sea_water_velocity" in out.data_vars
    assert "sea_water_temperature" in out.data_vars
    assert out["sea_water_temperature"].attrs["standard_name"] == "sea_water_temperature"
    assert "depth" not in out.dims          # surface layer selected


def test_prep_without_sst_still_works():
    out = prep(_ds(["uo", "vo"]), None)
    assert "sea_water_temperature" not in out.data_vars
    assert "x_sea_water_velocity" in out.data_vars


def test_prep_rejects_wrong_variables():
    with pytest.raises(ValueError, match="uo/vo"):
        prep(_ds(["foo", "bar"]))
    with pytest.raises(ValueError, match="thetao"):
        prep(_ds(["uo", "vo"]), _ds(["not_sst"]))
