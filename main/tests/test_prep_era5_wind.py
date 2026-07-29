"""prep_era5_wind: single-step rename + CF attrs (merged prep+patch)."""

import numpy as np
import pytest
import xarray as xr

from main.scripts.prep_era5_wind import prep


def _era5(with_valid_time=True):
    tname = "valid_time" if with_valid_time else "time"
    coords = {
        tname: np.array(["2025-01-01T00", "2025-01-01T01"], dtype="datetime64[ns]"),
        "latitude": [-23.0, -22.75],
        "longitude": [-41.0, -40.75],
    }
    shape = (2, 2, 2)
    dims = (tname, "latitude", "longitude")
    return xr.Dataset(
        {"u10": (dims, np.ones(shape, np.float32)),
         "v10": (dims, np.zeros(shape, np.float32))},
        coords=coords,
    )


def test_prep_renames_and_sets_cf_attrs():
    out = prep(_era5())
    assert "time" in out.coords and "valid_time" not in out.coords
    assert out["x_wind"].attrs["standard_name"] == "eastward_wind"
    assert out["y_wind"].attrs["standard_name"] == "northward_wind"
    assert out["x_wind"].attrs["units"] == "m s-1"


def test_prep_rejects_unknown_variables():
    ds = _era5().rename({"u10": "foo", "v10": "bar"})
    with pytest.raises(ValueError, match="u10/v10"):
        prep(ds)
