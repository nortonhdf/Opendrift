"""Tests for status flag decoding (audit finding #1: codes vary per file)."""

import numpy as np
import xarray as xr

from main.status_utils import (
    ACTIVE, MISSING_DATA, STRANDED,
    code_of, final_status, last_valid_index, status_map,
)


def _status_var(meanings: str, values=None):
    attrs = {"flag_meanings": meanings}
    if values is not None:
        attrs["flag_values"] = values
    return xr.DataArray(np.zeros(1), attrs=attrs)


def test_status_map_orders_vary_per_file():
    # The same integer means different things depending on occurrence order.
    assert status_map(_status_var("active missing_data")) == {
        0: ACTIVE, 1: MISSING_DATA}
    assert status_map(_status_var("active stranded")) == {
        0: ACTIVE, 1: STRANDED}
    assert status_map(_status_var("active stranded missing_data")) == {
        0: ACTIVE, 1: STRANDED, 2: MISSING_DATA}
    assert status_map(_status_var("active missing_data stranded")) == {
        0: ACTIVE, 1: MISSING_DATA, 2: STRANDED}


def test_status_map_accepts_plain_attrs_dict_and_missing_attrs():
    assert status_map({"flag_meanings": "active stranded"})[1] == STRANDED
    assert status_map({}) == {0: ACTIVE}          # run with no deactivation
    assert status_map(xr.DataArray(np.zeros(1))) == {0: ACTIVE}


def test_status_map_uses_flag_values_when_present():
    var = _status_var("active stranded", values=np.array([0, 3]))
    assert status_map(var) == {0: ACTIVE, 3: STRANDED}


def test_code_of():
    var = _status_var("active missing_data stranded")
    assert code_of(var, STRANDED) == 2
    assert code_of(var, MISSING_DATA) == 1
    assert code_of(var, "never_happened") is None


def test_last_valid_index():
    nan = np.nan
    lon = np.array([
        [1.0, 2.0, 3.0],     # survives to the end -> 2
        [1.0, 2.0, nan],     # deactivated after step 1 -> 1
        [1.0, nan, nan],     # deactivated after step 0 -> 0
        [nan, nan, nan],     # never valid -> -1
    ])
    assert last_valid_index(lon).tolist() == [2, 1, 0, -1]


def test_final_status_decodes_per_particle():
    # OpenDrift keeps the position valid at the step where the status flips
    # to a deactivation code, and pads with NaN afterwards (verified on the
    # real 288-file archive during the audit) — fixtures mirror that.
    nan = np.nan
    lon = np.array([[1.0, 2.0, 3.0],
                    [1.0, 2.0, nan],
                    [1.0, 2.0, nan]])
    status = np.array([[0, 0, 0],
                       [0, 2, 2],    # code 2 = stranded in this file
                       [0, 1, 1]])   # code 1 = missing_data in this file
    smap = {0: ACTIVE, 1: MISSING_DATA, 2: STRANDED}
    out = final_status(lon, status, smap)
    assert out.tolist() == [ACTIVE, STRANDED, MISSING_DATA]
