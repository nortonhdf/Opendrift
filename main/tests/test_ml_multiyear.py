"""Multi-year pipeline: registry dispatch and path conventions."""

import numpy as np
import pytest

from main.ml import multiyear


def test_forcing_paths_2025_keeps_production_names():
    cur, wnd = multiyear.forcing_paths(2025)
    assert cur.name == "currents.nc" and wnd.name == "wind_cf.nc"
    cur22, wnd22 = multiyear.forcing_paths(2022)
    assert cur22.name == "currents_2022.nc" and wnd22.name == "wind_cf_2022.nc"


def test_registry_dispatches_by_timestamp_year(monkeypatch):
    calls = []

    class _Fake:
        def __init__(self, tag):
            self.tag = tag

        def at(self, lon, lat, when):
            calls.append(self.tag)
            return {"u_cur": 0, "v_cur": 0, "u_wind": 0, "v_wind": 0, "sst": 24.0}

        def close(self):
            pass

    reg = multiyear.ForcingRegistry.__new__(multiyear.ForcingRegistry)
    reg._samplers = {2022: _Fake(2022), 2025: _Fake(2025)}
    reg.at(-41.0, -23.0, np.datetime64("2022-07-15T06:00"))
    reg.at(-41.0, -23.0, np.datetime64("2025-01-01T00:00"))
    assert calls == [2022, 2025]


def test_download_refuses_frozen_holdout_year():
    with pytest.raises(AssertionError, match="2024"):
        multiyear.download(2024)


def test_coverage_ok_detects_silent_clipping(tmp_path):
    import xarray as xr

    def _file(t0, t1, n):
        times = np.arange(np.datetime64(t0), np.datetime64(t1),
                          np.timedelta64(1, "D"))[:n]
        ds = xr.Dataset({"uo": (("time",), np.zeros(len(times)))},
                        coords={"time": times})
        p = tmp_path / f"{t0}.nc"
        ds.to_netcdf(p)
        return p

    full = _file("2022-01-01", "2023-01-01", 365)
    clipped = _file("2022-06-01", "2023-01-01", 214)   # the real 2022 anfc case
    assert multiyear._coverage_ok(full, 2022) is True
    assert multiyear._coverage_ok(clipped, 2022) is False
