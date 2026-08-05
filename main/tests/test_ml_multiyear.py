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
