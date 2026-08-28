"""Reading an archive whose releases are not one of the six fields.

`release_config` is what lets a scenario come from anywhere: the six-field
manifests name a field and look the rest up in fields_config, while the
seed-grid manifests carry lon/lat/api/depth because there is no registry
entry to look up. Getting this wrong does not crash — it silently forecasts
with the wrong oil or the wrong depth — so the precedence is pinned here.
"""

import numpy as np
import pytest

from main.fields_config import CAMPOS_FIELDS
from main.ml import footprint, scenario
from main.ml.seedgrid import GRID_DIR_TMPL


# ── release_config precedence ────────────────────────────────────────────────

def test_six_field_entry_reads_the_registry():
    entry = {"field": "Marlim", "season": "jan", "day": 15}
    cfg = scenario.release_config(entry)
    assert cfg["api"] == CAMPOS_FIELDS["Marlim"]["api"]
    assert cfg["water_depth_m"] == CAMPOS_FIELDS["Marlim"]["water_depth_m"]


def test_manifest_values_win_over_the_registry():
    """A future archive must be able to override without touching the builder."""
    entry = {"field": "Marlim", "season": "jan", "day": 15, "api": 12.5}
    assert scenario.release_config(entry)["api"] == 12.5


def test_grid_entry_needs_no_registry_entry():
    entry = {"field": "grid007", "season": "apr", "day": 5,
             "lon": -40.1, "lat": -22.4, "api": 28.0, "water_depth_m": None}
    cfg = scenario.release_config(entry)
    assert cfg["api"] == 28.0
    assert cfg["lon"] == -40.1 and cfg["lat"] == -22.4


def test_explicit_null_depth_becomes_nan_not_zero():
    """Unknown bathymetry must reach the model as NaN; 0 m would be a lie."""
    entry = {"field": "grid007", "season": "apr", "day": 5,
             "api": 28.0, "water_depth_m": None}
    assert np.isnan(scenario.release_config(entry)["water_depth_m"])


def test_missing_depth_key_also_becomes_nan():
    entry = {"field": "grid007", "season": "apr", "day": 5, "api": 28.0}
    assert np.isnan(scenario.release_config(entry)["water_depth_m"])


def test_unknown_field_without_api_is_refused():
    """Better to stop than to invent an oil type for an unnamed location."""
    with pytest.raises(KeyError):
        scenario.release_config({"field": "grid007", "season": "apr", "day": 5})


# ── archive selection ────────────────────────────────────────────────────────

def test_grid_years_select_the_grid_archive():
    for mod in (scenario, footprint):
        got = mod._archives(grid_years=[2025])
        assert got, f"{mod.__name__} não achou o arquivo de grade"
        for year, path in got:
            assert year == 2025
            assert GRID_DIR_TMPL.format(year=2025) in str(path)


def test_default_selection_is_still_the_six_field_archive():
    for mod in (scenario, footprint):
        for _, path in mod._archives():
            assert "grid" not in path.parent.name


def test_holdout_selection_is_the_frozen_year():
    for mod in (scenario, footprint):
        years = [y for y, _ in mod._archives(holdout=True)]
        assert years == [2024]


def test_missing_grid_archive_says_how_to_build_it():
    with pytest.raises(SystemExit, match="seedgrid"):
        scenario._archives(grid_years=[1999])
