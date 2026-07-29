"""Domain constants: one source of truth, consistent everywhere (audit #11)."""

from datetime import datetime

from main import domain_config as dc
from main.scripts import compute_beaching, compute_risk_grids


def test_grid_matches_forcing_box():
    assert dc.GRID_LON_MIN == dc.FORCING_LON_MIN
    assert dc.GRID_LON_MAX == dc.FORCING_LON_MAX
    assert dc.GRID_LAT_MIN == dc.FORCING_LAT_MIN
    assert dc.GRID_LAT_MAX == dc.FORCING_LAT_MAX


def test_audit_approved_box_values():
    assert (dc.FORCING_LON_MIN, dc.FORCING_LON_MAX) == (-45.0, -36.0)
    assert (dc.FORCING_LAT_MIN, dc.FORCING_LAT_MAX) == (-27.0, -19.0)
    assert dc.GRID_RES == 0.1


def test_aggregation_scripts_share_the_domain():
    for mod in (compute_beaching, compute_risk_grids):
        assert mod.LON_MIN == dc.GRID_LON_MIN
        assert mod.LON_MAX == dc.GRID_LON_MAX
        assert mod.LAT_MIN == dc.GRID_LAT_MIN
        assert mod.LAT_MAX == dc.GRID_LAT_MAX
        assert mod.GRID_RES == dc.GRID_RES
        assert list(mod.SEASONS) == ["jan", "apr", "jul", "oct"]


def test_season_and_ensemble_dates():
    assert dc.season_date("jul") == datetime(2025, 7, 15)
    dates = dc.ensemble_dates("oct")
    assert len(dates) == 10
    assert dates[0] == datetime(2025, 10, 1)
    assert dates[-1] == datetime(2025, 10, 28)
