"""Seed-location sampling: the invariants an archive depends on.

The archive is generated incrementally over hours, so the sample has to be
stable — a location list that changes when you ask for more locations would
silently orphan every run already computed, and the manifest would describe
points the NetCDFs were not run at.
"""

import numpy as np
import pytest

from main.ml import seedgrid as sg


# ── the load-bearing invariant ───────────────────────────────────────────────

def test_sample_of_n_is_a_prefix_of_a_larger_sample():
    """A pilot batch must be part of the full archive, not a different one.

    Latin-hypercube stratifies over the number of points drawn, so a pool
    sized as a multiple of n gives two unrelated samples for two values of n.
    seedgrid draws a fixed pool for exactly this reason.
    """
    small = sg.sample_locations(6)
    large = sg.sample_locations(30)
    assert small == large[:6]


def test_sampled_locations_are_distinct():
    locs = sg.sample_locations(30)
    assert len(set(locs)) == len(locs)


def test_sample_stays_inside_the_declared_region():
    for lon, lat in sg.sample_locations(30):
        assert sg.SAMPLE_LON[0] <= lon <= sg.SAMPLE_LON[1]
        assert sg.SAMPLE_LAT[0] <= lat <= sg.SAMPLE_LAT[1]


def test_sample_is_reproducible():
    assert sg.sample_locations(12) == sg.sample_locations(12)


# ── the land mask ────────────────────────────────────────────────────────────

def _mask(shape=(20, 20), land=None):
    """Water everywhere except an optional land block."""
    m = np.ones(shape, bool)
    if land:
        (i0, i1), (j0, j1) = land
        m[i0:i1, j0:j1] = False
    lats = np.linspace(-27.0, -19.0, shape[0])
    lons = np.linspace(-45.0, -36.0, shape[1])
    return lons, lats, m


def test_open_water_requires_a_margin_of_water():
    lons, lats, mask = _mask(land=((10, 12), (10, 12)))
    # A point sitting on the land block is rejected...
    assert not sg._is_open_water(lons, lats, mask, lons[10], lats[10])
    # ...and so is one a single cell away, because the margin still touches it.
    assert not sg._is_open_water(lons, lats, mask, lons[9], lats[9])
    # Far from it, the same grid is fine.
    assert sg._is_open_water(lons, lats, mask, lons[4], lats[4])


def test_points_at_the_grid_edge_are_rejected():
    """No margin can be checked there, so the answer must be no, not a crash."""
    lons, lats, mask = _mask()
    assert not sg._is_open_water(lons, lats, mask, lons[0], lats[0])
    assert not sg._is_open_water(lons, lats, mask, lons[-1], lats[-1])


def test_margin_zero_still_rejects_land():
    lons, lats, mask = _mask(land=((10, 12), (10, 12)))
    assert not sg._is_open_water(lons, lats, mask, lons[10], lats[10],
                                 margin=0)


# ── task expansion ───────────────────────────────────────────────────────────

def test_tasks_cover_every_location_season_and_day():
    locs = [(-40.0, -22.0), (-39.0, -23.0)]
    t = sg.tasks(locs, seasons=["jan", "jul"], days=[5, 15])
    assert len(t) == 2 * 2 * 2
    assert len({(k, s, d) for k, _, _, s, d in t}) == len(t)
    # Location index and coordinates travel together — the manifest key is
    # built from the index, so a mismatch would mislabel every run.
    for k, lon, lat, _, _ in t:
        assert (lon, lat) == locs[k]


def test_default_task_count_matches_the_declared_plan():
    locs = sg.sample_locations(5)
    assert len(sg.tasks(locs)) == 5 * len(sg.SEASON_MONTHS) * len(sg.START_DAYS)


def test_reference_api_maps_to_a_real_oil_type():
    from main.fields_config import oil_type_for_api
    assert isinstance(oil_type_for_api(sg.REF_API), str)
    assert oil_type_for_api(sg.REF_API)
