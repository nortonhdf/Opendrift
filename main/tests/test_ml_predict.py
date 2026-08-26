"""The live predictor behind the app tab: guard rails, not model quality.

Model quality is settled in test_ml_forecast_rigor and test_ml_footprint.
What matters here is that a forecast is REFUSED when the data behind it does
not exist — outside the forcing box, or in a year with no forcing file —
because a plausible-looking map drawn from nothing is worse than no map.
"""

import numpy as np
import pytest
from datetime import datetime

from main.ml import predict
from main.ml.dataset import KM_PER_DEG
from main.ml.scenario import feature_names


class _FakeSampler:
    """Antecedent block of the feature vector, without touching any file."""

    def features(self, year, lon, lat, release):
        n = len(feature_names()) - 8
        return [0.5] * n

    def close(self):
        pass


def _predictor(years=(2024,)):
    p = predict.Predictor.__new__(predict.Predictor)     # no artefacts needed
    p.years = list(years)
    p._sampler = _FakeSampler()
    p.fc = {"feature_names": feature_names(), "horizons_d": [1, 2, 3, 5, 7]}
    p.horizons = [1, 2, 3, 5, 7]
    return p


# ── refusals ─────────────────────────────────────────────────────────────────

def test_refuses_a_year_without_forcing():
    p = _predictor(years=[2024])
    with pytest.raises(ValueError, match="2019"):
        p.features(-40.0, -22.0, 28.0, 1000.0, datetime(2019, 5, 1))


def test_refuses_a_point_outside_the_forcing_box():
    p = _predictor()
    with pytest.raises(ValueError, match="fora da caixa"):
        p.features(-30.0, -22.0, 28.0, 1000.0, datetime(2024, 5, 1))
    with pytest.raises(ValueError, match="fora da caixa"):
        p.features(-40.0, -5.0, 28.0, 1000.0, datetime(2024, 5, 1))


def test_accepts_a_point_on_the_boundary():
    """The box edges are inclusive — the forcing covers them."""
    p = _predictor()
    x = p.features(predict.FORCING_LON_MIN, predict.FORCING_LAT_MIN,
                   28.0, 1000.0, datetime(2024, 5, 1))
    assert len(x) == len(feature_names())


def test_feature_vector_follows_the_declared_order():
    p = _predictor()
    names = feature_names()
    x = p.features(-40.1, -22.4, 27.5, 1234.0, datetime(2024, 5, 1))
    assert x[names.index("lon")] == pytest.approx(-40.1)
    assert x[names.index("lat")] == pytest.approx(-22.4)
    assert x[names.index("api")] == pytest.approx(27.5)
    assert x[names.index("water_depth_m")] == pytest.approx(1234.0)


# ── warnings ─────────────────────────────────────────────────────────────────

def test_short_lookback_is_reported_not_hidden():
    """A January release has no 90-day window inside its year — say so."""
    p = _predictor()
    names = feature_names()
    x = np.full(len(names), 1.0, np.float32)
    x[names.index("coverage_90d")] = 0.42
    msgs = p._coverage_warnings(x)
    assert any("90d" in m for m in msgs)
    assert not any("3d" in m for m in msgs)


def test_full_coverage_produces_no_warning():
    p = _predictor()
    names = feature_names()
    x = np.full(len(names), 1.0, np.float32)
    assert p._coverage_warnings(x) == []


# ── geometry ─────────────────────────────────────────────────────────────────

def test_offset_to_lonlat_inverts_the_km_frame():
    lon0, lat0 = -40.0, -22.0
    dx, dy = np.array([120.0, -75.0]), np.array([-60.0, 33.0])
    lon, lat = predict.offset_to_lonlat(lon0, lat0, dx, dy)
    back_dx = (lon - lon0) * KM_PER_DEG * np.cos(np.radians(lat0))
    back_dy = (lat - lat0) * KM_PER_DEG
    assert back_dx == pytest.approx(dx)
    assert back_dy == pytest.approx(dy)


def test_season_is_the_nearest_modelled_month_cyclically():
    assert predict.season_of(1) == "jan"
    assert predict.season_of(4) == "apr"
    assert predict.season_of(3) == "apr"       # 3 is one month from April
    assert predict.season_of(11) == "oct"
    assert predict.season_of(12) == "jan"      # wraps the year end
