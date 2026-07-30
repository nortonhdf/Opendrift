"""Analytic checks for the ML evaluation metrics."""

import numpy as np
import pytest

from main.ml.metrics import (
    brier, centroid_error_km, haversine_km, iou, liu_weisberg_ss,
)


def test_haversine_known_distance():
    # 1 degree of latitude ~ 111.2 km anywhere
    assert haversine_km(-41.0, -23.0, -41.0, -22.0) == pytest.approx(111.2, abs=0.5)
    assert haversine_km(-41.0, -23.0, -41.0, -23.0) == 0.0


def test_liu_weisberg_perfect_and_worthless():
    lon = np.array([-41.0, -40.9, -40.8, -40.7])
    lat = np.full(4, -23.0)
    # Identical trajectories -> perfect skill
    assert liu_weisberg_ss(lon, lat, lon, lat) == pytest.approx(1.0)
    # Stationary model vs moving obs: d_i == l_i at every step -> s=1 -> SS=0
    mod_lon = np.full(4, lon[0])
    assert liu_weisberg_ss(lon, lat, mod_lon, lat) == pytest.approx(0.0, abs=1e-6)


def test_liu_weisberg_half_speed_model():
    # Model moving at half the observed speed: d_i = l_i/2 -> s=0.5 -> SS=0.5
    lon = np.array([-41.0, -40.9, -40.8, -40.7])
    lat = np.full(4, -23.0)
    mod_lon = np.array([-41.0, -40.95, -40.90, -40.85])
    assert liu_weisberg_ss(lon, lat, mod_lon, lat) == pytest.approx(0.5, abs=0.01)


def test_liu_weisberg_stationary_obs():
    lon = np.full(4, -41.0)
    lat = np.full(4, -23.0)
    assert liu_weisberg_ss(lon, lat, lon, lat) == 1.0        # both parked
    mod = np.array([-41.0, -40.9, -40.8, -40.7])
    assert liu_weisberg_ss(lon, lat, mod, lat) == 0.0        # model wanders


def test_centroid_error():
    o_lon = np.array([-41.0, -40.8]); o_lat = np.array([-23.0, -23.0])
    m_lon = o_lon; m_lat = o_lat + 1.0
    assert centroid_error_km(o_lon, o_lat, m_lon, m_lat) == pytest.approx(111.2, abs=0.5)


def test_iou_cases():
    a = np.zeros((4, 4), bool); a[1, 1] = a[1, 2] = True
    b = np.zeros((4, 4), bool); b[1, 2] = b[1, 3] = True
    assert iou(a, a) == 1.0
    assert iou(a, ~a) == 0.0
    assert iou(a, b) == pytest.approx(1 / 3)
    assert iou(np.zeros((2, 2), bool), np.zeros((2, 2), bool)) == 1.0


def test_brier():
    assert brier([1.0, 0.0], [1, 0]) == 0.0
    assert brier([0.5, 0.5], [1, 0]) == pytest.approx(0.25)
    assert brier([0.0], [1]) == 1.0
