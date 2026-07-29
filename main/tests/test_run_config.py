"""Verify the EFFECTIVE model configuration (audit critical finding #2).

The old code had `if USE_3D and DISABLE_VERTICAL_MIXING:` with USE_3D=False —
never executed — and used a config key that does not exist. These tests pin
the real behaviour of configure_physics on a live OpenOil instance.
"""

import pytest

from opendrift.models.openoil import OpenOil

from main.run_open_oil import configure_physics, configure_spill


@pytest.fixture(scope="module")
def model():
    return OpenOil(loglevel=50, weathering_model="noaa")


def test_vertical_mixing_off_by_default_path(model):
    configure_physics(model, use_wind=True, use_waves=False, vertical_mixing=False)
    assert model.get_config("drift:vertical_mixing") is False
    assert model.get_config("drift:stokes_drift") is False


def test_vertical_mixing_can_be_enabled(model):
    configure_physics(model, use_wind=True, use_waves=False, vertical_mixing=True)
    assert model.get_config("drift:vertical_mixing") is True


def test_waves_enable_tabularised_stokes(model):
    configure_physics(model, use_wind=True, use_waves=True, vertical_mixing=False)
    assert model.get_config("drift:stokes_drift") is True
    assert model.get_config("drift:use_tabularised_stokes_drift") is True


def test_rk4_and_declared_uncertainties(model):
    configure_physics(model, use_wind=True, use_waves=False, vertical_mixing=False)
    assert model.get_config("drift:advection_scheme") == "runge-kutta4"
    assert model.get_config("drift:current_uncertainty") == 0.05
    assert model.get_config("drift:wind_uncertainty") == 0.5


def test_sst_fallback_is_campos_not_10C(model):
    configure_physics(model, use_wind=True, use_waves=False, vertical_mixing=False)
    assert model.get_config("environment:fallback:sea_water_temperature") == 24.0


def test_spill_volume_is_declared(model):
    configure_spill(model, 10.0)
    assert model.get_config("seed:m3_per_hour") == 10.0
    configure_spill(model, 2.5)
    assert model.get_config("seed:m3_per_hour") == 2.5
