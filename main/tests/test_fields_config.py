"""Field registry invariants (audit findings #9 and question #3/#4/#5)."""

from main.domain_config import (
    FORCING_LAT_MAX, FORCING_LAT_MIN, FORCING_LON_MAX, FORCING_LON_MIN,
)
from main.fields_config import CAMPOS_FIELDS, oil_type_for_api


def test_six_campos_fields_no_jubarte():
    assert len(CAMPOS_FIELDS) == 6
    assert "Jubarte" not in CAMPOS_FIELDS          # Espirito Santo Basin
    assert "Papa-Terra" in CAMPOS_FIELDS


def test_oil_type_rule():
    assert oil_type_for_api(13.0) == "GENERIC HEAVY CRUDE"
    assert oil_type_for_api(15.0) == "GENERIC MEDIUM CRUDE"
    assert oil_type_for_api(22.0) == "GENERIC MEDIUM CRUDE"
    assert oil_type_for_api(25.0) == "GENERIC LIGHT CRUDE"


def test_assigned_oil_types_follow_the_rule():
    # The audit found Roncador/Frade (18 API) labelled HEAVY against the
    # documented 15-22 -> MEDIUM rule; the rule now wins (author decision).
    for name, cfg in CAMPOS_FIELDS.items():
        assert cfg["oil_type"] == oil_type_for_api(cfg["api"]), name


def test_fields_inside_forcing_box():
    for name, cfg in CAMPOS_FIELDS.items():
        assert FORCING_LON_MIN < cfg["lon"] < FORCING_LON_MAX, name
        assert FORCING_LAT_MIN < cfg["lat"] < FORCING_LAT_MAX, name


def test_official_anp_coordinates_pinned():
    # Bounding-box centres of the ANP/EPE production polygons (2026-07-29).
    expected = {
        "Peregrino":  (-41.2816, -23.3130),
        "Marlim":     (-40.0889, -22.4210),
        "Roncador":   (-39.7790, -21.9275),
        "Papa-Terra": (-41.0655, -23.5265),
        "Frade":      (-39.8597, -21.9054),
        "Albacora":   (-39.9613, -22.1281),
    }
    for name, (lon, lat) in expected.items():
        assert abs(CAMPOS_FIELDS[name]["lon"] - lon) < 1e-4, name
        assert abs(CAMPOS_FIELDS[name]["lat"] - lat) < 1e-4, name
