"""Campos Basin oil field definitions.

Coordinates: centre of each field's official production-area polygon
("Contracted Areas" layer, EPE Webmap service WMS_Webmap_EPE_Data layer 59,
ANP data; UP = '<FIELD>-A', CLASSIFICA = 'Campos'; consulted 2026-07-29,
geometry export disabled so the bounding-box centre is used).
The audit found the previous hand-typed positions off by ~28-118 km for
Albacora/Roncador/Marlim/Frade; Peregrino was within ~2 km of the official
polygon centre (it matches the public FPSO position).

Jubarte (Espirito Santo Basin) was replaced by Papa-Terra to keep the project
strictly in the Campos Basin (author decision 2026-07-29).

ADIOS oil type follows the API rule (author decision: the rule wins):
    <15 deg API  -> GENERIC HEAVY CRUDE
    15-22        -> GENERIC MEDIUM CRUDE
    >22          -> GENERIC LIGHT CRUDE
"""


def oil_type_for_api(api: float) -> str:
    """Map API gravity to the project's generic ADIOS oil (documented rule)."""
    if api < 15:
        return "GENERIC HEAVY CRUDE"
    if api <= 22:
        return "GENERIC MEDIUM CRUDE"
    return "GENERIC LIGHT CRUDE"


CAMPOS_FIELDS = {
    "Peregrino": {
        "lon": -41.2816,
        "lat": -23.3130,
        "api": 13.0,
        "oil_type": "GENERIC HEAVY CRUDE",
        "operator": "PRIO / Equinor",
        "water_depth_m": 100,
        "description": "Heavy crude (~13 deg API), ~85 km offshore, SW Campos Basin",
    },
    "Marlim": {
        "lon": -40.0889,
        "lat": -22.4210,
        "api": 20.0,
        "oil_type": "GENERIC MEDIUM CRUDE",
        "operator": "Petrobras",
        "water_depth_m": 720,
        "description": "Medium crude (17-21 deg API), deep-water NE Campos",
    },
    "Roncador": {
        "lon": -39.7790,
        "lat": -21.9275,
        "api": 18.0,
        "oil_type": "GENERIC MEDIUM CRUDE",
        "operator": "Petrobras",
        "water_depth_m": 1800,
        "description": "Medium crude (18-31 deg API by module), ultra-deep NE Campos",
    },
    "Papa-Terra": {
        "lon": -41.0655,
        "lat": -23.5265,
        "api": 15.5,
        "oil_type": "GENERIC MEDIUM CRUDE",
        "operator": "Petrobras (EPE/ANP layer)",
        "water_depth_m": 1190,
        "description": "Heavy-medium crude (14-17.4 deg API), southern Campos Basin",
    },
    "Frade": {
        "lon": -39.8597,
        "lat": -21.9054,
        "api": 18.0,
        "oil_type": "GENERIC MEDIUM CRUDE",
        "operator": "PRIO (ex Petrobras/Chevron)",
        "water_depth_m": 1100,
        "description": "Medium-heavy crude (~18 deg API), northern Campos, west of Roncador",
    },
    "Albacora": {
        "lon": -39.9613,
        "lat": -22.1281,
        "api": 19.0,
        "oil_type": "GENERIC MEDIUM CRUDE",
        "operator": "Petrobras",
        "water_depth_m": 300,
        "description": "Medium crude (~19 deg API), NE Campos Basin",
    },
}
