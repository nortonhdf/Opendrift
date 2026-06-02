"""
Campos Basin oil field definitions.
ADIOS oil types mapped by closest API gravity match in the ADIOS database.
  <15° API  → GENERIC HEAVY CRUDE
  15-22° API → GENERIC MEDIUM CRUDE  (or BACHEQUERO HEAVY for very viscous)
  >22° API  → GENERIC LIGHT CRUDE
"""

CAMPOS_FIELDS = {
    "Peregrino": {
        "lon": -41.2593,
        "lat": -23.3183,
        "oil_type": "GENERIC HEAVY CRUDE",   # ~13° API
        "api": 13.0,
        "operator": "PRIO / Equinor",
        "water_depth_m": 100,
        "description": "Heavy crude (~13° API), ~85 km offshore, Campos Basin",
    },
    "Marlim": {
        "lon": -40.60,
        "lat": -22.60,
        "oil_type": "GENERIC MEDIUM CRUDE",  # ~20° API
        "api": 20.0,
        "operator": "Petrobras",
        "water_depth_m": 720,
        "description": "Medium-heavy crude (~20° API), deep-water Campos",
    },
    "Roncador": {
        "lon": -39.80,
        "lat": -22.40,
        "oil_type": "GENERIC HEAVY CRUDE",   # ~18° API
        "api": 18.0,
        "operator": "Petrobras",
        "water_depth_m": 1800,
        "description": "Heavy crude (~18° API), ultra-deep Campos Basin",
    },
    "Jubarte": {
        "lon": -39.60,
        "lat": -21.30,
        "oil_type": "GENERIC HEAVY CRUDE",   # ~16.5° API
        "api": 16.5,
        "operator": "Petrobras / Shell",
        "water_depth_m": 1300,
        "description": "Heavy crude (~16.5° API), Espírito Santo Basin",
    },
    "Frade": {
        "lon": -41.00,
        "lat": -22.10,
        "oil_type": "GENERIC HEAVY CRUDE",   # ~18° API
        "api": 18.0,
        "operator": "Petrobras / Chevron",
        "water_depth_m": 1100,
        "description": "Heavy crude (~18° API), northern Campos Basin",
    },
    "Albacora": {
        "lon": -40.20,
        "lat": -22.00,
        "oil_type": "GENERIC MEDIUM CRUDE",  # ~19° API
        "api": 19.0,
        "operator": "Petrobras",
        "water_depth_m": 300,
        "description": "Medium-heavy crude (~19° API), Campos Basin",
    },
}
