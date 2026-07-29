"""
Download ERA5 wave parameters for the Campos Basin region — full year.

Usage:
    python main/scripts/download_era5_waves.py            # full 2025
    python main/scripts/download_era5_waves.py 2024       # different year
"""

import sys
from pathlib import Path

import cdsapi

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from main.domain_config import (  # noqa: E402
    FORCING_LAT_MAX, FORCING_LAT_MIN, FORCING_LON_MAX, FORCING_LON_MIN,
)

YEAR  = sys.argv[1] if len(sys.argv) > 1 else "2025"
# [North, West, South, East] — audit-approved wide box (grave #4)
AREA  = [FORCING_LAT_MAX, FORCING_LON_MIN, FORCING_LAT_MIN, FORCING_LON_MAX]

def main():
    out = Path("main/inputs/waves_raw.nc")
    out.parent.mkdir(parents=True, exist_ok=True)

    c = cdsapi.Client()
    c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "format": "netcdf",
            "variable": [
                "significant_height_of_combined_wind_waves_and_swell",
                "mean_wave_period",
                "mean_wave_direction",
            ],
            "year": YEAR,
            "month": [f"{m:02d}" for m in range(1, 13)],
            "day":   [f"{d:02d}" for d in range(1, 32)],
            "time":  [f"{h:02d}:00" for h in range(24)],
            "area":  AREA,
        },
        str(out),
    )
    print("OK:", out.resolve())

if __name__ == "__main__":
    main()
