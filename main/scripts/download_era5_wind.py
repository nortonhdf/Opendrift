"""
Download ERA5 10m wind for the Campos Basin region — full year.

Usage:
    python main/scripts/download_era5_wind.py            # full 2025
    python main/scripts/download_era5_wind.py 2024       # different year
"""

import sys
from pathlib import Path
import cdsapi

YEAR  = sys.argv[1] if len(sys.argv) > 1 else "2025"
AREA  = [-21.0, -42.5, -24.5, -39.0]   # [North, West, South, East] — covers all 6 Campos fields

def main():
    out = Path("main/inputs/wind_raw.nc")
    out.parent.mkdir(parents=True, exist_ok=True)

    c = cdsapi.Client()
    c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "format": "netcdf",
            "variable": [
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
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
