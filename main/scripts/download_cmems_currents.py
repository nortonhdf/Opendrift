"""
Download CMEMS global ocean surface currents for the Campos Basin region — full year.

Usage:
    python main/scripts/download_cmems_currents.py            # full 2025
    python main/scripts/download_cmems_currents.py 2024       # different year
"""

import sys
from pathlib import Path
import copernicusmarine

YEAR = sys.argv[1] if len(sys.argv) > 1 else "2025"

def main():
    out_dir = Path("main/inputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Remove existing file so CMEMS doesn't auto-rename to currents_raw_(1).nc
    existing = out_dir / "currents_raw.nc"
    if existing.exists():
        existing.unlink()

    copernicusmarine.subset(
        dataset_id="cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m",
        variables=["uo", "vo"],
        minimum_longitude=-42.5,
        maximum_longitude=-39.0,
        minimum_latitude=-24.5,
        maximum_latitude=-21.0,
        start_datetime=f"{YEAR}-01-01T00:00:00",
        end_datetime=f"{YEAR}-12-31T23:00:00",
        minimum_depth=0,
        maximum_depth=1,
        output_filename="currents_raw.nc",
        output_directory=str(out_dir),
    )
    print("OK:", (out_dir / "currents_raw.nc").resolve())

if __name__ == "__main__":
    main()
