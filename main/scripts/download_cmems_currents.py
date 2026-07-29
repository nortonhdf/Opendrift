"""
Download CMEMS surface currents AND sea-surface temperature for the Campos
Basin region — full year.

Two subsets from sibling datasets on the same 1/12 deg daily grid:
  currents_raw.nc  <- cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m     (uo, vo)
  sst_raw.nc       <- cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m  (thetao)

SST feeds OpenOil weathering (evaporation/emulsification); without it the
model weathers oil at a 10 degC fallback (audit finding grave #3).

Usage:
    python main/scripts/download_cmems_currents.py            # full 2025
    python main/scripts/download_cmems_currents.py 2024       # different year
"""

import sys
from pathlib import Path
import copernicusmarine

YEAR = sys.argv[1] if len(sys.argv) > 1 else "2025"

# Forcing box approved in the 2026-07-29 audit (old 3.5x3.5 deg box lost
# ~16% of particles over the boundary within 120 h — finding grave #4).
LON_MIN, LON_MAX = -45.0, -36.0
LAT_MIN, LAT_MAX = -27.0, -19.0


def _subset(dataset_id: str, variables: list[str], filename: str, out_dir: Path):
    # Remove existing file so CMEMS doesn't auto-rename to name_(1).nc
    existing = out_dir / filename
    if existing.exists():
        existing.unlink()
    copernicusmarine.subset(
        dataset_id=dataset_id,
        variables=variables,
        minimum_longitude=LON_MIN,
        maximum_longitude=LON_MAX,
        minimum_latitude=LAT_MIN,
        maximum_latitude=LAT_MAX,
        start_datetime=f"{YEAR}-01-01T00:00:00",
        end_datetime=f"{YEAR}-12-31T23:00:00",
        minimum_depth=0,
        maximum_depth=1,
        output_filename=filename,
        output_directory=str(out_dir),
    )
    print("OK:", (out_dir / filename).resolve())


def main():
    out_dir = Path("main/inputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    _subset("cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m", ["uo", "vo"],
            "currents_raw.nc", out_dir)
    _subset("cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m", ["thetao"],
            "sst_raw.nc", out_dir)


if __name__ == "__main__":
    main()
