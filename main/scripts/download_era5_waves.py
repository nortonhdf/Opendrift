"""Download ERA5 wave parameters for the Campos Basin region — full year.

Downloads MONTH BY MONTH and concatenates. That is not a style choice: the
single whole-year request this script used to make is refused by the CDS with
``403 ... cost limits exceeded``. The wind download gets away with it because
it asks for two variables; three wave variables over the same box and hours
cross the limit. The script had never been run, so nobody had found out.

Monthly parts are cached in ``main/inputs/_waves_parts/``, so an interrupted
download resumes instead of starting over.

Usage:
    python main/scripts/download_era5_waves.py            # full 2025
    python main/scripts/download_era5_waves.py 2024       # different year
"""

import calendar
import hashlib
import sys
from pathlib import Path

import cdsapi
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from main.domain_config import (  # noqa: E402
    FORCING_LAT_MAX, FORCING_LAT_MIN, FORCING_LON_MAX, FORCING_LON_MIN,
)

YEAR = sys.argv[1] if len(sys.argv) > 1 else "2025"
# [North, West, South, East] — audit-approved wide box (grave #4)
AREA = [FORCING_LAT_MAX, FORCING_LON_MIN, FORCING_LAT_MIN, FORCING_LON_MAX]

# The two Stokes components are what actually move oil. Without them the
# file loads, the reader reports OK, and the trajectory is bit-identical to
# the wind parameterisation — measured, see CAMADA_IA.md. Height/period/
# direction feed the wave-dependent weathering terms, not the advection.
VARIABLES = [
    "significant_height_of_combined_wind_waves_and_swell",
    "mean_wave_period",
    "mean_wave_direction",
    "u_component_stokes_drift",
    "v_component_stokes_drift",
]

# Cached parts are keyed by the variable set: changing VARIABLES must not
# silently reuse parts downloaded without the new fields.
_TAG = hashlib.sha1(",".join(sorted(VARIABLES)).encode()).hexdigest()[:8]
PARTS = Path(f"main/inputs/_waves_parts_{_TAG}")


def out_path(year: str) -> Path:
    """2025 keeps the unsuffixed production name, as wind does."""
    return Path("main/inputs/waves_raw.nc" if year == "2025"
                else f"main/inputs/waves_raw_{year}.nc")


def main() -> None:
    PARTS.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client()
    parts = []

    for month in range(1, 13):
        part = PARTS / f"{YEAR}-{month:02d}.nc"
        if not part.exists():
            n_days = calendar.monthrange(int(YEAR), month)[1]
            print(f"[{month:02d}/12] baixando {YEAR}-{month:02d} "
                  f"({n_days} dias)…", flush=True)
            client.retrieve(
                "reanalysis-era5-single-levels",
                {
                    "product_type": "reanalysis",
                    "format": "netcdf",
                    "variable": VARIABLES,
                    "year": YEAR,
                    "month": [f"{month:02d}"],
                    "day": [f"{d:02d}" for d in range(1, n_days + 1)],
                    "time": [f"{h:02d}:00" for h in range(24)],
                    "area": AREA,
                },
                str(part),
            )
        else:
            print(f"[{month:02d}/12] {part.name} já existe — pulando",
                  flush=True)
        parts.append(part)

    out = out_path(YEAR)
    out.parent.mkdir(parents=True, exist_ok=True)
    ds = xr.open_mfdataset([str(p) for p in parts], combine="by_coords")
    time_name = "valid_time" if "valid_time" in ds.coords else "time"
    n = ds.sizes[time_name]
    expected = 366 * 24 if calendar.isleap(int(YEAR)) else 365 * 24
    ds.to_netcdf(out)
    ds.close()
    print(f"OK: {out.resolve()}  ({n} passos horários; esperado {expected})")
    if n != expected:
        print("[AVISO] cobertura incompleta — o CDS recorta pedidos "
              "silenciosamente; confira antes de usar.")


if __name__ == "__main__":
    main()
