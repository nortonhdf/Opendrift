"""
OpenDrift / OpenOil runner

Como usar (CLI):
    python main/run_open_oil.py

Como usar (programático):
    from main.run_open_oil import run_simulation
    result = run_simulation(field_name="Peregrino", duration_hours=120, use_wind=True)

Saídas:
    main/outputs/openoil_run.nc
    main/outputs/tracks.png
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from opendrift.models.openoil import OpenOil

# ── Default CLI configuration ──────────────────────────────────────────────────

CURRENTS_FILE = r"main\inputs\currents.nc"
WIND_FILE     = r"main\inputs\wind_cf.nc"
WAVES_FILE    = r"main\inputs\waves_cf.nc"   # optional; set None to skip

SEED_LON = -41.2593  # FPSO Peregrino mooring position (Campos Basin)
SEED_LAT = -23.3183

N_PARTICLES          = 1000
RADIUS_M             = 1
Z                    = 0

DURATION_HOURS       = 120  # 5 days — within the 7-day data window (Jan 1-7)
TIME_STEP_SEC        = 600
TIME_STEP_OUTPUT_SEC = 1800

USE_3D                  = False
DISABLE_VERTICAL_MIXING = True
MAX_SPEED               = 2.0

EXPORT_VARIABLES = ["lon", "lat", "status", "z"]

OIL_TYPE = None   # None → OpenOil default (GENERIC BUNKER C); pass ADIOS name to override


# ── Helpers ────────────────────────────────────────────────────────────────────

def ensure_dirs() -> tuple[Path, Path]:
    root    = Path(__file__).resolve().parents[1]
    outputs = root / "main" / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    return root, outputs


def resolve_oil_type(requested: Optional[str]) -> Optional[str]:
    """Return the ADIOS oil name to pass to OpenOil, or None for the default."""
    if not requested:
        return None
    try:
        from adios_db.scripting import get_all_oils
        names = {o.metadata.name.upper(): o.metadata.name for o in get_all_oils()}
        key = requested.upper()
        # Exact match first, then substring
        if key in names:
            return names[key]
        matches = [v for k, v in names.items() if key in k]
        if matches:
            return matches[0]
        print(f"[WARN] Oil '{requested}' not found in ADIOS — using default.")
        return None
    except Exception as e:
        print(f"[WARN] ADIOS lookup failed ({e}) — using default oil.")
        return None


def add_real_readers(
    o: OpenOil,
    currents_ref: Optional[str],
    wind_ref: Optional[str],
    waves_ref: Optional[str] = None,
) -> bool:
    from opendrift.readers import reader_netCDF_CF_generic

    readers = []
    for label, ref in [("correntes", currents_ref), ("vento", wind_ref), ("ondas", waves_ref)]:
        if not ref:
            continue
        path = Path(ref)
        if not path.exists():
            print(f"[SKIP] {label}: arquivo não encontrado ({ref})")
            continue
        try:
            r = reader_netCDF_CF_generic.Reader(str(ref))
            readers.append(r)
            print(f"\n[OK] Reader de {label} carregado: {ref}")
        except Exception as e:
            print(f"\n[ERRO] Falha ao carregar reader de {label} '{ref}':\n  {e}")

    if readers:
        o.add_reader(readers)
        return True
    return False


def add_smoke_test_reader(o: OpenOil) -> None:
    from opendrift.readers import reader_constant
    r = reader_constant.Reader({
        "x_sea_water_velocity": 0.3,
        "y_sea_water_velocity": 0.0,
        "x_wind": 5.0,
        "y_wind": 0.0,
        "sea_water_temperature": 20.0,
    })
    o.add_reader(r)
    print("\n[INFO] Rodando em modo SMOKE TEST (reader constante).")


# ── Main callable ──────────────────────────────────────────────────────────────

def run_simulation(
    seed_lon: float = SEED_LON,
    seed_lat: float = SEED_LAT,
    n_particles: int = N_PARTICLES,
    radius_m: float = RADIUS_M,
    start_time: Optional[datetime] = None,
    duration_hours: int = DURATION_HOURS,
    time_step_sec: int = TIME_STEP_SEC,
    time_step_output_sec: int = TIME_STEP_OUTPUT_SEC,
    oil_type: Optional[str] = OIL_TYPE,
    use_wind: bool = True,
    use_waves: bool = True,
    currents_file: Optional[str] = CURRENTS_FILE,
    wind_file: Optional[str] = WIND_FILE,
    waves_file: Optional[str] = WAVES_FILE,
    outfile: Optional[str] = None,
    figfile: Optional[str] = None,
    loglevel: int = 20,
) -> dict:
    """
    Run an OpenOil simulation and return paths to output files.

    Returns:
        {"netcdf": Path, "figure": Path, "start": datetime, "end": datetime}
    """
    _, outputs = ensure_dirs()

    if outfile is None:
        outfile = str(outputs / "openoil_run.nc")
    if figfile is None:
        figfile = str(outputs / "tracks.png")

    # Resolve oil type against ADIOS catalogue
    adios_oil = resolve_oil_type(oil_type)

    o = OpenOil(loglevel=loglevel, weathering_model="noaa")
    if adios_oil:
        o.set_config("seed:oil_type", adios_oil)
        print(f"[OIL] Using: {adios_oil}")

    o.max_speed = float(MAX_SPEED)

    if USE_3D and DISABLE_VERTICAL_MIXING:
        o.set_config("processes:vertical_mixing", False)

    # Stokes drift — must be explicitly disabled when not using waves;
    # OpenOil defaults to True and will demand wave variables otherwise.
    o.set_config("drift:stokes_drift", use_waves)
    if not use_waves:
        o.set_config("environment:fallback:sea_surface_wave_significant_height", 0)
        o.set_config("environment:fallback:sea_surface_wave_stokes_drift_x_velocity", 0)
        o.set_config("environment:fallback:sea_surface_wave_stokes_drift_y_velocity", 0)

    # Readers
    wind_ref  = wind_file  if use_wind  else None
    waves_ref = waves_file if use_waves else None
    used_real = add_real_readers(o, currents_file, wind_ref, waves_ref)
    if not used_real:
        add_smoke_test_reader(o)

    # OpenOil always needs wind variables for its weathering model (evaporation,
    # emulsification) even when wind-driven transport is disabled. Supply zero wind.
    if not use_wind:
        from opendrift.readers import reader_constant
        o.add_reader(reader_constant.Reader({"x_wind": 0.0, "y_wind": 0.0}))

    # Start time
    if start_time is None:
        start_time = datetime(2025, 1, 1, 0, 0, 0)

    o.seed_elements(
        lon=seed_lon,
        lat=seed_lat,
        time=start_time,
        number=n_particles,
        radius=radius_m,
        z=Z,
    )

    o.run(
        duration=timedelta(hours=duration_hours),
        time_step=timedelta(seconds=time_step_sec),
        time_step_output=timedelta(seconds=time_step_output_sec),
        outfile=outfile,
        export_variables=EXPORT_VARIABLES,
    )

    o.plot(filename=figfile)

    end_time = start_time + timedelta(hours=duration_hours)
    print(f"\n[OK] Finalizado.  NetCDF: {outfile}  |  Plot: {figfile}")

    return {
        "netcdf": Path(outfile),
        "figure": Path(figfile),
        "start":  start_time,
        "end":    end_time,
    }


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_simulation()
