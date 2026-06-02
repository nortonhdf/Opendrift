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

# Resolve input paths against the repo root so the runner works regardless of
# the current working directory.
_ROOT = Path(__file__).resolve().parents[1]
CURRENTS_FILE = str(_ROOT / "main" / "inputs" / "currents.nc")
WIND_FILE     = str(_ROOT / "main" / "inputs" / "wind_cf.nc")
WAVES_FILE    = str(_ROOT / "main" / "inputs" / "waves_cf.nc")   # optional; set None to skip

SEED_LON = -41.2593  # FPSO Peregrino mooring position (Campos Basin)
SEED_LAT = -23.3183

N_PARTICLES          = 1000
RADIUS_M             = 1
Z                    = 0

DURATION_HOURS       = 120  # 5 days — well within the full-year (2025) data window
TIME_STEP_SEC        = 600
TIME_STEP_OUTPUT_SEC = 1800

USE_3D                  = False
DISABLE_VERTICAL_MIXING = True
MAX_SPEED               = 2.0

# Trajectory variables plus the oil-mass variables needed to reconstruct the
# weathering budget (get_oil_budget needs status, z, mass_* and density;
# water_fraction/viscosity feed the emulsion panel).
EXPORT_VARIABLES = [
    "lon", "lat", "status", "z",
    "mass_oil", "mass_evaporated", "mass_dispersed", "mass_biodegraded",
    "density", "water_fraction", "viscosity",
]

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
        # OpenDrift bundles the ADIOS oil catalogue; query it via its own helper
        # (the adios_db.scripting.get_all_oils API was removed in newer versions).
        from opendrift.models.openoil import adios
        names = {n.upper(): n for n in adios.get_oil_names()}
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


def budget_path_for(outfile: str) -> Path:
    """Sidecar path holding the weathering budget time series for a run."""
    p = Path(outfile)
    return p.with_name(p.stem + "_budget.npz")


def save_oil_budget(o: OpenOil, outfile: str) -> Optional[Path]:
    """Aggregate the NOAA weathering budget into a compact .npz sidecar.

    Stores per-output-timestep oil mass (kg) split into surface / submerged /
    stranded / evaporated / dispersed / biodegraded, plus total and oil density.
    Much smaller than the trajectory NetCDF and trivial for the app to plot.
    """
    import numpy as np

    try:
        b = o.get_oil_budget()
    except Exception as e:
        print(f"[WARN] oil budget unavailable ({e}) — skipping sidecar.")
        return None
    if b is None:  # e.g. backwards simulation
        return None

    hours = (o.result.time - o.result.time[0]).dt.total_seconds().values / 3600.0
    out = budget_path_for(outfile)
    np.savez_compressed(
        out,
        hours          = np.asarray(hours, dtype=np.float32),
        mass_surface   = np.asarray(b["mass_surface"],   dtype=np.float32),
        mass_submerged = np.asarray(b["mass_submerged"], dtype=np.float32),
        mass_stranded  = np.asarray(b["mass_stranded"],  dtype=np.float32),
        mass_evaporated= np.asarray(b["mass_evaporated"],dtype=np.float32),
        mass_dispersed = np.asarray(b["mass_dispersed"], dtype=np.float32),
        mass_biodegraded=np.asarray(b["mass_biodegraded"],dtype=np.float32),
        mass_total     = np.asarray(b["mass_total"],     dtype=np.float32),
        oil_density    = np.asarray(b["oil_density"],    dtype=np.float32),
    )
    return out


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
    if use_waves:
        # Derive Stokes drift and significant wave height from the wind field
        # via OpenDrift's tabularised parameterisation — no separate wave
        # dataset (ERA5) is required. If a real waves file is supplied below,
        # its non-zero values take precedence and the parameterisation is skipped.
        o.set_config("drift:use_tabularised_stokes_drift", True)
        o.set_config("drift:tabularised_stokes_drift_fetch", "25000")
        if not use_wind:
            print("[WARN] use_waves=True but use_wind=False — parameterised "
                  "Stokes drift needs wind and will be ~0.")
    else:
        o.set_config("environment:fallback:sea_surface_wave_significant_height", 0)
        o.set_config("environment:fallback:sea_surface_wave_stokes_drift_x_velocity", 0)
        o.set_config("environment:fallback:sea_surface_wave_stokes_drift_y_velocity", 0)

    # Readers — a real waves file is optional; absence is fine when use_waves
    # relies on the wind-based parameterisation above.
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

    # Weathering budget sidecar (small .npz next to the NetCDF)
    budget_file = save_oil_budget(o, outfile)

    end_time = start_time + timedelta(hours=duration_hours)
    print(f"\n[OK] Finalizado.  NetCDF: {outfile}  |  Plot: {figfile}"
          + (f"  |  Budget: {budget_file}" if budget_file else ""))

    return {
        "netcdf": Path(outfile),
        "figure": Path(figfile),
        "budget": budget_file,
        "start":  start_time,
        "end":    end_time,
    }


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_simulation()
