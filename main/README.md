# Campos Basin Oil Spill Dispersion

Oil-spill trajectory, fate and risk modelling for six **Campos Basin** oil
fields, built on **OpenDrift / OpenOil** (used in-place from this repo, not
pip-installed). Everything in `main/` is the project layer on top of OpenDrift.

> A full technical audit (2026-07-29) lives in `docs/auditoria/` — read
> `DIAGNOSTICO.md` and `PLANO_DE_ACAO.md` before touching the science.

```
main/
├── app.py                 Streamlit app — 4 tabs (scenarios, risk, beaching, custom run)
├── fields_config.py       The 6 fields (official ANP/EPE polygon centres, API, ADIOS oil)
├── domain_config.py       Single source of truth: forcing box, grid, seasons
├── status_utils.py        Safe per-file decoding of OpenDrift status flags
├── run_open_oil.py        Simulation engine: run_simulation() → OpenOil
├── rebuild_all.ps1        Windows wrapper for the rebuild pipeline
├── inputs/                Forcing data (NetCDF, CF-renamed)
├── outputs/               scenarios/ ensemble/ risk_grids/ beaching/ (+ manifests)
├── scripts/               Download/prep + batch + aggregation + orchestrator
└── tests/                 pytest suite (run: python -m pytest main/tests -o addopts="")
```

## Fields

Peregrino, Marlim, Roncador, Papa-Terra, Frade, Albacora. Coordinates are the
centres of the official ANP production polygons (EPE Webmap layer 59,
consulted 2026-07-29). Oil type derives from API gravity via
`oil_type_for_api()`: <15° → GENERIC HEAVY CRUDE, 15–22° → MEDIUM, >22° → LIGHT.

## Model configuration (declared, tested in `main/tests/`)

- 2D surface transport (`drift:vertical_mixing=False` by default; enable with
  `run_simulation(vertical_mixing=True)`), NOAA weathering, RK4 advection.
- Reference spill: **10 m³ instantaneous** (`spill_m3` parameter).
- Spreading comes from the declared stochastic uncertainties
  (current 0.05 m/s, wind 0.5 m/s); horizontal diffusivity 0.
- SST from CMEMS `thetao` (merged into `currents.nc`); declared 24 °C fallback.
- Stokes drift: parameterised from wind when waves are enabled (no wave
  dataset needed); real ERA5 waves remain optional via `waves_cf.nc`.
- Forcing box lon −45..−36 / lat −27..−19 (`domain_config.py`);
  `run_simulation` warns if >2 % of particles exit the forcing coverage.
- **Status flags are per-file**: always decode with `status_utils` — never
  hard-code `1 == stranded`.

## Environment

Conda env `opendrift` (miniforge, Python 3.14). Recreate from
`environment.yml`. **Windows/BLAS gotcha**: numpy against Intel MKL crashes
natively — keep the OpenBLAS build (see the pin in `environment.yml`, and
force-reinstall `blas=*=openblas` if conda resolves MKL).

## Run the app

From the repo root, env active:

```
python -m streamlit run main/app.py
```

## Rebuild the precomputed products

⚠ The currently committed `outputs/` predate the audit fixes (status
decoding, 2D transport, SST, 10 m³, official coordinates, wide box) — the
beaching products in particular are invalid until regenerated.

```powershell
.\main\rebuild_all.ps1                 # show the plan, change nothing
.\main\rebuild_all.ps1 --fresh         # rebuild ALL (~3.5–4 h)
.\main\rebuild_all.ps1 --resume        # continue an interrupted rebuild
```

Stages: **scenarios** (48 runs) → **ensemble** (240 runs) → **risk** →
**beaching**. Safe to Ctrl-C and `--resume`.

## Refresh the forcing data (needs CMEMS + CDS credentials)

```
python main/scripts/download_cmems_currents.py   # currents_raw.nc + sst_raw.nc
python main/scripts/prep_cmems_currents.py        # → inputs/currents.nc (uo/vo + thetao, CF)
python main/scripts/download_era5_wind.py         # → inputs/wind_raw.nc (~/.cdsapirc)
python main/scripts/prep_era5_wind.py             # → inputs/wind_cf.nc (one step)
```

Pass a year to download other years (e.g. `... download_era5_wind.py 2024`);
2024 is reserved as the future ML hold-out year.

## Tests

```
python -m pytest main/tests -o addopts=""
```

(`-o addopts=""` neutralises upstream OpenDrift pytest options.)
