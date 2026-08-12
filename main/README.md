# Campos Basin Oil Spill Dispersion

Oil-spill trajectory, fate and risk modelling for six **Campos Basin** oil
fields, built on **OpenDrift / OpenOil** (used in-place from this repo, not
pip-installed). Everything in `main/` is the project layer on top of OpenDrift.

> **Start at `docs/auditoria/ESTADO_ATUAL.md`** — current state, inventory,
> and every published number with the command that reproduces it. The other
> files in `docs/auditoria/` are dated snapshots of the 2026-07-29 audit and
> describe bugs that have since been fixed.

```
main/
├── app.py                 Streamlit app — 4 tabs (scenarios, risk, beaching, custom run)
├── fields_config.py       The 6 fields (official ANP/EPE polygon centres, API, ADIOS oil)
├── domain_config.py       Single source of truth: forcing box, grid, seasons
├── status_utils.py        Safe per-file decoding of OpenDrift status flags
├── run_open_oil.py        Simulation engine: run_simulation() → OpenOil
├── rebuild_all.ps1        Windows wrapper for the rebuild pipeline
├── ml/                    ML layer: patch surrogate (v1–v3) + scenario forecasting (v4)
├── inputs/                Forcing data (NetCDF, CF-renamed), 2022–2025
├── outputs/               scenarios/ ensemble/ risk_grids/ beaching/ training168_*/ ml/
├── scripts/               Download/prep + batch + aggregation + orchestrator
└── tests/                 pytest suite (run: python -m pytest main/tests -o addopts="")
```

## ML layer (`main/ml/`)

Forecasts a slick from what is known **at release time only** (location, oil
API, depth, season, antecedent current statistics) — no future forcing, so
numerical advection is not a competitor. Horizons D+1..D+7. Headline result:
at a location never seen in training, gradient boosting beats season
climatology by 24–36% at every horizon; a linear control on the same features
falls *below* climatology, so the nonlinearity is what transfers in space.
Uncertainty comes with a conformal-calibrated P10–P90 envelope. Full record
and the numbers: `docs/auditoria/CAMADA_IA.md` §5e.

```
python -m main.ml.multiyear generate 2024   # 168-h archive for a year (resumable)
python -m main.ml.scenario [--holdout]      # build the scenario dataset
python -m main.ml.forecast                  # evaluate -> outputs/ml/forecast_report.json
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

The committed `outputs/` were regenerated on 2026-07-30 with the corrected
code and the new forcing (0 failures, 0 domain exits). Runs are deterministic
(`random_seed=0` by default), so a rebuild reproduces them exactly.

```powershell
.\main\rebuild_all.ps1                 # show the plan, change nothing
.\main\rebuild_all.ps1 --fresh         # rebuild ALL (~3.5–4 h)
.\main\rebuild_all.ps1 --resume        # continue an interrupted rebuild
```

Stages: **scenarios** (48 runs) → **ensemble** (672 runs) → **risk** →
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
