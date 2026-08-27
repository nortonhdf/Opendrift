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
├── app.py                 Streamlit app — 5 tabs (scenarios, risk, beaching, custom run, ML forecast)
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

A second layer forecasts the **footprint** — which 0.1° cells the oil
touches by D+n, as a probability per cell. At a location never seen in
training, a calibrated corridor around the predicted path beats season
climatology on IoU and on the area that must be searched (−23 % at D+1,
−12 % at D+7) at every horizon. Record: `docs/auditoria/CAMADA_IA.md` §5f.

```
python -m main.ml.multiyear generate 2024   # 168-h archive for a year (resumable)
python -m main.ml.scenario [--holdout]      # build the scenario dataset
python -m main.ml.forecast                  # evaluate -> outputs/ml/forecast_report.json
python -m main.ml.footprint [--holdout]     # build the footprint dataset (swept cells)
python -m main.ml.footprint_forecast        # evaluate -> outputs/ml/footprint_report.json
python -m main.ml.footprint_forecast --reliability   # export the product + check calibration
```

Both layers are exported as artefacts the app loads, so nothing is refitted at
run time. The **Forecast (ML)** tab takes an arbitrary release point, date, oil
API and depth, and answers in about a second — no simulation. It refuses to
forecast outside the forcing box or in a year with no forcing file.

```
python -m main.ml.forecast --export           # -> outputs/ml/forecast_product.joblib
python -m main.ml.footprint_forecast --export # -> outputs/ml/footprint_plume.joblib
python -m main.ml.predict --lat -22.4 --lon -40.1 --date 2024-03-10   # smoke check
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
  (`current_uncertainty` 0.05 m/s, `wind_uncertainty` 0.5 m/s, both
  parameters of `run_simulation`); horizontal diffusivity 0. Switching them
  off collapses the slick width to 0.2 % of it — the width of a slick in this
  model is a declared constant, not a prediction
  (`docs/auditoria/CAMADA_IA.md` §5h).
- SST from CMEMS `thetao` (merged into `currents.nc`); declared 24 °C fallback.
- Stokes drift: real ERA5 waves when `inputs/waves_cf.nc` is present
  (2025 is downloaded), otherwise parameterised from the wind. The two differ
  by 5.9 km at D+7 against a 27.7 km waves-on/waves-off effect, so the
  parameterisation is a fair stand-in — measured, `CAMADA_IA.md` §5h context.
  Every archived product was generated with `use_waves=False`.
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

## Deploy: what actually has to ship

The repository carries ~2.6 GB, but the app opens a small part of it at run
time. `main/scripts/deploy_bundle.py` assembles exactly that slice and
refuses to guess: `--dry-run` prints the plan and flags anything missing.

| bundle | size |
|---|---|
| all five tabs, four forecast years | **383 MB** |
| without the live-simulation tab, current year only | **156 MB** |

What is deliberately left behind: ~410 MB of `*_raw*.nc` (inputs to prep, never
opened by the app) and ~1.5 GB of run archives (`training168_*`, `ensemble/`,
`holdout_*` — evidence behind the published numbers, not app data). Only the
**Custom Run** tab needs the 86 MB wind field, because it is the only one that
runs a simulation; the forecast tab needs one currents file per year it must
answer for.

```
python main/scripts/deploy_bundle.py --dry-run
python main/scripts/deploy_bundle.py --out ../campos-deploy --tabs scenarios risk beaching forecast --years 2025
```

The bundle is data only — the code comes from a git clone and the environment
from `environment.yml`.

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
