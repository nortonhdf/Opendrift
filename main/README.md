# Campos Basin Oil Spill Dispersion

Oil-spill trajectory, fate and risk modelling for six Campos/Espírito Santo
Basin oil fields, built on **OpenDrift / OpenOil**. Everything in this `main/`
folder is the project layer on top of OpenDrift (which is used in-place from the
repo, not pip-installed).

```
main/
├── app.py                 Streamlit app — 4 tabs (scenarios, risk, beaching, custom run)
├── fields_config.py       The 6 oil fields (lon/lat, API, ADIOS oil type)
├── run_open_oil.py        Simulation engine: run_simulation() → OpenOil
├── rebuild_all.ps1        Convenience wrapper for the rebuild pipeline (Windows)
├── inputs/                Forcing data (NetCDF, CF-renamed)
│   ├── currents.nc        CMEMS surface currents, daily, full-year 2025
│   └── wind_cf.nc         ERA5 10 m wind, hourly, full-year 2025
├── outputs/
│   ├── scenarios/         48 precomputed runs (6 fields × 4 seasons × wind on/off)
│   ├── ensemble/          240 runs (6 × 4 × 10 start dates) — feeds risk & beaching
│   ├── risk_grids/        24 exposure/persistence probability grids
│   └── beaching/          24 coastal stranding grids
└── scripts/               Data download/prep + batch + analysis
```

## Environment

Conda env `opendrift` (miniforge). The app additionally needs `streamlit` and
`plotly` (now in `environment.yml`).

> **Windows / BLAS gotcha.** numpy linked against Intel **MKL** crashed natively
> on this machine (`Windows fatal exception 0xc06d007f`, even `np.dot`).
> Fixed by switching the BLAS backend to **OpenBLAS**:
> ```
> conda install -n opendrift -c conda-forge "blas=*=openblas" --force-reinstall
> ```
> `environment.yml` now pins `libblas=*=*openblas` so a fresh env avoids this.
> Also keep `adios_db < 1.2.7` (1.2.7 changed the API used here).

Python in this env: `C:\Users\nfreitas\AppData\Local\miniforge3\envs\opendrift\python.exe`

## Run the app

From the repo root, with the env active:

```
python -m streamlit run main/app.py
```

- **Pre-computed Scenarios** — browse the 48 runs; animation + density + oil budget.
- **Risk Maps** — ensemble exposure/persistence probability grids.
- **Beaching** — where/when oil reaches the coast (strongly seasonal, 0–89 %).
- **Custom Run** — live simulation; oil budget is computed on the fly.

## Rebuild the precomputed products

The committed `outputs/` were generated before recent fixes (per-field oil type,
oil-budget sidecars). Regenerate everything with one command:

```powershell
# Windows wrapper (handles env python + UTF-8 console):
.\main\rebuild_all.ps1                 # show the plan, change nothing
.\main\rebuild_all.ps1 --fresh         # rebuild ALL (~3.5–4 h)
.\main\rebuild_all.ps1 --fresh --only scenarios   # just the 48 scenarios (~47 min)
.\main\rebuild_all.ps1 --resume        # continue an interrupted rebuild
```

or call the orchestrator directly:

```
python main/scripts/rebuild_all.py [--fresh|--resume] [--only scenarios,ensemble,risk,beaching]
```

Stages, in order: **scenarios** (~47 min) → **ensemble** (~3 h) →
**risk** (~min) → **beaching** (~min). One scenario ≈ 1 min. Safe to Ctrl-C and
resume. `--fresh` deletes the scenario/ensemble manifests so runs are recomputed
and the NetCDFs overwritten in place.

## Data pipeline (only if refreshing the forcing data)

```
python main/scripts/download_cmems_currents.py   # → inputs/currents_raw.nc  (CMEMS creds)
python main/scripts/prep_cmems_currents.py        # → inputs/currents.nc      (CF names)
python main/scripts/download_era5_wind.py         # → inputs/wind_raw.nc       (CDS creds, ~/.cdsapirc)
python main/scripts/prep_era5_wind.py             # → inputs/wind.nc
python main/scripts/patch_wind_cf.py              # → inputs/wind_cf.nc        (CF attrs)
```

Wave (Stokes drift) data is **not** required: when waves are enabled the model
parameterises Stokes drift from the wind field
(`drift:use_tabularised_stokes_drift`).
