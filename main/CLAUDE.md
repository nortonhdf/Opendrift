# Project — Oil spill dispersion, Campos Basin (OpenDrift)

> Portable project context (travels in git). Rewritten 2026-07-29 after the
> technical audit — the previous version of this file claimed "2D surface, no
> vertical mixing" and "beaching 0–89% correct", both wrong at the time.
> Authoritative audit record: `docs/auditoria/` (root `CLAUDE.md` has the summary).

## Overview

Oil dispersion modelling for six **Campos Basin** fields (Peregrino, Marlim,
Roncador, Papa-Terra, Frade, Albacora) on **OpenDrift v1.14.7** used
*in-place* (NOT pip-installed, upstream untouched). All custom code in `main/`.
Research project; an ML layer (patch-transport surrogate + scenario summary
statistics) is planned on top of the regenerated products — see
`docs/auditoria/CAMADA_IA.md`.

**Components** — `app.py` (Streamlit, 4 tabs), `fields_config.py` (official
ANP/EPE coordinates + API→oil rule), `domain_config.py` (single source for
box/grid/seasons), `status_utils.py` (per-file status decoding),
`run_open_oil.py` (`run_simulation()`), `scripts/` (download/prep, 48
scenarios, 240-member ensemble, risk + beaching grids, `rebuild_all.py`
orchestrator), `tests/` (pytest).

## Declared model configuration (do not silently change)

- 2D surface transport: `vertical_mixing=False` default (parameter available).
- NOAA weathering; SST from CMEMS thetao, fallback 24 °C declared.
- Spill volume: `spill_m3=10` (reference scenario; type out of scope).
- RK4 advection; declared uncertainties current 0.05 / wind 0.5 m/s.
- Forcing box lon −45..−36 / lat −27..−19; >2 % domain-exit triggers a warning.
- No silent smoke fallback: missing forcing raises (pass `smoke_test=True`).
- **Status codes vary per output file** — decode via `main/status_utils.py`
  (`flag_meanings`); `active` is always 0; never assume `1 == stranded`.

## How to run

- Env conda `opendrift` (miniforge, Python 3.14). PATH python is NOT it.
- Always run from the repo root. App: `python -m streamlit run main/app.py`.
- Rebuild: `.\main\rebuild_all.ps1 [--fresh|--resume] [--only stages]`
  (finds the env python itself). Direct: `python main/scripts/rebuild_all.py`.
- Tests: `python -m pytest main/tests -o addopts=""`.

## ⚠ Environment gotchas (Windows)

1. **BLAS/MKL**: numpy linked against MKL crashes natively (exit 0xC06D007F,
   empty output). Keep OpenBLAS: `conda list blas` must show openblas;
   otherwise `conda install -n opendrift "blas=*=openblas" --force-reinstall`.
2. **Batch plotting**: `o.plot()` downloads coastline shapefiles (may fail
   offline) and leaks figures — already wrapped in try/except + close("all").
3. **Subprocess logging**: call the env's `python.exe` directly, not
   `powershell -File`; note `*>` redirects produce UTF-16 logs.

## State of outputs (2026-07-29)

The committed `outputs/` (48 scenarios + 240 ensemble + 24 risk + 24
beaching) were generated BEFORE the audit fixes. Known defects of that
generation: beaching grids counted domain exits as strandings (real beaching
only at Frade), vertical mixing was on, weathering ran at 10 °C, spill was
1 m³, old too-small box lost ~16 % of particles, and 4 of 6 field positions
were off by 28–118 km. **Regeneration pending**: new forcing download
(CMEMS currents+SST, ERA5 wind; wide box; year 2025 + 2024 as ML hold-out)
followed by `rebuild_all.ps1 --fresh` (~4 h). Until then treat all committed
products as superseded baselines, and keep them for comparison.

## Author decisions on record (2026-07-29)

See `docs/auditoria/PERGUNTAS_ABERTAS.md` for all 18. Highlights: outputs
stay in git; app is a final deliverable (UI polish after science+ML);
deploy platform flexible; waves = optional UI toggle; English standardised;
ML targets (a) patch-transport surrogate and (b) scenario summary stats;
2024 = hold-out year; metrics Liu–Weisberg SS + IoU + Brier.
