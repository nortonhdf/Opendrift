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

## State of outputs (2026-07-30) — REGENERATED ✓

All products were regenerated with the corrected code and new forcing
(wide box, SST, official coordinates): 48 scenarios + 240 ensemble + 24 risk
+ 24 beaching, **0 failures, 0 domain exits, 10 m³ in every run**, validated
by sweep (`docs/auditoria/REGENERACAO.md` has the full numbers and the two
incidents hit during regeneration — HDF5 inputs and a --fresh manifest bug,
both fixed and tested).

Key facts about the current generation (updated 2026-07-31):
- **Ensemble = 672 runs** (28 daily start dates 1..28 per month × 6 fields ×
  4 months). Convergence: IoU 0.78–0.82 for 14-vs-28 prob_any (much better
  than the old 10-member 0.63–0.73; residual uncertainty documented).
- **Beaching is near-zero and real**: only papa-terra_jan strands (3.41% of
  particles, 3 coastal cells) — a rare event only visible with daily
  sampling. The old "0–89%" was artifacts (domain exits + Frade's position
  ~118 km too close to shore).
- Scenario/ensemble PNGs are no longer generated (cartopy coastline 404) nor
  versioned; the app never used them.

## ML layer (main/ml/, since 2026-07-31)

Patch-transport surrogate (target a): metrics (Liu–Weisberg SS, IoU, Brier),
dataset builder (6-h patch transitions + local forcing; block key = field ×
season), baselines, HGB training with leave-one-block-out evaluation.
In-year LOBO (14,400 samples, 24 blocks): persistence 9.70 / advection 1.66
/ nearest 2.65 / HGB direct 1.40 / HGB residual 1.35 km per 6-h step.
**BLIND 2024 ROLLOUT (72 runs, 120 h): passive advection still wins**
(SS 0.93 / 10.9 km vs residual 0.91 / 15.9 km) — the learned correction is
training-year-specific (position-feature ablation ruled out as main cause).
v3 agenda: multi-year training (2022-2023 downloads) before any surrogate
claim; full record in docs/auditoria/CAMADA_IA.md §5b. Artefacts + hold-out
report in `main/outputs/ml/`. Rules: split by block only; 2024 = frozen
hold-out (reused ONLY for final evaluations); every model reported against
the baselines. scikit-learn is in environment.yml.

## Author decisions on record (2026-07-29)

See `docs/auditoria/PERGUNTAS_ABERTAS.md` for all 18. Highlights: outputs
stay in git; app is a final deliverable (UI polish after science+ML);
deploy platform flexible; waves = optional UI toggle; English standardised;
ML targets (a) patch-transport surrogate and (b) scenario summary stats;
2024 = hold-out year; metrics Liu–Weisberg SS + IoU + Brier.
