# Project — Oil spill dispersion, Campos Basin (OpenDrift)

> Portable project context (travels in git). Rewritten 2026-07-29 after the
> technical audit — the previous version of this file claimed "2D surface, no
> vertical mixing" and "beaching 0–89% correct", both wrong at the time.
> Current state, inventory and how to reproduce every published number:
> **`docs/auditoria/ESTADO_ATUAL.md`** (the other audit files are dated
> snapshots describing bugs that have since been fixed).

## Overview

Oil dispersion modelling for six **Campos Basin** fields (Peregrino, Marlim,
Roncador, Papa-Terra, Frade, Albacora) on **OpenDrift v1.14.7** used
*in-place* (NOT pip-installed, upstream untouched). All custom code in `main/`.
Research project; an ML layer (patch-transport surrogate + scenario summary
statistics) is planned on top of the regenerated products — see
`docs/auditoria/CAMADA_IA.md`.

**Components** — `app.py` (Streamlit, 5 tabs), `fields_config.py` (official
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
- Stokes drift: real ERA5 waves when `inputs/waves_cf.nc` exists (2025 is
  downloaded), otherwise parameterised from wind. Measured difference at D+7:
  5.9 km, against 27.7 km for waves-on vs waves-off. Every archived product
  was generated with `use_waves=False`, so waves change no published number.
  The download script requests the Stokes components (`ust`/`vst`): without
  them a wave file loads and changes nothing at all.
- No silent smoke fallback: missing forcing raises (pass `smoke_test=True`).
- **Status codes vary per output file** — decode via `main/status_utils.py`
  (`flag_meanings`); `active` is always 0; never assume `1 == stranded`.
- **Runs are deterministic**: OpenDrift seeds numpy's global RNG in its
  constructor (default 0) and the declared uncertainties draw from it, so a
  scenario re-run reproduces its archive bit-for-bit. Exposed as
  `run_simulation(random_seed=0)`; pass None for stochastic replicates.
  Note the seed must reach the OpenOil *constructor* — calling
  `np.random.seed()` before it is overwritten and silently does nothing.

## ⇒ START HERE if you are the next session (handover, 2026-09-02)

Read this block before touching anything. It is ordered: do 1 before 2.

**1. Run the test suite first. There is an unverified fix in the tree.**

```
python -m pytest main/tests -o addopts=""     # expect 129 + 4 new = 133
```

`block_mean()` in `main/ml/forecast.py` and
`main/tests/test_ml_climatology_nan.py` were written on the previous machine
**after** the defect was diagnosed but **before** the suite could be re-run
(the author stopped compute there). Treat both as unproven until that command
is green. If the new test fails, the test is more likely wrong than the fix —
the fix is a one-line `mean` → `nanmean` and the defect it addresses was
observed live, in `grid000` and `grid017` printing `nan` for the climatology
column at D+5 and D+7.

**What the defect was, because it matters for how you read any result:** a
fully beached slick has no drifting centroid, so its target is NaN at that
horizon. The climatology averaged its block with plain `mean`, so ONE such
scenario turned the entire block mean into NaN and the baseline emitted no
prediction at all. Every model then "beats" a baseline that never answered.
It could not fire while beaching was ~0 (the six fields); it fires
immediately on the seed grid, where coastal locations are deliberately
included and 11 of 480 scenarios are gone by D+7.

**2. Then re-run the evaluation that was aborted mid-flight.**

```
python -m main.ml.forecast --grid          # ~1 h: 40 folds + 6 reference folds
```

Its previous output was deleted on purpose — it was computed with the broken
climatology and would have looked like a win. `scenario_dataset_grid.npz` is
committed and valid (built from the complete 480-run archive), so the dataset
step can be skipped unless you change the builder.

**3. What that evaluation answers, and how to read it.** Leave-one-LOCATION-out
over 40 held-out sites, which is the attack on limitation #1 (`ESTADO_ATUAL.md`
§6.1: every spatial-generalisation claim so far rests on six neighbouring
fields, and six sites cannot separate "the model transfers in space" from
"the six sites are alike"). The report also carries a **reference arm**: the
same evaluation on the six fields restricted to the same year and the same
feature columns, so the two differ only in how many locations they cover.
Do not compare the grid numbers directly with §5e — that was 720 scenarios
over three years; this is 480 over one.

Two accounting details already instrumented, do not remove them: the count of
scenarios whose target exists per horizon (beached ones leave the evaluation),
and the dropped all-NaN feature (`water_depth_m`, unknown for arbitrary
points — it must reach the model as NaN, never as 0 m).

**4. Decisions waiting on the author, not on work:**
- The GitHub default branch is `main`, which holds a pre-audit version, so the
  repository landing page does not show this project. Options are: switch the
  default to `audit/revisao-completa`, merge via PR, or keep using the branch
  URL. Do not do any of these without being asked.
- Deploy platform. The technical objection is gone: measured, the app needs
  **156 MB** (products + ML forecast, current year) to **383 MB** (all five
  tabs, four years), not the 2.6 GB of the repo. `main/scripts/deploy_bundle.py
  --dry-run` prints the plan.

**5. Still open, in order of value:** more seeding locations *outside* the
Campos box (needs a wider forcing download — the current region is bounded by
what the forcing covers), validation against observed drift (needs external
drifter data the project does not have, and is the most serious declared
weakness, §6.5), and D+14 (needs 336-h runs; deferred by the author).

## Continuing on a different machine (written 2026-08-27)

Everything needed to carry on is in this repository **except three things
that cannot be**, so do these first:

1. **The environment.** `conda env create -f environment.yml` (miniforge).
   The file pins `libblas=*=*openblas` and that pin is load-bearing: with
   MKL, numpy crashes natively on Windows/py3.14 with exit `0xC06D007F` and
   no output. Verify with `conda list blas` before blaming any code.
2. **Credentials, which are deliberately not in git.** `~/.cdsapirc` for
   ERA5/CDS and `~/.copernicusmarine/` for CMEMS. Only the download scripts
   need them; everything already downloaded is committed, so the app, the
   ML layer and the tests run without them.
3. **Claude's local memory does not travel.** This file plus
   `docs/auditoria/ESTADO_ATUAL.md` are the portable context, which is why
   they are kept current in every commit.

**Work in flight when the machine changed:** the seed-location archive was
generating and is **231 of 480 runs** done and committed. It is resumable and
never recomputes what exists:

```
python -m main.ml.seedgrid generate 2025     # ~4 h for the remaining 249
python -m main.ml.scenario --grid 2025       # then build the dataset
python -m main.ml.forecast --grid            # leave-one-LOCATION-out
```

Measured rate on the previous machine: 60 s per run. A run interrupted
mid-write leaves a NetCDF with no manifest entry and is simply redone — the
resume check requires both. `scenario_dataset_grid.npz` and
`forecast_grid_report.json` were deleted on purpose before the handover:
they had been built from the 34-run pilot and would have looked
authoritative while describing 7 % of the archive.

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
Training data: 1,200 runs across 2022/2023/2025 -> 24,328 patch transitions
(`patch_dataset_multi.npz`). 2022 comes from the GLORYS `my` reanalysis (the
`anfc` analysis starts mid-2022 and CMEMS silently clips — `download()` now
verifies coverage).

**BLIND 2024 (72 runs, 120-h rollout) — the decisive table:**

| model | LW-SS | err 120 h | IoU |
|---|---|---|---|
| HGB direct (v1) | 0.90 | 16.0 km | 0.06 |
| HGB residual, 1 year (v2) | 0.91 | 15.9 km | 0.12 |
| HGB residual, 3 years (v3) | 0.94 | 10.7 km | 0.15 |
| single-point advection | 0.93 | 10.9 km | 0.16 |
| **midpoint advection (RK2)** | **0.97** | **4.6 km** | **0.35** |

v3 is statistically INDISTINGUISHABLE from advection (paired Wilcoxon
p=0.78/0.49/0.88; wins 39/72 runs). The diagnosis: the residual the model
was asked to learn is numerical path-integral error, not hidden physics —
switching to midpoint integration cuts the error 57% (p=8.6e-07) with zero
parameters. **RK2 is now the baseline any surrogate must beat (4.6 km, not
10.9 km).** v4 needs spatial features (current stencil, t and t+dt) and must
learn the residual over RK2. Full record: docs/auditoria/CAMADA_IA.md §5c.

Rules: split by block/year only; 2024 = frozen hold-out (final evaluations
only); every model reported against the baselines with a paired test.
scikit-learn is in environment.yml.

### v4 — scenario-level forecasting (the project's actual target)

`main/ml/scenario.py` + `main/ml/forecast.py`. Inputs known AT RELEASE ONLY:
location, oil API, depth, season, and antecedent current statistics over
3/7/30/90 days. Outputs: slick centroid displacement and spread at D+1, D+2,
D+3, D+5, D+7. No future forcing, so advection is not a competitor —
baselines are climatology, antecedent persistence, historical analogue and a
RidgeCV linear control. **Causality is enforced and tested**: features use
only data strictly before release; unavailable windows are NaN, never zero.

Archive: `training168_{2022,2023,2024,2025}`, 240 runs/year at **168 h**
(D+7 scope, author decision 2026-08-07). Training = 720 balanced scenarios
(2022+2023+2025); **blind 2024 = 240** scenarios (was 72 — the old holdout
lacked power). The 120-h archives stay untouched as the record behind §5a–5d.

**Headline — forecasting at a location never seen in training**
(leave-one-field-out, 720 held-out scenarios):

| horizon | HGB | ridge | climatology | HGB gain | HGB vs ridge |
|---|---|---|---|---|---|
| D+1 | **17.0 km** | 27.1 | 26.6 | +36% | p≈0 |
| D+3 | **46.9 km** | 69.4 | 62.6 | +25% | p≈0 |
| D+5 | **69.8 km** | 101.3 | 91.9 | +24% | p≈0 |
| D+7 | **82.5 km** | 138.4 | 113.8 | +28% | p≈0 |

HGB wins all 30 field × horizon cells. **The linear control is what makes
this claim sharp**: at a KNOWN location ridge ties HGB (p=0.07–0.74) and both
tie climatology past D+3, so nonlinearity buys nothing there; at a NEW
location ridge falls *below* climatology while HGB keeps 24–36%. The
nonlinearity is not fitting better — it is what transfers in space, which is
the deployment case.

Uncertainty: raw P10–P90 quantile boosters covered only 35–49% on the blind
year. Fixed with **split-conformal (CQR) calibrated on a held-out YEAR**
(fit 2022+2023, calibrate 2025) → 84–89% coverage, at the cost of ~3x wider
bands (±150 km at D+7 — that is the real uncertainty).

Still not possible: **D+14** (archive is 168 h). Spread is not predictable
from current features (MAE 1.28–1.38 km, every model). Full record:
docs/auditoria/CAMADA_IA.md §5e.

### v5 — footprint: which cells get oiled (since 2026-08-25)

`main/ml/footprint.py` (targets) + `main/ml/footprint_forecast.py` (models,
evaluation, exported product). No new simulation — same 168-h archives.

Two things were measured before anything was modelled, and they set the
design: the **instantaneous** slick is 1–2 cells on the 0.1° grid (RMS spread
0.35–1.25 km), so the target is the **swept** footprint (8 cells at D+1, 39
at D+7 — the same quantity the risk tab draws as `prob_any`); and everything
lives in a **release-relative km frame** (±501 km, 11.132 km cells, 0 of
960 runs left the frame), because a model reading absolute lon/lat cannot
transfer to a new site.

Every model returns a probability per cell. Metrics: Brier/BSS, IoU at an
operating point chosen on calibration data, and **capture area** — km² to
search, most likely cells first, to cover 80 % of the oiled cells. They
disagree on purpose: Brier asks if the number is right, capture area asks if
the ORDER is right.

**New location (leave-one-field-out, 720 held-out scenarios) — the
deployment case:**

| horizon | IoU corridor / clim | area corridor / clim | Δ area, p |
|---|---|---|---|
| D+1 | **0.610** / 0.553 | **1,239** / 1,611 km² | −124, 1e−15 |
| D+2 | **0.422** / 0.361 | **3,594** / 5,205 | −991, 1e−20 |
| D+3 | **0.330** / 0.288 | **7,745** / 9,728 | −1,363, 5e−15 |
| D+5 | 0.242 / 0.216 | **18,588** / 19,827 | −2,355, 6e−09 |
| D+7 | 0.203 / 0.182 | **29,927** / 33,893 | −3,718, 1e−10 |

"corridor" = isotonic-calibrated band around the path the v4 centroid model
predicts. It beats climatology on IoU AND area at all five horizons
(paired Wilcoxon). On the blind year — a location climatology has already
seen — it LOSES IoU (Δ −0.047 to −0.020) and still wins area: same split as
§5e, and the new-location column is the one that counts.

The two more ambitious models both fail at a new location, informatively:
the per-cell classifier learns lon/lat-conditioned shapes that do not
transfer; the 2-D plume kernel normalises its bins by the predicted
displacement, so at D+1 a bin is 1.5 km against an 11.1 km cell — it
estimates finer structure than the data has. Fix deferred, declared.

**The corridor is calibrated** (reliability measured through the exported
artefact on 2024: 0.03→0.03, 0.07→0.07, 0.13→0.12); its worse Brier is
sharpness, not miscalibration — deviations sit in bins holding <2 % of cells.

Full record: docs/auditoria/CAMADA_IA.md §5f.

### Serving it (since 2026-08-26) — §5g

Two artefacts, one per layer, both from ONE configuration (fit 2022+2023,
calibrate 2025 — structural, because a conformal correction measured on the
quantile models' own training data is a memory, not a correction):

| file | holds | build |
|---|---|---|
| `forecast_product.joblib` | centroid + quantile models + conformal corrections | `python -m main.ml.forecast --export` |
| `footprint_plume.joblib` | corridor isotonic, plume kernel, per-season climatology, operating points | `python -m main.ml.footprint_forecast --export` |

The footprint artefact does NOT carry its own centroid models — it consumes
the v4 product, so the corridor is calibrated against the very models the app
draws with (and 30 MB of duplicate regressors left git). Reliability came out
identical before and after that change, which is the check that they agreed.

`scenario.feature_row()` is the single source for the input vector: the
dataset builder and the live predictor call the same function, because a
feature assembled one way in training and another way in serving is the
quietest way to break a deployed model. The dataset was rebuilt after that
refactor and compared array by array to the previous one — identical.

`main/ml/predict.py` → `Predictor.forecast(lon, lat, api, water_depth_m,
when)` returns the track with its band per horizon plus the per-cell
probability field. It REFUSES to forecast outside the forcing box or in a
year with no forcing file, and reports (never hides) a short antecedent
window. App tab 5, "Forecast (ML)", is a thin layer over it — arbitrary
release point, ~1 s, no simulation.

**Do not draw the conformal band as a disc around the predicted point.** It
is calibrated on the DISTANCE travelled, so it is a range along the predicted
bearing; the spatial uncertainty is what the footprint field carries.

### Agenda v8 — two questions closed (2026-08-26), §5h

**The anisotropic plume is finished as a line of work.** Three hypotheses for
why it lost to the isotropic corridor at a new location; the first two were
tested and rejected — coarser bins changed nothing (LOFO D+1 IoU 0.464 →
0.469), and the bearing-error story is contradicted by the data (the plume
loses MOST where the bearing is best, r = −0.02). The third survives:
normalising the kernel axes by the PREDICTED displacement injects the model's
own error into the coordinate, so under-predicted runs smear probability
along the whole axis. Coordinates are now arc length and offset in km around
the polyline, which recovers most of the gap (D+7 IoU 0.162 → 0.193). It
still ties the corridor and never beats it, so the corridor remains the
shipped shape and the plume stays as the "does anisotropy pay?" control.
Reproduce: `python main/scripts/_plume_frames.py`.

**Spread is not forecastable, and now we know why** — the §5e limitation is
resolved, not merely restated. Three measurements
(`python main/scripts/_spread_decomposition.py`):
1. MAE/MAD ≈ 1.0–1.12 at every horizon: no model beats the best constant, so
   the "failure" is in the target, not the models.
2. Switching off the declared uncertainties collapses the slick width to
   **0.2 %** of it, and current-only ⊕ wind-only reproduce the total in
   quadrature (1.190 vs 1.141 km at D+7). The width is the diffusion constant
   this project declared, acting on a 1 m seed radius. Current dominates
   despite the smaller number: 0.05 m/s enters the drift directly, while the
   0.5 m/s of wind is multiplied by the ~3 % drag factor.
3. But the between-scenario variation is real: 32× the seed-to-seed scatter,
   and seed 0 reproduces the archive exactly. What sets it is strain along
   the FUTURE path — the same information barrier that keeps advection out
   of §5d.

`drift:current_uncertainty` and `drift:wind_uncertainty` are now parameters
of `run_simulation` (defaults unchanged at 0.05 / 0.5) so that claim is
testable rather than asserted. `predict_scenario` still returns `spread_km`,
with a docstring saying plainly that it must not be shown as a forecast; the
app does not display it.

## Author decisions on record (2026-07-29)

See `docs/auditoria/PERGUNTAS_ABERTAS.md` for all 18. Highlights: outputs
stay in git; app is a final deliverable (UI polish after science+ML);
deploy platform flexible; waves = optional UI toggle; English standardised;
ML targets (a) patch-transport surrogate and (b) scenario summary stats;
2024 = hold-out year; metrics Liu–Weisberg SS + IoU + Brier.
