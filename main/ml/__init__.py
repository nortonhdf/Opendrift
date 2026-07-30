"""ML layer for the Campos Basin dispersion project.

Targets (author decisions, docs/auditoria/CAMADA_IA.md):
  (a) patch-transport surrogate  — learn particle-patch displacement/spread
  (b) scenario summary statistics — needs the LHS sampling plan (future)

Ground rules baked into this package:
  - Split by block (field x month) or by whole month; NEVER by particle or
    timestep — ensemble members of the same month share lagged forcing.
  - Year 2024 is the frozen hold-out (main/inputs/*_2024.nc) and must never
    enter training.
  - Every model is reported against the baselines in baselines.py using the
    metrics in metrics.py (Liu-Weisberg SS, IoU, Brier).
"""
