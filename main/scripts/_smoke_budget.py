"""Smoke test: waves/Stokes parameterisation + oil budget sidecar."""
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from main.run_open_oil import run_simulation, budget_path_for

OUT = str(ROOT / "main" / "outputs" / "_smoke_budget.nc")

res = run_simulation(
    seed_lon=-41.2593, seed_lat=-23.3183,
    n_particles=50, start_time=datetime(2025, 1, 15),
    duration_hours=24, oil_type="GENERIC HEAVY CRUDE",
    use_wind=True, use_waves=True,
    outfile=OUT, figfile=str(ROOT / "main" / "outputs" / "_smoke_budget.png"),
    loglevel=30,
)

bp = budget_path_for(OUT)
print("\n=== VALIDATION ===")
print("budget sidecar exists:", bp.exists(), "->", bp.name)
d = np.load(bp)
print("keys:", list(d.files))
print("n_time:", len(d["hours"]), "| hours[-1]:", float(d["hours"][-1]))
for k in ["mass_surface", "mass_evaporated", "mass_dispersed", "mass_stranded", "mass_total"]:
    print(f"  {k:16s} t0={float(d[k][0]):.2f}  tEnd={float(d[k][-1]):.2f} kg")
print("oil_density:", float(np.asarray(d["oil_density"]).ravel()[0]))

# Confirm mass vars made it into the NetCDF too
import xarray as xr
ds = xr.open_dataset(OUT)
print("nc data_vars:", list(ds.data_vars))
ds.close()
print("SMOKE OK")
