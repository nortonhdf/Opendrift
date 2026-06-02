import sys
sys.path.insert(0, '.')
from datetime import datetime
from main.run_open_oil import run_simulation

run_simulation(
    seed_lon=-41.2593, seed_lat=-23.3183,
    n_particles=10, start_time=datetime(2025, 1, 15),
    duration_hours=6, oil_type="GENERIC HEAVY CRUDE",
    use_wind=False, use_waves=False,
    outfile="main/outputs/test_wind_off.nc",
    figfile="main/outputs/test_wind_off.png",
    loglevel=50,
)
print("SUCCESS")
