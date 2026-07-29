"""A run without loadable real forcing must fail loudly (audit finding #5).

The old behaviour silently substituted a constant 0.3 m/s eastward current,
which in a batch would generate plausible but physically fake scenarios.
"""

import pytest

from main.run_open_oil import run_simulation


def test_missing_forcing_raises_instead_of_smoke_fallback(tmp_path):
    with pytest.raises(RuntimeError, match="No real forcing reader"):
        run_simulation(
            n_particles=5,
            duration_hours=1,
            use_wind=False,
            use_waves=False,
            currents_file=str(tmp_path / "does_not_exist.nc"),
            wind_file=None,
            waves_file=None,
            outfile=str(tmp_path / "out.nc"),
            figfile=str(tmp_path / "out.png"),
            loglevel=50,
        )
