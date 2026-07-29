"""A corrupt-but-existing forcing file must abort immediately.

Observed in the 2026-07-29 rebuild: 'NetCDF: HDF error' on wind_cf.nc was
swallowed by add_real_readers, letting wind_on runs continue reader-less and
die later with a confusing seed-time ValueError (then segfault the batch).
"""

import pytest

from main.run_open_oil import run_simulation


def test_corrupt_forcing_file_raises_immediately(tmp_path):
    bad = tmp_path / "corrupt.nc"
    bad.write_bytes(b"this is not a netcdf file")
    with pytest.raises(RuntimeError, match="Failed to load correntes reader"):
        run_simulation(
            n_particles=5,
            duration_hours=1,
            use_wind=False,
            use_waves=False,
            currents_file=str(bad),
            wind_file=None,
            waves_file=None,
            outfile=str(tmp_path / "out.nc"),
            figfile=str(tmp_path / "out.png"),
            loglevel=50,
        )
