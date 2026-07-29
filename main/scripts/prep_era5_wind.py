"""Prepare ERA5 10 m wind for OpenDrift in ONE step.

Reads  main/inputs/wind_raw.nc  (u10/v10, from download_era5_wind.py)
writes main/inputs/wind_cf.nc   (x_wind/y_wind + CF standard_names)

This merges the old prep_era5_wind.py + patch_wind_cf.py pair (audit finding
#17: two halves of the same transformation with an orphan wind.nc between).
"""

from pathlib import Path

import xarray as xr

INP = Path("main/inputs/wind_raw.nc")
OUT = Path("main/inputs/wind_cf.nc")


def prep(ds: xr.Dataset) -> xr.Dataset:
    # ERA5 names the time coordinate valid_time on recent downloads
    if "valid_time" in ds.coords and "time" not in ds.coords:
        ds = ds.rename({"valid_time": "time"})

    if "u10" in ds.data_vars and "v10" in ds.data_vars:
        ds = ds.rename({"u10": "x_wind", "v10": "y_wind"})
    elif ("10m_u_component_of_wind" in ds.data_vars
          and "10m_v_component_of_wind" in ds.data_vars):
        ds = ds.rename({"10m_u_component_of_wind": "x_wind",
                        "10m_v_component_of_wind": "y_wind"})
    else:
        raise ValueError(
            f"u10/v10 not found. Variables present: {list(ds.data_vars)}")

    # CF attributes OpenDrift's generic reader keys on
    ds["x_wind"].attrs.update(standard_name="eastward_wind",
                              long_name="10m eastward wind")
    ds["y_wind"].attrs.update(standard_name="northward_wind",
                              long_name="10m northward wind")
    ds["x_wind"].attrs["units"] = ds["x_wind"].attrs.get("units", "m s-1")
    ds["y_wind"].attrs["units"] = ds["y_wind"].attrs.get("units", "m s-1")

    rename_coords = {}
    if "lon" in ds.coords and "longitude" not in ds.coords:
        rename_coords["lon"] = "longitude"
    if "lat" in ds.coords and "latitude" not in ds.coords:
        rename_coords["lat"] = "latitude"
    if rename_coords:
        ds = ds.rename(rename_coords)
    return ds


def main() -> None:
    out = prep(xr.open_dataset(INP))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_netcdf(OUT)
    print("OK ->", OUT.resolve())
    print("Vars:", list(out.data_vars))


if __name__ == "__main__":
    main()
