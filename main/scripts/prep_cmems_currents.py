"""Prepare CMEMS surface currents (+ SST when available) for OpenDrift.

Reads  main/inputs/currents_raw.nc  (uo/vo, from download_cmems_currents.py)
and    main/inputs/sst_raw.nc       (thetao, optional — same script)
writes main/inputs/currents.nc      with CF names OpenDrift's generic reader
recognises: x/y_sea_water_velocity and sea_water_temperature.

SST matters: without it OpenOil weathers the oil at the 10 °C fallback
instead of Campos Basin's ~24 °C (audit finding grave #3).
"""

from pathlib import Path

import xarray as xr

CUR_RAW = Path("main/inputs/currents_raw.nc")
SST_RAW = Path("main/inputs/sst_raw.nc")
OUT     = Path("main/inputs/currents.nc")


def _surface(ds: xr.Dataset) -> xr.Dataset:
    """Select the shallowest layer when a depth dimension is present."""
    if "depth" in ds.coords and "depth" in ds.dims:
        ds = ds.isel(depth=0)
    return ds


def prep(ds_cur: xr.Dataset, ds_sst: xr.Dataset | None = None) -> xr.Dataset:
    if "uo" not in ds_cur.data_vars or "vo" not in ds_cur.data_vars:
        raise ValueError(
            f"uo/vo not found. Variables present: {list(ds_cur.data_vars)}")

    ds = _surface(ds_cur).rename(
        {"uo": "x_sea_water_velocity", "vo": "y_sea_water_velocity"})

    if ds_sst is not None:
        if "thetao" not in ds_sst.data_vars:
            raise ValueError(
                f"thetao not found in SST file. Variables: {list(ds_sst.data_vars)}")
        sst = _surface(ds_sst)["thetao"]
        ds["sea_water_temperature"] = sst
        ds["sea_water_temperature"].attrs.update(
            standard_name="sea_water_temperature",
            units=sst.attrs.get("units", "degrees_C"),
        )

    rename_coords = {}
    if "lon" in ds.coords and "longitude" not in ds.coords:
        rename_coords["lon"] = "longitude"
    if "lat" in ds.coords and "latitude" not in ds.coords:
        rename_coords["lat"] = "latitude"
    if rename_coords:
        ds = ds.rename(rename_coords)
    return ds


def main() -> None:
    ds_cur = xr.open_dataset(CUR_RAW)
    ds_sst = xr.open_dataset(SST_RAW) if SST_RAW.exists() else None
    if ds_sst is None:
        print(f"[WARN] {SST_RAW} not found — currents.nc will carry no SST and "
              "weathering will use the declared 24 degC fallback. Run "
              "download_cmems_currents.py to fetch thetao.")

    out = prep(ds_cur, ds_sst).load()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    # NETCDF3: no HDF5 layer. Repeated same-process opens of an HDF5 file
    # corrupted netCDF state and segfaulted the 288-run rebuild (2026-07-29).
    out.to_netcdf(OUT, format="NETCDF3_64BIT",
                  encoding={c: {"_FillValue": None} for c in out.coords})
    print("OK ->", OUT.resolve())
    print("Vars:", list(out.data_vars))


if __name__ == "__main__":
    main()
