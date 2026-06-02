from pathlib import Path
import xarray as xr

INP = Path("main/inputs/waves_raw.nc")
OUT = Path("main/inputs/waves_cf.nc")

def main():
    ds = xr.open_dataset(INP)

    # Standardise time coordinate name
    if "valid_time" in ds.coords and "time" not in ds.coords:
        ds = ds.rename({"valid_time": "time"})

    # Map ERA5 wave variable names to CF standard_names OpenDrift expects
    rename_vars = {}
    cf_attrs = {}

    swh_candidates = [
        "significant_height_of_combined_wind_waves_and_swell", "swh", "shww"
    ]
    mwp_candidates = ["mean_wave_period", "mwp", "mpww"]
    mwd_candidates = ["mean_wave_direction", "mwd", "mdww"]

    for candidates, std_name, cf_name in [
        (swh_candidates, "sea_surface_wave_significant_height",            "sea_surface_wave_significant_height"),
        (mwp_candidates, "sea_surface_wave_mean_period_from_variance_spectral_density_first_frequency_moment", "sea_surface_wave_mean_period"),
        (mwd_candidates, "sea_surface_wave_from_direction",                "sea_surface_wave_from_direction"),
    ]:
        for c in candidates:
            if c in ds.data_vars:
                rename_vars[c] = cf_name
                cf_attrs[cf_name] = {"standard_name": std_name, "units": ds[c].attrs.get("units", "")}
                break

    if rename_vars:
        ds = ds.rename(rename_vars)

    for var, attrs in cf_attrs.items():
        if var in ds.data_vars:
            ds[var].attrs.update(attrs)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(OUT)
    print("OK ->", OUT.resolve())
    print("data_vars:", list(ds.data_vars))

if __name__ == "__main__":
    main()
