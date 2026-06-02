from pathlib import Path
import xarray as xr

inp = Path("main/inputs/wind_raw.nc")
out = Path("main/inputs/wind.nc")

ds = xr.open_dataset(inp)

# ERA5 normalmente vem como u10/v10
if "u10" in ds.data_vars and "v10" in ds.data_vars:
    ds = ds.rename({"u10": "x_wind", "v10": "y_wind"})
elif "10m_u_component_of_wind" in ds.data_vars and "10m_v_component_of_wind" in ds.data_vars:
    ds = ds.rename({"10m_u_component_of_wind": "x_wind", "10m_v_component_of_wind": "y_wind"})
else:
    raise ValueError(f"Não achei u10/v10. Variáveis encontradas: {list(ds.data_vars)}")

# Garantir coord names comuns
rename_coords = {}
if "lon" in ds.coords and "longitude" not in ds.coords:
    rename_coords["lon"] = "longitude"
if "lat" in ds.coords and "latitude" not in ds.coords:
    rename_coords["lat"] = "latitude"
if rename_coords:
    ds = ds.rename(rename_coords)

out.parent.mkdir(parents=True, exist_ok=True)
ds.to_netcdf(out)

print("OK ->", out.resolve())
print("Vars:", list(ds.data_vars))
