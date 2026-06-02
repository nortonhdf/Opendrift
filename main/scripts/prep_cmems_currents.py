from pathlib import Path
import xarray as xr

inp = Path("main/inputs/currents_raw.nc")
out = Path("main/inputs/currents.nc")

ds = xr.open_dataset(inp)

if "uo" not in ds.data_vars or "vo" not in ds.data_vars:
    raise ValueError(f"Não achei uo/vo. Variáveis encontradas: {list(ds.data_vars)}")

# Se tiver depth, pegar camada mais superficial (às vezes é depth=0; se der ruim trocamos)
if "depth" in ds.coords:
    ds = ds.isel(depth=0)

ds = ds.rename({"uo": "x_sea_water_velocity", "vo": "y_sea_water_velocity"})

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
