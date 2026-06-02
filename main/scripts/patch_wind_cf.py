from pathlib import Path
import xarray as xr

INP = Path("main/inputs/wind.nc")          # seu wind já gerado
OUT = Path("main/inputs/wind_cf.nc")       # saída nova (pra não destruir o original)

def main():
    ds = xr.open_dataset(INP)

    # 1) Padronizar tempo (opcional, mas ajuda)
    if "valid_time" in ds.coords and "time" not in ds.coords:
        ds = ds.rename({"valid_time": "time"})

    # 2) Garantir attrs CF para o OpenDrift reconhecer como vento
    ds["x_wind"].attrs["standard_name"] = "eastward_wind"
    ds["y_wind"].attrs["standard_name"] = "northward_wind"

    # units: ERA5 normalmente é m/s; aqui garantimos
    ds["x_wind"].attrs["units"] = ds["x_wind"].attrs.get("units", "m s-1")
    ds["y_wind"].attrs["units"] = ds["y_wind"].attrs.get("units", "m s-1")

    # (opcional) long_name ajuda debug
    ds["x_wind"].attrs["long_name"] = "10m eastward wind"
    ds["y_wind"].attrs["long_name"] = "10m northward wind"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(OUT)
    print("OK ->", OUT.resolve())
    print("data_vars:", list(ds.data_vars))
    print("coords:", list(ds.coords))

if __name__ == "__main__":
    main()
