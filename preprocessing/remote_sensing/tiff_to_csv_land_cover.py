import rasterio
import pandas as pd
import numpy as np

tiff_path = "data/land_cover/copernicus_land_cover.tif"

with rasterio.open(tiff_path) as src:
    band = src.read(1)  # first band
    transform = src.transform
    nodata = src.nodata


rows, cols = np.where(band != nodata)

xs, ys = rasterio.transform.xy(transform, rows, cols)

df_raster = pd.DataFrame({
    "x": xs,
    "y": ys,
    "value": band[rows, cols]
})

df_pid=pd.read_csv("data/lookup/PID_location_all.csv")

df_pid = df_pid.rename(columns={"lon": "x", "lat": "y"})

with rasterio.open(tiff_path) as src:
    coords = list(zip(df_pid["x"], df_pid["y"]))
    values = list(src.sample(coords))


df_pid["raster_value"] = [v[0] for v in values]

df_pid.rename(columns={"raster_value": "land_cover_value"}, inplace=True)

df_pid.to_csv("data/processed/PID_land_cover.csv", index=False)
