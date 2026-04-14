import pandas as pd
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin

# ---------------------------------------------------
# Settings
# ---------------------------------------------------
CSV_PATH = "data/lookup/PID_location_all.csv"
OUT_TIF = "PID_map.tif"
RESOLUTION = 0.001     # degrees (~1 km) adjust as needed
CRS = "EPSG:4326"     # WGS84 (same as lon/lat)
# ---------------------------------------------------

# Load CSV
df = pd.read_csv(CSV_PATH)

df.dropna(subset=["PID", "lon", "lat"], inplace=True)
df.drop_duplicates(subset=["PID"], inplace=True)

# encode PID
df["PID_code"] = df["PID"].astype("category").cat.codes

# save lookup table
df[["PID", "PID_code"]].drop_duplicates().to_csv("PID_lookup.csv", index=False)

# Convert to GeoDataFrame
gdf = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df["lon"], df["lat"]),
    crs=CRS
)

# Determine raster bounds
minx, miny, maxx, maxy = gdf.total_bounds

# Raster size
width  = int((maxx - minx) / RESOLUTION) + 1
height = int((maxy - miny) / RESOLUTION) + 1

# Create empty array
data = np.full((height, width), np.nan, dtype="float32")

# Transform definition
transform = from_origin(minx, maxy, RESOLUTION, RESOLUTION)

# Fill raster using nearest pixel
for _, row in gdf.iterrows():
    col = int((row["lon"] - minx) / RESOLUTION)
    row_idx = int((maxy - row["lat"]) / RESOLUTION)

    if 0 <= row_idx < height and 0 <= col < width:
        data[row_idx, col] = row["PID"]

# Write GeoTIFF
with rasterio.open(
    OUT_TIF,
    "w",
    driver="GTiff",
    height=height,
    width=width,
    count=1,
    dtype="float32",
    crs=CRS,
    transform=transform,
    nodata=np.nan
) as dst:
    dst.write(data, 1)

print(f"✅ Saved GeoTIFF: {OUT_TIF}")
