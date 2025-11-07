import pandas as pd
import geopandas as gpd
from shapely import wkt
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
import numpy as np


# User inputs

CSV_FILE = "data/merged_EVI_PID.csv"      # your CSV
POLYGON_COL = "buffer"       # column with POLYGON WKT strings
OUTPUT_TIF = "polygons_raster.tif"
RASTER_RES = 30                # resolution in CRS units (meters if projected)
VALUE_FIELD = None             # column to burn into raster (None = 1)
CRS = "EPSG:3857"              # projected CRS
# -----------------------------

# 1) Load CSV
df = pd.read_csv(CSV_FILE)

# 2) Convert WKT string to geometry
df["geometry"] = df[POLYGON_COL].apply(wkt.loads)

# 3) Convert to GeoDataFrame
gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=CRS)

# 4) Ensure projected CRS
if gdf.crs != CRS:
    gdf = gdf.to_crs(CRS)

# 5) Raster bounds
minx, miny, maxx, maxy = gdf.total_bounds
width  = int((maxx - minx) / RASTER_RES) + 1
height = int((maxy - miny) / RASTER_RES) + 1

# 6) Transform
transform = from_origin(minx, maxy, RASTER_RES, RASTER_RES)

# 7) Prepare shapes
if VALUE_FIELD:
    shapes = ((geom, val) for geom, val in zip(gdf.geometry, gdf[VALUE_FIELD]))
else:
    shapes = ((geom, 1) for geom in gdf.geometry)

# 8) Rasterize
raster = rasterize(
    shapes=shapes,
    out_shape=(height, width),
    transform=transform,
    fill=0,
    dtype="float32"
)

# 9) Save GeoTIFF
with rasterio.open(
    OUTPUT_TIF,
    "w",
    driver="GTiff",
    height=height,
    width=width,
    count=1,
    dtype="float32",
    crs=CRS,
    transform=transform,
    nodata=0
) as dst:
    dst.write(raster, 1)

print(f"✅ Saved GeoTIFF: {OUTPUT_TIF}")
