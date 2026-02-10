import rasterio
import pandas as pd
import numpy as np
import geopandas as gpd

tiff_path = "data/CSC/stitched_csc_PID.tif"
df_pid=pd.read_csv("data/lookup/PID_location_all.csv")

df_pid.drop_duplicated(inplace =True)

with rasterio.open(tiff_path) as src:
    csc = src.read(1)  # first band
    loc = src.read(2)
    transform = src.transform
    nodata = src.nodata


# Create mask of valid data
if nodata is not None:
    mask = csc != nodata
else:
    mask = np.ones_like(csc, dtype=bool)

# Get row and column indices of valid data
rows, cols = np.where(mask)

# Convert row/col to x/y coordinates
xs, ys = rasterio.transform.xy(transform, rows, cols)

# xs, ys are lists → convert to numpy arrays
xs = np.array(xs)
ys = np.array(ys)

# Build DataFrame
df_raster = pd.DataFrame({
    "lat": ys,
    "lon": xs,
    "csc": csc[rows, cols],
    "PID": loc[rows, cols].astype(np.int64)})


df_raster = df_raster[df_raster["csc"] != 0]
df_raster=df_raster[df_raster["PID"] != 0]


gdf1 = gpd.GeoDataFrame(df_pid, geometry=gpd.points_from_xy(df_pid.lon, df_pid.lat),
                         crs="EPSG:4326")
gdf2 = gpd.GeoDataFrame(df_raster, geometry=gpd.points_from_xy(df_raster.lon, df_raster.lat),
                         crs="EPSG:4326")

# Convert to a projected CRS in meters for distance
gdf1 = gdf1.to_crs(epsg=3395)
gdf2 = gdf2.to_crs(epsg=3395)

#Nearest Join in 500m since we have 500m resolution
df_joined = gpd.sjoin_nearest(gdf1, gdf2, max_distance=500, distance_col="dist")

df_joined.drop(columns=['Unnamed: 0', 
                        'PID_right', 
                        'lat_left', 
                        'lon_left', 
                        'geometry', 
                        'index_right',
                        'lat_right', 
                        'lon_right', 
                        'dist'], inplace=True)

df_joined.rename(columns={"PID_left": "PID"})
df_joined.to_csv('data/processed/PID_csc_upsampled.csv')
