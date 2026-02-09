import rasterio
import pandas as pd
import numpy as np
from shapely.geometry import Point, box
import geopandas as gpd

tiff_path = "data/CSC/global_forest_csc_upsampled.tif"
df_pid=pd.read_csv("data/lookup/PID_location_all.csv")

# Process Raster data --------------------------------------------------------------------------------------------------------------
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

df_raster.rename(columns={"value": "csc"}, inplace=True)
df_raster = df_raster[df_raster["csc"] != -3.4028235e+38]

csc_points = gpd.GeoDataFrame(df_raster, geometry=gpd.points_from_xy(df_raster['x'], df_raster['y']), crs='EPSG:4326')

# Process PID location data --------------------------------------------------------------------------------------------------------------

pid_gdf = gpd.GeoDataFrame(df_pid, geometry=gpd.points_from_xy(df_pid['lon'], df_pid['lat']), crs='EPSG:4326')

# Create 500m x 500m polygons around points
half_size = 250  # half of 500m
pid_gdf['polygon'] = pid_gdf.geometry.apply(lambda p: box(
    p.x - half_size, p.y - half_size,
    p.x + half_size, p.y + half_size
))

# Keep only polygon geometry
gdf_polygons = pid_gdf.set_geometry('polygon')

# Make sure both are in the same CRS
csc_points = csc_points.to_crs(gdf_polygons.crs)

# Spatial join: points inside polygons
points_in_poly = gpd.sjoin(csc_points, gdf_polygons[['polygon']], how='inner', predicate='within')


# Match the poly_id into points_in_poly
points_in_poly = points_in_poly.merge(
    gdf_polygons[['PID']],
    left_on='index_right',
    right_index=True,
    how='left'
)

points_in_poly.to_csv("wip.csv")