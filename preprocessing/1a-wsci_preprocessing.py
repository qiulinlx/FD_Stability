
import rasterio
import pandas as pd
import geopandas as gpd
from pyproj.crs import CRS
from rasterio.sample import sample_gen


tiff_path = "data/WSCI/GEDI_L4C_WSCI.tif"
df_pid=pd.read_csv("data/lookup/PID_location_all.csv")

df_pid.drop_duplicates(inplace =True)


gdf1 = gpd.GeoDataFrame(df_pid, geometry=gpd.points_from_xy(df_pid.lon, df_pid.lat),
                         crs="EPSG:4326")

with rasterio.open(tiff_path) as src:
    correct_crs = CRS.from_epsg(6933)
    
    # Reproject your points to match
    gdf1 = gdf1.to_crs(correct_crs)
    
    coords = [(geom.x, geom.y) for geom in gdf1.geometry]
    sampled = list(sample_gen(src, coords))
    # transform = src.transform
    # nodata = src.nodata

gdf1['WSCI'] = [val[0] for val in sampled]

gdf=gdf1.dropna(subset=['WSCI'])

gdf.drop_duplicates(subset=['PID'], inplace=True)
gdf.drop(columns=['geometry'], inplace=True)

gdf.to_csv("data/processed/PID_location_WSCI.csv", index=False)