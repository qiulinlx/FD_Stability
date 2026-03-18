from shapely.geometry import shape
import geopandas as gpd
import json
import pandas as pd

npp_df=pd.read_csv("npp_10_years.csv")
df_pid=pd.read_csv("data/lookup/PID_location_all.csv")
df_pid.drop_duplicates(inplace =True)


gdf1 = gpd.GeoDataFrame(df_pid, geometry=gpd.points_from_xy(df_pid.lon, df_pid.lat),
                         crs="EPSG:4326")
                         
npp_df[".geo"] = npp_df[".geo"].apply(json.loads)
npp_df[".geo"] = npp_df[".geo"].apply(shape)

gdf2 = gpd.GeoDataFrame(npp_df, geometry=".geo", crs="EPSG:4326")

gdf2.drop(columns=['system:index', 'b1'], inplace=True)
# Convert to a projected CRS in meters for distance
gdf1 = gdf1.to_crs(epsg=3395)
gdf2 = gdf2.to_crs(epsg=3395)
df_joined = gpd.sjoin_nearest(gdf1, gdf2, max_distance=250, distance_col="dist")
df_joined.drop(columns=['Unnamed: 0', 'geometry', 'lat', 'lon', 'geometry', 'index_right', 'dist'], inplace=True)

df_joined.to_csv('data/processed/PID_npp.csv')