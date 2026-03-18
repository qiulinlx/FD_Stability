import pandas as pd
import geopandas as gpd
import json

from utils.utils import parse_geo_string

"""

This script loads two CSV files containing latitude and longitude points:
1) A CSV with plot IDs (PID) and coordinates.
2) A CSV with EVI values, coordinates, and dates.

It creates a buffer of a specified radius (in meters) around each point,
performs a spatial join to retain only overlapping points between the two datasets,
and saves the merged result as a CSV with the PID, EVI, and date.

"""

dfA = pd.read_csv("data/lookup/PID_location.csv" )
dfB = pd.read_csv("data/raw/Summer_EVI_MEanStd_2005_2022.csv")

dfB["lon"], dfB["lat"] = zip(*dfB[".geo"].map(parse_geo_string))

dfB.drop(columns=[".geo", 'system:index'], inplace=True)

# Convert df to GeoDataFrames
gdfA = gpd.GeoDataFrame(
    dfA,
    geometry=gpd.points_from_xy(dfA.lon, dfA.lat),
    crs="EPSG:4326"
)

gdfB = gpd.GeoDataFrame(
    dfB,
    geometry=gpd.points_from_xy(dfB.lon, dfB.lat),
    crs="EPSG:4326"
)

# Project to a metric CRS (meters) for buffering
gdfA = gdfA.to_crs("EPSG:3857")
gdfB = gdfB.to_crs("EPSG:3857")

BUFFER_DIST = 500                            

gdfA["buffer"] = gdfA.geometry.buffer(BUFFER_DIST)
gdfB["buffer"] = gdfB.geometry.buffer(BUFFER_DIST)

# Spatial join: keep only points that overlap
joined = gpd.sjoin(
    gdfA.set_geometry("buffer"),
    gdfB.set_geometry("buffer"),
    how="inner",
    predicate="intersects"
)

joined.columns = joined.columns.str.strip()
joined['date'] = pd.to_datetime(joined['date'])

df_wide = joined.pivot_table(
    index='PID',
    columns='date',
    values='EVI',
    aggfunc='mean'
)

df_wide.to_csv("data/EVI_PID.csv", index=True)