import pandas as pd
import geopandas as gpd
import json
"""

This script loads two CSV files containing latitude and longitude points:
1) A CSV with plot IDs (PID) and coordinates.
2) A CSV with EVI values, coordinates, and dates.

It creates a buffer of a specified radius (in meters) around each point,
performs a spatial join to retain only overlapping points between the two datasets,
and saves the merged result as a CSV with the PID, EVI, and date.

"""

def parse_geo_string(geo_str):
    # Fix doubled quotes
    fixed = geo_str.replace('""', '"').strip('"')
    
    # Convert to dict
    d = json.loads(fixed)

    # Extract lon / lat
    lon, lat = d["coordinates"]
    return lon, lat


BUFFER_DIST = 500                               # Buffer radius in meters


# 1) Load CSVs
dfA = pd.read_csv("data/lookup/PID_location.csv" )
dfB = pd.read_csv("data/raw/Summer_EVI_MEanStd_2005_2022.csv")

dfB["lon"], dfB["lat"] = zip(*dfB[".geo"].map(parse_geo_string))

dfB.drop(columns=[".geo", 'system:index'], inplace=True)

# 2) Convert to GeoDataFrames
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

# 3) Project to a metric CRS (meters) for buffering
gdfA = gdfA.to_crs("EPSG:3857")
gdfB = gdfB.to_crs("EPSG:3857")

# 4) Create buffers around points
gdfA["buffer"] = gdfA.geometry.buffer(BUFFER_DIST)
gdfB["buffer"] = gdfB.geometry.buffer(BUFFER_DIST)

# 5) Spatial join: keep only points that overlap
joined = gpd.sjoin(
    gdfA.set_geometry("buffer"),
    gdfB.set_geometry("buffer"),
    how="inner",
    predicate="intersects"
)

print(joined.head(30))

joined = joined.rename(columns={"PID_left": "PID", "lat_left": "lat", "lon_left": "lon"})
print(joined.columns)

# 6) Select relevant columns
merged_df = joined[["PID", 'EVI_mean', 'EVI_stdDev', "year"]].copy()

# 7) Save merged CSV
merged_df.to_csv("data/merged_EVI_PID1.csv", index=False)

print("Saved merged CSV")