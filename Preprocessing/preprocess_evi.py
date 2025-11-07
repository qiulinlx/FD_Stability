import pandas as pd
import geopandas as gpd

"""

This script loads two CSV files containing latitude and longitude points:
1) A CSV with plot IDs (PID) and coordinates.
2) A CSV with EVI values, coordinates, and dates.

It creates a buffer of a specified radius (in meters) around each point,
performs a spatial join to retain only overlapping points between the two datasets,
and saves the merged result as a CSV with the PID, EVI, and date.

"""

CSV_A = "data/PID_location.csv"                 # Points CSV with 'PID', 'lat', 'lon'
CSV_B = "data/EVI_2018_2023_cleaned.csv"       # EVI CSV with 'lat', 'lon', 'EVI', 'date'
BUFFER_DIST = 500                               # Buffer radius in meters
OUTPUT_CSV = "data/merged_EVI_PID.csv"
# ================================

# 1) Load CSVs
dfA = pd.read_csv(CSV_A)
dfB = pd.read_csv(CSV_B)

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

# 6) Select relevant columns
merged_df = joined[["PID", "EVI", "date"]].copy()

# 7) Save merged CSV
merged_df.to_csv(OUTPUT_CSV, index=False)

print(f"✅ Saved merged CSV: {OUTPUT_CSV}")