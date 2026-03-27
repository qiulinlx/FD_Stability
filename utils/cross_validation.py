import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import box


def process_ecoregion(path: str):

    ecoregions = gpd.read_file(path)
    ecoregions = ecoregions.to_crs("EPSG:4326")

    bbox = (-170, 5, -50, 85)
    crop_box = box(*bbox)

    ecoregions = ecoregions[ecoregions.intersects(crop_box)].copy()
    return ecoregions


def assign_spatial_groups(df, grid_size=1.0):
    df = df.copy()

    df["lon_bin"] = (df["lon"] // grid_size) * grid_size
    df["lat_bin"] = (df["lat"] // grid_size) * grid_size
    
    df["spatial_group"] = (
        df["biome"].astype(str) + "_" +
        df["lon_bin"].astype(str) + "_" +
        df["lat_bin"].astype(str)
    )
    
    return df


def gridded_cross_validation(df, test_size, batch_size):
    grouped_df = (
        df
        .groupby("biome", group_keys=False)
        .apply(assign_spatial_groups, grid_size=5.0)
    )

    # unique groups
    groups = grouped_df["spatial_group"].unique()
    grouped_df = grouped_df.groupby("spatial_group").filter(lambda x: len(x) >= batch_size)


    # number to sample
    n_select = int(len(groups) * test_size)

    selected_groups = np.random.choice(groups, size=n_select, replace=False)
    test = grouped_df[grouped_df["spatial_group"].isin(selected_groups)]
    train = grouped_df[~grouped_df["spatial_group"].isin(selected_groups)]
    return train, test

def ecoregion_cross_validation(gdf, ecoregion, test_size, batch_size):

    grouped_df = gpd.sjoin(
        gdf,
        ecoregion,
        how="left",          # keep all PIDs
        predicate="within"   # point inside polygon
    )

    grouped_df.dropna(subset=['ECO_ID'], inplace=True)

    # unique groups
    groups = grouped_df["ECO_ID"].unique()
    print(groups)
    grouped_df = grouped_df.groupby("spatial_group").filter(lambda x: len(x) >= batch_size)
    # number to sample
    n_select = int(len(groups) * test_size)
    
    selected_groups = np.random.choice(groups, size=n_select, replace=False)
    print(selected_groups)
    test = grouped_df[grouped_df["ECO_ID"].isin(selected_groups)]
    train = grouped_df[grouped_df["ECO_ID"].isin(selected_groups)]
    
    return train, test

'''

Preparing for ecoregion cross validation 
from shapely.geometry import box

ecoregions = gpd.read_file("data/Ecoregions/Ecoregions2017.shp")
ecoregions = ecoregions.to_crs("EPSG:4326")

bbox = (-170, 5, -50, 85)
crop_box = box(*bbox)

na_ecoregions = ecoregions[ecoregions.intersects(crop_box)].copy()
na_ecoregions=na_ecoregions[['ECO_BIOME_', 'geometry']]

del ecoregions

TEST_SIZE = 0.3

BATCH_SIZE = 16 # Set to 0 if not doing any deep learning 

gdf = gpd.GeoDataFrame(
    PID_df,
    geometry=gpd.points_from_xy(PID_df["lon"], PID_df["lat"]),
    crs="EPSG:4326"  # WGS84
)

gdf=gdf[['PID', 'lon', 'lat','geometry']]

ecoregion_cross_validation(gdf, na_ecoregions, TEST_SIZE, BATCH_SIZE)

'''