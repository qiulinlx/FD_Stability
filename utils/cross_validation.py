import geopandas as gpd
import numpy as np
from shapely.geometry import box
import random
from math import radians, sin, cos, sqrt, asin


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
    grouped_df = grouped_df.groupby("ECO_ID").filter(lambda x: len(x) >= batch_size)
    # number to sample
    n_select = int(len(groups) * test_size)
    
    selected_groups = np.random.choice(groups, size=n_select, replace=False)
    print(selected_groups)
    test = grouped_df[grouped_df["ECO_ID"].isin(selected_groups)]
    train = grouped_df[grouped_df["ECO_ID"].isin(selected_groups)]
    
    return train, test


def haversine(lat1, lon1, lat2, lon2):
    """Calculate great-circle distance between two points in km"""
    R = 6371  # Earth's radius in km
    
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    
    return R * c

def select_points_stratified(df, n_per_biome=5, biome_col='biome_name', buffer_km=50,
                            lat_col='lat', lon_col='lon', random_seed=None):
    """
    Select exactly n_per_biome points from each biome.
    
    Parameters:
    - df: DataFrame with coordinates
    - n_per_biome: number of points to select per biome
    - biome_col: column name for biome
    - lat_col: latitude column name
    - lon_col: longitude column name
    - random_seed: random seed for reproducibility
    
    Returns:
    - selected_data: DataFrame with selected points
    - selected_indices: indices of selected points
    - biomes_sampled: dict with biome names and number sampled
    """
    
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Get biomes present
    biomes = df[biome_col].unique()
    
    selected_indices = []
    biomes_sampled = {}
    
    for biome in biomes:
        # Get indices for this biome
        biome_indices = df[df[biome_col] == biome].index.tolist()
        
        if len(biome_indices) < n_per_biome:
            print(f"Warning: Biome '{biome}' has only {len(biome_indices)} points, "
                  f"sampling all {len(biome_indices)}")
            n_to_sample = len(biome_indices)
        else:
            n_to_sample = n_per_biome
        
        # Randomly select points
        selected = np.random.choice(biome_indices, n_to_sample, replace=False)
        selected_indices.extend(selected)
        biomes_sampled[biome] = len(selected)
    
    # Get selected data
    selected_data = df.loc[selected_indices].copy()
    
    # Get unique coordinates (if needed)
    coords = selected_data[[lat_col, lon_col]].drop_duplicates().reset_index(drop=True)

    
    # Identify points within buffer
    selected_points = []
    removed_points = []
    remaining_points = []
    
    for idx, row in coords.iterrows():
        lat = row[lat_col]
        lon = row[lon_col]
        
        # Check distance to all selected points
        within_buffer = False
        for _, sel_row in coords.iterrows():
            dist = haversine(lat, lon, sel_row[lat_col], sel_row[lon_col])
            if dist <= buffer_km:
                within_buffer = True
                break
        
        if idx in selected_indices:
            selected_points.append(idx)
        elif within_buffer:
            removed_points.append(idx)
        else:
            remaining_points.append(idx)
    
    # Create DataFrames
    selected_df = df.iloc[selected_points].copy()
    removed_df = df.iloc[removed_points].copy()
    remaining_df = df.iloc[remaining_points].copy()
    
    return selected_df, remaining_df, removed_df


def select_points_with_buffer(df, n_points, buffer_km=100, lat_col='lat', lon_col='lon', 
                              random_seed=None):
    """
    Randomly select n_points and remove any points within buffer_km of selected points
    
    Parameters:
    - df: DataFrame with coordinates
    - n_points: number of points to randomly select
    - buffer_km: buffer radius in kilometers (default: 100)
    - lat_col: name of latitude column
    - lon_col: name of longitude column
    - random_seed: random seed for reproducibility
    
    Returns:
    - selected: DataFrame of selected points
    - remaining: DataFrame of points outside buffer
    - removed: DataFrame of points within buffer of selected points
    """
    
    if random_seed is not None:
        np.random.seed(random_seed)
        random.seed(random_seed)
    
    # Get unique coordinates (if multiple rows per location)
    coords = df[[lat_col, lon_col]].drop_duplicates().reset_index(drop=True)
    
    if len(coords) < n_points:
        raise ValueError(f"Only {len(coords)} unique points available, but {n_points} requested")
    
    # Randomly select n_points
    selected_indices = np.random.choice(len(coords), n_points, replace=False)
    selected_coords = coords.iloc[selected_indices].copy()
    
    # Identify points within buffer
    selected_points = []
    removed_points = []
    remaining_points = []
    
    for idx, row in coords.iterrows():
        lat = row[lat_col]
        lon = row[lon_col]
        
        # Check distance to all selected points
        within_buffer = False
        for _, sel_row in selected_coords.iterrows():
            dist = haversine(lat, lon, sel_row[lat_col], sel_row[lon_col])
            if dist <= buffer_km:
                within_buffer = True
                break
        
        if idx in selected_indices:
            selected_points.append(idx)
        elif within_buffer:
            removed_points.append(idx)
        else:
            remaining_points.append(idx)
    
    # Create DataFrames
    selected_df = df.iloc[selected_points].copy()
    removed_df = df.iloc[removed_points].copy()
    remaining_df = df.iloc[remaining_points].copy()
    
    return selected_df, remaining_df, removed_df



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