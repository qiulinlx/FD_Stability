
import geopandas as gpd
import utils.cross_validation as cval
import random 
import pandas as pd
from sklearn.preprocessing import StandardScaler
import random

ecoregions=cval.process_ecoregion("data/Ecoregions/Ecoregions2017.shp")

ecoregions=ecoregions[['ECO_NAME', 'geometry']]

def data_preprocessing (df, random_key=42):
    #Preprocessing the data for Random forest regression

    # df = df[~df['biome'].isin([13, 1, 2, 3])]

    df = df[df['biome'].isin([4,5,6,12])]


    # df["biome"] = df["biome"].replace(biome_mapping)

    df=cval.assign_spatial_groups(df, grid_size=1.0)

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:4326"  # WGS84
    )

    joined = gpd.sjoin(gdf, ecoregions, how="left", predicate="within")

    X_train, y_train, X_test, y_test = train_test_split(joined, random_key)
    return X_train, y_train, X_test, y_test

def data_processing_v2(df, biome_mapping, ecoregions):
     
    df["biome"] = df["biome"].replace(biome_mapping)

    df=cval.assign_spatial_groups(df, grid_size=1.0)

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:4326"  # WGS84
    )

    joined = gpd.sjoin(gdf, ecoregions, how="left", predicate="within")

    biome_dfs = {k: v for k, v in joined.groupby('biome')}
    return biome_dfs

def train_test_split_v1(sub_df, random_key):
    sub_df = sub_df.drop(columns=['BHAGE', 'managed', 'ownership', 'geometry', 
                                   'index_right', 'lat', 'lon', 'DIA',
                                   'lon_bin', 'lat_bin'])
    
    sets = list(set(sub_df['ECO_NAME']))
    set_b = list(set(sub_df['biome']))
    
    random.seed(random_key)
    
    test = []
    remaining_sets = sets.copy()
    n=0
    for biome in set_b:
            # Only consider candidates still in remaining_sets
            candidates = list(set(sub_df[sub_df['biome'] == biome]['ECO_NAME']) & set(remaining_sets))
            chosen = random.choice(candidates)
            test.append(chosen)
            remaining_sets.remove(chosen)
    
    test_set = sub_df[sub_df["ECO_NAME"].isin(test)]
    training_set = sub_df[~sub_df["ECO_NAME"].isin(test)]

    y=['transformed npp'] 

    X_train=training_set.drop(columns=y+['PID', 'spatial_group', 'ECO_NAME'])
    y_train = training_set['transformed npp'].values

    X_test= test_set.drop(columns=y+['PID', 'spatial_group', 'ECO_NAME'])
    y_test = test_set['transformed npp'].values

    return X_train, y_train, X_test, y_test



def train_test_split_v2(df, random_key, n_points_to_select=50, buffer_km=100):
    sub_df = df.drop(columns=['biome', 'lon_bin', 'lat_bin', 'geometry',
       'index_right', ])

    
    random_key =10

    selected, remaining, removed = cval.select_points_with_buffer(
        sub_df, 
        n_points=n_points_to_select,
        buffer_km=buffer_km,
        lat_col='lat',
        lon_col='lon',
        random_seed=random_key
    )

    test = selected.drop(columns=['lat', 'lon'])
    train= remaining.drop(columns=['lat', 'lon'])

    y=['transformed npp'] 
    X_train=train.drop(columns=y+['PID', 'spatial_group', 'ECO_NAME'])
    y_train = train['transformed npp'].values

    X_test= test.drop(columns=y+['PID', 'spatial_group', 'ECO_NAME'])
    y_test = test['transformed npp'].values
    
    scaler= StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    return X_train, y_train, X_test, y_test

