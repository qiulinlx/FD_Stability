
import geopandas as gpd
import utils.cross_validation as cval
import random 

ecoregions=cval.process_ecoregion("data/Ecoregions/Ecoregions2017.shp")

ecoregions=ecoregions[['ECO_NAME', 'geometry']]

def data_preprocessing (df, random_key=42):
    #Preprocessing the data for Random forest regression

    df = df[~df['biome'].isin([13, 1, 2, 3])]

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


def train_test_split(sub_df, random_key):
    sub_df = sub_df.drop(columns=['BHAGE', 'managed', 'ownership', 'geometry', 
                                   'index_right', 'lat', 'lon', 'DIA',
                                   'lon_bin', 'lat_bin', 'GC'])
    
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