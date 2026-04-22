import pandas as pd
import numpy as np
import geopandas as gpd
from sklearn.ensemble import RandomForestRegressor
from utils.data_utils import evaluate_rf
import utils.cross_validation as cval
import random 
from shapely.geometry import box

import threading

'''
List of biomes: ['Boreal and Tundra forests', 
    'Mediterranean woodlands',
    'Temperate broadleaf forests',
    'Temperate conifer forests', 
    'Temperate grasslands', 
    'Tropical', 
    'Xeric shrublands'] 
'''


def train_test_split(sub_df, random_key, test_size):
    sub_df.drop(columns=['BHAGE', 'managed',
        'ownership', 'biome', 'geometry', 'index_right', 'lat', 'lon', 'DIA',
        'lon_bin', 'lat_bin'], inplace=True)

    sets=list(set(sub_df['ECO_NAME']))

    random.seed(random_key)
    test_set = random.sample(sets, int(test_size * len(sets)))
    test = sub_df[sub_df["ECO_NAME"].isin(test_set)]
    train = sub_df[~sub_df["ECO_NAME"].isin(test_set)]

    y=['transformed npp', 'GC'] 

    X_train=train.drop(columns=y+['PID', 'spatial_group', 'ECO_NAME'])
    y_train = np.column_stack([train['transformed npp'], train['GC']])

    X_test= test.drop(columns=y+['PID', 'spatial_group', 'ECO_NAME'])
    y_test = np.column_stack([test['transformed npp'], test['GC']])

    return X_train, y_train, X_test, y_test

def experiment (b_list, biome_dfs, random_key, test_size):
    
    biome_cmap = {
        "Boreal and Tundra forests": "blue",
        "Temperate broadleaf forests": "green",
        "Tropical": "gold",
        "Xeric shrublands": "darkred",
        "Temperate conifer forests": "darkgreen"
    }

    Feature_importance_df=pd.DataFrame()

    for i in b_list:
        sub_df=biome_dfs[i]
        color = sub_df["biome"].map(biome_cmap)
        biome_name = i
        print(biome_name)
        X_train, y_train, X_test, y_test = train_test_split(sub_df, random_key, test_size)
        fd_reg = RandomForestRegressor(random_state=RANDOM_KEY, n_jobs=2)
        fd_reg.fit(X_train, y_train)
        fimp_df = evaluate_rf(X_test, y_test, fd_reg,
                     feature_names=X_train.columns, 
                     importance= True, 
                     div_type= 'f', 
                     biome=biome_name,
                     color=color)
        
        fimp_df["biome"] = biome_name
        Feature_importance_df = pd.concat([Feature_importance_df, fimp_df], ignore_index=True)

    return Feature_importance_df


def experiment_wrapper(name, *args):
    output = experiment(*args)
    results[name] = output


TEST_SIZE = 0.4
RANDOM_KEY = 21
BATCH_SIZE=16

biome_mapping = {
    0: 'Boreal and Tundra forests',
    1: 'Flooded grasslands',
    2: 'Mangroves',
    3: 'Mediterranean woodlands',
    4: 'Temperate broadleaf forests',
    5: 'Temperate conifer forests',
    6: 'Temperate grasslands',
    7: 'Tropical',
    8: 'Tropical',
    9: 'Tropical',
    10: 'Tropical',
    11: 'Boreal and Tundra forests',
    12: 'Xeric shrublands',
    13: np.nan
}


df= pd.read_csv("fd_df_gc.csv")
ecoregions=cval.process_ecoregion("data/Ecoregions/Ecoregions2017.shp")

ecoregions=ecoregions[['ECO_NAME', 'geometry']]

#Preprocessing the data for Random forest regression

df = df[~df['biome'].isin([13, 1, 2, 3])]

df["biome"] = df["biome"].replace(biome_mapping)

df=cval.assign_spatial_groups(df, grid_size=1.0)

gdf = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df["lon"], df["lat"]),
    crs="EPSG:4326"  # WGS84
)

joined = gpd.sjoin(gdf, ecoregions, how="left", predicate="within")

biome_dfs = {k: v for k, v in joined.groupby('biome')}


l1= ['Boreal and Tundra forests', 'Tropical', 'Xeric shrublands']
l2=['Temperate broadleaf forests','Temperate conifer forests',]

results = {}

t1 = threading.Thread(target=experiment_wrapper, args=("t1", l1, biome_dfs, RANDOM_KEY, TEST_SIZE))
t2 = threading.Thread(target=experiment_wrapper, args=("t2", l2, biome_dfs, RANDOM_KEY, TEST_SIZE))

t1.start()
t2.start()

t1.join()
t2.join()


df1=results["t1"]
df2=results["t2"]
df1.concat(df2, ignore_index=True).to_csv("feature_importance_results.csv", index=False)