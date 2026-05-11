import pandas as pd
import numpy as np
import geopandas as gpd
from sklearn.ensemble import RandomForestRegressor
from utils.data_utils import evaluate_rf
import utils.cross_validation as cval
import random 
from shapely.geometry import box

import threading
import utils.model_utils as mu

from sklearn.model_selection import RandomizedSearchCV


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
        'lon_bin', 'lat_bin', 'GC'], inplace=True)

    sets=list(set(sub_df['ECO_NAME']))

    random.seed(random_key)
    test_set = random.sample(sets, int(test_size * len(sets)))
    test = sub_df[sub_df["ECO_NAME"].isin(test_set)]
    train = sub_df[~sub_df["ECO_NAME"].isin(test_set)]

    y=['transformed npp'] 

    X_train=train.drop(columns=y+['PID', 'spatial_group', 'ECO_NAME'])
    y_train = train['transformed npp'].values

    X_test= test.drop(columns=y+['PID', 'spatial_group', 'ECO_NAME'])
    y_test = test['transformed npp'].values

    return X_train, y_train, X_test, y_test


def experiment (b_list, biome_dfs, random_key, test_size, param_grid):

    df=pd.DataFrame()
    for i in b_list:

        sub_df=biome_dfs[i]
        biome_name = i
        print(biome_name)
        

        X_train, y_train, X_test, y_test = train_test_split(sub_df, random_key, test_size)
        model = RandomForestRegressor(random_state=random_key, n_jobs=2)


        search = RandomizedSearchCV(
            model,
            param_distributions=param_grid,
            n_iter=20,           # number of random combinations to try
            cv=5,                 # 5-fold cross validation
            scoring='r2',         # or 'neg_mean_squared_error'
            n_jobs=4,            # use all cores
            verbose=2,
            random_state=RANDOM_KEY
        )

        search.fit(X_train, y_train)
        df.append({'biome': biome_name,
                'best_params': search.best_params_,
                'score': search.best_score_}, ignore_index=True )
        
    return df
    

def experiment_wrapper(name, *args):
    output = experiment(*args)
    f_df[name] = output
if __name__ == "__main__":

    TEST_SIZE = 0.3
    BATCH_SIZE=16
    RANDOM_KEY =42
    # ── Parameter grid ──
    param_grid = {
        'n_estimators':      [100, 200, 300, 500],
        'max_depth':         [5, 10, 20, 30, 40],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf':  [1, 2, 4],
        'max_features':      ['sqrt'],
        'bootstrap':         [True, False]
    }

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


    fd_df= pd.read_csv("data/final/fd_df.csv")
    sd_df = pd.read_csv("data/final/sd_df.csv")

    ecoregions=cval.process_ecoregion("data/Ecoregions/Ecoregions2017.shp")

    ecoregions=ecoregions[['ECO_NAME', 'geometry']]

    fd_df = fd_df.loc[:, ~fd_df.columns.str.contains(r'\.1')]
    sd_df = sd_df.loc[:, ~sd_df.columns.str.contains(r'\.1')]

    fd_df.drop(columns=['Mean Pairwise D'], inplace=True)
    fd_df.dropna(subset=['TPA_UNADJ'], inplace=True) # Maybe exclude

    sd_df.dropna(subset=['TPA_UNADJ'], inplace=True) # Maybe exclude
    sd_df = sd_df[sd_df["WSCI"] >= -1000]
    fd_df = fd_df[fd_df["WSCI"] >= -1000]


    sd_df.columns = sd_df.columns.str.replace('_x$', '', regex=True)
    sd_df.columns = sd_df.columns.str.replace('_y$', '', regex=True)

    sd_df = sd_df.loc[:, ~sd_df.columns.duplicated()]


    shared_pids = set(sd_df["PID"]).intersection(fd_df["PID"])

    # Filter both dataframes to only shared PIDs
    sd_df = sd_df[sd_df["PID"].isin(shared_pids)]
    fd_df = fd_df[fd_df["PID"].isin(shared_pids)]

    #Preprocessing the data for Random forest regression

    fbiome_dfs= mu.data_processing_v2(fd_df, biome_mapping)
    sbiome_dfs =mu.data_processing_v2(sd_df, biome_mapping)

    l1= ['Temperate broadleaf forests', 'Tropical' ]
    l2= ['Temperate conifer forests', 'Temperate grasslands','Xeric shrublands']

    f_df = {}
    s_df = {}

    t1 = threading.Thread(target=experiment_wrapper, args=("t1", l1, fbiome_dfs, RANDOM_KEY, TEST_SIZE, param_grid))
    t2 = threading.Thread(target=experiment_wrapper, args=("t2", l2, fbiome_dfs, RANDOM_KEY, TEST_SIZE, param_grid))
    t3 = threading.Thread(target=experiment_wrapper, args=("t3", l1, sbiome_dfs, RANDOM_KEY, TEST_SIZE, param_grid))
    t4 = threading.Thread(target=experiment_wrapper, args=("t4", l2, sbiome_dfs, RANDOM_KEY, TEST_SIZE, param_grid))


    t1.start()
    t2.start()
    t3.start()
    t4.start()

    t1.join()
    t2.join()
    t3.join()
    t4.join()


    df1 = pd.DataFrame(f_df["t1"])
    df2 = pd.DataFrame(f_df["t2"])

    df3 = pd.DataFrame(s_df["t3"])
    df4 = pd.DataFrame(s_df["t4"])


    f_df_params=pd.concat([df1, df2], ignore_index=True)
    s_df_params=pd.concat([df3, df4], ignore_index=True)


    f_df_params.to_csv("results/fd_rf_hyperparameters.csv", index=False)
    s_df_params.to_csv("results/sd_rf_hyperparameters.csv", index=False)


