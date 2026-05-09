import pandas as pd
import geopandas as gpd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

import utils.cross_validation as cval
import random 
import matplotlib.pyplot as plt

import utils.model_utils as mu
import utils.eval_utils as ev
import shap
import numpy as np
from sklearn.inspection import partial_dependence
import utils.model_utils as mu

RANDOM_KEY = 42

random.seed(RANDOM_KEY)

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

sX_train, sX_test, sy_train, sy_test = mu.data_preprocessing(sd_df, RANDOM_KEY)
fX_train, fX_test, fy_train, fy_test = mu.data_preprocessing(fd_df, RANDOM_KEY)

sd_reg=RandomForestRegressor(bootstrap=True,
    max_depth=20,
    max_features='sqrt', 
    min_samples_leaf=4, 
    min_samples_split=5, 
    n_estimators=300,
    random_state=RANDOM_KEY
)

fd_reg=RandomForestRegressor(
    n_estimators=300, 
    min_samples_split=5, 
    min_samples_leaf=2,
    max_features='sqrt',
    max_depth=20,
    bootstrap=False, 
    random_state=RANDOM_KEY
)

