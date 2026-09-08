import math
from xml.parsers.expat import model
import yaml
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from pathlib import Path
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import random 
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import utils.cross_validation as cval

def fit_predict_group(train_df, test_df, target_col, feature_cols, params, n_rounds):
    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]
    y_train = train_df[target_col].values
    y_test = test_df[target_col].values

    scaler = StandardScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index)
    X_test_s = pd.DataFrame(scaler.transform(X_test), columns=feature_cols, index=X_test.index)

    dtrain = xgb.DMatrix(X_train_s, y_train, enable_categorical=True)
    dtest = xgb.DMatrix(X_test_s, y_test, enable_categorical=True)

    model = xgb.train(params=params, dtrain=dtrain, num_boost_round=n_rounds)
    y_pred = model.predict(dtest)
    return y_test, y_pred  # arrays, one value per test row


def assign_spatial_groups(df, grid_size=1.0):
    df = df.copy()

    df["lon_bin"] = (df["lon"] // grid_size) * grid_size
    df["lat_bin"] = (df["lat"] // grid_size) * grid_size
    df["spatial_group"] = (
        df["lon_bin"].astype(str) + "_" +
        df["lat_bin"].astype(str)
    )
    return df

def fit_predict_one(train_df, test_df, target_col, feature_cols, params, n_rounds):
    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]
    y_train = train_df[target_col].values
    y_test = test_df[target_col].values

    scaler = StandardScaler()
    X_train_s = pd.DataFrame(
        scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index
    )
    X_test_s = pd.DataFrame(
        scaler.transform(X_test), columns=feature_cols, index=X_test.index
    )

    dtrain = xgb.DMatrix(X_train_s, y_train, enable_categorical=True)
    dtest = xgb.DMatrix(X_test_s, y_test, enable_categorical=True)

    model = xgb.train(params=params, dtrain=dtrain, num_boost_round=n_rounds)
    y_pred = model.predict(dtest)
    return y_test[0], y_pred[0]

with open("config/ex_config.yaml", "r") as f:
    config = yaml.safe_load(f)

biome_mapping = config["biome_mapping"]
random_key=config["random_key"]


# Data Cleaning and Preprocessing
fd_df = pd.read_csv("data/final/final_dataset_ba_v2.csv")
PID_df= pd.read_csv('data/lookup/PID_location_v3.csv')

fd_df=fd_df.merge(PID_df[['PID','lat','lon', 'biome', 'STDAGE', 'percent_conifer' ]], on='PID', how='left')

fd_df.drop(columns=['Unnamed: 0', 'managed', 'ownership', 'DIA', 'TPA_UNADJ','Functional_Richness', 'Shannon Equitabiltiy Index', 'transformed npp'], inplace=True)
fd_df.dropna(subset=['std npp', 'mean', 'Raos_Q', 'Functional_Evenness', 'Soil Moisture',
                      'Species Richness', 'Shannon Diversity', "Simpson's Index", "pet_std"], inplace=True)

fd_df = fd_df[fd_df["treecover2000"] > 30]
fd_df.drop_duplicates(subset=['PID'], inplace=True)

fd_df.rename(columns={'mean':'mean npp', 'pet_std': "PET sd", 'land_cover_value': "Land Cover", 'STDAGE': "Stand Age"}, inplace=True)
ecoregions=cval.process_ecoregion("data/Ecoregions/Ecoregions2017.shp")

ecoregions=ecoregions[['ECO_NAME', 'geometry']]
#Preprocessing the data for Random forest regression

fd_df['std npp']=np.log1p(fd_df['std npp'])
fd_df['mean npp']=np.log1p(fd_df['mean npp'])

fd_df['biome'] = fd_df['biome'].map(biome_mapping) # Only run this once ! 

biome_dfs = {k: v for k, v in fd_df.groupby('biome')}

fd_df = fd_df[fd_df["WSCI"] != 0]

fd_df.drop(columns=['WSCI'], inplace=True)  # Drop WSCI column after filtering

fd_df = fd_df[fd_df["disturbance_value"] != -2147483648]

params = {
    "objective": "reg:squarederror",
    "learning_rate": 0.01,
    "max_depth": 6,
    "min_child_weight": 2,
    "gamma": 0.1,
    "lambda": 1.0,
    "alpha": 0.0,
    "subsample": 0.7,
    "tree_method": "hist"
}
n_rounds = 500

diversity_vars = ["Species Richness", "Shannon Diversity", "Raos_Q", "Simpson's Index", "Functional_Evenness"]

target_cols = ['std npp', 'mean npp'] + diversity_vars
results = {col: [] for col in target_cols}

seed=42

rng = np.random.default_rng(seed)
random_key = int(rng.integers(0, 2**32 - 1))

n_bins=50

lon_range = fd_df["lon"].max() - fd_df["lon"].min()
lat_range = fd_df["lat"].max() - fd_df["lat"].min()
grid_size = max(lon_range, lat_range) / n_bins

fd_df= cval.assign_spatial_groups(
    fd_df, grid_size=grid_size
)
groups = fd_df['spatial_group'].unique()
print(groups.shape)
i=0
for group in groups:
    print(f'Fitting on group {i}...')
    test = fd_df[fd_df['spatial_group'] == group]
    test = test.drop(columns=['lat', 'lon', 'biome',
                               "lon_bin", "lat_bin", "spatial_group", 'treecover2000'])
    test_pid = test[['PID']]

    if test.empty:
        print(f"Skipping group {group}: no rows")
        continue

    train = fd_df[fd_df['spatial_group'] != group]
    train = train.drop(columns=['lat', 'lon', 'biome',
                                 "lon_bin", "lat_bin", "spatial_group", 'treecover2000'])


    exclude_cols = ['PID', 'std npp', 'mean npp'] + diversity_vars
    feature_cols = [c for c in train.columns if c not in exclude_cols]


    for target_col in target_cols:
        y_actual, y_pred = fit_predict_group(
            train, test, target_col, feature_cols, params, n_rounds
        )
        for pid, actual, pred in zip(test_pid['PID'].values, y_actual, y_pred):
            results[target_col].append({
                'group': group,
                'PID': pid,
                'y_actual': actual,
                'y_pred': pred,
            })
        print(f'Fitting on {target_col} for group {i} done')
    i+=1

# build the combined results DataFrame once, after all groups/targets are done
results_dfs = {target: pd.DataFrame(rows) for target, rows in results.items()}
all_results_df = pd.concat(
    [df.assign(target=target) for target, df in results_dfs.items()],
    ignore_index=True
)

all_results_df.to_csv('results/step_1_residuals.csv', index=False)

