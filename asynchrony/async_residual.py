import sys
from pathlib import Path

import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from sklearn.preprocessing import StandardScaler
import utils.cross_validation as cval


seed=42
n_bins=50
n_rounds = 1000

final_df=pd.read_csv("data/final/final_dataset_ba.csv")
df= pd.read_csv("results/merged_diversity_asynchrony.csv")
final_df=final_df.merge(df[['PID',"asynchrony_LdM"]], how='left', on='PID')

PID_df= pd.read_csv('data/lookup/PID_location_v3.csv')

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

fd_df=final_df
fd_df=final_df.merge(PID_df[['PID','lat','lon', 'biome']], on='PID', how='left')

fd_df.drop(columns=['Unnamed: 0', 'managed', 'ownership', 'DIA', 'TPA_UNADJ','Functional_Richness', 'Shannon Equitabiltiy Index', 'transformed npp'], inplace=True)
fd_df.dropna(subset=[ "asynchrony_LdM", 'std npp', 'mean', 'Raos_Q', 'Functional_Evenness', 'Soil Moisture',
                      'Species Richness', 'Shannon Diversity', "Simpson's Index", "pet_std"], inplace=True)

fd_df = fd_df[fd_df["treecover2000"] > 10]
fd_df.drop_duplicates(subset=['PID'], inplace=True)

fd_df.rename(columns={'mean':'mean npp', 'pet_std': "PET sd", 'land_cover_value': "Land Cover", 'STDAGE': "Stand Age"}, inplace=True)
ecoregions=cval.process_ecoregion("data/Ecoregions/Ecoregions2017.shp")

ecoregions=ecoregions[['ECO_NAME', 'geometry']]
#Preprocessing the data for Random forest regression

fd_df['std npp']=np.log1p(fd_df['std npp'])
fd_df['mean npp']=np.log1p(fd_df['mean npp'])

fd_df = fd_df[fd_df["WSCI"] != 0]
fd_df = fd_df[fd_df["disturbance_value"] != -2147483648]

diversity_vars = ["Species Richness", "Shannon Diversity", "Raos_Q", "Simpson's Index", "Functional_Evenness"]

target_cols = ["asynchrony_LdM"] 
results = {col: [] for col in target_cols}

lon_range = fd_df["lon"].max() - fd_df["lon"].min()
lat_range = fd_df["lat"].max() - fd_df["lat"].min()
grid_size = max(lon_range, lat_range) / n_bins

fd_df= cval.assign_spatial_groups(
    fd_df, grid_size=grid_size
)
groups = fd_df['spatial_group'].unique()

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

    exclude_cols = ['PID', "asynchrony_LdM", 'std npp', 'WSCI'] + diversity_vars
    feature_cols = [c for c in train.columns if c not in exclude_cols]

    for target_col in target_cols:
        y_actual, y_pred = cval.fit_predict_group(
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

all_results_df.to_csv('results/asynch_residuals.csv', index=False)

res_df=pd.read_csv("results/step_2_residuals_merged.csv")

all_results_df['residual'] = all_results_df['y_actual'] - all_results_df['y_pred']

res_df.merge(all_results_df[['PID', 'residual']], how='left', on=['PID',]).to_csv('results/step_3_residuals_merged.csv', index=False)
