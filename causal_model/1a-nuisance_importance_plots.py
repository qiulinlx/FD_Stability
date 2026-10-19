import math
import yaml
import sys
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[1]))

import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import utils.cross_validation as cval
import utils.feature_utils as futils
from utils.visualisation_utils import plot_importance_grouped_horizontal


try:
    from statsmodels.nonparametric.smoothers_lowess import lowess
    HAS_LOWESS = True
except ImportError:
    HAS_LOWESS = False

# ------------------------------------------------------------------
# Colours per target (matched to the SHAP importance figure).
# Simpson's Index and the NPP targets weren't in that figure, so
# those three colours are new picks -- change freely.
# ------------------------------------------------------------------

# "Shannon Diversity":   "#7DDFE7",   # light cyan

TARGET_COLORS = {
    "Functional_Evenness": "#9975BA",   # muted purple
    "Raos_Q":              "#4A16D8",   # deep indigo
    "Species Richness":    "#16D2B8",   # teal
}

N_PERM_REPEATS = 10   # shuffles per feature for permutation importance
TOP_N = 8

params = {
    "objective": "reg:squarederror",
    "learning_rate": 0.01,
    "max_depth": 6,
    "min_child_weight": 2,
    "gamma": 0.1,
    "lambda": 1.0,
    "alpha": 0.0,
    "subsample": 0.7,
    "tree_method": "approx"
}
n_rounds = 500


fd_df = pd.read_csv("data/final/final_dataset_ba.csv")
PID_df = pd.read_csv('data/lookup/PID_location_v3.csv')

fd_df = fd_df.merge(
    PID_df[['PID', 'lat', 'lon', 'biome', 'STDAGE']],
    on='PID', how='left'
)
fd_df.drop(columns=['Unnamed: 0', 'managed', 'ownership', 'DIA', 'TPA_UNADJ', 'Functional_Richness', 'Shannon Equitabiltiy Index', 'transformed npp'], inplace=True)
fd_df.dropna(subset=['std npp', 'mean', 'Raos_Q', 'Functional_Evenness', 'Soil Moisture',
                      'Species Richness', 'Shannon Diversity', "Simpson's Index", "pet_std"], inplace=True)

fd_df = fd_df[fd_df["treecover2000"] > 10]
fd_df.drop_duplicates(subset=['PID'], inplace=True)

fd_df.rename(columns={'mean': 'mean npp', 'pet_std': "PET sd",
                       'land_cover_value': "Land Cover", 'STDAGE': "Stand Age", "CHELSA_exBIO_GrowingSeasonLength": 'Growing Season Length'}, inplace=True)

fd_df['std npp'] = np.log1p(fd_df['std npp'])
fd_df['mean npp'] = np.log1p(fd_df['mean npp'])

ecoregions = cval.process_ecoregion("data/Ecoregions/Ecoregions2017.shp")
ecoregions = ecoregions[['ECO_NAME', 'geometry']]

fd_df = fd_df[fd_df["WSCI"] != 0]
fd_df.drop(columns=['WSCI'], inplace=True)  # Drop WSCI column after filtering
fd_df = fd_df[fd_df["disturbance_value"] != -2147483648]

diversity_vars = ["Raos_Q", "Functional_Evenness", "Species Richness"]
target_cols = diversity_vars

exclude_cols = ['PID', 'lat', 'lon', 'biome', 'treecover2000', 'std npp', 'mean npp',  "Simpson's Index","Shannon Diversity"] + diversity_vars
feature_cols = [c for c in fd_df.columns if c not in exclude_cols]

Path('results/shap_values').mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# Fit once per target on the full dataset; compute SHAP for every row,
# plus gain and permutation importance from the same model.
# ------------------------------------------------------------------
shap_dfs = {}
importance_rows = []   # mean |SHAP|
gain_rows = []         # normalised total gain
perm_rows = []         # permutation importance (drop in R²)

for target_col in target_cols:
    print(f"Fitting {target_col}...")
    model, shap_df, X_s = futils.fit_and_get_shap(fd_df, target_col, feature_cols, params, n_rounds)
    shap_dfs[target_col] = shap_df

    safe_name = target_col.replace(" ", "_").replace("'", "")
    shap_df.to_csv(f'results/shap_values/shap_values_{safe_name}.csv', index=False)

    gain = futils.gain_importance(model, feature_cols, normalize=True)
    perm = futils.permutation_importance_booster(
        model, X_s, fd_df[target_col].values, n_repeats=N_PERM_REPEATS
    )

    for feat in feature_cols:
        importance_rows.append({
            'target': target_col,
            'feature': feat,
            'mean_abs_shap': shap_df[f'shap_{feat}'].abs().mean(),
        })
        gain_rows.append({
            'target': target_col,
            'feature': feat,
            'gain_importance': gain[feat],
        })
        perm_rows.append({
            'target': target_col,
            'feature': feat,
            'perm_importance': np.abs(perm[feat][0]),
            'perm_importance_sd': perm[feat][1],
        })

importance_df = pd.DataFrame(importance_rows).sort_values(
    ['target', 'mean_abs_shap'], ascending=[True, False]
)

importance_df['shap_share'] = (
    importance_df['mean_abs_shap']
    / importance_df.groupby('target')['mean_abs_shap'].transform('sum')
)

gain_df = pd.DataFrame(gain_rows).sort_values(
    ['target', 'gain_importance'], ascending=[True, False]
)
perm_df = pd.DataFrame(perm_rows).sort_values(
    ['target', 'perm_importance'], ascending=[True, False]
)

importance_df.to_csv('results/shap_feature_importance.csv', index=False)
gain_df.to_csv('results/gain_feature_importance.csv', index=False)
perm_df.to_csv('results/permutation_feature_importance.csv', index=False)

# ------------------------------------------------------------------
# Importance figures (horizontal grouped bars, coloured by target)
# ------------------------------------------------------------------
plot_targets = diversity_vars          # drop e.g. "Simpson's Index" here to match your earlier figure

# plot_importance_grouped_horizontal(
#     importance_df, value_col="shap_share",
#     xlabel="Share of total mean |SHAP|",
#     title="SHAP Feature Importance by Environmental Variable, Grouped by Target",
#     savepath="results/figures/importance_shap.png",
#     targets=plot_targets,
# )
# plot_importance_grouped_horizontal(
#     gain_df, value_col="gain_importance",
#     xlabel="Gain importance (share of total gain)",
#     title="Gain (Gini-style) Feature Importance by Environmental Variable, Grouped by Target",
#     savepath="results/figures/importance_gain.png",
#     targets=plot_targets, 
# )

# plot_importance_grouped_horizontal(
#     perm_df, value_col="perm_importance", err_col="perm_importance_sd",
#     xlabel="Permutation importance (drop in R$^2$)",
#     title="Permutation Feature Importance by Environmental Variable, Grouped by Target",
#     savepath="results/figures/importance_permutation.png",
#     targets=plot_targets,
# )

# Importance bar chart: top 8 features (ranked by summed share across targets)
plot_importance_grouped_horizontal(
    importance_df, value_col="shap_share",
    xlabel="Share of total mean |SHAP|",
    title=f"SHAP Feature Importance (Top {TOP_N}), Grouped by Target",
    savepath=f"results/figures/importance_shap_top{TOP_N}.png",
    targets=plot_targets,
    top_n=TOP_N,
)