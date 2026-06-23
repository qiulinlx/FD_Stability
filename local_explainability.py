# Standard library
import random
import matplotlib.pyplot as plt
# Data manipulation
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score
)
from sklearn.model_selection import (
    KFold)

from sklearn.inspection import PartialDependenceDisplay
import math

# Custom utilities
from sklearn.preprocessing import StandardScaler

import utils.cross_validation as cval
from utils.model_utils import data_processing_v2, train_test_split_v2
import utils.eval_utils as eval
import xgboost as xgb
import shap

biome_mapping = {
    'Temperate broadleaf forests': 'Temperate broadleaf forests',
    'Temperate conifer forests': 'Temperate conifer forests',
    'Temperate grasslands': 'Temperate grasslands',
    'Xeric shrublands': 'Xeric shrublands',
    'Mediterranean woodlands': 'Mediterranean woodlands',
    'Flooded grasslands': 'Flooded grasslands',
    'Mangroves': 'Mangroves',
    'Tundra': 'Boreal and Tundra forests',
    'Boreal forests or taiga': 'Boreal and Tundra forests',
    'Tropical grasslands': 'Tropical',
    'Tropical coniferous forests': 'Tropical',
    'Tropical moist broadleaf forests': 'Tropical',
    'Tropical dry broadleaf forests': 'Tropical',
    np.nan: np.nan,  # Will be dropped
}

fd_df = pd.read_csv('data/final/final_dataset_with_aridity.csv')
# fd_df.drop(columns=[ 'managed', 'ownership','Functional_Divergences', 'Shannon Equitabiltiy Index', 'DIA'], inplace=True)
fd_df.dropna(subset=['Raos_Q', 'Functional_Evenness',
                     'Species Richness', 'Shannon Diversity', "Simpson's Index", 'biome'], inplace=True)

ecoregions=cval.process_ecoregion("data/Ecoregions/Ecoregions2017.shp")

ecoregions=ecoregions[['ECO_NAME', 'geometry']]

#Preprocessing the data for Random forest regression
# fbiome_dfs= data_processing_v2(fd_df, biome_mapping, ecoregions)

fd_df['biome'] = fd_df['biome'].map(biome_mapping) # Only run this once ! 

biome_dfs = {k: v for k, v in fd_df.groupby('biome')}


biome_list=["Temperate broadleaf forests", "Temperate conifer forests", "Temperate grasslands",
             "Xeric shrublands", "Boreal and Tundra forests", "Tropical",
               "Mediterranean woodlands"]


# Define biome-specific configurations in a dictionary
biome_configs = {
    'Temperate conifer forests': {
        'params': {
            'objective': 'reg:squarederror',
            'learning_rate': 0.025,
            'tree_method': 'approx'
        },
        'n_rounds': 240
    },
    'Boreal and Tundra forests': {
        'params':  {"objective": "reg:squarederror",
           "learning_rate": 0.05,  
          'max_depth': 6,
           'min_child_weight': 5,        # Increased from 3
            'gamma': 0.1,                # Added
            'reg_lambda': 2.0,           # Increased from 1
            'reg_alpha': 0.5,  "tree_method":'approx'} ,
        'n_rounds': 30
    },
    'Xeric shrublands': {
        'params': {
            'objective': 'reg:squarederror',
            'learning_rate': 0.05,
            'max_depth': 6,
            'min_child_weight': 5,
            'gamma': 0.1,
            'reg_lambda': 2.0,
            'reg_alpha': 0.5,
            'tree_method': 'approx'
        },
        'n_rounds': 50
    },
    'Temperate grasslands': {
        'params': {"objective": "reg:squarederror",
           "learning_rate": 0.05,  
          'max_depth': 6,
           'min_child_weight': 5,        # Increased from 3
            'gamma': 0.1,                # Added
            'reg_lambda': 2.0,           # Increased from 1
            'reg_alpha': 0.5,  "tree_method":'approx'
        } ,
        'n_rounds': 100
    },
    'Mediterranean woodlands': {
        'params': {
            'objective': 'reg:squarederror',
            'learning_rate': 0.05,
            'max_depth': 6,
            'min_child_weight': 5,
            'gamma': 0.1,
            'reg_lambda': 2.0,
            'reg_alpha': 0.5,
            'tree_method': 'approx'
        },
        'n_rounds': 50,
        'note': 'Random Forest actually does better wow'
    },
    'Temperate broadleaf forests': {
        'params': {
            'objective': 'reg:squarederror',
            'learning_rate': 0.001,
            'max_depth': 6,
            'min_child_weight': 5,
            'gamma': 0.1,
            'reg_lambda': 3.0,
            'reg_alpha': 0.5,
            'tree_method': 'approx'
        },
        'n_rounds': 10
    },
    'Tropical': {
        'params': {
            'objective': 'reg:squarederror',
            'learning_rate': 0.005,
            'max_depth': 6,
            'min_child_weight': 5,
            'gamma': 0.1,
            'reg_lambda': 2.0,
            'reg_alpha': 0.5,
            'tree_method': 'approx'
        },
        'n_rounds': 210
    }
}



random_key=12

biome_list= ['Temperate grasslands']# ["Temperate broadleaf forests", "Temperate conifer forests", "Temperate grasslands",
            # "Xeric shrublands", "Boreal and Tundra forests", "Tropical",
            #   "Mediterranean woodlands"]
# Select biome
# Use this versionl during finetuning and eval
for biome in biome_list:
    print(biome)
    # Get configuration
    config = biome_configs[biome]
    params = config['params']
    n_rounds = config['n_rounds']
    test_size=0.1
    # Training code
    n_points = round(len(biome_dfs[biome]) * test_size)
    fX_train, fy_train, fX_test, fy_test = train_test_split_v2(
        biome_dfs[biome], 
        random_key=random_key, 
        n_points_to_select=n_points, 
        buffer_km=50
    )

    # Create regression matrices
    dtrain_reg = xgb.DMatrix(fX_train, fy_train, enable_categorical=True)
    dtest_reg = xgb.DMatrix(fX_test, fy_test, enable_categorical=True)


    evals = [(dtrain_reg, "train"), (dtest_reg, "validation")]

    model = xgb.train(
    params=params,
    dtrain=dtrain_reg,
    num_boost_round=n_rounds,
    )
    fig, metrics = eval.evaluate_xgboost(
        model=model,
        dtrain=dtrain_reg,
        dtest=dtest_reg,
        X_test=fX_test,
        y_test=fy_test,
        feature_names=fX_test.columns.tolist(),
        biome_name='Overall',
        show_shap=True
    )
        # Compute SHAP values
    explainer = shap.TreeExplainer(model)
    print('pass1')

    X=biome_dfs[biome].drop(columns=['transformed npp', 'PID', 'lat', 'lon', 'biome'])
    shap_vals = explainer.shap_values(xgb.DMatrix(X), approximate=True)
    print('pass2')


    # Each tuple: (functional/top feature, species/bottom feature, title)
    pairs = [
    ('Elevation',               "Annual Temp",    "Raos Q vs Simpson's Index"),
    ('Functional_Evenness', 'Shannon Diversity',      "Functional Evenness vs Shannon Index"),
    ('Species Richness',     "Species Richness",   "Species Richness"),
    ('WSCI',                 "WSCI",               "WSCI"),
    ]

    colors_top    = "#1341ea"
    colors_bottom = "#c88f8f"
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes_flat = axes.flatten()
    fig.subplots_adjust(hspace=0.5, wspace=0.35)

    for ax, (top_feat, bot_feat, title) in zip(axes_flat, pairs):
        top_idx = X.columns.tolist().index(top_feat)
        bot_idx = X.columns.tolist().index(bot_feat)

        x_top = X[top_feat].values
        y_top = shap_vals[:, top_idx]
        x_bot = X[bot_feat].values
        y_bot = shap_vals[:, bot_idx]

        # Bottom axis — species diversity
        ax.hexbin(x_bot, y_bot, gridsize=50, cmap='inferno', alpha=0.7, mincnt=1)
        x_smooth_bot, y_smooth_bot = eval.smooth_pdp(x_bot, y_bot, window=200)
        ax.plot(x_smooth_bot, y_smooth_bot, color=colors_bottom, linewidth=2, zorder=5)
        ax.set_xlabel(bot_feat, color=colors_bottom)
        ax.tick_params(axis='x', colors=colors_bottom)
        ax.spines['bottom'].set_color(colors_bottom)

        # Top axis — functional diversity
        ax_top = ax.twiny()
        ax_top.hexbin(x_top, y_top, gridsize=50, cmap='viridis', alpha=0.7, mincnt=1)
        x_smooth_top, y_smooth_top = eval.smooth_pdp(x_top, y_top, window=200)
        ax_top.plot(x_smooth_top, y_smooth_top, color=colors_top, linewidth=2, zorder=5)
        ax_top.set_xlabel(top_feat, color=colors_top)
        ax_top.tick_params(axis='x', colors=colors_top)
        ax_top.spines['top'].set_color(colors_top)

        ax.set_ylabel("SHAP Value (influence on Stability)")
        ax.set_title(title, pad=25)
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax.spines['right'].set_visible(False)
        ax_top.spines['right'].set_visible(False)

    plt.suptitle("Functional vs Species Diversity — Influence on Stability",
                fontsize=13, y=1.02)
    # plt.savefig("shap_density.png", dpi=150, bbox_inches="tight")
    plt.show()