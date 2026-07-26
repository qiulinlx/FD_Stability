# Standard library
import random
import matplotlib.pyplot as plt
# Data manipulation
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

import math

# Custom utilities
from sklearn.preprocessing import StandardScaler

import utils.cross_validation as cval
import utils.umap_utils as uutil
import utils.exp_utils as exp
import xgboost as xgb
import yaml
import shap
from umap import UMAP


def plot_shap_beeswarm(shap_values, output_path="global_shap_beeswarm.png"):
    fig, axes = plt.subplots(2, 1, figsize=(10, 12))

    # Top: signed SHAP values
    shap.plots.beeswarm(
        shap_values,
        color="orchid",
        max_display=None,
        show=False,
        ax=axes[0]
    )
    axes[0].set_title("SHAP values")

    # Bottom: absolute SHAP values
    shap.plots.beeswarm(
        shap_values.abs,
        color="darkolivegreen",
        max_display=None,
        show=False,
        ax=axes[1]
    )
    axes[1].set_title("|SHAP values|")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":

    color_map = {
        'Species Richness': 'grey',
        'Shannon Diversity': "#8040a0",
        "Raos_Q": 'darkolivegreen',
        "Simpson's Index":  "#cca9dd",
        'Functional_Evenness': "#94BC92"
    }
    
    biome_colors = {
    'Temperate broadleaf forests': '#E67E22',      # Vibrant Orange
    'Temperate conifer forests': '#2E86C1',        # Bright Blue
    'Temperate grasslands': '#F1C40F',             # Bright Yellow
    'Xeric shrublands': '#E74C3C',                 # Bright Red
    'Boreal and Tundra forests': '#1ABC9C',        # Teal
    'Mediterranean woodlands': '#8E44AD',          # Purple
    'Tropical': '#2ECC71',                         # Emerald Green
    'Flooded grasslands': '#3498DB',               # Royal Blue
    'Mangroves': '#27AE60',                        # Dark Green
    'NaN': '#BDC3C7'                               # Light Gray
}


    with open("config/ex_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    biome_mapping = config["biome_mapping"]
    random_key=config["random_key"]

    # biome_list=["Temperate broadleaf forests", "Temperate conifer forests", "Temperate grasslands",
    #             "Xeric shrublands", "Boreal and Tundra forests", "Tropical",
    #             "Mediterranean woodlands"]
    

    fd_df = pd.read_csv('data/final/final_dataset_with_aridity.csv')
    # fd_df.drop(columns=[ 'managed', 'ownership','Functional_Divergences', 'Shannon Equitabiltiy Index', 'DIA'], inplace=True)
    fd_df.dropna(subset=['Raos_Q', 'Functional_Evenness', 'Soil Moisture',
                        'Species Richness', 'Shannon Diversity', "Simpson's Index", 'biome'], inplace=True)
    
    fd_df.drop(columns=['ownership'], inplace=True)
    
    ecoregions=cval.process_ecoregion("data/Ecoregions/Ecoregions2017.shp")

    ecoregions=ecoregions[['ECO_NAME', 'geometry']]

    #Preprocessing the data for Random forest regression
    # fbiome_dfs= data_processing_v2(fd_df, biome_mapping, ecoregions)

    fd_df['biome'] = fd_df['biome'].map(biome_mapping) # Only run this once ! 

    biome_dfs = {k: v for k, v in fd_df.groupby('biome')}

    X=fd_df.drop(columns=['transformed npp', 'PID', 'lat', 'lon', 'biome', 'BHAGE'])
    y=fd_df['transformed npp'].values
    scaler= StandardScaler()
    scaler.fit_transform(X)
    X = pd.DataFrame(
        X,
        columns=scaler.feature_names_in_
    )

    model = xgb.XGBRegressor(**config["global"])
    model.fit(X, y )

    explainer = shap.Explainer(model, approximate=True)

    shap_vals = explainer.shap_values(X, approximate=True)
    shap_values = explainer(X, approximate=True )

    # plot_shap_beeswarm(shap_values, output_path="global_shap_beeswarm.png")

    diversity_vars_f = ['Raos_Q', 'Functional_Evenness', 'WSCI', 'Annual Temp', 'Elevation', "Precipitation Seasonality"]

    diversity_vars_s =  ["Simpson's Index", 'Shannon Diversity', 'WSCI',  'Annual Temp','Elevation', "Precipitation Seasonality"]

    exp.plot_shap_with_polynomial_stats_v2(shap_vals, X, diversity_vars_f, diversity_vars_s, degree=3, name="global_shap_dependence_polynomial.png")

    feature_list = ['Raos_Q', 'Functional_Evenness', "Simpson's Index", 'Shannon Diversity']  

    # Define colors

    exp.map_shap(shap_vals, feature_list, X, fd_df, color_map, name="figures/global_dominant_groups_map.html")

    s_2d = UMAP(
    n_components=2, n_neighbors=30, min_dist=0.01, init='random', low_memory=True, random_state=random_key, n_jobs=-1
    ).fit_transform(shap_vals)

    feature_list = ['Raos_Q',  "Simpson's Index", 
        'Functional_Evenness', 'Shannon Diversity', 
        'Species Richness', 'WSCI']

    feature_list = ['Raos_Q',  "Simpson's Index", 
    'Functional_Evenness', 'Shannon Diversity', 
    'Species Richness', 'WSCI']

    uutil.plot_umap_by_features(s_2d, X, features=feature_list,cmap='viridis', name="figures/global_umap_shap_biotic.png")

    feature_list1 = [feat for feat in X.columns if feat not in feature_list]

    uutil.plot_umap_by_features(s_2d, X, features=feature_list1, cmap='inferno', name="figures/global_umap_shap_abiotic.png")