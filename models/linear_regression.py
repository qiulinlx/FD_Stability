# Standard library
import random
import matplotlib.pyplot as plt

# Data manipulation
import numpy as np
import pandas as pd

# Visualization
import seaborn as sns

# Machine learning
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score
)

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import KFold

import math

# Custom utilities
from sklearn.preprocessing import StandardScaler

import utils.cross_validation as cval

from math import radians, sin, cos, sqrt, asin


# Essential Params and Dicts

color_map = {
    'Species Richness': 'grey',
    'Shannon Diversity': "#8040a0",
    "Raos_Q": 'darkolivegreen',
    "Simpson's Index":  "#cca9dd",
    'Functional_Evenness': "#94BC92"
}
biome_colors = {
    'Temperate broadleaf forests': "#4881BE",   # Sea Green (mid-tone green)
    'Temperate conifer forests':   "#016742",   # Very Dark Green (near-black green)
    'Temperate grasslands':        '#DAA520',   # Goldenrod (yellow-brown)
    'Xeric shrublands':             '#B5651D',   # Burnt Orange-Brown (desert)
    'Boreal and Tundra forests':    '#4169E1',   # Royal Blue (cool, distinct from green)
    'Mediterranean woodlands':      '#9ACD32',   # Yellow-Green (olive, warm-leaning)
    'Tropical':                     '#FF00FF',   # Magenta (maximally distinct, non-natural but unmistakable)
    'NaN':                          '#D3D3D3'    # Light Gray
}



biome_list=["Temperate broadleaf forests", "Temperate conifer forests", "Temperate grasslands",
            "Xeric shrublands", "Boreal and Tundra forests", "Tropical",
            "Mediterranean woodlands"]



def select_one_point_remove_buffer(df,
                                    buffer_km=50,
                                    lat_col='lat', lon_col='lon',
                                    random_seed=None):
    """
    Select exactly ONE point and remove ALL other points within buffer_km.

    Parameters:
    - df: DataFrame with coordinates
    - buffer_km: radius in km to remove points (default: 50)
    - lat_col: latitude column name
    - lon_col: longitude column name
    - random_seed: random seed for reproducibility

    Returns:
    - selected_df: DataFrame with 1 selected point
    - remaining_df: DataFrame with points > buffer_km away (kept)
    - removed_df: DataFrame with points <= buffer_km away (removed)
    """

    if random_seed is not None:
        np.random.seed(random_seed)

    selected_idx = np.random.choice(df.index, 1, replace=False)[0]

    # Get selected point coordinates
    selected_point = df.loc[selected_idx]
    lat_sel = selected_point[lat_col]
    lon_sel = selected_point[lon_col]

    # Classify all points
    selected_points = [selected_idx]
    removed_points = []
    remaining_points = []

    for idx in df.index:
        if idx == selected_idx:
            continue

        # Calculate distance to selected point
        dist = haversine(lat_sel, lon_sel,
                          df.loc[idx, lat_col],
                          df.loc[idx, lon_col])

        if dist <= buffer_km:
            removed_points.append(idx)  # Too close → remove
        else:
            remaining_points.append(idx)  # Far enough → keep

    # Create DataFrames
    selected_df = df.loc[selected_points].copy()
    removed_df = df.loc[removed_points].copy()
    remaining_df = df.loc[remaining_points].copy()

    print(f"✓ Removed {len(removed_df)} points within {buffer_km}km")

    return selected_df, remaining_df, removed_df


def train_test_split_v3(df, random_key, n_points_to_select=2, buffer_km=100):

    selected, remaining, removed = select_one_point_remove_buffer(
        df,
        buffer_km=50,
        lat_col='lat', lon_col='lon',
        random_seed=random_key
    )

    test = selected.drop(columns=['lat', 'lon', 'biome', 'treecover2000'])
    test_biomes = selected[['biome']]
    train = remaining.drop(columns=['lat', 'lon', 'biome', 'treecover2000'])

    # --- Target changed to Species Richness ---
    y = ['Simpson\'s Index']  # Change target variable here
    X_train = train.drop(columns=y + ['PID'])
    y_train = train['Simpson\'s Index'].values

    X_test = test.drop(columns=y + ['PID'])
    y_test = test['Simpson\'s Index'].values

    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    return X_train, y_train, X_test, y_test, test_biomes, scaler


def haversine(lat1, lon1, lat2, lon2):
    """Calculate great-circle distance between two points in km"""
    R = 6371  # Earth's radius in km

    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))

    return R * c

if __name__ == "__main__":

    y=  "Simpson's Index"
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

    biome_list = ["Temperate broadleaf forests", "Temperate conifer forests", "Temperate grasslands",
                    "Xeric shrublands", "Boreal and Tundra forests", "Tropical",
                    "Mediterranean woodlands"]

    fd_df = pd.read_csv('data/final/final_dataset.csv')
    PID_df = pd.read_csv('data/lookup/PID_location_all.csv')

    fd_df = fd_df.merge(PID_df[['PID', 'lat', 'lon', 'biome']], on='PID', how='left')

    npp_df = pd.read_csv('data/processed/PID_npp_volatility.csv')
    npp_df = npp_df[['PID', 'mean']]

    fd_df = fd_df.merge(npp_df, on='PID', how='left')
    fd_df = fd_df[~fd_df["biome"].isin(["Mangroves", "Flooded grasslands"])]
    fd_df.drop_duplicates(subset=['PID'], inplace=True)

    fd_df.dropna(subset=['transformed npp', 'Raos_Q', 'Functional_Evenness', 'Soil Moisture',
                            'Species Richness', 'Shannon Diversity', "Simpson's Index", 'biome', "pet_std"],
                    inplace=True)

    fd_df.drop(columns=['Functional_Richness','Species Richness', 'Functional_Evenness', 'Raos_Q', "Shannon Diversity", 'WSCI', 'transformed npp'], inplace=True)

    fd_df = fd_df[fd_df["treecover2000"] > 30]

    ecoregions = cval.process_ecoregion("data/Ecoregions/Ecoregions2017.shp")
    ecoregions = ecoregions[['ECO_NAME', 'geometry']]

    fd_df['biome'] = fd_df['biome'].map(biome_mapping)  # Only run this once!

    biome_dfs = {k: v for k, v in fd_df.groupby('biome')}

    random_keys = random.sample(range(1, 100000), 200)  # Generate n random keys for reproducibility


    y_pred_all = []
    y_actual_all = []
    for i in range(len(random_keys)):

        random_key = random_keys[i]
        fX_train, fy_train, fX_test, fy_test, test_biomes, scaler = train_test_split_v3(
            fd_df, random_key=random_key, n_points_to_select=1, buffer_km=50
        )

        # --- Random Forest model ---
        model =  RandomForestRegressor(max_depth=6, n_jobs=-1, random_state=0)
        model.fit(fX_train, fy_train)

        # Make predictions
        y_pred = model.predict(fX_test)

        y_pred_all.append(y_pred)
        y_actual_all.append(fy_test)

        print(f"completed iteration {i + 1}")

    df = pd.DataFrame({
        'y_pred_all': y_pred_all,
        'y_actual_all': y_actual_all,
    })

    # If each value is a list like [1], [2], [3]
    # Extract the first (and only) element from each list
    df['y_actual_all'] = df['y_actual_all'].apply(lambda x: x[0])
    df['y_pred_all'] = df['y_pred_all'].apply(lambda x: x[0])

    # Clean the data before plotting
    df_clean = df[['y_actual_all', 'y_pred_all']].copy()

    # Force numeric conversion
    df_clean['y_actual_all'] = pd.to_numeric(df_clean['y_actual_all'], errors='coerce')
    df_clean['y_pred_all'] = pd.to_numeric(df_clean['y_pred_all'], errors='coerce')

    figsize = (12, 14)

    fig, ax_scatter = plt.subplots(1, 1, figsize=(figsize[0], figsize[1] // 1.5))

    hb = ax_scatter.scatter(df_clean['y_actual_all'],
                                df_clean['y_pred_all'],
                                color='darkolivegreen',
                                alpha=0.6,
                                s=30)

    # Perfect fit line
    min_val = min(df['y_actual_all'].min(), df['y_pred_all'].min())
    max_val = max(df['y_actual_all'].max(), df['y_pred_all'].max())
    ax_scatter.plot([min_val, max_val], [min_val, max_val], 'darkolivegreen', lw=2, label='Perfect fit')

    # Add 10% error bands (optional)
    ax_scatter.plot([min_val, max_val], [min_val * 1.1, max_val * 1.1], 'gray', lw=1, alpha=0.5, linestyle=':')
    ax_scatter.plot([min_val, max_val], [min_val * 0.9, max_val * 0.9], 'gray', lw=1, alpha=0.5, linestyle=':')

    ax_scatter.set_xlabel('Actual', fontsize=15)
    ax_scatter.set_ylabel('Predicted', fontsize=15)
    title = f'Random Forest: Predicted vs Actual ({y})'
    ax_scatter.set_title(title, fontsize=20, fontweight='bold')

    r2 = r2_score(df_clean['y_actual_all'], df_clean['y_pred_all'])
    mape = mean_absolute_percentage_error(df_clean['y_actual_all'], df_clean['y_pred_all']) * 100

    # --- ADD TEXT BOX AT BOTTOM ---
    metrics_text = f'R² = {r2:.4f}\nMAPE = {mape:.2f}%'
    ax_scatter.text(0.8, 0.1, metrics_text,
                        transform=ax_scatter.transAxes,
                        fontsize=12,
                        verticalalignment='top',
                        horizontalalignment='center',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

    plt.tight_layout()
    plt.savefig(f'results/random_forest_{y}_predicted_vs_actual.png', dpi=300)