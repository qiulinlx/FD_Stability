# Standard library
import random
import matplotlib.pyplot as plt

# Data manipulation
import numpy as np
import pandas as pd

# Visualization
import seaborn as sns

# Machine learning
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score
)
from sklearn.model_selection import KFold

import math
from sklearn.preprocessing import StandardScaler
import utils.cross_validation as cval
import xgboost as xgb
import shap
from matplotlib.patches import Patch
from math import radians, sin, cos, sqrt, asin


def select_one_point_remove_buffer(df,
                                    buffer_km=50,
                                    lat_col='lat', lon_col='lon',
                                    random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)

    selected_idx = np.random.choice(df.index, 1, replace=False)[0]
    selected_point = df.loc[selected_idx]
    lat_sel = selected_point[lat_col]
    lon_sel = selected_point[lon_col]

    selected_points = [selected_idx]
    removed_points = []
    remaining_points = []

    for idx in df.index:
        if idx == selected_idx:
            continue
        dist = haversine(lat_sel, lon_sel,
                          df.loc[idx, lat_col],
                          df.loc[idx, lon_col])
        if dist <= buffer_km:
            removed_points.append(idx)
        else:
            remaining_points.append(idx)

    selected_df = df.loc[selected_points].copy()
    removed_df = df.loc[removed_points].copy()
    remaining_df = df.loc[remaining_points].copy()

    return selected_df, remaining_df, removed_df


def train_test_split_v3(df, random_key, n_points_to_select=2, buffer_km=100):
    selected, remaining, removed = select_one_point_remove_buffer(
        df, buffer_km=50, lat_col='lat', lon_col='lon', random_seed=random_key
    )

    test = selected.drop(columns=['lat', 'lon', 'biome', 'treecover2000'])
    test_biomes = selected[['biome']]
    test_pid = selected[['PID']]  # keep track of WHICH point this is
    train = remaining.drop(columns=['lat', 'lon', 'biome', 'treecover2000'])

    y = ['transformed npp']
    X_train = train.drop(columns=y + ['PID'])
    y_train = train['transformed npp'].values

    X_test = test.drop(columns=y + ['PID'])
    y_test = test['transformed npp'].values

    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    return X_train, y_train, X_test, y_test, test_biomes, test_pid, scaler


def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return R * c


if __name__ == "__main__":

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
        np.nan: np.nan,
    }

    fd_df = pd.read_csv('data/final/final_dataset.csv')
    PID_df = pd.read_csv('data/lookup/PID_location_all.csv')
    fd_df.drop(columns=['Functional_Richness', 'WSCI'], inplace=True)
    fd_df = fd_df.merge(PID_df[['PID', 'lat', 'lon', 'biome']], on='PID', how='left')

    npp_df = pd.read_csv('data/processed/PID_npp_volatility.csv')
    npp_df = npp_df[['PID', 'mean']]

    fd_df = fd_df.merge(npp_df, on='PID', how='left')
    fd_df = fd_df[~fd_df["biome"].isin(["Mangroves", "Flooded grasslands"])]
    fd_df.drop_duplicates(subset=['PID'], inplace=True)

    fd_df.dropna(subset=['transformed npp', 'Raos_Q', 'Functional_Evenness', 'Soil Moisture',
                          'Species Richness', 'Shannon Diversity', "Simpson's Index",
                          'biome', "pet_std"], inplace=True)

    fd_df = fd_df[fd_df["treecover2000"] > 30]

    ecoregions = cval.process_ecoregion("data/Ecoregions/Ecoregions2017.shp")
    ecoregions = ecoregions[['ECO_NAME', 'geometry']]

    fd_df['biome'] = fd_df['biome'].map(biome_mapping)
    biome_dfs = {k: v for k, v in fd_df.groupby('biome')}

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
    n_rounds = 5000

    # ---- ROBUSTNESS CONFIG ----
    N_SEEDS = 50          # number of independent seed runs
    N_DRAWS_PER_SEED = 600  # buffered-LOO draws per seed (lower than 1000 for tractability;
                             # bump back up if you have the compute budget)

    all_results = []  # will hold one row per (seed, draw): seed, PID, y_actual, y_pred

    for seed in range(N_SEEDS):
        random.seed(seed)
        random_keys = random.sample(range(1, 100000), N_DRAWS_PER_SEED)

        for i in range(len(random_keys)):
            random_key = random_keys[i]

            fX_train, fy_train, fX_test, fy_test, test_biomes, test_pid, scaler = train_test_split_v3(
                fd_df, random_key=random_key, n_points_to_select=1, buffer_km=50
            )

            dtrain_reg = xgb.DMatrix(fX_train, fy_train, enable_categorical=True)
            dtest_reg = xgb.DMatrix(fX_test, fy_test, enable_categorical=True)

            model = xgb.train(
                params=params,
                dtrain=dtrain_reg,
                num_boost_round=n_rounds,
            )

            y_pred = model.predict(dtest_reg)

            all_results.append({
                'seed': seed,
                'PID': test_pid['PID'].values[0],
                'y_actual': fy_test[0],
                'y_pred': y_pred[0]
            })

        print(f"completed seed {seed+1}/{N_SEEDS}")

    df = pd.DataFrame(all_results)
    df.to_csv('results/global_xgboost_results_multiseed.csv', index=False)

    df['y_actual'] = pd.to_numeric(df['y_actual'], errors='coerce')
    df['y_pred'] = pd.to_numeric(df['y_pred'], errors='coerce')

    # ---- Per-seed metrics for the robustness plot ----
    metrics_rows = []
    for seed, group in df.groupby('seed'):
        r2 = r2_score(group['y_actual'], group['y_pred'])
        mape = mean_absolute_percentage_error(group['y_actual'], group['y_pred']) * 100
        rmse = np.sqrt(mean_squared_error(group['y_actual'], group['y_pred']))
        metrics_rows.append({'seed': seed, 'R2': r2, 'MAPE': mape, 'RMSE': rmse})
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv('results/global_xgboost_seed_metrics.csv', index=False)

    # ============================================================
    # PLOT 1: Predicted vs Actual, pooled across all seeds,
    #         with per-point mean ± std across seeds where a PID
    #         happened to get tested in more than one seed.
    # ============================================================
    figsize = (12, 14)
    fig, ax_scatter = plt.subplots(1, 1, figsize=(figsize[0], figsize[1] // 1.5))

    # Every individual prediction, faint, colored by seed
    palette = sns.color_palette("crest", n_colors=N_SEEDS)
    for seed, group in df.groupby('seed'):
        ax_scatter.scatter(group['y_actual'], group['y_pred'],
                            color=palette[seed], alpha=0.15, s=20)

    # Mean ± std per PID across seeds (robustness signal)
    agg = df.groupby('PID').agg(
        y_actual_mean=('y_actual', 'mean'),
        y_pred_mean=('y_pred', 'mean'),
        y_pred_std=('y_pred', 'std'),
        n_seeds_tested=('y_pred', 'count')
    ).reset_index()

    ax_scatter.errorbar(agg['y_actual_mean'], agg['y_pred_mean'],
                         yerr=agg['y_pred_std'].fillna(0),
                         fmt='o', color='darkolivegreen', alpha=0.6, markersize=4,
                         elinewidth=0.8, capsize=2, label='Mean ± SD across seeds')

    min_val = min(df['y_actual'].min(), df['y_pred'].min())
    max_val = max(df['y_actual'].max(), df['y_pred'].max())
    ax_scatter.plot([min_val, max_val], [min_val, max_val], 'darkolivegreen', lw=2, label='Perfect fit')
    ax_scatter.plot([min_val, max_val], [min_val*1.1, max_val*1.1], 'gray', lw=1, alpha=0.5, linestyle=':')
    ax_scatter.plot([min_val, max_val], [min_val*0.9, max_val*0.9], 'gray', lw=1, alpha=0.5, linestyle=':')

    ax_scatter.set_xlabel('Actual', fontsize=15)
    ax_scatter.set_ylabel('Predicted', fontsize=15)
    ax_scatter.set_title('XGBoost: Predicted vs Actual (pooled across seeds)', fontsize=20, fontweight='bold')

    r2_pooled = r2_score(df['y_actual'], df['y_pred'])
    mape_pooled = mean_absolute_percentage_error(df['y_actual'], df['y_pred']) * 100
    metrics_text = (f'Pooled R² = {r2_pooled:.4f}\nPooled MAPE = {mape_pooled:.2f}%\n'
                     f'Mean seed R² = {metrics_df["R2"].mean():.4f} ± {metrics_df["R2"].std():.4f}')
    ax_scatter.text(0.78, 0.15, metrics_text,
                     transform=ax_scatter.transAxes,
                     fontsize=12,
                     verticalalignment='top',
                     horizontalalignment='center',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

    ax_scatter.legend(loc='upper left', fontsize=10)
    plt.tight_layout()
    plt.savefig('results/global_xgboost_predicted_vs_actual_multiseed.png', dpi=300)
    plt.show()

    # ============================================================
    # PLOT 2: Robustness across seeds — R² and RMSE distributions
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.violinplot(y=metrics_df['R2'], color='darkolivegreen', alpha=0.3, ax=axes[0], inner=None)
    sns.stripplot(y=metrics_df['R2'], color='darkolivegreen', size=6, jitter=0.08, ax=axes[0])
    axes[0].set_ylabel('R² (spatially buffered LOO-CV)')
    axes[0].set_title(f'R² across {N_SEEDS} seeds\nmean = {metrics_df["R2"].mean():.3f} ± {metrics_df["R2"].std():.3f}')

    sns.violinplot(y=metrics_df['RMSE'], color='darkolivegreen', alpha=0.3, ax=axes[1], inner=None)
    sns.stripplot(y=metrics_df['RMSE'], color='darkolivegreen', size=6, jitter=0.08, ax=axes[1])
    axes[1].set_ylabel('RMSE (spatially buffered LOO-CV)')
    axes[1].set_title(f'RMSE across {N_SEEDS} seeds\nmean = {metrics_df["RMSE"].mean():.3f} ± {metrics_df["RMSE"].std():.3f}')

    plt.tight_layout()
    plt.savefig('results/global_xgboost_seed_robustness.png', dpi=300)
    plt.show()