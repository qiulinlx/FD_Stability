# Standard library
import random

# Data manipulation
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_percentage_error,
    r2_score
)
from sklearn.model_selection import (
    KFold)

# Custom utilities
import utils.cross_validation as cval
from utils.model_utils import data_processing_v2, train_test_split_v2

import threading
import queue

def train_rf_cv(X, y, n_splits=5, rf_params=None):
    if rf_params is None:
        rf_params = {
            'n_estimators': 200,
            'max_depth': None,
            'min_samples_leaf': 2,
            'n_jobs': -1,
            'random_state': 42
        }

    # Fix: reset index to avoid positional/label index mismatch
    X = np.array(X)
    y = np.array(y)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    model = RandomForestRegressor(**rf_params)

    fold_mape, fold_r2, fold_y_val= [], [], []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        X_train, X_val = X[train_idx], X[val_idx]  # use .iloc
        y_train, y_val = y[train_idx], y[val_idx]  # use .iloc

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        mape = mean_absolute_percentage_error(y_val, y_pred, multioutput='raw_values')
        r2  = r2_score(y_val, y_pred, multioutput='raw_values')
        y_mean = np.mean(y_val)
        fold_mape.append(mape*100)
        fold_r2.append(r2)
        fold_y_val.append(y_mean)
        
    fold_mape = np.array(fold_mape)
    fold_r2 = [float(x[0]) if isinstance(x, np.ndarray) else x for x in fold_r2]

    return model, fold_mape, fold_r2, fold_y_val


def scatter_with_color(data, **kwargs):
    # Get the biome name for this facet
    biome_name = data['biome'].iloc[0]
    color = biome_colors.get(biome_name, 'blue')  # Default to blue if not found
    plt.scatter(data['Stability'], data['mape'], 
               color=color, alpha=0.5, s=10)
    

def experiment (biome_list, biome_dfs, random_key, test_size, param_grid):

    result_df=pd.DataFrame(columns=['biome', 'mape', 'r2', 'Stability'])

    for biome in biome_list:
        print(f"Biome: {biome}")

        random.seed(random_key)
        random_keys = [random.randint(0, 500) for _ in range(100)]

        for key in random_keys:
            random_key=key
            n_points= round(len(biome_dfs[biome])*test_size)
            fX_train, fy_train, fX_test, fy_test = train_test_split_v2(fbiome_dfs[biome], random_key=random_key, n_points_to_select=n_points, buffer_km=50)
            
            _, mape, r2, y_val = train_rf_cv(fX_train, fy_train, 
                                             n_splits=10, rf_params=param_grid)

            mape = [item[0] for item in mape]
            r2 = r2
            
            sub_df = pd.DataFrame(
                list(zip([biome]*len(mape), mape, r2, y_val)),
                columns=['biome', 'mape', 'r2', 'Stability']
            )
            result_df=pd.concat([result_df, sub_df], ignore_index=True)
        
    return result_df
    

def experiment_wrapper(name, *args):
    output = experiment(*args)
    results[name] = output



if __name__ == "__main__":

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
        13: 'Unknown'
    }

    biome_colors = {
        'Boreal and Tundra forests': "#62D7E0",        # boreal blue
        'Flooded grasslands': "#4CAF50",                # wetland green
        'Mangroves': "#2E7D32",                        # coastal green
        'Mediterranean woodlands': "#F4A460",          # sandy brown
        'Temperate broadleaf forests': "#1B3111",      # dark temperate green
        'Temperate conifer forests': "#7E34A9",        # purple conifer
        'Temperate grasslands': "#D4A843",             # golden
        'Tropical': "#079331",                         # tropical green
        'Xeric shrublands': "#D2691E",                 # desert brown
    }
    
    RANDOM_KEY = 42
    TEST_SIZE = 0.1
    rf_params = {
            'n_estimators': 200,
            'max_depth': 10,
            'min_samples_leaf': 2,
            'n_jobs': -1,
            'random_state': RANDOM_KEY
        }


    fd_df = pd.read_csv('data/final/final_dataset.csv')

    fd_df.drop(columns=[ 'managed', 'ownership',
        'managed.1'], inplace=True)
    fd_df.dropna(inplace=True)


    ecoregions=cval.process_ecoregion("data/Ecoregions/Ecoregions2017.shp")

    ecoregions=ecoregions[['ECO_NAME', 'geometry']]

    #Preprocessing the data for Random forest regression
    fbiome_dfs= data_processing_v2(fd_df, biome_mapping, ecoregions)

    biomes_list1=['Temperate conifer forests', 'Xeric shrublands',]
    biomes_list2=['Mediterranean woodlands' , 'Temperate broadleaf forests' ]
    biomes_list3=[ 'Tropical', 'Temperate grasslands' ]
    biomes_list4=[ 'Unknown', 'Boreal and Tundra forests']

    result_queue = queue.Queue()
    results = {}

    t1 = threading.Thread(target=experiment_wrapper, args=("t1", biomes_list1, fbiome_dfs, RANDOM_KEY, TEST_SIZE, rf_params))
    t2 = threading.Thread(target=experiment_wrapper, args=("t2", biomes_list2, fbiome_dfs, RANDOM_KEY, TEST_SIZE, rf_params))
    t3 = threading.Thread(target=experiment_wrapper, args=("t3", biomes_list3, fbiome_dfs, RANDOM_KEY, TEST_SIZE, rf_params))
    t4 = threading.Thread(target=experiment_wrapper, args=("t4", biomes_list4, fbiome_dfs, RANDOM_KEY, TEST_SIZE, rf_params))

    t1.start()
    t2.start()
    t3.start()
    t4.start()

    t1.join()
    t2.join()
    t3.join()
    t4.join()

    # Collect results from queue
    while not result_queue.empty():
        name, output = result_queue.get()
        results[name] = output
    df1 = pd.DataFrame(results["t1"])
    df2 = pd.DataFrame(results["t2"])
    df3 = pd.DataFrame(results["t3"])
    df4 = pd.DataFrame(results["t4"])


    results_df=pd.concat([df1, df2], ignore_index=True)
    results_df=pd.concat([results_df, df3], ignore_index=True)
    results_df=pd.concat([results_df, df4], ignore_index=True)
    
    results_df.to_csv("results/cv_mape_values.csv", index=False)


    # Create FacetGrid with scatter plots
    g = sns.FacetGrid(
        results_df, 
        col="biome", 
        col_wrap=3, 
        height=4, 
        sharex=True, 
        sharey=True
    )

    g.map_dataframe(scatter_with_color)

    # Add MAPE=0 line centered in each subplot
    for ax in g.axes.flat:
        # Get y-axis limits
        ymin, ymax = ax.get_ylim()
        
        # Center the y-axis around 0 if needed
        max_abs = max(abs(ymin), abs(ymax))
        ax.set_ylim(-max_abs, max_abs)  # Center at 0
        
        # Add horizontal line at MAPE=0
        ax.axhline(y=0, color='gray', linewidth=2, linestyle='--', alpha=0.7)

    # Set axis labels
    g.set_axis_labels("Stability", "MAPE")

    # Adjust spacing - MORE SPACE between title and subplots
    g.fig.subplots_adjust(top=0.88)  # Increased from 0.9 to 0.88 (more space)

    # Add title with more space
    g.fig.suptitle(
        'Model MAPE vs Stability by Biome', 
        fontsize=16, 
        fontweight='bold',
        y=0.98  # Adjust y position (0.98 = higher, more space below)
    )

    plt.savefig("results/mape_vs_stability_by_biome.png", dpi=300, bbox_inches='tight')