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
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score
)
from sklearn.model_selection import (
    KFold)

# Custom utilities
import utils.cross_validation as cval
from utils.model_utils import data_processing_v2, train_test_split_v2
import shap

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



def quick_eval(model, X_test, y_test, biome_name=None, feature_names=None, 
               show_shap=True, figsize=(16, 10)):
    """
    Comprehensive model evaluation with hexbin scatter plot and SHAP feature importance.
    
    Parameters:
    - model: trained model
    - X_test: test features
    - y_test: test targets
    - biome_name: (optional) name for title
    - feature_names: (optional) feature names for SHAP plot
    - show_shap: whether to show SHAP importance
    - figsize: overall figure size
    
    Returns:
    - fig: matplotlib figure
    - metrics: dict with MAE and R2 scores
    """
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred, multioutput='raw_values')
    r2 = r2_score(y_test, y_pred, multioutput='raw_values')
    
    # Ensure arrays
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)
    n_outputs = y_test.shape[1] if y_test.ndim > 1 else 1
    
    # Determine layout
    if show_shap:
        # 2 rows: hexbin on top, SHAP on bottom
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, n_outputs, height_ratios=[1, 1.2], hspace=0.3)
    else:
        # Only hexbin plots
        fig, axes = plt.subplots(1, n_outputs, figsize=(5 * n_outputs, 4), squeeze=False)
        gs = None
    
    # --- HEXBIN SCATTER PLOTS ---
    if show_shap:
        hex_axes = [[fig.add_subplot(gs[0, i]) for i in range(n_outputs)]]
    else:
        hex_axes = axes
    
    for i in range(n_outputs):
        ax = hex_axes[0][i] if n_outputs > 1 else hex_axes[0][0]
        if n_outputs > 1:
            y_t = y_test[:, i]
            y_p = y_pred[:, i]
        else:
            y_t = y_test
            y_p = y_pred
        
        # Hexbin plot
        hb = ax.hexbin(y_t, y_p, gridsize=40, cmap='YlOrRd', mincnt=1)
        fig.colorbar(hb, ax=ax, label='Count')
        
        # Perfect fit line
        min_val = min(y_t.min(), y_p.min())
        max_val = max(y_t.max(), y_p.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=1.5, label='Perfect fit')
        
        # Labels and title
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title(f'Output {i+1} | MAE: {mae[i]:.3f} | R²: {r2[i]:.3f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # --- SHAP FEATURE IMPORTANCE ---
    if show_shap:
        # Create SHAP explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        
        # Handle feature names
        if feature_names is None:
            feature_names = X_test.columns if hasattr(X_test, 'columns') else [f'Feature {i}' for i in range(X_test.shape[1])]
        
        # Get mean absolute SHAP values
        if isinstance(shap_values, list):
            # Multi-output case - use first output or average
            shap_mean = np.abs(shap_values[0]).mean(axis=0)
        else:
            shap_mean = np.abs(shap_values).mean(axis=0)
        
        # Sort features by importance
        sorted_idx = np.argsort(shap_mean)[::-1]
        top_n = min(20, len(feature_names))
        
        # Create SHAP bar plot in the bottom subplot(s)
        if n_outputs > 1:
            # For multi-output, use the first subplot for SHAP
            ax_shap = fig.add_subplot(gs[1, 0])
            # Clear other bottom subplots
            for i in range(1, n_outputs):
                empty_ax = fig.add_subplot(gs[1, i])
                empty_ax.axis('off')
        else:
            ax_shap = fig.add_subplot(gs[1, 0])
        
        # Create bar plot
        colors = ['#4b72f1'] * top_n
        bars = ax_shap.barh(range(top_n), shap_mean[sorted_idx[:top_n]], color=colors)
        
        # Customize
        ax_shap.set_yticks(range(top_n))
        ax_shap.set_yticklabels([feature_names[i] for i in sorted_idx[:top_n]])
        ax_shap.set_xlabel('Mean |SHAP value|', fontsize=11)
        ax_shap.set_title(f'SHAP Feature Importance{": " + biome_name if biome_name else ""}', 
                         fontsize=13, fontweight='bold')
        ax_shap.invert_yaxis()  # Most important at top
        ax_shap.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, shap_mean[sorted_idx[:top_n]])):
            ax_shap.text(val + val*0.02, i, f'{val:.3f}', va='center', fontsize=8)
    
    # Overall title
    if show_shap:
        title = f'Model Performance & SHAP Importance'
        if biome_name:
            title += f' - {biome_name}'
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.show()
    
    # Return metrics
    metrics = {
        'MAE': mae.tolist() if hasattr(mae, 'tolist') else mae,
        'R2': r2.tolist() if hasattr(r2, 'tolist') else r2,
        'n_outputs': n_outputs
    }
    
    return fig, metrics

fd_df = pd.read_csv('data/final/final_dataset.csv')
fd_df.drop(columns=[ 'managed', 'ownership','Functional_Divergences'
       'managed.1'], inplace=True)

fd_df.dropna(inplace=True)

ecoregions=cval.process_ecoregion("data/Ecoregions/Ecoregions2017.shp")

ecoregions=ecoregions[['ECO_NAME', 'geometry']]

#Preprocessing the data for Random forest regression
fbiome_dfs= data_processing_v2(fd_df, biome_mapping, ecoregions)

biome_list=['Boreal and Tundra forests', 'Temperate conifer forests', 'Xeric shrublands',
            'Mediterranean woodlands' , 'Temperate broadleaf forests',
            'Tropical', 'Temperate grasslands',
            'Unknown', ]


