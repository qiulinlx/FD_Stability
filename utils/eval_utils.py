import shap
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
import pandas as pd
from sklearn.inspection import partial_dependence
import matplotlib.pyplot as plt
# Machine learning
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from math import radians, sin, cos, sqrt, asin


import numpy as np 

def SHAP_plots (model, X_test, colour="#4b72f1", plot_name="shap_bar_plot.png"):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test, approximate=True)

    # -----------------------------
    # Beeswarm plot
    # -----------------------------
    # shap.plots.beeswarm(shap_exp, max_display=len(X_test.columns), color=plt.get_cmap("RdYlGn"))
    shap.plots.bar(shap_values.abs.mean(0), max_display=len(X_test.columns))
    ax = plt.gca()
    for bar in ax.patches:
        bar.set_color(colour)

    shap_df = pd.DataFrame(
        shap_values,
        columns=X_test.columns,
        index=X_test.index
    )
    
    if plot_name:
        plt.savefig(plot_name, dpi=300, bbox_inches='tight')
    plt.close()

    return shap_df, explainer

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

def plot_pdp(div_f, div_s, fd_model, sd_model, fX_test, sX_test, name="pdp_plot.png"):

    paired_vars = list(zip(div_f, div_s))

    colors_f = "#0338f6"
    colors_s = "#c88f8f"

    # 2 rows x 3 cols
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes_flat = axes.flatten()  # flatten to iterate easily

    for ax, (f_var, s_var) in zip(axes_flat, paired_vars):

        pd_f = partial_dependence(fd_model, fX_test, features=[f_var], grid_resolution=200)
        pd_s = partial_dependence(sd_model, sX_test, features=[s_var], grid_resolution=200)

        x_f = pd_f["grid_values"][0]
        x_s = pd_s["grid_values"][0]
        y_f = pd_f["average"][0]
        y_s = pd_s["average"][0]

        # ── Bottom x-axis: species diversity ──
        ax.plot(x_s, y_s, color=colors_s, linewidth=2, linestyle="--")
        ax.set_xlabel(s_var)
        ax.tick_params(axis="x")

        # ── Top x-axis: functional diversity ──
        ax_top = ax.twiny()
        ax_top.plot(x_f, y_f, color=colors_f, linewidth=2)
        ax_top.set_xlabel(f_var)
        ax_top.tick_params(axis="x")

        # y label only on leftmost column
        ax.set_ylabel(" Influence on Predicted NPP" if ax in axes[:, 0] else "")
        ax.spines[["right"]].set_visible(False)
        ax_top.spines[["right"]].set_visible(False)

def smooth_pdp(x_vals, shap_vals, window=3):
    """Sort by x and apply rolling mean to smooth."""
    df = pd.DataFrame({'x': x_vals, 'y': shap_vals}).sort_values('x')
    df['y_smooth'] = df['y'].rolling(window, center=True, min_periods=1).mean()
    return df['x'].values, df['y_smooth'].values



def evaluate_xgboost(model, dtrain, dtest, X_test, y_test, feature_names=None, 
                     biome_name=None, show_shap=True, figsize=(12, 14)):
    """
    Comprehensive XGBoost model evaluation with hexbin scatter plot (top) 
    and SHAP feature importance (bottom).
    
    Parameters:
    - model: trained XGBoost model
    - dtrain: training DMatrix (for SHAP explainer)
    - dtest: test DMatrix (for SHAP explainer) 
    - X_test: test features (pandas DataFrame or numpy array)
    - y_test: test targets
    - feature_names: (optional) feature names for SHAP plot
    - biome_name: (optional) name for title
    - show_shap: whether to show SHAP importance
    - figsize: overall figure size (width, height)
    
    Returns:
    - fig: matplotlib figure
    - metrics: dict with MAE, R2, RMSE scores
    """
    
    # Make predictions
    y_pred = model.predict(dtest)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"=== XGBoost Model Performance ===")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    
    # Create figure with 2 rows: scatter plot on top, SHAP below
    if show_shap:
        fig, axes = plt.subplots(2, 1, figsize=figsize, 
                                 gridspec_kw={'height_ratios': [1.2, 1]})
        ax_scatter = axes[0]
        ax_shap = axes[1]
    else:
        fig, ax_scatter = plt.subplots(1, 1, figsize=(figsize[0], figsize[1]//1.5))
    
    # --- HEXBIN SCATTER PLOT (TOP) ---
    hb = ax_scatter.hexbin(y_test, y_pred, gridsize=40, cmap='YlOrRd', mincnt=1)
    cbar = fig.colorbar(hb, ax=ax_scatter, label='Count')
    cbar.ax.tick_params(labelsize=9)
    
    # Perfect fit line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax_scatter.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect fit')
    
    # Add 10% error bands (optional)
    ax_scatter.plot([min_val, max_val], [min_val*1.1, max_val*1.1], 'gray', lw=1, alpha=0.5, linestyle=':')
    ax_scatter.plot([min_val, max_val], [min_val*0.9, max_val*0.9], 'gray', lw=1, alpha=0.5, linestyle=':')
    
    ax_scatter.set_xlabel('Actual', fontsize=11)
    ax_scatter.set_ylabel('Predicted', fontsize=11)
    title = f'XGBoost: Predicted vs Actual'
    if biome_name:
        title += f' - {biome_name}'
    ax_scatter.set_title(title, fontsize=13, fontweight='bold')
    
    # Add metrics text box
    metrics_text = f'MAE: {mae:.4f}\nR²: {r2:.4f}\nRMSE: {rmse:.4f}\nn: {len(y_test)}'
    ax_scatter.text(0.05, 0.95, metrics_text, transform=ax_scatter.transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax_scatter.legend(loc='lower right', fontsize=9)
    ax_scatter.grid(True, alpha=0.3)
    
    # --- SHAP FEATURE IMPORTANCE (BOTTOM) ---
    if show_shap:
        try:
            import shap
            
            # Create SHAP explainer
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(dtest)
            
            # Get feature names
            if feature_names is None:
                if hasattr(X_test, 'columns'):
                    feature_names = X_test.columns.tolist()
                else:
                    feature_names = [f'Feature_{i}' for i in range(X_test.shape[1])]
            
            # Mean absolute SHAP values
            shap_mean = np.abs(shap_values).mean(axis=0)
            
            # Sort by importance
            sorted_idx = np.argsort(shap_mean)[::-1]
            top_n = min(15, len(feature_names))  # Reduced to 15 for better visibility
            
            # Create bar plot
            colors = plt.cm.Blues(np.linspace(0.4, 0.9, top_n))[::-1]
            bars = ax_shap.barh(range(top_n), shap_mean[sorted_idx[:top_n]], color=colors)
            
            # Customize
            ax_shap.set_yticks(range(top_n))
            ax_shap.set_yticklabels([feature_names[i] for i in sorted_idx[:top_n]], fontsize=10)
            ax_shap.set_xlabel('Mean |SHAP value|', fontsize=11)
            ax_shap.set_title('SHAP Feature Importance', fontsize=13, fontweight='bold')
            ax_shap.invert_yaxis()
            ax_shap.grid(True, alpha=0.3, axis='x')
            
            # Add value labels
            for i, (bar, val) in enumerate(zip(bars, shap_mean[sorted_idx[:top_n]])):
                ax_shap.text(val + val*0.02, i, f'{val:.3f}', va='center', fontsize=8)
                
        except Exception as e:
            ax_shap.text(0.5, 0.5, f'SHAP Error:\n{str(e)}', 
                        ha='center', va='center', transform=ax_shap.transAxes,
                        fontsize=12)
            print(f"Warning: SHAP plot failed - {e}")
    
    plt.tight_layout()
    plt.show()
    
    # Return metrics
    metrics = {
        'MAE': mae,
        'R2': r2,
        'RMSE': rmse,
        'n_samples': len(y_test)
    }
    
    return fig, metrics


def haversine(lat1, lon1, lat2, lon2):
    """Calculate great-circle distance between two points in km"""
    R = 6371  # Earth's radius in km
    
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    
    return R * c
