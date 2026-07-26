# Standard library
import matplotlib.pyplot as plt
# Data manipulation
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import math
from matplotlib.patches import Patch
import shap

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
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()


def shap_dependence_hexbin(feature_name, shap_values, X, figsize=(10, 6)):
    """
    Hexbin plot showing density of points in 2D space
    """
    # Get values
    if isinstance(X, pd.DataFrame):
        feature_vals = X[feature_name].values
        feature_idx = list(X.columns).index(feature_name)
    else:
        feature_vals = X[:, feature_idx]
    
    shap_vals_feature = shap_values[:, feature_idx].astype(np.float64)

    # Create hexbin plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # hb = ax.hexbin(feature_vals, shap_vals_feature, 
    #                gridsize=100, 
    #                cmap='viridis',
    #                mincnt=1,
    #                edgecolors='none')
    
    hb=ax.scatter(feature_vals, shap_vals_feature, 
                        alpha=1, s=20, c='lavender', edgecolors='black', linewidth=0.3)
    
    
    cbar = plt.colorbar(hb, ax=ax)
    cbar.set_label('Count', fontsize=12)
    
    ax.set_xlabel(feature_name, fontsize=12)
    ax.set_ylabel('SHAP value', fontsize=12)
    ax.set_title(f'SHAP Dependence Plot (Hexbin Density)\nFeature: {feature_name}', 
                fontsize=14)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    return fig, ax, shap_vals_feature, feature_vals

def plot_shap_with_polynomial_stats_v2(shap_values, X, div_f, div_s, degree=2, name=None):
    """Plot SHAP dependence with polynomial trend and statistics"""
    
    paired_vars = list(zip(div_f, div_s))

    colors_f = "#125F10"
    colors_f1= "#94BC92"
    colors_s = "#8040a0"
    colors_s1= "#cca9dd"

    # Get data
    n_plots = len(div_f)
    n_cols = 2  # or any number you prefer
    n_rows = math.ceil(n_plots / n_cols)

    if shap_values.shape[0] < 500:
        n_samples = shap_values.shape[0]
    else:
        n_samples = 2000

    # 2 rows x 2 cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5))
    axes_flat = axes.flatten()  # flatten to iterate easily
    fig.subplots_adjust(hspace=0.6)  # increase from default ~0.2

    for ax, (f_var, s_var) in zip(axes_flat, paired_vars):

        if f_var == s_var:
             # ── Bottom x-axis: species diversity ──
            feature_idx = list(X.columns).index(s_var)
            shap_feature = shap_values[:, feature_idx].astype(np.float64)
            feature_values = X[s_var].values.astype(np.float64)

            if np.all(feature_values == 0):
                # Skip this axis - do nothing
                pass
            else:
                # Fit polynomial
                coeffs = np.polyfit(feature_values, shap_feature, degree)
                poly_func = np.poly1d(coeffs)
            
                # Calculate R²
                y_pred = poly_func(feature_values)
                ss_res = np.sum((shap_feature - y_pred) ** 2)
                ss_tot = np.sum((shap_feature - np.mean(shap_feature)) ** 2)
                r2 = 1 - (ss_res / ss_tot)

                indices = np.random.choice(len(feature_values), n_samples, replace=False)
                feature_values_sample = feature_values[indices]
                shap_feature_sample = shap_feature[indices]

                max_abs = max(abs(ax.get_ylim()[0]), abs(ax.get_ylim()[1]))
                ax.set_ylim(-max_abs, max_abs)

                # # Scatter plot
                ax.scatter(feature_values_sample, shap_feature_sample, 
                                    alpha=0.5, s=20, c='lavender', linewidth=0.3)
                
            
                # Trend line
                x_smooth = np.linspace(feature_values.min(), feature_values.max(), 100)
                y_smooth = poly_func(x_smooth)
                ax.plot(x_smooth, y_smooth, c='slategray',  linewidth=2, 
                        label=f'Polynomial (degree {degree})\nR² = {r2:.4f}', zorder=10)
                ax.set_xlabel(s_var)
                ax.xaxis.set_label_position('top')  # Move label to top
                ax.tick_params(axis="x")
                ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)

        else:
            # ── Bottom x-axis: species diversity ──
            feature_idx = list(X.columns).index(s_var)
            shap_feature = shap_values[:, feature_idx].astype(np.float64)
            feature_values = X[s_var].values.astype(np.float64)

            # Fit polynomial
            coeffs = np.polyfit(feature_values, shap_feature, degree)
            poly_func = np.poly1d(coeffs)
        
            # Calculate R²
            y_pred = poly_func(feature_values)
            ss_res = np.sum((shap_feature - y_pred) ** 2)
            ss_tot = np.sum((shap_feature - np.mean(shap_feature)) ** 2)
            r2 = 1 - (ss_res / ss_tot)

            indices = np.random.choice(len(feature_values), n_samples, replace=False)
            feature_values_sample = feature_values[indices]
            shap_feature_sample = shap_feature[indices]

            # # Scatter plot
            ax.scatter(feature_values_sample, shap_feature_sample, 
                                alpha=0.2, s=30, c=colors_s1, linewidth=0.3, zorder=1)
            

        
            # Trend line
            x_smooth1 = np.linspace(feature_values.min(), feature_values.max(), 100)
            y_smooth1 = poly_func(x_smooth1)
            ax.set_xlabel(s_var)
            ax.tick_params(axis="x")

            ax.plot(x_smooth1, y_smooth1, c=colors_s,  linewidth=2, 
                label=f'Polynomial (degree {degree})\nR² = {r2:.4f}', zorder=10)


            # ── Top x-axis: functional diversity ──

            ax_top = ax.twiny()
            feature_idx = list(X.columns).index(f_var)
            shap_feature = shap_values[:, feature_idx]
            feature_values = X[f_var].values

            # Fit polynomial
            coeffs = np.polyfit(feature_values, shap_feature, degree)
            poly_func = np.poly1d(coeffs)
        
            # Calculate R²
            y_pred = poly_func(feature_values)
            ss_res = np.sum((shap_feature - y_pred) ** 2)
            ss_tot = np.sum((shap_feature - np.mean(shap_feature)) ** 2)
            r2 = 1 - (ss_res / ss_tot)

            indices = np.random.choice(len(feature_values), n_samples, replace=False)
            feature_values_sample = feature_values[indices]
            shap_feature_sample = shap_feature[indices]


            ax_top.scatter(feature_values_sample, shap_feature_sample, 
                                alpha=0.2, s=30, c=colors_f1, linewidth=0.3, zorder=1)


            # Trend line
            x_smooth = np.linspace(feature_values.min(), feature_values.max(), 100)
            y_smooth = poly_func(x_smooth)



            ax_top.plot(x_smooth, y_smooth, c=colors_f,  linewidth=2, 
                    label=f'Polynomial (degree {degree})\nR² = {r2:.4f}', zorder=10)
            

            y_min = min(np.min(y_smooth), np.min(y_smooth1))
            y_max = max(np.max(y_smooth), np.max(y_smooth1))

            padding = 0.1 * (y_max - y_min)

            ax.set_ylim(y_min - padding, y_max + padding)

            ax_top.set_xlabel(f_var)
            ax_top.tick_params(axis="x")
            
            # max_abs = max(abs(ax.get_ylim()[0]), abs(ax.get_ylim()[1]))
            # ax.set_ylim(-0.01, 0.01)

            # Reference line
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
            # Create legend handles
            legend_elements = [Patch(facecolor=colors_s, alpha=0.6, label='Species Diversity (sdiv)'),
                            Patch(facecolor=colors_f, alpha=0.6, label='Functional Diversity (fdiv)')]

            # Add legend to the figure
            fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=11)

            # Adjust layout to make room for legend
            plt.subplots_adjust(bottom=0.12)
    if name:
        plt.savefig(name, dpi=300, bbox_inches="tight")
    plt.show()


def map_shap(shap_vals, feature_list, X, fd_df, color_map, name="shap_map.html"):

    all_shap_df = pd.DataFrame(
        shap_vals,
        columns=[f'shap_{col}' for col in X.columns],
        index=X.index
    )

    shap_df = all_shap_df[[f'shap_{col}' for col in feature_list]]

    # Merge with PID, lat, lon
    results = pd.concat([
        fd_df[['PID', 'lat', 'lon']],
        shap_df
    ], axis=1)

    shap_cols = [col for col in results.columns if col.startswith('shap_')]

    results['dominant_feature'] = results.apply(get_dominant_feature, axis=1)


    # Create the map
    fig = px.scatter_map(
        results,
        lat='lat',
        lon='lon',
        color='dominant_feature',
        color_discrete_map=color_map,
        hover_data=['PID', 'dominant_feature'],
        title='Spatial Distribution of Dominant SHAP Features',
        zoom=6,
        center={'lat': results['lat'].mean(), 'lon': results['lon'].mean()},
        size=[5]*len(results),  # Consistent point size
        opacity=1
    )


    fig.update_traces(marker=dict(size=5, opacity=0.85))
    fig.update_layout(
        mapbox_style="white-bg",
        mapbox_layers=[
            {
                "below": "traces",
                "sourcetype": "raster",
                "sourceattribution": "Natural Earth",
                "source": [
                    "https://basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}.png"
                ]
            }
        ],
        width=1500, 
        height=900,
        legend=dict(
            title=dict(text="Species vs Functional Div", font=dict(size=16)),
            font=dict(size=14),
            itemsizing='constant'
        )
    )

    # Save to HTML
    fig.write_html(name)

