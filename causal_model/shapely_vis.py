import math
import yaml
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D

sys.path.append(str(Path(__file__).resolve().parents[1]))

import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import utils.cross_validation as cval

try:
    from statsmodels.nonparametric.smoothers_lowess import lowess
    HAS_LOWESS = True
except ImportError:
    HAS_LOWESS = False

def _trend_line(x, y, frac=0.4, n_bins=20):
    """Return smoothed (x_sorted, y_smooth) via LOWESS, or a binned-mean
    fallback if statsmodels isn't available."""
    order = np.argsort(x)
    x_sorted, y_sorted = x[order], y[order]

    if HAS_LOWESS:
        smoothed = lowess(y_sorted, x_sorted, frac=frac, return_sorted=True)
        return smoothed[:, 0], smoothed[:, 1]

    # Fallback: binned mean
    bins = np.linspace(x_sorted.min(), x_sorted.max(), n_bins + 1)
    bin_idx = np.digitize(x_sorted, bins) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    x_binned, y_binned = [], []
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() > 0:
            x_binned.append(x_sorted[mask].mean())
            y_binned.append(y_sorted[mask].mean())
    return np.array(x_binned), np.array(y_binned)


# Essential Params and Dicts
def make_sequential_group_colors(groups, cmap_name="Greens", low=0.25, high=0.95):
    """
    Generate a dict of {group: hex_color} using a sequential colormap.
    low/high trim the very lightest and darkest ends so colors stay
    visible against a white background and distinct from black text/lines.
    """
    cmap = plt.get_cmap(cmap_name)
    n = len(groups)
    colors = [cmap(low + (high - low) * i / (n - 1)) for i in range(n)]
    return {g: mcolors.to_hex(c) for g, c in zip(groups, colors)}


def plot_shap_with_line_stats_v4(
    df,
    features,
    color="#56B4E9",
    line_color="#65625F",
    y_axis_limits=None,
    shap_prefix="shap_",
    group=None,
    n_cols=5,
    lowess_frac=0.4,
    name=None,
):
    """
    Plot SHAP dependence with a LOWESS trend line, from a wide-format
    dataframe where each feature has a matching shap_<feature> column.
    Slope and zero-crossing are derived from the LOWESS curve rather
    than a separate linear regression.
    """
    slopes = []
    zeros = []
    n_plots = len(features)
    n_rows = math.ceil(n_plots / n_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.6 * n_cols, 4.2 * n_rows),
        squeeze=False
    )
    axes = axes.reshape(-1)

    for ax, var in zip(axes, features):
        shap_col = f"{shap_prefix}{var}"

        if var not in df.columns or shap_col not in df.columns:
            print(f"Skipping {var}: column(s) not found ({var}, {shap_col})")
            ax.set_visible(False)
            slopes.append(np.nan)
            zeros.append(np.nan)
            continue

        feature_values = df[var].values.astype(float)
        shap_feature = df[shap_col].values.astype(float)

        valid = ~(np.isnan(feature_values) | np.isnan(shap_feature))
        feature_values, shap_feature = feature_values[valid], shap_feature[valid]

        if len(feature_values) < 5 or np.all(feature_values == feature_values[0]):
            ax.set_visible(False)
            slopes.append(np.nan)
            zeros.append(np.nan)
            continue

        # --- LOWESS fit ---
        x_smooth, y_smooth, y_fitted_at_x = _lowess_fit(
            feature_values, shap_feature, frac=lowess_frac
        )
        slope, x_zero = _lowess_slope_and_zero(x_smooth, y_smooth)
        slopes.append(slope)
        zeros.append(x_zero)

        # R² of the LOWESS fit against the actual data
        ss_res = np.sum((shap_feature - y_fitted_at_x) ** 2)
        ss_tot = np.sum((shap_feature - shap_feature.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

        # --- Y limits ---
        if y_axis_limits:
            ax.set_ylim(*y_axis_limits)
        else:
            y_min, y_max = shap_feature.min(), shap_feature.max()
            pad = (y_max - y_min) * 0.08 if y_max > y_min else 0.01
            ax.set_ylim(y_min - pad, y_max + pad)

        # Points
        ax.scatter(
            feature_values,
            shap_feature,
            alpha=0.12,
            s=8,
            color=color,
            linewidth=0,
            zorder=1,
            rasterized=True,
        )

        # LOWESS trend line
        ax.plot(
            x_smooth,
            y_smooth,
            color=line_color,
            linestyle="--",
            linewidth=2.4,
            zorder=4,
            label=f"LOWESS (R²={r2:.2f}, slope={slope:.4f})",
        )

        ax.axhline(0, color="#333333", linestyle=":", linewidth=0.8, alpha=0.5, zorder=2)

        # Clean spines + gridlines
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        ax.grid(axis='y', color='#eeeeee', linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)

        ax.set_xlabel(var, fontweight='medium')
        ax.set_ylabel("SHAP value")
        ax.set_title(var, fontweight='bold', fontsize=11, pad=8)
        ax.legend(fontsize=7.5, frameon=False, loc="best")

    for ax in axes[n_plots:]:
        ax.set_visible(False)

    title = f"SHAP Dependence — {group}" if group is not None else "SHAP Dependence"
    fig.suptitle(title, y=1.02, fontsize=14, fontweight='bold')
    fig.subplots_adjust(hspace=0.55, wspace=0.35)

    if name:
        fig.savefig(name, dpi=300, bbox_inches="tight")

    return slopes, zeros
def _lowess_fit(x, y, frac=0.4, n_bins=20):
    """
    Fit a LOWESS smooth (or binned-mean fallback) and return:
    - x_sorted, y_smooth: the smooth curve for plotting
    - y_fitted_at_x: smooth values aligned back to the original x order,
      for R² and slope calculations against the actual data
    """
    order = np.argsort(x)
    x_sorted = x[order]

    if HAS_LOWESS:
        smoothed = lowess(y[order], x_sorted, frac=frac, return_sorted=True)
        x_smooth, y_smooth = smoothed[:, 0], smoothed[:, 1]
        # aligned fitted values at original x, for R² / slope stats
        y_fitted_at_x = lowess(y, x, frac=frac, return_sorted=False)
        return x_smooth, y_smooth, y_fitted_at_x

    # Fallback: binned mean, then interpolate back onto original x
    bins = np.linspace(x_sorted.min(), x_sorted.max(), n_bins + 1)
    bin_idx = np.clip(np.digitize(x_sorted, bins) - 1, 0, n_bins - 1)
    x_binned, y_binned = [], []
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() > 0:
            x_binned.append(x_sorted[mask].mean())
            y_binned.append(y[order][mask].mean())
    x_binned, y_binned = np.array(x_binned), np.array(y_binned)
    y_fitted_at_x = np.interp(x, x_binned, y_binned)
    return x_binned, y_binned, y_fitted_at_x



def _lowess_slope_and_zero(x_smooth, y_smooth):
    """
    Summarize a (possibly nonlinear) LOWESS curve with a single slope
    (least-squares line through the smoothed curve) and its zero-crossing.
    If the curve crosses zero multiple times, returns the first crossing
    found by interpolation; falls back to the linear estimate otherwise.
    """
    slope, intercept = np.polyfit(x_smooth, y_smooth, 1)

    # Look for an actual sign change along the smoothed curve first —
    # more faithful to a nonlinear LOWESS than the straight-line estimate.
    sign_changes = np.where(np.diff(np.sign(y_smooth)) != 0)[0]
    if len(sign_changes) > 0:
        i = sign_changes[0]
        x0, x1 = x_smooth[i], x_smooth[i + 1]
        y0, y1 = y_smooth[i], y_smooth[i + 1]
        x_zero = x0 + (0 - y0) * (x1 - x0) / (y1 - y0)
    elif slope != 0:
        x_zero = -intercept / slope
    else:
        x_zero = np.nan

    return slope, x_zero


def plot_shap_by_continuous(
    df,
    features,
    color_var,
    cmap="viridis",
    color_var_label=None,
    y_axis_limits=None,
    shap_prefix="shap_",
    group=None,
    n_cols=5,
    name=None,
    vmin=None,
    vmax=None,
    lowess_frac=0.4,
    trend_color="#65625F",
):
    n_plots = len(features)
    n_rows = math.ceil(n_plots / n_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.6 * n_cols, 4.2 * n_rows),
        squeeze=False
    )
    axes = axes.reshape(-1)

    # Shared color scale across all panels, so colors are comparable subplot-to-subplot
    if color_var not in df.columns:
        raise ValueError(f"color_var '{color_var}' not found in df.columns")
    
    c_all = df[color_var].values.astype(float)
    c_valid = c_all[~np.isnan(c_all)]
    vmin = c_valid.min() if vmin is None else vmin
    vmax = c_valid.max() if vmax is None else vmax
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    last_scatter = None

    for ax, var in zip(axes, features):
        shap_col = f"{shap_prefix}{var}"

        if var not in df.columns or shap_col not in df.columns:
            print(f"Skipping {var}: column(s) not found ({var}, {shap_col})")
            ax.set_visible(False)
            continue

        feature_values = df[var].values.astype(float)
        shap_feature = df[shap_col].values.astype(float)
        c_values = df[color_var].values.astype(float)

        valid = ~(np.isnan(feature_values) | np.isnan(shap_feature) | np.isnan(c_values))
        feature_values, shap_feature, c_values = (
            feature_values[valid], shap_feature[valid], c_values[valid]
        )

        if len(feature_values) < 5 or np.all(feature_values == feature_values[0]):
            ax.set_visible(False)
            continue

        # Points colored by the continuous variable
        last_scatter = ax.scatter(
            feature_values,
            shap_feature,
            c=c_values,
            cmap=cmap,
            norm=norm,
            alpha=0.1,
            s=6,
            linewidth=0,
            zorder=1,
            rasterized=True,  # keeps vector PDFs from bloating with dense scatter
        )

        # Single global trend line (pooled, ignores color_var)
        x_smooth, y_smooth = _trend_line(feature_values, shap_feature, frac=lowess_frac)
        ax.plot(
            x_smooth, y_smooth,
            color=trend_color,
            linewidth=2.4,
            zorder=4,
            solid_capstyle='round',
        )

        feature_ylim = None
        if y_axis_limits is not None:
            feature_ylim = y_axis_limits.get(var)

        if feature_ylim:
            ax.set_ylim(*feature_ylim)
        else:
            y_min, y_max = shap_feature.min(), shap_feature.max()
            pad = (y_max - y_min) * 0.08 if y_max > y_min else 0.01
            ax.set_ylim(y_min - pad, y_max + pad)

        ax.axhline(0, color="#333333", linestyle="--", linewidth=0.8, alpha=0.5, zorder=2)

        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        ax.grid(axis='y', color='#eeeeee', linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)

        ax.set_xlabel(var, fontweight='medium')
        ax.set_ylabel("SHAP value")
        ax.set_title(var, fontweight='bold', fontsize=11, pad=8)

    for ax in axes[n_plots:]:
        ax.set_visible(False)

    # Shared colorbar for the continuous variable
    if last_scatter is not None:
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(
            sm, ax=axes[:n_plots].tolist(),
            orientation='vertical',
            fraction=0.02, pad=0.06,
            location='right'
        )
        cbar.set_label(color_var_label or color_var, fontsize=10)
        cbar.ax.tick_params(labelsize=8)
    # Trend-line legend entry (kept separate from the colorbar)
    legend_elements = [
        Line2D([0], [0], color=trend_color, lw=2.4, label="Overall trend (LOWESS)")
    ]
    fig.legend(
        handles=legend_elements,
        loc='upper right',
        bbox_to_anchor=(0.99, 1.0),
        frameon=False,
        fontsize=10,
    )

    fig.suptitle("Overall SHAP Dependence Plots", y=1.1, fontsize=14, fontweight='bold')
    fig.subplots_adjust(hspace=0.55, wspace=0.35, top=0.85)

    if name:
        fig.savefig(name, dpi=300, bbox_inches="tight")

    return fig


# shap_df = pd.read_csv("results/shap_values_Shannon_Diversity.csv")

# wsci= pd.read_csv("data/processed/PID_location_WSCI.csv")

# shap_df = shap_df.merge(wsci[['PID', 'WSCI']], on='PID', how='left')

# shap_cols = [c for c in shap_df.columns if c.startswith('shap_')]


# features=['percent_conifer', 'Annual Temp', 'CHELSA_exBIO_GrowingSeasonLength', 
#           'Precipitation Seasonality', 'CGIAR_Aridity_Index']

import os
from pathlib import Path

folder = "results/shap_values/"  # change to your folder path

y_axis_limits = {
    'percent_conifer': (-0.5, 0.5),
    'Annual Temp': (-0.15, 0.15),
    'CHELSA_exBIO_GrowingSeasonLength': (-0.15, 0.15),
    'Precipitation Seasonality': (-0.2, 0.2),
    'CGIAR_Aridity_Index': (-0.05, 0.05),
}

features=['Precipitation Seasonality', 'Elevation', 
        'Stand Age', 'PET sd', 'Annual Temp']



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


fd_df= pd.read_csv("data/final/final_dataset_ba_v2.csv")

for filename in os.listdir(folder):
    if "shap_values" in filename:
        # split on 'shap_values_' and take what's after it
        name = filename.split("shap_values_", 1)[1]
        # strip the file extension too, if you don't want it
        name = Path(name).stem
        print(name)

        shap_df = pd.read_csv(f"{folder}{filename}")
        shap_df = shap_df.merge(fd_df[['PID', 'Species Richness']], on='PID', how='left')
        # wsci= pd.read_csv("data/processed/PID_location_WSCI.csv")
        # shap_df = shap_df.merge(wsci[['PID', 'WSCI']], on='PID', how='left')

        # PID_df= pd.read_csv('data/lookup/PID_location_v2.csv')

        # shap_df = shap_df.merge(PID_df[['PID', 'biome']], on='PID', how='left')

        shap_cols = [c for c in shap_df.columns if c.startswith('shap_')]

        # plot_shap_with_line_stats_v4(
        #     df=shap_df,
        #     features=features,
        #     color="#56B4E9",
        #     line_color="#65625F",
        #     y_axis_limits=y_axis_limits,
        #     shap_prefix="shap_",
        #     group=None,
        #     n_cols=5,
        #     lowess_frac=0.4,
        #     name=f"results2/{name}_shap_dependence.png",
        # )

        plot_shap_by_continuous(
        df=shap_df,
        features=features,
        color_var="Species Richness",
        cmap="Blues",
        color_var_label="Species Richness",
        y_axis_limits=y_axis_limits,
        name=f"results2/{name}_by_Species_Richness.png",
        )
