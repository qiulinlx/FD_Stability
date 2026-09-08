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


def plot_shap_by_continuous(
    df,
    features,
    color_var,               # name of the continuous column to color points by
    cmap="viridis",
    shap_prefix="shap_",
    n_cols=5,
    name=None,
    title=None,
    lowess_frac=0.4,
    trend_color="#D55E00",   # single global trend line color (contrasts with most cmaps)
    color_var_label=None,    # optional display label for the colorbar
    vmin=None,
    vmax=None,
):
    """
    Plot SHAP dependence with points colored by a continuous variable
    (shared colormap + colorbar across all panels), and a single pooled
    trend line per feature (fit across all data, ignoring color_var).
    """
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
            alpha=0.2,
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

        ax.set_ylim(-0.03, 0.03)
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
            orientation='horizontal',
            fraction=0.02, pad=0.06,
            location='top'
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

    fig.suptitle(title or "Overall SHAP Dependence Plots", y=1.1, fontsize=14, fontweight='bold')
    fig.subplots_adjust(hspace=0.55, wspace=0.35, top=0.85)

    if name:
        fig.savefig(name, dpi=300, bbox_inches="tight")

    return fig


shap_df = pd.read_csv("results/shap_values_Species_Richness.csv")

wsci= pd.read_csv("data/processed/PID_location_WSCI.csv")

shap_df = shap_df.merge(wsci[['PID', 'WSCI']], on='PID', how='left')

# shap_df # =shap_df.merge(fd_df[['PID', 'lat', 'lon', 'WSCI']], on='PID', how='left')
shap_cols = [c for c in shap_df.columns if c.startswith('shap_')]
features = [c.replace('shap_', '', 1) for c in shap_cols]

plot_shap_by_continuous(
df=shap_df,
features=features,
color_var="WSCI",
cmap="viridis",
color_var_label="WSCI",
name="results/species_richness_by_WSCI.png",
)
