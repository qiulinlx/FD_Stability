import math
import yaml
import sys
from pathlib import Path
import matplotlib.pyplot as plt

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


def _lowess_slope_and_zero(x_smooth, y_smooth):
    """
    Summarize a (possibly nonlinear) LOWESS curve with a single slope
    (least-squares line through the smoothed curve) and its zero-crossing.
    """
    slope, intercept = np.polyfit(x_smooth, y_smooth, 1)

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
        y_fitted_at_x = lowess(y, x, frac=frac, return_sorted=False)
        return x_smooth, y_smooth, y_fitted_at_x

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

        x_smooth, y_smooth, y_fitted_at_x = _lowess_fit(
            feature_values, shap_feature, frac=lowess_frac
        )
        slope, x_zero = _lowess_slope_and_zero(x_smooth, y_smooth)
        slopes.append(slope)
        zeros.append(x_zero)

        ss_res = np.sum((shap_feature - y_fitted_at_x) ** 2)
        ss_tot = np.sum((shap_feature - shap_feature.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

        if y_axis_limits:
            ax.set_ylim(*y_axis_limits)
        else:
            y_min, y_max = shap_feature.min(), shap_feature.max()
            pad = (y_max - y_min) * 0.08 if y_max > y_min else 0.01
            ax.set_ylim(y_min - pad, y_max + pad)

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

    plt.show()

    return slopes, zeros


def fit_and_get_shap(df, target_col, feature_cols, params, n_rounds):
    """
    Fit an XGBoost model on the full dataset for target_col, then compute
    native XGBoost SHAP contributions for every row. Returns a wide-format
    DataFrame: PID, target_col, feature columns (unscaled), shap_<feature>
    columns, and bias term.
    """
    X = df[feature_cols]
    y = df[target_col].values

    scaler = StandardScaler()
    X_s = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols, index=X.index)

    dmat = xgb.DMatrix(X_s, y, enable_categorical=True)
    model = xgb.train(params=params, dtrain=dmat, num_boost_round=n_rounds)

    # Native XGBoost SHAP: shape (n_rows, n_features + 1), last col = bias term
    contribs = model.predict(dmat, pred_contribs=True)
    shap_values = contribs[:, :-1]
    bias = contribs[:, -1]

    out = df[['PID', target_col]].copy()
    out[feature_cols] = X.values  # unscaled feature values, for dependence plots
    for i, feat in enumerate(feature_cols):
        out[f"shap_{feat}"] = shap_values[:, i]
    out['bias'] = bias

    return model, out


with open("config/ex_config.yaml", "r") as f:
    config = yaml.safe_load(f)

biome_mapping = config["biome_mapping"]

# ------------------------------------------------------------------
# Data cleaning and preprocessing
# ------------------------------------------------------------------
fd_df = pd.read_csv("data/final/final_dataset_ba_v2.csv")
PID_df = pd.read_csv('data/lookup/PID_location_v3.csv')

fd_df = fd_df.merge(
    PID_df[['PID', 'lat', 'lon', 'biome', 'STDAGE']],
    on='PID', how='left'
)
fd_df.drop(columns=['Unnamed: 0', 'managed', 'ownership', 'DIA', 'TPA_UNADJ', 'Functional_Richness', 'Shannon Equitabiltiy Index', 'transformed npp'], inplace=True)
fd_df.dropna(subset=['std npp', 'mean', 'Raos_Q', 'Functional_Evenness', 'Soil Moisture',
                      'Species Richness', 'Shannon Diversity', "Simpson's Index", "pet_std"], inplace=True)

fd_df = fd_df[fd_df["treecover2000"] > 30]
fd_df.drop_duplicates(subset=['PID'], inplace=True)

fd_df.rename(columns={'mean': 'mean npp', 'pet_std': "PET sd",
                       'land_cover_value': "Land Cover", 'STDAGE': "Stand Age"}, inplace=True)

fd_df['std npp']=np.log1p(fd_df['std npp'])
fd_df['mean npp']=np.log1p(fd_df['mean npp'])

ecoregions = cval.process_ecoregion("data/Ecoregions/Ecoregions2017.shp")
ecoregions = ecoregions[['ECO_NAME', 'geometry']]

fd_df['biome'] = fd_df['biome'].map(biome_mapping)  # Only run this once!

biome_dfs = {k: v for k, v in fd_df.groupby('biome')}

fd_df = fd_df[fd_df["WSCI"] != 0]
fd_df.drop(columns=['WSCI'], inplace=True)  # Drop WSCI column after filtering

fd_df = fd_df[fd_df["disturbance_value"] != -2147483648]

params = {
    "objective": "reg:squarederror",
    "learning_rate": 0.01,
    "max_depth": 6,
    "min_child_weight": 2,
    "gamma": 0.1,
    "lambda": 1.0,
    "alpha": 0.0,
    "subsample": 0.7,
    "tree_method": "approx"
}
n_rounds = 500

diversity_vars = ["Species Richness", "Shannon Diversity", "Raos_Q", "Simpson's Index", "Functional_Evenness"]
target_cols = ['std npp', 'mean npp'] + diversity_vars

exclude_cols = ['PID', 'lat', 'lon', 'biome', 'treecover2000', 'std npp', 'mean npp'] + diversity_vars
feature_cols = [c for c in fd_df.columns if c not in exclude_cols]

print(feature_cols)

Path('results').mkdir(exist_ok=True)

# ------------------------------------------------------------------
# Fit once per target on the full dataset, compute SHAP for every row
# ------------------------------------------------------------------
shap_dfs = {}
importance_rows = []

for target_col in target_cols:
    print(f"Fitting {target_col}...")
    model, shap_df = fit_and_get_shap(fd_df, target_col, feature_cols, params, n_rounds)
    shap_dfs[target_col] = shap_df

    safe_name = target_col.replace(" ", "_").replace("'", "")
    shap_df.to_csv(f'results/shap_values/shap_values_{safe_name}.csv', index=False)

    for feat in feature_cols:
        importance_rows.append({
            'target': target_col,
            'feature': feat,
            'mean_abs_shap': shap_df[f'shap_{feat}'].abs().mean(),
        })

importance_df = pd.DataFrame(importance_rows).sort_values(
    ['target', 'mean_abs_shap'], ascending=[True, False]
)
importance_df.to_csv('results/shap_feature_importance.csv', index=False)

# ------------------------------------------------------------------
# SHAP dependence plots, one figure per target
# ------------------------------------------------------------------
for target_col, shap_df in shap_dfs.items():
    safe_name = target_col.replace(" ", "_").replace("'", "")
    slopes, zeros = plot_shap_with_line_stats_v4(
        shap_df,
        features=feature_cols,
        group=target_col,
        name=None,
    )
    print(f"{target_col}: LOWESS slopes/zero-crossings computed for {len(feature_cols)} features")