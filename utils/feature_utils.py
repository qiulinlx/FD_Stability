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

try:
    from statsmodels.nonparametric.smoothers_lowess import lowess
    HAS_LOWESS = True
except ImportError:
    HAS_LOWESS = False

N_PERM_REPEATS = 20

def permutation_importance_booster(model, X_s, y, n_repeats=N_PERM_REPEATS, seed=42):
    """
    Permutation importance for a trained xgboost Booster.
    For each feature, shuffle its column `n_repeats` times, re-predict, and
    record the DROP IN R² relative to the unshuffled baseline. Larger drop =
    the model relies more on that feature. Using R² (rather than raw MSE)
    puts every target on a comparable, scale-free footing.
    Returns {feature: (mean_drop, sd_drop)}.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=float)
    ss_tot = np.sum((y - y.mean()) ** 2)

    def _r2(pred):
        return 1.0 - np.sum((y - pred) ** 2) / ss_tot

    base = _r2(model.predict(xgb.DMatrix(X_s, enable_categorical=True)))

    X_perm = X_s.copy()
    out = {}
    for feat in X_s.columns:
        original = X_s[feat].values.copy()
        drops = []
        for _ in range(n_repeats):
            X_perm[feat] = rng.permutation(original)
            pred = model.predict(xgb.DMatrix(X_perm, enable_categorical=True))
            drops.append(base - _r2(pred))
        X_perm[feat] = original
        out[feat] = (float(np.mean(drops)), float(np.std(drops)))
    return out

# ------------------------------------------------------------------
# Gain ("Gini-style") and permutation importance
# ------------------------------------------------------------------
def gain_importance(model, feature_cols, normalize=True):
    """
    Impurity-style importance for an XGBoost model: the total reduction in
    the loss (squared error) contributed by every split on each feature
    ('total_gain'). This is XGBoost's equivalent of Gini/impurity importance
    in random forests. Features never used in a split get 0.
    If normalize=True, values are each feature's SHARE of the model's total
    gain (sums to 1), which makes targets with different scales comparable.
    """
    raw = model.get_score(importance_type="total_gain")
    vals = np.array([raw.get(f, 0.0) for f in feature_cols], dtype=float)
    if normalize and vals.sum() > 0:
        vals = vals / vals.sum()
    return dict(zip(feature_cols, vals))

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



def fit_and_get_shap(df, target_col, feature_cols, params, n_rounds):
    """
    Fit an XGBoost model on the full dataset for target_col, then compute
    native XGBoost SHAP contributions for every row. Returns:
      model  : the trained Booster
      out    : wide-format DataFrame (PID, target, unscaled features,
               shap_<feature> columns, bias)
      X_s    : the standardised feature matrix the model was trained on
               (needed for permutation importance)
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

    return model, out, X


def _lowess_fit(x, y, frac=0.4, n_bins=20):
    """
    Fit a LOWESS smooth (or binned-mean fallback).
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


def _lowess_slope_and_zero(x_smooth, y_smooth):
    """
    Summarize a (possibly nonlinear) LOWESS curve with a single slope
    and its zero-crossing.
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