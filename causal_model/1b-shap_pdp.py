import math
import yaml
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import matplotlib.ticker as mticker
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

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

# ---------------------------------------------------------------
# Publication style -- applied globally, once, at import time.
# ---------------------------------------------------------------
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#1a1a1a",
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "legend.fontsize": 8.5,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

PANEL_LETTER_STYLE = dict(fontsize=11, fontweight="bold", va="top", ha="left")
TREND_COLOR_PUB = "#B9B9B9"

# ---------------------------------------------------------------
# Diversity-metric colours.
# Keys are NORMALISED names: lowercase, underscores -> spaces,
# apostrophes removed -- so "Raos_Q", "Rao's Q" and "shap_Raos_Q"
# all resolve to "raos q".
# ---------------------------------------------------------------
METRIC_COLORS = {
    "raos q":              "#4910D8",   # indigo
    "functional evenness": "#9975BA",   # muted purple
    "shannon diversity":   "#46BD5E",   # light cyan
    "species richness":    "#13D2B9",   # teal
}
METRIC_DISPLAY = {
    "raos q": "Rao's Q",
    "functional evenness": "Functional Evenness",
    "simpsons index": "Simpson's Index",
    "shannon diversity": "Shannon Diversity",
    "species richness": "Species Richness",
}
METRIC_ORDER = list(METRIC_COLORS)   # row order in the grid
FALLBACK_COLOR = "#888888"

# ---------------------------------------------------------------
# Features drawn on a LOG x-axis (right-skewed, spans orders of
# magnitude). Names must match the feature column names exactly.
#   * The axis is log10-scaled but labelled in original units
#     (1, 2, 5, 10, 20, 50, ...).
#   * LOWESS is fitted on log10(x), so smoothing is even across the axis
#     instead of being dominated by the dense young-stand end.
#   * The slope printed for these panels is per log10 unit, i.e. the SHAP
#     change for a TEN-FOLD increase in the feature. A slope per year is
#     not meaningful for a variable this skewed.
#   * Values <= 0 cannot be placed on a log axis and are dropped from
#     these panels (a message reports how many).
# ---------------------------------------------------------------
LOG_X_FEATURES = {"Stand Age"}


def _metric_key(label):
    """Normalise a row label / filename stem to a METRIC_COLORS key."""
    s = str(label).lower()
    if s.startswith("shap_"):
        s = s[len("shap_"):]
    for ch in ("'", "\u2019"):
        s = s.replace(ch, "")
    return s.replace("_", " ").strip()


def _save_publication_figure(fig, savepath_png):
    """
    Saves a figure as both the given PNG path and a sibling PDF
    (same name, .pdf extension).
    """
    os.makedirs(os.path.dirname(savepath_png) or ".", exist_ok=True)
    fig.savefig(savepath_png, dpi=300, bbox_inches="tight")
    savepath_pdf = os.path.splitext(savepath_png)[0] + ".pdf"
    fig.savefig(savepath_pdf, bbox_inches="tight")
    print(f"  Saved {savepath_png} and {savepath_pdf}")


def _trend_line(x, y, frac=0.4, n_bins=20):
    """Return smoothed (x_sorted, y_smooth) via LOWESS, or a binned-mean
    fallback if statsmodels isn't available."""
    order = np.argsort(x)
    x_sorted, y_sorted = x[order], y[order]

    if HAS_LOWESS:
        smoothed = lowess(y_sorted, x_sorted, frac=frac, return_sorted=True)
        return smoothed[:, 0], smoothed[:, 1]

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


def make_sequential_group_colors(groups, cmap_name="Greens", low=0.25, high=0.95):
    """
    Generate a dict of {group: hex_color} using a sequential colormap.
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
            linewidth=1,
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

    return slopes, zeros


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


def plot_shap_grid_colored(
    dfs,
    features,
    shap_prefix="shap_",
    lowess_frac=0.4,
    trend_color=TREND_COLOR_PUB,
    name=None,
    show_stats=True,
    point_alpha=0.3,
    x_pad_frac=0.03,
    y_pad_frac=0.08,
    y_shared_by="metric",
):
    """
    Publication-style grid of SHAP dependence plots.

    Rows = features (ONE VARIABLE PER ROW, so every diversity metric's SHAP
    values for that variable sit side by side), columns = diversity metrics
    (one per entry in `dfs`).

    AXES:
      * x-axis: shared along each row. A row is one feature, so every panel in
        it has the same units; the x-range is the union of that feature's data
        across all metrics. Features in LOG_X_FEATURES (e.g. Stand Age) use a
        log10 axis labelled in original units.
      * y-axis, SYMMETRIC about zero (zero sits at the vertical centre of every
        panel), shared according to `y_shared_by`:
          - "metric"  (default): shared down each COLUMN. Each metric has its
            own scale, so a metric with small SHAP values is not flattened by
            one with large values; panels of one metric are comparable across
            features.
          - "feature": shared along each ROW. All metrics for one variable are
            on the same scale, so SHAP magnitudes can be compared directly
            across metrics -- but small-SHAP metrics will look flat.

    COLOUR: each column (diversity metric) is drawn in its own colour from
    METRIC_COLORS, so a colour identifies the metric throughout the figure.

    dfs is a dict {metric_label: dataframe}; metric_label is matched to
    METRIC_COLORS after normalisation (e.g. "Raos_Q" -> "raos q").
    """
    if y_shared_by not in ("metric", "feature"):
        raise ValueError("y_shared_by must be 'metric' or 'feature'")

    # Order metric columns by METRIC_ORDER (unknown metrics go last, in input order)
    def _sort_key(label):
        k = _metric_key(label)
        return METRIC_ORDER.index(k) if k in METRIC_ORDER else len(METRIC_ORDER)

    metric_labels = sorted(dfs.keys(), key=_sort_key)
    n_rows = len(features)
    n_cols = len(metric_labels)

    # ---------------------------------------------------------------
    # PASS 1: clean every panel; collect the symmetric y-limits (per metric
    # AND per feature -- one of the two is used) and each feature's x-range.
    # ---------------------------------------------------------------
    panel_data = {}
    abs_max_metric = {m: 0.0 for m in metric_labels}   # max |SHAP| within each metric's column
    abs_max_feature = {v: 0.0 for v in features}       # max |SHAP| within each feature's row
    x_ranges = {}                                      # var -> [min, max] across all metrics

    for metric_label in metric_labels:
        df = dfs[metric_label]
        for var in features:
            shap_col = f"{shap_prefix}{var}"
            if var not in df.columns or shap_col not in df.columns:
                panel_data[(metric_label, var)] = None
                continue

            feature_values = df[var].values.astype(float)
            shap_feature = df[shap_col].values.astype(float)

            valid = ~(np.isnan(feature_values) | np.isnan(shap_feature))
            if var in LOG_X_FEATURES:
                n_nonpos = int((valid & (feature_values <= 0)).sum())
                if n_nonpos:
                    print(f"[plot_shap_grid_colored] {metric_label} / {var}: dropping {n_nonpos} "
                          f"non-positive values (cannot be shown on a log axis)")
                valid &= feature_values > 0
            feature_values, shap_feature = feature_values[valid], shap_feature[valid]

            if len(feature_values) < 5 or np.all(feature_values == feature_values[0]):
                panel_data[(metric_label, var)] = None
                continue

            panel_data[(metric_label, var)] = (feature_values, shap_feature)
            amax = np.abs(shap_feature).max()
            abs_max_metric[metric_label] = max(abs_max_metric[metric_label], amax)
            abs_max_feature[var] = max(abs_max_feature[var], amax)

            lo, hi = feature_values.min(), feature_values.max()
            if var in x_ranges:
                x_ranges[var][0] = min(x_ranges[var][0], lo)
                x_ranges[var][1] = max(x_ranges[var][1], hi)
            else:
                x_ranges[var] = [lo, hi]

    if all(v == 0.0 for v in abs_max_metric.values()):
        raise ValueError("No valid data found across any (target, feature) panel -- check inputs.")

    def _symmetric(m):
        m = m if m > 0 else 1.0   # group with no data: arbitrary, it is empty anyway
        return (-m * (1 + y_pad_frac), m * (1 + y_pad_frac))

    ylim_metric = {m: _symmetric(v) for m, v in abs_max_metric.items()}
    ylim_feature = {f: _symmetric(v) for f, v in abs_max_feature.items()}

    shared_xlim = {}
    for var, (lo, hi) in x_ranges.items():
        if var in LOG_X_FEATURES:
            # pad multiplicatively (in log space) so the margin looks even on a log axis
            llo, lhi = np.log10(lo), np.log10(hi)
            pad = (lhi - llo) * x_pad_frac
            shared_xlim[var] = (10 ** (llo - pad), 10 ** (lhi + pad))
        else:
            pad = (hi - lo) * x_pad_frac
            shared_xlim[var] = (lo - pad, hi + pad)

    groups = ylim_metric if y_shared_by == "metric" else ylim_feature
    for g, (lo, hi) in groups.items():
        print(f"[plot_shap_grid_colored] {g}: y-axis {lo:.4f} to {hi:.4f}")

    # ---------------------------------------------------------------
    # PASS 2: plotting
    # ---------------------------------------------------------------
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.9 * n_cols, 3.2 * n_rows),
        squeeze=False,
        sharex="row",
        sharey="col" if y_shared_by == "metric" else "row",
    )

    alphabet = [chr(ord('A') + i) for i in range(26)]
    panel_letters = []
    for i in range(n_rows * n_cols):
        panel_letters.append(alphabet[i] if i < 26 else alphabet[i // 26 - 1] + alphabet[i % 26])
    panel_idx = 0

    # Per-metric colour / display name (one per column)
    style = {}
    for metric_label in metric_labels:
        key = _metric_key(metric_label)
        color = METRIC_COLORS.get(key)
        if color is None:
            print(f"WARNING: no colour defined for '{metric_label}' (normalised '{key}'); "
                  f"using gray. Add it to METRIC_COLORS.")
            color = FALLBACK_COLOR
        style[metric_label] = (METRIC_DISPLAY.get(key, str(metric_label).replace("_", " ")), color)

    metrics_with_data = set()

    for row_idx, var in enumerate(features):
        is_log = var in LOG_X_FEATURES

        for col_idx, metric_label in enumerate(metric_labels):
            ax = axes[row_idx, col_idx]
            letter = panel_letters[panel_idx]
            panel_idx += 1
            display_name, metric_color = style[metric_label]

            data = panel_data.get((metric_label, var))
            if data is None:
                print(f"Skipping {metric_label} / {var}: insufficient or missing data")
                ax.set_visible(False)
                continue
            metrics_with_data.add(metric_label)

            feature_values, shap_feature = data

            ax.scatter(
                feature_values, shap_feature,
                color=metric_color, alpha=point_alpha, s=6,
                linewidth=0, zorder=1, rasterized=True,
            )

            if is_log:
                # Smooth in log10(x); plot back in original units on the log axis
                xs_log, y_smooth, y_fitted_at_x = _lowess_fit(
                    np.log10(feature_values), shap_feature, frac=lowess_frac)
                x_smooth = 10 ** xs_log
                x_for_slope = xs_log          # slope per log10 unit
            else:
                x_smooth, y_smooth, y_fitted_at_x = _lowess_fit(
                    feature_values, shap_feature, frac=lowess_frac)
                x_for_slope = x_smooth

            ax.plot(x_smooth, y_smooth, color=trend_color, linewidth=0.6, zorder=4,
                    solid_capstyle='round')
            if show_stats:
                slope, _ = _lowess_slope_and_zero(x_for_slope, y_smooth)
                ss_res = np.sum((shap_feature - y_fitted_at_x) ** 2)
                ss_tot = np.sum((shap_feature - shap_feature.mean()) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
                slope_unit = " / log10" if is_log else ""
                stats_text = f"$R^2$ = {r2:.2f}\nslope = {slope:.3f}{slope_unit}"
                ax.text(0.03, 0.03, stats_text, transform=ax.transAxes, fontsize=7,
                        va="bottom", ha="left", linespacing=1.3,
                        bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                                  edgecolor="#cccccc", alpha=0.85))

            # Log axis (labelled in original units) BEFORE setting limits
            if is_log:
                ax.set_xscale("log")
                ax.xaxis.set_major_locator(mticker.LogLocator(base=10, subs=(1.0, 3.0)))   # 1, 3, 10, 30, 100, ...
                ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
                ax.xaxis.set_minor_locator(mticker.LogLocator(base=10, subs=(2, 4, 5, 6, 7, 8, 9)))
                ax.xaxis.set_minor_formatter(mticker.NullFormatter())

            # Zero-centred y-limit (per metric column or per feature row) and row-shared x-limit
            ax.set_ylim(*(ylim_metric[metric_label] if y_shared_by == "metric" else ylim_feature[var]))
            ax.set_xlim(*shared_xlim[var])

            ax.axhline(0, color="#777777", linestyle="--", linewidth=0.8, zorder=2)
            ax.grid(axis='y', color='#eeeeee', linewidth=0.5, zorder=0)
            ax.set_axisbelow(True)

            # x-label on every panel: each row is a different variable
            ax.set_xlabel(f"{var} (log scale)" if is_log else var, fontsize=8.5)
            if col_idx == 0:
                ax.set_ylabel(f"{var}\nSHAP value", fontsize=9.5)
            else:
                ax.set_ylabel("")
            if row_idx == 0:
                ax.set_title(display_name, fontsize=10, pad=8)

            ax.text(-0.1, 1.12, letter, transform=ax.transAxes, **PANEL_LETTER_STYLE)

    # Legend: one patch per diversity metric that has data + the trend line
    handles = [Patch(facecolor=style[m][1], edgecolor="none", label=style[m][0])
               for m in metric_labels if m in metrics_with_data]
    handles.append(Line2D([0], [0], color=trend_color, lw=1.8, label="LOWESS trend"))
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=min(len(handles), 6),
        frameon=False,
        fontsize=9,
    )

    scale_note = ("y-axis shared within each metric's column" if y_shared_by == "metric"
                  else "y-axis shared within each variable's row")
    fig.suptitle(f"SHAP Dependence by Variable and Diversity Metric ({scale_note}, centred at zero)",
                 y=1.01, fontsize=13, fontweight='bold')
    fig.subplots_adjust(hspace=0.55, wspace=0.35, top=0.92)

    if name:
        _save_publication_figure(fig, name)

    return fig


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
folder = "results/shap_values/"  # change to your folder path

features = ['Precipitation Seasonality', 'Elevation',
            "Stand Age"
            ]

# Collect every target's shap_df into a dict first -- the single
# combined figure is drawn once, after the loop, from this dict.
shap_dfs = {}

for filename in os.listdir(folder):
    if "shap_values" in filename:
        name = filename.split("shap_values_", 1)[1]
        name = Path(name).stem
        print(name)

        shap_df = pd.read_csv(f"{folder}{filename}")
        shap_dfs[name] = shap_df

# Single combined figure: rows = features (one variable per row), columns =
# diversity metrics (each its own colour). x-axis shared along each row; y-axis
# shared down each metric's column (centred at zero) -- see y_shared_by.
plot_shap_grid_colored(
    dfs=shap_dfs,
    features=features,
    name="results2/all_targets_grid_by_variable.png",
)