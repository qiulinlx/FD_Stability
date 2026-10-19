import pandas as pd
import sys
import os
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats
from scipy.ndimage import gaussian_filter
import warnings


sys.path.append(str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import utils.visualisation_utils
from utils.feature_utils import _lowess_fit, _lowess_slope_and_zero


def _save_publication_figure(fig, savepath_png):
    if savepath_png is None:
        return
    fig.savefig(savepath_png, dpi=300, bbox_inches="tight")
    # savepath_pdf = os.path.splitext(savepath_png)[0] + ".pdf"
    # fig.savefig(savepath_pdf, bbox_inches="tight")
    print(f"  Saved {savepath_png}")

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
FALLBACK_COLOR = "#808080"      # used for metrics missing from METRIC_COLORS

# Density colouring of the points (Gaussian-smoothed, as in ecoregion_conifer.py)
COLOR_BY_DENSITY = True         # False = plain single-colour points (old look)
DENSITY_BINS = 100              # grid cells per axis; fewer = blurrier
DENSITY_SMOOTH = (3, 3)         # Gaussian blur (x, y, in grid cells); larger = smoother


def _metric_key(label):
    """Normalise a row label / filename stem to a METRIC_COLORS key."""
    s = str(label).lower()
    if s.startswith("shap_"):
        s = s[len("shap_"):]
    for ch in ("'", "\u2019"):
        s = s.replace(ch, "")
    return s.replace("_", " ").strip()


def _density_pct(x, y, bins=DENSITY_BINS, smooth=DENSITY_SMOOTH):
    """Per-point density percentile from a Gaussian-smoothed 2-D histogram.
    Percentiles (not raw counts) stop one very dense spot from washing out the rest."""
    h, xe, ye = np.histogram2d(x, y, bins=bins)
    h = gaussian_filter(h, smooth)
    xi = np.clip(np.searchsorted(xe, x) - 1, 0, bins - 1)
    yi = np.clip(np.searchsorted(ye, y) - 1, 0, bins - 1)
    return stats.rankdata(h[xi, yi]) / len(x)


def _metric_cmap(color):
    """Light tint -> metric colour -> darker shade, for density-coloured points."""
    c = np.array(mcolors.to_rgb(color))
    return mcolors.LinearSegmentedColormap.from_list(
        f"dens_{color}", [1 - 0.30 * (1 - c), c, 0.55 * c])


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
    color_by_density=COLOR_BY_DENSITY,
    density_smooth=DENSITY_SMOOTH,
):
    """

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
    With color_by_density=True the points shade from a light tint (sparse) to
    a dark shade (dense) of that colour, using a Gaussian-smoothed point
    density (blur set by density_smooth). For log-x features the density is
    computed in log10(x), so the blur looks even on the log axis.

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

            if color_by_density:
                x_dens = np.log10(feature_values) if is_log else feature_values
                dens = _density_pct(x_dens, shap_feature, smooth=density_smooth)
                order = np.argsort(dens)                 # densest points drawn on top
                ax.scatter(
                    feature_values[order], shap_feature[order],
                    c=dens[order], cmap=_metric_cmap(metric_color), vmin=0, vmax=1,
                    alpha=0.8, s=6, linewidth=0, zorder=1, rasterized=True,
                )
            else:
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

            # Dark halo under the trend line so it stays visible over dense (dark) points
            if color_by_density:
                ax.plot(x_smooth, y_smooth, color="#333333", linewidth=1.8, zorder=3,
                        solid_capstyle='round')
            ax.plot(x_smooth, y_smooth, color=trend_color, linewidth=0.6 if not color_by_density else 1.0,
                    zorder=4, solid_capstyle='round')
            if show_stats:
                slope, _ = _lowess_slope_and_zero(x_for_slope, y_smooth)
                ss_res = np.sum((shap_feature - y_fitted_at_x) ** 2)
                ss_tot = np.sum((shap_feature - shap_feature.mean()) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
                slope_unit = " / log10" if is_log else ""
                stats_text = f"$R^2$ = {r2:.2f}\nslope = {slope:.3f}{slope_unit}"
                ax.text(0.03, 0.03, stats_text, transform=ax.transAxes, fontsize=7,
                        va="bottom", ha="left", linespacing=1.3, zorder=6,
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
    if color_by_density:
        handles.append(Patch(facecolor="none", edgecolor="none",
                             label="Shade: light = sparse, dark = dense"))
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=min(len(handles), 6),
        frameon=False,
        fontsize=9,
    )

    fig.suptitle("SHAP Dependence by Variable and Diversity Metric",
                 y=1.01, fontsize=13, fontweight='bold')
    fig.subplots_adjust(hspace=0.55, wspace=0.35, top=0.92)

    if name:
        _save_publication_figure(fig, name)

    return fig



# ---------------------------------------------------------------
# Diversity-metric colours.
# Keys are NORMALISED names: lowercase, underscores -> spaces,
# apostrophes removed -- so "Raos_Q", "Rao's Q" and "shap_Raos_Q"
# all resolve to "raos q".
# ---------------------------------------------------------------
METRIC_COLORS = {
    "raos q":              "#4910D8",   # indigo
    "functional evenness": "#9975BA",   # muted purple
    "species richness":    "#13D2B9",   # teal
}

METRIC_DISPLAY = {
    "raos q": "Rao's Q",
    "functional evenness": "Functional Evenness",
    "species richness": "Species Richness",
}
METRIC_ORDER = list(METRIC_COLORS)   # column order in the grid

LOG_X_FEATURES = {"Stand Age"}

# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
if __name__ == "__main__":
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

    plot_shap_grid_colored(
        dfs=shap_dfs,
        features=features,
        name="shap_results/all_targets_grid_by_variable.png",
    )