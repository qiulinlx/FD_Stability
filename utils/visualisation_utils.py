from pathlib import Path
import numpy as np
import os

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from utils.feature_utils import _lowess_fit, _lowess_slope_and_zero

TARGET_COLORS = {
    "Functional_Evenness": "#9975BA",   # muted purple
    "Raos_Q":              "#4A16D8",   # deep indigo
    "Shannon Diversity":   "#7DDFE7",   # light cyan
    "Species Richness":    "#16D2B8",   # teal
}

TREND_COLOR_PUB = "#B9B9B9"


def _plot_ranking_bar(sub, xlabel, title, savepath, col="theta_pct", decimals=2):
    """Shared horizontal ranking bar chart for individual and mutually adjusted effects."""
    sub = sub.assign(_abs=sub[col].abs()).sort_values("_abs").reset_index(drop=True)
    lo_col, hi_col = f"{col}_ci_lower", f"{col}_ci_upper"

    y = np.arange(len(sub))
    heights = sub[col].values
    xerr = [heights - sub[lo_col].values,
            sub[hi_col].values - heights]
    colors = [COLOR_NEG if h < 0 else COLOR_POS for h in heights]

    fig, ax = plt.subplots(figsize=(6.5, 0.55 * len(sub) + 1.3))
    ax.barh(y, heights, height=0.55, xerr=xerr, capsize=3,
            color=colors, alpha=0.9, edgecolor="#1a1a1a", linewidth=0.6,
            error_kw=dict(ecolor="#1a1a1a", elinewidth=0.8, capthick=0.8))
    ax.axvline(0, color="#1a1a1a", linewidth=0.8, zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels(sub["diversity_metric"])
    ax.set_xlabel(xlabel)
    ax.set_title(title, pad=10)
    ax.grid(axis="y", visible=False)

    data_min = min(sub[lo_col].min(), 0)
    data_max = max(sub[hi_col].max(), 0)
    x_range = data_max - data_min if data_max > data_min else 1.0
    margin = 0.25 * x_range
    ax.set_xlim(data_min - margin, data_max + margin)

    offset = 0.03 * x_range
    for yi, h, lo, hi, sig in zip(y, heights, sub[lo_col].values,
                                  sub[hi_col].values, sub["significant_fdr"].values):
        label = f"{h:.{decimals}f}%" + ("*" if sig else "")
        ax.text(hi + offset if h >= 0 else lo - offset, yi, label, va="center",
                ha="left" if h >= 0 else "right", fontsize=8, color="#1a1a1a")

    fig.tight_layout()
    _save_publication_figure(fig, savepath)
    return fig


def _save_publication_figure(fig, savepath_png):
    if savepath_png is None:
        return
    fig.savefig(savepath_png, dpi=300, bbox_inches="tight")
    # savepath_pdf = os.path.splitext(savepath_png)[0] + ".pdf"
    # fig.savefig(savepath_pdf, bbox_inches="tight")
    print(f"  Saved {savepath_png}")


def plot_importance_grouped_horizontal(
    imp_df,
    value_col,
    xlabel,
    title,
    savepath=None,
    targets=None,
    top_n=None,
    err_col=None,
    colors=TARGET_COLORS,
):
    """
    Horizontal grouped bars: one group per feature (most important at the
    top), one bar per target inside each group, coloured by target.
    imp_df is long-format with columns: target, feature, <value_col>
    (and optionally <err_col> for error bars).
    """
    df = imp_df.copy()
    if targets is None:
        targets = list(dict.fromkeys(df["target"]))
    df = df[df["target"].isin(targets)]

    order = df.groupby("feature")[value_col].sum().sort_values(ascending=False)
    if top_n:
        order = order.head(top_n)
    features = list(order.index)

    n_f, n_t = len(features), len(targets)
    bar_h = 0.8 / n_t

    fig, ax = plt.subplots(figsize=(5,12))

    for i, t in enumerate(targets):
        sub = df[df["target"] == t].set_index("feature").reindex(features)
        vals = sub[value_col].fillna(0).values
        ypos = np.arange(n_f) + (i - (n_t - 1) / 2) * bar_h
        xerr = sub[err_col].fillna(0).values if err_col else None
        ax.barh(
            ypos, vals, height=bar_h,
            color=colors.get(t, "#888888"), label=t,
            xerr=xerr,
            error_kw=dict(ecolor="#444444", elinewidth=0.8, capsize=1.5),
        )

    ax.set_yticks(np.arange(n_f))
    ax.set_yticklabels(features)
    ax.invert_yaxis()                      # most important feature at the top
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", color="#eeeeee", linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    full_title = title
    ax.set_title(full_title, fontsize=11)
    ax.legend(title="Target", loc="lower right", frameon=False)

    fig.tight_layout()
    if savepath:
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
        fig.savefig(str(Path(savepath).with_suffix(".pdf")), bbox_inches="tight")
    return fig



def make_sequential_group_colors(groups, cmap_name="Greens", low=0.25, high=0.95):
    """
    Generate a dict of {group: hex_color} using a sequential colormap.
    """
    cmap = plt.get_cmap(cmap_name)
    n = len(groups)
    colors = [cmap(low + (high - low) * i / (n - 1)) for i in range(n)]
    return {g: mcolors.to_hex(c) for g, c in zip(groups, colors)}




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



def _metric_key(label):
    """Normalise a row label / filename stem to a METRIC_COLORS key."""
    s = str(label).lower()
    if s.startswith("shap_"):
        s = s[len("shap_"):]
    for ch in ("'", "\u2019"):
        s = s.replace(ch, "")
    return s.replace("_", " ").strip()

