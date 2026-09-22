"""
Final stage of the pipeline using OLS linear regression, looped over
ALL diversity metrics, and SEGMENTED by pct_rare_species groups -- to
test whether the diversity -> NPP-stability relationship depends on
the percentage of rare species in the community (H4), separately for
each diversity metric.

pct_rare_species is continuous in the raw data. This script bins it
into segments (default: quartiles/tertiles/etc., roughly equal sample
size per segment) and re-runs the total-effect / direct-effect OLS
analysis independently within each segment, producing:

  1. One combined dependence-plot figure per pct_rare_species segment
     (rows = diversity metric, columns = total effect / direct
     effect), with the actual data overlaid (scatter or heatmap, see
     below).
  2. One overlay comparison figure, 2 rows (total effect / direct
     effect) x N diversity metric columns, showing each segment's
     fitted line -- and its own scatter points -- on the same axes.
  3. A summary table (printed + saved to CSV) of p-values, betas, and
     sample sizes per metric x segment.

Y-AXIS LIMITS ARE PER DIVERSITY METRIC, NOT SHARED ACROSS METRICS:
each metric's column gets its own y-range (pooled across that
metric's own segments), because diversity metrics can differ by an
order of magnitude in effect size -- a single range shared across all
metrics flattens the smaller-effect ones to near-invisible flat
lines. Segments within a metric's column DO share that metric's
range, so segment-to-segment comparison for a given metric stays
valid; just don't compare panel heights ACROSS metric columns.

SCATTER POINTS ARE PARTIAL RESIDUALS, NOT RAW (e_div, e_sd) PAIRS: the
fitted line in each panel is a CENTERED partial effect -- for the
direct-effect model in particular, it's the diversity term's
contribution with npp_mean's contribution already factored out. Raw
NPP-stability values live on a different scale (they still contain
npp_mean's effect and the intercept) and would scatter well off the
fitted line even for a good fit. Instead, each point plotted is the
standard "component + residual" quantity:

    partial_residual_i = model.resid_i + beta_div * (e_div_i - e_div_bar)

which is exactly the model's residual plus the same centered
diversity contribution the fitted line traces out -- so the points
scatter AROUND the line the way you'd expect from a correctly-drawn
partial-effect plot.

WHY QUANTILE BINNING CAN SILENTLY GIVE YOU FEWER GROUPS THAN YOU ASKED
FOR (read this if you've set N_WSCI_GROUPS=8 and seen fewer come out,
or hit a "Categorical categories must be unique" crash): plain
pd.qcut(..., duplicates="drop") merges bins whose edges collide when
enough rows share the exact same pct_rare_species value (a pile-up at
0.0 is common for a rare-species fraction). This script avoids that by
ranking pct_rare_species first (df[col].rank(method="first")) before
calling qcut, which guarantees exactly N_WSCI_GROUPS groups are
produced. Labels are prefixed with an ordinal (Q1, Q2, ...) so they
stay unique even when two groups' underlying value ranges collide.

Uses statsmodels (pip install statsmodels) and scipy.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as mcolors
import statsmodels.api as sm
from scipy import stats

# ---------------------------------------------------------------
# Edit these to match your actual residual column names
# ---------------------------------------------------------------

DIVERSITY_METRICS = {
    "Rao's Q": "residual_raos_q",
    "Functional Evenness": "residual_functional_evenness",
    "Shannon": "residual_shannon",
    "Simpson": "residual_simpsons",
    "Species richness": "residual_species_richness",
}
STABILITY_COL = "residual_std_npp"   # e_sd
MEAN_COL = "residual_mean"           # e_mean
WSCI_COL = 'pct_rare_species'        # segmenting variable
PID_COL = "PID"                      # per-observation identifier, carried through to the effects CSV

# ---------------------------------------------------------------
# pct_rare_species segmentation settings
# ---------------------------------------------------------------

N_WSCI_GROUPS = 2                   # e.g. 4 -> quartiles, 8 -> octiles
BIN_METHOD = "quantile"             # "quantile" (rank-based qcut, always exactly n_groups) or "width" (cut, equal range)
MIN_N_PER_GROUP = 30                # warn if a segment has fewer rows than this
GROUP_LABELS = None                 # e.g. ["Low", "Medium", "High"]; None -> auto-generate from actual value ranges
YLIM_PAD_FRAC = 0.08                 # padding added above/below each metric's shared y-limit, as a fraction of its range
CI_ALPHA = 0.05                      # 95% confidence bands

# ---------------------------------------------------------------
# Overlay settings (the per-observation data drawn under the fitted line)
# ---------------------------------------------------------------

# Per-segment individual dependence plots (plot_metrics_for_group):
OVERLAY_MODE = "none"            # "heatmap", "scatter", or "none"
HEATMAP_GRIDSIZE = 70                # hexbin resolution -- higher = finer detail, needs more points to look smooth

# Multi-segment overlay comparison plot (plot_group_comparison) still uses
# scatter -- several overlapping heatmaps in different colors get muddy
# fast, so that figure keeps points instead.
SHOW_SCATTER = True                       # True = scatter points, False = no points (just the fitted lines)
SCATTER_ALPHA = 0.8
SCATTER_SIZE = 2


def assign_wsci_groups(df, wsci_col=WSCI_COL, n_groups=N_WSCI_GROUPS,
                        method=BIN_METHOD, labels=GROUP_LABELS):
    """
    Adds a 'WSCI_group' categorical column to df (a copy is returned;
    original df is not mutated). Drops rows with missing wsci_col, so
    check the row count before/after if that concerns you.
    """
    df = df.copy()
    n_missing = df[wsci_col].isna().sum()
    if n_missing:
        print(f"[assign_wsci_groups] Dropping {n_missing} rows with missing {wsci_col}")
    df = df.dropna(subset=[wsci_col])

    n_unique = df[wsci_col].nunique()
    if n_unique < n_groups:
        raise ValueError(
            f"{wsci_col} has only {n_unique} distinct values, which is fewer than "
            f"n_groups={n_groups}. Reduce n_groups or reconsider this variable as a "
            f"segmenting variable."
        )

    if method == "quantile":
        # Rank first so ties can never collapse bins -- see module docstring.
        ranked = df[wsci_col].rank(method="first")
        raw_groups, bin_edges = pd.qcut(ranked, q=n_groups, retbins=True)
        n_actual = len(raw_groups.cat.categories)
        if n_actual < n_groups:
            print(f"  WARNING: only produced {n_actual}/{n_groups} groups even after "
                  f"rank-based binning (n={len(df)} rows may be too small for "
                  f"n_groups={n_groups}).")

        # Recover human-readable labels from each group's ACTUAL value range,
        # prefixed with an ordinal (Q1, Q2, ...) so labels stay unique even
        # when two groups' value ranges collide -- see module docstring.
        if labels is None:
            edge_labels = []
            for i, cat in enumerate(raw_groups.cat.categories):
                vals = df.loc[raw_groups == cat, wsci_col]
                edge_labels.append(f"Q{i + 1}: {vals.min():.2f}-{vals.max():.2f}")
            raw_groups = raw_groups.cat.rename_categories(edge_labels)
        else:
            if len(set(labels)) != len(labels):
                raise ValueError(
                    f"labels must be unique (pandas categorical requires it), got: {labels}"
                )
            raw_groups = raw_groups.cat.rename_categories(labels)
        df["WSCI_group"] = raw_groups

        boundary_ties = df.groupby(wsci_col)["WSCI_group"].nunique().gt(1).sum()
        if boundary_ties:
            print(f"  NOTE: {boundary_ties} distinct {wsci_col} value(s) had tied rows "
                  f"split across group boundaries by rank order (unavoidable when "
                  f"n_groups={n_groups} doesn't evenly divide the tie structure).")

    elif method == "width":
        df["WSCI_group"], bin_edges = pd.cut(
            df[wsci_col], bins=n_groups, labels=labels, retbins=True
        )
        if labels is None:
            cats = df["WSCI_group"].cat.categories
            edge_labels = [f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}" for i in range(len(cats))]
            df["WSCI_group"] = df["WSCI_group"].cat.rename_categories(edge_labels)
    else:
        raise ValueError("method must be 'quantile' or 'width'")

    counts = df["WSCI_group"].value_counts().sort_index()
    print(f"[assign_wsci_groups] Sample size per {wsci_col} segment "
          f"({len(counts)} groups, requested={n_groups}):")
    print(counts.to_string())
    for grp, n in counts.items():
        if n < MIN_N_PER_GROUP:
            print(f"  WARNING: segment '{grp}' has only {n} rows (< {MIN_N_PER_GROUP}); "
                  f"OLS fit in this segment may be unstable.")

    return df, bin_edges


def fit_ols_total(e_sd, e_div):
    """y = e_sd, single predictor x = e_div (the 'total effect' model)."""
    X = pd.DataFrame({"e_div": np.asarray(e_div, dtype=float)})
    Xc = sm.add_constant(X, has_constant="add")
    model = sm.OLS(np.asarray(e_sd, dtype=float), Xc).fit()
    return model, X


def fit_ols_direct(e_sd, e_div, e_mean):
    """y = e_sd, predictors x = [e_div, e_mean] (the 'direct effect' model, npp_mean controlled)."""
    X = pd.DataFrame({
        "e_div": np.asarray(e_div, dtype=float),
        "e_mean": np.asarray(e_mean, dtype=float),
    })
    Xc = sm.add_constant(X, has_constant="add")
    model = sm.OLS(np.asarray(e_sd, dtype=float), Xc).fit()
    return model, X


def get_term_pvalue(model, term_name="e_div"):
    return model.pvalues[term_name]


def _term_centered_effect(model, term_name, x_values, x_bar, alpha=CI_ALPHA):
    """
    Linear-regression analogue of a GAM partial-dependence term: the
    fitted coefficient's contribution, centered on the term's mean in
    the fitting data (so the effect averages to ~0), with a pointwise
    CI built from the coefficient's own standard error.
    """
    beta = model.params[term_name]
    se = model.bse[term_name]
    dof = model.df_resid
    t_crit = stats.t.ppf(1 - alpha / 2, dof)
    x_values = np.asarray(x_values, dtype=float)
    effect = beta * (x_values - x_bar)
    se_effect = np.abs(x_values - x_bar) * se
    ci_lower = effect - t_crit * se_effect
    ci_upper = effect + t_crit * se_effect
    return effect, ci_lower, ci_upper


def compute_partial_dependence(model, X, term_name="e_div", n_grid=200):
    """
    Grid-based partial effect: returns (XX, pdep, confi) with XX
    shaped (n_grid, 1).
    """
    x_col = X[term_name].values.astype(float)
    x_bar = x_col.mean()
    grid = np.linspace(x_col.min(), x_col.max(), n_grid)
    effect, ci_lower, ci_upper = _term_centered_effect(model, term_name, grid, x_bar)
    XX = grid.reshape(-1, 1)
    confi = np.column_stack([ci_lower, ci_upper])
    return XX, effect, confi


def compute_partial_residuals(model, X, term_name="e_div"):
    """
    The "component + residual" quantity for the scatter overlay:
    model.resid + the diversity term's own centered contribution,
    evaluated at each ACTUAL observation (not a grid). Scatters around
    the same centered partial-effect line compute_partial_dependence
    draws, regardless of what else is in the model (npp_mean, for the
    direct-effect model).
    """
    x_col = X[term_name].values.astype(float)
    x_bar = x_col.mean()
    beta = model.params[term_name]
    effect_at_obs = beta * (x_col - x_bar)
    partial_resid = np.asarray(model.resid) + effect_at_obs
    return x_col, partial_resid


def fit_metrics_for_group(df, diversity_metrics=DIVERSITY_METRICS,
                           stability_col=STABILITY_COL, mean_col=MEAN_COL):
    """
    Fits (but does not plot) the total-effect and direct-effect OLS
    model for EVERY diversity metric, on whatever subset of df is
    passed in (i.e. one pct_rare_species segment).
    """
    e_sd = df[stability_col]
    e_mean = df[mean_col]

    results = {}
    for metric_name, col_name in diversity_metrics.items():
        e_div = df[col_name]

        model_total, X_total = fit_ols_total(e_sd, e_div)
        model_direct, X_direct = fit_ols_direct(e_sd, e_div, e_mean)

        results[metric_name] = {
            "n": len(df),
            "p_total": get_term_pvalue(model_total, "e_div"),
            "p_direct": get_term_pvalue(model_direct, "e_div"),
            "model_total": model_total,
            "X_total": X_total,
            "model_direct": model_direct,
            "X_direct": X_direct,
        }
    return results


def compute_effects_dataframe(df_group, group_label, results, diversity_metrics,
                               stability_col=STABILITY_COL, mean_col=MEAN_COL,
                               pid_col=PID_COL):
    """
    For one segment's already-fitted results, evaluates each fitted
    OLS model's centered partial effect (with its confidence band) at
    every actual observation, tied back to PID, for every diversity
    metric.
    """
    rows = []
    pids = df_group[pid_col].values
    npp_mean_vals = df_group[mean_col].values
    npp_sd_vals = df_group[stability_col].values

    for metric_name, col_name in diversity_metrics.items():
        res = results[metric_name]
        e_div = df_group[col_name].values.astype(float)

        x_bar_total = res["X_total"]["e_div"].values.astype(float).mean()
        eff_total, lo_total, hi_total = _term_centered_effect(
            res["model_total"], "e_div", e_div, x_bar_total
        )

        x_bar_direct = res["X_direct"]["e_div"].values.astype(float).mean()
        eff_direct, lo_direct, hi_direct = _term_centered_effect(
            res["model_direct"], "e_div", e_div, x_bar_direct
        )

        for i in range(len(pids)):
            rows.append({
                pid_col: pids[i],
                "WSCI_group": group_label,
                "diversity_metric": metric_name,
                "diversity_residual": e_div[i],
                "npp_mean_residual": npp_mean_vals[i],
                "npp_sd_residual_actual": npp_sd_vals[i],
                "ols_total_effect": eff_total[i],
                "ols_total_effect_ci_lower": lo_total[i],
                "ols_total_effect_ci_upper": hi_total[i],
                "ols_direct_effect": eff_direct[i],
                "ols_direct_effect_ci_lower": lo_direct[i],
                "ols_direct_effect_ci_upper": hi_direct[i],
            })

    return pd.DataFrame(rows)


def compute_ylim_per_metric(all_results, model_key, pad_frac=YLIM_PAD_FRAC):
    """
    Computes y-limits SEPARATELY FOR EACH DIVERSITY METRIC (pooled
    across every segment for that metric, but never pooled across
    DIFFERENT metrics). Diversity metrics can differ by an order of
    magnitude in effect size, so a single shared range flattens the
    weaker ones to near-invisible flat lines -- see module docstring.

    all_results is keyed [segment][metric_name] -> per-metric result
    dict (as built by fit_metrics_for_group). Returns
    {metric_name: (ymin, ymax)}.
    """
    X_key = "X_total" if model_key == "model_total" else "X_direct"

    metric_names = set()
    for metric_results in all_results.values():
        metric_names.update(metric_results.keys())

    ylims = {}
    for metric_name in metric_names:
        lows, highs = [], []
        for metric_results in all_results.values():
            if metric_name not in metric_results:
                continue
            res = metric_results[metric_name]
            _, _, confi = compute_partial_dependence(res[model_key], res[X_key])
            lows.append(np.nanmin(confi[:, 0]))
            highs.append(np.nanmax(confi[:, 1]))
        ymin, ymax = min(lows), max(highs)
        pad = (ymax - ymin) * pad_frac
        ylims[metric_name] = (ymin - pad, ymax + pad)
    return ylims


def make_density_cmap(color, max_alpha=0.9):
    """
    Builds a colormap that goes from fully transparent (0 points in a
    hexbin cell) up to `color` at max_alpha (the densest cell) --
    lets the heatmap sit on white axes without a background patch and
    keeps the panel's color identity (matches the fitted line).
    """
    rgba = mcolors.to_rgba(color)
    transparent = (rgba[0], rgba[1], rgba[2], 0.0)
    opaque = (rgba[0], rgba[1], rgba[2], max_alpha)
    return mcolors.LinearSegmentedColormap.from_list(f"density_{color}", [transparent, opaque])


def plot_dependence_on_ax(ax, model, X, feature_name, title, color="#D85A30",
                           overlay_mode=OVERLAY_MODE):
    """
    Per-segment figure panel: auto-scales its own y-axis (not tied to
    the comparison figure's shared ylim -- see plot_group_comparison
    for that). overlay_mode controls how the actual data (as partial
    residuals -- see compute_partial_residuals) is drawn underneath
    the fitted line and its confidence band:
      "heatmap" -- hexbin density, tinted to match `color`
      "scatter" -- individual translucent points (see the SCATTER_*
                   settings)
      "none"    -- line + CI band only, no per-observation overlay
    """
    XX, pdep, confi = compute_partial_dependence(model, X)
    p_value = get_term_pvalue(model, "e_div")

    if overlay_mode == "heatmap":
        x_obs_resid, partial_resid = compute_partial_residuals(model, X)
        cmap = make_density_cmap(color)
        ax.hexbin(x_obs_resid, partial_resid, gridsize=HEATMAP_GRIDSIZE, cmap=cmap,
                  mincnt=1, linewidths=0.1, zorder=1)
    elif overlay_mode == "scatter":
        x_obs_resid, partial_resid = compute_partial_residuals(model, X)
        ax.scatter(x_obs_resid, partial_resid, color=color, alpha=SCATTER_ALPHA,
                   s=SCATTER_SIZE, linewidths=0, zorder=1)
    elif overlay_mode != "none":
        raise ValueError('overlay_mode must be "heatmap", "scatter", or "none"')

    ax.plot(XX[:, 0], pdep, color=color, linewidth=2, zorder=3)
    ax.fill_between(XX[:, 0], confi[:, 0], confi[:, 1], color=color, alpha=0.2, zorder=2)


    # Rug placed at the CURRENT bottom of the axis, computed after the
    # heatmap/scatter (if any) has already been added, so it sits below
    # everything.
    x_obs = X["e_div"].values.astype(float)
    ax.scatter(x_obs, np.full_like(x_obs, ax.get_ylim()[0]), marker="|",
               color="gray", alpha=0.4, s=30, zorder=4)

    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xlabel(feature_name, fontsize=9)
    ax.set_ylabel("Partial effect on NPP sd", fontsize=9)
    ax.set_title(title, fontsize=10)

    sig_flag = "*" if p_value < 0.05 else ""
    ax.text(0.05, 0.95, f"p = {p_value:.3g}{sig_flag}",
            transform=ax.transAxes, verticalalignment="top", fontsize=9,
            fontweight="bold" if p_value < 0.05 else "normal",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7, edgecolor="gray"))

    return p_value


def plot_metrics_for_group(results, diversity_metrics,
                            savepath="ols_dependence.png",
                            suptitle=None):
    """
    Draws the combined figure (rows = diversity metric, columns =
    total / direct effect) for one segment's already-fitted results.
    Each panel auto-scales its own y-axis -- this figure is
    independent of the comparison figure's shared-per-metric ylim.
    """
    n_metrics = len(diversity_metrics)
    fig, axes = plt.subplots(n_metrics, 2, figsize=(11, 4 * n_metrics), squeeze=False)

    for row, metric_name in enumerate(diversity_metrics.keys()):
        res = results[metric_name]

        plot_dependence_on_ax(
            axes[row, 0], res["model_total"], res["X_total"],
            feature_name=f"{metric_name} residual",
            title=f"{metric_name} -- total effect",
        )
        plot_dependence_on_ax(
            axes[row, 1], res["model_direct"], res["X_direct"],
            feature_name=f"{metric_name} residual",
            title=f"{metric_name} -- direct effect (npp_mean controlled)",
        )

    if suptitle:
        fig.suptitle(suptitle, fontsize=13, y=1.02)
    fig.tight_layout()
    if savepath is not None:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")  # FIXED: was commented out, so per-segment figures were never written to disk
    return fig


def _plot_effect_row(axes_row, all_results, diversity_metrics, group_order,
                      colors, model_key, X_key, p_key, row_label, ylims_per_metric,
                      show_scatter=SHOW_SCATTER):
    """
    Shared helper: draws one row of panels (one per diversity metric),
    each overlaying every segment's fitted line (and, if show_scatter,
    its own partial-residual scatter) for a single effect type. Each
    column's y-limit is looked up from ylims_per_metric (see
    compute_ylim_per_metric) -- NOT shared across metric columns.
    """
    for col, metric_name in enumerate(diversity_metrics.keys()):
        ax = axes_row[col]
        ylim = ylims_per_metric[metric_name]
        for grp, color in zip(group_order, colors):
            if grp not in all_results or metric_name not in all_results[grp]:
                continue
            res = all_results[grp][metric_name]
            model = res[model_key]
            X = res[X_key]

            if show_scatter:
                x_obs_resid, partial_resid = compute_partial_residuals(model, X)
                ax.scatter(x_obs_resid, partial_resid, color=color, alpha=SCATTER_ALPHA * 0.7,
                           s=SCATTER_SIZE * 0.6, linewidths=0, zorder=1)

            XX, pdep, confi = compute_partial_dependence(model, X)
            ax.plot(XX[:, 0], pdep, color=color, linewidth=2, zorder=3,
                    label=f"{grp} (n={res['n']}, p={res[p_key]:.2g})")
            ax.fill_between(XX[:, 0], confi[:, 0], confi[:, 1], color=color, alpha=0.12, zorder=2)

        ax.set_ylim(ylim)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_xlabel(f"{metric_name} residual", fontsize=9)
        ax.set_ylabel("Partial effect on NPP sd", fontsize=9)
        ax.set_title(f"{metric_name} -- {row_label} by pct rare species segment", fontsize=10)
        ax.legend(fontsize=7, loc="best")


def plot_group_comparison(all_results, diversity_metrics, group_order,
                           ylims_total, ylims_direct,
                           savepath="ols_pct_rare_segment_comparison.png"):
    """
    Overlays every segment's partial-effect line on one set of axes
    per diversity metric -- one row for the total effect (npp_mean
    excluded), one row for the direct effect (npp_mean controlled
    for). ylims_total / ylims_direct are per-metric dicts (see
    compute_ylim_per_metric): each metric gets its own column range,
    not one range shared across every metric.
    """
    n_metrics = len(diversity_metrics)
    fig, axes = plt.subplots(2, n_metrics, figsize=(6 * n_metrics, 9), squeeze=False)

    colors = cm.BrBG(np.linspace(0.15, 0.85, len(group_order)))

    _plot_effect_row(axes[0, :], all_results, diversity_metrics, group_order,
                      colors, "model_total", "X_total", "p_total", "total effect",
                      ylims_total)
    _plot_effect_row(axes[1, :], all_results, diversity_metrics, group_order,
                      colors, "model_direct", "X_direct", "p_direct",
                      "direct effect (npp_mean controlled)", ylims_direct)

    fig.tight_layout()
    fig.savefig(savepath, dpi=150, bbox_inches="tight")
    return fig


def run_segmented_by_wsci(df, diversity_metrics=DIVERSITY_METRICS,
                           stability_col=STABILITY_COL, mean_col=MEAN_COL,
                           wsci_col=WSCI_COL, pid_col=PID_COL, outdir="results",
                           n_groups=N_WSCI_GROUPS, method=BIN_METHOD):
    """
    Top-level entry point:
      1. Bins into segments (guaranteed exactly n_groups for
         method="quantile" -- see assign_wsci_groups).
      2. Fits EVERY diversity metric's total/direct OLS model within
         each segment (fitting only -- no plotting yet).
      3. Computes y-limits SEPARATELY FOR EACH DIVERSITY METRIC (see
         compute_ylim_per_metric) -- not one range shared across all
         metrics.
      4. Draws the per-segment combined figure (auto-scaled, with
         scatter/heatmap) and the cross-segment comparison figure
         (per-metric shared ylims, with scatter).
      5. Saves a CSV summary of n and p-values per metric x segment.
      6. Saves a per-observation (per-PID) CSV of the fitted effect,
         total and direct, per metric, so results can be joined back
         to individual plots.
    """
    df_grouped, bin_edges = assign_wsci_groups(df, wsci_col=wsci_col,
                                                n_groups=n_groups, method=method)
    group_order = list(df_grouped["WSCI_group"].cat.categories)

    # --- fit everything first, no plotting yet; also compute per-PID effects ---
    all_results = {}
    effects_frames = []
    for grp in group_order:
        sub = df_grouped[df_grouped["WSCI_group"] == grp]
        if sub.empty:
            print(f"[run_segmented_by_wsci] Skipping empty segment '{grp}'")
            continue
        all_results[grp] = fit_metrics_for_group(
            sub, diversity_metrics=diversity_metrics,
            stability_col=stability_col, mean_col=mean_col,
        )
        effects_frames.append(compute_effects_dataframe(
            sub, grp, all_results[grp], diversity_metrics,
            stability_col=stability_col, mean_col=mean_col, pid_col=pid_col,
        ))

    # --- per-segment combined figures: each panel auto-scales its own y-axis ---
    for grp in group_order:
        if grp not in all_results:
            continue
        # savepath = f"{outdir}/ols_dependence_pct_rare_{str(grp).replace(' ', '_').replace('/', '-').replace(':', '')}.png"
        plot_metrics_for_group(
            all_results[grp], diversity_metrics,
            savepath=None,
            suptitle=f"pct_rare_species segment: {grp}  (n={all_results[grp][next(iter(diversity_metrics))]['n']})",
        )

    # --- per-metric y-limits: one range for the total-effect row, one for the
    #     direct-effect row, for EACH diversity metric column ---
    ylims_total = compute_ylim_per_metric(all_results, "model_total")
    ylims_direct = compute_ylim_per_metric(all_results, "model_direct")
    for metric_name in diversity_metrics:
        t = ylims_total[metric_name]
        d = ylims_direct[metric_name]
        print(f"[run_segmented_by_wsci] {metric_name}: "
              f"ylim total=({t[0]:.3g}, {t[1]:.3g}), direct=({d[0]:.3g}, {d[1]:.3g})")

    # --- summary table across segments x metrics ---
    rows = []
    for grp, metric_results in all_results.items():
        for metric_name, res in metric_results.items():
            rows.append({
                "pct_rare_species_segment": grp,
                "diversity_metric": metric_name,
                "n": res["n"],
                "p_total": res["p_total"],
                "p_direct": res["p_direct"],
                "beta_total": res["model_total"].params["e_div"],
                "beta_direct": res["model_direct"].params["e_div"],
                "r2_total": res["model_total"].rsquared,
                "r2_direct": res["model_direct"].rsquared,
            })
    summary_df = pd.DataFrame(rows)
    summary_path = f"{outdir}/ols_pct_rare_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n[run_segmented_by_wsci] Summary saved to {summary_path}")
    print(summary_df.to_string(index=False))

    # --- per-observation (per-PID) OLS effects, combined across all segments x metrics ---
    effects_df = pd.concat(effects_frames, ignore_index=True)
    effects_path = f"{outdir}/ols_pct_rare_effects_per_observation.csv"
    effects_df.to_csv(effects_path, index=False)
    print(f"[run_segmented_by_wsci] Per-observation effects saved to {effects_path} "
          f"({len(effects_df)} rows: {df_grouped.shape[0]} PIDs x {len(diversity_metrics)} metrics)")

    # --- overlay comparison plot, per-metric ylims across segments ---
    plot_group_comparison(all_results, diversity_metrics, group_order,
                           ylims_total=ylims_total, ylims_direct=ylims_direct,
                           savepath=f"{outdir}/ols_pct_rare_segment_comparison.png")

    return all_results, summary_df, effects_df, group_order, ylims_total, ylims_direct


if __name__ == "__main__":
    residuals_df = pd.read_csv('results/step_2_residuals_merged2.csv')

    abundance_df = pd.read_parquet('data/final/dataset_relba_10.parquet')
    residuals_df = residuals_df.merge(abundance_df, on='PID', how='inner')

    all_results, summary_df, effects_df, group_order, ylims_total, ylims_direct = run_segmented_by_wsci(residuals_df)

    plt.show()