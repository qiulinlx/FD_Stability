"""
Final stage of the pipeline using a GAM (Generalized Additive Model)
with a diversity x WSCI TENSOR SMOOTH, looped across all diversity
metrics, to test whether the diversity -> NPP-stability relationship
depends on structural complexity (H4) -- without discretizing WSCI
into tertile groups, and without assuming the interaction is linear.

This is the GAM analogue of the OLS-interaction script. Rather than

    e_sd ~ e_div + e_wsci + e_div:e_wsci          (linear interaction)

this fits

    Total effect:   e_sd ~ s(e_div) + s(e_wsci) + te(e_div, e_wsci)
    Direct effect:  e_sd ~ s(e_div) + s(e_mean) + s(e_wsci) + te(e_div, e_wsci)

using pyGAM (`pip install pygam`), where `s()` is a univariate smooth
and `te()` is a tensor-product smooth capturing a possibly-nonlinear
interaction surface between diversity and WSCI.

WHAT CHANGES VS. THE LINEAR VERSION (read this before trusting output)
------------------------------------------------------------------
1. There is no single interaction *coefficient* anymore -- te() is a
   smooth surface, not a slope times a slope. H4 ("does the diversity
   effect depend on structural complexity") is instead tested by
   comparing the full model (with te()) against a reduced additive
   model (s(div) + s(wsci), no interaction) via AIC/GCV. A materially
   lower AIC for the interaction model is evidence for H4; this
   script reports both and flags the comparison, but does not invent
   a p-value the way the OLS interaction term had one.

2. "Simple slopes" become "simple curves" (partial dependence lines):
   at each WSCI cut level w, we hold WSCI fixed at w and trace out the
   (possibly nonlinear) predicted effect of diversity, centered so the
   curve equals 0 at e_div = 0 (the diversity mean) -- same convention
   as before. We also report a *numerical derivative at e_div = 0* as
   the closest analogue to "the slope at average diversity."

3. Instead of three fixed levels (low/median/high WSCI), you can set
   N_WSCI_CUTS to any number of evenly spaced WSCI quantiles. Cuts are
   automatically generated and labeled.

4. Confidence bands come from pyGAM's built-in `confidence_intervals`,
   which uses the smooth's posterior covariance (Wood-style Bayesian
   CIs for GAMs) -- not a closed-form delta method as in the linear
   case. These are on the *centered curve* (predicted - predicted at
   e_div=0), computed as a contrast, so treat them as approximate
   (pyGAM does not natively expose covariance between two prediction
   points, so the band is built by propagating each point's own CI --
   this is conservative but not exact; for publication-grade CIs on
   the contrast, a parametric bootstrap over gam.coef_ / gam.statistics_
   would be more rigorous).

5. Johnson-Neyman becomes a NUMERICAL SCAN rather than a closed-form
   quadratic solve: for a fine grid of raw WSCI values, we compute a
   fixed diversity contrast (effect of moving e_div from its 10th to
   90th percentile) and its CI, then find where that CI stops
   excluding zero by interpolating sign changes across the grid.

6. Y-AXIS LIMITS ARE PER DIVERSITY METRIC, NOT SHARED GLOBALLY. Each
   row of the combined figure (one diversity metric) gets its own
   y-range for its total-effect panel and its own y-range for its
   direct-effect panel, computed from that metric's own WSCI-cut
   curves only. This trades away strict cross-metric comparability of
   effect size (a "steep" Rao's Q curve and a "flat" Simpson curve no
   longer sit on the same visual scale) in exchange for every metric's
   curve shape being legible on its own axes -- useful when metrics
   differ by an order of magnitude in effect size, which would
   otherwise flatten the weaker ones to invisibility under one shared
   range. If you need cross-metric magnitude comparisons, read them
   off the summary CSV's slope/AIC columns rather than the plot axes.

Uses pyGAM (pip install pygam), numpy, pandas, scipy, matplotlib.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from pygam import LinearGAM, s, te
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
WSCI_COL = "residual_wsci"                 # e_wsci (continuous moderator)
PID_COL = "PID"                      # per-observation identifier, carried through to the effects CSV

# ---------------------------------------------------------------
# Simple-curve / Johnson-Neyman-scan settings
# ---------------------------------------------------------------

# How many WSCI levels to plot/report the diversity partial curve at.
# Cuts are evenly spaced QUANTILES of e_wsci between WSCI_QUANTILE_RANGE.
# e.g. N_WSCI_CUTS = 3 with range (0.1, 0.9) reproduces the old
# low(p10)/median(p50)/high(p90) behavior. Set this to any n >= 2.
N_WSCI_CUTS = 8
WSCI_QUANTILE_RANGE = (0.10, 0.90)   # low/high bounds for the evenly spaced cuts

DIV_CONTRAST_QUANTILES = (0.10, 0.90)  # diversity contrast used for JN scan: p10 -> p90 of e_div
JN_GRID_N = 400                        # resolution of the WSCI scan for the JN boundary search

YLIM_PAD_FRAC = 0.0                  # padding above/below each metric's y-limit, as a fraction of its range
CI_ALPHA = 0.05                      # 95% confidence bands / significance tests
N_SPLINES = 12                       # basis functions per smooth term (tune if curves look over/under-fit
                                      # -- pyGAM's gridsearch below tunes the penalty, not spline count)


def generate_wsci_cuts(n_cuts=N_WSCI_CUTS, qrange=WSCI_QUANTILE_RANGE):
    """
    Builds n_cuts evenly spaced quantile levels between qrange[0] and
    qrange[1] (inclusive), with human-readable labels. n_cuts=1
    collapses to the midpoint; n_cuts>=2 always includes both ends.
    Returns an ordered dict-like list of (label, quantile) pairs.
    """
    if n_cuts < 1:
        raise ValueError("N_WSCI_CUTS must be >= 1")
    if n_cuts == 1:
        q = (qrange[0] + qrange[1]) / 2
        return [(f"WSCI (p{int(round(q * 100))})", q)]
    quantiles = np.linspace(qrange[0], qrange[1], n_cuts)
    cuts = []
    for q in quantiles:
        pct = int(round(q * 100))
        if np.isclose(q, 0.5):
            label = "Median WSCI"
        elif q < 0.5:
            label = f"Low WSCI (p{pct})"
        else:
            label = f"High WSCI (p{pct})"
        cuts.append((label, float(q)))
    return cuts


SLOPE_QUANTILES = dict(generate_wsci_cuts())  # label -> quantile, same shape the plotting code expects


def prepare_centered_data(df, diversity_metrics=DIVERSITY_METRICS,
                           stability_col=STABILITY_COL, mean_col=MEAN_COL,
                           wsci_col=WSCI_COL):
    """
    Drops rows missing any needed column, then adds mean-centered
    versions of e_wsci, e_mean, and each diversity metric. Centering
    keeps "effect = 0 at the mean" interpretable and keeps the GAM's
    smooth basis well-conditioned around zero.
    Returns (df_centered, means) where means is a dict of the raw
    column means used for centering (needed to convert JN boundaries
    back to raw WSCI units).
    """
    needed_cols = [stability_col, mean_col, wsci_col] + list(diversity_metrics.values())
    df = df.copy()
    n_before = len(df)
    df = df.dropna(subset=needed_cols)
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"[prepare_centered_data] Dropped {n_dropped} rows with missing values "
              f"in {needed_cols}")

    means = {
        "wsci": df[wsci_col].mean(),
        "mean": df[mean_col].mean(),
    }
    df["e_wsci_c"] = df[wsci_col] - means["wsci"]
    df["e_mean_c"] = df[mean_col] - means["mean"]

    for metric_name, col_name in diversity_metrics.items():
        div_mean = df[col_name].mean()
        means[col_name] = div_mean
        df[f"e_div_c__{col_name}"] = df[col_name] - div_mean

    print(f"[prepare_centered_data] n = {len(df)}")
    return df, means


def _fit_gam(X, y, terms, n_splines=N_SPLINES):
    """Fits a LinearGAM with the given term spec, tuning smoothing via gridsearch."""
    gam = LinearGAM(terms)
    gam.gridsearch(X, y, progress=False)
    return gam


def fit_gam_total(e_sd, e_div_c, e_wsci_c):
    """
    Total effect model. Columns: [0]=e_div, [1]=e_wsci.
    Terms: s(div) + s(wsci) + te(div, wsci)   -- the interaction surface.
    Also fits the additive-only reduced model (no te term) for the
    AIC-based H4 comparison.
    """
    X = np.column_stack([np.asarray(e_div_c, dtype=float),
                          np.asarray(e_wsci_c, dtype=float)])
    y = np.asarray(e_sd, dtype=float)

    full_terms = s(0, n_splines=N_SPLINES) + s(1, n_splines=N_SPLINES) + te(0, 1, n_splines=N_SPLINES)
    reduced_terms = s(0, n_splines=N_SPLINES) + s(1, n_splines=N_SPLINES)

    gam_full = _fit_gam(X, y, full_terms)
    gam_reduced = _fit_gam(X, y, reduced_terms)
    return gam_full, gam_reduced, X


def fit_gam_direct(e_sd, e_div_c, e_wsci_c, e_mean_c):
    """
    Direct effect model (npp_mean controlled). Columns:
    [0]=e_div, [1]=e_mean, [2]=e_wsci.
    Terms: s(div) + s(mean) + s(wsci) + te(div, wsci).
    """
    X = np.column_stack([np.asarray(e_div_c, dtype=float),
                          np.asarray(e_mean_c, dtype=float),
                          np.asarray(e_wsci_c, dtype=float)])
    y = np.asarray(e_sd, dtype=float)

    full_terms = (s(0, n_splines=N_SPLINES) + s(1, n_splines=N_SPLINES)
                  + s(2, n_splines=N_SPLINES) + te(0, 2, n_splines=N_SPLINES))
    reduced_terms = s(0, n_splines=N_SPLINES) + s(1, n_splines=N_SPLINES) + s(2, n_splines=N_SPLINES)

    gam_full = _fit_gam(X, y, full_terms)
    gam_reduced = _fit_gam(X, y, reduced_terms)
    return gam_full, gam_reduced, X


def _predict_with_ci(gam, X_pred, alpha=CI_ALPHA):
    """Wraps pyGAM prediction + confidence interval into (mean, lower, upper)."""
    mean = gam.predict(X_pred)
    ci = gam.confidence_intervals(X_pred, width=1 - alpha)
    return mean, ci[:, 0], ci[:, 1]


def diversity_partial_curve(gam, e_div_grid, w, div_idx=0, wsci_idx=1,
                             other_fixed=None, n_cols=2, alpha=CI_ALPHA):
    """
    Traces the diversity partial-dependence curve at a fixed
    (centered) WSCI value w: holds WSCI at w (and any other predictor
    at `other_fixed`), sweeps e_div over e_div_grid, and centers the
    curve so effect = 0 at e_div = 0 (matching the zero-mean
    convention used throughout this pipeline).

    other_fixed: dict {col_idx: value} for any additional predictor
    columns (e.g. e_mean in the direct model). Defaults those columns
    to 0 (their centered mean) if not supplied.
    """
    e_div_grid = np.asarray(e_div_grid, dtype=float)
    n = len(e_div_grid)

    X_pred = np.zeros((n, n_cols))
    X_pred[:, div_idx] = e_div_grid
    X_pred[:, wsci_idx] = w
    if other_fixed:
        for idx, val in other_fixed.items():
            X_pred[:, idx] = val

    mean, lo, hi = _predict_with_ci(gam, X_pred, alpha)

    # Baseline row: e_div = 0, everything else identical, to center the curve.
    X_base = X_pred.copy()
    X_base[:, div_idx] = 0.0
    base_mean, base_lo, base_hi = _predict_with_ci(gam, X_base, alpha)
    base = base_mean[0]

    effect = mean - base
    # Propagate each point's own half-width onto the contrast (approximate,
    # see module docstring point 4) rather than pretending independence
    # gives an exact combined SE.
    half_width_pred = (hi - lo) / 2
    half_width_base = (base_hi[0] - base_lo[0]) / 2
    half_width = np.sqrt(half_width_pred ** 2 + half_width_base ** 2)

    ci_lower = effect - half_width
    ci_upper = effect + half_width

    # Numerical derivative at e_div = 0 (closest analogue to "the slope")
    h = max(1e-4, (e_div_grid.max() - e_div_grid.min()) / (2 * n))
    X_plus = np.zeros((1, n_cols)); X_plus[:, div_idx] = h; X_plus[:, wsci_idx] = w
    X_minus = np.zeros((1, n_cols)); X_minus[:, div_idx] = -h; X_minus[:, wsci_idx] = w
    if other_fixed:
        for idx, val in other_fixed.items():
            X_plus[:, idx] = val
            X_minus[:, idx] = val
    slope_at_mean = float((gam.predict(X_plus)[0] - gam.predict(X_minus)[0]) / (2 * h))

    return effect, ci_lower, ci_upper, slope_at_mean


def _johnson_neyman_scan_impl(gam, e_wsci_c_series, div_lo, div_hi, div_idx=0, wsci_idx=1,
                               other_fixed=None, n_cols=2, n_grid=JN_GRID_N, alpha=CI_ALPHA):
    """
    Numerical Johnson-Neyman analogue for a GAM. Fixes a diversity
    contrast (moving e_div from its low quantile to its high quantile,
    e.g. p10 -> p90, per DIV_CONTRAST_QUANTILES) and scans raw
    (centered) WSCI values, computing the contrast's predicted value
    and CI at each point. Returns the (centered) WSCI value(s) where
    the CI crosses zero, found by linear interpolation between
    adjacent grid points that flip significance -- the boundary of
    statistical significance for the diversity contrast, as a
    function of structural complexity.
    """
    w_grid = np.linspace(e_wsci_c_series.min(), e_wsci_c_series.max(), n_grid)

    X_hi = np.zeros((n_grid, n_cols)); X_hi[:, div_idx] = div_hi; X_hi[:, wsci_idx] = w_grid
    X_lo = np.zeros((n_grid, n_cols)); X_lo[:, div_idx] = div_lo; X_lo[:, wsci_idx] = w_grid
    if other_fixed:
        for idx, val in other_fixed.items():
            X_hi[:, idx] = val
            X_lo[:, idx] = val

    mean_hi, lo_hi, hi_hi = _predict_with_ci(gam, X_hi, alpha)
    mean_lo, lo_lo, hi_lo = _predict_with_ci(gam, X_lo, alpha)

    contrast = mean_hi - mean_lo
    hw = np.sqrt(((hi_hi - lo_hi) / 2) ** 2 + ((hi_lo - lo_lo) / 2) ** 2)
    ci_lower = contrast - hw
    ci_upper = contrast + hw

    # Significant where the CI excludes zero
    significant = (ci_lower > 0) | (ci_upper < 0)

    # Find sign-change boundaries via interpolation
    boundaries = []
    for i in range(len(w_grid) - 1):
        if significant[i] != significant[i + 1]:
            # linearly interpolate the crossing point between w_grid[i], w_grid[i+1]
            # using whichever bound (lower or upper) is closest to 0 at the flip
            f = lambda idx: ci_lower[idx] if abs(ci_lower[idx]) < abs(ci_upper[idx]) else ci_upper[idx]
            y0, y1 = f(i), f(i + 1)
            if y1 != y0:
                frac = -y0 / (y1 - y0)
                w_cross = w_grid[i] + frac * (w_grid[i + 1] - w_grid[i])
                boundaries.append(float(w_cross))

    return sorted(boundaries), significant, w_grid, contrast, ci_lower, ci_upper


def fit_metrics(df, diversity_metrics=DIVERSITY_METRICS,
                 stability_col=STABILITY_COL, slope_quantiles=SLOPE_QUANTILES):
    """
    Fits total and direct GAMs (full + additive-reduced) for every
    diversity metric, computes diversity partial curves at each
    requested WSCI cut, and a numerical Johnson-Neyman scan for both
    effect types. Fitting kept separate from plotting so per-metric
    y-limits can be computed before any figure is drawn.
    """
    e_sd = df[stability_col]
    e_wsci_c = df["e_wsci_c"]
    e_mean_c = df["e_mean_c"]

    results = {}
    for metric_name, col_name in diversity_metrics.items():
        print(f"[fit_metrics] Fitting GAMs for {metric_name}...")
        e_div_c = df[f"e_div_c__{col_name}"]

        gam_total_full, gam_total_reduced, X_total = fit_gam_total(e_sd, e_div_c, e_wsci_c)
        gam_direct_full, gam_direct_reduced, X_direct = fit_gam_direct(e_sd, e_div_c, e_wsci_c, e_mean_c)

        # H4 comparison: does adding the interaction surface improve fit
        # beyond an additive model? Lower AIC / GCV = better fit.
        aic_total_full, aic_total_reduced = gam_total_full.statistics_["AIC"], gam_total_reduced.statistics_["AIC"]
        aic_direct_full, aic_direct_reduced = gam_direct_full.statistics_["AIC"], gam_direct_reduced.statistics_["AIC"]

        w_values = {label: df["e_wsci_c"].quantile(q) for label, q in slope_quantiles.items()}

        curves_total = {}
        for label, w in w_values.items():
            curves_total[label] = (w,) + diversity_partial_curve(
                gam_total_full, _div_grid_placeholder(df, col_name), w,
                div_idx=0, wsci_idx=1, n_cols=2,
            )

        mean_c_val = 0.0  # e_mean is already centered; hold at its own mean
        curves_direct = {}
        for label, w in w_values.items():
            curves_direct[label] = (w,) + diversity_partial_curve(
                gam_direct_full, _div_grid_placeholder(df, col_name), w,
                div_idx=0, wsci_idx=2, other_fixed={1: mean_c_val}, n_cols=3,
            )

        div_col_c = f"e_div_c__{col_name}"
        div_lo = df[div_col_c].quantile(DIV_CONTRAST_QUANTILES[0])
        div_hi = df[div_col_c].quantile(DIV_CONTRAST_QUANTILES[1])

        jn_total = _johnson_neyman_scan_impl(
            gam_total_full, df["e_wsci_c"], div_lo, div_hi,
            div_idx=0, wsci_idx=1, n_cols=2,
        )
        jn_direct = _johnson_neyman_scan_impl(
            gam_direct_full, df["e_wsci_c"], div_lo, div_hi,
            div_idx=0, wsci_idx=2, other_fixed={1: mean_c_val}, n_cols=3,
        )

        results[metric_name] = {
            "n": len(df),
            "e_div_col_c": div_col_c,
            "gam_total_full": gam_total_full,
            "gam_total_reduced": gam_total_reduced,
            "gam_direct_full": gam_direct_full,
            "gam_direct_reduced": gam_direct_reduced,
            "aic_total_full": aic_total_full,
            "aic_total_reduced": aic_total_reduced,
            "aic_total_delta": aic_total_reduced - aic_total_full,  # positive => interaction improves fit
            "aic_direct_full": aic_direct_full,
            "aic_direct_reduced": aic_direct_reduced,
            "aic_direct_delta": aic_direct_reduced - aic_direct_full,
            "curves_total": curves_total,    # label -> (w, effect, ci_lower, ci_upper, slope_at_mean)
            "curves_direct": curves_direct,
            "jn_total_boundaries": jn_total[0],   # centered WSCI units
            "jn_direct_boundaries": jn_direct[0],
        }
    return results


def _div_grid_placeholder(df, col_name, n_grid=200):
    """Diversity grid (centered units) spanning the observed range for this metric."""
    x = df[f"e_div_c__{col_name}"].values.astype(float)
    return np.linspace(x.min(), x.max(), n_grid)


def compute_ylim_per_metric(all_metric_results, pad_frac=YLIM_PAD_FRAC):
    """
    Computes y-limits SEPARATELY FOR EACH DIVERSITY METRIC, rather
    than one range shared across all metrics. Within a metric, the
    total-effect panel and the direct-effect panel still get their
    own independent range (they are not forced to match each other,
    same as before) -- but neither range is pooled across metrics
    anymore.

    Diversity metrics can differ by an order of magnitude in effect
    size (e.g. Rao's Q vs. Simpson), so a single shared range tends to
    flatten the weaker metrics' curves to near-invisible flat lines.
    Per-metric scaling keeps every row's curve shape legible on its
    own axes, at the cost of cross-metric visual comparability -- use
    the summary CSV's slope/AIC columns if you need to compare
    magnitudes across metrics.

    Returns {metric_name: {"total": (ymin, ymax), "direct": (ymin, ymax)}}.
    """
    ylims = {}
    for metric_name, res in all_metric_results.items():
        metric_ylims = {}
        for curves_key, effect_label in [("curves_total", "total"), ("curves_direct", "direct")]:
            lows, highs = [], []
            for label, (w, effect, ci_lower, ci_upper, slope) in res[curves_key].items():
                lows.append(np.nanmin(ci_lower))
                highs.append(np.nanmax(ci_upper))
            ymin, ymax = min(lows), max(highs)
            pad = (ymax - ymin) * pad_frac
            metric_ylims[effect_label] = (ymin - pad, ymax + pad)
        ylims[metric_name] = metric_ylims
    return ylims


def plot_metrics(df, all_metric_results, diversity_metrics, ylims_per_metric,
                  savepath="gam_interaction_dependence.png"):
    """
    Combined figure: rows = diversity metric, columns = total effect /
    direct effect. Each panel overlays the diversity partial curve at
    each WSCI cut. Y-limits are looked up per metric (and per effect
    type) from ylims_per_metric -- see compute_ylim_per_metric.
    """
    n_metrics = len(diversity_metrics)
    fig, axes = plt.subplots(n_metrics, 2, figsize=(12, 4.2 * n_metrics), squeeze=False)
    colors = cm.viridis(np.linspace(0.15, 0.85, len(SLOPE_QUANTILES)))

    for row, metric_name in enumerate(diversity_metrics.keys()):
        res = all_metric_results[metric_name]
        col_name = diversity_metrics[metric_name]
        e_div_grid = _div_grid_placeholder(df, col_name)
        metric_ylims = ylims_per_metric[metric_name]

        for col, (curves_key, ylim_key, row_label, aic_delta_key) in enumerate([
            ("curves_total", "total", "total effect", "aic_total_delta"),
            ("curves_direct", "direct", "direct effect (npp_mean controlled)", "aic_direct_delta"),
        ]):
            ax = axes[row, col]
            ylim = metric_ylims[ylim_key]
            for (label, (w, effect, ci_lower, ci_upper, slope)), color in zip(res[curves_key].items(), colors):
                ax.plot(e_div_grid, effect, color=color, linewidth=2,
                        label=f"{label} (slope@0={slope:.3f})")
                ax.fill_between(e_div_grid, ci_lower, ci_upper, color=color, alpha=0.12)

            ax.set_ylim(ylim)
            ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
            ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
            ax.set_xlabel(f"{metric_name} residual", fontsize=9)
            ax.set_ylabel("Partial effect on NPP sd", fontsize=9)

            aic_delta = res[aic_delta_key]
            sig_flag = "*" if aic_delta > 2 else ""  # commonly used AIC-improvement heuristic
            ax.set_title(f"{metric_name} -- {row_label}\n"
                          f"interaction AIC improvement = {aic_delta:.2f}{sig_flag}", fontsize=9.5)
            ax.legend(fontsize=6, loc="best")

    fig.tight_layout()
    fig.savefig(savepath, dpi=150, bbox_inches="tight")
    return fig


def compute_effects_dataframe(df, all_metric_results, diversity_metrics,
                               stability_col=STABILITY_COL, mean_col=MEAN_COL,
                               wsci_col=WSCI_COL, pid_col=PID_COL, alpha=CI_ALPHA):
    """
    Per-observation effects: for every PID x diversity metric, predicts
    the GAM at that observation's own (e_div, e_wsci[, e_mean]) values
    and centers against the e_div = 0 baseline at that same WSCI --
    the continuous-moderator analogue of the per-PID effects CSV.
    """
    rows = []
    pids = df[pid_col].values
    wsci_raw = df[wsci_col].values
    wsci_c = df["e_wsci_c"].values
    mean_c = df["e_mean_c"].values
    npp_mean_vals = df[mean_col].values
    npp_sd_vals = df[stability_col].values

    for metric_name, col_name in diversity_metrics.items():
        res = all_metric_results[metric_name]
        e_div_c = df[res["e_div_col_c"]].values.astype(float)
        e_div_raw = df[col_name].values.astype(float)

        gam_total = res["gam_total_full"]
        gam_direct = res["gam_direct_full"]

        # Total model: X = [e_div, e_wsci]
        X_total_obs = np.column_stack([e_div_c, wsci_c])
        X_total_base = np.column_stack([np.zeros_like(e_div_c), wsci_c])
        mean_t, lo_t, hi_t = _predict_with_ci(gam_total, X_total_obs, alpha)
        base_t, _, _ = _predict_with_ci(gam_total, X_total_base, alpha)
        total_effect = mean_t - base_t

        # Direct model: X = [e_div, e_mean, e_wsci]
        X_direct_obs = np.column_stack([e_div_c, mean_c, wsci_c])
        X_direct_base = np.column_stack([np.zeros_like(e_div_c), mean_c, wsci_c])
        mean_d, lo_d, hi_d = _predict_with_ci(gam_direct, X_direct_obs, alpha)
        base_d, _, _ = _predict_with_ci(gam_direct, X_direct_base, alpha)
        direct_effect = mean_d - base_d

        for i in range(len(pids)):
            rows.append({
                pid_col: pids[i],
                "diversity_metric": metric_name,
                "diversity_residual": e_div_raw[i],
                "wsci_residual": wsci_raw[i],
                "npp_mean_residual": npp_mean_vals[i],
                "npp_sd_residual_actual": npp_sd_vals[i],
                "gam_total_effect": total_effect[i],
                "gam_total_effect_ci_lower": total_effect[i] - (hi_t[i] - lo_t[i]) / 2,
                "gam_total_effect_ci_upper": total_effect[i] + (hi_t[i] - lo_t[i]) / 2,
                "gam_direct_effect": direct_effect[i],
                "gam_direct_effect_ci_lower": direct_effect[i] - (hi_d[i] - lo_d[i]) / 2,
                "gam_direct_effect_ci_upper": direct_effect[i] + (hi_d[i] - lo_d[i]) / 2,
            })

    return pd.DataFrame(rows)


def build_summary_dataframe(all_metric_results, means, slope_quantiles=SLOPE_QUANTILES):
    """
    One row per diversity metric: AIC-based interaction test results,
    derivative-at-mean ("slope") at each WSCI cut for both effect
    types, and Johnson-Neyman boundaries converted back to raw WSCI
    units.
    """
    wsci_mean = means["wsci"]
    rows = []
    for metric_name, res in all_metric_results.items():
        row = {
            "diversity_metric": metric_name,
            "n": res["n"],
            "aic_total_full": res["aic_total_full"],
            "aic_total_reduced": res["aic_total_reduced"],
            "aic_total_delta": res["aic_total_delta"],
            "interaction_improves_fit_total": res["aic_total_delta"] > 2,
            "aic_direct_full": res["aic_direct_full"],
            "aic_direct_reduced": res["aic_direct_reduced"],
            "aic_direct_delta": res["aic_direct_delta"],
            "interaction_improves_fit_direct": res["aic_direct_delta"] > 2,
        }
        for label in slope_quantiles:
            w, effect, ci_lower, ci_upper, slope_t = res["curves_total"][label]
            w_d, effect_d, ci_lower_d, ci_upper_d, slope_d = res["curves_direct"][label]
            key = label.lower().replace(" ", "_").replace("(", "").replace(")", "")
            row[f"slope_at_mean_{key}_total"] = slope_t
            row[f"slope_at_mean_{key}_direct"] = slope_d

        jn_t_raw = [round(w + wsci_mean, 3) for w in res["jn_total_boundaries"]]
        jn_d_raw = [round(w + wsci_mean, 3) for w in res["jn_direct_boundaries"]]
        row["johnson_neyman_wsci_total"] = jn_t_raw if jn_t_raw else "none"
        row["johnson_neyman_wsci_direct"] = jn_d_raw if jn_d_raw else "none"

        rows.append(row)
    return pd.DataFrame(rows)


def run_interaction_analysis(df, diversity_metrics=DIVERSITY_METRICS,
                              stability_col=STABILITY_COL, mean_col=MEAN_COL,
                              wsci_col=WSCI_COL, pid_col=PID_COL, outdir="results"):
    """
    Top-level entry point:
      1. Centers e_div / e_mean / e_wsci and drops missing rows.
      2. Fits total/direct GAMs (full tensor-interaction + reduced
         additive) for every diversity metric on the full sample.
      3. Computes y-limits SEPARATELY FOR EACH DIVERSITY METRIC (see
         compute_ylim_per_metric) -- no longer one range shared across
         all metrics.
      4. Draws the combined dependence figure (partial curves at each
         of N_WSCI_CUTS WSCI levels, per metric, total vs. direct),
         with each row scaled to its own metric's y-range.
      5. Saves a CSV summary of AIC-based interaction comparisons,
         derivative-at-mean "slopes", and Johnson-Neyman boundaries.
      6. Saves a per-observation (per-PID) CSV of each diversity
         metric's effect at that observation's own WSCI value.
    """
    df_c, means = prepare_centered_data(df, diversity_metrics=diversity_metrics,
                                         stability_col=stability_col, mean_col=mean_col,
                                         wsci_col=wsci_col)

    all_results = fit_metrics(df_c, diversity_metrics=diversity_metrics,
                               stability_col=stability_col)

    ylims_per_metric = compute_ylim_per_metric(all_results)
    for metric_name, metric_ylims in ylims_per_metric.items():
        t = metric_ylims["total"]
        d = metric_ylims["direct"]
        print(f"[run_interaction_analysis] {metric_name}: "
              f"ylim total=({t[0]:.3g}, {t[1]:.3g}), direct=({d[0]:.3g}, {d[1]:.3g})")

    plot_metrics(df_c, all_results, diversity_metrics, ylims_per_metric,
                 savepath=f"{outdir}/gam_interaction_dependence.png")

    summary_df = build_summary_dataframe(all_results, means)
    summary_path = f"{outdir}/gam_interaction_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n[run_interaction_analysis] Summary saved to {summary_path}")
    print(summary_df.to_string(index=False))

    effects_df = compute_effects_dataframe(df_c, all_results, diversity_metrics,
                                            stability_col=stability_col, mean_col=mean_col,
                                            wsci_col=wsci_col, pid_col=pid_col)
    effects_path = f"{outdir}/gam_interaction_effects_per_observation.csv"
    effects_df.to_csv(effects_path, index=False)
    print(f"[run_interaction_analysis] Per-observation effects saved to {effects_path} "
          f"({len(effects_df)} rows: {len(df_c)} PIDs x {len(diversity_metrics)} metrics)")

    return all_results, summary_df, effects_df, ylims_per_metric


if __name__ == "__main__":
    residuals_df = pd.read_csv('results/step_2_residuals_merged2.csv')

    df = pd.read_csv('data/final/final_dataset_ba_v2.csv')
    # residuals_df = residuals_df.merge(df[['PID', 'pet_std']], on='PID', how='left')

    all_results, summary_df, effects_df, ylims_per_metric = run_interaction_analysis(residuals_df)

    plt.show()