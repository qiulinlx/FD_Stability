"""
Final stage of the pipeline using OLS linear regression with a
diversity x WSCI INTERACTION TERM, looped across all diversity
metrics, to test whether the diversity -> NPP-stability relationship
depends on structural complexity (H4) -- without discretizing WSCI
into tertile groups.

  Total effect:   e_sd ~ e_div + e_wsci + e_div:e_wsci
  Direct effect:  e_sd ~ e_div + e_mean + e_wsci + e_div:e_wsci

H4 is tested directly by the interaction coefficient's p-value: if
diversity's effect on stability is constant across structural
complexity, the interaction term should be ~0 and non-significant.

Both e_div and e_wsci (and e_mean) are mean-centered before fitting,
so:
  - the diversity main-effect coefficient = the diversity slope AT
    AVERAGE WSCI (not at WSCI = 0, which may be outside the data),
  - the WSCI main-effect coefficient = the WSCI slope at average
    diversity,
  - the interaction coefficient = how much the diversity slope
    changes per unit increase in WSCI.

Instead of a single fitted line, each panel plots three "simple
slopes" -- the diversity effect on stability at low / median / high
WSCI (10th / 50th / 90th percentile by default) -- with delta-method
confidence bands.

A Johnson-Neyman analysis is also computed per metric/effect: the
range of (raw) WSCI values over which the diversity effect is
statistically significant, found by solving where the simple-slope
confidence band stops excluding zero.

Y-AXIS LIMITS ARE PER DIVERSITY METRIC, NOT SHARED GLOBALLY. Each row
of the combined figure (one diversity metric) gets its own y-range for
its total-effect panel and its own y-range for its direct-effect
panel, computed from that metric's own simple-slope lines only.
Diversity metrics can differ by an order of magnitude in effect size,
so pooling the range across all metrics tends to flatten the weaker
ones to near-invisible flat lines; per-metric scaling keeps every
row's shape legible on its own axes. If you need cross-metric
magnitude comparisons, read them off the summary CSV's beta/slope
columns rather than off the plot axes.

Uses statsmodels (pip install statsmodels) and scipy.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
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
WSCI_COL = "residual_wsci"           # e_wsci (continuous moderator)
PID_COL = "PID"                      # per-observation identifier, carried through to the effects CSV

# ---------------------------------------------------------------
# Simple-slope / Johnson-Neyman settings
# ---------------------------------------------------------------

# Representative WSCI levels at which to plot/report the diversity
# slope, given as quantiles of e_wsci. Labels are used in legends/CSV.
SLOPE_QUANTILES = {
    "Low WSCI (p10)": 0.10,
    "Median WSCI": 0.50,
    "High WSCI (p90)": 0.90,
}
YLIM_PAD_FRAC = 0.08                 # padding above/below each metric's y-limit, as a fraction of its range
CI_ALPHA = 0.05                      # 95% confidence bands / significance tests


def prepare_centered_data(df, diversity_metrics=DIVERSITY_METRICS,
                           stability_col=STABILITY_COL, mean_col=MEAN_COL,
                           wsci_col=WSCI_COL):
    """
    Drops rows missing any needed column, then adds mean-centered
    versions of e_wsci, e_mean, and each diversity metric. Centering
    is required for the main-effect coefficients (and the simple
    slopes below) to be interpretable, and it reduces collinearity
    between the main effects and the interaction term.
    Returns (df_centered, means) where means is a dict of the raw
    column means used for centering (needed to convert Johnson-Neyman
    boundaries back to raw WSCI units).
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


def fit_interaction_total(e_sd, e_div_c, e_wsci_c):
    """y = e_sd ~ e_div + e_wsci + e_div:e_wsci  (total effect model)."""
    X = pd.DataFrame({
        "e_div": np.asarray(e_div_c, dtype=float),
        "e_wsci": np.asarray(e_wsci_c, dtype=float),
    })
    X["e_div:e_wsci"] = X["e_div"] * X["e_wsci"]
    Xc = sm.add_constant(X, has_constant="add")
    model = sm.OLS(np.asarray(e_sd, dtype=float), Xc).fit()
    return model, X


def fit_interaction_direct(e_sd, e_div_c, e_wsci_c, e_mean_c):
    """y = e_sd ~ e_div + e_mean + e_wsci + e_div:e_wsci  (direct effect, npp_mean controlled)."""
    X = pd.DataFrame({
        "e_div": np.asarray(e_div_c, dtype=float),
        "e_mean": np.asarray(e_mean_c, dtype=float),
        "e_wsci": np.asarray(e_wsci_c, dtype=float),
    })
    X["e_div:e_wsci"] = X["e_div"] * X["e_wsci"]
    Xc = sm.add_constant(X, has_constant="add")
    model = sm.OLS(np.asarray(e_sd, dtype=float), Xc).fit()
    return model, X


def get_term_pvalue(model, term_name):
    return model.pvalues[term_name]


def simple_slope(model, w, div_name="e_div", int_name="e_div:e_wsci"):
    """
    Diversity slope (and its SE, via the delta method) at a given
    (centered) WSCI value w:
        slope(w) = beta_div + beta_interaction * w
        Var(slope(w)) = Var(beta_div) + w^2 * Var(beta_int)
                         + 2*w*Cov(beta_div, beta_int)
    """
    b_div = model.params[div_name]
    b_int = model.params[int_name]
    cov = model.cov_params()
    var = (cov.loc[div_name, div_name]
           + w ** 2 * cov.loc[int_name, int_name]
           + 2 * w * cov.loc[div_name, int_name])
    slope = b_div + b_int * w
    se = np.sqrt(max(var, 0.0))
    return slope, se


def compute_slope_effect(model, e_div_grid, w, div_name="e_div", int_name="e_div:e_wsci",
                          alpha=CI_ALPHA):
    """
    The diversity partial effect at a fixed WSCI level w, evaluated
    over a grid of (centered) diversity values, with a delta-method
    confidence band. Centered so effect = 0 at e_div = 0 (the
    diversity mean), matching the zero-mean convention used
    throughout this pipeline.
    """
    slope, se_slope = simple_slope(model, w, div_name, int_name)
    dof = model.df_resid
    t_crit = stats.t.ppf(1 - alpha / 2, dof)
    e_div_grid = np.asarray(e_div_grid, dtype=float)
    effect = slope * e_div_grid
    se_effect = np.abs(e_div_grid) * se_slope
    ci_lower = effect - t_crit * se_effect
    ci_upper = effect + t_crit * se_effect
    return effect, ci_lower, ci_upper, slope, se_slope


def johnson_neyman_bounds(model, div_name="e_div", int_name="e_div:e_wsci", alpha=CI_ALPHA):
    """
    Solves for the (centered) WSCI value(s) w at which the diversity
    simple slope's confidence interval crosses zero, i.e. where
        slope(w)^2 == t_crit^2 * Var(slope(w))
    This is a quadratic in w; returns a sorted list of real roots
    (0, 1, or 2 of them). Outside these roots (or everywhere, if no
    real roots exist and the slope is significant at w=0) the
    diversity effect is statistically significant at `alpha`.
    """
    b_div = model.params[div_name]
    b_int = model.params[int_name]
    cov = model.cov_params()
    var_div = cov.loc[div_name, div_name]
    var_int = cov.loc[int_name, int_name]
    cov_div_int = cov.loc[div_name, int_name]
    dof = model.df_resid
    t_crit = stats.t.ppf(1 - alpha / 2, dof)
    t2 = t_crit ** 2

    A = b_int ** 2 - t2 * var_int
    B = 2 * b_div * b_int - 2 * t2 * cov_div_int
    C = b_div ** 2 - t2 * var_div

    if abs(A) < 1e-12:
        if abs(B) < 1e-12:
            return []
        return [-C / B]

    disc = B ** 2 - 4 * A * C
    if disc < 0:
        return []
    sqrt_disc = np.sqrt(disc)
    roots = sorted([(-B - sqrt_disc) / (2 * A), (-B + sqrt_disc) / (2 * A)])
    return roots


def fit_metrics(df, diversity_metrics=DIVERSITY_METRICS,
                 stability_col=STABILITY_COL, wsci_mean=0.0,
                 slope_quantiles=SLOPE_QUANTILES):
    """
    Fits the total-effect and direct-effect interaction model for
    every diversity metric on the (already centered) df, and computes
    simple slopes at each requested WSCI quantile plus Johnson-Neyman
    bounds for both effect types. Fitting is kept separate from
    plotting so per-metric y-limits can be computed before any figure
    is drawn.
    """
    e_sd = df[stability_col]
    e_wsci_c = df["e_wsci_c"]
    e_mean_c = df["e_mean_c"]

    results = {}
    for metric_name, col_name in diversity_metrics.items():
        e_div_c = df[f"e_div_c__{col_name}"]

        model_total, X_total = fit_interaction_total(e_sd, e_div_c, e_wsci_c)
        model_direct, X_direct = fit_interaction_direct(e_sd, e_div_c, e_wsci_c, e_mean_c)

        w_values = {label: df["e_wsci_c"].quantile(q) for label, q in slope_quantiles.items()}

        slopes_total = {
            label: (*simple_slope(model_total, w), w)
            for label, w in w_values.items()
        }
        slopes_direct = {
            label: (*simple_slope(model_direct, w), w)
            for label, w in w_values.items()
        }

        jn_total = johnson_neyman_bounds(model_total)
        jn_direct = johnson_neyman_bounds(model_direct)

        results[metric_name] = {
            "n": len(df),
            "e_div_col_c": f"e_div_c__{col_name}",
            "model_total": model_total,
            "X_total": X_total,
            "model_direct": model_direct,
            "X_direct": X_direct,
            "p_interaction_total": get_term_pvalue(model_total, "e_div:e_wsci"),
            "p_interaction_direct": get_term_pvalue(model_direct, "e_div:e_wsci"),
            "p_div_at_mean_wsci_total": get_term_pvalue(model_total, "e_div"),
            "p_div_at_mean_wsci_direct": get_term_pvalue(model_direct, "e_div"),
            "slopes_total": slopes_total,     # label -> (slope, se, w)
            "slopes_direct": slopes_direct,
            "jn_total_centered": jn_total,    # roots in centered WSCI units
            "jn_direct_centered": jn_direct,
        }
    return results


def _div_grid_for(res, df, n_grid=200):
    col = res["e_div_col_c"]
    x = df[col].values.astype(float)
    return np.linspace(x.min(), x.max(), n_grid)


def compute_ylim_per_metric(df, all_metric_results, pad_frac=YLIM_PAD_FRAC):
    """
    Computes y-limits SEPARATELY FOR EACH DIVERSITY METRIC, rather
    than one range shared across all metrics. Within a metric, the
    total-effect panel and the direct-effect panel still get their
    own independent range (they are not forced to match each other)
    -- but neither range is pooled across metrics anymore.

    Diversity metrics can differ by an order of magnitude in effect
    size, so a single shared range tends to flatten the weaker
    metrics' lines to near-invisible flat lines. Per-metric scaling
    keeps every row's curve shape legible on its own axes, at the
    cost of cross-metric visual comparability -- use the summary
    CSV's beta/slope columns if you need to compare magnitudes across
    metrics.

    Returns {metric_name: {"total": (ymin, ymax), "direct": (ymin, ymax)}}.
    """
    ylims = {}
    for metric_name, res in all_metric_results.items():
        e_div_grid = _div_grid_for(res, df)
        metric_ylims = {}
        for model_key, slopes_key, effect_label in [
            ("model_total", "slopes_total", "total"),
            ("model_direct", "slopes_direct", "direct"),
        ]:
            model = res[model_key]
            lows, highs = [], []
            for label, (slope, se, w) in res[slopes_key].items():
                _, ci_lower, ci_upper, _, _ = compute_slope_effect(model, e_div_grid, w)
                lows.append(np.nanmin(ci_lower))
                highs.append(np.nanmax(ci_upper))
            ymin, ymax = min(lows), max(highs)
            pad = (ymax - ymin) * pad_frac
            metric_ylims[effect_label] = (ymin - pad, ymax + pad)
        ylims[metric_name] = metric_ylims
    return ylims


def plot_metrics(df, all_metric_results, diversity_metrics, ylims_per_metric,
                  savepath="ols_interaction_dependence.png"):
    """
    Combined figure: rows = diversity metric, columns = total effect /
    direct effect. Each panel overlays the diversity simple-slope line
    at low / median / high WSCI. Y-limits are looked up per metric
    (and per effect type) from ylims_per_metric -- see
    compute_ylim_per_metric.
    """
    n_metrics = len(diversity_metrics)
    fig, axes = plt.subplots(2, n_metrics, figsize=(4.6 * n_metrics, 8.5), squeeze=False)
    colors = cm.PRGn(np.linspace(0.15, 0.85, len(SLOPE_QUANTILES)))
 
    for col, metric_name in enumerate(diversity_metrics.keys()):
        res = all_metric_results[metric_name]
        e_div_grid = _div_grid_for(res, df)
        metric_ylims = ylims_per_metric[metric_name]
 
        for row, (model_key, slopes_key, ylim_key, row_label) in enumerate([
            ("model_total", "slopes_total", "total", "total effect"),
            ("model_direct", "slopes_direct", "direct", "direct effect (npp_mean controlled)"),
        ]):
            ax = axes[row, col]
            model = res[model_key]
            ylim = metric_ylims[ylim_key]
            for (label, (slope, se, w)), color in zip(res[slopes_key].items(), colors):
                effect, ci_lower, ci_upper, _, _ = compute_slope_effect(model, e_div_grid, w)
                dof = model.df_resid
                t_stat = slope / se if se > 0 else np.nan
                p_slope = 2 * (1 - stats.t.cdf(abs(t_stat), dof)) if se > 0 else np.nan
                ax.plot(e_div_grid, effect, color=color, linewidth=2,
                        label=f"{label} (slope={slope:.3f}, p={p_slope:.2g})")
                ax.fill_between(e_div_grid, ci_lower, ci_upper, color=color, alpha=0.12)

            ax.set_ylim(ylim)
            ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
            ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
            ax.set_xlabel(f"{metric_name} residual", fontsize=9)
            ax.set_ylabel("Partial effect on NPP sd", fontsize=9)

            p_int = res[f"p_interaction_{model_key.split('_')[1]}"]
            sig_flag = "*" if p_int < 0.05 else ""
            ax.set_title(f"{metric_name} -- {row_label}\n"
                          f"diversity x WSCI interaction p = {p_int:.3g}{sig_flag}", fontsize=9.5)
            ax.legend(fontsize=6.5, loc="best")

    fig.tight_layout()
    fig.savefig(savepath, dpi=150, bbox_inches="tight")
    return fig


def compute_effects_dataframe(df, all_metric_results, diversity_metrics,
                               stability_col=STABILITY_COL, mean_col=MEAN_COL,
                               wsci_col=WSCI_COL, pid_col=PID_COL):
    """
    Per-observation effects: for every PID x diversity metric, uses
    that observation's OWN (continuous) WSCI value to compute the
    diversity partial effect via the interaction model's simple slope
    at that exact WSCI level -- not a fixed tertile or quantile.
    """
    rows = []
    pids = df[pid_col].values
    wsci_raw = df[wsci_col].values
    wsci_c = df["e_wsci_c"].values
    npp_mean_vals = df[mean_col].values
    npp_sd_vals = df[stability_col].values

    for metric_name, col_name in diversity_metrics.items():
        res = all_metric_results[metric_name]
        e_div_c = df[res["e_div_col_c"]].values.astype(float)
        e_div_raw = df[col_name].values.astype(float)

        model_total = res["model_total"]
        model_direct = res["model_direct"]

        for i in range(len(pids)):
            w_i = wsci_c[i]
            eff_t, se_t = simple_slope(model_total, w_i)
            eff_d, se_d = simple_slope(model_direct, w_i)
            dof_t = model_total.df_resid
            dof_d = model_direct.df_resid
            t_crit_t = stats.t.ppf(1 - CI_ALPHA / 2, dof_t)
            t_crit_d = stats.t.ppf(1 - CI_ALPHA / 2, dof_d)

            total_effect = eff_t * e_div_c[i]
            total_se = abs(e_div_c[i]) * se_t
            direct_effect = eff_d * e_div_c[i]
            direct_se = abs(e_div_c[i]) * se_d

            rows.append({
                pid_col: pids[i],
                "diversity_metric": metric_name,
                "diversity_residual": e_div_raw[i],
                "wsci_residual": wsci_raw[i],
                "npp_mean_residual": npp_mean_vals[i],
                "npp_sd_residual_actual": npp_sd_vals[i],
                "diversity_slope_at_this_wsci_total": eff_t,
                "ols_total_effect": total_effect,
                "ols_total_effect_ci_lower": total_effect - t_crit_t * total_se,
                "ols_total_effect_ci_upper": total_effect + t_crit_t * total_se,
                "diversity_slope_at_this_wsci_direct": eff_d,
                "ols_direct_effect": direct_effect,
                "ols_direct_effect_ci_lower": direct_effect - t_crit_d * direct_se,
                "ols_direct_effect_ci_upper": direct_effect + t_crit_d * direct_se,
            })

    return pd.DataFrame(rows)


def build_summary_dataframe(all_metric_results, means, slope_quantiles=SLOPE_QUANTILES):
    """
    One row per diversity metric: interaction test results, slopes at
    each requested WSCI level (both effect types), and Johnson-Neyman
    boundaries converted back to raw WSCI units.
    """
    wsci_mean = means["wsci"]
    rows = []
    for metric_name, res in all_metric_results.items():
        row = {
            "diversity_metric": metric_name,
            "n": res["n"],
            "beta_div_at_mean_wsci_total": res["model_total"].params["e_div"],
            "p_div_at_mean_wsci_total": res["p_div_at_mean_wsci_total"],
            "beta_interaction_total": res["model_total"].params["e_div:e_wsci"],
            "p_interaction_total": res["p_interaction_total"],
            "r2_total": res["model_total"].rsquared,
            "beta_div_at_mean_wsci_direct": res["model_direct"].params["e_div"],
            "p_div_at_mean_wsci_direct": res["p_div_at_mean_wsci_direct"],
            "beta_interaction_direct": res["model_direct"].params["e_div:e_wsci"],
            "p_interaction_direct": res["p_interaction_direct"],
            "r2_direct": res["model_direct"].rsquared,
        }
        for label in slope_quantiles:
            slope_t, se_t, w_t = res["slopes_total"][label]
            slope_d, se_d, w_d = res["slopes_direct"][label]
            dof_t = res["model_total"].df_resid
            dof_d = res["model_direct"].df_resid
            p_t = 2 * (1 - stats.t.cdf(abs(slope_t / se_t), dof_t)) if se_t > 0 else np.nan
            p_d = 2 * (1 - stats.t.cdf(abs(slope_d / se_d), dof_d)) if se_d > 0 else np.nan
            key = label.split(" ")[0].lower()  # "low" / "median" / "high"
            row[f"slope_{key}_wsci_total"] = slope_t
            row[f"p_slope_{key}_wsci_total"] = p_t
            row[f"slope_{key}_wsci_direct"] = slope_d
            row[f"p_slope_{key}_wsci_direct"] = p_d

        jn_t_raw = [round(w + wsci_mean, 3) for w in res["jn_total_centered"]]
        jn_d_raw = [round(w + wsci_mean, 3) for w in res["jn_direct_centered"]]
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
      2. Fits total/direct interaction models for every diversity
         metric on the full sample (fitting only -- no plotting yet).
      3. Computes y-limits SEPARATELY FOR EACH DIVERSITY METRIC (see
         compute_ylim_per_metric) -- no longer one range shared
         across all metrics.
      4. Draws the combined dependence figure (simple slopes at low /
         median / high WSCI, per metric, total vs. direct), with each
         row scaled to its own metric's y-range.
      5. Saves a CSV summary of interaction tests, simple slopes, and
         Johnson-Neyman bounds per metric.
      6. Saves a per-observation (per-PID) CSV of each diversity
         metric's effect at that observation's own WSCI value.
    """
    df_c, means = prepare_centered_data(df, diversity_metrics=diversity_metrics,
                                         stability_col=stability_col, mean_col=mean_col,
                                         wsci_col=wsci_col)

    all_results = fit_metrics(df_c, diversity_metrics=diversity_metrics,
                               stability_col=stability_col)

    ylims_per_metric = compute_ylim_per_metric(df_c, all_results)
    for metric_name, metric_ylims in ylims_per_metric.items():
        t = metric_ylims["total"]
        d = metric_ylims["direct"]
        print(f"[run_interaction_analysis] {metric_name}: "
              f"ylim total=({t[0]:.3g}, {t[1]:.3g}), direct=({d[0]:.3g}, {d[1]:.3g})")

    plot_metrics(df_c, all_results, diversity_metrics, ylims_per_metric,
                 savepath=f"{outdir}/ols_interaction_dependence.png")

    summary_df = build_summary_dataframe(all_results, means)
    summary_path = f"{outdir}/ols_interaction_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n[run_interaction_analysis] Summary saved to {summary_path}")
    print(summary_df.to_string(index=False))

    effects_df = compute_effects_dataframe(df_c, all_results, diversity_metrics,
                                            stability_col=stability_col, mean_col=mean_col,
                                            wsci_col=wsci_col, pid_col=pid_col)
    effects_path = f"{outdir}/ols_interaction_effects_per_observation.csv"
    effects_df.to_csv(effects_path, index=False)
    print(f"[run_interaction_analysis] Per-observation effects saved to {effects_path} "
          f"({len(effects_df)} rows: {len(df_c)} PIDs x {len(diversity_metrics)} metrics)")

    return all_results, summary_df, effects_df, ylims_per_metric


if __name__ == "__main__":
    residuals_df = pd.read_csv('results/step_2_residuals_merged2.csv')
    # wsci = pd.read_csv('data/processed/PID_location_WSCI.csv')
    # residuals_df = residuals_df.merge(wsci[['PID', 'WSCI']], on='PID', how='left')
    # Expect residuals_df to already contain a `residual_wsci` column
    # (WSCI residualized the same way as the other predictors upstream
    # in the pipeline). If not, compute/merge it in before calling
    # run_interaction_analysis.

    all_results, summary_df, effects_df, ylims_per_metric = run_interaction_analysis(residuals_df)

    plt.show()