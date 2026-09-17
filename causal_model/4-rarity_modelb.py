"""
Final stage of the pipeline using OLS linear regression, restricted
to a SINGLE diversity metric (Species richness), with a NESTED
segmentation:

  OUTER segmentation: pct_rare_species (default: quartiles) --
    produces one complete set of outputs per pct_rare_species bin.
  INNER segmentation: WSCI (default: tertiles) -- WITHIN each
    pct_rare_species bin, the species-richness -> NPP-stability
    relationship is re-fit separately per WSCI segment and plotted as
    overlaid lines, exactly like the original WSCI-segmented script.

In other words: this answers "does the species richness effect
depend on structural complexity (WSCI), and does THAT dependency
itself look different depending on how many rare species are in the
community (pct_rare_species)?" by producing a separate WSCI-segment
comparison figure for each pct_rare_species bin, rather than mixing
the two moderators into one model.

For each pct_rare_species bin you get (written to
{outdir}/pct_rare_<bin_label>/):
  1. One combined dependence-plot figure per WSCI segment within that
     bin (rows = Species richness, columns = total / direct effect).
  2. One WSCI-segment overlay comparison figure (2 rows x 1 column)
     for that pct_rare_species bin.
  3. A per-bin summary CSV and per-observation effects CSV.

A combined summary CSV and combined per-observation effects CSV
(tagged with both the pct_rare_species bin and the WSCI segment) are
also written to {outdir} directly, so you can compare across
pct_rare_species bins without digging into each subfolder.

Why quantile bins (qcut) rather than fixed-width bins by default:
  Small segments make any regression's estimates noisier. Quantile
  binning keeps sample size roughly balanced across segments even if
  the raw variable is skewed. Switch to fixed-width bins (method =
  "width") if equal-width categories are more interpretable for your
  write-up, but check n per bin before trusting the fit -- this
  matters MORE here than in the single-segmentation version, since
  nesting two 3-4-way splits can leave individual cells with only a
  few dozen rows.

Uses statsmodels (pip install statsmodels) and scipy.
"""

import os
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
    "Species richness": "residual_species_richness",
}
STABILITY_COL = "residual_std_npp"     # e_sd
MEAN_COL = "residual_mean"             # e_mean
WSCI_COL = "residual_wsci"                   # INNER segmenting variable (structural complexity)
PCT_RARE_COL = "pct_rare_species"      # OUTER segmenting variable (one figure set per bin)
PID_COL = "PID"                        # per-observation identifier, carried through to the effects CSV

# ---------------------------------------------------------------
# Segmentation settings
# ---------------------------------------------------------------

N_WSCI_GROUPS = 3                  # inner: e.g. 3 -> low / medium / high WSCI
WSCI_BIN_METHOD = "quantile"       # "quantile" (qcut, balanced n) or "width" (cut, equal WSCI range)

N_PCT_RARE_GROUPS = 2             # outer: e.g. 4 -> quartiles of pct_rare_species
PCT_RARE_BIN_METHOD = "quantile"

MIN_N_PER_GROUP = 30               # warn if any inner-x-outer cell has fewer rows than this
GROUP_LABELS = None                # None -> auto-generate labels from bin edges
YLIM_PAD_FRAC = 0.08               # padding added above/below the shared y-limit, as a fraction of its range
CI_ALPHA = 0.05                    # 95% confidence bands


def assign_quantile_groups(df, col, n_groups, method="quantile", labels=None,
                            group_col_name="group", min_n_per_group=MIN_N_PER_GROUP,
                            segment_kind="segment"):
    """
    Generic version of the tertile-binning helper: adds a
    `group_col_name` categorical column to df (a copy is returned;
    original df is not mutated), binning `col` into `n_groups` via
    quantiles (qcut) or equal-width bins (cut). Drops rows with
    missing `col`.
    """
    df = df.copy()
    n_missing = df[col].isna().sum()
    if n_missing:
        print(f"[assign_quantile_groups] Dropping {n_missing} rows with missing {col}")
    df = df.dropna(subset=[col])

    if method == "quantile":
        df[group_col_name], bin_edges = pd.qcut(
            df[col], q=n_groups, labels=labels, retbins=True, duplicates="drop"
        )
    elif method == "width":
        df[group_col_name], bin_edges = pd.cut(
            df[col], bins=n_groups, labels=labels, retbins=True
        )
    else:
        raise ValueError("method must be 'quantile' or 'width'")

    if labels is None:
        cats = df[group_col_name].cat.categories
        edge_labels = [f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}" for i in range(len(cats))]
        df[group_col_name] = df[group_col_name].cat.rename_categories(edge_labels)

    counts = df[group_col_name].value_counts().sort_index()
    print(f"[assign_quantile_groups] Sample size per {segment_kind}:")
    print(counts.to_string())
    for grp, n in counts.items():
        if n < min_n_per_group:
            print(f"  WARNING: {segment_kind} '{grp}' has only {n} rows (< {min_n_per_group}); "
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
    """Grid-based partial effect: returns (XX, pdep, confi) with XX shaped (n_grid, 1)."""
    x_col = X[term_name].values.astype(float)
    x_bar = x_col.mean()
    grid = np.linspace(x_col.min(), x_col.max(), n_grid)
    effect, ci_lower, ci_upper = _term_centered_effect(model, term_name, grid, x_bar)
    XX = grid.reshape(-1, 1)
    confi = np.column_stack([ci_lower, ci_upper])
    return XX, effect, confi


def fit_metrics_for_group(df, diversity_metrics=DIVERSITY_METRICS,
                           stability_col=STABILITY_COL, mean_col=MEAN_COL):
    """
    Fits (but does not plot) the total-effect and direct-effect OLS
    model for every diversity metric (just Species richness here), on
    whatever subset of df is passed in (i.e. one WSCI segment within
    one pct_rare_species bin).
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
    For one WSCI segment's already-fitted results, evaluates each
    fitted OLS model's centered partial effect (with its confidence
    band) at every actual observation, tied back to PID.
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


def compute_ylim_for_effect(all_results, model_key, pad_frac=YLIM_PAD_FRAC):
    """
    Scans every fitted model of a single effect type across every
    WSCI segment (within one pct_rare_species bin) and returns a
    padded (ymin, ymax) for that effect type only.
    """
    X_key = "X_total" if model_key == "model_total" else "X_direct"
    lows, highs = [], []
    for metric_results in all_results.values():
        for res in metric_results.values():
            _, _, confi = compute_partial_dependence(res[model_key], res[X_key])
            lows.append(np.nanmin(confi[:, 0]))
            highs.append(np.nanmax(confi[:, 1]))
    ymin, ymax = min(lows), max(highs)
    pad = (ymax - ymin) * pad_frac
    return (ymin - pad, ymax + pad)


def plot_dependence_on_ax(ax, model, X, feature_name, title, color="#D85A30"):
    """Per-segment figure panel: auto-scales its own y-axis."""
    XX, pdep, confi = compute_partial_dependence(model, X)
    p_value = get_term_pvalue(model, "e_div")

    ax.plot(XX[:, 0], pdep, color=color, linewidth=2)
    ax.fill_between(XX[:, 0], confi[:, 0], confi[:, 1], color=color, alpha=0.2)

    x_obs = X["e_div"].values.astype(float)
    ax.scatter(x_obs, np.full_like(x_obs, ax.get_ylim()[0]), marker="|",
               color="gray", alpha=0.4, s=30)

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


def plot_metrics_for_group(results, diversity_metrics, savepath, suptitle=None):
    """
    Draws the combined figure (rows = diversity metric -- just Species
    richness here, columns = total / direct effect) for one WSCI
    segment's already-fitted results.
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
    fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def _plot_effect_row(axes_row, all_results, diversity_metrics, group_order,
                      colors, model_key, X_key, p_key, row_label, ylim):
    """
    Draws one row of panels (one per diversity metric -- just Species
    richness here), each overlaying every WSCI segment's fitted line
    for a single effect type, all sharing the same ylim.
    """
    for col, metric_name in enumerate(diversity_metrics.keys()):
        ax = axes_row[col]
        for grp, color in zip(group_order, colors):
            if grp not in all_results or metric_name not in all_results[grp]:
                continue
            res = all_results[grp][metric_name]
            XX, pdep, confi = compute_partial_dependence(res[model_key], res[X_key])
            ax.plot(XX[:, 0], pdep, color=color, linewidth=2,
                    label=f"{grp} (n={res['n']}, p={res[p_key]:.2g})")
            ax.fill_between(XX[:, 0], confi[:, 0], confi[:, 1], color=color, alpha=0.12)

        ax.set_ylim(ylim)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_xlabel(f"{metric_name} residual", fontsize=9)
        ax.set_ylabel("Partial effect on NPP sd", fontsize=9)
        ax.set_title(f"{metric_name} -- {row_label} by WSCI segment", fontsize=10)
        ax.legend(fontsize=7, loc="best")


def plot_group_comparison(all_results, diversity_metrics, group_order,
                           ylim_total, ylim_direct, savepath):
    """
    Overlays every WSCI segment's partial-effect line on one set of
    axes -- one row for the total effect, one row for the direct
    effect. With only one diversity metric, this is a 2x1 figure.
    """
    n_metrics = len(diversity_metrics)
    fig, axes = plt.subplots(2, n_metrics, figsize=(6 * n_metrics, 9), squeeze=False)

    colors = cm.BrBG(np.linspace(0.15, 0.85, len(group_order)))

    _plot_effect_row(axes[0, :], all_results, diversity_metrics, group_order,
                      colors, "model_total", "X_total", "p_total", "total effect",
                      ylim_total)
    _plot_effect_row(axes[1, :], all_results, diversity_metrics, group_order,
                      colors, "model_direct", "X_direct", "p_direct",
                      "direct effect (npp_mean controlled)", ylim_direct)

    fig.tight_layout()
    fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def run_segmented_by_wsci(df, diversity_metrics=DIVERSITY_METRICS,
                           stability_col=STABILITY_COL, mean_col=MEAN_COL,
                           wsci_col=WSCI_COL, pid_col=PID_COL, outdir="results",
                           n_groups=N_WSCI_GROUPS, method=WSCI_BIN_METHOD,
                           label_prefix="ols_species_richness"):
    """
    INNER analysis, run once per pct_rare_species bin (or once total,
    if called directly without the outer loop):
      1. Bins wsci_col into segments.
      2. Fits Species richness's total/direct OLS model within each
         WSCI segment.
      3. Computes shared y-limits (total-effect row, direct-effect
         row) across WSCI segments.
      4. Draws the per-segment combined figure and the cross-segment
         comparison figure.
      5. Saves a CSV summary and a per-observation effects CSV.
    """
    os.makedirs(outdir, exist_ok=True)
    df_grouped, bin_edges = assign_quantile_groups(
        df, col=wsci_col, n_groups=n_groups, method=method,
        group_col_name="WSCI_group", segment_kind="WSCI segment",
    )
    group_order = list(df_grouped["WSCI_group"].cat.categories)

    all_results = {}
    effects_frames = []
    for grp in group_order:
        sub = df_grouped[df_grouped["WSCI_group"] == grp]
        if sub.empty:
            print(f"[run_segmented_by_wsci] Skipping empty WSCI segment '{grp}'")
            continue
        all_results[grp] = fit_metrics_for_group(
            sub, diversity_metrics=diversity_metrics,
            stability_col=stability_col, mean_col=mean_col,
        )
        effects_frames.append(compute_effects_dataframe(
            sub, grp, all_results[grp], diversity_metrics,
            stability_col=stability_col, mean_col=mean_col, pid_col=pid_col,
        ))

    for grp in group_order:
        if grp not in all_results:
            continue
        savepath = f"{outdir}/{label_prefix}_WSCI_{str(grp).replace(' ', '_').replace('/', '-')}.png"
        plot_metrics_for_group(
            all_results[grp], diversity_metrics, savepath=savepath,
            suptitle=f"WSCI segment: {grp}  (n={all_results[grp][next(iter(diversity_metrics))]['n']})",
        )

    ylim_total = compute_ylim_for_effect(all_results, "model_total")
    ylim_direct = compute_ylim_for_effect(all_results, "model_direct")
    print(f"[run_segmented_by_wsci] Shared ylim, total-effect row: "
          f"({ylim_total[0]:.3g}, {ylim_total[1]:.3g})")
    print(f"[run_segmented_by_wsci] Shared ylim, direct-effect row: "
          f"({ylim_direct[0]:.3g}, {ylim_direct[1]:.3g})")

    rows = []
    for grp, metric_results in all_results.items():
        for metric_name, res in metric_results.items():
            rows.append({
                "WSCI_segment": grp,
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
    summary_path = f"{outdir}/{label_prefix}_WSCI_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"[run_segmented_by_wsci] Summary saved to {summary_path}")
    print(summary_df.to_string(index=False))

    effects_df = pd.concat(effects_frames, ignore_index=True)
    effects_path = f"{outdir}/{label_prefix}_WSCI_effects_per_observation.csv"
    effects_df.to_csv(effects_path, index=False)
    print(f"[run_segmented_by_wsci] Per-observation effects saved to {effects_path} "
          f"({len(effects_df)} rows)")

    plot_group_comparison(all_results, diversity_metrics, group_order,
                           ylim_total=ylim_total, ylim_direct=ylim_direct,
                           savepath=f"{outdir}/{label_prefix}_WSCI_segment_comparison.png")

    return all_results, summary_df, effects_df, group_order, ylim_total, ylim_direct


def run_nested_segmented_analysis(df, diversity_metrics=DIVERSITY_METRICS,
                                   stability_col=STABILITY_COL, mean_col=MEAN_COL,
                                   wsci_col=WSCI_COL, pct_rare_col=PCT_RARE_COL,
                                   pid_col=PID_COL, outdir="results",
                                   n_wsci_groups=N_WSCI_GROUPS, wsci_method=WSCI_BIN_METHOD,
                                   n_pct_rare_groups=N_PCT_RARE_GROUPS,
                                   pct_rare_method=PCT_RARE_BIN_METHOD):
    """
    Top-level entry point (OUTER x INNER segmentation):
      1. Bins pct_rare_col into segments (OUTER, e.g. quartiles).
      2. For each pct_rare_species bin, runs the full WSCI-segmented
         analysis (run_segmented_by_wsci) on that subset -- its own
         set of per-WSCI-segment figures, comparison figure, summary
         CSV, and effects CSV, written under
         {outdir}/pct_rare_<bin_label>/.
      3. Stitches every bin's summary and effects tables together
         into one combined CSV each (tagged with the pct_rare_species
         bin), written directly under {outdir}.
    """
    os.makedirs(outdir, exist_ok=True)
    df_pct, pct_edges = assign_quantile_groups(
        df, col=pct_rare_col, n_groups=n_pct_rare_groups, method=pct_rare_method,
        group_col_name="PctRare_group", segment_kind="pct_rare_species segment",
    )
    pct_groups = list(df_pct["PctRare_group"].cat.categories)

    combined_summary_frames = []
    combined_effects_frames = []

    for pct_grp in pct_groups:
        sub = df_pct[df_pct["PctRare_group"] == pct_grp]
        if sub.empty:
            print(f"[run_nested_segmented_analysis] Skipping empty pct_rare_species bin '{pct_grp}'")
            continue

        print(f"\n{'='*70}\n[run_nested_segmented_analysis] pct_rare_species bin: {pct_grp} "
              f"(n={len(sub)})\n{'='*70}")

        bin_tag = str(pct_grp).replace(' ', '_').replace('/', '-')
        bin_outdir = f"{outdir}/pct_rare_{bin_tag}"

        _, summary_df, effects_df, _, _, _ = run_segmented_by_wsci(
            sub, diversity_metrics=diversity_metrics, stability_col=stability_col,
            mean_col=mean_col, wsci_col=wsci_col, pid_col=pid_col, outdir=bin_outdir,
            n_groups=n_wsci_groups, method=wsci_method,
            label_prefix="ols_species_richness",
        )

        summary_df = summary_df.copy()
        summary_df["pct_rare_segment"] = str(pct_grp)
        effects_df = effects_df.copy()
        effects_df["pct_rare_segment"] = str(pct_grp)

        combined_summary_frames.append(summary_df)
        combined_effects_frames.append(effects_df)

    combined_summary = pd.concat(combined_summary_frames, ignore_index=True)
    combined_summary_path = f"{outdir}/ols_species_richness_nested_pct_rare_x_WSCI_summary.csv"
    combined_summary.to_csv(combined_summary_path, index=False)
    print(f"\n[run_nested_segmented_analysis] Combined summary saved to {combined_summary_path}")
    print(combined_summary.to_string(index=False))

    combined_effects = pd.concat(combined_effects_frames, ignore_index=True)
    combined_effects_path = f"{outdir}/ols_species_richness_nested_pct_rare_x_WSCI_effects_per_observation.csv"
    combined_effects.to_csv(combined_effects_path, index=False)
    print(f"[run_nested_segmented_analysis] Combined per-observation effects saved to "
          f"{combined_effects_path} ({len(combined_effects)} rows)")

    return combined_summary, combined_effects, pct_groups


if __name__ == "__main__":
    residuals_df = pd.read_csv('results/step_2_residuals_merged2.csv')

    abundance_df = pd.read_parquet('data/final/dataset_relba.parquet')
    residuals_df = residuals_df.merge(abundance_df, on='PID', how='inner')

    # wsci_df = pd.read_csv('data/final/final_dataset_ba_v2.csv')
    # residuals_df = residuals_df.merge(wsci_df[['PID', 'pet_std']], on='PID', how='left')

    combined_summary, combined_effects, pct_groups = run_nested_segmented_analysis(residuals_df)

    plt.show()