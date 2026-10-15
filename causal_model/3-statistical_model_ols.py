"""
DML final-stage (target model) estimates of the effect of diversity on
NPP variability, consistent with the equations

  Individual effect (Eq. individual_effect), one model per metric j:
      S~ = theta_j * D~_j + eps

  Mutually adjusted (Eq. mutually_adjusted), one model with all metrics:
      S~ = sum_j theta_j * D~_j + eps

where S~ is the residualised stability outcome (STABILITY_COL) and D~_j
the residualised diversity metrics (DIVERSITY_METRICS), both produced by
the cross-fitted XGBoost nuisance models.

INPUTS / OUTPUTS
  All input paths, the output directory and every output file name are
  defined in the "Input and output files" block directly below the
  imports. Figures are saved as PNG plus a sibling PDF with the same stem.

NOTES
  * No intercept (INCLUDE_INTERCEPT = False), matching both equations.
  * R^2 is the CENTRED R^2 (statsmodels reports an uncentred R^2 when
    there is no intercept).
  * HC3 robust standard errors.
  * The outcome is on the log scale (log1p of NPP SD), so
    % change = 100 * (exp(theta_j * sd(D~_j)) - 1).
  * REPORT_AS_STABILITY = True flips the outcome's sign, so all thetas,
    CIs, % changes, CSVs and plots are in terms of stability (inverse SD):
    positive = diversity drives NPP SD down.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.tools import add_constant
from scipy import stats

# ===============================================================
# Input and output files  (edit names here only)
# ===============================================================

# Inputs
RAW_DATA_PATH = "data/final/final_dataset_ba.csv"
RESIDUALS_PATH = "results/step_2_residuals_merged.csv"

# Output directory
OUTDIR = "results/deciduous"

# Individual-effect model outputs
OUT_INDIVIDUAL_EFFECTS = "individual_effects.csv"
OUT_INDIVIDUAL_PER_OBS = "individual_effects_per_observation.csv"
OUT_INDIVIDUAL_SCENARIOS = "individual_scenario_effects.csv"
OUT_INDIVIDUAL_RANKING_FIG = "ranking_individual.png"          # + .pdf sibling

# Mutually adjusted model outputs
OUT_ADJUSTED_EFFECTS = "mutually_adjusted_effects.csv"
OUT_ADJUSTED_VIF = "mutually_adjusted_vif.csv"
OUT_ADJUSTED_PER_OBS = "mutually_adjusted_effects_per_observation.csv"
OUT_ADJUSTED_SCENARIOS = "mutually_adjusted_scenario_effects.csv"
OUT_ADJUSTED_RANKING_FIG = "ranking_mutually_adjusted.png"     # + .pdf sibling

# Relative importance (LMG) outputs
OUT_RELATIVE_IMPORTANCE = "relative_importance.csv"
OUT_RELATIVE_IMPORTANCE_FIG = "relative_importance.png"        # + .pdf sibling


def _out(outdir, filename):
    """Full path of an output file inside outdir."""
    return os.path.join(outdir, filename)


# ---------------------------------------------------------------
# Publication style
# ---------------------------------------------------------------
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#1a1a1a",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "legend.fontsize": 8,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

COLOR_POS = "#7ddfe7"   # light blue: increases stability (drives NPP SD down)
COLOR_NEG = "#4910d8"   # purple: decreases stability (drives NPP SD up)


def _save_publication_figure(fig, savepath_png):
    """Saves the figure as a 300 dpi PNG and a sibling vector PDF (skipped if savepath is None)."""
    if savepath_png is None:
        return
    fig.savefig(savepath_png, dpi=300, bbox_inches="tight")
    savepath_pdf = os.path.splitext(savepath_png)[0] + ".pdf"
    fig.savefig(savepath_pdf, bbox_inches="tight")
    print(f"  Saved {savepath_png} and {savepath_pdf}")


# ---------------------------------------------------------------
# Column names
# ---------------------------------------------------------------

DIVERSITY_METRICS = {
    "Rao's Q": "residual_raos_q",
    "Functional Evenness": "residual_functional_evenness",
    "Shannon": "residual_shannon",
    "Species richness": "residual_species_richness",
}
STABILITY_COL = "residual_std_npp"   # S~
PID_COL = "PID"

# ---------------------------------------------------------------
# Model settings (keep consistent with the equations)
# ---------------------------------------------------------------

INCLUDE_INTERCEPT = False   # equations have no intercept

REPORT_AS_STABILITY = True
OUTCOME_SIGN = -1 if REPORT_AS_STABILITY else 1

# Raw (un-residualised) data, used for the percentile scenario contrasts.
raw_df = pd.read_csv(RAW_DATA_PATH)

PID_df= pd.read_csv('data/lookup/PID_location_v3.csv')
raw_df=raw_df.merge(PID_df[['PID', 'STDAGE', 'percent_conifer']], on='PID', how='left')

raw_df.drop(columns=['Unnamed: 0', 'managed', 'ownership', 'DIA', 'TPA_UNADJ', "Simpson's Index",
                     'Functional_Richness', 'Shannon Equitabiltiy Index', 'transformed npp'],
            inplace=True, errors="ignore")
raw_df.dropna(subset=['std npp', 'mean', 'Raos_Q', 'Functional_Evenness', 'Soil Moisture',
                      'Species Richness', 'Shannon Diversity', "pet_std"],
              inplace=True)

raw_df=raw_df[raw_df.percent_conifer > 60]

CI_ALPHA = 0.05
FDR_ALPHA = 0.05
JOINT_VIF_WARN_THRESHOLD = 10

# Scale for the two ranking charts: each metric's effect as a % of the NET
# effect of diversity on stability.
#   "sum_abs"     : net = sum over metrics of |% change per 1 SD| from the
#                   same model, so each bar is a signed share of the total
#                   magnitude (|bars| sum to 100%). Always well defined.
#   "joint_shift" : net = % change in stability when ALL metrics increase
#                   by 1 SD together, from the mutually adjusted model.
#                   Unstable if opposing effects cancel (net close to 0).
NET_EFFECT_SCALE = "sum_abs"

# Relative importance (share of the net diversity effect)
N_BOOT_LMG = 200            # bootstrap resamples for LMG share CIs (0 to skip)
RANDOM_SEED = 42

# ---------------------------------------------------------------
# Ecological scenarios
# ---------------------------------------------------------------
# Raw (un-residualised) column for each diversity metric in raw_df, used
# to compute percentile contrasts in natural units.
RAW_METRIC_COLS = {
    "Rao's Q": "Raos_Q",
    "Functional Evenness": "Functional_Evenness",
    "Shannon": "Shannon Diversity",
    "Species richness": "Species Richness",
}
# Percentile contrast applied to every metric: low -> high diversity.
SCENARIO_PERCENTILES = (10, 90)
# Fixed changes in natural units: {metric: [(label, delta), ...]}.
FIXED_SCENARIOS = {
    "Species richness": [("Loss of 5 species", -5), ("Gain of 5 species", 5)],
    "Shannon": [("Decrease of 0.5 in Shannon", -0.5)],
}


# ---------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------

def _design(X):
    """Adds an intercept only if INCLUDE_INTERCEPT is True (equations have none)."""
    return sm.add_constant(X, has_constant="add") if INCLUDE_INTERCEPT else X


def _fit(y, X):
    """OLS with HC3 robust standard errors."""
    return sm.OLS(np.asarray(y, dtype=float), X).fit(cov_type="HC3")


def _crit(model, alpha=CI_ALPHA):
    """Critical value matching the model's inference (t for classical, z for robust)."""
    if getattr(model, "use_t", False):
        return stats.t.ppf(1 - alpha / 2, model.df_resid)
    return stats.norm.ppf(1 - alpha / 2)


def centred_r2(model, y):
    """Centred R^2 = 1 - SSR / sum((y - mean(y))^2), valid with or without an intercept."""
    y = np.asarray(y, dtype=float)
    return 1 - np.sum(np.asarray(model.resid) ** 2) / np.sum((y - y.mean()) ** 2)


def _to_pct(change):
    """Convert a change on the (log) outcome scale into a % change."""
    return (np.exp(change) - 1) * 100


def _pct_label(suffix=""):
    """x-axis label for the ranking charts, matching the reporting scale."""
    what = "stability (1/NPP SD)" if REPORT_AS_STABILITY else "NPP SD"
    return f"% change in {what} per 1 SD of " + r"$\tilde{D}_j$" + suffix


def _outcome_name():
    return "stability" if REPORT_AS_STABILITY else "NPP SD"


def _orient_outcome(df, stability_col=STABILITY_COL):
    """Applies OUTCOME_SIGN so that positive effects mean increased stability."""
    df = df.copy()
    df[stability_col] = OUTCOME_SIGN * df[stability_col].astype(float)
    return df


def _term_centered_effect(model, term_name, x_values, x_bar, alpha=CI_ALPHA):
    """Centred fitted contribution theta * (x - x_bar) with a pointwise CI."""
    theta = model.params[term_name]
    se = model.bse[term_name]
    crit = _crit(model, alpha)
    x_values = np.asarray(x_values, dtype=float)
    effect = theta * (x_values - x_bar)
    se_effect = np.abs(x_values - x_bar) * se
    return effect, effect - crit * se_effect, effect + crit * se_effect


# ---------------------------------------------------------------
# Individual-effect models:  S~ = theta_j D~_j + eps
# ---------------------------------------------------------------

def fit_individual(s, d):
    """Fits S~ = theta_j D~_j + eps for one diversity metric."""
    X = pd.DataFrame({"D": np.asarray(d, dtype=float)})
    y = np.asarray(s, dtype=float)
    return _fit(y, _design(X)), X, y


def fit_individual_models(df, diversity_metrics=DIVERSITY_METRICS, stability_col=STABILITY_COL):
    """One individual-effect model per diversity metric."""
    results = {}
    for metric_name, col_name in diversity_metrics.items():
        model, X, y = fit_individual(df[stability_col], df[col_name])
        results[metric_name] = {
            "n": len(df), "model": model, "X": X, "y": y,
            "theta": model.params["D"], "p": model.pvalues["D"],
        }
    return results


# ---------------------------------------------------------------
# Ranking figure
# ---------------------------------------------------------------

def joint_shift_pct(df, diversity_metrics=DIVERSITY_METRICS, stability_col=STABILITY_COL):
    """% change in stability when every D~_j rises by 1 SD together (mutually adjusted model)."""
    model, X, _ = fit_mutually_adjusted(df, diversity_metrics, stability_col)
    return _to_pct(sum(model.params[t] * X[t].std(ddof=1) for t in X.columns))


def add_net_scale(effect_df, net):
    """
    Adds rel_net, rel_net_ci_lower, rel_net_ci_upper: each metric's % change
    (and CI) expressed as a % of |net|. The sign of each bar is kept, so
    positive still means stabilising. The denominator is treated as fixed,
    so these CIs reflect uncertainty in each metric's effect only.
    """
    out = effect_df.copy()
    denom = abs(net)
    for src, dst in [("theta_pct", "rel_net"), ("theta_pct_ci_lower", "rel_net_ci_lower"),
                     ("theta_pct_ci_upper", "rel_net_ci_upper")]:
        out[dst] = 100 * out[src] / denom if denom > 0 else np.nan
    out["net_effect_pct"] = net
    out["net_effect_definition"] = NET_EFFECT_SCALE
    return out


def _net_for(effect_df, df):
    """Net diversity effect used as the denominator, per NET_EFFECT_SCALE."""
    if NET_EFFECT_SCALE == "joint_shift":
        net = joint_shift_pct(df)
        if abs(net) < 1e-3 * effect_df["theta_pct"].abs().max():
            print("  WARNING: joint-shift net effect is ~0 (opposing effects cancel); "
                  "relative values will be unstable. Consider NET_EFFECT_SCALE = 'sum_abs'.")
        return net
    return effect_df["theta_pct"].abs().sum()


def _net_label(suffix=""):
    if NET_EFFECT_SCALE == "joint_shift":
        base = "Effect as % of net diversity effect\n(all metrics +1 SD together)"
    else:
        base = "Effect as % of net diversity effect\n(sum of absolute effects)"
    return base + suffix


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


# ---------------------------------------------------------------
# Summaries and % change effect sizes
# ---------------------------------------------------------------

def _pct_rows(model, X, terms, alpha=CI_ALPHA):
    """theta_j with CI, plus % change in NPP SD per 1 SD of each D~_j."""
    rows = []
    ci = model.conf_int(alpha=alpha)
    for term, label in terms:
        x_sd = X[term].astype(float).std(ddof=1)
        lo, hi = ci.loc[term].iloc[0], ci.loc[term].iloc[1]
        rows.append({
            "diversity_metric": label,
            "theta": model.params[term],
            "theta_ci_lower": lo,
            "theta_ci_upper": hi,
            "theta_pct": _to_pct(model.params[term] * x_sd),
            "theta_pct_ci_lower": _to_pct(lo * x_sd),
            "theta_pct_ci_upper": _to_pct(hi * x_sd),
            "p_value_raw": model.pvalues[term],
        })
    return rows


def _add_fdr(df, fdr_alpha=FDR_ALPHA):
    reject, p_adj, _, _ = multipletests(df["p_value_raw"].values, alpha=fdr_alpha, method="fdr_bh")
    df["p_value_fdr"] = p_adj
    df["significant_fdr"] = reject
    return df


def build_individual_summary(all_results):
    rows = []
    for metric_name, res in all_results.items():
        row = _pct_rows(res["model"], res["X"], [("D", metric_name)])[0]
        row.update({"model": "individual", "n": res["n"],
                    "r2_centred": centred_r2(res["model"], res["y"])})
        rows.append(row)
    return _add_fdr(pd.DataFrame(rows))


def compute_effects_dataframe(df, all_results, diversity_metrics=DIVERSITY_METRICS,
                              stability_col=STABILITY_COL, pid_col=PID_COL):
    """Per-PID fitted individual effect for every diversity metric (long format)."""
    rows = []
    pids, s_vals = df[pid_col].values, df[stability_col].values
    for metric_name, col_name in diversity_metrics.items():
        res = all_results[metric_name]
        d = df[col_name].values.astype(float)
        eff, lo, hi = _term_centered_effect(res["model"], "D", d, res["X"]["D"].mean())
        for i in range(len(pids)):
            rows.append({pid_col: pids[i], "diversity_metric": metric_name,
                         "diversity_residual": d[i], "stability_residual": s_vals[i],
                         "individual_effect": eff[i], "individual_effect_ci_lower": lo[i],
                         "individual_effect_ci_upper": hi[i]})
    return pd.DataFrame(rows)


def compute_scenario_effects(summary_df, raw_df=None, residual_df=None,
                             raw_cols=RAW_METRIC_COLS, percentiles=SCENARIO_PERCENTILES,
                             fixed=FIXED_SCENARIOS, alpha=CI_ALPHA):
    """
    Effect of concrete changes in diversity, Delta D_j, on stability:
        % change = 100 * (exp(theta_j * Delta D_j) - 1)   (log outcome)
    with the CI obtained by applying the same transform to the CI of theta_j.

    Scenarios:
      * percentile contrast: Delta D_j = P_high - P_low of the RAW metric
      * fixed changes in natural units from FIXED_SCENARIOS
    Also reports Delta D_j relative to the SD of the RESIDUALISED metric,
    to flag scenarios that extrapolate well beyond the variation actually
    used to estimate theta_j (values >> 2 deserve a caveat).
    """
    lo_p, hi_p = percentiles
    rows = []
    for _, r in summary_df.iterrows():
        metric = r["diversity_metric"]
        scenarios = []
        if raw_df is not None and metric in raw_cols and raw_cols[metric] in raw_df:
            x = raw_df[raw_cols[metric]].astype(float).dropna()
            p_lo, p_hi = np.percentile(x, [lo_p, hi_p])
            scenarios.append((f"P{lo_p} -> P{hi_p} ({p_lo:.3g} -> {p_hi:.3g})", p_hi - p_lo))
        scenarios += fixed.get(metric, [])

        resid_sd = np.nan
        if residual_df is not None and metric in DIVERSITY_METRICS:
            resid_sd = residual_df[DIVERSITY_METRICS[metric]].astype(float).std(ddof=1)

        for label, delta in scenarios:
            ends = sorted([_to_pct(r["theta_ci_lower"] * delta), _to_pct(r["theta_ci_upper"] * delta)])
            rows.append({
                "diversity_metric": metric,
                "scenario": label,
                "delta_D": delta,
                "delta_in_residual_SDs": abs(delta) / resid_sd if resid_sd else np.nan,
                "pct_change": _to_pct(r["theta"] * delta),
                "pct_change_ci_lower": ends[0],
                "pct_change_ci_upper": ends[1],
                "significant_fdr": r.get("significant_fdr", np.nan),
            })
    return pd.DataFrame(rows)


def run_individual_effect_analysis(df, diversity_metrics=DIVERSITY_METRICS,
                                   stability_col=STABILITY_COL, pid_col=PID_COL, outdir=OUTDIR):
    """Individual-effect models (Eq. individual_effect) for every diversity metric."""
    os.makedirs(outdir, exist_ok=True)
    df = df.dropna(subset=[stability_col] + list(diversity_metrics.values())).reset_index(drop=True)
    df = _orient_outcome(df, stability_col)
    print(f"[run_individual_effect_analysis] n = {len(df)}")

    all_results = fit_individual_models(df, diversity_metrics, stability_col)

    summary_df = build_individual_summary(all_results)
    print(summary_df.to_string(index=False))

    summary_df = add_net_scale(summary_df, _net_for(summary_df, df))
    summary_df.to_csv(_out(outdir, OUT_INDIVIDUAL_EFFECTS), index=False)
    _plot_ranking_bar(summary_df,
                      xlabel=_net_label(),
                      title=f"Individual effects of diversity metrics on {_outcome_name()}",
                      savepath=_out(outdir, OUT_INDIVIDUAL_RANKING_FIG),
                      col="rel_net", decimals=0)

    effects_df = compute_effects_dataframe(df, all_results, diversity_metrics, stability_col, pid_col)
    effects_df.to_csv(_out(outdir, OUT_INDIVIDUAL_PER_OBS), index=False)
    scenario_df = compute_scenario_effects(summary_df, raw_df=raw_df, residual_df=df)
    scenario_df.to_csv(_out(outdir, OUT_INDIVIDUAL_SCENARIOS), index=False)
    print("\n[run_individual_effect_analysis] Scenario effects:")
    print(scenario_df.to_string(index=False))

    return all_results, summary_df, effects_df


# ---------------------------------------------------------------
# Mutually adjusted model:  S~ = sum_j theta_j D~_j + eps
# ---------------------------------------------------------------

def fit_mutually_adjusted(df, diversity_metrics=DIVERSITY_METRICS, stability_col=STABILITY_COL):
    """One model with all D~_j as simultaneous predictors (no intercept)."""
    X = df[list(diversity_metrics.values())].astype(float).copy()
    X.columns = list(diversity_metrics.keys())
    y = df[stability_col].astype(float).values
    return _fit(y, _design(X)), X, y


def compute_vifs(X, vif_warn_threshold=JOINT_VIF_WARN_THRESHOLD):
    """VIFs for the mutually adjusted design (computed with a constant, as is standard)."""
    Xc = add_constant(X)
    vif_df = pd.DataFrame({"variable": Xc.columns,
                           "VIF": [variance_inflation_factor(Xc.values, i) for i in range(Xc.shape[1])]})
    vif_df = vif_df[vif_df["variable"] != "const"].reset_index(drop=True)
    flagged = vif_df[vif_df["VIF"] > vif_warn_threshold]
    if len(flagged):
        print(f"  [compute_vifs] WARNING: VIF > {vif_warn_threshold} for "
              f"{', '.join(flagged['variable'])}; their theta_j are likely unstable.")
    return vif_df


def compute_adjusted_effects_dataframe(df, model, X, stability_col=STABILITY_COL, pid_col=PID_COL):
    """Per-PID fitted effect of each D~_j from the mutually adjusted model (long format)."""
    rows = []
    pids, s_vals = df[pid_col].values, df[stability_col].values
    for term in X.columns:
        eff, lo, hi = _term_centered_effect(model, term, X[term].values, X[term].mean())
        for i in range(len(pids)):
            rows.append({pid_col: pids[i], "diversity_metric": term,
                         "stability_residual": s_vals[i],
                         "adjusted_effect": eff[i], "adjusted_effect_ci_lower": lo[i],
                         "adjusted_effect_ci_upper": hi[i]})
    return pd.DataFrame(rows)


def run_mutually_adjusted_analysis(df, diversity_metrics=DIVERSITY_METRICS,
                                   stability_col=STABILITY_COL, pid_col=PID_COL, outdir=OUTDIR):
    """Mutually adjusted model (Eq. mutually_adjusted). Check VIFs before interpreting."""
    os.makedirs(outdir, exist_ok=True)
    df = df.dropna(subset=[stability_col] + list(diversity_metrics.values())).reset_index(drop=True)
    df = _orient_outcome(df, stability_col)

    model, X, y = fit_mutually_adjusted(df, diversity_metrics, stability_col)

    vif_df = compute_vifs(X)
    vif_df.to_csv(_out(outdir, OUT_ADJUSTED_VIF), index=False)
    print(vif_df.to_string(index=False))

    adj_df = _add_fdr(pd.DataFrame(_pct_rows(model, X, [(t, t) for t in X.columns])))
    adj_df["model"] = "mutually_adjusted"
    adj_df["n"] = len(df)
    adj_df["r2_centred"] = centred_r2(model, y)
    print(adj_df.to_string(index=False))

    adj_df = add_net_scale(adj_df, _net_for(adj_df, df))
    adj_df.to_csv(_out(outdir, OUT_ADJUSTED_EFFECTS), index=False)
    _plot_ranking_bar(adj_df,
                      xlabel=_net_label(", others held constant"),
                      title=f"Mutually adjusted effects of diversity metrics on {_outcome_name()}",
                      savepath=_out(outdir, OUT_ADJUSTED_RANKING_FIG),
                      col="rel_net", decimals=0)

    effects_df = compute_adjusted_effects_dataframe(df, model, X, stability_col, pid_col)
    effects_df.to_csv(_out(outdir, OUT_ADJUSTED_PER_OBS), index=False)
    scenario_df = compute_scenario_effects(adj_df, raw_df=raw_df, residual_df=df)
    scenario_df.to_csv(_out(outdir, OUT_ADJUSTED_SCENARIOS), index=False)
    ri_df = compute_relative_importance(df, model, diversity_metrics, stability_col)
    ri_df.to_csv(_out(outdir, OUT_RELATIVE_IMPORTANCE), index=False)
    print("\n[run_mutually_adjusted_analysis] Share of net diversity effect:")
    print(ri_df.to_string(index=False))
    plot_relative_importance(ri_df, savepath=_out(outdir, OUT_RELATIVE_IMPORTANCE_FIG))

    return model, vif_df, adj_df, effects_df


# ---------------------------------------------------------------
# Relative importance: share of the NET diversity effect (LMG / Shapley)
# ---------------------------------------------------------------

def _subset_r2(Xv, yv):
    """Centred R^2 of S~ on every subset of the D~_j (no intercept), keyed by bitmask."""
    # Variables are mean-centred so that every subset model's R^2 is
    # non-negative and nested models never lose fit (with residualised
    # inputs the means are ~0, so this barely changes the fits).
    p = Xv.shape[1]
    yv = yv - yv.mean()
    Xv = Xv - Xv.mean(axis=0)
    sst = np.sum(yv ** 2)
    r2 = {0: 0.0}
    for mask in range(1, 2 ** p):
        cols = [k for k in range(p) if mask >> k & 1]
        Xs = Xv[:, cols]
        coef, *_ = np.linalg.lstsq(Xs, yv, rcond=None)
        r2[mask] = 1 - np.sum((yv - Xs @ coef) ** 2) / sst
    return r2


def _lmg_shares(Xv, yv):
    """
    LMG / Shapley decomposition of the diversity block's R^2: each metric's
    average R^2 gain over all orders of entry. Returns (net_r2, shares)
    where shares sum to net_r2.
    """
    from math import factorial
    p = Xv.shape[1]
    r2 = _subset_r2(Xv, yv)
    shares = np.zeros(p)
    for j in range(p):
        bit = 1 << j
        for mask in range(2 ** p):
            if mask & bit:
                continue
            size = bin(mask).count("1")
            w = factorial(size) * factorial(p - size - 1) / factorial(p)
            shares[j] += w * (r2[mask | bit] - r2[mask])
    return r2[2 ** p - 1], shares


def compute_relative_importance(df, model, diversity_metrics=DIVERSITY_METRICS,
                                stability_col=STABILITY_COL, n_boot=N_BOOT_LMG,
                                seed=RANDOM_SEED, alpha=CI_ALPHA):
    """
    Net diversity effect = centred R^2 of S~ explained by all D~_j together.
    Each metric's share (%) = its LMG contribution / net R^2 * 100.
    Direction is taken from the sign of theta_j in the mutually adjusted model.
    Bootstrap CIs (resampling plots) for both the shares and the net R^2.
    """
    names = list(diversity_metrics.keys())
    Xv = df[list(diversity_metrics.values())].astype(float).values
    yv = df[stability_col].astype(float).values

    net_r2, shares = _lmg_shares(Xv, yv)
    pct = 100 * shares / net_r2 if net_r2 > 0 else np.full(len(names), np.nan)

    boot_pct, boot_net = [], []
    if n_boot:
        rng = np.random.default_rng(seed)
        n = len(yv)
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            b_net, b_sh = _lmg_shares(Xv[idx], yv[idx])
            boot_net.append(b_net)
            boot_pct.append(100 * b_sh / b_net if b_net > 0 else np.full(len(names), np.nan))
        boot_pct = np.array(boot_pct)

    q = [100 * alpha / 2, 100 * (1 - alpha / 2)]
    rows = []
    for j, name in enumerate(names):
        theta = model.params[name]
        rows.append({
            "diversity_metric": name,
            "share_of_net_effect_pct": pct[j],
            "share_ci_lower": np.nanpercentile(boot_pct[:, j], q[0]) if n_boot else np.nan,
            "share_ci_upper": np.nanpercentile(boot_pct[:, j], q[1]) if n_boot else np.nan,
            "lmg_r2": shares[j],
            "direction": "stabilising" if theta > 0 else "destabilising",
            "theta_adjusted": theta,
            "net_diversity_r2": net_r2,
            "net_r2_ci_lower": np.percentile(boot_net, q[0]) if n_boot else np.nan,
            "net_r2_ci_upper": np.percentile(boot_net, q[1]) if n_boot else np.nan,
        })
    return pd.DataFrame(rows)


def plot_relative_importance(ri_df, savepath):
    """Bar chart of each metric's share (%) of the net diversity effect, coloured by direction."""
    sub = ri_df.sort_values("share_of_net_effect_pct").reset_index(drop=True)
    y = np.arange(len(sub))
    heights = sub["share_of_net_effect_pct"].values
    colors = [COLOR_POS if d == "stabilising" else COLOR_NEG for d in sub["direction"]]

    fig, ax = plt.subplots(figsize=(6.5, 0.55 * len(sub) + 1.6))
    kw = dict(height=0.55, color=colors, alpha=0.9, edgecolor="#1a1a1a", linewidth=0.6)
    if sub["share_ci_lower"].notna().all():
        kw.update(xerr=[heights - sub["share_ci_lower"].values,
                        sub["share_ci_upper"].values - heights],
                  capsize=3, error_kw=dict(ecolor="#1a1a1a", elinewidth=0.8, capthick=0.8))
    ax.barh(y, heights, **kw)
    ax.set_yticks(y)
    ax.set_yticklabels(sub["diversity_metric"])
    ax.set_xlim(0, max(100, np.nanmax(sub.get("share_ci_upper", heights))) * 1.05)
    ax.set_xlabel("Share of net diversity effect on stability (%)")
    net, lo, hi = sub["net_diversity_r2"].iloc[0], sub["net_r2_ci_lower"].iloc[0], sub["net_r2_ci_upper"].iloc[0]
    ci_txt = f" [{lo:.2g}, {hi:.2g}]" if np.isfinite(lo) else ""
    ax.set_title(f"Relative importance of diversity metrics\n(net diversity $R^2$ = {net:.2g}{ci_txt})", pad=10)
    ax.grid(axis="y", visible=False)
    for yi, h in zip(y, heights):
        ax.text(h + 1.5, yi, f"{h:.0f}%", va="center", ha="left", fontsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOR_POS, alpha=0.9, label="Stabilising"),
               plt.Rectangle((0, 0), 1, 1, color=COLOR_NEG, alpha=0.9, label="Destabilising")]
    ax.legend(handles=handles, loc="lower right")
    fig.tight_layout()
    _save_publication_figure(fig, savepath)
    return fig


if __name__ == "__main__":
    residuals_df = pd.read_csv(RESIDUALS_PATH)

    residuals_df.drop(columns=["residual_pct_rare_species"], inplace=True, errors="ignore")
    residuals_df=residuals_df.merge(PID_df[['PID', 'STDAGE', 'percent_conifer']], on='PID', how='left')
    residuals_df=residuals_df[residuals_df.percent_conifer < 40]


    ind_results, ind_summary, ind_effects = run_individual_effect_analysis(residuals_df, outdir=OUTDIR)
    adj_model, vif_df, adj_summary, adj_effects = run_mutually_adjusted_analysis(residuals_df, outdir=OUTDIR)

    plt.show()