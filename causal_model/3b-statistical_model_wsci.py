"""
Effects of the diversity metrics on an outcome, switchable with OUTCOME at the top:
  "wsci"      : residualised structural complexity (WSCI), standardised; effects are the
                change in WSCI (in SD) per 1 SD of each diversity metric. Outputs in results/wsci.
  "stability" : the original analysis (residualised log NPP SD x -1; % change in stability).
Same models, gradients, scenarios and figures for either outcome. WSCI should be the
residual from the same nuisance models as the diversity metrics (residualised on the
covariates, NOT on diversity), so the slopes are the a-paths of the mediation analysis.
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
# Outcome
# ===============================================================
# What the diversity metrics predict. Everything else (output folder, signs, units,
# labels) follows from this choice.
#   "stability": residualised log NPP SD x -1 (+ = more stable), effects as % change
#   "wsci"     : residualised structural complexity (+ = more complex than expected),
#                standardised to 1 SD, effects in SD of WSCI per 1 SD of diversity
OUTCOME = "wsci"

OUTCOMES = {
    "stability": dict(col="residual_std_npp", sign=-1, log=True, standardise=False,
                      name="stability", unit="% change in stability",
                      pos="stabilising", neg="destabilising", outdir="results/ols",
                      coords_csv="results/mutually_adjusted_effects_per_observation_with_coords.csv"),
    "wsci": dict(col="residual_wsci", sign=1, log=False, standardise=True,
                 name="structural complexity (WSCI)", unit="change in WSCI (SD)",
                 pos="increases WSCI", neg="decreases WSCI", outdir="results/wsci",
                 coords_csv="results/wsci/mutually_adjusted_effects_per_observation_with_coords.csv"),
}
_OC = OUTCOMES[OUTCOME]

# ===============================================================
# Input and output files  (edit names here only)
# ===============================================================

# Inputs
RAW_DATA_PATH = "data/final/final_dataset_ba.csv"
RESIDUALS_PATH = "results/step_2_residuals_merged.csv"  # 'results/asynch_residuals_cleaned.csv'
RARITY_PATH = "data/final/dataset_rarity.parquet"
PID_LOOKUP_PATH = "data/lookup/PID_location_v3.csv"



# Output directories
OUTDIR = _OC["outdir"]            # individual / mutually adjusted outputs (per outcome)
OUT_GRADIENT_DIR = _OC["outdir"]  # gradient outputs

# Individual-effect model outputs (no figure)
OUT_INDIVIDUAL_EFFECTS = "individual_effects.csv"
OUT_INDIVIDUAL_PER_OBS = "individual_effects_per_observation.csv"
OUT_INDIVIDUAL_SCENARIOS = "individual_scenario_effects.csv"

# Mutually adjusted model outputs
OUT_ADJUSTED_EFFECTS = "mutually_adjusted_effects.csv"
OUT_ADJUSTED_VIF = "mutually_adjusted_vif.csv"
OUT_ADJUSTED_PER_OBS = "mutually_adjusted_effects_per_observation.csv"
OUT_ADJUSTED_PER_OBS_COORDS = _OC["coords_csv"]   # full path; per outcome, so the stability file
                                                  #   used by the maps is never overwritten
OUT_ADJUSTED_SCENARIOS = "mutually_adjusted_scenario_effects.csv"
OUT_ADJUSTED_RANKING_FIG = "ranking_mutually_adjusted.png"

# Relative importance (LMG) outputs
OUT_RELATIVE_IMPORTANCE = "relative_importance.csv"
OUT_RELATIVE_IMPORTANCE_FIG = "relative_importance.png"

# ---------------------------------------------------------------
# Gradient analyses (use ALL plots): the individual and mutually adjusted
# models are refitted in windows along a plot-level gradient, and the slope
# of stability on each diversity metric is plotted against the gradient. Three figures:
# one panel per metric (individual + mutually adjusted lines), and all metrics in one
# panel for the individual models and for the mutually adjusted model.
# ---------------------------------------------------------------
CONIFER_COL = "percent_conifer"          # from PID_LOOKUP_PATH
PCT_RARE_RAW_COL = "pct_rare_species"    # from RARITY_PATH
WSCI_COL = "residual_wsci"               # structural complexity: "residual_wsci" (in the residuals
                                         #   table) or a raw WSCI column from RAW_DATA_PATH

GRADIENTS = {
    "conifer": dict(
        col=CONIFER_COL, label="% conifer",
        range=(0, 100), width=20, step=10, min_n=40,   # windows in the gradient's units
        invert=False, axis_label="% conifer  \u2192  more conifer",
        direction=1, unit=10, unit_label="10 pp more conifer",   # for the interaction test
        out_csv="effects_vs_percent_conifer.csv",
        out_inter="conifer_interaction_tests.csv",
        out_fig_by_metric="effects_vs_percent_conifer.png",            # one panel per metric
        out_fig_individual="effects_vs_percent_conifer_individual.png",
        out_fig_adjusted="effects_vs_percent_conifer_mutually_adjusted.png",
    ),
    "pct_rare": dict(
        col=PCT_RARE_RAW_COL, label="% rare species",
        range=None, width=None, step=0.15, min_n=40,  # None = from the data (1st-99th pct),
                                                      # width 1/5 and step 1/20 of that range
        invert=False, axis_label="% rare species  \u2192  more rare species",
        direction=1, unit=10, unit_label="10 pp more rare species",  # use unit=0.1 if a proportion
        out_csv="effects_vs_pct_rare_species.csv",
        out_inter="pct_rare_species_interaction_tests.csv",
        out_fig_by_metric="effects_vs_pct_rare_species.png",           # one panel per metric
        out_fig_individual="effects_vs_pct_rare_species_individual.png",
        out_fig_adjusted="effects_vs_pct_rare_species_mutually_adjusted.png",
    ),
    "wsci": dict(
        col=WSCI_COL, label="WSCI",
        range=None, width=None, step=None, min_n=40,  # None = from the data (1st-99th pct),
                                                      # width 1/5 and step 1/20 of that range
        invert=False, axis_label="WSCI  \u2192  more structurally complex",
        direction=1, unit="sd", unit_label="1 SD more WSCI",   # unit="sd": per 1 SD of the gradient
        out_csv="effects_vs_wsci.csv",
        out_inter="wsci_interaction_tests.csv",
        out_fig_by_metric="effects_vs_wsci.png",
        out_fig_individual="effects_vs_wsci_individual.png",
        out_fig_adjusted="effects_vs_wsci_mutually_adjusted.png",
    ),
}
RUN_GRADIENTS = ["conifer", "pct_rare" ]   # gradients to run (= rows of the combined figure)
if OUTCOME == "wsci":                              # WSCI cannot moderate its own prediction
    RUN_GRADIENTS = [g for g in RUN_GRADIENTS if g != "wsci"]

# One figure with every gradient in RUN_GRADIENTS stacked (rows) and the metrics side by side,
# plus a right-hand column with the COMBINED effect of diversity: the % change in stability
# when ALL metrics rise by 1 SD together, from the mutually adjusted model in each window.
OUT_GRADIENTS_COMBINED_FIG = "effects_vs_gradients_combined.png"
TOTAL_LABEL = "All metrics +1 SD"
TOTAL_COLOR = "#222222"
SHOW_METRICS_IN_TOTAL_PANEL = True        # thin adjusted metric lines behind the combined line
# Truncate the ends of each row in the combined figure: only windows whose median lies between
# these percentiles of the plots' gradient values are drawn, and the x-axis is fitted to them
# (sparse, wide-CI windows at the extremes are dropped). None = draw every window.
# Override per gradient with plot_trim=(lo, hi) in its GRADIENTS entry.
COMBINED_TRIM_PCT = (5, 95)
COMBINED_X_PAD = 0.1                      # padding either side, as a share of the plotted range


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


def _save_publication_figure(fig, savepath_png):
    if savepath_png is None:
        return
    fig.savefig(savepath_png, dpi=300, bbox_inches="tight")
    print(f"  Saved {savepath_png}")


COLOR_POS = "#7ddfe7"   # light blue: increases stability (drives NPP SD down)
COLOR_NEG = "#4910d8"   # purple: decreases stability (drives NPP SD up)

DIVERSITY_METRICS = {
    "Rao's Q": "residual_raos_q",
    "Functional Evenness": "residual_functional_evenness",
    # "Shannon": "residual_shannon",
    "Species richness": "residual_species_richness",
}

# Per-metric colours (gradient figures)
METRIC_COLORS = {
    "Rao's Q": "#4B2BC4",
    "Functional Evenness": "#A47CC7",
    "Species richness": "#14B8A0",
}

STABILITY_COL = _OC["col"]           # the outcome column (name kept for the rest of the code)
PID_COL = "PID"

CI_ALPHA = 0.05
FDR_ALPHA = 0.05
JOINT_VIF_WARN_THRESHOLD = 10

NET_EFFECT_SCALE = "sum_abs"

N_BOOT_LMG = 200            # bootstrap resamples for LMG share CIs (0 to skip)
RANDOM_SEED = 42

# ---------------------------------------------------------------
# Model settings (keep consistent with the equations)
# ---------------------------------------------------------------

INCLUDE_INTERCEPT = False   # equations have no intercept

REPORT_AS_STABILITY = OUTCOME == "stability"
OUTCOME_SIGN = _OC["sign"]          # -1 turns log NPP SD into stability; +1 keeps WSCI as is
OUTCOME_IS_LOG = _OC["log"]

# Raw (un-residualised) data, used for the percentile scenario contrasts.
raw_df = pd.read_csv(RAW_DATA_PATH)

PID_df = pd.read_csv(PID_LOOKUP_PATH)
raw_df = raw_df.merge(PID_df[["PID", "STDAGE", "percent_conifer"]], on="PID", how="left")

raw_df.drop(columns=["Unnamed: 0", "managed", "ownership", "DIA", "TPA_UNADJ", "Simpson's Index",
                     "Functional_Richness", "Shannon Equitabiltiy Index", "transformed npp"],
            inplace=True, errors="ignore")
raw_df.dropna(subset=["std npp", "mean", "Raos_Q", "Functional_Evenness", "Soil Moisture",
                      "Species Richness", "Shannon Diversity", "pet_std"],
              inplace=True)

# ---------------------------------------------------------------
# Ecological scenarios
# ---------------------------------------------------------------
# Raw (un-residualised) column for each diversity metric in raw_df, used
# to compute percentile contrasts in natural units.
RAW_METRIC_COLS = {
    "Rao's Q": "Raos_Q",
    "Functional Evenness": "Functional_Evenness",
    "Species richness": "Species Richness",
}

# Percentile contrast applied to every metric: low -> high diversity.
SCENARIO_PERCENTILES = (10, 90)
# Fixed changes in natural units: {metric: [(label, delta), ...]}.
FIXED_SCENARIOS = {
    "Species richness": [("Loss of 5 species", -5), ("Gain of 5 species", 5)],
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
    """Change on the outcome scale -> reporting units: % change for the log outcome
    (stability); unchanged (SD units, as the outcome is standardised) for WSCI."""
    if not OUTCOME_IS_LOG:
        return change
    return (np.exp(change) - 1) * 100


def _outcome_name():
    return _OC["name"]


def _unit():
    """Axis wording of the effect units, e.g. '% change in stability'."""
    return _OC["unit"]


def _orient_outcome(df, stability_col=STABILITY_COL):
    """Applies OUTCOME_SIGN so that positive effects mean more of the outcome (stability / WSCI)."""
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
        base = "Effect as % of net diversity effect"
    return base + suffix


def _plot_ranking_bar(sub, xlabel, title, savepath, col="theta_pct", decimals=2):
    """Horizontal ranking bar chart (used for the mutually adjusted effects)."""
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
            "effect_units": f"{_unit()} per 1 SD of the metric",
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


# ---------------------------------------------------------------
# Effects along a gradient (% conifer, % rare species, ...)
# ---------------------------------------------------------------

def _gradient_windows(values, cfg):
    """Sliding windows (lo, hi) over the gradient, in its own units."""
    if cfg.get("range"):
        lo, hi = cfg["range"]
    else:
        lo, hi = np.nanpercentile(values, [1, 99])
    span = hi - lo
    width = cfg.get("width") or span / 5
    step = cfg.get("step") or span / 20
    starts = list(np.arange(lo, hi - width + 1e-9, step))
    if not starts or starts[-1] + width < hi - 1e-9:
        starts.append(hi - width)            # make sure the last window reaches the top
    return [(a, a + width) for a in starts]


def _gradient_row(model, term, metric, model_name, ref_sd, lo, hi, n, g_median, alpha):
    ci = model.conf_int(alpha=alpha).loc[term]
    return {
        "diversity_metric": metric, "model": model_name,
        "window_lower": lo, "window_upper": hi, "gradient_median": g_median, "n": n,
        "theta": model.params[term],
        "pct": _to_pct(model.params[term] * ref_sd),
        "pct_ci_lower": _to_pct(ci.iloc[0] * ref_sd),
        "pct_ci_upper": _to_pct(ci.iloc[1] * ref_sd),
        "p_value": model.pvalues[term],
    }


def compute_gradient(df, cfg, diversity_metrics=DIVERSITY_METRICS, stability_col=STABILITY_COL,
                     alpha=CI_ALPHA):
    """
    Refits the individual and mutually adjusted models in sliding windows of the
    gradient. Effects are scaled by the FULL-sample SD of each D~_j, so changes
    across windows reflect changes in theta_j, not changes in the spread of D~_j.
    """
    gcol = cfg["col"]
    cols = [stability_col, gcol] + list(diversity_metrics.values())
    df = _orient_outcome(df.dropna(subset=cols).reset_index(drop=True), stability_col)
    ref_sd = {m: df[c].astype(float).std(ddof=1) for m, c in diversity_metrics.items()}
    g = df[gcol].astype(float)

    rows = []
    for lo, hi in _gradient_windows(g.values, cfg):
        sub = df[(g >= lo) & (g <= hi)]
        n = len(sub)
        if n < cfg["min_n"]:
            print(f"  [{cfg['label']} gradient] skipping {lo:.3g}-{hi:.3g} (n = {n} < {cfg['min_n']})")
            continue
        g_med = sub[gcol].median()

        for m, col in diversity_metrics.items():
            model, _, _ = fit_individual(sub[stability_col], sub[col])
            rows.append(_gradient_row(model, "D", m, "individual", ref_sd[m], lo, hi, n, g_med, alpha))

        model, X, _ = fit_mutually_adjusted(sub, diversity_metrics, stability_col)
        for m in X.columns:
            rows.append(_gradient_row(model, m, m, "mutually_adjusted", ref_sd[m], lo, hi, n, g_med, alpha))
        rows.append(_total_row(model, list(X.columns), ref_sd, lo, hi, n, g_med, alpha))
    return pd.DataFrame(rows)


def _total_row(model, terms, ref_sd, lo, hi, n, g_median, alpha):
    """
    Combined effect of diversity in one window: every metric +1 full-sample SD together,
    from the mutually adjusted model, sum_j theta_j * SD_j, with a CI from the robust
    covariance of the thetas (so it accounts for their correlation).
    """
    s = np.array([ref_sd[t] for t in terms])
    b = model.params[terms].to_numpy(float)
    cov = model.cov_params().loc[terms, terms].to_numpy(float)
    est = float(s @ b)
    se = float(np.sqrt(max(s @ cov @ s, 0)))
    crit = _crit(model, alpha)
    z = est / se if se > 0 else np.nan
    return {
        "diversity_metric": TOTAL_LABEL, "model": "total",
        "window_lower": lo, "window_upper": hi, "gradient_median": g_median, "n": n,
        "theta": est,
        "pct": _to_pct(est), "pct_ci_lower": _to_pct(est - crit * se),
        "pct_ci_upper": _to_pct(est + crit * se),
        "p_value": float(2 * stats.norm.sf(abs(z))) if np.isfinite(z) else np.nan,
    }


def gradient_interaction_tests(df, cfg, diversity_metrics=DIVERSITY_METRICS,
                               stability_col=STABILITY_COL, alpha=CI_ALPHA):
    """
    Formal test of whether each effect changes along the gradient (CSV only, not plotted):
        S~ = theta D~ + gamma D~*z + delta z,   z = direction * gradient / unit, centred
    gamma = change in theta per `unit_label` (e.g. per 10 pp less conifer).
    """
    gcol = cfg["col"]
    cols = [stability_col, gcol] + list(diversity_metrics.values())
    df = _orient_outcome(df.dropna(subset=cols).reset_index(drop=True), stability_col)
    g_raw = df[gcol].astype(float).values
    unit = g_raw.std(ddof=1) if cfg["unit"] == "sd" else cfg["unit"]
    z = cfg["direction"] * g_raw / unit
    z = z - z.mean()
    y = df[stability_col].astype(float).values

    rows = []
    for m, col in diversity_metrics.items():
        d = df[col].astype(float).values
        X = pd.DataFrame({"D": d, "D_x_gradient": d * z, "gradient": z})
        model = _fit(y, _design(X))
        ci = model.conf_int(alpha=alpha).loc["D_x_gradient"]
        sd = d.std(ddof=1)
        rows.append({
            "gradient": cfg["label"],
            "gamma_per": cfg["unit_label"],
            "diversity_metric": m,
            "theta_at_mean_gradient": model.params["D"],
            "gamma": model.params["D_x_gradient"],
            "gamma_ci_lower": ci.iloc[0], "gamma_ci_upper": ci.iloc[1],
            # change in the per-1-SD effect (log scale; x100 ~ percentage points)
            "effect_change_per_SD_per_unit": model.params["D_x_gradient"] * sd,
            "p_value_raw": model.pvalues["D_x_gradient"],
            "n": len(y),
        })
    return _add_fdr(pd.DataFrame(rows))


def plot_gradient(grad_df, cfg, model_name, savepath=None):
    """
    One panel: the slope of stability on each diversity metric (y, as % change in
    stability per 1 full-sample SD of D~_j) against the gradient (x), with its CI.
    model_name = "individual" or "mutually_adjusted".
    """
    sub = grad_df[grad_df["model"] == model_name]
    metrics = list(dict.fromkeys(sub["diversity_metric"]))
    ci_pct = round((1 - CI_ALPHA) * 100)

    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    for m in metrics:
        d = sub[sub["diversity_metric"] == m].sort_values("gradient_median")
        color = METRIC_COLORS.get(m, COLOR_NEG)
        ax.fill_between(d["gradient_median"], d["pct_ci_lower"], d["pct_ci_upper"],
                        color=color, alpha=0.15, linewidth=0)
        ax.plot(d["gradient_median"], d["pct"], color=color, lw=1.6, marker="o", ms=3.5, label=m)
    ax.axhline(0, color="#1a1a1a", lw=0.8, zorder=0)

    if cfg.get("range"):
        ax.set_xlim(*cfg["range"])
    if cfg.get("invert"):
        ax.invert_xaxis()
    ax.set_xlabel(cfg["axis_label"])
    ax.set_ylabel(f"Effect of diversity on {_outcome_name()}\n"
                  f"({_unit()} per 1 SD of diversity)", fontsize=9)
    ax.set_title("Individual effects" if model_name == "individual" else "Mutually adjusted effects",
                 fontsize=10)
    ax.legend(loc="best", fontsize=7.5, title=f"Shading: {ci_pct}% CI", title_fontsize=7)
    fig.tight_layout()
    _save_publication_figure(fig, savepath)
    return fig


def plot_gradient_by_metric(grad_df, cfg, savepath=None):
    """
    One panel per diversity metric: slope of stability on that metric (y) against the
    gradient (x). Mutually adjusted = metric colour with CI band; individual = grey
    dashed line with a light grey CI band.
    """
    grad_df = grad_df[grad_df["model"] != "total"]
    metrics = list(dict.fromkeys(grad_df["diversity_metric"]))
    n = len(metrics)
    ci_pct = round((1 - CI_ALPHA) * 100)
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 3.0), sharex=True, sharey=True,
                             squeeze=False)
    axes = axes[0]

    for ax, m in zip(axes, metrics):
        color = METRIC_COLORS.get(m, COLOR_NEG)
        sel = grad_df["diversity_metric"] == m
        ind = grad_df[sel & (grad_df["model"] == "individual")].sort_values("gradient_median")
        adj = grad_df[sel & (grad_df["model"] == "mutually_adjusted")].sort_values("gradient_median")

        if len(ind):
            ax.fill_between(ind["gradient_median"], ind["pct_ci_lower"], ind["pct_ci_upper"],
                            color="#7f7f7f", alpha=0.12, linewidth=0)
            ax.plot(ind["gradient_median"], ind["pct"], color="#7f7f7f", ls="--", lw=1.1,
                    marker="o", ms=2.5, label="Individual")
        if len(adj):
            ax.fill_between(adj["gradient_median"], adj["pct_ci_lower"], adj["pct_ci_upper"],
                            color=color, alpha=0.2, linewidth=0)
            ax.plot(adj["gradient_median"], adj["pct"], color=color, lw=1.6, marker="o", ms=3,
                    label="Mutually adjusted")
        ax.axhline(0, color="#1a1a1a", lw=0.8, zorder=0)
        ax.set_title(m, fontsize=9, color=color)

    if cfg.get("range"):
        axes[0].set_xlim(*cfg["range"])
    if cfg.get("invert"):
        axes[0].invert_xaxis()   # shared x, so every panel is inverted
    axes[0].set_ylabel(f"Effect of diversity on {_outcome_name()}\n"
                       f"({_unit()} per 1 SD of diversity)", fontsize=9)
    axes[0].legend(loc="best", fontsize=7, title=f"Shading: {ci_pct}% CI", title_fontsize=6.5)
    fig.supxlabel(cfg["axis_label"], fontsize=9)
    fig.tight_layout()
    _save_publication_figure(fig, savepath)
    return fig


def _draw_metric_lines(ax, gm, color):
    """Individual (grey dashed) and mutually adjusted (metric colour) slope along the gradient."""
    ind = gm[gm["model"] == "individual"].sort_values("gradient_median")
    adj = gm[gm["model"] == "mutually_adjusted"].sort_values("gradient_median")
    if len(ind):
        ax.fill_between(ind["gradient_median"], ind["pct_ci_lower"], ind["pct_ci_upper"],
                        color="#7f7f7f", alpha=0.12, linewidth=0)
        ax.plot(ind["gradient_median"], ind["pct"], color="#7f7f7f", ls="--", lw=1.1,
                marker="o", ms=2.5)
    if len(adj):
        ax.fill_between(adj["gradient_median"], adj["pct_ci_lower"], adj["pct_ci_upper"],
                        color=color, alpha=0.2, linewidth=0)
        ax.plot(adj["gradient_median"], adj["pct"], color=color, lw=1.6, marker="o", ms=3)


def _trim_windows(gd, cfg, gradient_values=None):
    """Keeps windows whose median lies within the trim percentiles of the gradient's values."""
    trim = cfg.get("plot_trim", COMBINED_TRIM_PCT)
    if not trim:
        return gd
    vals = (np.asarray(gradient_values, dtype=float) if gradient_values is not None
            else gd["gradient_median"].to_numpy(float))      # fallback: the window medians
    vals = vals[np.isfinite(vals)]
    lo, hi = np.percentile(vals, trim)
    keep = gd["gradient_median"].between(lo, hi)
    dropped = gd.loc[~keep, ["window_lower", "window_upper"]].drop_duplicates()
    if len(dropped):
        print(f"  [combined figure] {cfg['label']}: dropped {len(dropped)} end window(s) outside "
              f"P{trim[0]:g}-P{trim[1]:g} ({lo:.3g}-{hi:.3g})")
    return gd[keep]


def plot_gradients_combined(grad_results, keys, savepath=None, gradient_values=None):
    """
    Rows = gradients (keys, e.g. % conifer and % rare species), columns = diversity metrics
    (individual = grey dashed, mutually adjusted = metric colour, each with its CI band),
    plus a right-hand column with the COMBINED effect of all metrics rising by 1 SD
    together (mutually adjusted, black, with CI). Metric panels share a y-axis within each
    row; the combined panel has its own y-axis, because it is a sum of the metric effects.
    gradient_values = {key: raw gradient values of the plots}, used to truncate the ends
    (COMBINED_TRIM_PCT); without it the window medians are used instead.
    """
    ci_pct = round((1 - CI_ALPHA) * 100)
    metrics = list(DIVERSITY_METRICS)
    n_r, n_m = len(keys), len(metrics)
    fig = plt.figure(figsize=(2.6 * n_m + 3.0, 2.9 * n_r + 0.9))
    gs = fig.add_gridspec(n_r, n_m + 1, width_ratios=[1] * n_m + [1.15], wspace=0.2, hspace=0.45)

    for r, key in enumerate(keys):
        cfg = GRADIENTS[key]
        gd = _trim_windows(grad_results[key], cfg, (gradient_values or {}).get(key))
        row_axes = []
        for j, m in enumerate(metrics):
            ax = fig.add_subplot(gs[r, j], sharey=row_axes[0] if row_axes else None,
                                 sharex=row_axes[0] if row_axes else None)
            row_axes.append(ax)
            color = METRIC_COLORS.get(m, COLOR_NEG)
            _draw_metric_lines(ax, gd[gd["diversity_metric"] == m], color)
            ax.axhline(0, color="#1a1a1a", lw=0.8, zorder=0)
            if r == 0:
                ax.set_title(m, fontsize=9.5, color=color)
            if j == 0:
                ax.set_ylabel(f"Along {cfg['label']}\n{_unit()}\n"
                              f"per 1 SD of diversity", fontsize=8.5)
            else:
                ax.tick_params(labelleft=False)
            ax.set_xlabel(cfg["axis_label"], fontsize=7.5)

        # Combined effect of all metrics (+1 SD each), with the adjusted metric lines behind it
        axt = fig.add_subplot(gs[r, n_m], sharex=row_axes[0])
        tot = gd[gd["model"] == "total"].sort_values("gradient_median")
        if SHOW_METRICS_IN_TOTAL_PANEL:
            for m in metrics:
                adj = gd[(gd["model"] == "mutually_adjusted") & (gd["diversity_metric"] == m)]
                adj = adj.sort_values("gradient_median")
                axt.plot(adj["gradient_median"], adj["pct"], color=METRIC_COLORS.get(m, COLOR_NEG),
                         lw=0.9, alpha=0.6)
        axt.fill_between(tot["gradient_median"], tot["pct_ci_lower"], tot["pct_ci_upper"],
                         color=TOTAL_COLOR, alpha=0.15, linewidth=0)
        axt.plot(tot["gradient_median"], tot["pct"], color=TOTAL_COLOR, lw=2.2, marker="o", ms=3.5)
        axt.axhline(0, color="#1a1a1a", lw=0.8, zorder=0)
        if r == 0:
            axt.set_title(f"Combined effect\n({TOTAL_LABEL}, adjusted)", fontsize=9.5,
                          color=TOTAL_COLOR, fontweight="bold")
        axt.set_xlabel(cfg["axis_label"], fontsize=7.5)
        axt.yaxis.set_label_position("right")
        axt.yaxis.tick_right()
        axt.spines["right"].set_visible(True)
        axt.spines["left"].set_visible(False)
        axt.set_ylabel(_unit(), fontsize=8.5)

        x_lo, x_hi = gd["gradient_median"].min(), gd["gradient_median"].max()
        pad = COMBINED_X_PAD * (x_hi - x_lo) if x_hi > x_lo else 1.0
        row_axes[0].set_xlim(x_lo - pad, x_hi + pad)   # fitted to the windows actually drawn
        if cfg.get("invert"):
            row_axes[0].invert_xaxis()     # shared x, so the whole row is inverted
        for ax in row_axes + [axt]:
            ax.tick_params(labelsize=7.5)

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Line2D([], [], color="#555555", lw=1.6, marker="o", ms=3,
                      label="Mutually adjusted (metric colour)"),
               Line2D([], [], color="#7f7f7f", lw=1.1, ls="--", marker="o", ms=2.5,
                      label="Individual"),
               Line2D([], [], color=TOTAL_COLOR, lw=2.2, marker="o", ms=3.5,
                      label=f"Combined: {TOTAL_LABEL} (adjusted)"),
               Patch(color="#555555", alpha=0.2, label=f"{ci_pct}% CI")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=4,
               fontsize=8, frameon=False)
    _save_publication_figure(fig, savepath)
    if savepath:
        fig.savefig(os.path.splitext(savepath)[0] + ".pdf", bbox_inches="tight")
    return fig


def run_gradient_analysis(df, key, outdir=OUT_GRADIENT_DIR):
    cfg = GRADIENTS[key]
    os.makedirs(outdir, exist_ok=True)
    if cfg["col"] not in df.columns:
        raise KeyError(f"Gradient '{key}': column '{cfg['col']}' not found in the residuals table.")

    grad_df = compute_gradient(df, cfg)
    grad_df.to_csv(_out(outdir, cfg["out_csv"]), index=False)

    inter_df = gradient_interaction_tests(df, cfg)
    inter_df.to_csv(_out(outdir, cfg["out_inter"]), index=False)
    print(f"\n[run_gradient_analysis] Interaction with {cfg['label']} (gamma per {cfg['unit_label']}):")
    print(inter_df.to_string(index=False))

    plot_gradient_by_metric(grad_df, cfg, savepath=_out(outdir, cfg["out_fig_by_metric"]))
    plot_gradient(grad_df, cfg, "individual", savepath=_out(outdir, cfg["out_fig_individual"]))
    plot_gradient(grad_df, cfg, "mutually_adjusted", savepath=_out(outdir, cfg["out_fig_adjusted"]))
    return grad_df, inter_df


# ---------------------------------------------------------------
# Scenario effects
# ---------------------------------------------------------------

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
                "pct_change": _to_pct(r["theta"] * delta),      # in _unit() (see effect_units)
                "effect_units": _unit(),
                "pct_change_ci_lower": ends[0],
                "pct_change_ci_upper": ends[1],
                "significant_fdr": r.get("significant_fdr", np.nan),
            })
    return pd.DataFrame(rows)


def run_individual_effect_analysis(df, diversity_metrics=DIVERSITY_METRICS,
                                   stability_col=STABILITY_COL, pid_col=PID_COL, outdir=OUTDIR):
    """Individual-effect models (Eq. individual_effect) for every diversity metric. CSVs only."""
    os.makedirs(outdir, exist_ok=True)
    df = df.dropna(subset=[stability_col] + list(diversity_metrics.values())).reset_index(drop=True)
    df = _orient_outcome(df, stability_col)
    print(f"[run_individual_effect_analysis] n = {len(df)}")

    all_results = fit_individual_models(df, diversity_metrics, stability_col)

    summary_df = build_individual_summary(all_results)
    print(summary_df.to_string(index=False))

    summary_df = add_net_scale(summary_df, _net_for(summary_df, df))
    summary_df.to_csv(_out(outdir, OUT_INDIVIDUAL_EFFECTS), index=False)

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
                         "diversity_residual": X[term].values[i], "stability_residual": s_vals[i],
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
            "direction": _OC["pos"] if theta > 0 else _OC["neg"],
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
    colors = [COLOR_POS if d == _OC["pos"] else COLOR_NEG for d in sub["direction"]]

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
    ax.set_xlabel(f"Share of net diversity effect on {_outcome_name()} (%)")
    net, lo, hi = sub["net_diversity_r2"].iloc[0], sub["net_r2_ci_lower"].iloc[0], sub["net_r2_ci_upper"].iloc[0]
    ci_txt = f" [{lo:.2g}, {hi:.2g}]" if np.isfinite(lo) else ""
    ax.set_title(f"Relative importance of diversity metrics\n(net diversity $R^2$ = {net:.2g}{ci_txt})", pad=10)
    ax.grid(axis="y", visible=False)
    for yi, h in zip(y, heights):
        ax.text(h + 1.5, yi, f"{h:.0f}%", va="center", ha="left", fontsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOR_POS, alpha=0.9, label=_OC["pos"].capitalize()),
               plt.Rectangle((0, 0), 1, 1, color=COLOR_NEG, alpha=0.9, label=_OC["neg"].capitalize())]
    ax.legend(handles=handles, loc="lower right")
    fig.tight_layout()
    _save_publication_figure(fig, savepath)
    return fig


if __name__ == "__main__":

    residuals_all = pd.read_csv(RESIDUALS_PATH)
    residuals_all.drop(columns=["residual_pct_rare_species"], inplace=True, errors="ignore")
    residuals_all = residuals_all.merge(PID_df[["PID", "STDAGE", "percent_conifer"]], on="PID", how="left")

    # Raw % rare species per plot (the gradient uses the un-residualised value)
    if "pct_rare" in RUN_GRADIENTS:
        abundance_df = pd.read_parquet(RARITY_PATH)
        residuals_all = residuals_all.merge(
            abundance_df[["PID", PCT_RARE_RAW_COL]].drop_duplicates("PID"), on="PID", how="left")

    # WSCI gradient: use the residuals table's column, or merge a raw WSCI column if named
    if "wsci" in RUN_GRADIENTS and WSCI_COL not in residuals_all.columns:
        raw_all = pd.read_csv(RAW_DATA_PATH)
        if WSCI_COL not in raw_all.columns:
            raise KeyError(f"WSCI_COL '{WSCI_COL}' is in neither {RESIDUALS_PATH} nor {RAW_DATA_PATH}.")
        residuals_all = residuals_all.merge(raw_all[["PID", WSCI_COL]].drop_duplicates("PID"),
                                            on="PID", how="left")

    # Outcome scale: standardise WSCI so every effect is in SD of WSCI per 1 SD of diversity
    if _OC["standardise"]:
        sd_out = residuals_all[STABILITY_COL].std(ddof=1)
        residuals_all[STABILITY_COL] = residuals_all[STABILITY_COL] / sd_out
        print(f"[outcome] {STABILITY_COL} standardised (SD = {sd_out:.4g}); effects in SD of WSCI")
    print(f"[outcome] predicting {_outcome_name()} ({STABILITY_COL}); outputs in {OUTDIR}")

    # Slope of each diversity metric along each gradient (individual + mutually adjusted figures)
    gradient_results = {key: run_gradient_analysis(residuals_all, key) for key in RUN_GRADIENTS}
    plot_gradients_combined({k: v[0] for k, v in gradient_results.items()}, RUN_GRADIENTS,
                            savepath=_out(OUT_GRADIENT_DIR, OUT_GRADIENTS_COMBINED_FIG),
                            gradient_values={k: residuals_all[GRADIENTS[k]["col"]].dropna().to_numpy()
                                             for k in RUN_GRADIENTS})

    residuals_df = residuals_all
    ind_results, ind_summary, ind_effects = run_individual_effect_analysis(residuals_df, outdir=OUTDIR)
    adj_model, vif_df, adj_summary, adj_effects = run_mutually_adjusted_analysis(residuals_df, outdir=OUTDIR)

    adj_effects.merge(PID_df[["PID", "lon", "lat"]], on="PID", how="left"
    ).to_csv(OUT_ADJUSTED_PER_OBS_COORDS, index=False)

    plt.show()