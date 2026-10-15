"""
Continuous moderation of the diversity -> stability effect, estimated in
the DML final stage as

    Y~_i = (theta_0 + theta_1 * V_i) * D~_i + eps_i
         =  theta_0 * D~_i + theta_1 * (V_i * D~_i) + eps_i

where
    Y~  = residualised stability       (STABILITY_COL)
    D~  = residualised diversity metric (one of DIVERSITY_METRICS)
    (With REPORT_AS_STABILITY = True, Y~ is multiplied by -1 so that
     positive effects mean diversity drives NPP SD down, i.e. increases
     stability = inverse SD.)
    V   = moderator (WSCI or rarity), MEAN-CENTRED (and optionally scaled
          to unit SD), so that
            theta_0 = effect of diversity at the AVERAGE level of V
            theta_1 = change in that effect per 1 unit (or 1 SD) of V

SCALE  (NET_EFFECT_SCALE)
  "own_total" (default): every effect of metric j is expressed as a % of that
  SAME metric's unmoderated, UNADJUSTED effect, |theta_total_j|, from
      Y~ = theta_total_j * D~_j + eps
  fitted on the same rows / standardisation / sign. So 100% = the metric's
  original effect without moderation; theta_1 = -50% means each 1 SD of the
  moderator shifts the effect by half of its original size. The sign is kept
  (positive = stabilising). Unstable when theta_total_j is close to 0: the
  script warns and you should report raw units for that metric.

  "total_sum_abs": % of the TOTAL diversity effect = sum over metrics j of
  |theta_total_j| (the "sum of absolute effects" scale used for the individual /
  mutually adjusted ranking charts). The denominator is treated as fixed, so
  CIs reflect uncertainty in each metric's own effect only.

  "none": raw theta units.

INPUTS / OUTPUTS
  The input path, the output directory (OUTDIR) and every output file name
  are defined in the "Input and output files" block below the imports.
  Names containing "{moderator}" are filled in per moderator (lower case).

TIER BOXPLOTS (bootstrap, mutually adjusted for the other metrics)
  Tiers are tertiles of each moderator (rank-based, so ties cannot merge bins):
  Low / Medium / High WSCI and rarity. WITHIN each tier, the slope of Y~ on the
  focal metric is re-estimated on N_BOOT bootstrap resamples, with the OTHER
  metrics entered as separate covariates, so the focal slope is the effect with
  the others held constant. All metrics are fitted on ONE common set of rows.
  Box = interquartile range, line = median, whiskers = 2.5th to 97.5th
  percentiles. Asterisk = two-sided bootstrap p < BOOT_ALPHA (not FDR-corrected).

AXES: the conditional-effect plots share x / y limits across all moderators.
STANDARD ERRORS: HC3.
MULTIPLE TESTING: Benjamini-Hochberg FDR across all theta_1 tests.

ASSUMPTIONS TO CHECK
  * Linearity of theta(V) in V (see tertile-slope check on the conditional plots).
  * V should be among the nuisance-model covariates; otherwise set
    INCLUDE_MODERATOR_MAIN_EFFECT = True.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.patches import Patch
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from scipy import stats

# ===============================================================
# Input and output files  (edit names here only)
# ===============================================================

# Input
RESIDUALS_PATH = "results/step_2_residuals_merged.csv"

# Output directory (all outputs below are written here)
OUTDIR = "results/deciduous"

# Output file names. "{moderator}" is replaced by the moderator name in lower case.
OUT_SUMMARY = "moderation_summary.csv"
OUT_CONDITIONAL_FIG = "moderation_conditional_{moderator}.png"
OUT_THETA1_FOREST_FIG = "moderation_theta1_forest_marginal.png"
OUT_TIER_BOOT_CSV = "moderation_tier_bootstrap_{moderator}.csv"
OUT_TIER_BOX_FIG = "moderation_tier_boxplots_{moderator}.png"


def _out(filename, outdir=None, moderator=None):
    """Full output path: OUTDIR (or outdir) + filename, filling in {moderator} if given."""
    if moderator is not None:
        filename = filename.format(moderator=moderator.lower())
    return os.path.join(outdir or OUTDIR, filename)


# ---------------------------------------------------------------
# Column names
# ---------------------------------------------------------------

DIVERSITY_METRICS = {
    "Rao's Q": "residual_raos_q",
    "Functional Evenness": "residual_functional_evenness",
    "Shannon": "residual_shannon",
    "Species richness": "residual_species_richness",
}
STABILITY_COL = "residual_std_npp"   # Y~

# Residualised moderators: "high WSCI" = more structurally complex than the
# environment predicts. Point these at raw columns for moderation by actual values.
MODERATORS = {
    "WSCI": "residual_wsci",
    "Rarity": "residual_pct_rare_species",
}

# ---------------------------------------------------------------
# Settings
# ---------------------------------------------------------------

STANDARDISE_MODERATOR = True
STANDARDISE_TREATMENT = True
INCLUDE_INTERCEPT = False
INCLUDE_MODERATOR_MAIN_EFFECT = False

REPORT_AS_STABILITY = True
OUTCOME_SIGN = -1 if REPORT_AS_STABILITY else 1

# "own_total": % of each metric's own unmoderated effect (100% = original effect).
# "total_sum_abs": % of the total diversity effect (sum of |unmoderated effects|).
# "none": raw theta units.
NET_EFFECT_SCALE = "own_total"
# Warn when a metric's own unmoderated effect is this small relative to the
# largest one (its rescaled values will be very large and unstable).
OWN_TOTAL_WARN_FRAC = 0.1

CI_ALPHA = 0.05
FDR_ALPHA = 0.05
N_GRID = 200
N_BINS_CHECK = 3
RUG_MAX_POINTS = 3000
RANDOM_SEED = 42
GRID_N_COLS = 2

AXIS_PAD_FRAC = 0.05

MODERATOR_COLORS = {"WSCI": "#9975BA", "Rarity": "#7DDFE7"}

# ---- Bootstrap tier boxplot settings ----
N_BOOT = 1000
BOOT_ALPHA = 0.05
BOX_WHIS = (2.5, 97.5)
TIER_LABELS = ["Low", "Medium", "High"]
METRIC_COLORS = {
    "Rao's Q": "#4B2BC4",
    "Functional Evenness": "#A47CC7",
    "Shannon": "#5ED6DA",
    "Species richness": "#14B8A0",
}


# ---------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------

def _prepare(df, div_col, mod_cols, stability_col=STABILITY_COL):
    """Y (signed outcome), D (optionally standardised), V_<name> (centred / scaled)."""
    cols = [stability_col, div_col] + list(mod_cols.values())
    sub = df[cols].dropna().reset_index(drop=True)

    out = pd.DataFrame({"Y": OUTCOME_SIGN * sub[stability_col].astype(float)})
    d = sub[div_col].astype(float)
    d_sd = d.std(ddof=1) if STANDARDISE_TREATMENT else 1.0
    out["D"] = d / d_sd

    mod_info = {}
    for name, col in mod_cols.items():
        v = sub[col].astype(float)
        v_mean = v.mean()
        v_sd = v.std(ddof=1) if STANDARDISE_MODERATOR else 1.0
        out[f"V_{name}"] = (v - v_mean) / v_sd
        mod_info[name] = {"mean": v_mean, "sd": v_sd, "raw": v.values}

    return out, d_sd, mod_info


def _prepare_joint(df, diversity_metrics=DIVERSITY_METRICS, moderators=MODERATORS,
                   stability_col=STABILITY_COL):
    """
    ALL metrics and moderators on ONE common set of rows -- needed so every
    metric can be adjusted for the others within a tier.
    Returns {"Y", "D" (DataFrame, one standardised column per metric),
    "V" (DataFrame of moderators), "theta_total", "n"}.
    """
    cols = [stability_col] + list(diversity_metrics.values()) + list(moderators.values())
    sub = df[cols].dropna().reset_index(drop=True)

    Y = OUTCOME_SIGN * sub[stability_col].astype(float).values
    D = pd.DataFrame(index=sub.index)
    for name, col in diversity_metrics.items():
        d = sub[col].astype(float)
        D[name] = d / (d.std(ddof=1) if STANDARDISE_TREATMENT else 1.0)
    V = pd.DataFrame({name: sub[col].astype(float) for name, col in moderators.items()})
    theta_total = {m: _ols_slope(D[m].values, Y) for m in D.columns}
    print(f"[_prepare_joint] n = {len(sub)} plots with every metric and moderator present")
    return {"Y": Y, "D": D, "V": V, "theta_total": theta_total, "n": len(sub)}


def _design(prep, mod_names):
    X = pd.DataFrame({"D": prep["D"]})
    for name in mod_names:
        X[f"DxV_{name}"] = prep["D"] * prep[f"V_{name}"]
        if INCLUDE_MODERATOR_MAIN_EFFECT:
            X[f"V_{name}"] = prep[f"V_{name}"]
    if INCLUDE_INTERCEPT:
        X = sm.add_constant(X, has_constant="add")
    return X


def _fit(y, X):
    """OLS with HC3 robust standard errors."""
    return sm.OLS(np.asarray(y, dtype=float), X).fit(cov_type="HC3")


def _ols_slope(D, Y):
    """Closed-form OLS slope of Y on D (with or without intercept)."""
    if INCLUDE_INTERCEPT:
        Dc = D - D.mean()
        return float((Dc * (Y - Y.mean())).sum() / (Dc * Dc).sum())
    return float((D * Y).sum() / (D * D).sum())


# ---------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------

def fit_marginal_models(df, diversity_metrics=DIVERSITY_METRICS, moderators=MODERATORS):
    """One model per (metric, moderator): Y~ = (theta_0 + theta_1 V) D~ + eps."""
    results = {}
    for metric_name, div_col in diversity_metrics.items():
        results[metric_name] = {}
        for mod_name, mod_col in moderators.items():
            prep, d_sd, mod_info = _prepare(df, div_col, {mod_name: mod_col})
            res = _fit(prep["Y"], _design(prep, [mod_name]))
            results[metric_name][mod_name] = {
                "res": res, "prep": prep, "d_sd": d_sd,
                "mod_info": mod_info[mod_name], "n": len(prep),
                # unmoderated total effect on the same rows / scale / sign
                "theta_total": _ols_slope(prep["D"].values, prep["Y"].values),
            }
    return results


# ---------------------------------------------------------------
# Scale: % of the total diversity effect
# ---------------------------------------------------------------

def total_diversity_effect(marginal_results, mod_name):
    """Total diversity effect for one moderator's rows: sum_j |theta_total_j|."""
    return sum(abs(mods[mod_name]["theta_total"]) for mods in marginal_results.values())


_WARNED = set()


def scale_factor(marginal_results, mod_name, metric_name):
    """Multiplier from theta units to the reporting scale for one metric (1 if unscaled)."""
    if NET_EFFECT_SCALE == "total_sum_abs":
        denom = total_diversity_effect(marginal_results, mod_name)
    elif NET_EFFECT_SCALE == "own_total":
        denom = abs(marginal_results[metric_name][mod_name]["theta_total"])
        largest = max(abs(m[mod_name]["theta_total"]) for m in marginal_results.values())
        key = (metric_name, mod_name)
        if denom < OWN_TOTAL_WARN_FRAC * largest and key not in _WARNED:
            _WARNED.add(key)
            print(f"  WARNING: {metric_name}'s unmoderated effect ({mod_name} rows) is close to 0; "
                  f"its values relative to that effect will be very large and unstable.")
    else:
        return 1.0
    if denom <= 0:
        return 1.0
    return 100.0 / denom


def joint_scale_factors(joint):
    """
    Scale factors {metric: multiplier} for the TIER boxplots, computed from the
    common-row (joint) data. 100% = the metric's own unmoderated, UNADJUSTED
    effect ("own_total"), or the sum of |those effects| ("total_sum_abs").
    """
    th = joint["theta_total"]
    if NET_EFFECT_SCALE == "total_sum_abs":
        denom = sum(abs(v) for v in th.values())
        return {m: (100.0 / denom if denom > 0 else 1.0) for m in th}
    if NET_EFFECT_SCALE == "own_total":
        largest = max(abs(v) for v in th.values())
        k = {}
        for m, v in th.items():
            if abs(v) < OWN_TOTAL_WARN_FRAC * largest:
                print(f"  WARNING: {m}'s unmoderated effect is close to 0; its tier slopes "
                      f"relative to that effect will be very large and unstable.")
            k[m] = 100.0 / abs(v) if abs(v) > 0 else 1.0
        return k
    return {m: 1.0 for m in th}


def _scaled():
    return NET_EFFECT_SCALE in ("total_sum_abs", "own_total")


def _scale_name():
    return ("% of original (unmoderated) effect" if NET_EFFECT_SCALE == "own_total"
            else "% of total diversity effect")


# ---------------------------------------------------------------
# Conditional effects and linearity check
# ---------------------------------------------------------------

def conditional_effect(res, mod_name, v_grid, alpha=CI_ALPHA):
    """theta(v) = theta_0 + theta_1 v with a pointwise CI from the coefficient covariance."""
    t1_name = f"DxV_{mod_name}"
    b0, b1 = res.params["D"], res.params[t1_name]
    cov = res.cov_params()
    v00, v11, v01 = cov.loc["D", "D"], cov.loc[t1_name, t1_name], cov.loc["D", t1_name]
    v_grid = np.asarray(v_grid, dtype=float)
    effect = b0 + b1 * v_grid
    se = np.sqrt(np.maximum(v00 + v_grid ** 2 * v11 + 2 * v_grid * v01, 0))
    z = stats.norm.ppf(1 - alpha / 2)
    return effect, effect - z * se, effect + z * se


def binned_slopes(prep, mod_name, n_bins=N_BINS_CHECK, alpha=CI_ALPHA):
    """Slope of Y~ on D~ within quantile bins of V: (median_v, slope, ci_lo, ci_hi, n)."""
    v = prep[f"V_{mod_name}"]
    bins = pd.qcut(v.rank(method="first"), q=n_bins, labels=False)
    z = stats.norm.ppf(1 - alpha / 2)
    out = []
    for b in range(n_bins):
        mask = (bins == b).values
        if mask.sum() < 10:
            continue
        Xb = pd.DataFrame({"D": prep.loc[mask, "D"].values})
        if INCLUDE_INTERCEPT:
            Xb = sm.add_constant(Xb, has_constant="add")
        rb = _fit(prep.loc[mask, "Y"].values, Xb)
        slope, se = rb.params["D"], rb.bse["D"]
        out.append((float(np.median(v[mask])), slope, slope - z * se, slope + z * se, int(mask.sum())))
    return out


# ---------------------------------------------------------------
# Bootstrap slopes within moderator tiers, adjusted for the other metrics
# ---------------------------------------------------------------

def _boot_p_two_sided(slopes):
    n = len(slopes)
    lower = np.sum(slopes <= 0) / n
    upper = np.sum(slopes >= 0) / n
    return float(min(1.0, 2 * min(lower, upper)))


def _tier_base(n_bins):
    return TIER_LABELS if n_bins == len(TIER_LABELS) else [f"Tier {i + 1}" for i in range(n_bins)]


def _tier_names(n_bins, mod_name):
    return [f"{t} {mod_name}" for t in _tier_base(n_bins)]


def _tier_ids(values, n_bins=N_BINS_CHECK):
    """Rank-based tertile ids 0..n_bins-1 (ties cannot merge bins)."""
    return pd.qcut(pd.Series(values).rank(method="first"), q=n_bins, labels=False).values


def _controls(D_all, focal):
    """The other metrics as separate covariates (None if there are none)."""
    others = D_all.drop(columns=[focal])
    return others.values if others.shape[1] else None


def _adj_slope(D, Y, C=None):
    """OLS coefficient on D in Y ~ D (+ C), with or without an intercept (per INCLUDE_INTERCEPT)."""
    if C is None:
        return _ols_slope(D, Y)
    X = np.column_stack([D, C])
    if INCLUDE_INTERCEPT:
        X = np.column_stack([np.ones(len(D)), X])
    k = 1 if INCLUDE_INTERCEPT else 0
    try:
        beta = np.linalg.solve(X.T @ X, X.T @ Y)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(X, Y, rcond=None)[0]
    return float(beta[k])


def bootstrap_tier_slopes_adjusted(Y, D, C, tier_ids, n_boot=N_BOOT,
                                   n_bins=N_BINS_CHECK, seed=RANDOM_SEED):
    """Within each tier, bootstrap the slope of Y on the focal metric D, adjusted for C."""
    rng = np.random.default_rng(seed)
    out = []
    for b in range(n_bins):
        idx_b = np.where(tier_ids == b)[0]
        Db, Yb = D[idx_b], Y[idx_b]
        Cb = C[idx_b] if C is not None else None
        n = len(idx_b)
        slopes = np.empty(n_boot)
        for k in range(n_boot):
            i = rng.integers(0, n, n)
            slopes[k] = _adj_slope(Db[i], Yb[i], Cb[i] if Cb is not None else None)
        out.append({"tier": b, "n": n, "slopes": slopes,
                    "estimate": _adj_slope(Db, Yb, Cb), "p": _boot_p_two_sided(slopes)})
    return out


def run_bootstrap_tiers(joint, mod_name, n_boot=N_BOOT):
    """Tier slopes for every metric, tiers cut on moderator `mod_name`. Returns {metric: [tier dicts]}."""
    tier_ids = _tier_ids(joint["V"][mod_name].values)
    boot = {}
    for metric in joint["D"].columns:
        boot[metric] = bootstrap_tier_slopes_adjusted(
            joint["Y"], joint["D"][metric].values, _controls(joint["D"], metric),
            tier_ids, n_boot=n_boot)
    return boot


def bootstrap_summary_table(boot, mod_name, k=None):
    """One row per metric x tier; *_rel_total columns are on the plotted scale."""
    k_map = k or {}
    rows = []
    for metric_name, tiers in boot.items():
        kk = k_map.get(metric_name, 1.0)
        names = _tier_names(len(tiers), mod_name)
        for t in tiers:
            lo, hi = np.percentile(t["slopes"], BOX_WHIS)
            med = float(np.median(t["slopes"]))
            rows.append({
                "tiers_cut_on": mod_name,
                "tier": names[t["tier"]], "diversity_metric": metric_name,
                "n": t["n"], "slope_estimate": t["estimate"], "boot_median": med,
                "boot_ci_lower": lo, "boot_ci_upper": hi,
                "slope_estimate_rel_total": kk * t["estimate"], "boot_median_rel_total": kk * med,
                "boot_ci_lower_rel_total": kk * lo, "boot_ci_upper_rel_total": kk * hi,
                "boot_p_two_sided": t["p"], "significant": t["p"] < BOOT_ALPHA,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------
# Labels
# ---------------------------------------------------------------

def _target():
    return "stability (1/NPP SD)" if REPORT_AS_STABILITY else "NPP SD"


def _effect_label():
    if _scaled():
        return r"Effect of diversity, $\theta(V)$" + f"\n({_scale_name()})"
    return (f"Effect of diversity on {_target()}, " + r"$\theta(V)$"
            + ("\n(per 1 SD of diversity)" if STANDARDISE_TREATMENT else ""))


def _slope_label():
    base = "Tier slope, adjusted for the other metrics (bootstrap)"
    return f"{base}\n{_scale_name()}" if _scaled() else base


def _moderator_label(mod_name):
    return f"{mod_name} (" + ("SD from mean" if STANDARDISE_MODERATOR else "centred") + ")"


# ---------------------------------------------------------------
# Tier boxplots
# ---------------------------------------------------------------

def _draw_box(ax, slopes, pos, color):
    """One horizontal bootstrap box; returns the whisker ends (for axis limits)."""
    bp = ax.boxplot(slopes, positions=[pos], vert=False, widths=0.7, whis=BOX_WHIS,
                    patch_artist=True, showfliers=False, manage_ticks=False)
    for box in bp["boxes"]:
        box.set(facecolor=color, edgecolor="#555555", alpha=0.85, linewidth=1)
    for med in bp["medians"]:
        med.set(color="#333333", linewidth=1.3)
    for part in bp["whiskers"] + bp["caps"]:
        part.set(color="#555555", linewidth=1)
    lo, hi = np.percentile(slopes, BOX_WHIS)
    return lo, hi


def _finish_tier_axes(ax, x_extent, xlim, yticks, ylabels, title, handles):
    ax.axvline(0, color="gray", linestyle=":", linewidth=1)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_xlabel(_slope_label())
    ax.set_title(title, fontsize=10)
    if xlim is None:
        pad = (max(x_extent) - min(x_extent)) * 0.08
        xlim = (min(x_extent + [0.0]) - pad, max(x_extent + [0.0]) + pad)
    ax.set_xlim(*xlim)
    ax.legend(handles=handles, loc="upper right", fontsize=8, frameon=True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def plot_tier_boxplots(boot, mod_name, savepath, k=None, xlim=None):
    """Tiers cut on a MODERATOR: one block per tier, one box per metric; k = {metric: scale factor}."""
    k = k or {}
    metric_names = list(boot.keys())
    n_m = len(metric_names)
    n_tiers = len(next(iter(boot.values())))
    tier_names = _tier_names(n_tiers, mod_name)
    fallback = plt.cm.tab10(np.arange(n_m))

    fig, ax = plt.subplots(figsize=(9, 0.55 * n_m * n_tiers + 2.5))
    yticks, ylabels, x_extent = [], [], []

    for t in range(n_tiers):
        base = (n_tiers - 1 - t) * (n_m + 1)
        for m_i, metric_name in enumerate(metric_names):
            res = boot[metric_name][t]
            slopes = k.get(metric_name, 1.0) * res["slopes"]
            color = METRIC_COLORS.get(metric_name, fallback[m_i])
            pos = base + m_i
            lo, hi = _draw_box(ax, slopes, pos, color)
            x_extent += [lo, hi]
            yticks.append(pos)
            ylabels.append(f"{tier_names[t]} - {metric_name} (n={res['n']})")
            if res["p"] < BOOT_ALPHA:
                ax.text(hi, pos, "  *", va="center", ha="left", fontsize=13, fontweight="bold")

    handles = [Patch(facecolor=METRIC_COLORS.get(m, fallback[i]), edgecolor="#555555", label=m)
               for i, m in enumerate(metric_names)]
    _finish_tier_axes(ax, x_extent, xlim, yticks, ylabels,
                      f"Diversity-stability slope by {mod_name} tier, all metrics combined "
                      f"(* = p < {BOOT_ALPHA:g}, two-sided bootstrap)", handles)
    fig.tight_layout()
    fig.savefig(savepath, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------
# Shared axis limits (in plotted, scaled units)
# ---------------------------------------------------------------

def _panel_extent(r, mod_name, k=1.0):
    v = r["prep"][f"V_{mod_name}"].values
    lo_v, hi_v = np.percentile(v, [1, 99])
    grid = np.linspace(lo_v, hi_v, N_GRID)
    _, lo, hi = conditional_effect(r["res"], mod_name, grid)
    ys = [k * lo.min(), k * hi.max()]
    for _, _, s_lo, s_hi, _ in binned_slopes(r["prep"], mod_name):
        ys += [k * s_lo, k * s_hi]
    return lo_v, hi_v, min(ys + [0.0]), max(ys + [0.0])


def compute_shared_limits(marginal_results, mod_names, pad_frac=AXIS_PAD_FRAC):
    x_lo, x_hi, y_lo, y_hi = [], [], [], []
    for metric_name, mods in marginal_results.items():
        for mod_name in mod_names:
            k = scale_factor(marginal_results, mod_name, metric_name)
            a, b, c, d = _panel_extent(mods[mod_name], mod_name, k)
            x_lo.append(a); x_hi.append(b); y_lo.append(c); y_hi.append(d)
    x0, x1, y0, y1 = min(x_lo), max(x_hi), min(y_lo), max(y_hi)
    xpad, ypad = (x1 - x0) * pad_frac, (y1 - y0) * pad_frac
    return (x0 - xpad, x1 + xpad), (y0 - ypad, y1 + ypad)


# ---------------------------------------------------------------
# Summary
# ---------------------------------------------------------------

def _coef_row(res, name, alpha=CI_ALPHA):
    ci = res.conf_int(alpha=alpha).loc[name]
    return res.params[name], ci.iloc[0], ci.iloc[1], res.pvalues[name]


def build_summary(marginal_results, fdr_alpha=FDR_ALPHA):
    rows = []
    for metric_name, mods in marginal_results.items():
        for mod_name, r in mods.items():
            res = r["res"]
            t0, t0_lo, t0_hi, t0_p = _coef_row(res, "D")
            t1, t1_lo, t1_hi, t1_p = _coef_row(res, f"DxV_{mod_name}")
            rows.append({
                "model": "marginal", "diversity_metric": metric_name, "moderator": mod_name,
                "n": r["n"], "theta_total": r["theta_total"],
                "theta_0": t0, "theta_0_ci_lower": t0_lo, "theta_0_ci_upper": t0_hi, "theta_0_p": t0_p,
                "theta_1": t1, "theta_1_ci_lower": t1_lo, "theta_1_ci_upper": t1_hi, "theta_1_p": t1_p,
                "moderator_mean_raw": r["mod_info"]["mean"], "moderator_sd_raw": r["mod_info"]["sd"],
                "treatment_sd_raw": r["d_sd"],
            })

    summary = pd.DataFrame(rows)
    reject, p_adj, _, _ = multipletests(summary["theta_1_p"].values, alpha=fdr_alpha, method="fdr_bh")
    summary["theta_1_p_fdr"] = p_adj
    summary["theta_1_significant_fdr"] = reject

    mods = summary["moderator"].unique()
    k = summary.apply(lambda r: scale_factor(marginal_results, r["moderator"],
                                             r["diversity_metric"]), axis=1)
    summary["scale_factor"] = k
    summary["scale_definition"] = NET_EFFECT_SCALE
    summary["total_diversity_effect"] = summary["moderator"].map(
        {m: total_diversity_effect(marginal_results, m) for m in mods})
    for c in ["theta_total", "theta_0", "theta_0_ci_lower", "theta_0_ci_upper",
              "theta_1", "theta_1_ci_lower", "theta_1_ci_upper"]:
        summary[f"{c}_rel_total"] = summary[c] * k
    return summary


# ---------------------------------------------------------------
# Plots
# ---------------------------------------------------------------

def _create_metric_grid(n_metrics, n_cols=GRID_N_COLS, panel_width=5.5, panel_height=4.0):
    n_rows = int(np.ceil(n_metrics / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(panel_width * n_cols, panel_height * n_rows),
                             squeeze=False)
    axes_flat = list(axes.flatten())
    for ax in axes_flat[n_metrics:]:
        ax.set_visible(False)
    return fig, axes_flat[:n_metrics]


def plot_conditional_effects(marginal_results, mod_name, summary, savepath, xlim, ylim):
    """theta(V) with CI band, tertile-slope check and rug, all in scaled units, shared axes."""
    rng = np.random.default_rng(RANDOM_SEED)
    metric_names = list(marginal_results.keys())
    color = MODERATOR_COLORS.get(mod_name, "navy")
    fig, axes = _create_metric_grid(len(metric_names))

    for ax, metric_name in zip(axes, metric_names):
        r = marginal_results[metric_name][mod_name]
        k = scale_factor(marginal_results, mod_name, metric_name)
        v = r["prep"][f"V_{mod_name}"].values
        lo_v, hi_v = np.percentile(v, [1, 99])
        grid = np.linspace(lo_v, hi_v, N_GRID)
        eff, lo, hi = (k * a for a in conditional_effect(r["res"], mod_name, grid))

        ax.fill_between(grid, lo, hi, color=color, alpha=0.2, linewidth=0)
        ax.plot(grid, eff, color=color, linewidth=2, label="Linear moderation model")
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, zorder=0)

        for med_v, slope, s_lo, s_hi, _ in binned_slopes(r["prep"], mod_name):
            slope, s_lo, s_hi = k * slope, k * s_lo, k * s_hi
            ax.errorbar(med_v, slope, yerr=[[slope - s_lo], [s_hi - slope]], fmt="o",
                        color="black", markersize=4, capsize=3, linewidth=1, zorder=4)
        ax.plot([], [], "o", color="black", markersize=4, label="Tertile slopes (check)")

        idx = rng.choice(len(v), size=min(RUG_MAX_POINTS, len(v)), replace=False)
        trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
        ax.plot(v[idx], np.full(len(idx), 0.01), "|", color=color, alpha=0.15,
                markersize=6, transform=trans)

        row = summary[(summary.diversity_metric == metric_name) & (summary.moderator == mod_name)].iloc[0]
        star = " *" if row["theta_1_significant_fdr"] else ""
        if _scaled():
            t1 = (f" = {row['theta_1_rel_total']:.1f}% "
                  f"[{row['theta_1_ci_lower_rel_total']:.1f}, {row['theta_1_ci_upper_rel_total']:.1f}]")
        else:
            t1 = f" = {row['theta_1']:.3g} [{row['theta_1_ci_lower']:.3g}, {row['theta_1_ci_upper']:.3g}]"
        ax.set_title(f"{metric_name}\n" + r"$\theta_1$" + t1 + star, fontsize=10)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel(_moderator_label(mod_name), fontsize=9)
        ax.set_ylabel(_effect_label(), fontsize=9)

    axes[0].legend(fontsize=8, loc="best")
    fig.suptitle(f"Conditional effect of diversity on stability across {mod_name}  "
                 f"(* = moderation significant after FDR correction)", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(savepath, dpi=150, bbox_inches="tight")
    return fig


def plot_theta1_forest(summary, savepath):
    """Forest plot of theta_1 for every metric x moderator, on the reporting scale."""
    sfx = "_rel_total" if _scaled() else ""
    metrics = list(dict.fromkeys(summary["diversity_metric"]))
    mods = list(dict.fromkeys(summary["moderator"]))
    offsets = np.linspace(-0.15, 0.15, len(mods)) if len(mods) > 1 else [0.0]

    fig, ax = plt.subplots(figsize=(7, 0.8 * len(metrics) + 1.5))
    for m_i, mod_name in enumerate(mods):
        rows = summary[summary.moderator == mod_name].set_index("diversity_metric").loc[metrics]
        y = np.arange(len(metrics))[::-1] + offsets[m_i]
        est = rows[f"theta_1{sfx}"].values
        ci_lo, ci_hi = rows[f"theta_1_ci_lower{sfx}"].values, rows[f"theta_1_ci_upper{sfx}"].values
        ax.errorbar(est, y, xerr=[est - ci_lo, ci_hi - est], fmt="o",
                    color=MODERATOR_COLORS.get(mod_name, "navy"), capsize=3, label=mod_name)
        for yi, xi, sig in zip(y, ci_hi, rows["theta_1_significant_fdr"].values):
            if sig:
                ax.text(xi, yi, "  *", va="center", fontsize=12, fontweight="bold")

    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_yticks(np.arange(len(metrics))[::-1])
    ax.set_yticklabels(metrics)
    per = " per 1 SD of moderator" if STANDARDISE_MODERATOR else ""
    if _scaled():
        ax.set_xlabel(r"Moderation effect $\theta_1$" + f" ({_scale_name()}{per})")
    else:
        ax.set_xlabel(r"Moderation effect $\theta_1$" + (f" ({per.strip()})" if per else ""))
    ax.set_title("Moderation of the diversity-stability effect\n* = significant after FDR correction",
                 fontsize=11)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(savepath, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------

def run_moderation_analysis(df, outdir=OUTDIR):
    os.makedirs(outdir, exist_ok=True)
    marginal = fit_marginal_models(df)

    summary = build_summary(marginal)
    summary_path = _out(OUT_SUMMARY, outdir)
    summary.to_csv(summary_path, index=False)
    print(f"[run_moderation_analysis] Summary saved to {summary_path}")
    cols = ["diversity_metric", "moderator", "n", "total_diversity_effect",
            "theta_total_rel_total", "theta_0_rel_total", "theta_1_rel_total",
            "theta_1_ci_lower_rel_total", "theta_1_ci_upper_rel_total", "theta_1_p_fdr"]
    print(summary[cols].to_string(index=False))

    # Conditional-effect plots share axes across all moderators
    xlim, ylim = compute_shared_limits(marginal, list(MODERATORS))
    for mod_name in MODERATORS:
        plot_conditional_effects(marginal, mod_name, summary,
                                 savepath=_out(OUT_CONDITIONAL_FIG, outdir, mod_name),
                                 xlim=xlim, ylim=ylim)
    plot_theta1_forest(summary, _out(OUT_THETA1_FOREST_FIG, outdir))

    # ---- Tier boxplots: slope of each metric within moderator tiers, adjusted for the others ----
    joint = _prepare_joint(df)
    k_joint = joint_scale_factors(joint)
    print(f"[run_moderation_analysis] Tier slopes are adjusted for the other metrics; "
          f"scale = {NET_EFFECT_SCALE}")

    for mod_name in MODERATORS:
        print(f"[run_moderation_analysis] Bootstrapping tier slopes, tiers cut on {mod_name} "
              f"({N_BOOT} resamples x {len(DIVERSITY_METRICS)} metrics x {N_BINS_CHECK} tiers)...")
        boot = run_bootstrap_tiers(joint, mod_name)
        bootstrap_summary_table(boot, mod_name, k_joint).to_csv(
            _out(OUT_TIER_BOOT_CSV, outdir, mod_name), index=False)
        plot_tier_boxplots(boot, mod_name, k=k_joint,
                           savepath=_out(OUT_TIER_BOX_FIG, outdir, mod_name))

    return marginal, summary


if __name__ == "__main__":
    residuals_df = pd.read_csv(RESIDUALS_PATH)
    residuals_df = residuals_df.drop(columns=["residual_simpsons"], errors="ignore")

    PID_df= pd.read_csv('data/lookup/PID_location_v3.csv')
    residuals_df=residuals_df.merge(PID_df[['PID', 'STDAGE', 'percent_conifer']], on='PID', how='left')

    residuals_df=residuals_df[residuals_df.percent_conifer < 40]

    marginal, summary = run_moderation_analysis(residuals_df)
    plt.show()