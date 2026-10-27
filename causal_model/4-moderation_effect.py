"""
Moderation of the diversity -> stability effect by RARITY and % CONIFER.

FIGURE 1: coefficient (forest) plot
  One model per diversity metric, fitted on the DML residuals:

      Y~ = a + theta_0 D~ + theta_R (D~ x R) + theta_C (D~ x C) [+ theta_RC (D~ x R x C)]
             [+ R + C (+ R x C)]                     (moderator main effects, optional)
             [+ the other diversity metrics]         (optional adjustment)
             + eps

  R = residual % rare species, C = % conifer; both mean-centred and scaled to 1 SD, so
      theta_0  = effect of diversity at AVERAGE rarity and AVERAGE % conifer
      theta_R  = change in that effect per 1 SD of rarity     (at average % conifer)
      theta_C  = change in that effect per 1 SD of % conifer  (at average rarity)
      theta_RC = change in theta_R per 1 SD of % conifer       (if INCLUDE_THREE_WAY)
  Point = estimate, bar = (1 - CI_ALPHA) CI from HC3 standard errors.
  Asterisk: main effect p < CI_ALPHA; interactions significant after Benjamini-Hochberg FDR
  (across all interaction terms of all metrics).

FIGURE 2: slopes by forest type x rarity tier
  Forest types from RAW % conifer (FOREST_TYPES):
      Broadleaf  < 10 % conifer,   Mixed 25-75 %,   Conifer-dominated > 90 %.
  Rarity tiers (RARITY_TIERS, rank-based percentiles of rarity): bottom 10 % and top 10 %,
  computed WITHIN each forest type (RARITY_TIERS_WITHIN_FOREST_TYPE = True) or across all
  plots (False). In every forest type x rarity cell the slope of Y~ on D~ is fitted by OLS
  (HC3 CI). Above each forest type: FDR-corrected p of the high-vs-low rarity difference
  (D~ x high-rarity interaction within that forest type).

SCALE (NET_EFFECT_SCALE)
  "own_total": effects expressed as % of the metric's own unmoderated, unadjusted slope
               |theta_total| (Y~ on D~ on the same rows). 100 % = the original effect.
  "none":      raw units (change in the stability residual per 1 SD of diversity).

With REPORT_AS_STABILITY = True the outcome is multiplied by -1, so positive = diversity
lowers NPP SD = more stable. All metrics and both moderators use ONE common set of rows.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.lines import Line2D
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from scipy import stats

# ===============================================================
# Input and output files
# ===============================================================

RESIDUALS_PATH = "results/step_2_residuals_merged.csv"
PID_LOOKUP_PATH = "data/lookup/PID_location_v3.csv"
OUTDIR = "results/moderation"

OUT_COEF_CSV = "moderation_coefficients_rarity_conifer.csv"
OUT_COEF_FIG = "moderation_coefficients_rarity_conifer.png"          # + .pdf
OUT_CELL_CSV = "moderation_slopes_foresttype_x_rarity.csv"
OUT_CELL_FIG = "moderation_slopes_foresttype_x_rarity.png"           # + .pdf

# ===============================================================
# Columns
# ===============================================================

DIVERSITY_METRICS = {
    "Rao's Q": "residual_raos_q",
    "Functional Evenness": "residual_functional_evenness",
    "Species richness": "residual_species_richness",
}
STABILITY_COL = "residual_std_npp"
RARITY_COL = "residual_pct_rare_species"
CONIFER_COL = "percent_conifer"          # raw %, 0-100 (0-1 is detected and converted)

# ===============================================================
# Settings
# ===============================================================

STANDARDISE_TREATMENT = True
STANDARDISE_MODERATOR = True
INCLUDE_INTERCEPT = True
# % conifer is not a residual, so it is probably not in the nuisance models: keep its
# main effect (and rarity's) in the model so the interactions do not absorb it.
INCLUDE_MODERATOR_MAIN_EFFECTS = True
INCLUDE_THREE_WAY = True                 # D x Rarity x Conifer term in Figure 1
ADJUST_FOR_OTHER_METRICS = False         # True = other metrics as covariates (both figures)

REPORT_AS_STABILITY = True
OUTCOME_SIGN = -1 if REPORT_AS_STABILITY else 1

NET_EFFECT_SCALE = "own_total"           # "own_total" or "none"
OWN_TOTAL_WARN_FRAC = 0.1

CI_ALPHA = 0.05
FDR_ALPHA = 0.05

# Forest types in RAW % conifer: (label, lower, upper, closed), closed as in pd.Interval
# ("left", "right", "both", "neither").
FOREST_TYPES = [
    ("Broadleaf (< 10% conifer)", -np.inf, 10, "neither"),
    ("Mixed (25\u201375% conifer)", 25, 75, "both"),
    ("Conifer-dominated (> 90% conifer)", 90, np.inf, "neither"),
]
# Rarity tiers as percentile ranges (rank-based): (label, lower pct, upper pct)
RARITY_TIERS = [
    ("Low rarity (bottom 10%)", 0, 10),
    ("High rarity (top 10%)", 90, 100),
]
RARITY_TIERS_WITHIN_FOREST_TYPE = True
MIN_CELL_N = 30

METRIC_COLORS = {
    "Rao's Q": "#4B2BC4",
    "Functional Evenness": "#A47CC7",
    "Shannon": "#5ED6DA",
    "Species richness": "#14B8A0",
}
TERM_LABELS = {
    "D": "Main effect (avg. rarity & conifer)",
    "DxRarity": "\u00d7 Rarity",
    "DxConifer": "\u00d7 % conifer",
    "DxRarityxConifer": "\u00d7 Rarity \u00d7 % conifer",
}
TERM_COLORS = {
    "D": "#333333",
    "DxRarity": "#1C33B5",
    "DxConifer": "#087127",
    "DxRarityxConifer": "#B5651C",
}
GROUP_TINT = 0.40     # lightest group colour = this share of the metric colour
GROUP_SHADE = 0.55    # darkest group colour = metric colour mixed with this much black


def _out(filename, outdir=None):
    return os.path.join(outdir or OUTDIR, filename)


# ===============================================================
# Data preparation
# ===============================================================

def _ols_slope(D, Y):
    D, Y = np.asarray(D, float), np.asarray(Y, float)
    if INCLUDE_INTERCEPT:
        Dc = D - D.mean()
        return float((Dc * (Y - Y.mean())).sum() / (Dc * Dc).sum())
    return float((D * Y).sum() / (D * D).sum())


def prepare(df):
    """Y, standardised D (one column per metric), standardised and raw moderators, on common rows."""
    cols = [STABILITY_COL] + list(DIVERSITY_METRICS.values()) + [RARITY_COL, CONIFER_COL]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Columns not found in the data: {missing}")
    sub = df[cols].dropna().reset_index(drop=True)

    conifer = sub[CONIFER_COL].astype(float)
    if conifer.max() <= 1.0:
        print("  NOTE: % conifer looks like a 0-1 proportion; multiplying by 100.")
        conifer = conifer * 100
    rarity = sub[RARITY_COL].astype(float)

    Y = pd.Series(OUTCOME_SIGN * sub[STABILITY_COL].astype(float).values)
    D, d_sd = pd.DataFrame(index=sub.index), {}
    for name, col in DIVERSITY_METRICS.items():
        d = sub[col].astype(float)
        d_sd[name] = d.std(ddof=1) if STANDARDISE_TREATMENT else 1.0
        D[name] = d / d_sd[name]

    raw = pd.DataFrame({"Rarity": rarity, "Conifer": conifer})
    V, v_info = pd.DataFrame(index=sub.index), {}
    for name in raw.columns:
        v = raw[name]
        sd = v.std(ddof=1) if STANDARDISE_MODERATOR else 1.0
        V[name] = (v - v.mean()) / sd
        v_info[name] = {"mean": v.mean(), "sd": sd}

    theta_total = {m: _ols_slope(D[m], Y) for m in D.columns}
    print(f"[prepare] n = {len(sub)} plots with every metric and moderator present")
    return {"Y": Y, "D": D, "V": V, "raw": raw, "d_sd": d_sd, "v_info": v_info,
            "theta_total": theta_total, "n": len(sub)}


def scale_factors(prep):
    """{metric: multiplier} from raw theta units to the reporting scale."""
    th = prep["theta_total"]
    if NET_EFFECT_SCALE != "own_total":
        return {m: 1.0 for m in th}
    largest = max(abs(v) for v in th.values())
    k = {}
    for m, v in th.items():
        if abs(v) < OWN_TOTAL_WARN_FRAC * largest:
            print(f"  WARNING: {m}'s unmoderated effect is close to 0; its % values will be "
                  f"very large and unstable. Consider NET_EFFECT_SCALE = 'none'.")
        k[m] = 100.0 / abs(v) if abs(v) > 0 else 1.0
    return k


def _scaled():
    return NET_EFFECT_SCALE == "own_total"


# ===============================================================
# Model helpers
# ===============================================================

def _design(cols):
    X = pd.DataFrame(cols)
    if INCLUDE_INTERCEPT:
        X = sm.add_constant(X, has_constant="add")
    return X


def _fit(y, X):
    """OLS with HC3 robust standard errors."""
    return sm.OLS(np.asarray(y, dtype=float), X).fit(cov_type="HC3")


def _control_cols(prep, focal, mask=None):
    """The other metrics as covariates (empty unless ADJUST_FOR_OTHER_METRICS)."""
    if not ADJUST_FOR_OTHER_METRICS:
        return {}
    out = {}
    for other in prep["D"].columns:
        if other != focal:
            vals = prep["D"][other].to_numpy(float)
            out[f"adj_{other}"] = vals if mask is None else vals[mask]
    return out


def _coef(res, name):
    lo, hi = res.conf_int(alpha=CI_ALPHA).loc[name]
    return float(res.params[name]), float(lo), float(hi), float(res.pvalues[name])


def _p_text(p, prefix=""):
    return prefix + ("$p_{FDR}$ < 0.001" if p < 0.001 else f"$p_{{FDR}}$ = {p:.2g}")


def _group_colors(base_color, n):
    """n ordered shades of base_color: light tint -> base -> dark shade."""
    base = np.array(plt.matplotlib.colors.to_rgb(base_color))
    tint = 1 - GROUP_TINT * (1 - base)
    shade = base * (1 - GROUP_SHADE)
    if n == 1:
        return [tuple(base)]
    out = []
    for t in np.linspace(0, 1, n):
        c = tint + (base - tint) * (t / 0.5) if t <= 0.5 else base + (shade - base) * ((t - 0.5) / 0.5)
        out.append(tuple(np.clip(c, 0, 1)))
    return out


# ===============================================================
# Figure 1: main effect + interactions with rarity and % conifer
# ===============================================================

def fit_interaction_model(prep, metric):
    D = prep["D"][metric]
    R, C = prep["V"]["Rarity"], prep["V"]["Conifer"]
    cols = {"D": D, "DxRarity": D * R, "DxConifer": D * C}
    if INCLUDE_THREE_WAY:
        cols["DxRarityxConifer"] = D * R * C
    if INCLUDE_MODERATOR_MAIN_EFFECTS:
        cols["Rarity"], cols["Conifer"] = R, C
        if INCLUDE_THREE_WAY:
            cols["RarityxConifer"] = R * C
    cols.update(_control_cols(prep, metric))
    return _fit(prep["Y"], _design(cols))


def build_coef_table(prep, fits, k):
    rows = []
    for metric, res in fits.items():
        for term, label in TERM_LABELS.items():
            if term not in res.params.index:
                continue
            est, lo, hi, p = _coef(res, term)
            kk = k[metric]
            rows.append({
                "diversity_metric": metric, "term": term, "term_label": label, "n": prep["n"],
                "estimate": est, "ci_lower": lo, "ci_upper": hi, "se_hc3": float(res.bse[term]),
                "p": p, "theta_total": prep["theta_total"][metric], "scale_factor": kk,
                "scale_definition": NET_EFFECT_SCALE,
                "estimate_scaled": kk * est, "ci_lower_scaled": kk * lo, "ci_upper_scaled": kk * hi,
                "treatment_sd_raw": prep["d_sd"][metric],
                "rarity_mean_raw": prep["v_info"]["Rarity"]["mean"],
                "rarity_sd_raw": prep["v_info"]["Rarity"]["sd"],
                "conifer_mean_raw": prep["v_info"]["Conifer"]["mean"],
                "conifer_sd_raw": prep["v_info"]["Conifer"]["sd"],
            })
    tab = pd.DataFrame(rows)
    inter = (tab["term"] != "D").to_numpy()
    tab["p_fdr"] = np.nan
    rej, p_adj, _, _ = multipletests(tab.loc[inter, "p"].values, alpha=FDR_ALPHA, method="fdr_bh")
    tab.loc[inter, "p_fdr"] = p_adj
    tab["significant"] = False
    tab.loc[inter, "significant"] = rej
    tab.loc[~inter, "significant"] = tab.loc[~inter, "p"] < CI_ALPHA
    return tab


def plot_coefficients(tab, savepath):
    metrics = list(dict.fromkeys(tab["diversity_metric"]))
    terms = [t for t in TERM_LABELS if t in set(tab["term"])]
    n_t = len(terms)
    block = n_t + 1                                     # + 1 row for the metric header

    fig, ax = plt.subplots(figsize=(8.5, 0.42 * len(metrics) * block + 1.8))
    trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
    yticks, ylabels, x_ext = [], [], [0.0]

    for i, metric in enumerate(metrics):
        base = (len(metrics) - 1 - i) * block
        if i > 0:
            ax.axhline(base + block - 0.5, color="#e5e5e5", lw=0.8, zorder=0)
        for j, term in enumerate(terms):
            row = tab[(tab.diversity_metric == metric) & (tab.term == term)].iloc[0]
            pos = base + (n_t - 1 - j)
            est, lo, hi = row["estimate_scaled"], row["ci_lower_scaled"], row["ci_upper_scaled"]
            ax.errorbar(est, pos, xerr=[[est - lo], [hi - est]], fmt="o",
                        color=TERM_COLORS[term], ecolor=TERM_COLORS[term], markersize=6.5,
                        markeredgecolor="#333333", markeredgewidth=0.7,
                        elinewidth=2, capsize=3.5, zorder=3)
            if row["significant"]:
                ax.text(hi, pos, "  *", va="center", ha="left", fontsize=12, fontweight="bold")
            yticks.append(pos)
            ylabels.append(TERM_LABELS[term])
            x_ext += [lo, hi]
        ax.text(0.01, base + n_t - 0.15, metric, transform=trans, ha="left", va="center",
                fontsize=10.5, fontweight="bold", color=METRIC_COLORS.get(metric, "black"))

    ax.axvline(0, color="gray", ls=":", lw=1)
    if _scaled():
        ax.axvline(100, color="#bbbbbb", ls="--", lw=0.8)
    pad = (max(x_ext) - min(x_ext)) * 0.1
    ax.set_xlim(min(x_ext) - pad, max(x_ext) + pad)
    ax.set_ylim(-0.7, len(metrics) * block - 0.4)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=8.5)
    per = " per 1 SD" if STANDARDISE_MODERATOR else ""
    if _scaled():
        ax.set_xlabel("Effect on stability, % of the metric's unmoderated effect\n"
                      f"(interactions: change{per} of the moderator; dashed line = 100%)")
    else:
        ax.set_xlabel("Effect on stability residual per 1 SD of diversity\n"
                      f"(interactions: change{per} of the moderator)")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ci_pct = round((1 - CI_ALPHA) * 100)
    ax.set_title(f"Diversity effects and their moderation by rarity and % conifer\n"
                 f"(estimate \u00b1 {ci_pct}% CI, HC3)", fontsize=11)
    fig.text(0.01, -0.01, f"* main effect: p < {CI_ALPHA:g}; interactions: significant after "
             f"FDR (BH, q = {FDR_ALPHA:g}).", fontsize=7.5, color="#555555")
    fig.tight_layout()
    fig.savefig(savepath, dpi=300, bbox_inches="tight")
    fig.savefig(os.path.splitext(savepath)[0] + ".pdf", bbox_inches="tight")
    print(f"[figure] saved {savepath}")
    return fig


# ===============================================================
# Figure 2: slopes in forest type x rarity-tier cells
# ===============================================================

def _in_interval(x, lo, hi, closed):
    left = x >= lo if closed in ("left", "both") else x > lo
    right = x <= hi if closed in ("right", "both") else x < hi
    return left & right


def _pct_tier_ids(values, tiers):
    """Tier id per value from rank-based percentile ranges; -1 = in no tier."""
    v = pd.Series(np.asarray(values, dtype=float))
    pr = 100 * (v.rank(method="first") - 0.5) / len(v)
    ids = np.full(len(v), -1)
    for k, (_, lo, hi) in enumerate(tiers):
        upper = (pr <= hi) if hi == 100 else (pr < hi)
        ids[((pr >= lo) & upper).to_numpy()] = k
    return ids


def forest_type_ids(conifer_raw):
    conifer_raw = np.asarray(conifer_raw, dtype=float)
    ids = np.full(len(conifer_raw), -1)
    for k, (_, lo, hi, closed) in enumerate(FOREST_TYPES):
        ids[_in_interval(conifer_raw, lo, hi, closed) & (ids == -1)] = k
    return ids


def rarity_tier_ids(rarity_raw, f_ids):
    rarity_raw = np.asarray(rarity_raw, dtype=float)
    if not RARITY_TIERS_WITHIN_FOREST_TYPE:
        return _pct_tier_ids(rarity_raw, RARITY_TIERS)
    ids = np.full(len(rarity_raw), -1)
    for f in range(len(FOREST_TYPES)):
        m = f_ids == f
        if m.any():
            ids[m] = _pct_tier_ids(rarity_raw[m], RARITY_TIERS)
    return ids


def _difference_test(prep, metric, Y, D, f_mask, r_ids):
    """High-vs-low rarity slope difference within one forest type: D x High in
    Y ~ D + High + D x High (+ controls). Returns (difference, p)."""
    lo_id, hi_id = 0, len(RARITY_TIERS) - 1
    if min((f_mask & (r_ids == lo_id)).sum(), (f_mask & (r_ids == hi_id)).sum()) < MIN_CELL_N:
        return np.nan, np.nan
    m = f_mask & ((r_ids == lo_id) | (r_ids == hi_id))
    H = (r_ids[m] == hi_id).astype(float)
    cols = {"D": D[m], "High": H, "DxHigh": D[m] * H}
    cols.update(_control_cols(prep, metric, m))
    res = _fit(Y[m], _design(cols))
    return float(res.params["DxHigh"]), float(res.pvalues["DxHigh"])


def cell_slopes(prep, k):
    Y = prep["Y"].to_numpy(float)
    f_ids = forest_type_ids(prep["raw"]["Conifer"])
    r_ids = rarity_tier_ids(prep["raw"]["Rarity"], f_ids)
    rows = []
    for metric in prep["D"].columns:
        D = prep["D"][metric].to_numpy(float)
        kk = k[metric]
        for f, (f_lab, *_) in enumerate(FOREST_TYPES):
            f_mask = f_ids == f
            diff, diff_p = _difference_test(prep, metric, Y, D, f_mask, r_ids)
            for r, (r_lab, _, _) in enumerate(RARITY_TIERS):
                m = f_mask & (r_ids == r)
                n = int(m.sum())
                row = {"diversity_metric": metric, "forest_type": f_lab, "rarity_tier": r_lab,
                       "n": n, "n_forest_type": int(f_mask.sum()),
                       "rarity_raw_min": np.nan, "rarity_raw_max": np.nan,
                       "slope": np.nan, "slope_ci_lower": np.nan, "slope_ci_upper": np.nan,
                       "slope_p": np.nan, "scale_factor": kk,
                       "diff_high_minus_low": diff, "diff_high_minus_low_scaled": kk * diff,
                       "diff_p": diff_p}
                if n:
                    rr = prep["raw"]["Rarity"].to_numpy(float)[m]
                    row["rarity_raw_min"], row["rarity_raw_max"] = rr.min(), rr.max()
                if n >= MIN_CELL_N:
                    cols = {"D": D[m]}
                    cols.update(_control_cols(prep, metric, m))
                    est, lo, hi, p = _coef(_fit(Y[m], _design(cols)), "D")
                    row.update(slope=est, slope_ci_lower=lo, slope_ci_upper=hi, slope_p=p)
                row["slope_scaled"] = kk * row["slope"]
                row["slope_ci_lower_scaled"] = kk * row["slope_ci_lower"]
                row["slope_ci_upper_scaled"] = kk * row["slope_ci_upper"]
                rows.append(row)
    tab = pd.DataFrame(rows)

    # FDR across the high-vs-low tests (one per metric x forest type)
    key = tab.drop_duplicates(["diversity_metric", "forest_type"])[
        ["diversity_metric", "forest_type", "diff_p"]].copy()
    key["diff_p_fdr"] = np.nan
    ok = key["diff_p"].notna()
    if ok.any():
        key.loc[ok, "diff_p_fdr"] = multipletests(key.loc[ok, "diff_p"].values,
                                                  alpha=FDR_ALPHA, method="fdr_bh")[1]
    tab = tab.merge(key[["diversity_metric", "forest_type", "diff_p_fdr"]],
                    on=["diversity_metric", "forest_type"], how="left")

    print("[cells] plots per forest type x rarity tier (first metric):")
    first = tab[tab.diversity_metric == tab.diversity_metric.iloc[0]]
    print(first[["forest_type", "rarity_tier", "n", "n_forest_type"]].to_string(index=False))
    return tab


def plot_cell_slopes(tab, savepath, panel_width=4.6, panel_height=4.4):
    metrics = list(dict.fromkeys(tab["diversity_metric"]))
    f_labels = [f[0] for f in FOREST_TYPES]
    r_labels = [r[0] for r in RARITY_TIERS]
    n_r = len(r_labels)
    offs = np.linspace(-0.17, 0.17, n_r) if n_r > 1 else [0.0]

    fig, axes = plt.subplots(1, len(metrics), figsize=(panel_width * len(metrics), panel_height),
                             squeeze=False)
    axes = axes[0]

    for ax, metric in zip(axes, metrics):
        colors = _group_colors(METRIC_COLORS.get(metric, "#555555"), n_r)
        sub = tab[tab.diversity_metric == metric]
        # own y-range per metric (one unstable metric must not squash the others)
        ys = np.r_[sub["slope_ci_lower_scaled"].dropna(), sub["slope_ci_upper_scaled"].dropna(), 0.0]
        span = (ys.max() - ys.min()) or 1.0
        ylim = (ys.min() - 0.12 * span, ys.max() + 0.18 * span)   # headroom for the p labels
        trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
        for f, f_lab in enumerate(f_labels):
            for r, r_lab in enumerate(r_labels):
                row = sub[(sub.forest_type == f_lab) & (sub.rarity_tier == r_lab)].iloc[0]
                x = f + offs[r]
                if not np.isfinite(row["slope_scaled"]):
                    ax.text(x, 0, f"n={row['n']}\n(too few)", ha="center", va="center",
                            fontsize=6.5, color="#888888")
                    continue
                est = row["slope_scaled"]
                lo, hi = row["slope_ci_lower_scaled"], row["slope_ci_upper_scaled"]
                ax.errorbar(x, est, yerr=[[est - lo], [hi - est]], fmt="o", color=colors[r],
                            ecolor=colors[r], markersize=7, markeredgecolor="#333333",
                            markeredgewidth=0.7, elinewidth=2, capsize=4, zorder=3)
                ax.annotate(f"n={row['n']}", (x, lo), xytext=(0, -3), textcoords="offset points",
                            ha="center", va="top", fontsize=6.5, color="#555555")
            p = sub[sub.forest_type == f_lab]["diff_p_fdr"].iloc[0]
            if np.isfinite(p):
                ax.text(f, 0.98, _p_text(p, "\u0394 "), transform=trans, ha="center", va="top",
                        fontsize=7.5, fontweight="bold" if p < FDR_ALPHA else "normal")

        ax.axhline(0, color="gray", ls=":", lw=1, zorder=0)
        if _scaled():
            ax.axhline(100, color="#bbbbbb", ls="--", lw=0.8, zorder=0)
        ax.set_xticks(range(len(f_labels)))
        ax.set_xticklabels([lab.replace(" (", "\n(") for lab in f_labels], fontsize=8)
        ax.set_xlim(-0.6, len(f_labels) - 0.4)
        ax.set_ylim(*ylim)
        ax.set_title(metric, fontsize=11, fontweight="bold", color=METRIC_COLORS.get(metric, "black"))
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    if _scaled():
        axes[0].set_ylabel("Slope of stability on diversity\n(% of the metric's unmoderated "
                           "effect;\ndashed line = 100%)", fontsize=9)
    else:
        axes[0].set_ylabel("Slope of stability residual\nper 1 SD of diversity", fontsize=9)

    greys = _group_colors("#555555", n_r)
    within = " (within each forest type)" if RARITY_TIERS_WITHIN_FOREST_TYPE else " (all plots)"
    handles = [Line2D([], [], color=g, marker="o", lw=2, markeredgecolor="#333333",
                      markersize=6, label=lab + within) for g, lab in zip(greys, r_labels)]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0),
               ncol=n_r, fontsize=8.5, frameon=False)
    ci_pct = round((1 - CI_ALPHA) * 100)
    fig.suptitle(f"Diversity-stability slope by forest type and rarity (OLS \u00b1 {ci_pct}% CI, "
                 f"HC3; \u0394 = high vs low rarity, FDR)", fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(savepath, dpi=300, bbox_inches="tight")
    fig.savefig(os.path.splitext(savepath)[0] + ".pdf", bbox_inches="tight")
    print(f"[figure] saved {savepath}")
    return fig


# ===============================================================
# Entry point
# ===============================================================

def run_moderation_analysis(df, outdir=OUTDIR):
    os.makedirs(outdir, exist_ok=True)
    prep = prepare(df)
    k = scale_factors(prep)

    fits = {m: fit_interaction_model(prep, m) for m in prep["D"].columns}
    coef = build_coef_table(prep, fits, k)
    coef.to_csv(_out(OUT_COEF_CSV, outdir), index=False)
    plot_coefficients(coef, _out(OUT_COEF_FIG, outdir))

    cells = cell_slopes(prep, k)
    cells.to_csv(_out(OUT_CELL_CSV, outdir), index=False)
    plot_cell_slopes(cells, _out(OUT_CELL_FIG, outdir))
    return coef, cells


if __name__ == "__main__":
    residuals_df = pd.read_csv(RESIDUALS_PATH)
    residuals_df = residuals_df.drop(columns=["residual_simpsons"], errors="ignore")

    PID_df = pd.read_csv(PID_LOOKUP_PATH)
    residuals_df = residuals_df.merge(PID_df[["PID", "STDAGE", "percent_conifer"]], on="PID", how="left")

    coef, cells = run_moderation_analysis(residuals_df)
    plt.show()