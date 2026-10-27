"""
Moderated mediation of the diversity -> stability effect via structural
complexity (WSCI), with BOTH paths allowed to change along a plot-level
moderator V (% conifer, % rare species). Estimated on the DML residuals.

    a-path:   M~ = a0 D~ + a1 D~V + a2 V                     (+ other metrics, x V) + e1
    outcome:  Y~ = c'0 D~ + c'1 D~V + b0 M~ + b1 M~V + g V   (+ other metrics, x V) + e2

    conditional indirect effect   IE(V) = (a0 + a1 V) * (b0 + b1 V)
    conditional direct effect     DE(V) = c'0 + c'1 V
    conditional total effect      TE(V) = IE(V) + DE(V)

where
    Y~ = residualised log NPP SD x -1 = residualised log(1/NPP SD), i.e. stability
    D~ = residualised diversity metric, per 1 SD
    M~ = residualised WSCI, per 1 SD
    V  = moderator, standardised for fitting; plotted and reported in its own units

Because the indirect effect is a product of two moderated paths, it can change
non-linearly (quadratically) along V, and there is no single "index of moderated
mediation". The script therefore reports
  * IE(V), DE(V), TE(V) on a grid of V with percentile-bootstrap bands,
  * a contrast: IE at the end of the plotted axis minus IE at its start
    (P10 -> P90 of the moderator, in the plotted direction), with bootstrap CI and p,
  * a1 (does diversity -> WSCI change with V?), b1 (does WSCI -> stability change
    with V?) and c'1 with HC3 p-values,
  * a CHECK: ordinary (unmoderated) mediation refitted in quantile bins of V,
    shown as points; if they stray from the curve, the linear moderation is too
    simple (as for the conifer moderation of the diversity effects).

GROUPED ESTIMATES (forest type x rarity class)
  Ordinary (unmoderated) mediation refitted within each combination of a forest type
  (FOREST_CLASSES, cut on % conifer) and a rarity class (tertiles, or RARITY_CUTS),
  for each model in GP_MODELS, with the same pairs bootstrap. No linearity assumption
  along either moderator. Contrasts: high minus low rarity within each forest type, and
  broadleaf minus conifer within each rarity class (indirect effect, FDR within model).
  Plots between the FOREST_CLASSES ranges (e.g. 10-25% and 75-90% conifer) are left out.

MODELS
  "marginal": each metric on its own.
  "adjusted": the other metrics (and their interactions with V) are covariates in
              both regressions, so each metric's paths are net of the others.

INFERENCE
  Pairs bootstrap (N_BOOT resamples), the SAME resample for both regressions and
  all metrics/models, so IE, DE and TE stay consistent. Benjamini-Hochberg FDR
  across all contrast tests (moderators x models x metrics).

ASSUMPTIONS (as in the plain mediation script, plus one)
  * WSCI residualised on the same covariates as D and Y but NOT on diversity.
  * No unmeasured confounding of WSCI -> stability, at any level of V.
  * No diversity x WSCI interaction on stability.
  * The moderator is not itself caused by diversity. % rare species is a
    diversity property, so moderation by it partly reflects how the metrics are
    linked; interpret those results cautiously.

OUTPUTS (in OUTDIR)
  per moderator <key>:
    modmed_<key>_curves.csv   IE / DE / TE with CIs along the moderator
    modmed_<key>_bins.csv     binned (unmoderated) indirect effects, the check
    modmed_<key>_tests.csv    contrasts, a1 / b1 / c'1, FDR
  per model in FIG_MODELS:
    modmed_indirect_combined_<model>.png/.pdf   indirect effect along both moderators
  per model in MED_FIG_MODELS:
    mediation_summary_<model>.csv / .png / .pdf  scatter + mediated / direct / total effects
                                                 per metric (unmoderated, all plots), plus the
                                                 NET diversity effect (all metrics +1 SD, joint
                                                 model) and the % of it mediated by WSCI
  per model in GP_MODELS:
    modmed_grouped_<model>.csv                  estimates per forest type x rarity class
    modmed_grouped_contrasts_<model>.csv        contrasts of the indirect effect, FDR
    modmed_grouped_<model>.png/.pdf             grouped plot
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.colors as mcolors
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

# ===============================================================
# Input and output files
# ===============================================================
RESIDUALS_PATH = "results/step_2_residuals_merged.csv"
LOOKUP_PATH = "data/lookup/PID_location_v3.csv"          # PID -> percent_conifer
RARITY_PATH = "data/final/dataset_rarity.parquet"        # PID -> pct_rare_species
OUTDIR = "results/mediation/"

# Forest type classes on % conifer: (label, lower bound incl., upper bound excl.), left to right
FOREST_CLASSES = [
    ("Conifer-\ndominated", 90, np.inf),
    ("Mixed", 25, 75),
    ("Broadleaf-\ndominated", -np.inf, 10),
]
# Rarity classes: None = tertiles of the whole sample; or explicit cut points in %, e.g. [30, 60]
RARITY_CUTS = None
RARITY_COLORS = ["#9ECAE1", "#4292C6", "#08306B"]   # low -> high (extended automatically if needed)
GP_DODGE = 0.26                                     # spread of the rarity points around each x

_GP_QIDX = {"indirect": 0, "direct": 1, "total": 2}

# ---------------------------------------------------------------
# Column names
# ---------------------------------------------------------------
DIVERSITY_METRICS = {
    "Rao's Q": "residual_raos_q",
    "Functional Evenness": "residual_functional_evenness",
    "Species richness": "residual_species_richness",
}
STABILITY_COL = "residual_std_npp"     # Y~
MEDIATOR_COL = "residual_wsci"         # M~
MEDIATOR_NAME = "Structural complexity "  # for plots and tables
PID_COL = "PID"

# Moderators (raw, un-residualised plot values)
MODERATORS = {
    "conifer": dict(
        col="percent_conifer", label="% conifer", source="lookup",
        range=(0, 100), invert=True,                 # plotted 100 -> 0 (less conifer to the right)
        axis_label="% conifer  \u2192  less conifer",
    ),
    "pct_rare": dict(
        col="pct_rare_species", label="% rare species", source="rarity",
        multiply=100,                                # stored as a proportion (0-1); shown as %
        range=None, invert=False,                    # None = 1st-99th percentile of the data
        axis_label="% rare species  \u2192  more rare species",
    ),
}
RUN_MODERATORS = ["conifer", "pct_rare"]

# ---------------------------------------------------------------
# Settings
# ---------------------------------------------------------------
MODELS = ["marginal", "adjusted"]
FIG_MODELS = ["adjusted", "marginal"]  # one combined indirect-effect figure for each of these
STANDARDISE_TREATMENT = True           # D per 1 SD
STANDARDISE_MEDIATOR = True            # M per 1 SD
INCLUDE_INTERCEPT = False              # residuals are centred
REPORT_AS_STABILITY = True
OUTCOME_SIGN = -1 if REPORT_AS_STABILITY else 1

N_BOOT = 1000                          # lower (e.g. 200) for quick runs
CI = (2.5, 97.5)
ALPHA = 0.05
RANDOM_SEED = 42
N_GRID = 60                            # points along the moderator for the curves
CONTRAST_PCT = (10, 90)                # contrast between these percentiles of the moderator
N_BINS = 4                             # quantile bins for the check
MIN_BIN_N = 30                         # skip a bin (in a resample) with fewer rows
ZERO_A_WARN = 0.01                     # warn: a-path possibly removed by residualisation
WEAK_A_WARN = 0.05                     # note: a-path too weak for a meaningful indirect effect

# Grouped estimates (forest type x rarity class)
GP_MODELS = ["adjusted"]               # "adjusted" and/or "marginal"
GP_ROWS = ["indirect", "direct"]       # any of "indirect", "direct", "total"; one row each
GP_N_BOOT = N_BOOT
GP_MIN_N = 100                         # groups with fewer plots are not estimated

# Mediation summary figure (scatter + mediated / direct / total), one row per metric
MED_FIG_MODELS = ["adjusted"]            # "adjusted" and/or "marginal"
MED_QUANTITIES = ["Mediated", "Direct", "Total"]   # rows of the effects panel, top to bottom
MED_SCATTER_PATH = "a"                   # "a": structural complexity vs the metric (differs per row);
                                         # "b": stability vs structural complexity (the same in every
                                         #   row in the adjusted model, as b is shared)
MED_SCATTER_MAX_POINTS = 4000            # random plots drawn in each scatter
MED_SCATTER_ALPHA = 0.35
MED_SCATTER_CLIP = (0.5, 99.5)           # axis range = these percentiles of the data
MED_STAR_RULE = "fdr"                    # "fdr": FDR-corrected bootstrap p < ALPHA; "ci": CI excludes 0
# Net diversity effect: every metric +1 SD together, from the joint model with all metrics
#   a-path  M~ = sum_j a_j D~_j            outcome  Y~ = sum_j c'_j D~_j + b M~
#   net mediated = b * sum_j a_j,  net direct = sum_j c'_j,  net total = their sum,
#   proportion mediated = net mediated / net total (bootstrap CI).
SHOW_NET_EFFECT = True
NET_LABEL = "Net diversity (all metrics +1 SD)"
NET_COLOR = "#222222"

METRIC_COLORS = {
    "Rao's Q": "#4B2BC4",
    "Functional Evenness": "#A47CC7",
    "Species richness": "#14B8A0",
}
DIRECT_COLOR = "#8a8a8a"
# "split": indirect and direct effects in separate rows, each with its own y-scale, so the
#          (usually much smaller) indirect effect is readable. "combined": one row, shared axis.
PANEL_LAYOUT = "split"


# ---------------------------------------------------------------
# Data
# ---------------------------------------------------------------
def load_data():
    """Residuals plus the raw moderator columns."""
    df = pd.read_csv(RESIDUALS_PATH)
    for key in RUN_MODERATORS:
        cfg = MODERATORS[key]
        col = cfg["col"]
        if col in df.columns:
            continue
        if cfg["source"] == "lookup":
            lk = pd.read_csv(LOOKUP_PATH, usecols=[PID_COL, col])
        elif cfg["source"] == "rarity":
            lk = pd.read_parquet(RARITY_PATH, columns=[PID_COL, col])
        else:
            raise ValueError(f"Unknown source for moderator '{key}': {cfg['source']}")
        df = df.merge(lk.drop_duplicates(PID_COL), on=PID_COL, how="left")
    return df


def prepare(df, mkey):
    """Common rows for one moderator: Y, M, D (per 1 SD), raw moderator g and standardised v."""
    cfg = MODERATORS[mkey]
    cols = [STABILITY_COL, MEDIATOR_COL, cfg["col"]] + list(DIVERSITY_METRICS.values())
    sub = df[cols].dropna().reset_index(drop=True)
    print(f"\n[{mkey}] n = {len(sub):,} plots with stability, {MEDIATOR_NAME}, "
          f"{cfg['label']} and every metric")

    Y = OUTCOME_SIGN * sub[STABILITY_COL].astype(float).values
    m = sub[MEDIATOR_COL].astype(float)
    M = (m / (m.std(ddof=1) if STANDARDISE_MEDIATOR else 1.0)).values
    D = pd.DataFrame(index=sub.index)
    for name, col in DIVERSITY_METRICS.items():
        d = sub[col].astype(float)
        D[name] = d / (d.std(ddof=1) if STANDARDISE_TREATMENT else 1.0)

    g = sub[cfg["col"]].astype(float).values * cfg.get("multiply", 1)
    g_mean, g_sd = g.mean(), g.std(ddof=1)
    v = (g - g_mean) / g_sd

    corr = {k: np.corrcoef(D[k].values, M)[0, 1] for k in D.columns}
    print(f"[{mkey}] corr(D~, {MEDIATOR_NAME}~): "
          + ", ".join(f"{k} {c:+.3f}" for k, c in corr.items()))
    weak = [k for k, c in corr.items() if abs(c) < WEAK_A_WARN]
    if weak:
        print(f"  note: weak diversity -> {MEDIATOR_NAME} link (|r| < {WEAK_A_WARN}) for "
              f"{', '.join(weak)}: the indirect effect a*b will be near 0 whatever b is.")
    if all(abs(c) < ZERO_A_WARN for c in corr.values()):
        print(f"  WARNING: {MEDIATOR_NAME} residuals are uncorrelated with every metric; check "
              f"that {MEDIATOR_COL} was NOT residualised on diversity.")
    return dict(Y=Y, M=M, D=D, g=g, v=v, g_mean=g_mean, g_sd=g_sd, n=len(sub))


# ---------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------
def _stack(*cols):
    out = []
    for c in cols:
        if c is None:
            continue
        c = np.asarray(c, dtype=float)
        out.append(c[:, None] if c.ndim == 1 else c)
    return np.hstack(out)


def _ols(y, X):
    if INCLUDE_INTERCEPT:
        X = np.hstack([np.ones((len(y), 1)), X])
    try:
        beta = np.linalg.solve(X.T @ X, X.T @ y)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return beta[1:] if INCLUDE_INTERCEPT else beta


def _controls(Dv, j, model):
    """Other metrics as covariates (adjusted model), else None. Dv is an array."""
    if model == "adjusted" and Dv.shape[1] > 1:
        return np.delete(Dv, j, axis=1)
    return None


def fit_moderated(d, M, Y, v, C=None):
    """Returns [a0, a1, b0, b1, cp0, cp1]."""
    Cv = None if C is None else C * v[:, None]
    ba = _ols(M, _stack(d, d * v, v, C, Cv))
    by = _ols(Y, _stack(d, d * v, M, M * v, v, C, Cv))
    return np.array([ba[0], ba[1], by[2], by[3], by[0], by[1]])


def fit_joint(Dv, M, Y):
    """Net effect of all metrics +1 SD together: (net mediated b*sum(a), net direct sum(c'))."""
    a = _ols(M, Dv)
    by = _ols(Y, _stack(Dv, M))
    return by[-1] * a.sum(), by[:Dv.shape[1]].sum()


def fit_plain(d, M, Y, C=None):
    """Unmoderated mediation: (indirect a*b, direct c')."""
    a = _ols(M, _stack(d, C))[0]
    by = _ols(Y, _stack(d, M, C))
    return a * by[1], by[0]


def conditional(p, vgrid):
    """IE, DE, TE along vgrid; p has shape (..., 6)."""
    p = np.asarray(p, dtype=float)
    a0, a1, b0, b1, cp0, cp1 = [p[..., k][..., None] for k in range(6)]
    ie = (a0 + a1 * vgrid) * (b0 + b1 * vgrid)
    de = cp0 + cp1 * vgrid
    return ie, de, ie + de


def hc3_moderation(d, M, Y, v, C=None):
    """HC3 tests of the moderation terms a1 (D x V on M), b1 (M x V on Y) and c'1 (D x V on Y)."""
    ctrl, ctrl_v = {}, {}
    if C is not None:
        for j in range(C.shape[1]):
            ctrl[f"ctrl_{j}"] = C[:, j]
            ctrl_v[f"ctrl_{j}_xV"] = C[:, j] * v
    Xa = pd.DataFrame({"D": d, "DxV": d * v, "V": v, **ctrl, **ctrl_v})
    Xy = pd.DataFrame({"D": d, "DxV": d * v, "M": M, "MxV": M * v, "V": v, **ctrl, **ctrl_v})
    if INCLUDE_INTERCEPT:
        Xa, Xy = sm.add_constant(Xa, has_constant="add"), sm.add_constant(Xy, has_constant="add")
    ra = sm.OLS(M, Xa).fit(cov_type="HC3")
    ry = sm.OLS(Y, Xy).fit(cov_type="HC3")
    return {"a1": ra.params["DxV"], "a1_p": ra.pvalues["DxV"],
            "b1": ry.params["MxV"], "b1_p": ry.pvalues["MxV"],
            "cp1": ry.params["DxV"], "cp1_p": ry.pvalues["DxV"]}


def _boot_p(x):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    return float(min(1.0, 2 * min(np.mean(x <= 0), np.mean(x >= 0)))) if len(x) else np.nan


# ---------------------------------------------------------------
# One moderator
# ---------------------------------------------------------------
def run_moderator(df, mkey, n_boot=N_BOOT):
    cfg = MODERATORS[mkey]
    data = prepare(df, mkey)
    Y, M, D, g, v, n = data["Y"], data["M"], data["D"], data["g"], data["v"], data["n"]
    g_mean, g_sd = data["g_mean"], data["g_sd"]
    metrics = list(D.columns)
    Dv = D.values

    # Grid and contrast points in the moderator's own units
    lo, hi = cfg["range"] if cfg.get("range") else np.percentile(g, [1, 99])
    grid = np.linspace(max(lo, g.min()), min(hi, g.max()), N_GRID)
    vgrid = (grid - g_mean) / g_sd
    p_lo, p_hi = np.percentile(g, CONTRAST_PCT)
    start, end = (p_hi, p_lo) if cfg.get("invert") else (p_lo, p_hi)   # plotted direction
    v_start, v_end = (start - g_mean) / g_sd, (end - g_mean) / g_sd

    # Check bins
    bins = pd.qcut(g, N_BINS, labels=False, duplicates="drop")
    bins = np.asarray(bins)
    n_bins = int(bins.max()) + 1
    bin_x = np.array([g[bins == k].mean() for k in range(n_bins)])
    if n_bins < N_BINS:
        print(f"  note: many tied {cfg['label']} values; {n_bins} bins instead of {N_BINS}")

    keys = [(model, j, metric) for model in MODELS for j, metric in enumerate(metrics)]

    # Point estimates
    point, point_bins, tests = {}, {}, []
    for model, j, metric in keys:
        C = _controls(Dv, j, model)
        d = Dv[:, j]
        point[(model, metric)] = fit_moderated(d, M, Y, v, C)
        point_bins[(model, metric)] = np.array([
            fit_plain(d[bins == k], M[bins == k], Y[bins == k],
                      None if C is None else C[bins == k])[0] for k in range(n_bins)])
        a0, a1, b0, b1, cp0, cp1 = point[(model, metric)]
        tests.append({"moderator": cfg["label"], "model": model, "diversity_metric": metric, "n": n,
                      "a0": a0, "b0": b0, "cp0": cp0,          # paths at the moderator mean
                      **hc3_moderation(d, M, Y, v, C)})

    # Bootstrap
    print(f"[{mkey}] bootstrap: {n_boot} resamples x {len(metrics)} metrics x {len(MODELS)} models ...")
    rng = np.random.default_rng(RANDOM_SEED)
    draws = {k[::2]: np.empty((n_boot, 6)) for k in keys}
    bin_draws = {k[::2]: np.full((n_boot, n_bins), np.nan) for k in keys}
    for i in range(n_boot):
        if (i + 1) % max(1, n_boot // 10) == 0:
            print(f"  {i + 1}/{n_boot}")
        idx = rng.integers(0, n, n)
        Yb, Mb, vb, Db, bb = Y[idx], M[idx], v[idx], Dv[idx], bins[idx]
        for model, j, metric in keys:
            Cb = _controls(Db, j, model)
            d = Db[:, j]
            draws[(model, metric)][i] = fit_moderated(d, Mb, Yb, vb, Cb)
            for k in range(n_bins):
                mk = bb == k
                if mk.sum() >= MIN_BIN_N:
                    bin_draws[(model, metric)][i, k] = fit_plain(
                        d[mk], Mb[mk], Yb[mk], None if Cb is None else Cb[mk])[0]

    # Curves
    curve_rows, bin_rows = [], []
    for model, j, metric in keys:
        key = (model, metric)
        ie, de, te = conditional(point[key], vgrid)          # each shape (N_GRID,)
        ie_b, de_b, te_b = conditional(draws[key], vgrid)
        q = {name: np.nanpercentile(x, CI, axis=0) for name, x in
             [("indirect", ie_b), ("direct", de_b), ("total", te_b)]}
        for gi in range(N_GRID):
            curve_rows.append({
                "model": model, "diversity_metric": metric, "moderator_value": grid[gi],
                "indirect": ie[gi], "indirect_ci_lower": q["indirect"][0, gi],
                "indirect_ci_upper": q["indirect"][1, gi],
                "direct": de[gi], "direct_ci_lower": q["direct"][0, gi],
                "direct_ci_upper": q["direct"][1, gi],
                "total": te[gi], "total_ci_lower": q["total"][0, gi],
                "total_ci_upper": q["total"][1, gi],
            })
        bq = np.nanpercentile(bin_draws[key], CI, axis=0)
        for k in range(n_bins):
            bin_rows.append({"model": model, "diversity_metric": metric, "bin": k + 1,
                             "n": int((bins == k).sum()), "moderator_mean": bin_x[k],
                             "indirect": point_bins[key][k],
                             "indirect_ci_lower": bq[0, k], "indirect_ci_upper": bq[1, k]})

    # Contrasts (plotted direction: end minus start)
    for t in tests:
        key = (t["model"], t["diversity_metric"])
        ie_pt = conditional(point[key], np.array([v_start, v_end]))[0]   # shape (2,)
        ie_bt = conditional(draws[key], np.array([v_start, v_end]))[0]
        diff_b = ie_bt[:, 1] - ie_bt[:, 0]
        t.update({
            "contrast_from": start, "contrast_to": end,
            "indirect_at_from": ie_pt[0], "indirect_at_to": ie_pt[1],
            "indirect_contrast": ie_pt[1] - ie_pt[0],
            "indirect_contrast_ci_lower": np.nanpercentile(diff_b, CI[0]),
            "indirect_contrast_ci_upper": np.nanpercentile(diff_b, CI[1]),
            "p_value_raw": _boot_p(diff_b),
        })

    return (pd.DataFrame(curve_rows), pd.DataFrame(bin_rows), pd.DataFrame(tests), g)


# ---------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------
def _scale_for(values):
    """Power-of-ten scale so tick labels stay short."""
    vmax = np.nanmax(np.abs(np.asarray(values, dtype=float)))
    exp = int(np.floor(np.log10(vmax))) if np.isfinite(vmax) and vmax > 0 else 0
    return exp, 10.0 ** exp


def _add_hist(ax, g, x_lo, x_hi, color):
    counts, edges = np.histogram(g[(g >= x_lo) & (g <= x_hi)], bins=np.linspace(x_lo, x_hi, 26))
    hax = ax.inset_axes([0, 0, 1, 0.13], sharex=ax)
    hax.bar(edges[:-1], counts, width=np.diff(edges), align="edge", color=color, alpha=0.35,
            linewidth=0)
    hax.set_ylim(0, max(counts.max(), 1) * 1.05)
    hax.set_axis_off()
    hax.patch.set_alpha(0)


def _style(ax):
    ax.grid(alpha=0.25, lw=0.5)
    ax.axhline(0, color="#1a1a1a", lw=0.8, ls=":", zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


# ---------------------------------------------------------------
# Grouped estimates: forest type x rarity class
# ---------------------------------------------------------------
def gp_prepare(df):
    cc, rc = MODERATORS["conifer"], MODERATORS["pct_rare"]
    missing = [c for c in (cc["col"], rc["col"]) if c not in df.columns]
    if missing:
        raise KeyError(f"{missing} not in the data: add both 'conifer' and 'pct_rare' to RUN_MODERATORS.")
    cols = [STABILITY_COL, MEDIATOR_COL, cc["col"], rc["col"]] + list(DIVERSITY_METRICS.values())
    sub = df[cols].dropna().reset_index(drop=True)

    con = sub[cc["col"]].to_numpy(float) * cc.get("multiply", 1)
    rar = sub[rc["col"]].to_numpy(float) * rc.get("multiply", 1)
    if np.nanmax(con) <= 1:
        con = con * 100
    if np.nanmax(rar) <= 1:
        rar = rar * 100

    Y = OUTCOME_SIGN * sub[STABILITY_COL].to_numpy(float)
    m = sub[MEDIATOR_COL].astype(float)
    M = (m / (m.std(ddof=1) if STANDARDISE_MEDIATOR else 1.0)).to_numpy()
    D = pd.DataFrame(index=sub.index)
    for name, col in DIVERSITY_METRICS.items():
        d = sub[col].astype(float)
        D[name] = d / (d.std(ddof=1) if STANDARDISE_TREATMENT else 1.0)

    fclass = np.full(len(sub), -1)
    for k, (_, lo, hi) in enumerate(FOREST_CLASSES):
        fclass[(con >= lo) & (con < hi)] = k

    cuts = list(np.percentile(rar, [100 / 3, 200 / 3])) if RARITY_CUTS is None else list(RARITY_CUTS)
    rclass = np.digitize(rar, cuts)
    names = ["Low", "Medium", "High"] if len(cuts) == 2 else [f"Class {k + 1}" for k in range(len(cuts) + 1)]
    rlabels = []
    for k, nm in enumerate(names):
        if k == 0:
            rlabels.append(f"{nm} rarity (< {cuts[0]:.0f}%)")
        elif k == len(cuts):
            rlabels.append(f"{nm} rarity (\u2265 {cuts[-1]:.0f}%)")
        else:
            rlabels.append(f"{nm} rarity ({cuts[k - 1]:.0f}\u2013{cuts[k]:.0f}%)")

    print(f"\n[grouped] n = {len(sub):,} plots ({int((fclass >= 0).sum()):,} in a forest class); "
          f"rarity cut points: " + ", ".join(f"{c:.1f}%" for c in cuts))
    return dict(Y=Y, M=M, D=D, fclass=fclass, rclass=rclass, rlabels=rlabels)


def gp_estimate(Y, M, Dv, fclass, rclass, model, n_f, n_r):
    """Array (metrics, forest classes, rarity classes, 3) of indirect, direct, total."""
    p = Dv.shape[1]
    out = np.full((p, n_f, n_r, 3), np.nan)
    for f in range(n_f):
        for r in range(n_r):
            mk = (fclass == f) & (rclass == r)
            if mk.sum() < GP_MIN_N:
                continue
            for j in range(p):
                C = _controls(Dv[mk], j, model)
                ie, de = fit_plain(Dv[mk, j], M[mk], Y[mk], C)
                out[j, f, r] = ie, de, ie + de
    return out


def run_grouped(df, model):
    d = gp_prepare(df)
    Y, M, Dv = d["Y"], d["M"], d["D"].values
    fclass, rclass = d["fclass"], d["rclass"]
    metrics = list(d["D"].columns)
    n_f, n_r = len(FOREST_CLASSES), len(d["rlabels"])
    counts = np.array([[int(((fclass == f) & (rclass == r)).sum()) for r in range(n_r)]
                       for f in range(n_f)])

    point = gp_estimate(Y, M, Dv, fclass, rclass, model, n_f, n_r)
    print(f"[grouped] bootstrap ({model}): {GP_N_BOOT} resamples ...")
    rng = np.random.default_rng(RANDOM_SEED)
    n = len(Y)
    draws = np.full((GP_N_BOOT,) + point.shape, np.nan)
    for i in range(GP_N_BOOT):
        if (i + 1) % max(1, GP_N_BOOT // 10) == 0:
            print(f"  {i + 1}/{GP_N_BOOT}")
        idx = rng.integers(0, n, n)
        draws[i] = gp_estimate(Y[idx], M[idx], Dv[idx], fclass[idx], rclass[idx], model, n_f, n_r)
    lo, hi = np.nanpercentile(draws, CI, axis=0)

    # Long table of group estimates
    rows = []
    for j, m in enumerate(metrics):
        for f in range(n_f):
            for r in range(n_r):
                for q, qi in _GP_QIDX.items():
                    rows.append({"model": model, "diversity_metric": m,
                                 "forest_type": FOREST_CLASSES[f][0].replace("\n", ""),
                                 "rarity_class": d["rlabels"][r], "n": counts[f, r], "quantity": q,
                                 "estimate": point[j, f, r, qi],
                                 "ci_lower": lo[j, f, r, qi], "ci_upper": hi[j, f, r, qi],
                                 "p_boot": _boot_p(draws[:, j, f, r, qi])})
    groups = pd.DataFrame(rows)

    # Contrasts of the indirect effect
    crow = []
    ii = _GP_QIDX["indirect"]
    for j, m in enumerate(metrics):
        for f in range(n_f):
            diff = draws[:, j, f, -1, ii] - draws[:, j, f, 0, ii]
            crow.append({"model": model, "diversity_metric": m,
                         "contrast": f"{FOREST_CLASSES[f][0].replace(chr(10), '')}: "
                                     f"{d['rlabels'][-1]} minus {d['rlabels'][0]}",
                         "estimate": point[j, f, -1, ii] - point[j, f, 0, ii],
                         "ci_lower": np.nanpercentile(diff, CI[0]),
                         "ci_upper": np.nanpercentile(diff, CI[1]), "p_raw": _boot_p(diff)})
        for r in range(n_r):
            diff = draws[:, j, -1, r, ii] - draws[:, j, 0, r, ii]
            crow.append({"model": model, "diversity_metric": m,
                         "contrast": f"{d['rlabels'][r]}: "
                                     f"{FOREST_CLASSES[-1][0].replace(chr(10), '')} minus "
                                     f"{FOREST_CLASSES[0][0].replace(chr(10), '')}",
                         "estimate": point[j, -1, r, ii] - point[j, 0, r, ii],
                         "ci_lower": np.nanpercentile(diff, CI[0]),
                         "ci_upper": np.nanpercentile(diff, CI[1]), "p_raw": _boot_p(diff)})
    contrasts = pd.DataFrame(crow)
    ok = contrasts["p_raw"].notna()
    contrasts["p_fdr"], contrasts["significant_fdr"] = np.nan, False
    if ok.any():
        rej, padj, _, _ = multipletests(contrasts.loc[ok, "p_raw"], alpha=ALPHA, method="fdr_bh")
        contrasts.loc[ok, "p_fdr"], contrasts.loc[ok, "significant_fdr"] = padj, rej
    return groups, contrasts, metrics, d["rlabels"], counts


# ---------------------------------------------------------------
# Combined indirect-effect figure (both moderators)
# ---------------------------------------------------------------
def plot_indirect_combined(results, all_tests, model, savepath, mkeys=None):
    """
    Indirect effect via the mediator only. Rows = moderators (% conifer, % rare species),
    columns = diversity metrics. One shared y-scale across all panels.
    """
    mkeys = mkeys or [k for k in ("conifer", "pct_rare") if k in results]
    metrics = list(DIVERSITY_METRICS)
    ind_cols = ["indirect", "indirect_ci_lower", "indirect_ci_upper"]

    # Shared power-of-ten scale over every curve, band and check point
    vals = []
    for k in mkeys:
        curves, bins_df, _, _ = results[k]
        vals.append(curves.loc[curves.model == model, ind_cols].to_numpy().ravel())
        vals.append(bins_df.loc[bins_df.model == model, ind_cols].to_numpy().ravel())
    exp, sc = _scale_for(np.concatenate(vals))
    unit = f" (\u00d710$^{{{exp}}}$)" if exp else ""

    nrows, ncols = len(mkeys), len(metrics)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.0 * nrows + 1.0),
                             sharey=True, squeeze=False)
    xinfo = {}
    for r, k in enumerate(mkeys):
        cfg = MODERATORS[k]
        curves, bins_df, _, g = results[k]
        tests = all_tests[(all_tests["_key"] == k) & (all_tests.model == model)]
        x_lo, x_hi = cfg["range"] if cfg.get("range") else np.percentile(g, [1, 99])
        xinfo[k] = (x_lo, x_hi, g)
        pct = "%" if cfg["label"].startswith("%") else ""

        for j, m in enumerate(metrics):
            ax = axes[r, j]
            col = METRIC_COLORS.get(m, "navy")
            cm = curves[(curves.model == model) & (curves.diversity_metric == m)]
            bm = bins_df[(bins_df.model == model) & (bins_df.diversity_metric == m)]
            tm = tests[tests.diversity_metric == m].iloc[0]

            ax.fill_between(cm.moderator_value, cm.indirect_ci_lower / sc,
                            cm.indirect_ci_upper / sc, color=col, alpha=0.25, linewidth=0)
            ax.plot(cm.moderator_value, cm.indirect / sc, color=col, lw=2.0)
            ax.errorbar(bm.moderator_mean, bm.indirect / sc,
                        yerr=[(bm.indirect - bm.indirect_ci_lower) / sc,
                              (bm.indirect_ci_upper - bm.indirect) / sc],
                        fmt="o", ms=4.5, mfc="white", mec=col, mew=1.2, ecolor=col,
                        elinewidth=0.9, capsize=2.5, zorder=4, clip_on=False)
            _style(ax)

            pad = 0.05 * (x_hi - x_lo)
            ax.set_xlim(x_lo - pad, x_hi + pad)
            if cfg.get("invert"):
                ax.invert_xaxis()
            ax.set_xlabel(cfg["axis_label"], fontsize=8)
            ax.tick_params(labelsize=7.5)

            star = " *" if tm["significant_fdr"] else ""
            ax.text(0.03, 0.97,
                    f"\u0394 indirect, {tm.contrast_from:.3g}\u2192{tm.contrast_to:.3g}{pct}: "
                    f"{tm.indirect_contrast / sc:+.2g}{star}\n"
                    f"[{tm.indirect_contrast_ci_lower / sc:+.2g}, "
                    f"{tm.indirect_contrast_ci_upper / sc:+.2g}]   "
                    f"p(a\u2081) = {tm.a1_p:.2g}, p(b\u2081) = {tm.b1_p:.2g}",
                    transform=ax.transAxes, va="top", ha="left", fontsize=6.5,
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85,
                              edgecolor="none"), zorder=5)
            if r == 0:
                ax.set_title(m, fontsize=10, fontweight="bold", color=col)
            if j == 0:
                ax.set_ylabel(f"Along {cfg['label']}\nIndirect via {MEDIATOR_NAME}{unit}",
                              fontsize=8.5)

    # Room at the bottom for the plot-distribution histograms (shared y, so set once)
    y_lo, y_hi = axes[0, 0].get_ylim()
    axes[0, 0].set_ylim(y_lo - 0.20 * (y_hi - y_lo), y_hi + 0.15 * (y_hi - y_lo))
    for r, k in enumerate(mkeys):
        x_lo, x_hi, g = xinfo[k]
        for j, m in enumerate(metrics):
            _add_hist(axes[r, j], g, x_lo, x_hi, METRIC_COLORS.get(m, "navy"))

    target = "stability" if REPORT_AS_STABILITY else "NPP SD"
    per = "per 1 SD of diversity" if STANDARDISE_TREATMENT else ""
    fig.supylabel(f"Indirect effect on {target} {per}", fontsize=9)
    bin_name = {3: "tertiles", 4: "quartiles", 5: "quintiles"}.get(
        int(results[mkeys[0]][1]["bin"].max()), "bins")
    handles = [Line2D([], [], color="#555555", lw=2.0, label=f"Indirect via {MEDIATOR_NAME} (metric colour)"),
               Line2D([], [], marker="o", ls="none", mfc="white", mec="#555555",
                      label=f"Indirect in {bin_name} (check)"),
               Patch(color="#555555", alpha=0.25, label=f"{CI[1] - CI[0]:.0f}% bootstrap band"),
               Patch(color="#555555", alpha=0.35, label="Distribution of plots")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=4,
               fontsize=7.5, frameon=False)
    model_txt = "adjusted for the other metrics" if model == "adjusted" else "each metric on its own"
    fig.suptitle(f"Indirect effect of diversity on {target} via {MEDIATOR_NAME} "
                 f"along % conifer and % rare species ({model_txt})\n"
                 f"* = change in indirect effect significant; all panels share one y-scale",
                 fontsize=10, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(savepath, dpi=300, bbox_inches="tight")
    fig.savefig(savepath.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"[figure] saved {savepath}")
    return fig


# ---------------------------------------------------------------
# Grouped figure
# ---------------------------------------------------------------
def _rarity_colors(n):
    """RARITY_COLORS, extended along the same light -> dark ramp if more classes are needed."""
    if n <= len(RARITY_COLORS):
        return RARITY_COLORS[:n]
    cmap = mcolors.LinearSegmentedColormap.from_list("rar", RARITY_COLORS)
    return [mcolors.to_hex(cmap(t)) for t in np.linspace(0, 1, n)]


def plot_grouped(groups, contrasts, metrics, rlabels, counts, model, savepath):
    """
    Group estimates: rows = GP_ROWS (indirect / direct / total), columns = diversity metrics,
    x = forest type (left to right as in FOREST_CLASSES), one point per rarity class (dodged,
    light -> dark = low -> high rarity) with its bootstrap CI.
    Filled point = CI excludes 0; open point = CI includes 0. Groups with < GP_MIN_N plots
    are not estimated (no point). In the indirect row, * above a forest type = the
    high-minus-low-rarity contrast of the indirect effect is significant after FDR.
    Each row has its own y-scale.
    """
    rows = [q for q in GP_ROWS if q in _GP_QIDX]
    n_f, n_r = len(FOREST_CLASSES), len(rlabels)
    colors = _rarity_colors(n_r)
    offsets = (np.arange(n_r) - (n_r - 1) / 2) * GP_DODGE
    forest_names = [lab.replace("\n", "") for lab, _, _ in FOREST_CLASSES]

    fig, axes = plt.subplots(len(rows), len(metrics),
                             figsize=(3.4 * len(metrics), 2.7 * len(rows) + 1.3),
                             sharex=True, sharey="row", squeeze=False)
    g_mod = groups[groups.model == model]
    c_mod = contrasts[contrasts.model == model]

    for ri, q in enumerate(rows):
        gq = g_mod[g_mod.quantity == q]
        exp, sc = _scale_for(gq[["estimate", "ci_lower", "ci_upper"]].to_numpy().ravel())
        unit = f" (\u00d710$^{{{exp}}}$)" if exp else ""
        for j, m in enumerate(metrics):
            ax = axes[ri, j]
            gm = gq[gq.diversity_metric == m]
            for r, lab in enumerate(rlabels):
                gr = (gm[gm.rarity_class == lab].set_index("forest_type")
                      .reindex(forest_names))                     # forest order, NaN if missing
                x = np.arange(n_f) + offsets[r]
                est = gr["estimate"].to_numpy(float) / sc
                lo = gr["ci_lower"].to_numpy(float) / sc
                hi = gr["ci_upper"].to_numpy(float) / sc
                sig = (lo > 0) | (hi < 0)
                ax.errorbar(x, est, yerr=[est - lo, hi - est], fmt="none", ecolor=colors[r],
                            elinewidth=1.0, capsize=2.5, zorder=3)
                ax.scatter(x[sig], est[sig], s=28, color=colors[r], edgecolor=colors[r],
                           linewidth=1.1, zorder=4)
                ax.scatter(x[~sig], est[~sig], s=28, facecolor="white", edgecolor=colors[r],
                           linewidth=1.1, zorder=4)
            _style(ax)
            ax.grid(axis="x", visible=False)
            for xb in np.arange(n_f - 1) + 0.5:                  # separate the forest types
                ax.axvline(xb, color="#d9d9d9", lw=0.6, zorder=0)

            if q == "indirect":                                  # stars for the rarity contrast
                cm = c_mod[c_mod.diversity_metric == m]
                for f, fname in enumerate(forest_names):
                    hit = cm[cm.contrast.str.startswith(f"{fname}:")]
                    if len(hit) and bool(hit["significant_fdr"].iloc[0]):
                        ax.text(f, 0.97, "*", transform=ax.get_xaxis_transform(), ha="center",
                                va="top", fontsize=13, fontweight="bold")
            if ri == 0:
                ax.set_title(m, fontsize=10, fontweight="bold", color=METRIC_COLORS.get(m, "black"))
            if j == 0:
                ax.set_ylabel(f"{q.capitalize()}{' via ' + MEDIATOR_NAME.strip() if q == 'indirect' else ''}"
                              f"{unit}", fontsize=8.5)

    ticklabels = [f"{FOREST_CLASSES[f][0]}\n(n = {counts[f].sum():,})" for f in range(n_f)]
    for ax in axes[-1]:
        ax.set_xticks(np.arange(n_f))
        ax.set_xticklabels(ticklabels, fontsize=7.5)
        ax.set_xlim(-0.5, n_f - 0.5)
    for ax in axes.ravel():
        ax.tick_params(axis="y", labelsize=7.5)

    target = "stability" if REPORT_AS_STABILITY else "NPP SD"
    per = "per 1 SD of diversity" if STANDARDISE_TREATMENT else ""
    fig.supylabel(f"Effect on {target} {per}", fontsize=9)
    handles = [Line2D([], [], marker="o", ls="none", color=colors[r], markersize=6, label=lab)
               for r, lab in enumerate(rlabels)]
    handles += [Line2D([], [], marker="o", ls="none", color="#555555", markersize=6,
                       label=f"{CI[1] - CI[0]:.0f}% CI excludes 0"),
                Line2D([], [], marker="o", ls="none", mfc="white", mec="#555555", markersize=6,
                       label="CI includes 0")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0),
               ncol=min(len(handles), 5), fontsize=7.5, frameon=False)
    model_txt = "adjusted for the other metrics" if model == "adjusted" else "each metric on its own"
    fig.suptitle(f"Mediation via {MEDIATOR_NAME.strip()} by forest type and rarity ({model_txt})\n"
                 f"* = indirect effect differs between high and low rarity (FDR); "
                 f"rows have their own y-scale", fontsize=10, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(savepath, dpi=300, bbox_inches="tight")
    fig.savefig(savepath.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"[figure] saved {savepath}")
    return fig


# ---------------------------------------------------------------
# Mediation summary figure: scatter + mediated / direct / total effects
# ---------------------------------------------------------------
def med_prepare(df):
    """Y, M, D (per 1 SD) on the rows with stability, the mediator and every metric."""
    cols = [STABILITY_COL, MEDIATOR_COL] + list(DIVERSITY_METRICS.values())
    sub = df[cols].dropna().reset_index(drop=True)
    Y = OUTCOME_SIGN * sub[STABILITY_COL].to_numpy(float)
    m = sub[MEDIATOR_COL].astype(float)
    M = (m / (m.std(ddof=1) if STANDARDISE_MEDIATOR else 1.0)).to_numpy()
    D = pd.DataFrame(index=sub.index)
    for name, col in DIVERSITY_METRICS.items():
        d = sub[col].astype(float)
        D[name] = d / (d.std(ddof=1) if STANDARDISE_TREATMENT else 1.0)
    print(f"\n[mediation summary] n = {len(sub):,} plots")
    return Y, M, D


def run_mediation_summary(df, model, n_boot=N_BOOT):
    """
    Ordinary (unmoderated) mediation on all plots, per metric: mediated (a*b), direct (c')
    and total effects with pairs-bootstrap CIs and p. Returns (table, Y, M, D).
    """
    Y, M, D = med_prepare(df)
    Dv, metrics, n = D.values, list(D.columns), len(Y)
    point = np.array([fit_plain(Dv[:, j], M, Y, _controls(Dv, j, model)) for j in range(len(metrics))])
    if SHOW_NET_EFFECT:                                   # net effect: joint model, all metrics
        point = np.vstack([point, fit_joint(Dv, M, Y)])
    names = metrics + ([NET_LABEL] if SHOW_NET_EFFECT else [])
    rng = np.random.default_rng(RANDOM_SEED)
    draws = np.empty((n_boot, len(names), 2))
    print(f"[mediation summary] bootstrap ({model}): {n_boot} resamples ...")
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        Db, Mb, Yb = Dv[idx], M[idx], Y[idx]
        for j in range(len(metrics)):
            draws[i, j] = fit_plain(Db[:, j], Mb, Yb, _controls(Db, j, model))
        if SHOW_NET_EFFECT:
            draws[i, -1] = fit_joint(Db, Mb, Yb)
    rows = []
    for j, m in enumerate(names):
        est = {"Mediated": point[j, 0], "Direct": point[j, 1], "Total": point[j].sum()}
        bd = {"Mediated": draws[:, j, 0], "Direct": draws[:, j, 1], "Total": draws[:, j].sum(axis=1)}
        for q in MED_QUANTITIES:
            lo, hi = np.percentile(bd[q], CI)
            rows.append({"model": "joint (all metrics)" if m == NET_LABEL else model,
                         "diversity_metric": m, "effect": q, "estimate": est[q],
                         "ci_lower": lo, "ci_upper": hi, "p_boot": _boot_p(bd[q]), "n": n})
    tab = pd.DataFrame(rows)
    # Proportion of the net effect that runs through the mediator (only meaningful when the
    # mediated and total effects have the same sign; the CI shows how well it is pinned down)
    if SHOW_NET_EFFECT:
        with np.errstate(invalid="ignore", divide="ignore"):
            prop_b = 100 * draws[:, -1, 0] / draws[:, -1].sum(axis=1)
        prop = 100 * point[-1, 0] / point[-1].sum()
        lo, hi = np.nanpercentile(prop_b, CI)
        tab.attrs["net_prop_mediated"] = (prop, lo, hi)
        print(f"[mediation summary] net effect: {prop:.1f}% mediated by {MEDIATOR_NAME.strip()} "
              f"[{lo:.1f}, {hi:.1f}]")
    rej, padj, _, _ = multipletests(tab["p_boot"].fillna(1.0), alpha=ALPHA, method="fdr_bh")
    tab["p_fdr"], tab["significant_fdr"] = padj, rej
    tab["ci_excludes_0"] = (tab["ci_lower"] > 0) | (tab["ci_upper"] < 0)
    if SHOW_NET_EFFECT:
        prop, lo, hi = tab.attrs["net_prop_mediated"]
        tab = pd.concat([tab, pd.DataFrame([{
            "model": "joint (all metrics)", "diversity_metric": NET_LABEL,
            "effect": "Proportion mediated (%)", "estimate": prop, "ci_lower": lo, "ci_upper": hi,
            "n": n}])], ignore_index=True)
        tab.attrs["net_prop_mediated"] = (prop, lo, hi)
    return tab, Y, M, D


def _partial(x, y, Z):
    """Frisch-Waugh partial residuals of x and y on Z (no intercept unless INCLUDE_INTERCEPT);
    the slope of y on x then equals the coefficient on x in y ~ x + Z."""
    if Z is None:
        return x, y
    Z = np.asarray(Z, dtype=float)
    Z = Z[:, None] if Z.ndim == 1 else Z
    if INCLUDE_INTERCEPT:
        Z = np.hstack([np.ones((len(x), 1)), Z])
    bx = np.linalg.lstsq(Z, x, rcond=None)[0]
    by = np.linalg.lstsq(Z, y, rcond=None)[0]
    return x - Z @ bx, y - Z @ by


def plot_mediation_summary(tab, Y, M, D, model, savepath):
    """
    One row per diversity metric:
      left : scatter of the path in MED_SCATTER_PATH with its OLS line
             "a": mediator vs the metric (does diversity raise structural complexity?)
             "b": stability vs the mediator, controlling for the metric
             (adjusted model: both also partial out the other metrics, so the line's slope
              is exactly the path coefficient used in the indirect effect)
      right: mediated, direct and total effects with bootstrap CIs; star = significant.
    """
    metrics = list(D.columns)
    Dv = D.values
    rows_ = metrics + ([NET_LABEL] if SHOW_NET_EFFECT and NET_LABEL in set(tab.diversity_metric) else [])
    n = len(rows_)
    eff_tab = tab[tab.effect.isin(MED_QUANTITIES)]
    exp, sc = _scale_for(eff_tab[["estimate", "ci_lower", "ci_upper"]].to_numpy().ravel())
    rng = np.random.default_rng(RANDOM_SEED)
    med = MEDIATOR_NAME.strip()

    fig, axes = plt.subplots(n, 2, figsize=(7.2, 2.55 * n + 0.4), squeeze=False,
                             gridspec_kw=dict(width_ratios=[1.05, 1], wspace=0.45, hspace=0.6))
    for j, m in enumerate(rows_):
        is_net = m == NET_LABEL
        col = NET_COLOR if is_net else METRIC_COLORS.get(m, "#2b6a2b")
        C = None if is_net else _controls(Dv, j, model)
        d = None if is_net else Dv[:, j]
        if is_net:
            # the shared b-path: stability vs the mediator with ALL metrics partialled out;
            # its slope b times sum(a_j) is the net mediated effect
            x, y = _partial(M, Y, Dv)
            xlab = f"{med} (residual, SD)  [partial]"
            ylab = "Stability residual\n(+ = more stable)"
        elif MED_SCATTER_PATH == "a":
            x, y = _partial(d, M, C)
            xlab = f"{m} (residual, SD)"
            ylab = f"{med}\n(residual, SD)"
        else:
            Z = d if C is None else np.column_stack([d, C])
            x, y = _partial(M, Y, Z)
            xlab = f"{med} (residual, SD)"
            ylab = "Stability residual\n(+ = more stable)"
        if not is_net and (model == "adjusted" or MED_SCATTER_PATH == "b"):
            xlab += "  [partial]"

        # Scatter: a random subsample of plots, axes clipped to the central data
        ax = axes[j, 0]
        idx = rng.choice(len(x), size=min(MED_SCATTER_MAX_POINTS, len(x)), replace=False)
        ax.scatter(x[idx], y[idx], s=4, color=col, alpha=MED_SCATTER_ALPHA, linewidths=0,
                   rasterized=True)
        slope = float((x * y).sum() / (x * x).sum()) if not INCLUDE_INTERCEPT else np.polyfit(x, y, 1)[0]
        x0, x1 = np.percentile(x, MED_SCATTER_CLIP)
        y0, y1 = np.percentile(y, MED_SCATTER_CLIP)
        ax.plot([x0, x1], [slope * x0, slope * x1], color=col, lw=1.6)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_xlabel(xlab, fontsize=8)
        ax.set_ylabel(ylab, fontsize=8)
        ax.text(0.97, 0.95, f"slope = {slope:.2g}", transform=ax.transAxes, ha="right", va="top",
                fontsize=7, bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8,
                                      edgecolor="none"))
        ax.tick_params(labelsize=7.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

        # Effects: mediated / direct / total
        ax = axes[j, 1]
        tm = eff_tab[eff_tab.diversity_metric == m].set_index("effect").loc[MED_QUANTITIES]
        ypos = np.arange(len(MED_QUANTITIES))[::-1]
        est, lo, hi = (tm[c].to_numpy(float) / sc for c in ("estimate", "ci_lower", "ci_upper"))
        ax.errorbar(est, ypos, xerr=[est - lo, hi - est], fmt="o", color=col, ecolor=col,
                    ms=5, elinewidth=1.6, capsize=4, capthick=1.4, zorder=3)
        ax.axvline(0, color="#1a1a1a", lw=0.7, ls=":", zorder=0)
        ax.set_yticks(ypos)
        ax.set_yticklabels(MED_QUANTITIES, fontsize=9)
        ax.set_ylim(-0.6, len(MED_QUANTITIES) - 0.4)
        sig_col = "significant_fdr" if MED_STAR_RULE == "fdr" else "ci_excludes_0"
        for yy, s_ in zip(ypos, tm[sig_col].to_numpy(bool)):
            if s_:
                ax.text(1.04, yy, "*", transform=ax.get_yaxis_transform(), ha="left", va="center",
                        fontsize=15, fontweight="bold", color=col)
        per_txt = "with every metric +1 SD" if is_net else f"per 1 SD of {m}"
        ax.set_xlabel(f"Effect on stability {per_txt}" + (f" (\u00d710$^{{{exp}}}$)" if exp else ""),
                      fontsize=8)
        if is_net and "net_prop_mediated" in tab.attrs:
            prop, plo, phi = tab.attrs["net_prop_mediated"]
            total_sig = bool(tm.loc["Total", "ci_excludes_0"]) if "Total" in tm.index else True
            txt = (f"{prop:.0f}% mediated by {med}\n[{plo:.0f}, {phi:.0f}]" if total_sig else
                   f"% mediated not reported:\ntotal effect CI includes 0")
            ax.set_title(txt.replace("\n", " "), fontsize=7, color=col, loc="right", pad=4)
        ax.tick_params(axis="x", labelsize=7.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

        for k, a in enumerate(axes[j]):
            a.text(-0.32 if k == 0 else -0.42, 1.08, "ABCDEFGHIJ"[2 * j + k], transform=a.transAxes,
                   fontsize=11, fontweight="bold", va="bottom")
        axes[j, 0].set_title(m, fontsize=9.5, fontweight="bold", color=col, loc="left", pad=6)

    # Same x-range for every effects panel, so metrics can be compared
    xs = np.concatenate([a.get_xlim() for a in axes[:, 1]])
    for a in axes[:, 1]:
        a.set_xlim(xs.min(), xs.max())

    model_txt = "adjusted for the other metrics" if model == "adjusted" else "each metric on its own"
    rule = "FDR-corrected p < 0.05" if MED_STAR_RULE == "fdr" else f"{CI[1] - CI[0]:.0f}% CI excludes 0"
    fig.text(0.5, 0.005, f"Mediation via {med} ({model_txt}); error bars = {CI[1] - CI[0]:.0f}% "
             f"bootstrap CI; * = {rule}; scatter = random {MED_SCATTER_MAX_POINTS:,} plots",
             ha="center", va="bottom", fontsize=7, color="#555555")
    fig.savefig(savepath, dpi=300, bbox_inches="tight")
    fig.savefig(os.path.splitext(savepath)[0] + ".pdf", bbox_inches="tight")
    print(f"[figure] saved {savepath}")
    return fig


# ---------------------------------------------------------------
# Run
# ---------------------------------------------------------------
def run(outdir=OUTDIR, n_boot=N_BOOT):
    os.makedirs(outdir, exist_ok=True)
    df = load_data()

    results = {}
    for mkey in RUN_MODERATORS:
        results[mkey] = run_moderator(df, mkey, n_boot)

    # FDR across all contrast tests (moderators x models x metrics)
    all_tests = pd.concat([r[2].assign(_key=k) for k, r in results.items()], ignore_index=True)
    reject, p_adj, _, _ = multipletests(all_tests["p_value_raw"].fillna(1.0), alpha=ALPHA, method="fdr_bh")
    all_tests["p_value_fdr"], all_tests["significant_fdr"] = p_adj, reject

    for mkey, (curves, bins_df, _, g) in results.items():
        tests = all_tests[all_tests._key == mkey].drop(columns="_key")
        curves.to_csv(os.path.join(outdir, f"modmed_{mkey}_curves.csv"), index=False)
        bins_df.to_csv(os.path.join(outdir, f"modmed_{mkey}_bins.csv"), index=False)
        tests.to_csv(os.path.join(outdir, f"modmed_{mkey}_tests.csv"), index=False)
        print(f"\n[{mkey}] moderation of the paths and change in the indirect effect:")
        show = ["model", "diversity_metric", "a0", "b0", "a1_p", "b1_p", "indirect_contrast",
                "indirect_contrast_ci_lower", "indirect_contrast_ci_upper", "p_value_fdr"]
        print(tests[show].to_string(index=False, float_format=lambda x: f"{x:.3g}"))

    for model in FIG_MODELS:
        if model in MODELS:
            plot_indirect_combined(results, all_tests, model,
                                   os.path.join(outdir, f"modmed_indirect_combined_{model}.png"))

    # Grouped estimates: forest type x rarity class
    grouped = {}
    for model in GP_MODELS:
        groups, contrasts, metrics, rlabels, counts = run_grouped(df, model)
        groups.to_csv(os.path.join(outdir, f"modmed_grouped_{model}.csv"), index=False)
        contrasts.to_csv(os.path.join(outdir, f"modmed_grouped_contrasts_{model}.csv"), index=False)
        print(f"\n[grouped] contrasts of the indirect effect ({model}):")
        print(contrasts.to_string(index=False, float_format=lambda x: f"{x:.3g}"))
        plot_grouped(groups, contrasts, metrics, rlabels, counts, model,
                     os.path.join(outdir, f"modmed_grouped_{model}.png"))
        grouped[model] = (groups, contrasts)
    # Mediation summary: scatter + mediated / direct / total effects per metric
    for model in MED_FIG_MODELS:
        tab, Y, M, D = run_mediation_summary(df, model, n_boot)
        tab.to_csv(os.path.join(outdir, f"mediation_summary_{model}.csv"), index=False)
        print(f"\n[mediation summary] {model}:")
        print(tab.to_string(index=False, float_format=lambda x: f"{x:.3g}"))
        plot_mediation_summary(tab, Y, M, D, model,
                               os.path.join(outdir, f"mediation_summary_{model}.png"))
    return results, all_tests, grouped


if __name__ == "__main__":
    run()
    plt.show()