"""
Direction and size of each diversity metric's effect on stability across % conifer (x)
and % rare species (y), one panel per metric (panel title in the metric's colour).
Three figures:

  1. HEATMAP  Gaussian-smoothed surface (SMOOTH_HEATMAP), or square cells
              (CONIFER_CELL x RARE_CELL) with SMOOTH_HEATMAP = False
  2. HEXBIN   hexagons (HEX_GRIDSIZE)
  3. GROUPED  point-range plot: forest type (x) x rarity class (colour), 95% bootstrap CIs

COLOUR is a diverging, colour-blind-friendly scale (based on PiYG), symmetric about 0:
  destabilising (magenta) <- near-white (no effect) -> stabilising (green)
With SCALE_PER_METRIC = False (default) one range is shared by all panels, so the same
shade means the same effect everywhere; True gives each panel its own range.

DENSITY: with FADE_BY_DENSITY, areas fade toward white where they rest on fewer plots
(fully opaque from FADE_FULL_N plots), so sparsely sampled areas do not look as reliable
as densely sampled ones. Areas below the minimum number of plots are hatched.

CONTOURS outline the strongest effects in EACH direction, from the smoothed surface:
  solid dark-green lines    = strongest STABILISING areas
  dashed dark-magenta lines = strongest DESTABILISING areas
EFFECT_CONTOUR_QUANTILES = (0.50, 0.90) draws the strongest 50% and 10% on each side
(thicker = stronger).

THE MODEL is taken from EFFECTS_FILE:
  individual-effects file        -> each metric on its own
  mutually-adjusted effects file -> each metric with the other metrics held constant

WHAT IS MAPPED (EFFECT_MODE)
  "local_slope" (default): the slope of stability on diversity WITHIN each area
      (no intercept, as in the model equations), as % change in stability per 1
      full-sample SD of D~ (assumes the stability residual is on a log scale).
        individual       : theta_local = sum(D~ S~) / sum(D~^2)
        mutually adjusted: a local multiple regression of S~ on ALL D~_j together,
                           theta_local = (X'X)^-1 X'S~ within the area
      Needs diversity_residual and stability_residual in EFFECTS_FILE.
  "contribution": the mean per-plot effect, theta_j * (D~_j - mean D~_j) from ONE pooled
      model. Its sign only says whether plots there have above- or below-expected
      diversity (times the sign of theta_j); it is NOT a change in the effect itself.

GROUPED FIGURE: the same quantity computed from the sums of each forest type x rarity
group. Percentile pairs bootstrap (plots resampled, treated as independent). Contrasts:
high minus low rarity within each forest type, broadleaf- minus conifer-dominated within
each rarity class; Benjamini-Hochberg FDR across all contrasts.

RICHNESS CONTOURS (SHOW_RICHNESS_CONTOURS): lines of mean raw species richness per area.
"""

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.patheffects as pe
from scipy.ndimage import gaussian_filter
from statsmodels.stats.multitest import multipletests

# ===============================================================
# Input and output files  (edit names here only)
# ===============================================================
EFFECTS_FILE = "results/ols/mutually_adjusted_effects_per_observation.csv"
CONIFER_LOOKUP_PATH = "data/lookup/PID_location_v3.csv"
RARE_PATH = "data/final/dataset_rarity.parquet"
DIVERSITY_DATA_PATH = "data/final/final_dataset_ba.csv"   # raw species richness per plot

OUTDIR = "results/heatmaps"
# "{model}" = individual / mutually_adjusted (from EFFECTS_FILE), "{mode}" = EFFECT_MODE;
# figures are saved as .png and .pdf
OUT_HEATMAP_NAME = "effects_conifer_rarity_{model}_{mode}_heatmap"
OUT_HEXBIN_NAME = "effects_conifer_rarity_{model}_{mode}_hexbin"
OUT_GROUPED_NAME = "effects_forest_rarity_{model}_{mode}_grouped"
OUT_CELLS_CSV = "effects_conifer_rarity_{model}_{mode}_cells.csv"

# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------
EFFECT_MODE = "local_slope"             # "local_slope" or "contribution" (see docstring)
EFFECT_COL = None                       # "contribution" only; None = adjusted_effect or
                                        #   individual_effect, whichever the file has
D_RESID_COL = "diversity_residual"      # "local_slope"
S_RESID_COL = "stability_residual"      # "local_slope" (already signed: + = stabilising)

CONIFER_COL = "percent_conifer"
RARE_COL = "pct_rare_species"
RARE_MIN_RICHNESS = None                # e.g. 5: only plots with >= this many species
RICHNESS_COL = "Species Richness"

X_LABEL = "% conifer\n\u2190 conifer-dominated          broadleaf-dominated \u2192"
Y_LABEL = "% rare species"

# Contours of mean species richness (raw, un-residualised) per area
SHOW_RICHNESS_CONTOURS = False
RICHNESS_LEVELS = None                  # e.g. (4, 6, 8, 10); None = ~5 automatic whole-number levels
RICHNESS_LINE_COLOR = "#000000"
RICHNESS_LINE_WIDTH = 1.0
RICHNESS_LINE_STYLE = (0, (5, 1.5, 1.2, 1.5))
RICHNESS_HALO = True

METRIC_ORDER = ["Rao's Q", "Functional Evenness", "Species richness"]
METRIC_COLORS = {                       # panel titles only
    "Rao's Q": "#4910D8",
    "Functional Evenness": "#9975BA",
    "Species richness": "#43D9DC",
}

# Diverging colours: destabilising (magenta) <- neutral -> stabilising (green), based on PiYG
DESTAB_COLOR = "#C51B7D"
STAB_COLOR = "#4D9221"
NEUTRAL_COLOR = "#F7F5F2"
DIV_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "destab_stab", [DESTAB_COLOR, "#E9A3C9", NEUTRAL_COLOR, "#A1D76A", STAB_COLOR])
STAB_LINE_COLOR = "#2E5A12"             # dark green, solid contours and labels
DESTAB_LINE_COLOR = "#7A1050"           # dark magenta, dashed contours and labels

SCALE_PER_METRIC = False       # False: one shared range (comparable panels); True: own range per panel
RANGE_PCT_ABS = 98             # colour range = +/- this percentile of |values|

# Fade sparsely sampled areas toward white (alpha on a log scale of the plot count)
FADE_BY_DENSITY = True
FADE_FULL_N = 300              # fully opaque from this many plots
FADE_MIN_ALPHA = 0.30          # opacity at the minimum number of plots

EFFECT_CONTOUR_QUANTILES = (0.50, 0.90)     # () = no contours
CONTOUR_LABELS = False                      # inline "50%" / "10%" labels on the contours
MIN_CONTOUR_POINTS = 10

RARE_MAX_PCT = 99              # % rare axis stops at this percentile

# 1. Heatmap
CONIFER_CELL = 20
RARE_CELL = 10
MIN_PLOTS_PER_CELL = 30
SMOOTH_HEATMAP = True
SMOOTH_SD = (6.0, 4.0)         # Gaussian SD in percentage points (% conifer, % rare)
SMOOTH_STEP = 1.0

# 2. Hexbin
HEX_GRIDSIZE = (18, 12)
MIN_PLOTS_PER_HEX = 10

# 3. Grouped point-range
FOREST_CLASSES = [             # (label, lower bound incl., upper bound excl.) on % conifer, left to right
    ("Conifer-\ndominated", 75, np.inf),
    ("Mixed", 25, 75),
    ("Broadleaf-\ndominated", -np.inf, 25),
]
RARITY_CUTS = None             # None = tertiles of % rare; or cut points in %, e.g. [30, 60]
RARITY_COLORS = ["#C6C6C6", "#7A7A7A", "#1F1F1F"]   # low -> high rarity (neutral greys)
GROUP_MIN_N = 100
N_BOOT = 1000
CI = (2.5, 97.5)
ALPHA = 0.05
SEED = 42
DODGE = 0.24

FIG_WIDTH_IN = 7.2
DPI = 600

# Filled by load_data()
_STATE = {"model": None, "metrics": [], "sd": {}, "moments": []}


def _out(name):
    return os.path.join(OUTDIR, name.format(model=_STATE["model"], mode=EFFECT_MODE))


# ---------------------------------------------------------------
# Data: one row per plot with x, y and the "moments" summed over each area
# ---------------------------------------------------------------
def _xx(j, k):
    j, k = sorted((j, k))
    return f"xx|{j}|{k}"


def load_data():
    """
    Returns P (one row per plot) with CONIFER_COL, RARE_COL, 'n' (= 1) and moment columns.
    An area's value comes from the SUMS of the moments over its plots (_values()):
      local_slope, individual       : xs|m = D_m S,  xx|m|m = D_m^2
      local_slope, mutually adjusted: xs|j = D_j S,  xx|j|k = D_j D_k  (all pairs)
      contribution                  : c|m = the plot's signed effect as % change
    """
    eff = pd.read_csv(EFFECTS_FILE)
    model = "mutually_adjusted" if "adjusted_effect" in eff.columns else "individual"
    metrics = [m for m in METRIC_ORDER if m in eff["diversity_metric"].unique()]
    _STATE.update(model=model, metrics=metrics)

    if EFFECT_MODE == "local_slope":
        missing = [c for c in (D_RESID_COL, S_RESID_COL) if c not in eff.columns]
        if missing:
            raise KeyError(f"{EFFECTS_FILE} lacks {missing}. Rerun the updated effects script "
                           f"(it now writes diversity_residual to both per-observation files), "
                           f"or set EFFECT_MODE = 'contribution'.")
        D = eff.pivot(index="PID", columns="diversity_metric", values=D_RESID_COL)[metrics]
        S = eff.groupby("PID")[S_RESID_COL].first()
        W = D.join(S.rename("_S")).dropna()
        _STATE["sd"] = {m: W[m].std(ddof=1) for m in metrics}
        mom = {}
        for j in metrics:
            mom[f"xs|{j}"] = W[j] * W["_S"]
            pairs = metrics if model == "mutually_adjusted" else [j]
            for k in pairs:
                mom[_xx(j, k)] = W[j] * W[k]
        P = pd.DataFrame(mom)
    else:
        col = EFFECT_COL or ("adjusted_effect" if model == "mutually_adjusted" else "individual_effect")
        if col not in eff.columns:
            raise KeyError(f"{EFFECTS_FILE} has no column '{col}'.")
        E = eff.pivot(index="PID", columns="diversity_metric", values=col)[metrics].dropna()
        P = ((np.exp(E) - 1) * 100).add_prefix("c|")
    P["n"] = 1.0
    _STATE["moments"] = list(P.columns)
    P = P.reset_index()

    lk = pd.read_csv(CONIFER_LOOKUP_PATH, usecols=["PID", CONIFER_COL]).drop_duplicates("PID")
    rare = pd.read_parquet(RARE_PATH, columns=["PID", RARE_COL]).drop_duplicates("PID")
    P = P.merge(lk, on="PID", how="left").merge(rare, on="PID", how="left")
    if RARE_MIN_RICHNESS or SHOW_RICHNESS_CONTOURS:
        sr = pd.read_csv(DIVERSITY_DATA_PATH, usecols=["PID", RICHNESS_COL]).drop_duplicates("PID")
        P = P.merge(sr, on="PID", how="left")
    if RARE_MIN_RICHNESS:
        P = P[P[RICHNESS_COL] >= RARE_MIN_RICHNESS]
    P = P.dropna(subset=[CONIFER_COL, RARE_COL])
    for c in (CONIFER_COL, RARE_COL):
        if P[c].max() <= 1:
            P[c] = P[c] * 100
    rare_max = float(np.ceil(np.percentile(P[RARE_COL], RARE_MAX_PCT) / RARE_CELL) * RARE_CELL)
    P = P[P[RARE_COL] <= rare_max].reset_index(drop=True)
    print(f"[load] {len(P):,} plots; model = {model}; mode = {EFFECT_MODE}; metrics = {metrics}")
    return P, rare_max


def _values(M):
    """{metric: value array (% change in stability, signed)} from summed moments M."""
    metrics, sd = _STATE["metrics"], _STATE["sd"]
    shape = np.shape(M["n"])
    out = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        if EFFECT_MODE != "local_slope":
            return {m: np.asarray(M[f"c|{m}"]) / np.asarray(M["n"]) for m in metrics}
        if _STATE["model"] == "individual":
            for m in metrics:
                th = np.asarray(M[f"xs|{m}"]) / np.asarray(M[_xx(m, m)])
                out[m] = (np.exp(th * sd[m]) - 1) * 100
            return out
        p = len(metrics)
        A = np.empty((int(np.prod(shape)), p, p))
        b = np.empty((A.shape[0], p))
        for j, mj in enumerate(metrics):
            b[:, j] = np.ravel(M[f"xs|{mj}"])
            for k, mk in enumerate(metrics):
                A[:, j, k] = np.ravel(M[_xx(mj, mk)])
        theta = np.full(b.shape, np.nan)
        good = np.isfinite(A).all(axis=(1, 2)) & np.isfinite(b).all(axis=1) & (np.ravel(M["n"]) > 0)
        if good.any():
            theta[good] = (np.linalg.pinv(A[good]) @ b[good][..., None])[..., 0]
        for j, m in enumerate(metrics):
            out[m] = ((np.exp(theta[:, j] * sd[m]) - 1) * 100).reshape(shape)
    return out


def _value_label():
    if EFFECT_MODE == "local_slope":
        return "Local effect (% change in stability per 1 SD)"
    return "Mean per-plot effect (% change)"


def _model_note():
    what = ("others held constant (local multiple regression)" if _STATE["model"] == "mutually_adjusted"
            else "each metric on its own")
    kind = "Local slopes" if EFFECT_MODE == "local_slope" else "Mean per-plot contributions"
    return f"{kind}, {what}"


# ---------------------------------------------------------------
# Colour and density
# ---------------------------------------------------------------
def make_norm(values):
    v = np.abs(np.asarray(values, dtype=float))
    v = v[np.isfinite(v)]
    vmax = float(np.percentile(v, RANGE_PCT_ABS)) if v.size else 1.0
    vmax = vmax if vmax > 0 else 1e-12
    return mcolors.Normalize(-vmax, vmax)


def div_colors(v, norm):
    return DIV_CMAP(norm(np.asarray(v, dtype=float)))


def density_alpha(n, n_min):
    """Opacity from the plot count: FADE_MIN_ALPHA at n_min, 1 from FADE_FULL_N (log scale)."""
    n = np.asarray(n, dtype=float)
    if not FADE_BY_DENSITY:
        return np.ones_like(n)
    lo, hi = np.log(max(n_min, 1)), np.log(max(FADE_FULL_N, n_min + 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (np.log(np.maximum(n, 1)) - lo) / (hi - lo)
    return np.clip(FADE_MIN_ALPHA + (1 - FADE_MIN_ALPHA) * t, FADE_MIN_ALPHA, 1.0)


# ---------------------------------------------------------------
# Shared drawing helpers
# ---------------------------------------------------------------
def _style():
    have = {f.name for f in font_manager.fontManager.ttflist}
    font = next((n for n in ("Arial", "Helvetica", "Liberation Sans", "DejaVu Sans") if n in have),
                "sans-serif")
    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": [font],
                         "pdf.fonttype": 42, "ps.fonttype": 42, "hatch.linewidth": 0.3,
                         "axes.linewidth": 0.5, "xtick.major.width": 0.4, "ytick.major.width": 0.4})


def _new_figure(n):
    fig, axes = plt.subplots(1, n, figsize=(FIG_WIDTH_IN, 3.0), sharey=True, squeeze=False)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.88, bottom=0.45, wspace=0.10)
    return fig, axes[0]


def _finish_axes(ax, i, m, rare_max):
    ax.set_xlim(100, 0)
    ax.set_ylim(0, rare_max)
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(20))
    ax.set_xlabel(X_LABEL, fontsize=6.0, labelpad=1.5)
    if i == 0:
        ax.set_ylabel(Y_LABEL, fontsize=6.5)
    for name, sp in ax.spines.items():
        sp.set_visible(name in ("left", "bottom"))
    ax.tick_params(labelsize=6, length=2, pad=1.5)
    ax.set_title(m, fontsize=7.5, color=METRIC_COLORS.get(m, "black"), fontweight="bold", pad=3)
    ax.text(-0.14, 1.06, "abcdef"[i], transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")


def _panel_colorbar(fig, ax, norm):
    pos = ax.get_position()
    cax = fig.add_axes([pos.x0 + 0.1 * pos.width, 0.19, 0.8 * pos.width, 0.03])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=DIV_CMAP), cax=cax,
                      orientation="horizontal", extend="both")
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(labelsize=5.3, length=2, width=0.4, pad=1)
    cb.ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(5, symmetric=True))
    cb.ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(
        lambda v, _: "0" if abs(v) < 1e-12 else f"{v:+.2g}".replace("-", "\u2212")))
    cax.xaxis.set_label_position("top")
    cax.set_xlabel(_value_label(), fontsize=5.8, labelpad=2)
    cax.text(0, -1.9, "\u2190 destabilising", transform=cax.transAxes, ha="left", va="top",
             fontsize=5, color=DESTAB_LINE_COLOR)
    cax.text(1, -1.9, "stabilising \u2192", transform=cax.transAxes, ha="right", va="top",
             fontsize=5, color=STAB_LINE_COLOR)


def _hatch_background(ax, rare_max):
    ax.add_patch(plt.Rectangle((0, 0), 100, rare_max, facecolor="white", edgecolor="#d0d0d0",
                               linewidth=0, hatch="//////", zorder=0))


def _draw_rgba(ax, val, n, norm, extent, n_min, interpolation):
    """White underlay where there are values (so fading goes to white, not to the hatching),
    then the colours with density-based opacity."""
    ok = np.isfinite(val)
    under = np.zeros(val.shape + (4,))
    under[ok] = (1.0, 1.0, 1.0, 1.0)
    img = np.zeros(val.shape + (4,))
    img[..., :3] = 1.0
    img[ok] = div_colors(val[ok], norm)
    img[ok, 3] = density_alpha(n[ok], n_min)
    ax.imshow(under, origin="lower", extent=extent, aspect="auto", interpolation="nearest", zorder=0.5)
    ax.imshow(img, origin="lower", extent=extent, aspect="auto", interpolation=interpolation, zorder=1)


def _grid_edges(rare_max):
    ex = np.arange(0, 100 + SMOOTH_STEP, SMOOTH_STEP)
    ey = np.arange(0, rare_max + SMOOTH_STEP, SMOOTH_STEP)
    return ex, ey


def smooth_surfaces(P, rare_max):
    """{metric: value grid}, ey, and the effective number of plots per grid point."""
    ex, ey = _grid_edges(rare_max)
    sig = (SMOOTH_SD[1] / SMOOTH_STEP, SMOOTH_SD[0] / SMOOTH_STEP)       # (rows = rare, cols = conifer)
    M = {}
    for c in _STATE["moments"]:
        h, _, _ = np.histogram2d(P[RARE_COL], P[CONIFER_COL], bins=(ey, ex), weights=P[c])
        M[c] = gaussian_filter(h, sig, mode="constant")
    vals = _values(M)
    n_eff = M["n"] * 2 * np.pi * sig[0] * sig[1]
    for v in vals.values():
        v[n_eff < MIN_PLOTS_PER_CELL] = np.nan
    return vals, ey, n_eff


def effect_contours(ax, val, rare_max):
    if not EFFECT_CONTOUR_QUANTILES:
        return
    ex, ey = _grid_edges(rare_max)
    xc, yc = (ex[:-1] + ex[1:]) / 2, (ey[:-1] + ey[1:]) / 2
    z = np.ma.masked_invalid(val)
    finite = val[np.isfinite(val)]
    qs = sorted(EFFECT_CONTOUR_QUANTILES)
    widths_by_q = np.linspace(0.6, 1.1, len(qs))

    for sign, color, ls in ((1, STAB_LINE_COLOR, "solid"), (-1, DESTAB_LINE_COLOR, "dashed")):
        side = sign * finite[sign * finite > 0]
        if side.size < MIN_CONTOUR_POINTS:
            continue
        mags = [float(np.quantile(side, q)) for q in qs]
        if len(set(mags)) < len(mags):
            continue
        levels = [sign * mg for mg in mags]
        order = np.argsort(levels)
        levels = [levels[k] for k in order]
        widths = [widths_by_q[k] for k in order]
        cs = ax.contour(xc, yc, z, levels=levels, colors=color, linewidths=widths,
                        linestyles=ls, zorder=3)
        if CONTOUR_LABELS:
            names = {sign * mg: f"{round((1 - q) * 100)}%" for mg, q in zip(mags, qs)}
            ax.clabel(cs, fmt=lambda lv: names.get(lv, ""), fontsize=4.5, inline=True, inline_spacing=1)


def richness_surface(P, rare_max):
    if "richness" in _STATE:
        return _STATE["richness"]
    ex, ey = _grid_edges(rare_max)
    sig = (SMOOTH_SD[1] / SMOOTH_STEP, SMOOTH_SD[0] / SMOOTH_STEP)
    R = P.dropna(subset=[RICHNESS_COL])
    cnt, _, _ = np.histogram2d(R[RARE_COL], R[CONIFER_COL], bins=(ey, ex))
    tot, _, _ = np.histogram2d(R[RARE_COL], R[CONIFER_COL], bins=(ey, ex), weights=R[RICHNESS_COL])
    c_s = gaussian_filter(cnt, sig, mode="constant")
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = gaussian_filter(tot, sig, mode="constant") / c_s
    mean[c_s * 2 * np.pi * sig[0] * sig[1] < MIN_PLOTS_PER_CELL] = np.nan
    _STATE["richness"] = mean
    return mean


def richness_contours(ax, P, rare_max):
    if not SHOW_RICHNESS_CONTOURS or RICHNESS_COL not in P:
        return
    val = richness_surface(P, rare_max)
    finite = val[np.isfinite(val)]
    if finite.size < MIN_CONTOUR_POINTS:
        return
    lo, hi = float(finite.min()), float(finite.max())
    levels = (RICHNESS_LEVELS if RICHNESS_LEVELS is not None
              else mpl.ticker.MaxNLocator(5, integer=True).tick_values(lo, hi))
    levels = [lv for lv in levels if lo < lv < hi]
    if not levels:
        return
    ex, ey = _grid_edges(rare_max)
    xc, yc = (ex[:-1] + ex[1:]) / 2, (ey[:-1] + ey[1:]) / 2
    cs = ax.contour(xc, yc, np.ma.masked_invalid(val), levels=levels, colors=RICHNESS_LINE_COLOR,
                    linewidths=RICHNESS_LINE_WIDTH, linestyles=[RICHNESS_LINE_STYLE], zorder=5)
    if RICHNESS_HALO:
        halo = [pe.Stroke(linewidth=RICHNESS_LINE_WIDTH + 1.6, foreground="white"), pe.Normal()]
        for artist in getattr(cs, "collections", None) or [cs]:
            artist.set_path_effects(halo)
    labels = ax.clabel(cs, fmt=lambda lv: f"{lv:g} spp", fontsize=5, inline=True, inline_spacing=2)
    for t in labels:
        t.set_fontweight("bold")
        t.set_color(RICHNESS_LINE_COLOR)
        t.set_bbox(dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.85))
        t.set_zorder(6)


def _legend(fig, n_min):
    handles = [Patch(facecolor="white", edgecolor="#d0d0d0", hatch="//////", linewidth=0.4,
                     label=f"< {n_min} plots")]
    if FADE_BY_DENSITY:
        handles.append(Patch(facecolor=STAB_COLOR, alpha=FADE_MIN_ALPHA, linewidth=0,
                             label=f"faded: fewer plots (opaque \u2265 {FADE_FULL_N})"))
    if EFFECT_CONTOUR_QUANTILES:
        tops = " / ".join(f"{round((1 - q) * 100)}" for q in sorted(EFFECT_CONTOUR_QUANTILES))
        handles += [Line2D([], [], color=STAB_LINE_COLOR, lw=0.9, ls="solid",
                           label=f"Strongest stabilising {tops}%"),
                    Line2D([], [], color=DESTAB_LINE_COLOR, lw=0.9, ls="dashed",
                           label=f"Strongest destabilising {tops}%")]
    if SHOW_RICHNESS_CONTOURS:
        handles.append(Line2D([], [], color=RICHNESS_LINE_COLOR, lw=RICHNESS_LINE_WIDTH,
                              ls=RICHNESS_LINE_STYLE, label="Mean species richness",
                              path_effects=[pe.Stroke(linewidth=RICHNESS_LINE_WIDTH + 1.6,
                                                      foreground="white"), pe.Normal()]))
    fig.legend(handles=handles, loc="lower right", ncol=len(handles), bbox_to_anchor=(0.99, 0.0),
               frameon=False, fontsize=5.4, handlelength=2.0, columnspacing=1.0)


def _footnote(fig, extra=""):
    fig.text(0.08, 0.07, _model_note() + extra, fontsize=5.4, color="#666666")


def _save(fig, stem):
    os.makedirs(os.path.dirname(stem) or ".", exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=DPI, facecolor="white")
    print(f"[figure] saved {stem}.png / .pdf")


# ---------------------------------------------------------------
# 1. Heatmap
# ---------------------------------------------------------------
def heatmap_cells(P, rare_max):
    cx = np.arange(0, 100 + CONIFER_CELL, CONIFER_CELL)
    cy = np.arange(0, rare_max + RARE_CELL, RARE_CELL)
    Q = P.assign(bx=pd.cut(P[CONIFER_COL], cx, include_lowest=True, labels=False),
                 by=pd.cut(P[RARE_COL], cy, include_lowest=True, labels=False))
    g = Q.groupby(["bx", "by"])[_STATE["moments"]].sum().reset_index()
    vals = _values({c: g[c].to_numpy(float) for c in _STATE["moments"]})
    rows = []
    for m in _STATE["metrics"]:
        rows.append(pd.DataFrame({
            "diversity_metric": m, "bx": g["bx"].astype(int), "by": g["by"].astype(int),
            "conifer_lo": cx[g["bx"].astype(int)], "rare_lo": cy[g["by"].astype(int)],
            "n": g["n"].astype(int), "value_pct": vals[m],
            "enough": g["n"] >= MIN_PLOTS_PER_CELL,
            "model": _STATE["model"], "mode": EFFECT_MODE}))
    return pd.concat(rows, ignore_index=True), cx, cy


def plot_heatmap(P, rare_max, stem):
    cells, cx, cy = heatmap_cells(P, rare_max)
    os.makedirs(OUTDIR, exist_ok=True)
    cells.drop(columns=["bx", "by"]).to_csv(_out(OUT_CELLS_CSV), index=False)
    if SMOOTH_HEATMAP:
        return plot_heatmap_smooth(P, rare_max, stem)

    metrics = _STATE["metrics"]
    ok = cells[cells.enough & np.isfinite(cells.value_pct)]
    shared = make_norm(ok["value_pct"])
    surf = smooth_surfaces(P, rare_max)[0] if EFFECT_CONTOUR_QUANTILES else {}
    fig, axes = _new_figure(len(metrics))
    shape = (len(cy) - 1, len(cx) - 1)
    for i, (ax, m) in enumerate(zip(axes, metrics)):
        c = ok[ok.diversity_metric == m]
        norm = make_norm(c["value_pct"]) if SCALE_PER_METRIC else shared
        val = np.full(shape, np.nan)
        cnt = np.zeros(shape)
        val[c.by, c.bx] = c["value_pct"]
        cnt[c.by, c.bx] = c["n"]
        _hatch_background(ax, rare_max)
        _draw_rgba(ax, val, cnt, norm, (0, 100, 0, cy[-1]), MIN_PLOTS_PER_CELL, "nearest")
        for x in cx[1:-1]:
            ax.axvline(x, color="white", lw=0.3, zorder=2)
        for y in cy[1:-1]:
            ax.axhline(y, color="white", lw=0.3, zorder=2)
        if m in surf:
            effect_contours(ax, surf[m], rare_max)
        richness_contours(ax, P, rare_max)
        _finish_axes(ax, i, m, rare_max)
        _panel_colorbar(fig, ax, norm)
    _legend(fig, MIN_PLOTS_PER_CELL)
    _footnote(fig)
    _save(fig, stem)
    return fig


def plot_heatmap_smooth(P, rare_max, stem):
    metrics = _STATE["metrics"]
    surf, ey, n_eff = smooth_surfaces(P, rare_max)
    shared = make_norm(np.concatenate([s.ravel() for s in surf.values()]))

    fig, axes = _new_figure(len(metrics))
    for i, (ax, m) in enumerate(zip(axes, metrics)):
        val = surf[m]
        norm = make_norm(val.ravel()) if SCALE_PER_METRIC else shared
        _hatch_background(ax, rare_max)
        _draw_rgba(ax, val, n_eff, norm, (0, 100, 0, ey[-1]), MIN_PLOTS_PER_CELL, "bilinear")
        effect_contours(ax, val, rare_max)
        richness_contours(ax, P, rare_max)
        _finish_axes(ax, i, m, rare_max)
        _panel_colorbar(fig, ax, norm)
    _legend(fig, MIN_PLOTS_PER_CELL)
    _footnote(fig, f"; Gaussian-smoothed (SD = {SMOOTH_SD[0]:g} pp conifer, {SMOOTH_SD[1]:g} pp rare)")
    _save(fig, stem)
    return fig


# ---------------------------------------------------------------
# 2. Hexbin
# ---------------------------------------------------------------
def _hex_kw(rare_max):
    return dict(gridsize=HEX_GRIDSIZE, extent=(0, 100, 0, rare_max), mincnt=MIN_PLOTS_PER_HEX,
                reduce_C_function=np.sum)


def _hex_values(P, rare_max):
    tmp_fig, tmp_ax = plt.subplots()
    kw = _hex_kw(rare_max)
    M = {c: np.asarray(tmp_ax.hexbin(P[CONIFER_COL], P[RARE_COL], C=P[c], **kw).get_array(), float)
         for c in _STATE["moments"]}
    plt.close(tmp_fig)
    return _values(M), M["n"]


def plot_hexbin(P, rare_max, stem):
    metrics = _STATE["metrics"]
    vals, counts = _hex_values(P, rare_max)
    shared = make_norm(np.concatenate(list(vals.values())))
    surf = smooth_surfaces(P, rare_max)[0] if EFFECT_CONTOUR_QUANTILES else {}

    fig, axes = _new_figure(len(metrics))
    for i, (ax, m) in enumerate(zip(axes, metrics)):
        _hatch_background(ax, rare_max)
        v = vals[m]
        ok = np.isfinite(v)
        # white underlay so faded hexagons fade to white, not to the hatching
        under = ax.hexbin(P[CONIFER_COL], P[RARE_COL], C=P["n"], linewidths=0,
                          **_hex_kw(rare_max))
        under.set_array(None)
        under.set_facecolor(np.where(ok[:, None], [1, 1, 1, 1], [0, 0, 0, 0]))
        under.set_zorder(0.5)

        hb = ax.hexbin(P[CONIFER_COL], P[RARE_COL], C=P["n"], linewidths=0.25,
                       edgecolors="white", **_hex_kw(rare_max))
        norm = make_norm(v) if SCALE_PER_METRIC else shared
        fc = div_colors(np.nan_to_num(v), norm)
        fc[:, 3] = np.where(ok, density_alpha(counts, MIN_PLOTS_PER_HEX), 0.0)   # unsolvable: hatched
        hb.set_array(None)
        hb.set_facecolor(fc)
        hb.set_zorder(1)
        if m in surf:
            effect_contours(ax, surf[m], rare_max)
        richness_contours(ax, P, rare_max)
        _finish_axes(ax, i, m, rare_max)
        _panel_colorbar(fig, ax, norm)
    _legend(fig, MIN_PLOTS_PER_HEX)
    _footnote(fig)
    _save(fig, stem)
    return fig


# ---------------------------------------------------------------
# 3. Grouped point-range: forest type x rarity class
# ---------------------------------------------------------------
def _groups(P):
    con, rar = P[CONIFER_COL].to_numpy(float), P[RARE_COL].to_numpy(float)
    n_f = len(FOREST_CLASSES)
    fclass = np.full(len(P), -1)
    for k, (_, lo, hi) in enumerate(FOREST_CLASSES):
        fclass[(con >= lo) & (con < hi)] = k
    cuts = list(np.percentile(rar, [100 / 3, 200 / 3])) if RARITY_CUTS is None else list(RARITY_CUTS)
    rclass = np.digitize(rar, cuts)
    n_r = len(cuts) + 1
    names = ["Low", "Medium", "High"] if n_r == 3 else [f"Class {k + 1}" for k in range(n_r)]
    rlabels = []
    for k, nm in enumerate(names):
        if k == 0:
            rlabels.append(f"{nm} rarity (< {cuts[0]:.0f}%)")
        elif k == n_r - 1:
            rlabels.append(f"{nm} rarity (\u2265 {cuts[-1]:.0f}%)")
        else:
            rlabels.append(f"{nm} rarity ({cuts[k - 1]:.0f}\u2013{cuts[k]:.0f}%)")
    gid = np.where(fclass >= 0, fclass * n_r + rclass, -1)
    return gid, n_f, n_r, rlabels


def _group_values(P, gid, n_groups, w=None):
    keep = gid >= 0
    ww = np.ones(len(P)) if w is None else w
    M = {c: np.bincount(gid[keep], weights=(ww * P[c].to_numpy(float))[keep], minlength=n_groups)
         for c in _STATE["moments"]}
    return _values(M)


def _boot_p(x):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    return float(min(1.0, 2 * min(np.mean(x <= 0), np.mean(x >= 0)))) if len(x) else np.nan


def plot_grouped(P, stem):
    metrics = _STATE["metrics"]
    gid, n_f, n_r, rlabels = _groups(P)
    G = n_f * n_r
    counts = np.bincount(gid[gid >= 0], minlength=G)
    small = counts < GROUP_MIN_N

    point = _group_values(P, gid, G)
    rng = np.random.default_rng(SEED)
    n = len(P)
    draws = {m: np.empty((N_BOOT, G)) for m in metrics}
    print(f"[grouped] bootstrap: {N_BOOT} resamples ...")
    for i in range(N_BOOT):
        w = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
        vb = _group_values(P, gid, G, w)
        for m in metrics:
            draws[m][i] = vb[m]

    rows, crow = [], []
    est, lo, hi = {}, {}, {}
    for m in metrics:
        e = np.where(small, np.nan, point[m])
        l_, h_ = np.nanpercentile(draws[m], CI, axis=0)
        l_[small], h_[small] = np.nan, np.nan
        est[m], lo[m], hi[m] = e, l_, h_
        for f in range(n_f):
            for r in range(n_r):
                g = f * n_r + r
                rows.append({"model": _STATE["model"], "mode": EFFECT_MODE, "diversity_metric": m,
                             "forest_type": FOREST_CLASSES[f][0].replace("\n", ""),
                             "rarity_class": rlabels[r], "n": int(counts[g]), "estimate": e[g],
                             "ci_lower": l_[g], "ci_upper": h_[g],
                             "p_boot": np.nan if small[g] else _boot_p(draws[m][:, g])})
        for f in range(n_f):
            g0, g1 = f * n_r, f * n_r + n_r - 1
            if small[g0] or small[g1]:
                continue
            diff = draws[m][:, g1] - draws[m][:, g0]
            crow.append({"diversity_metric": m,
                         "contrast": f"{FOREST_CLASSES[f][0].replace(chr(10), '')}: "
                                     f"{rlabels[-1]} minus {rlabels[0]}",
                         "estimate": point[m][g1] - point[m][g0],
                         "ci_lower": np.nanpercentile(diff, CI[0]),
                         "ci_upper": np.nanpercentile(diff, CI[1]), "p_raw": _boot_p(diff)})
        for r in range(n_r):
            g0, g1 = r, (n_f - 1) * n_r + r
            if small[g0] or small[g1]:
                continue
            diff = draws[m][:, g1] - draws[m][:, g0]
            crow.append({"diversity_metric": m,
                         "contrast": f"{rlabels[r]}: {FOREST_CLASSES[-1][0].replace(chr(10), '')} "
                                     f"minus {FOREST_CLASSES[0][0].replace(chr(10), '')}",
                         "estimate": point[m][g1] - point[m][g0],
                         "ci_lower": np.nanpercentile(diff, CI[0]),
                         "ci_upper": np.nanpercentile(diff, CI[1]), "p_raw": _boot_p(diff)})
    groups = pd.DataFrame(rows)
    contrasts = pd.DataFrame(crow)
    if len(contrasts):
        rej, padj, _, _ = multipletests(contrasts["p_raw"].fillna(1.0), alpha=ALPHA, method="fdr_bh")
        contrasts["p_fdr"], contrasts["significant_fdr"] = padj, rej
    groups.to_csv(f"{stem}_groups.csv", index=False)
    contrasts.to_csv(f"{stem}_contrasts.csv", index=False)
    print(contrasts.to_string(index=False, float_format=lambda x: f"{x:.3g}"))

    colors = (RARITY_COLORS * n_r)[:n_r]
    offsets = np.linspace(-DODGE, DODGE, n_r) if n_r > 1 else np.zeros(1)
    fig, axes = plt.subplots(1, len(metrics), figsize=(FIG_WIDTH_IN, 2.9), sharey=True, squeeze=False)
    axes = axes[0]
    fig.subplots_adjust(left=0.09, right=0.99, top=0.86, bottom=0.30, wspace=0.10)
    for i, (ax, m) in enumerate(zip(axes, metrics)):
        for f in range(n_f):
            for r in range(n_r):
                g = f * n_r + r
                x = f + offsets[r]
                ax.text(x, 0.99, f"{counts[g]:,}", transform=ax.get_xaxis_transform(),
                        rotation=90, ha="center", va="top", fontsize=4.6, color="#777777")
                if not np.isfinite(est[m][g]):
                    continue
                excl = (lo[m][g] > 0) or (hi[m][g] < 0)
                ax.errorbar(x, est[m][g], yerr=[[est[m][g] - lo[m][g]], [hi[m][g] - est[m][g]]],
                            fmt="o", ms=3.8, color=colors[r], mec=colors[r],
                            mfc=colors[r] if excl else "white", mew=0.9, elinewidth=0.9,
                            capsize=2, zorder=3)
        ax.axhline(0, color="#1a1a1a", lw=0.6, ls=":", zorder=0)
        ax.axhspan(0, 1e9, color=STAB_COLOR, alpha=0.04, zorder=-1, lw=0)
        ax.axhspan(-1e9, 0, color=DESTAB_COLOR, alpha=0.04, zorder=-1, lw=0)
        for xb in np.arange(0.5, n_f - 0.5):
            ax.axvline(xb, color="#e3e3e3", lw=0.5, zorder=0)
        ax.set_xticks(range(n_f))
        ax.set_xticklabels([c[0] for c in FOREST_CLASSES], fontsize=6)
        ax.set_xlim(-0.6, n_f - 0.4)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(labelsize=6, length=2, pad=1.5)
        ax.set_title(m, fontsize=7.5, color=METRIC_COLORS.get(m, "black"), fontweight="bold", pad=3)
        ax.text(-0.14, 1.06, "abcdef"[i], transform=ax.transAxes, fontsize=9, fontweight="bold",
                va="bottom")
    # keep the shaded halves within the data range
    lo_y = np.nanmin([np.nanmin(lo[m]) for m in metrics])
    hi_y = np.nanmax([np.nanmax(hi[m]) for m in metrics])
    pad = 0.12 * (hi_y - lo_y) if np.isfinite(hi_y - lo_y) and hi_y > lo_y else 1.0
    axes[0].set_ylim(min(lo_y - pad, -pad), max(hi_y + 2.2 * pad, pad))
    axes[0].set_ylabel(_value_label(), fontsize=6.5)

    handles = [Line2D([], [], marker="o", ls="none", color=c, mfc=c, ms=4, label=lab)
               for c, lab in zip(colors, rlabels)]
    handles += [Line2D([], [], marker="o", ls="none", color="#555555", mfc="#555555", ms=4,
                       label=f"{CI[1] - CI[0]:.0f}% CI excludes 0"),
                Line2D([], [], marker="o", ls="none", color="#555555", mfc="white", ms=4,
                       label="CI includes 0")]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
               fontsize=5.4, bbox_to_anchor=(0.5, 0.06), columnspacing=1.0, handletextpad=0.3)
    fig.text(0.09, 0.015, _model_note() + f"; numbers = plots per group; groups with "
             f"< {GROUP_MIN_N} plots omitted; {N_BOOT} bootstrap resamples",
             fontsize=5.0, color="#666666")
    _save(fig, stem)
    return fig


# ---------------------------------------------------------------
# Run
# ---------------------------------------------------------------
def run():
    _style()
    P, rare_max = load_data()
    return [plot_heatmap(P, rare_max, _out(OUT_HEATMAP_NAME)),
            plot_hexbin(P, rare_max, _out(OUT_HEXBIN_NAME)),
            plot_grouped(P, _out(OUT_GROUPED_NAME))]


if __name__ == "__main__":
    run()
    plt.show()