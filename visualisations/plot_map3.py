"""
Ecoregion map of the contiguous US showing the TOTAL effect of diversity
on NPP stability in each ecoregion (panel a), plus a trend panel (b) showing how
that total effect -- and each metric's contribution to it -- changes along the
% conifer gradient.

  (a) Colour runs from destabilising (DESTAB_COLOR) through white to stabilising
      (STAB_COLOR); the stronger the effect, the more intense the colour.
  (b) Mean total effect per % conifer bin (thick line, markers coloured on the
      same scale as the map, CI band), with each metric's mean signed
      contribution as thin lines in METRIC_COLORS. Bins run from conifer (left)
      to deciduous (right), matching the dominant-metric figure.

Companion to the dominant-metric map script: same plots, ecoregion assignment,
MIN_PLOTS_PER_ECOREGION, eastern limit and CONUS filter, so the maps line up.

Input: per-observation effects from the joint (mutually adjusted) model, long
format -- one row per plot x diversity metric with columns
    PID, diversity_metric, adjusted_effect, lon, lat
adjusted_effect = theta_j * (D~_j - mean D~_j), oriented so that positive =
stabilising (REPORT_AS_STABILITY = True in the analysis script).
Panel (b) also needs a PID -> percent_conifer lookup (CONIFER_LOOKUP_PATH).

Steps
  1. For every plot, total diversity effect = sum over metrics of the SIGNED
     adjusted_effect, i.e. the model's combined diversity contribution to
     stability for that plot.
  2. Assign plots to WWF Ecoregions2017 polygons (point-in-polygon; plots just
     outside are snapped to the nearest ecoregion within NEAREST_MAX_M).
  3. Within each ecoregion take the mean total effect and convert it to a %
     change in stability: 100 * (exp(mean) - 1). Colour = that % on a symmetric
     diverging scale.
  4. Ecoregions whose CI_LEVEL CI of the mean (mean +/- z SE across plots)
     includes 0 are stippled (SHOW_UNCERTAIN).
  5. Panel (b): the same total and metric contributions, averaged over plots in
     CONIFER_BIN_WIDTH-wide % conifer bins (bins with < MIN_PLOTS_PER_BIN plots
     are left out).

How to read it
  * The effects are centred, so the total is RELATIVE to a plot with nationally
    average (residualised) diversity. A stabilising ecoregion is one whose plots
    are, on average, more diverse in the stabilising direction than the
    national average -- not one where diversity "works" differently. The
    per-plot effects use one pooled set of theta_j.
  * In (b) the metric contributions are converted to % one by one, so they add
    up to the total only approximately (exactly on the log scale).
  * The CIs reflect spread among plots only: they ignore uncertainty in theta_j
    and spatial autocorrelation, so they are optimistic.
  * Ecoregions with fewer than MIN_PLOTS_PER_ECOREGION plots are hatched;
    ecoregions with no plots are grey.

Outputs: <OUT_STEM>.png/.pdf (600 dpi PNG and vector PDF), <OUT_STEM>_summary.csv
(per ecoregion) and <OUT_STEM>_conifer_trend.csv (per % conifer bin).
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import font_manager
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator
from shapely.geometry import box
from scipy.stats import norm as _norm

# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------
EFFECTS_CSV = "results/mutually_adjusted_effects_per_observation_with_coords.csv"
ECOREGIONS_PATH = "data/Ecoregions/Ecoregions2017.shp"
ECO_NAME_COL = "ECO_NAME"
STATES_PATH = None            # e.g. "data/states/cb_2018_us_state_20m.shp"

EFFECT_SIGN = 1               # 1 if adjusted_effect is already "+ = stabilising"; -1 to flip

# Per-metric colours (sliding-window and conditional-effect figures)
METRIC_COLORS = {
    "Rao's Q": "#4B2BC4",
    "Functional Evenness": "#A47CC7",
    "Species richness": "#14B8A0",
}

# Diverging colours chosen to sit apart from the metric hues (indigo 253°, purple 272°,
# teal 171°): burnt orange for destabilising, deep petrol blue for stabilising, white at 0.
# Orange vs blue also stays distinguishable with red-green colour blindness.
DESTAB_COLOR = "#C0622F"      # burnt orange
STAB_COLOR = "#1F5F7A"        # deep petrol blue
NEUTRAL_COLOR = "#F7F5F2"     # warm white at zero effect
TOTAL_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "destab_stab", [DESTAB_COLOR, "#E6B48F", NEUTRAL_COLOR, "#8DB3C7", STAB_COLOR])

# Map colouring on a signed log scale (symlog): small effects get more colour contrast,
# large ones are compressed. Only the map colours, the colour bar and the panel-(b) markers
# use it; all reported values and the panel-(b) y-axis stay linear.
LOG_COLOR_SCALE = True
LOG_LINTHRESH = None          # |% change| below this is coloured linearly; None = automatic

SHOW_UNCERTAIN = False        # stipple ecoregions whose CI of the mean includes 0
UNCERTAIN_HATCH = "..."
UNCERTAIN_HATCH_COLOR = "#4d4d4d"

CI_LEVEL = 0.90                                # e.g. 0.90, 0.95, 0.99
Z_CI = _norm.ppf(1 - (1 - CI_LEVEL) / 2)       # 1.645 for 90%, 1.960 for 95%
CI_PCT = round(CI_LEVEL * 100)

MIN_PLOTS_PER_ECOREGION = 20  # keep the same as the dominant-metric map
NEAREST_MAX_M = 5000          # snap unmatched plots to an ecoregion this close (m); None = don't
MAP_CRS = "EPSG:5070"         # CONUS Albers equal-area
SIMPLIFY_M = 500              # simplify polygons for DRAWING only (joins use full detail)
FIG_WIDTH_IN = 7.0            # ~178 mm, double-column width
DPI = 600
OUT_STEM = "results/total_diversity_effect_ecoregions"

# Eastern edge of the map: nothing is drawn east of this ecoregion (None = no limit)
EAST_LIMIT_ECOREGION = "New England-Acadian forests"

# Keep only contiguous-US plots (drops Hawaii and Alaska)
CONUS_LON = (-130.0, -60.0)
CONUS_LAT = (23.0, 50.0)

# Trend panel (b): total effect along the % conifer gradient
SHOW_CONIFER_TREND = True
CONIFER_LOOKUP_PATH = "data/lookup/PID_location_v3.csv"   # PID -> percent_conifer
CONIFER_COL = "percent_conifer"
CONIFER_BIN_WIDTH = 10        # % conifer per bin
MIN_PLOTS_PER_BIN = 20
SHOW_METRIC_CONTRIBUTIONS = True   # thin lines: each metric's signed contribution


def _to_pct(change):
    """Change on the (log) stability scale -> % change in stability."""
    return (np.exp(change) - 1) * 100


# ---------------------------------------------------------------
# Data
# ---------------------------------------------------------------
def load_plot_totals(csv_path):
    """One row per plot: lon/lat, the signed effect of every metric, and their sum ('total')."""
    long = pd.read_csv(csv_path, usecols=["PID", "diversity_metric", "adjusted_effect", "lon", "lat"])
    wide = EFFECT_SIGN * long.pivot(index="PID", columns="diversity_metric", values="adjusted_effect")

    n_all = len(wide)
    wide = wide.dropna()  # the total needs every metric
    if len(wide) < n_all:
        print(f"[load] dropped {n_all - len(wide)} plots missing an effect for one or more metrics")
    metrics = list(wide.columns)
    wide["total"] = wide[metrics].sum(axis=1)

    coords = long.drop_duplicates("PID").set_index("PID")[["lon", "lat"]]
    plots = coords.loc[wide.index].join(wide).reset_index()

    conus = plots.lon.between(*CONUS_LON) & plots.lat.between(*CONUS_LAT)
    if (~conus).any():
        print(f"[load] dropped {int((~conus).sum())} plots outside the contiguous US (Hawaii/Alaska)")
    plots = plots[conus].reset_index(drop=True)
    print(f"[load] {len(plots)} plots, total effect summed over: {metrics}")
    return plots, metrics


def attach_conifer(plots, lookup_path=CONIFER_LOOKUP_PATH, col=CONIFER_COL):
    """Adds % conifer (0-100) to each plot from the PID lookup."""
    if col not in plots.columns:
        lk = pd.read_csv(lookup_path, usecols=["PID", col]).drop_duplicates("PID")
        plots = plots.merge(lk, on="PID", how="left")
    if plots[col].max() <= 1:
        plots[col] = plots[col] * 100          # 0-1 fraction -> %
    print(f"[conifer] {int(plots[col].isna().sum())} plots without % conifer")
    return plots


def read_ecoregions(path, plots, pad_deg=2.0):
    """Read only the ecoregions near the plots (the full shapefile is global and large)."""
    file_crs = gpd.read_file(path, rows=1).crs
    study_box = gpd.GeoSeries(
        [box(plots.lon.min() - pad_deg, plots.lat.min() - pad_deg,
             plots.lon.max() + pad_deg, plots.lat.max() + pad_deg)], crs="EPSG:4326"
    ).to_crs(file_crs)
    eco = gpd.read_file(path, bbox=tuple(study_box.total_bounds))
    eco = eco.to_crs(MAP_CRS).reset_index(drop=True)
    eco["geometry"] = eco.geometry.make_valid()
    eco["eco_id"] = np.arange(len(eco))
    print(f"[ecoregions] {len(eco)} polygons intersect the study area")
    return eco


def _union(geoseries):
    return geoseries.union_all() if hasattr(geoseries, "union_all") else geoseries.unary_union


def clip_east_of(eco_draw, name=EAST_LIMIT_ECOREGION, name_col=ECO_NAME_COL):
    """Removes everything east of the eastern edge of the named ecoregion."""
    if not name or name_col not in eco_draw:
        return eco_draw
    ref = eco_draw[eco_draw[name_col].str.casefold() == name.casefold()]
    if ref.empty:
        print(f"[map] WARNING: '{name}' not found in {name_col}; no eastern limit applied")
        return eco_draw
    east_x = ref.total_bounds[2]
    minx, miny, _, maxy = eco_draw.total_bounds
    out = gpd.clip(eco_draw, box(minx - 1, miny - 1, east_x, maxy + 1))
    out = out[~out.geometry.is_empty]
    print(f"[map] eastern limit = edge of {name}: "
          f"{len(eco_draw) - len(out)} ecoregions further east removed (others trimmed at the edge)")
    return out


def load_us_outline(states_path):
    """Contiguous-US polygon in MAP_CRS (drops AK, HI, territories and non-US rows)."""
    s = gpd.read_file(states_path)
    if "iso_a2" in s.columns:
        s = s[s["iso_a2"] == "US"]
    elif "admin" in s.columns:
        s = s[s["admin"].str.contains("United States", na=False)]
    s = s.to_crs("EPSG:4326")
    c = s.geometry.representative_point()
    s = s[(c.x > -130) & (c.x < -60) & (c.y > 23) & (c.y < 50)]
    return _union(s.to_crs(MAP_CRS).geometry)


def assign_ecoregions(plots, eco, nearest_max_m=NEAREST_MAX_M):
    pts = gpd.GeoDataFrame(plots, geometry=gpd.points_from_xy(plots.lon, plots.lat),
                           crs="EPSG:4326").to_crs(eco.crs)
    ids = eco[["eco_id", "geometry"]]

    j = gpd.sjoin(pts, ids, how="left", predicate="within").drop(columns="index_right")
    j = j[~j.index.duplicated(keep="first")]
    missing = j["eco_id"].isna()
    n_direct = int((~missing).sum())

    n_snapped = 0
    if missing.any() and nearest_max_m:
        near = gpd.sjoin_nearest(pts.loc[missing[missing].index], ids, how="left",
                                 max_distance=nearest_max_m)
        near = near[~near.index.duplicated(keep="first")].dropna(subset=["eco_id"])
        j.loc[near.index, "eco_id"] = near["eco_id"].values
        n_snapped = len(near)

    n_lost = int(j["eco_id"].isna().sum())
    print(f"[join] {n_direct} plots inside an ecoregion, {n_snapped} snapped to the nearest "
          f"(<= {nearest_max_m} m), {n_lost} unassigned and dropped")
    return j.dropna(subset=["eco_id"])


def summarize_total(joined, eco, metrics, min_plots=MIN_PLOTS_PER_ECOREGION):
    """Per ecoregion: mean total effect, its SE and CI, % change, and metric contributions."""
    grp = joined.groupby("eco_id")
    stats_df = pd.DataFrame({
        "n_plots": grp.size(),
        "mean_total": grp["total"].mean(),
        "sd_total": grp["total"].std(ddof=1),
    })
    contrib = grp[metrics].mean().add_prefix("contrib__")   # signed, so they add up to mean_total
    eco = eco.merge(pd.concat([stats_df, contrib], axis=1), left_on="eco_id",
                    right_index=True, how="left")
    eco["n_plots"] = eco["n_plots"].fillna(0).astype(int)

    eco["se_total"] = eco["sd_total"] / np.sqrt(eco["n_plots"].where(eco["n_plots"] > 0))
    eco["ci_lower"] = eco["mean_total"] - Z_CI * eco["se_total"]
    eco["ci_upper"] = eco["mean_total"] + Z_CI * eco["se_total"]
    eco["pct"] = _to_pct(eco["mean_total"])
    eco["pct_ci_lower"] = _to_pct(eco["ci_lower"])
    eco["pct_ci_upper"] = _to_pct(eco["ci_upper"])
    eco["direction"] = np.where(eco["mean_total"] > 0, "stabilising",
                                np.where(eco["mean_total"] < 0, "destabilising", None))

    eco["no_plots"] = eco["n_plots"] == 0
    eco["low_n"] = (eco["n_plots"] > 0) & (eco["n_plots"] < min_plots)
    eco["uncertain"] = (~eco["no_plots"] & ~eco["low_n"]
                        & (eco["ci_lower"] <= 0) & (eco["ci_upper"] >= 0))

    eco["area_km2"] = eco.geometry.area / 1e6
    eco["plots_per_10k_km2"] = eco["n_plots"] / eco["area_km2"] * 1e4
    return eco


def conifer_trend(plots, metrics, width=CONIFER_BIN_WIDTH, min_plots=MIN_PLOTS_PER_BIN):
    """Per % conifer bin: mean total effect (+ CI) and each metric's mean contribution, in %."""
    d = plots.dropna(subset=[CONIFER_COL])
    edges = np.arange(0, 100 + width, width)
    d = d.assign(bin=pd.cut(d[CONIFER_COL], edges, include_lowest=True, labels=False))
    g = d.groupby("bin")
    t = pd.DataFrame({"n_plots": g.size(), "mean_total": g["total"].mean(),
                      "se_total": g["total"].std(ddof=1) / np.sqrt(g.size())})
    t["mid"] = edges[t.index.astype(int)] + width / 2
    t["pct"] = _to_pct(t["mean_total"])
    t["pct_ci_lower"] = _to_pct(t["mean_total"] - Z_CI * t["se_total"])
    t["pct_ci_upper"] = _to_pct(t["mean_total"] + Z_CI * t["se_total"])
    for m in metrics:
        t[f"pct__{m}"] = _to_pct(g[m].mean())
    dropped = int((t["n_plots"] < min_plots).sum())
    if dropped:
        print(f"[trend] {dropped} % conifer bins with < {min_plots} plots left out")
    return t[t["n_plots"] >= min_plots].reset_index(drop=True)


# ---------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------
def _pick_font():
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"):
        if name in have:
            return name
    return "sans-serif"


def _set_style():
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": [_pick_font()],
        "pdf.fonttype": 42, "ps.fonttype": 42, "hatch.linewidth": 0.3,
    })


def _scale_bar(ax, xmax, ymin, width, height, length_m=500_000):
    x1, y = xmax - 0.04 * width, ymin + 0.055 * height
    x0 = x1 - length_m
    ax.plot([x0, x1], [y, y], color="black", lw=1.0, solid_capstyle="butt", zorder=10)
    for x in (x0, x1):
        ax.plot([x, x], [y - 0.008 * height, y + 0.008 * height], color="black", lw=1.0, zorder=10)
    ax.text((x0 + x1) / 2, y + 0.018 * height, f"{length_m // 1000:d} km",
            ha="center", va="bottom", fontsize=7, zorder=10)


def _map_extent(eco, us_poly):
    xmin, ymin, xmax, ymax = us_poly.bounds if us_poly is not None else eco.total_bounds
    pad = 0.01 * (xmax - xmin)
    return xmin - pad, xmax + pad, ymin - pad, ymax + pad


def _new_figure(extent, legend_in, trend_in=0.0):
    """Map on top, optional trend strip (trend_in inches) and a legend strip (legend_in) beneath."""
    xmin, xmax, ymin, ymax = extent
    map_h_in = FIG_WIDTH_IN * (ymax - ymin) / (xmax - xmin)
    fig_h_in = map_h_in + trend_in + legend_in
    fig = plt.figure(figsize=(FIG_WIDTH_IN, fig_h_in))
    ax = fig.add_axes([0, (legend_in + trend_in) / fig_h_in, 1, map_h_in / fig_h_in])
    return fig, ax, fig_h_in


def _save(fig, out_stem):
    os.makedirs(os.path.dirname(out_stem) or ".", exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_stem}.{ext}", dpi=DPI, bbox_inches=None, facecolor="white")
    print(f"[figure] saved {out_stem}.png / .pdf")


def _pct_fmt(v, _=None):
    return "0" if abs(v) < 1e-12 else f"{v:+g}".replace("-", "−")


# ---------------------------------------------------------------
# Trend panel (b)
# ---------------------------------------------------------------
def draw_conifer_trend(ax, trend, metrics, norm):
    """Total effect along the % conifer gradient, conifer (left) -> deciduous (right)."""
    ax.axhline(0, color="#777777", lw=0.7, ls=(0, (3, 2)), zorder=1)
    metrics = [m for m in METRIC_COLORS if m in metrics] + [m for m in metrics if m not in METRIC_COLORS]

    if SHOW_METRIC_CONTRIBUTIONS:
        for m in metrics:
            ax.plot(trend["mid"], trend[f"pct__{m}"], color=METRIC_COLORS.get(m, "#888888"),
                    lw=1.1, marker="o", ms=2.6, alpha=0.9, zorder=2)

    ax.fill_between(trend["mid"], trend["pct_ci_lower"], trend["pct_ci_upper"],
                    color="#222222", alpha=0.12, lw=0, zorder=2)
    ax.plot(trend["mid"], trend["pct"], color="#222222", lw=1.8, zorder=3)
    ax.scatter(trend["mid"], trend["pct"], c=[TOTAL_CMAP(norm(v)) for v in trend["pct"]],
               s=34, edgecolors="#222222", linewidths=0.7, zorder=4)

    ax.set_xlim(100, 0)                              # conifer on the left, as in the other figure
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(10))
    ax.yaxis.set_major_locator(MaxNLocator(4, symmetric=True))
    ax.yaxis.set_major_formatter(FuncFormatter(_pct_fmt))
    ax.set_xlabel("% conifer  →  less conifer", fontsize=7.5, labelpad=2)
    ax.set_ylabel("Diversity effect on\nstability (% change)", fontsize=7.5, linespacing=1.15)
    ax.tick_params(labelsize=7, length=2.5, width=0.5, pad=1.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_linewidth(0.6)

    handles = [Line2D([], [], color="#222222", lw=1.8, marker="o", ms=5, mfc="white",
                      mec="#222222", label="Total (marker = map colour)"),
               Patch(facecolor="#222222", alpha=0.12, edgecolor="none", label=f"{CI_PCT}% CI")]
    if SHOW_METRIC_CONTRIBUTIONS:
        handles += [Line2D([], [], color=METRIC_COLORS.get(m, "#888888"), lw=1.1, marker="o", ms=3,
                           label=m) for m in metrics]
    ax.legend(handles=handles, fontsize=6.5, frameon=False, ncol=1,
              loc="upper left", bbox_to_anchor=(1.01, 1.0), handlelength=1.8, borderaxespad=0)


def _nice_up(x):
    """Smallest 1, 2 or 5 x 10^k that is >= x."""
    k = np.floor(np.log10(x))
    for m in (1, 2, 5, 10):
        if m * 10 ** k >= x * (1 - 1e-9):
            return m * 10 ** k


def _color_norm(vals):
    """Colour normalisation and colour-bar ticks for the map values (in % change)."""
    vmax_raw = float(vals.abs().max())
    if not LOG_COLOR_SCALE:
        ticks = MaxNLocator(4, symmetric=True, steps=[1, 2, 2.5, 5, 10]).tick_values(-vmax_raw, vmax_raw)
        return mcolors.Normalize(-ticks[-1], ticks[-1]), ticks

    vmax = _nice_up(vmax_raw)
    lt = LOG_LINTHRESH
    if lt is None:
        # Decade below the 10th percentile of |effect|: most ecoregions fall in the log part
        nz = vals.abs()[vals.abs() > 0]
        q = np.percentile(nz, 10) if len(nz) else vmax / 100
        lt = 10 ** np.floor(np.log10(q))
    lt = min(lt, vmax / 10)                    # keep at least one decade of log range
    norm = mcolors.SymLogNorm(linthresh=lt, linscale=0.5, vmin=-vmax, vmax=vmax, base=10)

    decades = 10.0 ** np.arange(np.ceil(np.log10(lt)), np.floor(np.log10(vmax)) + 1)
    pos = sorted(set(np.round(np.r_[decades, vmax], 12)))
    if len(pos) > 3:                           # avoid a crowded colour bar
        pos = pos[-3:]
    ticks = np.r_[-np.array(pos[::-1]), 0.0, pos]
    print(f"[map] signed-log colour scale: linear within ±{lt:g}%, range ±{vmax:g}%")
    return norm, ticks


# ---------------------------------------------------------------
# Map: total diversity effect
# ---------------------------------------------------------------
def draw_total_map(eco, us_poly, min_plots=MIN_PLOTS_PER_ECOREGION, out_stem=OUT_STEM,
                   trend=None, metrics=None):
    _set_style()
    colored = eco[~eco.no_plots & ~eco.low_n].copy()
    vals = colored["pct"].dropna()
    if vals.empty:
        raise ValueError(
            f"No ecoregion has >= {min_plots} plots, so there is nothing to colour "
            f"({int(eco.low_n.sum())} ecoregions have 1-{min_plots - 1} plots). Lower "
            f"MIN_PLOTS_PER_ECOREGION, or check the [load]/[join] lines above.")

    # Symmetric scale around 0 (white = no net effect): linear or signed-log
    norm, ticks = _color_norm(vals)

    has_trend = trend is not None and len(trend) > 0
    trend_in = 2.0 if has_trend else 0.0
    extent = _map_extent(eco, us_poly)
    xmin, xmax, ymin, ymax = extent
    fig, ax, H = _new_figure(extent, legend_in=0.80, trend_in=trend_in)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_axis_off()

    if eco.no_plots.any():
        eco[eco.no_plots].plot(ax=ax, facecolor="#e6e6e6", edgecolor="#9a9a9a", linewidth=0.2)
    if eco.low_n.any():
        eco[eco.low_n].plot(ax=ax, facecolor="#f7f7f7", edgecolor="#808080",
                            linewidth=0.2, hatch="//////")
    colored.plot(ax=ax, color=[TOTAL_CMAP(norm(v)) for v in colored["pct"]],
                 edgecolor="#5a5a5a", linewidth=0.2)
    if SHOW_UNCERTAIN and eco.uncertain.any():
        eco[eco.uncertain].plot(ax=ax, facecolor="none", edgecolor=UNCERTAIN_HATCH_COLOR,
                                hatch=UNCERTAIN_HATCH, linewidth=0)
    if us_poly is not None:
        gpd.GeoSeries([us_poly], crs=eco.crs).boundary.plot(ax=ax, color="black", linewidth=0.5)
    _scale_bar(ax, xmax, ymin, xmax - xmin, ymax - ymin)

    if has_trend:
        fig.text(0.01, 1 - 0.05 / H, "a", fontsize=11, fontweight="bold", va="top")
        tax = fig.add_axes([0.14, (0.80 + 0.45) / H, 0.60, (trend_in - 0.65) / H])
        draw_conifer_trend(tax, trend, metrics or [], norm)
        fig.text(0.01, (0.80 + trend_in - 0.05) / H, "b", fontsize=11, fontweight="bold", va="top")

    # Diverging colour bar
    cax = fig.add_axes([0.22, 0.24 / H, 0.42, 0.10 / H])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=TOTAL_CMAP), cax=cax, orientation="horizontal")
    cb.set_ticks(ticks)
    cb.ax.xaxis.set_major_formatter(FuncFormatter(_pct_fmt))
    cb.ax.tick_params(labelsize=7, length=2, width=0.4, pad=1.5)
    cb.outline.set_linewidth(0.4)
    cax.xaxis.set_label_position("top")
    scale_note = ", log scale" if LOG_COLOR_SCALE else ""
    cax.set_xlabel("Total diversity effect on stability (% change vs. a plot with\n"
                   f"average diversity; mean of plots in ecoregion{scale_note})",
                   fontsize=7.5, labelpad=3, linespacing=1.2)
    cax.text(-0.02, 0.5, "Destabilising", transform=cax.transAxes, ha="right", va="center", fontsize=7)
    cax.text(1.02, 0.5, "Stabilising", transform=cax.transAxes, ha="left", va="center", fontsize=7)

    # Status key
    handles = []
    if SHOW_UNCERTAIN:
        handles.append(Patch(facecolor="white", edgecolor=UNCERTAIN_HATCH_COLOR, hatch=UNCERTAIN_HATCH,
                             linewidth=0.4, label=f"{CI_PCT}% CI includes 0"))
    handles.append(Patch(facecolor="#f7f7f7", edgecolor="#808080", hatch="//////", linewidth=0.4,
                         label=f"< {min_plots} plots"))
    if eco.no_plots.any():
        handles.append(Patch(facecolor="#e6e6e6", edgecolor="#9a9a9a", linewidth=0.4, label="No plots"))
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.80, 0.10 / H), frameon=False,
               fontsize=7, handlelength=1.6, handleheight=1.0, borderaxespad=0)

    _save(fig, out_stem)
    return fig


# ---------------------------------------------------------------
# Run
# ---------------------------------------------------------------
def run(effects_csv=EFFECTS_CSV, ecoregions_path=ECOREGIONS_PATH, states_path=STATES_PATH,
        min_plots=MIN_PLOTS_PER_ECOREGION, out_stem=OUT_STEM):
    plots, metrics = load_plot_totals(effects_csv)
    eco = read_ecoregions(ecoregions_path, plots)
    joined = assign_ecoregions(plots, eco)
    eco = summarize_total(joined, eco, metrics, min_plots)

    ok = eco[~eco.no_plots & ~eco.low_n]
    print(f"[result] {len(ok)} ecoregions with >= {min_plots} plots: "
          f"{int((ok.direction == 'stabilising').sum())} stabilising, "
          f"{int((ok.direction == 'destabilising').sum())} destabilising, "
          f"{int(ok.uncertain.sum())} with a {CI_PCT}% CI that includes 0")
    print(f"[result] national mean total effect: {_to_pct(plots['total'].mean()):+.3g}% "
          f"(~0 by construction: the effects are centred)")

    trend = None
    if SHOW_CONIFER_TREND:
        # Same plots as the map: only those assigned to an ecoregion
        trend_plots = attach_conifer(pd.DataFrame(joined.drop(columns="geometry")))
        trend = conifer_trend(trend_plots, metrics)
        trend.to_csv(f"{out_stem}_conifer_trend.csv", index=False)
        print(f"[trend] saved {out_stem}_conifer_trend.csv")

    # Simplify for drawing only (assignment above used the full-detail polygons)
    eco_draw = eco.copy()
    eco_draw["geometry"] = eco.geometry.simplify(SIMPLIFY_M, preserve_topology=True)
    eco_draw = clip_east_of(eco_draw)

    us_poly = None
    if states_path:
        us_poly = load_us_outline(states_path)
        eco_draw = gpd.clip(eco_draw, us_poly)
        eco_draw = eco_draw[~eco_draw.geometry.is_empty]
    else:
        print("[map] no STATES_PATH: showing only ecoregions that contain plots, unclipped "
              "(border-straddling ecoregions will extend beyond the US)")
        eco_draw = eco_draw[~eco_draw.no_plots]

    fig = draw_total_map(eco_draw, us_poly, min_plots, out_stem, trend=trend, metrics=metrics)

    keep = [c for c in (ECO_NAME_COL,) if c in eco.columns] + \
           ["n_plots", "plots_per_10k_km2", "mean_total", "se_total", "ci_lower", "ci_upper",
            "pct", "pct_ci_lower", "pct_ci_upper", "direction", "uncertain", "low_n"] + \
           [f"contrib__{m}" for m in metrics]
    keep = [c for c in keep if c in eco.columns]
    eco.loc[~eco.no_plots, keep].sort_values("pct", ascending=False).to_csv(
        f"{out_stem}_summary.csv", index=False)
    print(f"[summary] saved {out_stem}_summary.csv")
    return eco, fig


if __name__ == "__main__":
    run()
    plt.show()