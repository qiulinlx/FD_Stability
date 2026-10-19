"""
% conifer by ecoregion (contiguous US; Hawaii and Alaska removed; nothing drawn
east of the New England-Acadian forests) and diversity scatter plots split by
ecoregion conifer class.
  (a) Map: mean % conifer of the plots in each ecoregion, brown (deciduous) ->
      green (conifer). The conifer-class breaks are marked on the colour bar.
  (b) Scatter plots: species richness vs Rao's Q and vs Functional evenness,
      one column per ecoregion conifer class (deciduous / mixed / conifer).

Companion script: ecoregion_effect_map.py (dominant-metric map). By default this
script takes its plots and coordinates from the same effects CSV, so both use
the same plots, ecoregion assignment and MIN_PLOTS_PER_ECOREGION.

Inputs
  * PLOTS_CSV: PID + lon/lat for every plot (default: the effects CSV; only plots
    with an effect for every metric are kept, matching ecoregion_effect_map.py)
  * CONIFER_LOOKUP_PATH: PID -> percent_conifer
  * DIVERSITY_DATA_PATH: PID + plot-level species richness, Rao's Q, FEve

Caveats
  * Ecoregions with fewer than MIN_PLOTS_PER_ECOREGION plots are hatched and get
    no conifer class, so their plots are left out of the scatter plots;
    ecoregions with no plots are grey.
  * The class is the ECOREGION's mean % conifer, so a deciduous plot in a
    conifer ecoregion is shown under "Conifer".

Outputs: <OUT_STEM_CONIFER>.png/.pdf (map), <OUT_STEM_SCATTER>.png/.pdf
(scatter plots), 600 dpi PNG and vector PDF, plus <OUT_STEM_CONIFER>_summary.csv.
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import font_manager
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import MaxNLocator
from scipy import stats
from scipy.ndimage import gaussian_filter
from shapely.geometry import box

# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------
PLOTS_CSV = "results/mutually_adjusted_effects_per_observation_with_coords.csv"
LON_COL, LAT_COL = "lon", "lat"
ECOREGIONS_PATH = "data/Ecoregions/Ecoregions2017.shp"
ECO_NAME_COL = "ECO_NAME"
STATES_PATH = None            # e.g. "data/states/cb_2018_us_state_20m.shp"

# % conifer
CONIFER_LOOKUP_PATH = "data/lookup/PID_location_v3.csv"   # PID -> percent_conifer
CONIFER_COL = "percent_conifer"
# Brown (deciduous) -> green (conifer); the pale middle stays distinct from the grey "no plots"
CONIFER_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "deciduous_conifer", ["#8c510a", "#d8b365", "#f6e8c3", "#a6dba0", "#1b7837"])

MIN_PLOTS_PER_ECOREGION = 20  # keep the same in ecoregion_effect_map.py
NEAREST_MAX_M = 5000          # snap unmatched plots to an ecoregion this close (m); None = don't
MAP_CRS = "EPSG:5070"         # CONUS Albers equal-area
SIMPLIFY_M = 500              # simplify polygons for DRAWING only (joins use full detail)
FIG_WIDTH_IN = 7.0            # ~178 mm, double-column width
DPI = 600
OUT_STEM_CONIFER = "results/percent_conifer_ecoregions"

# Eastern edge of the map: nothing is drawn east of this ecoregion (None = no limit)
EAST_LIMIT_ECOREGION = "New England-Acadian forests"

# Keep only contiguous-US plots (drops Hawaii and Alaska)
CONUS_LON = (-130.0, -60.0)
CONUS_LAT = (23.0, 50.0)

# Scatter plots: species richness vs Rao's Q / FEve, one column per conifer class
DIVERSITY_DATA_PATH = "data/final/final_dataset_ba.csv"   # PID + plot-level diversity values
SCATTER_Y = ("Species Richness", "Species richness")       # (column, axis label)
SCATTER_X = [("Raos_Q", "Rao’s Q"),                        # one row of panels per entry
             ("Functional_Evenness", "Functional evenness")]
CONIFER_CLASS_BREAKS = [0, 33.3, 66.7, 100]   # ecoregion mean % conifer; e.g. [0, 50, 100] for two classes
CONIFER_CLASS_NAMES = ["Deciduous", "Mixed", "Conifer"]
CONIFER_CLASS_COLORS = ["#8c510a", "#8f8f2f", "#1b7837"]   # brown, olive, green (matches the map)
SCATTER_JITTER = 0.35         # SD of vertical jitter for integer richness (display only)
SCATTER_N_BINS = 20           # bins for the binned-mean line
SCATTER_SHOW_ALL = True       # dashed grey line = binned mean of all classes, for comparison
OUT_STEM_SCATTER = "results/SR_scatter_by_conifer_class"


# ---------------------------------------------------------------
# Data
# ---------------------------------------------------------------
def load_plots(csv_path=PLOTS_CSV):
    """One row per plot: PID, lon, lat (contiguous US only)."""
    head = pd.read_csv(csv_path, nrows=0).columns
    if {"diversity_metric", "adjusted_effect"} <= set(head):
        # Effects CSV: keep plots with an effect for every metric, as in ecoregion_effect_map.py
        long = pd.read_csv(csv_path, usecols=["PID", "diversity_metric", "adjusted_effect",
                                              LON_COL, LAT_COL])
        complete = (long.pivot(index="PID", columns="diversity_metric", values="adjusted_effect")
                    .dropna().index)
        plots = long[long.PID.isin(complete)].drop_duplicates("PID")[["PID", LON_COL, LAT_COL]]
    else:
        plots = pd.read_csv(csv_path, usecols=["PID", LON_COL, LAT_COL]).drop_duplicates("PID")
    plots = plots.rename(columns={LON_COL: "lon", LAT_COL: "lat"}).dropna()

    conus = plots.lon.between(*CONUS_LON) & plots.lat.between(*CONUS_LAT)
    if (~conus).any():
        print(f"[load] dropped {int((~conus).sum())} plots outside the contiguous US (Hawaii/Alaska)")
    plots = plots[conus].reset_index(drop=True)
    print(f"[load] {len(plots)} plots")
    return plots


def attach_conifer(plots, lookup_path=CONIFER_LOOKUP_PATH, col=CONIFER_COL):
    """Adds % conifer to each plot from the PID lookup (if not already in the effects CSV)."""
    if col not in plots.columns:
        lk = pd.read_csv(lookup_path, usecols=["PID", col]).drop_duplicates("PID")
        plots = plots.merge(lk, on="PID", how="left")
    c = plots[col]
    print(f"[conifer] {int(c.isna().sum())} plots without % conifer; "
          f"range {c.min():.0f}-{c.max():.0f}%")
    if c.max() - c.min() < 50:
        print("[conifer] WARNING: narrow % conifer range -- were these effects fitted on a "
              "subset (e.g. deciduous only)?")
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


def summarize(joined, eco, min_plots=MIN_PLOTS_PER_ECOREGION):
    """Per ecoregion: number of plots and mean % conifer."""
    grp = joined.groupby("eco_id")
    stats_ = pd.concat([grp.size().rename("n_plots"),
                        grp[CONIFER_COL].mean().rename("mean_conifer")], axis=1)
    eco = eco.merge(stats_, left_on="eco_id", right_index=True, how="left")
    eco["n_plots"] = eco["n_plots"].fillna(0).astype(int)
    eco["no_plots"] = eco["n_plots"] == 0
    eco["low_n"] = (eco["n_plots"] > 0) & (eco["n_plots"] < min_plots)
    return eco


def classify_conifer(eco):
    """Deciduous / mixed / conifer class per ecoregion from its mean % conifer.
    Ecoregions with no plots or < MIN_PLOTS_PER_ECOREGION plots get no class."""
    if "mean_conifer" not in eco:
        eco["conifer_class"] = pd.NA
        print("[classes] no mean_conifer column: conifer classes skipped")
        return eco
    usable = ~eco["no_plots"] & ~eco["low_n"]
    eco["conifer_class"] = pd.cut(eco["mean_conifer"].where(usable), CONIFER_CLASS_BREAKS,
                                  labels=CONIFER_CLASS_NAMES, include_lowest=True)
    counts = eco.loc[usable].groupby("conifer_class", observed=False).agg(
        ecoregions=("eco_id", "size"), plots=("n_plots", "sum"))
    print("[classes] ecoregions and plots per conifer class:\n" + counts.to_string())
    return eco


# ---------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------
def _pick_font():
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"):
        if name in have:
            return name
    return "sans-serif"


def _scale_bar(ax, xmax, ymin, width, height, length_m=500_000):
    x1, y = xmax - 0.04 * width, ymin + 0.055 * height
    x0 = x1 - length_m
    ax.plot([x0, x1], [y, y], color="black", lw=1.0, solid_capstyle="butt", zorder=10)
    for x in (x0, x1):
        ax.plot([x, x], [y - 0.008 * height, y + 0.008 * height], color="black", lw=1.0, zorder=10)
    ax.text((x0 + x1) / 2, y + 0.018 * height, f"{length_m // 1000:d} km",
            ha="center", va="bottom", fontsize=7, zorder=10)


def _setup_map_ax(ax, eco, extent):
    """Shared frame plus the grey (no plots) and hatched (low n) ecoregions."""
    xmin, xmax, ymin, ymax = extent
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_axis_off()
    if eco.no_plots.any():
        eco[eco.no_plots].plot(ax=ax, facecolor="#e6e6e6", edgecolor="#9a9a9a", linewidth=0.2)
    if eco.low_n.any():
        eco[eco.low_n].plot(ax=ax, facecolor="#f7f7f7", edgecolor="#808080",
                            linewidth=0.2, hatch="//////")


def _draw_outline(ax, us_poly, crs):
    if us_poly is not None:
        gpd.GeoSeries([us_poly], crs=crs).boundary.plot(ax=ax, color="black", linewidth=0.5)


def _map_extent(eco, us_poly):
    xmin, ymin, xmax, ymax = us_poly.bounds if us_poly is not None else eco.total_bounds
    pad = 0.01 * (xmax - xmin)
    return xmin - pad, xmax + pad, ymin - pad, ymax + pad


def _new_map_figure(extent, legend_in):
    """Figure with the map on top and a legend strip of legend_in inches beneath it."""
    xmin, xmax, ymin, ymax = extent
    map_h_in = FIG_WIDTH_IN * (ymax - ymin) / (xmax - xmin)
    fig_h_in = map_h_in + legend_in
    fig = plt.figure(figsize=(FIG_WIDTH_IN, fig_h_in))
    ax = fig.add_axes([0, legend_in / fig_h_in, 1, map_h_in / fig_h_in])
    return fig, ax, fig_h_in


def _save(fig, out_stem):
    os.makedirs(os.path.dirname(out_stem) or ".", exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_stem}.{ext}", dpi=DPI, bbox_inches=None, facecolor="white")
    print(f"[figure] saved {out_stem}.png / .pdf")


def _set_style():
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": [_pick_font()],
        "pdf.fonttype": 42, "ps.fonttype": 42, "hatch.linewidth": 0.3,
    })


def _status_key(fig, eco, min_plots, anchor):
    handles = []
    handles.append(Patch(facecolor="#f7f7f7", edgecolor="#808080", hatch="//////", linewidth=0.4,
                         label=f"< {min_plots} plots"))
    if eco.no_plots.any():
        handles.append(Patch(facecolor="#e6e6e6", edgecolor="#9a9a9a", linewidth=0.4, label="No plots"))
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=anchor, frameon=False,
               fontsize=7, handlelength=1.6, handleheight=1.0, borderaxespad=0)


# ---------------------------------------------------------------
# Map: mean % conifer, brown (deciduous) -> green (conifer)
# ---------------------------------------------------------------
def draw_conifer_map(eco, us_poly, min_plots=MIN_PLOTS_PER_ECOREGION, out_stem=OUT_STEM_CONIFER):
    _set_style()
    extent = _map_extent(eco, us_poly)
    xmin, xmax, ymin, ymax = extent
    fig, ax, H = _new_map_figure(extent, legend_in=0.70)

    _setup_map_ax(ax, eco, extent)
    cnorm = mcolors.Normalize(0, 100)
    con = eco[~eco.no_plots & ~eco.low_n]
    if "mean_conifer" in con:
        con = con[con["mean_conifer"].notna()]
        if len(con):
            con.plot(ax=ax, color=[CONIFER_CMAP(cnorm(v)) for v in con["mean_conifer"]],
                     edgecolor="#5a5a5a", linewidth=0.2)
    _draw_outline(ax, us_poly, eco.crs)
    _scale_bar(ax, xmax, ymin, xmax - xmin, ymax - ymin)

    cax = fig.add_axes([0.25, 0.22 / H, 0.40, 0.10 / H])
    cb = fig.colorbar(ScalarMappable(norm=cnorm, cmap=CONIFER_CMAP), cax=cax, orientation="horizontal")
    cb.set_ticks([0, 25, 50, 75, 100])
    # Class breaks used for the scatter plots (c), marked on the colour bar
    for b in CONIFER_CLASS_BREAKS[1:-1]:
        cax.axvline(b, color="black", lw=0.8)
    cb.ax.tick_params(labelsize=7, length=2, width=0.4, pad=1.5)
    cb.outline.set_linewidth(0.4)
    cax.xaxis.set_label_position("top")
    cax.set_xlabel("Mean % conifer of plots in ecoregion", fontsize=7.5, labelpad=3)
    cax.text(-0.02, 0.5, "Deciduous", transform=cax.transAxes, ha="right", va="center", fontsize=7)
    cax.text(1.02, 0.5, "Conifer", transform=cax.transAxes, ha="left", va="center", fontsize=7)

    _status_key(fig, eco, min_plots, (0.80, 0.12 / H))
    _save(fig, out_stem)
    return fig


# ---------------------------------------------------------------
# Scatter plots: species richness vs Rao's Q / FEve by conifer class
# ---------------------------------------------------------------
def _spearman_ci(x, y, alpha=0.05):
    """Spearman rho with Fisher-z confidence interval (Fieller et al. 1957 SE)."""
    rho, p = stats.spearmanr(x, y)
    n = len(x)
    se = 1.06 / np.sqrt(n - 3)
    zc = stats.norm.ppf(1 - alpha / 2)
    lo, hi = np.tanh([np.arctanh(rho) - zc * se, np.arctanh(rho) + zc * se])
    return rho, lo, hi, p, n


def _is_discrete(v, max_levels=60):
    return np.allclose(v, np.round(v)) and len(np.unique(v)) <= max_levels


def _density_pct(x, y, bins=100, smooth=(3, 5)):
    """Per-point density percentile from a smoothed 2-D histogram (fast for large n)."""
    h, xe, ye = np.histogram2d(x, y, bins=bins)
    h = gaussian_filter(h, smooth)
    xi = np.clip(np.searchsorted(xe, x) - 1, 0, bins - 1)
    yi = np.clip(np.searchsorted(ye, y) - 1, 0, bins - 1)
    return stats.rankdata(h[xi, yi]) / len(x)


def _binned_means(x, y, n_bins=SCATTER_N_BINS):
    """Equal-count bins along x: mean x, mean y and 95% CI of y per bin."""
    n_bins = int(min(n_bins, max(3, len(x) // 50)))
    edges = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
    idx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)
    m = (pd.DataFrame({"x": x, "y": y, "b": idx}).groupby("b")
         .agg(xm=("x", "mean"), ym=("y", "mean"), sd=("y", "std"), n=("y", "size")))
    m["ci"] = 1.96 * m["sd"].fillna(0) / np.sqrt(m["n"])
    return m


def _class_cmap(color):
    """Light tint -> class colour -> darker shade, for density-coloured points."""
    c = np.array(mcolors.to_rgb(color))
    return mcolors.LinearSegmentedColormap.from_list(
        f"cls_{color}", [1 - 0.30 * (1 - c), c, 0.55 * c])


def _class_titles():
    b, out = CONIFER_CLASS_BREAKS, []
    for i, name in enumerate(CONIFER_CLASS_NAMES):
        lo, hi = b[i], b[i + 1]
        if i == 0:
            rng_txt = f"< {hi:.0f}% conifer"
        elif i == len(CONIFER_CLASS_NAMES) - 1:
            rng_txt = f"> {lo:.0f}% conifer"
        else:
            rng_txt = f"{lo:.0f}–{hi:.0f}% conifer"
        out.append(f"{name} ({rng_txt})")
    return out


def load_diversity_by_class(joined, eco):
    """Plot-level diversity values for every plot in a classed ecoregion."""
    ycol = SCATTER_Y[0]
    xcols = [c for c, _ in SCATTER_X]
    div = (pd.read_csv(DIVERSITY_DATA_PATH, usecols=["PID", ycol] + xcols)
           .drop_duplicates("PID"))
    cls = eco[["eco_id", "conifer_class"]].dropna()
    d = (pd.DataFrame(joined[["PID", "eco_id"]])
         .merge(cls, on="eco_id")
         .merge(div, on="PID"))
    n0 = len(d)
    d = d.dropna(subset=[ycol] + xcols)          # complete cases, so every panel uses the same plots
    print(f"[scatter] {len(d):,} plots with diversity data in classed ecoregions "
          f"({n0 - len(d):,} dropped for missing values)")
    return d


def draw_scatter_by_class(joined, eco, out_stem=OUT_STEM_SCATTER):
    """Rows = x variables (Rao's Q, FEve); columns = ecoregion conifer classes."""
    _set_style()
    if eco["conifer_class"].isna().all():
        print("[scatter] no classed ecoregions: scatter plots skipped")
        return None
    d = load_diversity_by_class(joined, eco)
    ycol, ylabel = SCATTER_Y
    titles = dict(zip(CONIFER_CLASS_NAMES, _class_titles()))
    colors = dict(zip(CONIFER_CLASS_NAMES, CONIFER_CLASS_COLORS))
    classes = [c for c in CONIFER_CLASS_NAMES if (d["conifer_class"] == c).any()]
    if not classes:
        print("[scatter] no plots left after merging diversity data: scatter plots skipped")
        return None

    rng = np.random.default_rng(42)
    y_all = d[ycol].to_numpy(float)
    jitter = SCATTER_JITTER if (SCATTER_JITTER and _is_discrete(y_all)) else 0
    d["_y_plot"] = y_all + (rng.normal(0, jitter, len(d)) if jitter else 0)

    n_rows, n_cols = len(SCATTER_X), len(classes)
    rc = {"font.size": 7, "axes.labelsize": 8, "axes.linewidth": 0.6,
          "xtick.major.width": 0.6, "ytick.major.width": 0.6,
          "xtick.labelsize": 7, "ytick.labelsize": 7,
          "axes.spines.top": False, "axes.spines.right": False}
    with plt.rc_context(rc):
        fig, axes = plt.subplots(n_rows, n_cols, squeeze=False, sharey=True, sharex="row",
                                 figsize=(FIG_WIDTH_IN, 2.2 * n_rows + 0.5),
                                 constrained_layout=True)
        letters = iter("abcdefghijklmnopqrstuvwxyz")
        for r, (xcol, xlabel) in enumerate(SCATTER_X):
            ref = _binned_means(d[xcol].to_numpy(float), y_all) if SCATTER_SHOW_ALL else None
            for c, cls in enumerate(classes):
                ax = axes[r, c]
                sub = d[d["conifer_class"] == cls]
                x = sub[xcol].to_numpy(float)
                y = sub[ycol].to_numpy(float)
                yj = sub["_y_plot"].to_numpy(float)

                dens = _density_pct(x, yj)
                o = np.argsort(dens)                     # densest points drawn on top
                ax.scatter(x[o], yj[o], c=dens[o], cmap=_class_cmap(colors[cls]), vmin=0, vmax=1,
                           s=4, linewidths=0, alpha=0.8, rasterized=True)

                if r == 0:
                    k = int(eco.loc[eco["conifer_class"] == cls, "eco_id"].nunique())
                    ax.set_title(f"{titles[cls]}\n{k} ecoregions", fontsize=7.5,
                                 color=colors[cls], fontweight="bold", linespacing=1.3)
                ax.set_xlabel(xlabel)
                if c == 0:
                    ax.set_ylabel(ylabel)
                ax.margins(x=0.02)
                ax.tick_params(length=2.5, pad=1.5)

        if jitter:
            axes[0, 0].yaxis.set_major_locator(MaxNLocator(integer=True))
        # Cap y at the 99.9th percentile so a few outliers don't squash the panels
        axes[0, 0].set_ylim(min(0, np.nanmin(y_all)) - 0.8, np.nanpercentile(y_all, 99.9) + 1)

        handles = [Line2D([], [], color="black", lw=1.3, marker="o", ms=2,
                          label="Binned mean ± 95% CI (this class)")]
        if SCATTER_SHOW_ALL:
            handles.append(Line2D([], [], color="0.3", lw=1.0, ls=(0, (3, 2)),
                                  label="Binned mean, all classes"))
        try:
            fig.legend(handles=handles, loc="outside lower center", ncol=len(handles),
                       frameon=False, fontsize=7)
        except ValueError:                          # matplotlib < 3.7
            fig.legend(handles=handles, loc="lower center", ncol=len(handles),
                       frameon=False, fontsize=7)

    _save(fig, out_stem)
    return fig


# ---------------------------------------------------------------
# Run
# ---------------------------------------------------------------
def run(plots_csv=PLOTS_CSV, ecoregions_path=ECOREGIONS_PATH, states_path=STATES_PATH,
        min_plots=MIN_PLOTS_PER_ECOREGION, out_stem_conifer=OUT_STEM_CONIFER,
        out_stem_scatter=OUT_STEM_SCATTER):
    plots = load_plots(plots_csv)
    plots = attach_conifer(plots)                 # must come before assign_ecoregions
    eco = read_ecoregions(ecoregions_path, plots)
    joined = assign_ecoregions(plots, eco)
    eco = summarize(joined, eco, min_plots)
    eco = classify_conifer(eco)

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

    fig_map = draw_conifer_map(eco_draw, us_poly, min_plots, out_stem_conifer)
    # Scatter plots use the plot -> ecoregion assignment (joined), not the clipped drawing layer
    fig_scatter = draw_scatter_by_class(joined, eco, out_stem_scatter)

    keep = [c for c in (ECO_NAME_COL,) if c in eco.columns] + \
           ["n_plots", "mean_conifer", "conifer_class", "low_n"]
    keep = [c for c in keep if c in eco.columns]
    out_csv = f"{out_stem_conifer}_summary.csv"
    eco.loc[~eco.no_plots, keep].sort_values("mean_conifer", ascending=False).to_csv(out_csv, index=False)
    print(f"[summary] saved {out_csv}")
    return eco, fig_map, fig_scatter


if __name__ == "__main__":
    run()
    plt.show()