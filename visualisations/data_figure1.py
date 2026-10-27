"""
Two figures along the conifer / rarity gradients (same plots, same colours):

  FIGURE 1  <OUT_STEM>_gradients.png / .pdf
    (a-c)  Left column: Rao's Q, Functional evenness and Species richness along the
           % conifer gradient (conifer left -> deciduous right): plots as density-shaded
           points, median per bin (line) and interquartile range (band). With
           SHOW_STANDARDISED an extra panel shows all metrics as median vs. overall median (SD).
    (d-f)  Right column (SHOW_RARE): the same three metrics along % rare species,
           kernel-smoothed median and IQR, rows aligned with a-c.

  FIGURE 2  <OUT_STEM>_map_scatter.png / .pdf
    (a)    Mean % conifer of the plots in each ecoregion (contiguous US, nothing east of the
           New England-Acadian forests), conifer-class breaks on the colour bar; with SHOW_RARE
           each ecoregion also gets a circle for its mean % rare species. Keys to the right.
    (b)    One row: species richness vs Rao's Q (three conifer classes), then vs Functional
           evenness (three conifer classes); classes = ecoregion mean % conifer, as in the map.

ONE set of plots is used throughout: plots in PLOTS_CSV (default: the effects CSV,
keeping plots with an effect for every metric) inside the contiguous US, with % conifer
and every diversity value, and assigned to an ecoregion. The scatter plots use the subset in
ecoregions with >= MIN_PLOTS_PER_ECOREGION plots (those get a conifer class).

The conifer-class colours are taken from CONIFER_CMAP at each class's midpoint (deepened
if too pale to read), so the scatter columns always match the map.

Outputs: the two figures (600 dpi PNG + vector PDF), <OUT_STEM>_bins.csv
(per metric x % conifer bin), <OUT_STEM>_rare_smooth.csv and <OUT_STEM>_ecoregions.csv.
"""

import colorsys
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import font_manager
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, MultipleLocator
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
CONIFER_LOOKUP_PATH = "data/lookup/PID_location_v3.csv"
CONIFER_COL = "percent_conifer"
DIVERSITY_DATA_PATH = "data/final/final_dataset_ba.csv"

# Diversity columns -> (label, colour), in panel order (left column)
METRICS = {
    "Raos_Q": ("Rao’s Q", "#4B2BC4"),
    "Functional_Evenness": ("Functional evenness", "#A47CC7"),
    "Species Richness": ("Species richness", "#14B8A0"),
}
# Rare species: extra left-column panel + circles on the map (raw %, not the residual)
RARE_PATH = "data/final/dataset_rarity.parquet"
RARE_COL = "pct_rare_species"
RARE_LABEL = "Rare species (%)"
SHOW_RARE = True              # False = figure as before, without rarity
# Map symbols: ecoregions grouped into classes of mean % rare; each class gets a bigger AND
# darker circle (double encoding), with a dark outline so it reads on both green and brown.
RARE_CLASS_BREAKS = None      # None = quartiles of the ecoregion means; or e.g. [0, 5, 10, 20, 100]
RARE_CMAP = mcolors.LinearSegmentedColormap.from_list("rare", ["#F7D6E6", "#D9619B", "#8E1B5C", "#540035"])
RARE_SIZES_PT2 = [8, 20, 38, 62]   # marker area per class (low -> high); one per class
RARE_EDGE = "#2A0A1C"
MAP_FADE = 0.30               # with rarity shown, map fill is mixed this much toward white so dots stand out
# Panel f: % rare species is a ratio of small counts (1/3, 1/2, 2/3 ...), so plots pile up on a few
# exact values with very different richness. Bins then zig-zag; a kernel smooth does not.
RARE_BANDWIDTH = 8.0          # Gaussian kernel SD (percentage points) for the running median / IQR
RARE_GRID_N = 60              # points along % rare at which the smooth is evaluated
RARE_X_JITTER = 2.0           # horizontal jitter (pp) for the points in f, display only
RARE_MIN_RICHNESS = None      # e.g. 5: panel f only uses plots with >= this many species (see note)

SCATTER_Y = "Species Richness"                         # y in panel f
SCATTER_X = ["Raos_Q", "Functional_Evenness"]          # one row of panel f per entry

# Map colours (0% -> 100% conifer). Class colours below are sampled from this.
CONIFER_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "conifer_deciduous",  ["#1b7837", "#3a8f8a", "#6b73e5", "#9a6a7a", "#8c510a"])
CONIFER_CLASS_BREAKS = [0, 33.3, 66.7, 100]
CONIFER_CLASS_NAMES = ["Deciduous", "Mixed", "Conifer"]
CONIFER_CLASS_COLORS = None   # None = sample CONIFER_CMAP; or a list, e.g. ["#389440", "#8f8f2f", "#8c510a"]

# Left column
CONIFER_BIN_WIDTH = 10
MIN_PLOTS_PER_BIN = 20
MAX_POINTS = 40000            # points drawn per panel (all plots used for the summaries)
JITTER = 0.35                 # SD of vertical jitter for integer metrics (display only)
CLIP_Y_PCT = (0.5, 99.5)      # y-limits from these percentiles

# Map / scatter
MIN_PLOTS_PER_ECOREGION = 20
NEAREST_MAX_M = 5000
MAP_CRS = "EPSG:5070"
SIMPLIFY_M = 500
EAST_LIMIT_ECOREGION = "New England-Acadian forests"
CONUS_LON = (-130.0, -60.0)
CONUS_LAT = (23.0, 50.0)
DENSITY_SMOOTH = (3, 5)       # Gaussian blur of point density (x, y grid cells)

SHOW_STANDARDISED = False     # True = extra left panel: all metrics as median vs. overall median (SD)

FIG_WIDTH_IN = 7.2
DPI = 600
OUT_STEM = "results/figures/conifer_gradient_figure"


def panel_vars():
    """Variables shown in the left column (and the standardised panel), in order."""
    return dict(METRICS)


# ---------------------------------------------------------------
# Data
# ---------------------------------------------------------------
def load_plots():
    """PID, lon, lat, % conifer and every diversity value; contiguous US; complete cases."""
    head = pd.read_csv(PLOTS_CSV, nrows=0).columns
    if {"diversity_metric", "adjusted_effect"} <= set(head):
        long = pd.read_csv(PLOTS_CSV, usecols=["PID", "diversity_metric", "adjusted_effect",
                                               LON_COL, LAT_COL])
        complete = long.pivot(index="PID", columns="diversity_metric", values="adjusted_effect").dropna().index
        plots = long[long.PID.isin(complete)].drop_duplicates("PID")[["PID", LON_COL, LAT_COL]]
    else:
        plots = pd.read_csv(PLOTS_CSV, usecols=["PID", LON_COL, LAT_COL]).drop_duplicates("PID")
    plots = plots.rename(columns={LON_COL: "lon", LAT_COL: "lat"})
    plots = plots[plots.lon.between(*CONUS_LON) & plots.lat.between(*CONUS_LAT)]

    lk = pd.read_csv(CONIFER_LOOKUP_PATH, usecols=["PID", CONIFER_COL]).drop_duplicates("PID")
    div = pd.read_csv(DIVERSITY_DATA_PATH, usecols=["PID"] + list(METRICS)).drop_duplicates("PID")
    plots = plots.merge(lk, on="PID", how="left").merge(div, on="PID", how="left")
    if SHOW_RARE:
        rare = pd.read_parquet(RARE_PATH, columns=["PID", RARE_COL]).drop_duplicates("PID")
        plots = plots.merge(rare, on="PID", how="left")
    n0 = len(plots)
    plots = plots.dropna(subset=["lon", "lat", CONIFER_COL] + list(METRICS)).reset_index(drop=True)
    if plots[CONIFER_COL].max() <= 1:
        plots[CONIFER_COL] = plots[CONIFER_COL] * 100
    if SHOW_RARE:
        # Rarity is NOT a complete-case requirement, so the plot set is the same as without it
        if plots[RARE_COL].max() <= 1:
            plots[RARE_COL] = plots[RARE_COL] * 100
        print(f"[rare] {int(plots[RARE_COL].isna().sum()):,} plots without {RARE_COL} "
              f"(left out of the rarity panel and circles only)")
    print(f"[load] {len(plots):,} plots with coordinates, % conifer and all metrics "
          f"({n0 - len(plots):,} dropped)")
    return plots


def read_ecoregions(path, plots, pad_deg=2.0):
    file_crs = gpd.read_file(path, rows=1).crs
    study_box = gpd.GeoSeries(
        [box(plots.lon.min() - pad_deg, plots.lat.min() - pad_deg,
             plots.lon.max() + pad_deg, plots.lat.max() + pad_deg)], crs="EPSG:4326").to_crs(file_crs)
    eco = gpd.read_file(path, bbox=tuple(study_box.total_bounds)).to_crs(MAP_CRS).reset_index(drop=True)
    eco["geometry"] = eco.geometry.make_valid()
    eco["eco_id"] = np.arange(len(eco))
    return eco


def _union(gs):
    return gs.union_all() if hasattr(gs, "union_all") else gs.unary_union


def clip_east_of(eco_draw, name=EAST_LIMIT_ECOREGION, name_col=ECO_NAME_COL):
    if not name or name_col not in eco_draw:
        return eco_draw
    ref = eco_draw[eco_draw[name_col].str.casefold() == name.casefold()]
    if ref.empty:
        print(f"[map] WARNING: '{name}' not found in {name_col}; no eastern limit applied")
        return eco_draw
    minx, miny, _, maxy = eco_draw.total_bounds
    out = gpd.clip(eco_draw, box(minx - 1, miny - 1, ref.total_bounds[2], maxy + 1))
    return out[~out.geometry.is_empty]


def load_us_outline(states_path):
    s = gpd.read_file(states_path)
    if "iso_a2" in s.columns:
        s = s[s["iso_a2"] == "US"]
    elif "admin" in s.columns:
        s = s[s["admin"].str.contains("United States", na=False)]
    s = s.to_crs("EPSG:4326")
    c = s.geometry.representative_point()
    s = s[(c.x > -130) & (c.x < -60) & (c.y > 23) & (c.y < 50)]
    return _union(s.to_crs(MAP_CRS).geometry)


def assign_ecoregions(plots, eco):
    pts = gpd.GeoDataFrame(plots, geometry=gpd.points_from_xy(plots.lon, plots.lat),
                           crs="EPSG:4326").to_crs(eco.crs)
    ids = eco[["eco_id", "geometry"]]
    j = gpd.sjoin(pts, ids, how="left", predicate="within").drop(columns="index_right")
    j = j[~j.index.duplicated(keep="first")]
    missing = j["eco_id"].isna()
    if missing.any() and NEAREST_MAX_M:
        near = gpd.sjoin_nearest(pts.loc[missing[missing].index], ids, how="left",
                                 max_distance=NEAREST_MAX_M)
        near = near[~near.index.duplicated(keep="first")].dropna(subset=["eco_id"])
        j.loc[near.index, "eco_id"] = near["eco_id"].values
    n_lost = int(j["eco_id"].isna().sum())
    print(f"[join] {len(j) - n_lost:,} plots assigned to an ecoregion, {n_lost:,} dropped")
    return pd.DataFrame(j.dropna(subset=["eco_id"]).drop(columns="geometry")).reset_index(drop=True)


def summarize_ecoregions(joined, eco):
    grp = joined.groupby("eco_id")
    parts = [grp.size().rename("n_plots"), grp[CONIFER_COL].mean().rename("mean_conifer")]
    if SHOW_RARE:
        parts.append(grp[RARE_COL].mean().rename("mean_rare"))
    s = pd.concat(parts, axis=1)
    eco = eco.merge(s, left_on="eco_id", right_index=True, how="left")
    eco["n_plots"] = eco["n_plots"].fillna(0).astype(int)
    eco["no_plots"] = eco["n_plots"] == 0
    eco["low_n"] = (eco["n_plots"] > 0) & (eco["n_plots"] < MIN_PLOTS_PER_ECOREGION)
    usable = ~eco["no_plots"] & ~eco["low_n"]
    eco["conifer_class"] = pd.cut(eco["mean_conifer"].where(usable), CONIFER_CLASS_BREAKS,
                                  labels=CONIFER_CLASS_NAMES, include_lowest=True)
    return eco


def bin_summary(df):
    edges = np.arange(0, 100 + CONIFER_BIN_WIDTH, CONIFER_BIN_WIDTH)
    b = pd.cut(df[CONIFER_COL], edges, include_lowest=True, labels=False)
    rows = []
    for col in panel_vars():
        x = df[col].dropna()
        # z of the bin median relative to the OVERALL median, so skewed variables (e.g. % rare)
        # are not offset by the gap between their mean and median
        mu, sd = x.median(), x.std(ddof=1)
        for k, v in x.groupby(b.loc[x.index]):
            lo, hi = edges[int(k)], edges[int(k) + 1]
            rows.append({"metric": col, "lo": lo, "hi": hi, "mid": (lo + hi) / 2, "n": len(v),
                         "median": v.median(), "q25": v.quantile(0.25), "q75": v.quantile(0.75),
                         "mean": v.mean(), "z_median": (v.median() - mu) / sd})
    out = pd.DataFrame(rows)
    out["enough"] = out["n"] >= MIN_PLOTS_PER_BIN
    return out


def _weighted_quantiles(y, w, qs):
    o = np.argsort(y)
    cw = np.cumsum(w[o])
    cw /= cw[-1]
    return np.interp(qs, cw, y[o])


def _rare_plots(df):
    d = df.dropna(subset=[RARE_COL])
    if RARE_MIN_RICHNESS:
        d = d[d[SCATTER_Y] >= RARE_MIN_RICHNESS]
    return d


def rare_bin_summary(df):
    """Kernel-smoothed median and IQR of each metric along % rare species.
    At each grid value g, every plot gets weight exp(-0.5 ((x - g) / RARE_BANDWIDTH)^2); the
    weighted median / quartiles are reported where the effective n >= MIN_PLOTS_PER_BIN."""
    d = _rare_plots(df)
    x = d[RARE_COL].to_numpy(float)
    grid = np.linspace(0, np.percentile(x, 99), RARE_GRID_N)
    # Integer metrics (richness): spread each value over [v - 0.5, v + 0.5] so the quantiles move
    # smoothly instead of jumping between whole numbers (mid-distribution quantiles)
    rs = np.random.default_rng(0)
    ys = {}
    for col in METRICS:
        y = d[col].to_numpy(float)
        ys[col] = y + rs.uniform(-0.5, 0.5, len(y)) if _is_discrete(y) else y
    rows = []
    for g in grid:
        w = np.exp(-0.5 * ((x - g) / RARE_BANDWIDTH) ** 2)
        keep = w > 1e-4
        wk = w[keep]
        n_eff = wk.sum() ** 2 / (wk ** 2).sum()
        for col in METRICS:
            y = ys[col][keep]
            q25, med, q75 = _weighted_quantiles(y, wk, [0.25, 0.5, 0.75])
            rows.append({"metric": col, "rare_mid": g, "n_eff": n_eff,
                         "median": med, "q25": q25, "q75": q75})
    out = pd.DataFrame(rows)
    out["enough"] = out["n_eff"] >= MIN_PLOTS_PER_BIN
    return out


# ---------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------
def _pick_font():
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"):
        if name in have:
            return name
    return "sans-serif"


def _style(ax, size=6):
    for name, sp in ax.spines.items():
        sp.set_visible(name in ("left", "bottom"))
        sp.set_linewidth(0.5)
    ax.tick_params(labelsize=size, length=2, width=0.4, pad=1.5)


def _is_discrete(v, max_levels=60):
    return np.allclose(v, np.round(v)) and len(np.unique(v)) <= max_levels


def _density_pct(x, y, bins=100, smooth=DENSITY_SMOOTH):
    h, xe, ye = np.histogram2d(x, y, bins=bins)
    h = gaussian_filter(h, smooth)
    xi = np.clip(np.searchsorted(xe, x) - 1, 0, bins - 1)
    yi = np.clip(np.searchsorted(ye, y) - 1, 0, bins - 1)
    return stats.rankdata(h[xi, yi]) / len(x)


def class_colors():
    """Class colour = CONIFER_CMAP at the class midpoint, darkened until readable on white."""
    if CONIFER_CLASS_COLORS:
        return dict(zip(CONIFER_CLASS_NAMES, CONIFER_CLASS_COLORS))
    b = CONIFER_CLASS_BREAKS
    out = []
    for i in range(len(CONIFER_CLASS_NAMES)):
        c = mcolors.to_rgb(CONIFER_CMAP(((b[i] + b[i + 1]) / 2) / 100))
        h, l, s = colorsys.rgb_to_hls(*c)
        # Keep the hue, but deepen pale colours (e.g. the cream middle of the map) so they read
        out.append(mcolors.to_hex(colorsys.hls_to_rgb(h, min(l, 0.40), max(s, 0.45))))
    return dict(zip(CONIFER_CLASS_NAMES, out))


def map_cmap():
    """Conifer colour map as drawn on the map: faded toward white when rarity dots are shown."""
    if not SHOW_RARE or not MAP_FADE:
        return CONIFER_CMAP
    c = CONIFER_CMAP(np.linspace(0, 1, 256))[:, :3]
    return mcolors.ListedColormap(1 - (1 - MAP_FADE) * (1 - c), name="conifer_faded")


def _tint_cmap(color, light=0.15, dark=0.55):
    c = np.array(mcolors.to_rgb(color))
    return mcolors.LinearSegmentedColormap.from_list(f"t_{color}", [1 - light * (1 - c), 1 - dark * (1 - c)])


def _class_cmap(color):
    c = np.array(mcolors.to_rgb(color))
    return mcolors.LinearSegmentedColormap.from_list(f"cls_{color}", [1 - 0.30 * (1 - c), c, 0.55 * c])


def _class_title(i):
    b, name = CONIFER_CLASS_BREAKS, CONIFER_CLASS_NAMES[i]
    if i == 0:
        rng = f"< {b[1]:.0f}%"
    elif i == len(CONIFER_CLASS_NAMES) - 1:
        rng = f"> {b[i]:.0f}%"
    else:
        rng = f"{b[i]:.0f}–{b[i + 1]:.0f}%"
    return f"{name} ({rng} conifer)"


# ---------------------------------------------------------------
# Panels
# ---------------------------------------------------------------
def draw_left_column(fig, gs, df, summ, rng):
    """One panel per variable (+ the standardised panel if SHOW_STANDARDISED), sharing x."""
    pv = panel_vars()
    n_ax = len(pv) + int(SHOW_STANDARDISED)
    axes = [fig.add_subplot(gs[0])]
    axes += [fig.add_subplot(gs[i], sharex=axes[0]) for i in range(1, n_ax)]

    idx = rng.choice(len(df), size=min(MAX_POINTS, len(df)), replace=False)
    x_pts = df[CONIFER_COL].to_numpy(float)[idx]
    for ax, (col, (label, color)) in zip(axes, pv.items()):
        s = summ[(summ.metric == col) & summ.enough]
        y_all = df[col].to_numpy(float)
        y = y_all[idx]
        ok = np.isfinite(y)
        xp, y = x_pts[ok], y[ok]
        y_all = y_all[np.isfinite(y_all)]
        if JITTER and _is_discrete(y_all):
            y = y + rng.normal(0, JITTER, len(y))
        d = _density_pct(xp, y, smooth=(2, 4))
        o = np.argsort(d)
        ax.scatter(xp[o], y[o], c=d[o], cmap=_tint_cmap(color), s=0.4, linewidths=0,
                   rasterized=True, zorder=1)
        ax.fill_between(s["mid"], s["q25"], s["q75"], color=color, alpha=0.30, lw=0, zorder=2)
        ax.plot(s["mid"], s["median"], color="white", lw=2.3, zorder=3)
        ax.plot(s["mid"], s["median"], color=color, lw=1.2, marker="o", ms=2.5, mec="white", mew=0.4,
                zorder=4)
        lo, hi = np.percentile(y_all, CLIP_Y_PCT)
        pad = 0.05 * (hi - lo)
        ax.set_ylim(lo - pad, hi + pad)
        ax.yaxis.set_major_locator(MaxNLocator(3))
        ax.set_ylabel(label, fontsize=6.5)
        _style(ax)
        ax.tick_params(axis="x", labelbottom=False)

    ax = axes[-1]
    if SHOW_STANDARDISED:
        ax.axhline(0, color="#888888", lw=0.5, ls=(0, (3, 2)))
        for col, (label, color) in pv.items():
            s = summ[(summ.metric == col) & summ.enough]
            ax.plot(s["mid"], s["z_median"], color=color, lw=1.2, marker="o", ms=2.5, mec="white",
                    mew=0.4, label=label)
        ax.set_ylabel("Median vs. overall\nmedian (SD)", fontsize=6.5, linespacing=1.1)
        ax.yaxis.set_major_locator(MaxNLocator(4, symmetric=True))
        ax.legend(fontsize=5.3, frameon=False, loc="best", handlelength=1.3, borderaxespad=0.2)
        _style(ax)
    ax.tick_params(axis="x", labelbottom=True)
    ax.set_xlim(100, 0)
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.set_xlabel("% conifer  \u2192  less conifer", fontsize=6.5, labelpad=2)
    return axes


def draw_map(ax, eco, extent, us_poly):
    xmin, xmax, ymin, ymax = extent
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_axis_off()
    if eco.no_plots.any():
        eco[eco.no_plots].plot(ax=ax, facecolor="#e6e6e6", edgecolor="#9a9a9a", linewidth=0.15)
    if eco.low_n.any():
        eco[eco.low_n].plot(ax=ax, facecolor="#f7f7f7", edgecolor="#808080", linewidth=0.15,
                            hatch="//////")
    con = eco[~eco.no_plots & ~eco.low_n & eco["mean_conifer"].notna()]
    cmap = map_cmap()
    con.plot(ax=ax, color=[cmap(v / 100) for v in con["mean_conifer"]],
             edgecolor="#5a5a5a", linewidth=0.15)
    if us_poly is not None:
        gpd.GeoSeries([us_poly], crs=eco.crs).boundary.plot(ax=ax, color="black", linewidth=0.4)
    rare_handles = draw_rare_circles(ax, con) if SHOW_RARE else []
    # scale bar
    w, h = xmax - xmin, ymax - ymin
    x1, y = xmax - 0.04 * w, ymin + 0.06 * h
    ax.plot([x1 - 500_000, x1], [y, y], color="black", lw=0.8, solid_capstyle="butt")
    for x in (x1 - 500_000, x1):
        ax.plot([x, x], [y - 0.012 * h, y + 0.012 * h], color="black", lw=0.8)
    ax.text(x1 - 250_000, y + 0.025 * h, "500 km", ha="center", va="bottom", fontsize=5.5)
    return rare_handles


def rare_classes(values):
    """Class breaks for the ecoregion mean % rare (quartiles unless RARE_CLASS_BREAKS is set)."""
    if RARE_CLASS_BREAKS:
        return np.asarray(RARE_CLASS_BREAKS, float)
    q = np.quantile(values, np.linspace(0, 1, len(RARE_SIZES_PT2) + 1))
    return np.unique(np.round(q, 1))


def draw_rare_circles(ax, con):
    """Graduated symbols: ecoregion mean % rare species in classes, bigger and darker per class."""
    r = con[con["mean_rare"].notna()]
    if r.empty:
        return []
    br = rare_classes(r["mean_rare"])
    k = np.clip(np.searchsorted(br, r["mean_rare"].to_numpy(), side="right") - 1, 0, len(br) - 2)
    n_cls = len(br) - 1
    sizes = np.interp(np.arange(n_cls), np.linspace(0, n_cls - 1, len(RARE_SIZES_PT2)), RARE_SIZES_PT2)
    cols = [RARE_CMAP(t) for t in np.linspace(0.15, 1.0, n_cls)]
    pts = r.geometry.representative_point()
    order = np.argsort(-k)                                    # big circles first, small on top
    ax.scatter(pts.x.to_numpy()[order], pts.y.to_numpy()[order], s=sizes[k[order]],
               c=[cols[i] for i in k[order]], edgecolor=RARE_EDGE, linewidth=0.45, zorder=6)

    def lab(i):
        lo, hi = br[i], br[i + 1]
        return f"{lo:g}–{hi:g}"
    return [ax.scatter([], [], s=sizes[i], color=cols[i], edgecolor=RARE_EDGE, linewidth=0.45,
                       label=lab(i)) for i in range(n_cls)]


def draw_rarity_stack(fig, gs, df, rb, rng):
    """(rarity column) One panel per metric against % rare species, same size and style as a-c:
    density-shaded points, kernel-smoothed median (line) and IQR (band)."""
    d = _rare_plots(df)
    x_all = d[RARE_COL].to_numpy(float)
    x_hi = float(rb["rare_mid"].max())
    idx = rng.choice(len(d), size=min(MAX_POINTS, len(d)), replace=False)
    axes = []
    for i, (col, (label, color)) in enumerate(METRICS.items()):
        ax = fig.add_subplot(gs[i], sharex=axes[0] if axes else None)
        axes.append(ax)
        x, y = x_all[idx], d[col].to_numpy(float)[idx]
        if RARE_X_JITTER:                                  # break up the stripes at 1/3, 1/2, ...
            x = np.clip(x + rng.normal(0, RARE_X_JITTER, len(x)), 0, None)
        y_full = df[col].to_numpy(float)
        if JITTER and _is_discrete(y_full[np.isfinite(y_full)]):
            y = y + rng.normal(0, JITTER, len(y))
        keep = x <= x_hi
        dd = _density_pct(x[keep], y[keep], smooth=(2, 4))
        o = np.argsort(dd)
        ax.scatter(x[keep][o], y[keep][o], c=dd[o], cmap=_tint_cmap(color), s=0.4, linewidths=0,
                   rasterized=True, zorder=1)
        s = rb[(rb.metric == col) & rb.enough]
        ax.fill_between(s["rare_mid"], s["q25"], s["q75"], color=color, alpha=0.30, lw=0, zorder=2)
        ax.plot(s["rare_mid"], s["median"], color="white", lw=2.3, zorder=3)
        ax.plot(s["rare_mid"], s["median"], color=color, lw=1.2, zorder=4)
        lo, hi = np.nanpercentile(y_full, CLIP_Y_PCT)
        pad = 0.05 * (hi - lo)
        ax.set_ylim(lo - pad, hi + pad)
        ax.yaxis.set_major_locator(MaxNLocator(3))
        ax.set_ylabel(label, fontsize=6.5)
        _style(ax)
        if i < len(METRICS) - 1:
            ax.tick_params(axis="x", labelbottom=False)
    axes[-1].set_xlim(0, x_hi)
    axes[-1].xaxis.set_major_locator(MaxNLocator(4))
    axes[-1].set_xlabel("% rare species", fontsize=6.5, labelpad=2)
    return axes


def draw_colorbar(fig, rect, eco):
    cax = fig.add_axes(rect)
    cb = fig.colorbar(ScalarMappable(norm=mcolors.Normalize(0, 100), cmap=map_cmap()), cax=cax,
                      orientation="horizontal")
    cb.set_ticks([0, 25, 50, 75, 100])
    for b in CONIFER_CLASS_BREAKS[1:-1]:
        cax.axvline(b, color="black", lw=0.7)
    cb.ax.tick_params(labelsize=5.5, length=2, width=0.4, pad=1)
    cb.outline.set_linewidth(0.4)
    cax.xaxis.set_label_position("top")
    cax.set_xlabel("Mean % conifer in ecoregion (lines = class breaks)", fontsize=6, labelpad=2)
    cax.text(-0.02, 0.5, "Deciduous", transform=cax.transAxes, ha="right", va="center", fontsize=5.5)
    cax.text(1.02, 0.5, "Conifer", transform=cax.transAxes, ha="left", va="center", fontsize=5.5)


def draw_scatter_grid(fig, gs, joined, eco, rng, one_row=True):
    """Species richness vs each SCATTER_X, one column per ecoregion conifer class (f)."""
    cls = eco[["eco_id", "conifer_class"]].dropna()
    d = joined.merge(cls, on="eco_id")
    colors = class_colors()
    classes = [c for c in CONIFER_CLASS_NAMES if (d["conifer_class"] == c).any()]
    y_all = d[SCATTER_Y].to_numpy(float)
    yj_all = y_all + (rng.normal(0, JITTER, len(d)) if JITTER and _is_discrete(y_all) else 0)
    d = d.assign(_yj=yj_all)
    ylim = (min(0, np.nanmin(y_all)) - 0.8, np.nanpercentile(y_all, 99.9) + 1)

    axes = np.empty((len(SCATTER_X), len(classes)), dtype=object)
    for r, xcol in enumerate(SCATTER_X):
        xr = d[xcol].to_numpy(float)
        xlim = (np.nanmin(xr), np.nanmax(xr))
        pad = 0.03 * (xlim[1] - xlim[0])
        for c, name in enumerate(classes):
            # one row: block per x variable, separated by a narrow spacer column
            ax = fig.add_subplot(gs[0, r * (len(classes) + 1) + c] if one_row else gs[r, c])
            axes[r, c] = ax
            sub = d[d["conifer_class"] == name]
            x, y = sub[xcol].to_numpy(float), sub["_yj"].to_numpy(float)
            if len(x) > MAX_POINTS:
                k = rng.choice(len(x), MAX_POINTS, replace=False)
                x, y = x[k], y[k]
            dens = _density_pct(x, y)
            o = np.argsort(dens)
            ax.scatter(x[o], y[o], c=dens[o], cmap=_class_cmap(colors[name]), vmin=0, vmax=1,
                       s=1.0, linewidths=0, alpha=0.85, rasterized=True)
            rho = stats.spearmanr(sub[xcol], sub[SCATTER_Y])[0]
            ax.text(0.04, 0.96, f"ρ = {rho:.2f}".replace("-", "−"), transform=ax.transAxes,
                    fontsize=5, va="top", ha="left")
            ax.set_xlim(xlim[0] - pad, xlim[1] + pad)
            ax.set_ylim(*ylim)
            ax.xaxis.set_major_locator(MaxNLocator(3))
            ax.yaxis.set_major_locator(MaxNLocator(4, integer=True))
            _style(ax, size=5.5)
            ax.set_xlabel(METRICS[xcol][0], fontsize=6, labelpad=1.5)
            if c == 0:
                ax.set_ylabel(METRICS[SCATTER_Y][0], fontsize=6)
            else:
                ax.tick_params(axis="y", labelleft=False)
            if one_row and r > 0 and c == 0:
                ax.tick_params(axis="y", labelleft=True)
                ax.set_ylabel("")
            if r == 0 or one_row:
                k = int(eco.loc[eco["conifer_class"] == name, "eco_id"].nunique())
                i = CONIFER_CLASS_NAMES.index(name)
                ax.set_title(f"{_class_title(i)}\n{k} ecoregions, {len(sub):,} plots", fontsize=5.6,
                             color=colors[name], fontweight="bold", linespacing=1.2, pad=3)
    return axes


# ---------------------------------------------------------------
# Figure
# ---------------------------------------------------------------
def _save(fig, stem):
    os.makedirs(os.path.dirname(stem) or ".", exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=DPI, facecolor="white")
    print(f"[figure] saved {stem}.png / .pdf")


def _set_rc():
    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": [_pick_font()],
                         "pdf.fonttype": 42, "ps.fonttype": 42, "hatch.linewidth": 0.3})


def build_gradient_figure(df, summ, stem):
    """FIGURE 1: metrics along % conifer (left column) and along % rare species (right column),
    one panel per metric, rows aligned."""
    _set_rc()
    rng = np.random.default_rng(42)
    W = FIG_WIDTH_IN
    n_left = len(panel_vars()) + int(SHOW_STANDARDISED)
    panel_h_in = 1.15
    top_in = n_left * panel_h_in + (n_left - 1) * 0.15
    bottom_in = 0.40
    H = 0.10 + top_in + bottom_in
    fig = plt.figure(figsize=(W, H))
    top, bottom = 1 - 0.10 / H, bottom_in / H

    left_x = (0.09, 0.47) if SHOW_RARE else (0.09, 0.97)
    rare_x = (0.59, 0.97)
    gs_left = fig.add_gridspec(n_left, 1, left=left_x[0], right=left_x[1], top=top, bottom=bottom,
                               hspace=0.15 / panel_h_in)
    axes = list(draw_left_column(fig, gs_left, df, summ, rng))
    if SHOW_RARE:
        gs_rare = fig.add_gridspec(n_left, 1, left=rare_x[0], right=rare_x[1], top=top, bottom=bottom,
                                   hspace=0.15 / panel_h_in)
        axes += list(draw_rarity_stack(fig, [gs_rare[i] for i in range(len(METRICS))], df,
                                       rare_bin_summary(df), rng))

    letter = dict(fontsize=9, fontweight="bold", va="bottom", ha="left")
    for ax, L in zip(axes, "abcdefghijklmnop"):
        pos = ax.get_position()
        fig.text(pos.x0 - 0.075, pos.y1, L, **letter)
    _save(fig, stem)
    return fig


def build_map_scatter_figure(joined, eco, eco_draw, us_poly, stem):
    """FIGURE 2: map (a) with its keys to the right, scatter row (b) below."""
    _set_rc()
    rng = np.random.default_rng(42)
    xmin, ymin, xmax, ymax = us_poly.bounds if us_poly is not None else eco_draw.total_bounds
    p = 0.01 * (xmax - xmin)
    extent = (xmin - p, xmax + p, ymin - p, ymax + p)

    W = FIG_WIDTH_IN
    map_x = (0.06, 0.66)
    map_w_in = (map_x[1] - map_x[0]) * W
    map_h_in = map_w_in * (extent[3] - extent[2]) / (extent[1] - extent[0])
    gap_in, sc_title_in, sc_h_in, bottom_in = 0.15, 0.40, 1.25, 0.38
    H = 0.08 + map_h_in + gap_in + sc_title_in + sc_h_in + bottom_in
    fig = plt.figure(figsize=(W, H))
    map_top = 1 - 0.08 / H

    # Map
    map_ax = fig.add_axes([map_x[0], map_top - map_h_in / H, map_x[1] - map_x[0], map_h_in / H])
    rare_handles = draw_map(map_ax, eco_draw, extent, us_poly)

    # Keys to the right of the map, stacked and vertically centred on it
    kx0, kx1 = 0.72, 0.97
    kcx = (kx0 + kx1) / 2
    map_mid = map_top - map_h_in / (2 * H)
    block = 0.55 + (0.75 if SHOW_RARE and rare_handles else 0) + 0.25      # inches
    y = map_mid + block / (2 * H) - 0.30 / H
    draw_colorbar(fig, [kx0 + 0.035, y, kx1 - kx0 - 0.07, 0.07 / H], eco_draw)
    y -= 0.30 / H
    if SHOW_RARE and rare_handles:
        leg = fig.legend(handles=rare_handles, title=f"Mean {RARE_LABEL.lower()}\nin ecoregion (circles)",
                         loc="upper center", ncol=2, bbox_to_anchor=(kcx, y), frameon=False,
                         fontsize=5.5, title_fontsize=6, handletextpad=0.3, columnspacing=1.0,
                         labelspacing=0.9, borderaxespad=0, scatterpoints=1)
        fig.add_artist(leg)
        y -= 0.62 / H
    handles = [Patch(facecolor="#f7f7f7", edgecolor="#808080", hatch="//////", linewidth=0.4,
                     label=f"< {MIN_PLOTS_PER_ECOREGION} plots")]
    if eco_draw.no_plots.any():
        handles.append(Patch(facecolor="#e6e6e6", edgecolor="#9a9a9a", linewidth=0.4, label="No plots"))
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), bbox_to_anchor=(kcx, y),
               frameon=False, fontsize=5.5, handlelength=1.3, handleheight=0.8, borderaxespad=0)

    # Scatter row
    n_cls = len(CONIFER_CLASS_NAMES)
    ratios = []
    for r in range(len(SCATTER_X)):
        ratios += [1] * n_cls + ([0.28] if r < len(SCATTER_X) - 1 else [])
    sc_top = (bottom_in + sc_h_in) / H
    gs_sc = fig.add_gridspec(1, len(ratios), width_ratios=ratios, left=0.075, right=0.985,
                             top=sc_top, bottom=bottom_in / H, wspace=0.12)
    draw_scatter_grid(fig, gs_sc, joined, eco, rng, one_row=True)

    letter = dict(fontsize=9, fontweight="bold", va="top", ha="left")
    fig.text(0.008, map_top, "a", **letter)
    fig.text(0.008, sc_top + (sc_title_in + 0.05) / H, "b", **letter)
    _save(fig, stem)
    return fig


def run(out_stem=OUT_STEM):
    plots = load_plots()
    eco = read_ecoregions(ECOREGIONS_PATH, plots)
    joined = assign_ecoregions(plots, eco)          # the one set of plots used throughout
    eco = summarize_ecoregions(joined, eco)
    summ = bin_summary(joined)

    usable = ~eco.no_plots & ~eco.low_n
    print("[classes] ecoregions / plots per conifer class:\n"
          + eco[usable].groupby("conifer_class", observed=False)
          .agg(ecoregions=("eco_id", "size"), plots=("n_plots", "sum")).to_string())
    rho = {lab: stats.spearmanr(joined[CONIFER_COL], joined[c], nan_policy="omit")[0]
           for c, (lab, _) in panel_vars().items()}
    print("[result] Spearman rho with % conifer: " + ", ".join(f"{k} {v:+.2f}" for k, v in rho.items()))

    us_poly = load_us_outline(STATES_PATH) if STATES_PATH else None
    eco_draw = eco.copy()
    eco_draw["geometry"] = eco.geometry.simplify(SIMPLIFY_M, preserve_topology=True)
    eco_draw = clip_east_of(eco_draw)
    if us_poly is not None:
        eco_draw = gpd.clip(eco_draw, us_poly)
        eco_draw = eco_draw[~eco_draw.geometry.is_empty]
    else:
        eco_draw = eco_draw[~eco_draw.no_plots]

    fig_gradients = build_gradient_figure(joined, summ, f"{out_stem}_gradients")
    fig_map_scatter = build_map_scatter_figure(joined, eco, eco_draw, us_poly, f"{out_stem}_map_scatter")

    summ.to_csv(f"{out_stem}_bins.csv", index=False)
    if SHOW_RARE:
        rare_bin_summary(joined).to_csv(f"{out_stem}_rare_smooth.csv", index=False)
    if SHOW_RARE:
        ok = eco[usable & eco["mean_rare"].notna()]
        r_eco = stats.spearmanr(ok["mean_conifer"], ok["mean_rare"])[0]
        print(f"[result] across ecoregions, Spearman rho(mean % conifer, mean % rare) = {r_eco:+.2f}")
    keep = [c for c in (ECO_NAME_COL,) if c in eco.columns] + \
        ["n_plots", "mean_conifer"] + (["mean_rare"] if SHOW_RARE else []) + ["conifer_class", "low_n"]
    eco.loc[~eco.no_plots, keep].sort_values("mean_conifer", ascending=False).to_csv(
        f"{out_stem}_ecoregions.csv", index=False)
    print(f"[summary] saved {out_stem}_bins.csv and {out_stem}_ecoregions.csv")
    return fig_gradients, fig_map_scatter


if __name__ == "__main__":
    run()
    plt.show()