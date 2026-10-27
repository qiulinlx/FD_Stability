"""
Figures of the diversity -> NPP stability effects across the contiguous US
(Hawaii and Alaska removed; nothing drawn east of the New England-Acadian forests).

FIGURE 1 -- combined (OUT_COMBINED_NAME)
  (a) Dominant metric along the % conifer gradient: winner strip (hue = metric,
      intensity = magnitude, tied bins in FRAGILE_COLOR, n plots printed) above the
      mean |effect| of every metric per bin.
  (b) Total diversity effect along the same gradient: mean total effect per bin
      (thick line + CI band, markers coloured as in map d) and each metric's signed
      contribution (thin lines).
  (c) Map: WHICH metric has the largest mean |effect| in each ecoregion (hue) and
      HOW LARGE it is (intensity); tied ecoregions in FRAGILE_COLOR.
  (d) Map: TOTAL diversity effect in each ecoregion as % change in stability,
      destabilising (orange) <- white -> stabilising (blue), optionally on a
      signed-log colour scale.

  (e) Heatmap (SHOW_DIRECTION_HEATMAP), right of the maps: share of PLOTS whose total
      effect is stabilising / neutral / destabilising, per forest type (FOREST_TYPE_CLASSES
      on each plot's own % conifer). Neutral = total effect inside NEUTRAL_INTERVAL around 0
      (e.g. -0.001% to +0.001%), also marked on map d's colour bar. Shares sum to 100% within
      each forest type. Counts and shares saved to OUT_DIRECTION_FOREST_CSV.

FIGURE 3 -- heatmaps by rarity (OUT_HEATMAP_RARITY_NAME, SHOW_RARITY_HEATMAPS)
  The plot-level heatmap (e) repeated for low / medium / high rarity (tertiles of the plots'
  % rare species, or RARITY_CUTS), side by side; shares within forest type x rarity class.

FIGURE 4 -- map + rarity heatmaps (OUT_MAP_RARITY_NAME, SHOW_RARITY_HEATMAPS)
  (a) total-effect map with a vertical colour bar on its left; (b-d) the three rarity
  heatmaps of Figure 3 underneath.

FIGURE 5 -- map + WSCI heatmaps (OUT_MAP_WSCI_NAME, SHOW_WSCI_HEATMAPS)
  As Figure 4, but the heatmaps are segmented by structural complexity (WSCI_COL;
  tertiles or WSCI_CUTS) instead of rarity. The heatmaps alone go to OUT_HEATMAP_WSCI_NAME.

FIGURE 6 -- bubble map (OUT_BUBBLE_MAP_NAME, SHOW_BUBBLE_MAP)
  Ecoregions filled by the total diversity effect on stability (as map d); on each a circle
  coloured by its dominant metric (as map c), circle area proportional to that metric's
  mean |effect|. Tied ecoregions: FRAGILE_COLOR circles; < MIN_PLOTS ecoregions: no circle.

FIGURE 2 -- maps only (OUT_MAPS_NAME)
  The total-effect map with its colour key, plus the heatmap, without the conifer panels
  or the dominant-metric map: (a) total effect, (b) heatmap. MAPS_FIG_SHOW_DOMINANT = True
  puts the dominant-metric map back as (a). Uses exactly the same colour scales as
  Figure 1, so the two figures can be compared directly.

Panels (a) and (b) share the x-axis (conifer on the left -> deciduous on the right);
the maps share the extent and plots. All panels use ONE set of plots: those
with an effect for every metric, inside the contiguous US and assigned to an ecoregion.

Input: per-observation effects from the joint (mutually adjusted) model, long
format -- one row per plot x diversity metric with columns
    PID, diversity_metric, adjusted_effect, lon, lat
adjusted_effect = theta_j * (D~_j - mean D~_j), positive = stabilising
(EFFECT_SIGN flips it if needed), plus a PID -> percent_conifer lookup.

How to read it
  * Dominant metric uses |effect|: the effects are centred, so magnitude is what
    carries information. Total effect uses the SIGNED sum over metrics, relative
    to a plot with nationally average (residualised) diversity.
  * The per-plot effects come from one pooled set of theta_j, so differences between
    ecoregions / bins reflect how each residualised metric varies there, not a
    different theta_j.
  * CIs reflect spread among plots only (no theta_j uncertainty or spatial
    autocorrelation), so they are optimistic.
  * Ecoregions / bins with fewer than the minimum plots are hatched; ecoregions with
    no plots are grey.

Outputs (folder and names set in the "Output files" block below): each figure as a
600 dpi PNG and a vector PDF, plus four summary CSVs.
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
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FuncFormatter, MaxNLocator, MultipleLocator
from shapely.geometry import box
from scipy.stats import norm as _norm

# ===============================================================
# Input and output files  (edit names here only)
# ===============================================================
EFFECTS_CSV = "results/mutually_adjusted_effects_per_observation_with_coords.csv"          # stability
WSCI_EFFECTS_CSV = "results/wsci/mutually_adjusted_effects_per_observation_with_coords.csv"  # WSCI (from
                                                     #   the effects script run with OUTCOME = "wsci")
ECOREGIONS_PATH = "data/Ecoregions/Ecoregions2017.shp"
STATES_PATH = None            # e.g. "data/states/cb_2018_us_state_20m.shp"
CONIFER_LOOKUP_PATH = "data/lookup/PID_location_v3.csv"   # PID -> percent_conifer
RARE_PATH = "data/final/dataset_rarity.parquet"           # PID -> pct_rare_species (rarity heatmaps)
RAW_DATA_PATH = "data/final/final_dataset_ba.csv"         # PID -> raw NPP SD and raw WSCI
WSCI_PATH = RAW_DATA_PATH                                 # PID -> WSCI for the WSCI-class heatmaps

OUTDIR = "results"
# Figures: name WITHOUT extension; each is saved as .png and .pdf
OUT_COMBINED_NAME = "diversity_effects_combined"    # panels a-d
OUT_MAPS_NAME = "diversity_effects_maps"            # maps + colour keys only
# Summary tables
OUT_DOMINANT_CSV = "diversity_effects_dominant_summary.csv"
OUT_TOTAL_CSV = "diversity_effects_total_summary.csv"
OUT_CONIFER_DOMINANT_CSV = "diversity_effects_conifer_dominant.csv"
OUT_CONIFER_TOTAL_CSV = "diversity_effects_conifer_total.csv"
OUT_DIRECTION_FOREST_CSV = "diversity_effects_direction_by_forest.csv"   # counts behind the heatmap
OUT_HEATMAP_RARITY_NAME = "diversity_effects_heatmap_by_rarity"           # 3 heatmaps, one per rarity class
OUT_MAP_RARITY_NAME = "diversity_effects_map_rarity"   # total-effect map (a) above the 3 rarity heatmaps (b-d)
OUT_HEATMAP_WSCI_NAME = "diversity_effects_heatmap_by_wsci"       # 3 heatmaps, one per WSCI class
OUT_MAP_WSCI_NAME = "diversity_effects_map_wsci"                  # map (a) above the 3 WSCI heatmaps (b-d)
OUT_BUBBLE_MAP_NAME = "diversity_effects_bubble_map"   # total effect (fill) + dominant metric (circles)
OUT_BUBBLE_MAP_WSCI_NAME = "diversity_effects_bubble_map_wsci"   # the same for effects on WSCI
OUT_DIRECTION_FOREST_WSCI_CSV = "diversity_effects_direction_by_forest_wsci.csv"
OUT_DIRECTION_FOREST_RARITY_CSV = "diversity_effects_direction_by_forest_rarity.csv"


def _out(name, outdir=None):
    return os.path.join(outdir or OUTDIR, name)


# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------
ECO_NAME_COL = "ECO_NAME"
CONIFER_COL = "percent_conifer"

EFFECT_SIGN = 1               # 1 if adjusted_effect is already "+ = stabilising"; -1 to flip

# One metric palette for the whole figure (keys must match diversity_metric in the CSV)
METRIC_COLORS = {
    "Rao's Q": "#4B2BC4",
    "Functional Evenness": "#A47CC7",
    "Species richness": "#14B8A0",
}

# Dominant-metric encoding
INTENSITY_FLOOR = 0.18        # weakest effects still show 18% of the hue
FRAGILE_MARGIN = 1.10         # tied: winner mean |effect| < FRAGILE_MARGIN x runner-up
FRAGILE_COLOR = "#4675DA"

# Palettes chosen to sit alongside the metric colours (violet, lilac, teal)
# Total effect on stability: brick red <- warm white -> slate blue
DESTAB_COLOR = "#B23A2E"
STAB_COLOR = "#2E5E7E"
NEUTRAL_COLOR = "#F5F3EE"
TOTAL_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "destab_stab", [DESTAB_COLOR, "#DE9A8F", NEUTRAL_COLOR, "#93AFC2", STAB_COLOR])
# Total effect on WSCI: cool grey-slate (less complex) <- warm white -> dark bark (more complex)
WSCI_LESS_COLOR, WSCI_MORE_COLOR = "#5E6C84", "#6B4A33"
WSCI_EFFECT_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "wsci_effect", [WSCI_LESS_COLOR, "#AEB6C4", NEUTRAL_COLOR, "#C2A07A", WSCI_MORE_COLOR])
# Forest type, rarity and WSCI classes
FOREST_COLORS = {"Broadleaf": "#D9963A", "Mixed": "#8A9A5B", "Conifer": "#2F5233"}
RARITY_RAMP = ["#FFF6D5", "#F0C94A", "#B08A00"]   # low -> high rarity
WSCI_RAMP = ["#F3EBDD", "#C2A07A", "#6B4A33"]     # low -> high WSCI
CONIFER_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "conifer", [FOREST_COLORS["Broadleaf"], FOREST_COLORS["Mixed"], FOREST_COLORS["Conifer"]])
LOG_COLOR_SCALE = True        # signed-log colouring of the total map (and the markers in b)
LOG_LINTHRESH = None          # |% change| below this is coloured linearly; None = automatic
SHOW_UNCERTAIN = False        # stipple ecoregions in the total map whose CI includes 0
UNCERTAIN_HATCH = "..."
UNCERTAIN_HATCH_COLOR = "#4d4d4d"
CI_LEVEL = 0.90
Z_CI = _norm.ppf(1 - (1 - CI_LEVEL) / 2)
CI_PCT = round(CI_LEVEL * 100)
SHOW_METRIC_CONTRIBUTIONS = True   # thin metric lines in b

# Heatmap to the right of the maps: share of PLOTS whose total diversity effect is
# stabilising / neutral / destabilising, per forest type (each plot's own % conifer; plots
# between the classes are not counted). Only plots in the ecoregions drawn on the map.
SHOW_DIRECTION_HEATMAP = True
NEUTRAL_INTERVAL = 0.001                # neutral = total effect (% change in stability) within this
                                        #   interval around 0: x -> [-x, +x], or (lo, hi), e.g. (-0.002, 0.001)
MARK_NEUTRAL_ON_COLORBAR = True         # bracket the neutral interval on map d's colour bar

# The same heatmap repeated per rarity class (separate figure, OUT_HEATMAP_RARITY_NAME)
SHOW_RARITY_HEATMAPS = True            # computed for the bubble map's left column
SAVE_RARITY_HEATMAP_FIGURE = False     # also save them as their own figure
RARE_COL = "pct_rare_species"           # raw % (or proportion) of rare species per plot
RARITY_CUTS = None                      # None = tertiles of the plots; or cut points in %, e.g. [20, 40]
RARITY_NAMES = ["Low rarity", "Medium rarity", "High rarity"]

# The same heatmaps segmented by structural complexity (WSCI) instead of rarity
SHOW_WSCI_HEATMAPS = True

# Bubble map: ecoregions filled by the total diversity effect on stability (as map d), with a
# circle on each one coloured by its dominant metric (as map c) and sized by that metric's
# mean |effect|. Tied ecoregions get a FRAGILE_COLOR circle; < MIN_PLOTS ecoregions none.
SHOW_BUBBLE_MAP = True
SHOW_WSCI_BUBBLE_MAP = True   # bubble map of the diversity effects on WSCI (needs WSCI_EFFECTS_CSV)

# Which figures to save. Kept: the combined figure, the bubble maps and the stability heatmaps
# (by rarity and by WSCI class). The others can be switched back on here.
SHOW_MAPS_FIGURE = False          # maps-only figure (Figure 2)
SHOW_MAP_HEATMAP_FIGURES = False  # map above the rarity / WSCI heatmaps (Figures 4 and 5)

# Labels and units per outcome (the bubble maps use the matching one)
OUTCOME_STYLES = {
    "stability": dict(log=True, title="Total diversity effect on stability",
                      units="% change vs. average diversity", pos="Stabilising", neg="Destabilising",
                      cmap=TOTAL_CMAP, pos_color=STAB_COLOR, neg_color=DESTAB_COLOR),
    "wsci": dict(log=False, title="Total diversity effect on WSCI",
                 units="SD vs. average diversity", pos="More complex", neg="Less complex",
                 cmap=WSCI_EFFECT_CMAP, pos_color=WSCI_MORE_COLOR, neg_color=WSCI_LESS_COLOR),
}
_OUTCOME = dict(OUTCOME_STYLES["stability"])     # current outcome (set while drawing)
BUBBLE_MAX_SIZE = 140         # circle area (pt^2) at the BUBBLE_SIZE_PCT percentile of |effect|
BUBBLE_MIN_SIZE = 6           # smallest circle area (pt^2)
BUBBLE_SIZE_PCT = 98          # |effect| percentile that gets BUBBLE_MAX_SIZE (larger ones are capped)
BUBBLE_ALPHA = 0.95
# Bubble-map fill: "raw" = the ecoregion's median RAW outcome (stability = 1 / NPP SD on the
# stability map, raw WSCI on the WSCI map), circles = diversity effects as before;
# "effect" = the total diversity effect (the earlier version).
BUBBLE_FILL = "raw"
# Circle colour: "conifer" = the ecoregion's mean % conifer (broadleaf amber -> mixed sage ->
# conifer pine); "metric" = the dominant diversity metric (the earlier version).
# Circle size is in both cases the dominant metric's mean |effect|.
BUBBLE_COLOUR_BY = "metric"
# Direction on the bubbles from LOCAL slopes: in each ecoregion the slope of the residual outcome
# on each metric is refitted from that ecoregion's plots (BUBBLE_LOCAL_MODEL: "adjusted" = all
# metrics together, "individual" = each on its own; with intercept; HC3 SEs). The dominant metric
# is the one with the largest local |effect| per 1 national SD; its sign gives the marker:
#   up triangle = stabilising (CI above 0), down triangle = destabilising (CI below 0),
#   small grey dot = CI includes 0. Colour = that metric, size = its local |effect|.
# False = the earlier circles from the nationally pooled effects (no direction).
BUBBLE_DIRECTION = True
BUBBLE_LOCAL_MODEL = "adjusted"
BUBBLE_LOCAL_CI = 0.95
BUBBLE_NEUTRAL_SIZE = 10      # marker area (pt^2) of the grey "CI includes 0" dots
OUT_BUBBLE_LOCAL_CSV = "diversity_effects_bubble_local_slopes{suffix}.csv"
RAW_FILL_PCT = (2, 98)        # colour range = these percentiles of the ecoregion medians
RAW_STABILITY_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "raw_stab", ["#F2F5F7", "#93AFC2", "#2E5E7E", "#173246"])          # light -> slate blue
RAW_WSCI_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "raw_wsci", ["#F3EBDD", "#C2A07A", "#6B4A33", "#3E2A1D"])          # sand -> dark bark
RAW_WSCI_COL = "WSCI"                   # raw structural complexity column in RAW_DATA_PATH
RAW_STD_NPP_COL = "std npp"             # raw NPP SD in RAW_DATA_PATH (stability = 1 / NPP SD)
WSCI_COL = RAW_WSCI_COL                 # WSCI classes in the heatmaps use RAW WSCI (set to
                                        #   "residual_wsci" with WSCI_PATH = the residuals file to undo)
WSCI_CUTS = None                        # None = tertiles of the plots; or cut points in WSCI units
WSCI_NAMES = ["Low WSCI", "Medium WSCI", "High WSCI"]
MAPS_FIG_SHOW_DOMINANT = False          # Figure 2: False = total-effect map + heatmap only
FOREST_TYPE_CLASSES = [                 # (label, lower incl., upper excl.), left to right
    ("Conifer\n\u2265 90%", 90, np.inf),
    ("Mixed\n40\u201360%", 40, 60),
    ("Broadleaf\n< 10%", -np.inf, 10),
]
NEUTRAL_HEAT_COLOR = "#6b6b6b"
HEATMAP_INTENSITY = "column_share"      # cell intensity: "column_share" (share of the forest type's
                                        #   plots) or "count" (raw count, scaled to the largest)

# Shared
MIN_PLOTS_PER_ECOREGION = 20
MIN_PLOTS_PER_BIN = 20
CONIFER_BIN_WIDTH = 10        # % conifer per bin (a and b)
NEAREST_MAX_M = 5000          # snap unmatched plots to an ecoregion this close (m); None = don't
MAP_CRS = "EPSG:5070"         # CONUS Albers equal-area
SIMPLIFY_M = 500              # simplify polygons for DRAWING only (joins use full detail)
EAST_LIMIT_ECOREGION = "New England-Acadian forests"   # None = no eastern limit
CONUS_LON = (-130.0, -60.0)
CONUS_LAT = (23.0, 50.0)
FIG_WIDTH_IN = 7.2            # double-column width
DPI = 600

NO_PLOTS_FC, NO_PLOTS_EC = "#e6e6e6", "#9a9a9a"
LOW_N_FC, LOW_N_EC, LOW_N_HATCH = "#f7f7f7", "#808080", "//////"
EDGE = "#5a5a5a"


def _to_pct(change):
    if not _OUTCOME["log"]:                       # WSCI effects are already in SD units
        return change
    """Change on the (log) stability scale -> % change in stability."""
    return (np.exp(change) - 1) * 100


def _bins():
    return np.arange(0, 100 + CONIFER_BIN_WIDTH, CONIFER_BIN_WIDTH)


# ---------------------------------------------------------------
# Data
# ---------------------------------------------------------------
def load_plots(csv_path):
    """One row per plot: lon/lat, the SIGNED effect of every metric and their sum ('total')."""
    long = pd.read_csv(csv_path, usecols=["PID", "diversity_metric", "adjusted_effect", "lon", "lat"])
    wide = EFFECT_SIGN * long.pivot(index="PID", columns="diversity_metric", values="adjusted_effect")

    n_all = len(wide)
    wide = wide.dropna()          # every metric needed for the comparison and the total
    if len(wide) < n_all:
        print(f"[load] dropped {n_all - len(wide)} plots missing an effect for one or more metrics")
    unknown = [m for m in wide.columns if m not in METRIC_COLORS]
    if unknown:
        raise KeyError(f"No colour defined in METRIC_COLORS for: {unknown}")
    metrics = [m for m in METRIC_COLORS if m in wide.columns]
    wide["total"] = wide[metrics].sum(axis=1)

    coords = long.drop_duplicates("PID").set_index("PID")[["lon", "lat"]]
    plots = coords.loc[wide.index].join(wide).reset_index()
    conus = plots.lon.between(*CONUS_LON) & plots.lat.between(*CONUS_LAT)
    if (~conus).any():
        print(f"[load] dropped {int((~conus).sum())} plots outside the contiguous US (Hawaii/Alaska)")
    plots = plots[conus].reset_index(drop=True)
    print(f"[load] {len(plots)} plots, metrics: {metrics}")
    return plots, metrics


def attach_conifer(plots, lookup_path=CONIFER_LOOKUP_PATH, col=CONIFER_COL):
    """Adds % conifer (0-100) to each plot from the PID lookup."""
    if col not in plots.columns:
        lk = pd.read_csv(lookup_path, usecols=["PID", col]).drop_duplicates("PID")
        plots = plots.merge(lk, on="PID", how="left")
    if plots[col].max() <= 1:
        plots[col] = plots[col] * 100
    c = plots[col]
    print(f"[conifer] {int(c.isna().sum())} plots without % conifer; range {c.min():.0f}-{c.max():.0f}%")
    if c.max() - c.min() < 50:
        print("[conifer] WARNING: narrow % conifer range -- were these effects fitted on a subset?")
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
    return out[~out.geometry.is_empty]


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
    n_direct, n_snapped = int((~missing).sum()), 0
    if missing.any() and nearest_max_m:
        near = gpd.sjoin_nearest(pts.loc[missing[missing].index], ids, how="left",
                                 max_distance=nearest_max_m)
        near = near[~near.index.duplicated(keep="first")].dropna(subset=["eco_id"])
        j.loc[near.index, "eco_id"] = near["eco_id"].values
        n_snapped = len(near)
    n_lost = int(j["eco_id"].isna().sum())
    print(f"[join] {n_direct} plots inside an ecoregion, {n_snapped} snapped to the nearest "
          f"(<= {nearest_max_m} m), {n_lost} unassigned and dropped")
    return pd.DataFrame(j.dropna(subset=["eco_id"]).drop(columns="geometry"))


# ---------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------
def _rank_metrics(out, mags, metrics, min_plots):
    """Adds winner, value, runner_up, margin, no_plots, low_n and fragile columns."""
    has = ~np.isnan(mags).all(axis=1)
    order = np.argsort(-np.nan_to_num(mags, nan=-np.inf), axis=1)
    top, second = order[:, 0], order[:, 1]
    rows = np.arange(len(out))
    out["winner"] = [metrics[k] if h else None for k, h in zip(top, has)]
    out["value"] = np.where(has, mags[rows, top], np.nan)
    out["runner_up"] = [metrics[k] if h else None for k, h in zip(second, has)]
    out["margin"] = np.where(has, mags[rows, top] / mags[rows, second], np.nan)
    out["no_plots"] = out["n_plots"] == 0
    out["low_n"] = (out["n_plots"] > 0) & (out["n_plots"] < min_plots)
    out["fragile"] = ~out["no_plots"] & ~out["low_n"] & (out["margin"] < FRAGILE_MARGIN)
    return out


def summarize_dominant(joined, eco, metrics, min_plots=MIN_PLOTS_PER_ECOREGION):
    """Per ecoregion: mean |effect| per metric, winner, magnitude, margin and tie flag."""
    grp = joined.assign(**{m: joined[m].abs() for m in metrics}).groupby("eco_id")
    stats_ = pd.concat([grp.size().rename("n_plots"), grp[metrics].mean().add_prefix("mag__")], axis=1)
    out = eco.merge(stats_, left_on="eco_id", right_index=True, how="left")
    out["n_plots"] = out["n_plots"].fillna(0).astype(int)
    mags = out[[f"mag__{m}" for m in metrics]].to_numpy(dtype=float)
    return _rank_metrics(out, mags, metrics, min_plots)


def summarize_total(joined, eco, metrics, min_plots=MIN_PLOTS_PER_ECOREGION):
    """Per ecoregion: mean total effect, SE, CI, % change and signed metric contributions."""
    grp = joined.groupby("eco_id")
    parts = [grp.size().rename("n_plots"), grp["total"].mean().rename("mean_total"),
             grp["total"].std(ddof=1).rename("sd_total"),
             grp[metrics].mean().add_prefix("contrib__")]
    if CONIFER_COL in joined.columns:
        parts.append(grp[CONIFER_COL].mean().rename("conifer_mean"))
    stats_ = pd.concat(parts, axis=1)
    out = eco.merge(stats_, left_on="eco_id", right_index=True, how="left")
    out["n_plots"] = out["n_plots"].fillna(0).astype(int)
    out["se_total"] = out["sd_total"] / np.sqrt(out["n_plots"].where(out["n_plots"] > 0))
    out["ci_lower"] = out["mean_total"] - Z_CI * out["se_total"]
    out["ci_upper"] = out["mean_total"] + Z_CI * out["se_total"]
    out["pct"] = _to_pct(out["mean_total"])
    out["pct_ci_lower"] = _to_pct(out["ci_lower"])
    out["pct_ci_upper"] = _to_pct(out["ci_upper"])
    out["direction"] = np.where(out["mean_total"] > 0, "stabilising",
                                np.where(out["mean_total"] < 0, "destabilising", None))
    out["no_plots"] = out["n_plots"] == 0
    out["low_n"] = (out["n_plots"] > 0) & (out["n_plots"] < min_plots)
    out["uncertain"] = (~out["no_plots"] & ~out["low_n"]
                        & (out["ci_lower"] <= 0) & (out["ci_upper"] >= 0))
    return out


def conifer_dominant(plots, metrics, min_plots=MIN_PLOTS_PER_BIN):
    """Per % conifer bin: mean |effect| per metric, winner, magnitude, margin and tie flag."""
    bins = _bins()
    p = plots.dropna(subset=[CONIFER_COL])
    p = p.assign(_bin=pd.cut(p[CONIFER_COL], bins, include_lowest=True))
    g = p.assign(**{m: p[m].abs() for m in metrics}).groupby("_bin", observed=False)
    mags = g[metrics].mean()
    out = pd.DataFrame({"lo": bins[:-1], "hi": bins[1:], "n_plots": g.size().to_numpy()})
    for m in metrics:
        out[f"mag__{m}"] = mags[m].to_numpy()
    return _rank_metrics(out, mags.to_numpy(dtype=float), metrics, min_plots)


def conifer_total(plots, metrics, min_plots=MIN_PLOTS_PER_BIN):
    """Per % conifer bin: mean total effect (+ CI) and each metric's mean contribution, in %."""
    bins = _bins()
    d = plots.dropna(subset=[CONIFER_COL])
    d = d.assign(bin=pd.cut(d[CONIFER_COL], bins, include_lowest=True, labels=False))
    g = d.groupby("bin")
    t = pd.DataFrame({"n_plots": g.size(), "mean_total": g["total"].mean(),
                      "se_total": g["total"].std(ddof=1) / np.sqrt(g.size())})
    t["mid"] = bins[t.index.astype(int)] + CONIFER_BIN_WIDTH / 2
    t["pct"] = _to_pct(t["mean_total"])
    t["pct_ci_lower"] = _to_pct(t["mean_total"] - Z_CI * t["se_total"])
    t["pct_ci_upper"] = _to_pct(t["mean_total"] + Z_CI * t["se_total"])
    for m in metrics:
        t[f"pct__{m}"] = _to_pct(g[m].mean())
    return t[t["n_plots"] >= min_plots].reset_index(drop=True)


DIRECTIONS = ["Stabilising", "Neutral", "Destabilising"]


def neutral_interval():
    """(lo, hi) in % change: total effects with lo <= effect <= hi count as neutral."""
    x = NEUTRAL_INTERVAL
    lo, hi = (-abs(float(x)), abs(float(x))) if np.isscalar(x) else (float(x[0]), float(x[1]))
    if not lo <= 0 <= hi:
        raise ValueError(f"NEUTRAL_INTERVAL must contain 0, got ({lo:g}, {hi:g})")
    return lo, hi


def _interval_text(iv):
    lo, hi = iv
    if np.isclose(-lo, hi):
        return f"within \u00b1{hi:g}%"
    return f"within {_pct_fmt(lo)}% to {_pct_fmt(hi)}%"


def direction_by_forest(plots, iv):
    """
    NOTE: the direction comes from each plot's TOTAL DIVERSITY EFFECT ('total' = sum of the
    adjusted effects theta_j * (D~_j - mean)), i.e. the DML model fitted on residualised
    stability. Raw stability (1 / NPP SD) is used only for the bubble-map fill, never here.
    Plots by direction of their total effect (stabilising / neutral / destabilising, with
    effects inside the neutral interval iv = (lo, hi) = neutral) and forest type.
    Returns (counts array [direction, forest type], long table for the CSV).
    """
    p = plots.dropna(subset=[CONIFER_COL, "total"])
    pct = _to_pct(p["total"].to_numpy(float))
    lo_n, hi_n = iv
    direction = np.where((pct >= lo_n) & (pct <= hi_n), "Neutral",
                         np.where(pct > hi_n, "Stabilising", "Destabilising"))
    conif = p[CONIFER_COL].to_numpy(float)
    counts = np.zeros((len(DIRECTIONS), len(FOREST_TYPE_CLASSES)), dtype=int)
    for f, (_, lo, hi) in enumerate(FOREST_TYPE_CLASSES):
        in_f = (conif >= lo) & (conif < hi)
        for d, name in enumerate(DIRECTIONS):
            counts[d, f] = int((in_f & (direction == name)).sum())
    print(f"[heatmap] {counts.sum():,} plots in a forest class ({len(p) - counts.sum():,} between "
          f"classes, not counted); neutral = effect {_interval_text(iv)}")
    rows = []
    for f, (label, _, _) in enumerate(FOREST_TYPE_CLASSES):
        tot = counts[:, f].sum()
        for d, name in enumerate(DIRECTIONS):
            rows.append({"forest_type": label.replace("\n", " "), "direction": name,
                         "n_plots": counts[d, f],
                         "share_of_forest_type": counts[d, f] / tot if tot else np.nan,
                         "neutral_lower_pct": lo_n, "neutral_upper_pct": hi_n})
    return counts, pd.DataFrame(rows)


# ---------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------
def blend(base_hex, t, floor=INTENSITY_FLOOR):
    """White -> base colour; t in [0, 1] is the normalised magnitude."""
    base = np.array(mcolors.to_rgb(base_hex))
    a = floor + (1 - floor) * np.clip(t, 0, 1)
    return tuple(1 - a * (1 - base))


def _text_color(fc):
    r, g, b = mcolors.to_rgb(fc)
    return "white" if 0.299 * r + 0.587 * g + 0.114 * b < 0.5 else "#1a1a1a"


def _pct_fmt(v, _=None):
    return "0" if abs(v) < 1e-12 else f"{v:+g}".replace("-", "−")


def _nice_up(x):
    k = np.floor(np.log10(x))
    for m in (1, 2, 5, 10):
        if m * 10 ** k >= x * (1 - 1e-9):
            return m * 10 ** k


def magnitude_norm(eco_dom, conf):
    """Shared |effect| scale for the dominant-metric panels, from clear winners (falls back to ties)."""
    vals = None
    for include_tied in (False, True):
        parts = []
        for df in (eco_dom, conf):
            keep = ~df["no_plots"] & ~df["low_n"]
            if not include_tied:
                keep &= ~df["fragile"]
            parts.append(df.loc[keep, "value"])
        vals = pd.concat(parts).dropna()
        if len(vals):
            break
    if vals is None or vals.empty:
        raise ValueError("No ecoregion or conifer bin has enough plots to colour; lower the minimums.")
    exp = int(np.floor(np.log10(vals.max())))
    scale = 10.0 ** exp
    ticks = MaxNLocator(5, steps=[1, 2, 2.5, 5, 10]).tick_values(vals.min() / scale, vals.max() / scale)
    return mcolors.Normalize(ticks[0] * scale, ticks[-1] * scale), ticks, scale, exp


def total_norm(eco_tot):
    """Diverging scale for the total effect, linear or signed-log, symmetric about 0."""
    vals = eco_tot.loc[~eco_tot.no_plots & ~eco_tot.low_n, "pct"].dropna()
    vmax_raw = float(vals.abs().max())
    if not LOG_COLOR_SCALE:
        ticks = MaxNLocator(4, symmetric=True, steps=[1, 2, 2.5, 5, 10]).tick_values(-vmax_raw, vmax_raw)
        return mcolors.Normalize(-ticks[-1], ticks[-1]), ticks
    vmax = _nice_up(vmax_raw)
    lt = LOG_LINTHRESH
    if lt is None:
        nz = vals.abs()[vals.abs() > 0]
        q = np.percentile(nz, 10) if len(nz) else vmax / 100
        lt = 10 ** np.floor(np.log10(q))
    lt = min(lt, vmax / 10)
    norm = mcolors.SymLogNorm(linthresh=lt, linscale=0.5, vmin=-vmax, vmax=vmax, base=10)
    decades = 10.0 ** np.arange(np.ceil(np.log10(lt)), np.floor(np.log10(vmax)) + 1)
    pos = sorted(set(np.round(np.r_[decades, vmax], 12)))[-3:]
    print(f"[map] signed-log colour scale: linear within ±{lt:g}%, range ±{vmax:g}%")
    return norm, np.r_[-np.array(pos[::-1]), 0.0, pos]


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


def _scale_bar(ax, extent, length_m=500_000):
    xmin, xmax, ymin, ymax = extent
    width, height = xmax - xmin, ymax - ymin
    x1, y = xmax - 0.04 * width, ymin + 0.06 * height
    x0 = x1 - length_m
    ax.plot([x0, x1], [y, y], color="black", lw=0.9, solid_capstyle="butt", zorder=10)
    for x in (x0, x1):
        ax.plot([x, x], [y - 0.012 * height, y + 0.012 * height], color="black", lw=0.9, zorder=10)
    ax.text((x0 + x1) / 2, y + 0.025 * height, f"{length_m // 1000:d} km",
            ha="center", va="bottom", fontsize=6, zorder=10)


def _style_axes(ax):
    for name, sp in ax.spines.items():
        sp.set_visible(name in ("left", "bottom"))
        sp.set_linewidth(0.5)
    ax.tick_params(labelsize=6.5, length=2, width=0.4, pad=1.5)


def _setup_map(ax, eco, extent):
    xmin, xmax, ymin, ymax = extent
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_axis_off()
    if eco.no_plots.any():
        eco[eco.no_plots].plot(ax=ax, facecolor=NO_PLOTS_FC, edgecolor=NO_PLOTS_EC, linewidth=0.15)
    if eco.low_n.any():
        eco[eco.low_n].plot(ax=ax, facecolor=LOW_N_FC, edgecolor=LOW_N_EC, linewidth=0.15,
                            hatch=LOW_N_HATCH)


def _outline(ax, us_poly, crs):
    if us_poly is not None:
        gpd.GeoSeries([us_poly], crs=crs).boundary.plot(ax=ax, color="black", linewidth=0.4)


def _map_extent(eco, us_poly):
    xmin, ymin, xmax, ymax = us_poly.bounds if us_poly is not None else eco.total_bounds
    pad = 0.01 * (xmax - xmin)
    return (xmin - pad, xmax + pad, ymin - pad, ymax + pad)


def _save(fig, stem):
    os.makedirs(os.path.dirname(stem) or ".", exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=DPI, bbox_inches=None, facecolor="white")
    print(f"[figure] saved {stem}.png / .pdf")


# ---------------------------------------------------------------
# Panels
# ---------------------------------------------------------------
def panel_dominant_conifer(fig, rect, conf, metrics, mnorm, scale, exp):
    """(a) Winner strip above mean |effect| lines along the % conifer gradient."""
    x0, y0, w, h = rect
    strip_h, gap = 0.20 * h, 0.05 * h
    sax = fig.add_axes([x0, y0 + h - strip_h, w, strip_h])
    lax = fig.add_axes([x0, y0, w, h - strip_h - gap], sharex=sax)

    for r in conf.itertuples():
        if r.no_plots:
            fc, ec, hatch = NO_PLOTS_FC, NO_PLOTS_EC, None
        elif r.low_n:
            fc, ec, hatch = LOW_N_FC, LOW_N_EC, LOW_N_HATCH
        elif r.fragile:
            fc, ec, hatch = FRAGILE_COLOR, EDGE, None
        else:
            fc, ec, hatch = blend(METRIC_COLORS[r.winner], mnorm(r.value)), EDGE, None
        sax.add_patch(Rectangle((r.lo, 0), r.hi - r.lo, 1, facecolor=fc, edgecolor=ec,
                                linewidth=0.3, hatch=hatch))
        if r.n_plots:
            sax.text((r.lo + r.hi) / 2, 0.5, f"{r.n_plots:,}", ha="center", va="center",
                     fontsize=5, color=_text_color(fc))
    sax.set_ylim(0, 1)
    sax.set_yticks([])
    sax.tick_params(axis="x", length=0, labelbottom=False)
    sax.set_ylabel("Dominant\nmetric", rotation=0, ha="right", va="center", fontsize=6.5, labelpad=4)
    sax.text(1.01, 0.5, "n plots", transform=sax.transAxes, fontsize=6, va="center", ha="left")
    for sp in sax.spines.values():
        sp.set_visible(False)

    for r in conf[conf["fragile"]].itertuples():
        lax.axvspan(r.lo, r.hi, color=FRAGILE_COLOR, alpha=0.15, linewidth=0, zorder=0)

    mids = ((conf["lo"] + conf["hi"]) / 2).to_numpy()
    ok = (~conf["no_plots"] & ~conf["low_n"]).to_numpy()
    low = conf["low_n"].to_numpy()
    for m in metrics:
        y = conf[f"mag__{m}"].to_numpy(dtype=float) / scale
        c = METRIC_COLORS[m]
        lax.plot(mids, np.where(ok, y, np.nan), color=c, lw=1.2, marker="o", ms=2.8, zorder=3)
        if low.any():
            lax.plot(mids[low], y[low], ls="none", marker="o", ms=2.8, mfc="white", mec=c, mew=0.7,
                     zorder=3)

    lax.set_xlim(100, 0)
    lax.xaxis.set_major_locator(MultipleLocator(10))
    lax.set_ylim(bottom=0)
    lax.yaxis.set_major_locator(MaxNLocator(3))
    lax.set_ylabel(f"Mean |effect|\n(\u00d710$^{{{exp}}}$)", fontsize=6.5, linespacing=1.15)
    _style_axes(lax)

    handles = [Line2D([], [], color=METRIC_COLORS[m], lw=1.2, marker="o", ms=3, label=m) for m in metrics]
    if conf["fragile"].any():
        handles.append(Patch(facecolor=FRAGILE_COLOR, alpha=0.15, linewidth=0, label="Tied bin"))
    lax.legend(handles=handles, fontsize=6, frameon=False, loc="upper left",
               bbox_to_anchor=(1.01, 1.0), handlelength=1.6, borderaxespad=0)
    return lax


def panel_total_conifer(ax, trend, metrics, tnorm):
    """(b) Total effect (+ metric contributions) along the % conifer gradient."""
    ax.axhline(0, color="#777777", lw=0.6, ls=(0, (3, 2)), zorder=1)
    if SHOW_METRIC_CONTRIBUTIONS:
        for m in metrics:
            ax.plot(trend["mid"], trend[f"pct__{m}"], color=METRIC_COLORS[m],
                    lw=1.0, marker="o", ms=2.4, alpha=0.9, zorder=2)
    ax.fill_between(trend["mid"], trend["pct_ci_lower"], trend["pct_ci_upper"],
                    color="#222222", alpha=0.12, lw=0, zorder=2)
    ax.plot(trend["mid"], trend["pct"], color="#222222", lw=1.6, zorder=3)
    ax.scatter(trend["mid"], trend["pct"], c=[TOTAL_CMAP(tnorm(v)) for v in trend["pct"]],
               s=26, edgecolors="#222222", linewidths=0.6, zorder=4)

    ax.set_xlim(100, 0)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_major_locator(MaxNLocator(4, symmetric=True))
    ax.yaxis.set_major_formatter(FuncFormatter(_pct_fmt))
    ax.set_xlabel("% conifer  \u2192  less conifer", fontsize=7, labelpad=2)
    ax.set_ylabel("Total effect on\nstability (% change)", fontsize=6.5, linespacing=1.15)
    _style_axes(ax)

    handles = [Line2D([], [], color="#222222", lw=1.6, marker="o", ms=4.5, mfc="white", mec="#222222",
                      label="Total (marker = map d colour)"),
               Patch(facecolor="#222222", alpha=0.12, edgecolor="none", label=f"{CI_PCT}% CI")]
    if SHOW_METRIC_CONTRIBUTIONS:
        handles += [Line2D([], [], color=METRIC_COLORS[m], lw=1.0, marker="o", ms=2.8,
                           label=f"{m} contribution") for m in metrics]
    ax.legend(handles=handles, fontsize=6, frameon=False, loc="upper left",
              bbox_to_anchor=(1.01, 1.0), handlelength=1.6, borderaxespad=0)


def map_dominant(ax, eco, extent, us_poly, mnorm):
    """Dominant metric per ecoregion."""
    _setup_map(ax, eco, extent)
    fragile = eco[eco.fragile]
    colored = eco[~eco.no_plots & ~eco.low_n & ~eco.fragile]
    if len(fragile):
        fragile.plot(ax=ax, facecolor=FRAGILE_COLOR, edgecolor=EDGE, linewidth=0.15)
    if len(colored):
        colored.plot(ax=ax, color=[blend(METRIC_COLORS[m], mnorm(v))
                                   for m, v in zip(colored["winner"], colored["value"])],
                     edgecolor=EDGE, linewidth=0.15)
    _outline(ax, us_poly, eco.crs)


def map_total(ax, eco, extent, us_poly, tnorm):
    """Total diversity effect per ecoregion."""
    _setup_map(ax, eco, extent)
    colored = eco[~eco.no_plots & ~eco.low_n]
    colored.plot(ax=ax, color=[_OUTCOME["cmap"](tnorm(v)) for v in colored["pct"]],
                 edgecolor=EDGE, linewidth=0.15)
    if SHOW_UNCERTAIN and eco.uncertain.any():
        eco[eco.uncertain].plot(ax=ax, facecolor="none", edgecolor=UNCERTAIN_HATCH_COLOR,
                                hatch=UNCERTAIN_HATCH, linewidth=0)
    _outline(ax, us_poly, eco.crs)
    _scale_bar(ax, extent)


def _forest_color(label):
    return next((c for k, c in FOREST_COLORS.items() if label.startswith(k)), "#1a1a1a")


def panel_direction_heatmap(ax, counts, thr, title=None, footnote=True, ylabels=True,
                            compact=False, swatch=None):
    """Plots by direction (rows) and forest type (columns): share of each forest type's plots."""
    row_colors = {"Stabilising": STAB_COLOR, "Neutral": NEUTRAL_HEAT_COLOR,
                  "Destabilising": DESTAB_COLOR}
    n_d, n_f = counts.shape
    col_tot = counts.sum(axis=0)
    for d, name in enumerate(DIRECTIONS):
        for f in range(n_f):
            c = counts[d, f]
            share = c / col_tot[f] if col_tot[f] else 0.0
            t = share if HEATMAP_INTENSITY == "column_share" else c / max(counts.max(), 1)
            fc = blend(row_colors[name], t, floor=0.06) if c else "white"
            ax.add_patch(Rectangle((f, d), 1, 1, facecolor=fc, edgecolor="white", linewidth=1.2))
            if col_tot[f]:
                ax.text(f + 0.5, d + 0.5, f"{share:.0%}", ha="center", va="center",
                        fontsize=6.5, color=_text_color(fc))
    ax.set_xlim(0, n_f)
    ax.set_ylim(n_d, 0)
    ax.set_aspect("equal")
    ax.set_yticks(np.arange(n_d) + 0.5)
    if ylabels:
        ax.set_yticklabels(DIRECTIONS, fontsize=6)
        for d, name in enumerate(DIRECTIONS):
            ax.get_yticklabels()[d].set_color(row_colors[name])
    else:
        ax.set_yticklabels([])
    ax.set_xticks(np.arange(n_f) + 0.5)
    if compact:            # one short line per column: forest type + n
        xl = [f"{lab.split(chr(10))[0]}\nn={col_tot[f]:,}" for f, (lab, _, _) in enumerate(FOREST_TYPE_CLASSES)]
    else:
        xl = [f"{lab}\n(n = {col_tot[f]:,})" for f, (lab, _, _) in enumerate(FOREST_TYPE_CLASSES)]
    ax.set_xticklabels(xl, fontsize=5 if compact else 5.5, linespacing=1.05)
    for t, (lab, _, _) in zip(ax.get_xticklabels(), FOREST_TYPE_CLASSES):
        t.set_color(_forest_color(lab))
        t.set_fontweight("bold")
    ax.tick_params(length=0, pad=2)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(title or "Plots by direction of the\ntotal effect and forest type", fontsize=6.5,
                 linespacing=1.15, pad=4)
    if swatch is not None:   # thin class-colour bar along the top of the heatmap
        ax.add_patch(Rectangle((0, 1.02), 1, 0.045, transform=ax.transAxes, facecolor=swatch,
                               edgecolor="#555555", linewidth=0.4, clip_on=False))
        ax.set_title(ax.get_title(), fontsize=6.5, linespacing=1.15, pad=9)   # clear of the bar
    if footnote:
        ax.annotate(f"% of each forest type's plots\nneutral = effect {_interval_text(thr)}\n"
                    f"forest type = plot % conifer",
                    xy=(0.5, 0), xycoords="axes fraction", xytext=(0, -30), textcoords="offset points",
                    ha="center", va="top", fontsize=5, color="#555555", linespacing=1.1)


def legend_dominant(fig, H, metrics, mnorm, mticks, scale, exp, national_means, y_in, tag="c"):
    """Metric gradient key + national mean |effect| bars (under the dominant-metric map)."""
    L = _layout()
    n = len(metrics)
    row_h = 0.14 / H
    lax = fig.add_axes([L["key_x"], y_in / H, L["key_w"], n * row_h])
    t = np.linspace(0, 1, 256)
    img = np.array([[blend(METRIC_COLORS[m], tt) for tt in t] for m in metrics])
    vmin, vmax = mnorm.vmin / scale, mnorm.vmax / scale
    lax.imshow(img, aspect="auto", extent=(vmin, vmax, n, 0), interpolation="nearest")
    for r in range(1, n):
        lax.axhline(r, color="white", lw=1.0)
    lax.set_yticks(np.arange(n) + 0.5)
    lax.set_yticklabels(metrics, fontsize=6.5)
    lax.set_xticks(mticks)
    lax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    lax.xaxis.tick_top()
    lax.xaxis.set_label_position("top")
    key_txt = (f"({tag}) Dominant metric (hue)\nand mean |effect|\n(intensity, \u00d710$^{{{exp}}}$)"
               if SHOW_DIRECTION_HEATMAP else
               f"({tag}) Dominant metric (hue) and\nmean |effect| (intensity, \u00d710$^{{{exp}}}$)")
    lax.set_xlabel(key_txt,
                   fontsize=6.5, labelpad=3, loc="left", linespacing=1.15)
    lax.tick_params(axis="x", length=2, width=0.4, labelsize=6, pad=1)
    lax.tick_params(axis="y", length=0, pad=2)
    for sp in lax.spines.values():
        sp.set_linewidth(0.4)

    bax = fig.add_axes([L["bar_x"], y_in / H, L["bar_w"], n * row_h])
    v = np.array([national_means[m] for m in metrics]) / scale
    ypos = np.arange(n) + 0.5
    bax.barh(ypos, v, height=0.72, color=[METRIC_COLORS[m] for m in metrics])
    bax.set_ylim(n, 0)
    bax.set_xlim(0, v.max() * 1.35)
    for yy, vv in zip(ypos, v):
        bax.text(vv + v.max() * 0.04, yy, f"{vv:.2f}", va="center", ha="left", fontsize=5.5)
    bax.set_yticks([])
    bax.xaxis.set_major_locator(MaxNLocator(2))
    bax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
    bax.xaxis.tick_top()
    bax.xaxis.set_label_position("top")
    bax.set_xlabel("National mean\n|effect|", fontsize=6.5, labelpad=3, loc="left", linespacing=1.15)
    bax.tick_params(axis="x", length=2, width=0.4, labelsize=6, pad=1)
    for name_, sp in bax.spines.items():
        sp.set_visible(name_ == "left")
        sp.set_linewidth(0.4)


def legend_total(fig, H, tnorm, tticks, y_in, tag="d", neutral=None):
    """Diverging colour bar (under the total-effect map); neutral = (lo, hi) to bracket."""
    L = _layout()
    cax = fig.add_axes([L["cbar_x"], y_in / H, L["cbar_w"], 0.085 / H])
    cb = fig.colorbar(ScalarMappable(norm=tnorm, cmap=TOTAL_CMAP), cax=cax, orientation="horizontal")
    cb.set_ticks(tticks)
    cb.ax.xaxis.set_major_formatter(FuncFormatter(_pct_fmt))
    cb.ax.tick_params(labelsize=5.5, length=2, width=0.4, pad=1)
    cb.outline.set_linewidth(0.4)
    cax.xaxis.set_label_position("top")
    scale_note = ", log colour scale" if LOG_COLOR_SCALE else ""
    mark = neutral is not None and MARK_NEUTRAL_ON_COLORBAR
    cax.set_xlabel(f"({tag}) Total diversity effect on stability\n(% change vs. average diversity{scale_note})",
                   fontsize=6.5, labelpad=9 if mark else 3, linespacing=1.15)   # room for the bracket
    cax.text(-0.03, 0.5, "Destabilising", transform=cax.transAxes, ha="right", va="center", fontsize=6)
    cax.text(1.03, 0.5, "Stabilising", transform=cax.transAxes, ha="left", va="center", fontsize=6)
    if neutral is not None and MARK_NEUTRAL_ON_COLORBAR:
        lo, hi = (float(np.clip(x, tnorm.vmin, tnorm.vmax)) for x in neutral)
        x0, x1 = (cax.transAxes.inverted().transform(cax.transData.transform((x, 0)))[0]
                  for x in (lo, hi))
        for x in (x0, x1):          # thin black ticks spanning the bar at the interval ends
            cax.plot([x, x], [-0.15, 1.15], color="#1a1a1a", lw=0.7, transform=cax.transAxes,
                     clip_on=False, zorder=5)
        cax.plot([x0, x1], [1.15, 1.15], color="#1a1a1a", lw=0.7, transform=cax.transAxes,
                 clip_on=False, zorder=5)
        cax.text((x0 + x1) / 2, 1.25, "neutral", transform=cax.transAxes, ha="center",
                 va="bottom", fontsize=5, zorder=5)


def legend_status(fig, H, eco_any_no_plots, y_in, tied_panels="a, c", total_panel="d"):
    """Tied / low n / no plots key, shared by both maps (and the strip in a of Figure 1)."""
    pct = round((FRAGILE_MARGIN - 1) * 100)
    handles = []
    if tied_panels:
        handles.append(Patch(facecolor=FRAGILE_COLOR, edgecolor=EDGE, linewidth=0.4,
                             label=f"Tied (< {pct}% ahead; {tied_panels})"))
    handles.append(Patch(facecolor=LOW_N_FC, edgecolor=LOW_N_EC, hatch=LOW_N_HATCH, linewidth=0.4,
                         label=f"< {MIN_PLOTS_PER_ECOREGION} plots"))
    if eco_any_no_plots:
        handles.append(Patch(facecolor=NO_PLOTS_FC, edgecolor=NO_PLOTS_EC, linewidth=0.4, label="No plots"))
    if SHOW_UNCERTAIN:
        handles.append(Patch(facecolor="white", edgecolor=UNCERTAIN_HATCH_COLOR, hatch=UNCERTAIN_HATCH,
                             linewidth=0.4, label=f"{CI_PCT}% CI includes 0 ({total_panel})"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(_layout()["status_x"], y_in / H),
               ncol=len(handles),
               frameon=False, fontsize=6, handlelength=1.4, handleheight=0.9, columnspacing=1.0,
               borderaxespad=0)


# ---------------------------------------------------------------
# Figures
# ---------------------------------------------------------------
_LAYOUT_STATE = {"dominant": True}      # set by each figure builder


def _layout():
    """Horizontal positions (figure fractions) of the maps, heatmap and legends."""
    if not _LAYOUT_STATE["dominant"]:       # total-effect map + heatmap only
        mw = 0.6 if SHOW_DIRECTION_HEATMAP else 0.8
        return dict(map_w=mw, map_c_x=None, map_d_x=0.005,
                    heat_x=0.005 + mw + 0.12, heat_w=0.24,
                    key_x=None, key_w=None, bar_x=None, bar_w=None,
                    cbar_x=0.15, cbar_w=0.3, status_x=0.32)
    if SHOW_DIRECTION_HEATMAP:
        mw = 0.375
        return dict(map_w=mw, map_c_x=0.005, map_d_x=0.005 + mw + 0.01,
                    heat_x=0.005 + 2 * mw + 0.01 + 0.085, heat_w=0.145,
                    key_x=0.155, key_w=0.13, bar_x=0.305, bar_w=0.07,
                    cbar_x=0.475, cbar_w=0.2, status_x=0.6)
    return dict(map_w=0.49, map_c_x=0.005, map_d_x=0.505, heat_x=None, heat_w=None,
                key_x=0.15, key_w=0.17, bar_x=0.355, bar_w=0.095,
                cbar_x=0.64, cbar_w=0.24, status_x=0.76)


def _map_dims(extent, metrics):
    """Map width / height and legend strip height in inches, shared by both figures."""
    map_w_in = FIG_WIDTH_IN * _layout()["map_w"]
    map_h_in = map_w_in * (extent[3] - extent[2]) / (extent[1] - extent[0])
    legend_in = 0.80 + 0.14 * len(metrics)
    return map_h_in, legend_in


def _draw_maps_row(fig, H, y_maps_in, map_h_in, eco_dom, eco_tot, extent, us_poly, mnorm, tnorm,
                   heat_counts=None, heat_thr=None):
    L = _layout()
    if L["map_c_x"] is not None:
        ax_c = fig.add_axes([L["map_c_x"], y_maps_in / H, L["map_w"], map_h_in / H])
        map_dominant(ax_c, eco_dom, extent, us_poly, mnorm)
    ax_d = fig.add_axes([L["map_d_x"], y_maps_in / H, L["map_w"], map_h_in / H])
    map_total(ax_d, eco_tot, extent, us_poly, tnorm)
    if SHOW_DIRECTION_HEATMAP and heat_counts is not None:
        n_d, n_f = heat_counts.shape
        h_frac = L["heat_w"] * FIG_WIDTH_IN * n_d / n_f / H        # square cells
        y0 = (y_maps_in + 0.5 * map_h_in) / H - 0.5 * h_frac + 0.1 / H
        ax_h = fig.add_axes([L["heat_x"], y0, L["heat_w"], h_frac])
        panel_direction_heatmap(ax_h, heat_counts, heat_thr)


def build_figure(eco_dom, eco_tot, conf, trend, metrics, national_means, us_poly, norms, out_stem,
                 heat_counts=None, heat_thr=None):
    """Figure 1: panels a-d (+ e)."""
    _set_style()
    _LAYOUT_STATE["dominant"] = True
    (mnorm, mticks, scale, exp), (tnorm, tticks) = norms
    extent = _map_extent(eco_dom, us_poly)

    # Vertical layout in inches (bottom -> top): legends, maps, panel b, panel a
    W = FIG_WIDTH_IN
    map_h_in, legend_in = _map_dims(extent, metrics)
    maps_gap_in = 0.22
    b_in, a_in = 1.55, 1.85
    H = legend_in + map_h_in + maps_gap_in + b_in + a_in
    fig = plt.figure(figsize=(W, H))

    y_maps = legend_in
    y_b = y_maps + map_h_in + maps_gap_in
    y_a = y_b + b_in
    px, pw = 0.15, 0.58                         # shared x-position of panels a and b

    lax_a = panel_dominant_conifer(fig, [px, (y_a + 0.22) / H, pw, (a_in - 0.32) / H],
                                   conf, metrics, mnorm, scale, exp)
    ax_b = fig.add_axes([px, (y_b + 0.38) / H, pw, (b_in - 0.48) / H], sharex=lax_a)
    panel_total_conifer(ax_b, trend, metrics, tnorm)

    _draw_maps_row(fig, H, y_maps, map_h_in, eco_dom, eco_tot, extent, us_poly, mnorm, tnorm,
                   heat_counts, heat_thr)

    legend_dominant(fig, H, metrics, mnorm, mticks, scale, exp, national_means, y_in=0.10, tag="c")
    legend_total(fig, H, tnorm, tticks, y_in=0.42, tag="d", neutral=heat_thr)
    legend_status(fig, H, bool(eco_dom.no_plots.any()), y_in=0.06, tied_panels="a, c", total_panel="d")

    letter = dict(fontsize=10, fontweight="bold", va="top", ha="left")
    fig.text(0.01, (y_a + a_in - 0.02) / H, "a", **letter)
    fig.text(0.01, (y_b + b_in - 0.02) / H, "b", **letter)
    L = _layout()
    fig.text(L["map_c_x"] + 0.005, (y_maps + map_h_in + maps_gap_in - 0.02) / H, "c", **letter)
    fig.text(L["map_d_x"] + 0.005, (y_maps + map_h_in + maps_gap_in - 0.02) / H, "d", **letter)
    if SHOW_DIRECTION_HEATMAP and heat_counts is not None:
        fig.text(L["heat_x"] - 0.07, (y_maps + map_h_in + maps_gap_in - 0.02) / H, "e", **letter)

    _save(fig, out_stem)
    return fig


def build_maps_figure(eco_dom, eco_tot, metrics, national_means, us_poly, norms, out_stem,
                      heat_counts=None, heat_thr=None):
    """Figure 2: total-effect map (+ heatmap); the dominant-metric map only if MAPS_FIG_SHOW_DOMINANT."""
    _set_style()
    _LAYOUT_STATE["dominant"] = MAPS_FIG_SHOW_DOMINANT
    (mnorm, mticks, scale, exp), (tnorm, tticks) = norms
    extent = _map_extent(eco_dom, us_poly)

    # Vertical layout in inches (bottom -> top): legends, maps, space for panel letters
    W = FIG_WIDTH_IN
    map_h_in, legend_in = _map_dims(extent, metrics)
    top_in = 0.22
    H = legend_in + map_h_in + top_in
    fig = plt.figure(figsize=(W, H))

    y_maps = legend_in
    _draw_maps_row(fig, H, y_maps, map_h_in, eco_dom, eco_tot, extent, us_poly, mnorm, tnorm,
                   heat_counts, heat_thr)

    letter = dict(fontsize=10, fontweight="bold", va="top", ha="left")
    L = _layout()
    letters = iter("abc")
    if MAPS_FIG_SHOW_DOMINANT:
        tag_dom = next(letters)
        fig.text(L["map_c_x"] + 0.005, (H - 0.02) / H, tag_dom, **letter)
        legend_dominant(fig, H, metrics, mnorm, mticks, scale, exp, national_means, y_in=0.10,
                        tag=tag_dom)
    tag_tot = next(letters)
    fig.text(L["map_d_x"] + 0.005, (H - 0.02) / H, tag_tot, **letter)
    legend_total(fig, H, tnorm, tticks, y_in=0.42, tag=tag_tot, neutral=heat_thr)
    legend_status(fig, H, bool(eco_dom.no_plots.any()), y_in=0.06,
                  tied_panels=tag_dom if MAPS_FIG_SHOW_DOMINANT else None, total_panel=tag_tot)
    if SHOW_DIRECTION_HEATMAP and heat_counts is not None:
        fig.text(L["heat_x"] - 0.07, (H - 0.02) / H, next(letters), **letter)

    _save(fig, out_stem)
    return fig


def prepare_for_drawing(eco, us_poly):
    """Simplified, east-limited and (if STATES_PATH) US-clipped copy, for drawing only."""
    d = eco.copy()
    d["geometry"] = eco.geometry.simplify(SIMPLIFY_M, preserve_topology=True)
    d = clip_east_of(d)
    if us_poly is not None:
        d = gpd.clip(d, us_poly)
        return d[~d.geometry.is_empty]
    return d[~d.no_plots]


# ---------------------------------------------------------------
# Heatmaps by rarity class
# ---------------------------------------------------------------
def attach_rarity(plots, path=RARE_PATH, col=RARE_COL):
    """Adds % rare species (0-100) to each plot."""
    rare = pd.read_parquet(path, columns=["PID", col]).drop_duplicates("PID")
    out = plots.merge(rare, on="PID", how="left")
    if out[col].max() <= 1:
        out[col] = out[col] * 100
    print(f"[rarity] {int(out[col].isna().sum())} plots without % rare species")
    return out


def segment_classes(values, cuts, names, fmt, unit=""):
    """(class id per plot, labels with ranges) for any segmenting variable: tertiles when
    cuts is None, else the given cut points. fmt formats a cut value, unit follows it."""
    v = np.asarray(values, dtype=float)
    cuts = list(np.nanpercentile(v, [100 / 3, 200 / 3])) if cuts is None else list(cuts)
    ids = np.digitize(v, cuts)
    if len(names) != len(cuts) + 1:
        names = [f"Class {k + 1}" for k in range(len(cuts) + 1)]
    f = lambda x: format(x, fmt).replace("-", "\u2212")
    labels = []
    for k, nm in enumerate(names):
        if k == 0:
            labels.append(f"{nm}\n(< {f(cuts[0])}{unit})")
        elif k == len(cuts):
            labels.append(f"{nm}\n(\u2265 {f(cuts[-1])}{unit})")
        else:
            labels.append(f"{nm}\n({f(cuts[k - 1])} to {f(cuts[k])}{unit})")
    return ids, labels


def rarity_classes(values):
    """(class id per plot, labels with ranges): tertiles, or RARITY_CUTS."""
    return segment_classes(values, RARITY_CUTS, RARITY_NAMES, ".0f", "% rare")


def wsci_classes(values):
    """(class id per plot, labels with ranges): tertiles, or WSCI_CUTS."""
    return segment_classes(values, WSCI_CUTS, WSCI_NAMES, ".2g")


def attach_wsci(plots, path=WSCI_PATH, col=WSCI_COL):
    """Adds the WSCI value of each plot (from a .csv or .parquet with PID and col)."""
    w = (pd.read_parquet(path, columns=["PID", col]) if str(path).endswith(".parquet")
         else pd.read_csv(path, usecols=["PID", col]))
    out = plots.drop(columns=[col], errors="ignore").merge(w.drop_duplicates("PID"), on="PID",
                                                            how="left")
    print(f"[wsci] {int(out[col].isna().sum())} plots without {col}")
    return out


RARITY_ROW_LEFT, RARITY_ROW_RIGHT, RARITY_ROW_GAP = 0.12, 0.99, 0.03   # figure fractions
RARITY_ROW_TOP_IN, RARITY_ROW_BOTTOM_IN = 0.5, 0.75                      # titles; tick labels + footnote


def segmented_heatmap_counts(plots, iv, col, classes_fn, seg_name):
    """Per class of col: direction x forest-type counts. Returns (list of counts, labels, table)."""
    p = plots.dropna(subset=[col])
    ids, labels = classes_fn(p[col])
    tables, counts = [], []
    for k, lab in enumerate(labels):
        print(f"[{seg_name} heatmap] {lab.replace(chr(10), ' ')}:")
        c, t = direction_by_forest(p[ids == k], iv)
        counts.append(c)
        tables.append(t.assign(**{f"{seg_name}_class": lab.replace("\n", " ")}))
    return counts, labels, pd.concat(tables, ignore_index=True)


def rarity_heatmap_counts(plots, iv):
    return segmented_heatmap_counts(plots, iv, RARE_COL, rarity_classes, "rarity")


def _rarity_row_height_in(counts):
    n = len(counts)
    n_d, n_f = counts[0].shape
    w = (RARITY_ROW_RIGHT - RARITY_ROW_LEFT - RARITY_ROW_GAP * (n - 1)) / n
    return w, w * FIG_WIDTH_IN * n_d / n_f                   # square cells


def _draw_rarity_row(fig, H, y0_in, counts, labels, iv, letters):
    """The rarity heatmaps side by side, bottoms at y0_in (inches); letters above each."""
    w, h_in = _rarity_row_height_in(counts)
    for k in range(len(counts)):
        x = RARITY_ROW_LEFT + k * (w + RARITY_ROW_GAP)
        ax = fig.add_axes([x, y0_in / H, w, h_in / H])
        ramp = RARITY_RAMP if "rar" in labels[k].lower() else WSCI_RAMP
        panel_direction_heatmap(ax, counts[k], iv, title=labels[k], footnote=False, ylabels=(k == 0),
                                swatch=ramp[k % len(ramp)])
        fig.text(x - (0.1 if k == 0 else 0.0), (y0_in + h_in + RARITY_ROW_TOP_IN - 0.05) / H,
                 letters[k], fontsize=10, fontweight="bold", va="top", ha="left")


def _rarity_footnote(fig, H, iv, seg="rarity class"):
    fig.text(0.5, 0.04 / H, f"% of each forest type's plots within the {seg}; neutral = effect "
             f"{_interval_text(iv)}; forest type = plot % conifer; n = plots",
             ha="center", va="bottom", fontsize=5.5, color="#555555")


def build_rarity_heatmaps(counts, labels, iv, out_stem, seg="rarity class"):
    """The plot-level direction x forest-type heatmap repeated for each rarity class, side by side."""
    _set_style()
    _, h_in = _rarity_row_height_in(counts)
    H = h_in + RARITY_ROW_TOP_IN + RARITY_ROW_BOTTOM_IN
    fig = plt.figure(figsize=(FIG_WIDTH_IN, H))
    _draw_rarity_row(fig, H, RARITY_ROW_BOTTOM_IN, counts, labels, iv, "abcdef")
    _rarity_footnote(fig, H, iv, seg)
    _save(fig, out_stem)
    return fig


def _vertical_total_colorbar(fig, rect, tnorm, tticks, neutral=None):
    """Vertical version of map d's colour bar (stabilising at the top), with the neutral bracket."""
    cax = fig.add_axes(rect)
    cb = fig.colorbar(ScalarMappable(norm=tnorm, cmap=_OUTCOME["cmap"]), cax=cax, orientation="vertical")
    cb.set_ticks(tticks)
    cb.ax.yaxis.set_major_formatter(FuncFormatter(_pct_fmt))
    cb.ax.tick_params(labelsize=5.5, length=2, width=0.4, pad=1)
    cb.outline.set_linewidth(0.4)
    scale_note = ", log colour scale" if LOG_COLOR_SCALE else ""
    cax.yaxis.set_label_position("left")
    cax.set_ylabel(f"{_OUTCOME['title']}\n({_OUTCOME['units']}{scale_note})",
                   fontsize=6.5, labelpad=12 if neutral is not None else 4, linespacing=1.15)
    cax.text(0.5, 1.03, _OUTCOME["pos"], transform=cax.transAxes, ha="center", va="bottom", fontsize=6,
             color=_OUTCOME["pos_color"])
    cax.text(0.5, -0.03, _OUTCOME["neg"], transform=cax.transAxes, ha="center", va="top", fontsize=6,
             color=_OUTCOME["neg_color"])
    if neutral is not None and MARK_NEUTRAL_ON_COLORBAR:
        lo, hi = (float(np.clip(x, tnorm.vmin, tnorm.vmax)) for x in neutral)
        y0, y1 = (cax.transAxes.inverted().transform(cax.transData.transform((0, y)))[1]
                  for y in (lo, hi))
        for y in (y0, y1):
            cax.plot([-0.25, 1.15], [y, y], color="#1a1a1a", lw=0.7, transform=cax.transAxes,
                     clip_on=False, zorder=5)
        cax.plot([-0.25, -0.25], [y0, y1], color="#1a1a1a", lw=0.7, transform=cax.transAxes,
                 clip_on=False, zorder=5)
        cax.text(-0.4, (y0 + y1) / 2, "neutral", transform=cax.transAxes, rotation=90,
                 ha="right", va="center", fontsize=5, zorder=5)
    return cax


def _anchor_point(geom):
    """A point inside the ecoregion's largest part (keeps circles off small islands)."""
    if geom.geom_type == "MultiPolygon":
        geom = max(geom.geoms, key=lambda g: g.area)
    return geom.representative_point()


_RAW_CACHE = {}


def _read_raw(cols):
    """PID + the requested raw columns from RAW_DATA_PATH (cached)."""
    key = tuple(cols)
    if key not in _RAW_CACHE:
        head = pd.read_csv(RAW_DATA_PATH, nrows=0).columns
        missing = [c for c in cols if c not in head]
        if missing:
            raise KeyError(f"{missing} not in {RAW_DATA_PATH}. Available columns include: "
                           f"{', '.join(map(str, list(head)[:40]))}")
        _RAW_CACHE[key] = pd.read_csv(RAW_DATA_PATH, usecols=["PID"] + list(cols)).drop_duplicates("PID")
    return _RAW_CACHE[key]


def raw_fill(joined, eco_draw, kind):
    """
    Ecoregion median of the RAW outcome for the bubble-map fill. kind = "stability"
    (1 / NPP SD) or "wsci". Returns a dict with the ecoregions (column raw_fill), norm,
    colour map and labels.
    """
    if kind == "stability":
        r = _read_raw([RAW_STD_NPP_COL])
        v = 1.0 / r[RAW_STD_NPP_COL].where(r[RAW_STD_NPP_COL] > 0)
        cmap, label, hi, lo = RAW_STABILITY_CMAP, "Stability (1 / NPP SD)", "More stable", "Less stable"
    else:
        r = _read_raw([RAW_WSCI_COL])
        v = r[RAW_WSCI_COL]
        cmap, label, hi, lo = RAW_WSCI_CMAP, "Structural complexity (WSCI)", "More complex", "Less complex"
    per_plot = pd.DataFrame({"PID": r["PID"], "_v": v}).merge(joined[["PID", "eco_id"]], on="PID")
    med = per_plot.groupby("eco_id")["_v"].median()
    eco = eco_draw.copy()
    eco["raw_fill"] = eco["eco_id"].map(med)
    ok = eco.loc[~eco.no_plots & ~eco.low_n, "raw_fill"].dropna()
    lo_v, hi_v = np.percentile(ok, RAW_FILL_PCT) if len(ok) else (0.0, 1.0)
    print(f"[bubble map] fill = ecoregion median raw {label}: {lo_v:.3g} to {hi_v:.3g} "
          f"(P{RAW_FILL_PCT[0]}-P{RAW_FILL_PCT[1]})")
    return dict(eco=eco, norm=mcolors.Normalize(lo_v, hi_v), cmap=cmap,
                label=f"{label}\necoregion median (raw)", hi=hi, lo=lo)


def map_raw(ax, eco, extent, us_poly, fill):
    """Ecoregions coloured by their median raw outcome (sequential)."""
    _setup_map(ax, eco, extent)
    colored = eco[~eco.no_plots & ~eco.low_n & eco["raw_fill"].notna()]
    colored.plot(ax=ax, color=[fill["cmap"](fill["norm"](v)) for v in colored["raw_fill"]],
                 edgecolor=EDGE, linewidth=0.15)
    _outline(ax, us_poly, eco.crs)
    _scale_bar(ax, extent)


def _vertical_seq_colorbar(fig, rect, fill):
    cax = fig.add_axes(rect)
    cb = fig.colorbar(ScalarMappable(norm=fill["norm"], cmap=fill["cmap"]), cax=cax,
                      orientation="vertical", extend="both")
    cb.ax.tick_params(labelsize=5.5, length=2, width=0.4, pad=1)
    cb.ax.yaxis.set_major_locator(MaxNLocator(5))
    cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.3g}"))
    cb.outline.set_linewidth(0.4)
    cax.set_ylabel(fill["label"], fontsize=6.5, labelpad=4, linespacing=1.15)
    cax.text(0.5, 1.05, fill["hi"], transform=cax.transAxes, ha="center", va="bottom", fontsize=6,
             color=fill["cmap"](0.85))
    cax.text(0.5, -0.05, fill["lo"], transform=cax.transAxes, ha="center", va="top", fontsize=6,
             color="#7f7f7f")
    return cax


def _ols_hc3(X, y):
    """OLS coefficients and HC3 standard errors (numpy only)."""
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    e = y - X @ beta
    h = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
    w = (e / np.clip(1 - h, 1e-8, None)) ** 2
    cov = XtX_inv @ (X.T * w) @ X @ XtX_inv
    return beta, np.sqrt(np.clip(np.diag(cov), 0, None))


def local_slopes(effects_csv, joined, metrics, min_plots=MIN_PLOTS_PER_ECOREGION):
    """
    Per ecoregion: the slope of the residual outcome on each metric from that ecoregion's plots,
    as the effect of +1 national SD of the metric, with a CI; the dominant metric (largest
    |effect|), its direction (+1 / -1, 0 = CI includes 0) and whether it is tied.
    """
    cols = ["PID", "diversity_metric", "diversity_residual", "stability_residual"]
    head = pd.read_csv(effects_csv, nrows=0).columns
    missing = [c for c in cols if c not in head]
    if missing:
        raise KeyError(f"{effects_csv} lacks {missing}: rerun the effects script (it writes the "
                       f"residuals to the per-observation files), or set BUBBLE_DIRECTION = False.")
    e = pd.read_csv(effects_csv, usecols=cols)
    D = e.pivot(index="PID", columns="diversity_metric", values="diversity_residual")[metrics]
    S = e.groupby("PID")["stability_residual"].first()      # already signed: + = more of the outcome
    W = D.join(S.rename("_S")).dropna()
    sd = W[metrics].std(ddof=1)                              # national SD of each metric
    W = W.join(joined.drop_duplicates("PID").set_index("PID")["eco_id"], how="inner")
    z = _norm.ppf(1 - (1 - BUBBLE_LOCAL_CI) / 2)
    rows = []
    for eco_id, g in W.groupby("eco_id"):
        n = len(g)
        if n < max(min_plots, len(metrics) + 5):
            continue
        y = g["_S"].to_numpy(float)
        est, se = {}, {}
        if BUBBLE_LOCAL_MODEL == "adjusted":
            X = np.column_stack([np.ones(n), g[metrics].to_numpy(float)])
            b, s_ = _ols_hc3(X, y)
            for k, m in enumerate(metrics):
                est[m], se[m] = b[k + 1], s_[k + 1]
        else:
            for m in metrics:
                X = np.column_stack([np.ones(n), g[m].to_numpy(float)])
                b, s_ = _ols_hc3(X, y)
                est[m], se[m] = b[1], s_[1]
        eff = {m: est[m] * sd[m] for m in metrics}           # per 1 national SD of the metric
        ranked = sorted(metrics, key=lambda m: -abs(eff[m]))
        win, run = ranked[0], ranked[1] if len(ranked) > 1 else ranked[0]
        lo, hi = (est[win] - z * se[win]) * sd[win], (est[win] + z * se[win]) * sd[win]
        direction = 1 if lo > 0 else (-1 if hi < 0 else 0)
        rows.append({"eco_id": eco_id, "n_plots": n, "winner": win, "runner_up": run,
                     "effect": eff[win], "ci_lower": lo, "ci_upper": hi, "value": abs(eff[win]),
                     "direction": direction,
                     "fragile": abs(eff[win]) < FRAGILE_MARGIN * abs(eff[run]) if run != win else False,
                     **{f"effect__{m}": eff[m] for m in metrics}})
    out = pd.DataFrame(rows)
    if len(out):
        print(f"[local slopes] {len(out)} ecoregions; dominant metric up / down / CI includes 0: "
              f"{int((out.direction > 0).sum())} / {int((out.direction < 0).sum())} / "
              f"{int((out.direction == 0).sum())}")
    return out


def build_bubble_map(eco_dom, eco_tot, us_poly, norms, metrics, out_stem, iv=None, side=None,
                     fill=None, local=None):
    """
    One map: ecoregion fill = total diversity effect (colour bar on the right); circle =
    dominant metric (colour) and its mean |effect| (circle AREA proportional to |effect|).
    side = (counts, labels, iv) adds the rarity heatmaps stacked on the left (panels a-c,
    map = d); None = map only.
    """
    _set_style()
    _, (tnorm, tticks) = norms
    extent = _map_extent(eco_tot, us_poly)
    W = FIG_WIDTH_IN
    aspect = (extent[3] - extent[2]) / (extent[1] - extent[0])

    if side is None:
        map_x, map_w, cb_x = 0.05, 0.78, 0.86
        map_h_in = W * map_w * aspect
        legend_in, top_in = 0.75, 0.12
        H = legend_in + map_h_in + top_in
        y_map = legend_in
    else:
        counts, labels, s_iv = side
        n_d, n_f = counts[0].shape
        heat_x, heat_w = 0.115, 0.135                      # stacked heatmaps (left column)
        cell_in = heat_w * W / n_f
        heat_h_in = cell_in * n_d
        title_in, xlab_in, gap_in = 0.34, 0.30, 0.10
        block_in = title_in + heat_h_in + xlab_in + gap_in
        foot_in = 0.30
        stack_in = len(counts) * block_in + foot_in
        map_x, map_w, cb_x = 0.29, 0.555, 0.865
        map_h_in = W * map_w * aspect
        legend_in = 0.75
        H = max(stack_in, legend_in + map_h_in + 0.3)
        y_map = legend_in + 0.5 * (H - legend_in - map_h_in)
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([map_x, y_map / H, map_w, map_h_in / H])
    if fill is None:
        map_total(ax, eco_tot, extent, us_poly, tnorm)          # fill = total diversity effect
    else:
        map_raw(ax, fill["eco"], extent, us_poly, fill)         # fill = median raw outcome

    # Circles: one per ecoregion with enough plots
    if local is not None and len(local):        # local slopes: dominant metric + direction
        geo = eco_dom[~eco_dom.no_plots][["eco_id", "geometry"]]
        geo = geo.dissolve(by="eco_id").reset_index() if geo["eco_id"].duplicated().any() else geo
        b = geo.merge(local, on="eco_id", how="inner")
        eco_dom = b                             # (only used below for the circles)
    b = eco_dom[eco_dom["value"].notna()].copy() if local is not None else \
        eco_dom[~eco_dom.no_plots & ~eco_dom.low_n & eco_dom["value"].notna()].copy()
    b = b.dissolve(by="eco_id", aggfunc="first") if b["eco_id"].duplicated().any() else b
    pts = b.geometry.apply(_anchor_point)
    vals = b["value"].to_numpy(float)
    vref = float(np.nanpercentile(vals, BUBBLE_SIZE_PCT)) if len(vals) else 1.0
    size = lambda v: BUBBLE_MIN_SIZE + (BUBBLE_MAX_SIZE - BUBBLE_MIN_SIZE) * np.clip(v / vref, 0, 1)
    by_conifer = BUBBLE_COLOUR_BY == "conifer"
    if by_conifer:
        src = fill["eco"] if fill is not None else eco_tot
        conif = (src.drop_duplicates("eco_id").set_index("eco_id")["conifer_mean"]
                 if "conifer_mean" in src else pd.Series(dtype=float))
        b_ids = b.index if b.index.name == "eco_id" else b["eco_id"]
        cvals = np.asarray(pd.Series(b_ids).map(conif), dtype=float)
        cnorm = mcolors.Normalize(0, 100)
        colors = [CONIFER_CMAP(cnorm(c)) if np.isfinite(c) else "#bdbdbd" for c in cvals]
    else:
        colors = [FRAGILE_COLOR if f else METRIC_COLORS.get(w, "#555555")
                  for w, f in zip(b["winner"], b["fragile"])]
    order = np.argsort(-vals)                                   # big circles underneath
    px, py = pts.x.to_numpy(), pts.y.to_numpy()
    if local is not None and "direction" in b:
        dirs = b["direction"].to_numpy()
        for d, mk in ((1, "^"), (-1, "v")):
            o = [i for i in order if dirs[i] == d]
            if o:
                ax.scatter(px[o], py[o], s=size(vals)[o] * 1.3, marker=mk, c=[colors[i] for i in o],
                           alpha=BUBBLE_ALPHA, edgecolors="white", linewidths=0.6, zorder=6)
        o = [i for i in order if dirs[i] == 0]
        if o:
            ax.scatter(px[o], py[o], s=BUBBLE_NEUTRAL_SIZE, marker="o", c="#9e9e9e",
                       edgecolors="white", linewidths=0.4, zorder=6)
    else:
        ax.scatter(px[order], py[order], s=size(vals)[order],
                   c=[colors[i] for i in order], alpha=BUBBLE_ALPHA, edgecolors="white",
                   linewidths=0.6, zorder=6)

    # Colour bar (fill) on the right of the map, with the "< 20 plots" key under it
    cb_h = 0.62 * map_h_in
    cb_y = y_map + 0.5 * (map_h_in - cb_h) + 0.1
    if fill is None:
        cax = _vertical_total_colorbar(fig, [cb_x, cb_y / H, 0.016, cb_h / H], tnorm, tticks, neutral=iv)
    else:
        cax = _vertical_seq_colorbar(fig, [cb_x, cb_y / H, 0.016, cb_h / H], fill)
    cax.yaxis.set_label_position("right")
    hatch = [Patch(facecolor=LOW_N_FC, edgecolor=LOW_N_EC, hatch=LOW_N_HATCH, linewidth=0.4,
                   label=f"< {MIN_PLOTS_PER_ECOREGION} plots")]
    fig.legend(handles=hatch, loc="upper left", bbox_to_anchor=(cb_x - 0.02, (cb_y - 0.22) / H),
               frameon=False, fontsize=6, handlelength=1.4, handleheight=0.9, borderaxespad=0)

    # Legends under the map: circle colour (dominant metric) and circle size (|effect|)
    exp = int(np.floor(np.log10(vref))) if vref > 0 else 0
    sc = 10.0 ** exp
    col_mk = "s" if local is not None else "o"
    col_handles = [Line2D([], [], ls="none", marker=col_mk, markersize=6.5, markerfacecolor=METRIC_COLORS[m],
                          markeredgecolor="white", label=m) for m in metrics]
    if b["fragile"].any():
        pct = round((FRAGILE_MARGIN - 1) * 100)
        col_handles.append(Line2D([], [], ls="none", marker=col_mk, markersize=6.5,
                                  markerfacecolor=FRAGILE_COLOR, markeredgecolor="white",
                                  label=f"Tied (< {pct}% ahead)"))
    if local is not None:                       # marker shape = direction of the local effect
        ci_txt = f"{round(BUBBLE_LOCAL_CI * 100)}% CI"
        col_handles += [
            Line2D([], [], ls="none", marker="^", markersize=7, markerfacecolor="#6b6b6b",
                   markeredgecolor="white", label=f"{_OUTCOME['pos']} ({ci_txt} > 0)"),
            Line2D([], [], ls="none", marker="v", markersize=7, markerfacecolor="#6b6b6b",
                   markeredgecolor="white", label=f"{_OUTCOME['neg']} ({ci_txt} < 0)"),
            Line2D([], [], ls="none", marker="o", markersize=np.sqrt(BUBBLE_NEUTRAL_SIZE),
                   markerfacecolor="#9e9e9e", markeredgecolor="white", label=f"{ci_txt} includes 0")]
    ref_vals = [v for v in MaxNLocator(3, steps=[1, 2, 2.5, 5, 10]).tick_values(0, vref / sc)[1:]
                if v * sc <= vref * 1.0001][:3] or [vref / sc]
    size_handles = [Line2D([], [], ls="none", marker="^" if local is not None else "o",
                           markersize=np.sqrt(size(v * sc) * (1.3 if local is not None else 1)),
                           markerfacecolor="#bdbdbd", markeredgecolor="white", label=f"{v:g}")
                    for v in ref_vals]
    if by_conifer:      # small colour bar for % conifer under the map
        kax = fig.add_axes([map_x + 0.07, 0.2 / H, 0.17, 0.07 / H])
        kb = fig.colorbar(ScalarMappable(norm=mcolors.Normalize(0, 100), cmap=CONIFER_CMAP), cax=kax,
                          orientation="horizontal")
        kb.set_ticks([0, 25, 50, 75, 100])
        kb.ax.tick_params(labelsize=5.5, length=2, width=0.4, pad=1)
        kb.outline.set_linewidth(0.4)
        kax.xaxis.set_label_position("top")
        kax.set_xlabel("Circle colour: ecoregion % conifer", fontsize=6.5, labelpad=3)
        kax.text(-0.03, 0.5, "Broadleaf", transform=kax.transAxes, ha="right", va="center",
                 fontsize=5.5, color=FOREST_COLORS["Broadleaf"])
        kax.text(1.03, 0.5, "Conifer", transform=kax.transAxes, ha="left", va="center", fontsize=5.5,
                 color=FOREST_COLORS["Conifer"])
    else:
        leg1 = fig.legend(handles=col_handles, loc="lower left", bbox_to_anchor=(map_x, 0.12 / H),
                          ncol=2 if side is not None else len(col_handles), frameon=False, fontsize=6.5,
                          handletextpad=0.3, columnspacing=1.0, borderaxespad=0,
                          title=("Colour: dominant metric (local slope); shape: its direction"
                                 if local is not None else "Circle colour: dominant diversity metric"),
                          title_fontsize=6.5)
        leg1._legend_box.align = "left"
    fig.legend(handles=size_handles, loc="lower right", bbox_to_anchor=(map_x + map_w, 0.12 / H),
               ncol=len(size_handles), frameon=False, fontsize=6.5, handletextpad=0.3,
               columnspacing=1.0, borderaxespad=0,
               title=(f"Size: |local effect| per 1 SD (\u00d710$^{{{exp}}}$)" if local is not None
                      else f"Circle size: |effect| of dominant metric (\u00d710$^{{{exp}}}$)"
                      if by_conifer else f"Circle size: mean |effect| (\u00d710$^{{{exp}}}$)"),
               title_fontsize=6.5)

    # Rarity heatmaps stacked on the left (a-c), map = d
    if side is not None:
        top = H
        for k in range(len(counts)):
            y0 = top - (k + 1) * block_in + xlab_in + gap_in
            hax = fig.add_axes([heat_x, y0 / H, heat_w, heat_h_in / H])
            panel_direction_heatmap(hax, counts[k], s_iv, title=labels[k].replace("\n", " "),
                                    footnote=False, compact=True,
                                    swatch=RARITY_RAMP[k % len(RARITY_RAMP)])
            fig.text(0.01, (y0 + heat_h_in + title_in - 0.04) / H, "abcdefgh"[k], fontsize=10,
                     fontweight="bold", va="top", ha="left")
        fig.text(map_x - 0.01, (y_map + map_h_in + 0.05) / H, "abcdefgh"[len(counts)], fontsize=10,
                 fontweight="bold", va="bottom", ha="left")
        fig.text(heat_x + heat_w / 2, 0.04 / H,
                 f"% of each forest type's plots within the rarity class;\nneutral = effect "
                 f"{_interval_text(s_iv)}; n = plots", ha="center", va="bottom", fontsize=5,
                 color="#555555", linespacing=1.1)
    _save(fig, out_stem)
    return fig


def build_map_rarity_figure(eco_tot, us_poly, norms, counts, labels, iv, out_stem, seg="rarity class"):
    """
    (a) total-effect map with a vertical colour bar on its left, above (b-d) the three rarity
    heatmaps. Same colour scale and neutral interval as the other figures.
    """
    _set_style()
    _, (tnorm, tticks) = norms
    extent = _map_extent(eco_tot, us_poly)
    W = FIG_WIDTH_IN
    map_x, map_w = 0.24, 0.72                                   # figure fractions
    map_w_in = W * map_w
    map_h_in = map_w_in * (extent[3] - extent[2]) / (extent[1] - extent[0])
    _, heat_h_in = _rarity_row_height_in(counts)
    top_in, gap_in = 0.12, 0.25
    y_heat = RARITY_ROW_BOTTOM_IN
    y_map = y_heat + heat_h_in + RARITY_ROW_TOP_IN + gap_in
    H = y_map + map_h_in + top_in
    fig = plt.figure(figsize=(W, H))

    ax = fig.add_axes([map_x, y_map / H, map_w, map_h_in / H])
    map_total(ax, eco_tot, extent, us_poly, tnorm)
    fig.text(map_x - 0.005, (H - 0.04) / H, "a", fontsize=10, fontweight="bold", va="top", ha="left")

    cb_h = 0.62 * map_h_in
    _vertical_total_colorbar(fig, [0.155, (y_map + 0.5 * (map_h_in - cb_h) + 0.1) / H, 0.016, cb_h / H],
                             tnorm, tticks, neutral=iv)
    handles = [Patch(facecolor=LOW_N_FC, edgecolor=LOW_N_EC, hatch=LOW_N_HATCH, linewidth=0.4,
                     label=f"< {MIN_PLOTS_PER_ECOREGION} plots")]
    if eco_tot.no_plots.any():
        handles.append(Patch(facecolor=NO_PLOTS_FC, edgecolor=NO_PLOTS_EC, linewidth=0.4, label="No plots"))
    fig.legend(handles=handles, loc="upper left",
               bbox_to_anchor=(0.12, (y_map + 0.5 * (map_h_in - cb_h) - 0.12) / H),
               frameon=False, fontsize=6, handlelength=1.4, handleheight=0.9, borderaxespad=0)

    _draw_rarity_row(fig, H, y_heat, counts, labels, iv, "bcdefg")
    _rarity_footnote(fig, H, iv, seg)
    _save(fig, out_stem)
    return fig


# ---------------------------------------------------------------
# Run
# ---------------------------------------------------------------
def run(effects_csv=EFFECTS_CSV, ecoregions_path=ECOREGIONS_PATH, states_path=STATES_PATH,
        outdir=OUTDIR):
    plots, metrics = load_plots(effects_csv)
    plots = attach_conifer(plots)
    eco = read_ecoregions(ecoregions_path, plots)
    joined = assign_ecoregions(plots, eco)          # the ONE set of plots used by every panel

    eco_dom = summarize_dominant(joined, eco, metrics)
    eco_tot = summarize_total(joined, eco, metrics)
    conf = conifer_dominant(joined, metrics)
    trend = conifer_total(joined, metrics)
    national_means = joined[metrics].abs().mean()

    ok = eco_dom[~eco_dom.no_plots & ~eco_dom.low_n]
    print(f"[result] {int(ok['fragile'].sum())} of {len(ok)} ecoregions tied; won (excl. ties):\n"
          + ok.loc[~ok["fragile"], "winner"].value_counts().to_string())
    okt = eco_tot[~eco_tot.no_plots & ~eco_tot.low_n]
    print(f"[result] total effect: {int((okt.direction == 'stabilising').sum())} stabilising, "
          f"{int((okt.direction == 'destabilising').sum())} destabilising, "
          f"{int(okt.uncertain.sum())} with a {CI_PCT}% CI including 0")

    us_poly = load_us_outline(states_path) if states_path else None
    if us_poly is None:
        print("[map] no STATES_PATH: showing only ecoregions that contain plots, unclipped")
    dom_draw = prepare_for_drawing(eco_dom, us_poly)
    tot_draw = prepare_for_drawing(eco_tot, us_poly)

    # One set of colour scales for both figures, so they match exactly
    norms = (magnitude_norm(dom_draw, conf), total_norm(tot_draw))

    # Plots by direction x forest type: plots in the ecoregions drawn on the total-effect map
    heat_counts, heat_thr = None, None
    os.makedirs(outdir or OUTDIR, exist_ok=True)
    if SHOW_DIRECTION_HEATMAP:
        heat_thr = neutral_interval()
        drawn = joined[joined["eco_id"].isin(tot_draw["eco_id"])]
        heat_counts, heat_df = direction_by_forest(drawn, heat_thr)
        heat_df.to_csv(_out(OUT_DIRECTION_FOREST_CSV, outdir), index=False)
        print(heat_df.to_string(index=False))

    fig = build_figure(dom_draw, tot_draw, conf, trend, metrics, national_means, us_poly, norms,
                       _out(OUT_COMBINED_NAME, outdir), heat_counts, heat_thr)
    fig_maps = None
    if SHOW_MAPS_FIGURE:
        fig_maps = build_maps_figure(dom_draw, tot_draw, metrics, national_means, us_poly, norms,
                                     _out(OUT_MAPS_NAME, outdir), heat_counts, heat_thr)

    # The same heatmap per rarity class (same plots and neutral interval)
    if SHOW_RARITY_HEATMAPS:
        drawn = joined[joined["eco_id"].isin(tot_draw["eco_id"])]
        iv = neutral_interval()
        r_counts, r_labels, rar_df = rarity_heatmap_counts(attach_rarity(drawn), iv)
        rar_df.to_csv(_out(OUT_DIRECTION_FOREST_RARITY_CSV, outdir), index=False)
        if SAVE_RARITY_HEATMAP_FIGURE:
            build_rarity_heatmaps(r_counts, r_labels, iv, _out(OUT_HEATMAP_RARITY_NAME, outdir))
        if SHOW_MAP_HEATMAP_FIGURES:          # map (a) above the rarity heatmaps (b-d)
            build_map_rarity_figure(tot_draw, us_poly, norms, r_counts, r_labels, iv,
                                    _out(OUT_MAP_RARITY_NAME, outdir))

    # Bubble map: total effect (fill) + dominant metric and its size (circles)
    if SHOW_BUBBLE_MAP:
        side = (r_counts, r_labels, neutral_interval()) if SHOW_RARITY_HEATMAPS else None
        fill = raw_fill(joined, tot_draw, "stability") if BUBBLE_FILL == "raw" else None
        local = None
        if BUBBLE_DIRECTION:
            local = local_slopes(effects_csv, joined, metrics)
            local.to_csv(_out(OUT_BUBBLE_LOCAL_CSV.format(suffix=""), outdir), index=False)
        build_bubble_map(dom_draw, tot_draw, us_poly, norms, metrics,
                         _out(OUT_BUBBLE_MAP_NAME, outdir),
                         iv=neutral_interval() if SHOW_DIRECTION_HEATMAP else None, side=side,
                         fill=fill, local=local)

    # The same, segmented by structural complexity (WSCI) instead of rarity
    if SHOW_WSCI_HEATMAPS:
        drawn = joined[joined["eco_id"].isin(tot_draw["eco_id"])]
        iv = neutral_interval()
        w_counts, w_labels, w_df = segmented_heatmap_counts(attach_wsci(drawn), iv, WSCI_COL,
                                                            wsci_classes, "wsci")
        w_df.to_csv(_out(OUT_DIRECTION_FOREST_WSCI_CSV, outdir), index=False)
        build_rarity_heatmaps(w_counts, w_labels, iv, _out(OUT_HEATMAP_WSCI_NAME, outdir),
                              seg="WSCI class")
        if SHOW_MAP_HEATMAP_FIGURES:
            build_map_rarity_figure(tot_draw, us_poly, norms, w_counts, w_labels, iv,
                                    _out(OUT_MAP_WSCI_NAME, outdir), seg="WSCI class")

    # Bubble map of the diversity effects on WSCI (own effects file, own colour scale)
    if SHOW_WSCI_BUBBLE_MAP:
        build_wsci_bubble_map(ecoregions_path, us_poly, _out(OUT_BUBBLE_MAP_WSCI_NAME, outdir))

    name = [c for c in (ECO_NAME_COL,) if c in eco.columns]
    eco_dom.loc[~eco_dom.no_plots, name + ["n_plots", "winner", "value", "runner_up", "margin",
                                           "fragile", "low_n"] + [f"mag__{m}" for m in metrics]] \
        .sort_values("value", ascending=False).to_csv(_out(OUT_DOMINANT_CSV, outdir), index=False)
    eco_tot.loc[~eco_tot.no_plots, name + ["n_plots", "mean_total", "se_total", "ci_lower", "ci_upper",
                                           "pct", "pct_ci_lower", "pct_ci_upper", "direction",
                                           "uncertain", "low_n"] + [f"contrib__{m}" for m in metrics]] \
        .sort_values("pct", ascending=False).to_csv(_out(OUT_TOTAL_CSV, outdir), index=False)
    conf.drop(columns="no_plots").to_csv(_out(OUT_CONIFER_DOMINANT_CSV, outdir), index=False)
    trend.to_csv(_out(OUT_CONIFER_TOTAL_CSV, outdir), index=False)
    print(f"[summary] saved CSVs to {outdir or OUTDIR}/")
    return fig, fig_maps


def build_wsci_bubble_map(ecoregions_path, us_poly, out_stem, effects_csv=None):
    """Bubble map for the diversity effects on WSCI: same steps as for stability, WSCI labels."""
    effects_csv = effects_csv or WSCI_EFFECTS_CSV
    if not os.path.exists(effects_csv):
        print(f"[wsci bubble map] {effects_csv} not found: run the effects script with "
              f"OUTCOME = 'wsci' first. Skipped.")
        return None
    _OUTCOME.update(OUTCOME_STYLES["wsci"])
    try:
        plots, metrics = load_plots(effects_csv)
        plots = attach_conifer(plots)
        eco = read_ecoregions(ecoregions_path, plots)
        joined = assign_ecoregions(plots, eco)
        dom_draw = prepare_for_drawing(summarize_dominant(joined, eco, metrics), us_poly)
        tot_draw = prepare_for_drawing(summarize_total(joined, eco, metrics), us_poly)
        norms = (magnitude_norm(dom_draw, conifer_dominant(joined, metrics)), total_norm(tot_draw))
        fill = raw_fill(joined, tot_draw, "wsci") if BUBBLE_FILL == "raw" else None
        local = None
        if BUBBLE_DIRECTION:
            local = local_slopes(effects_csv, joined, metrics)
            local.to_csv(os.path.join(os.path.dirname(out_stem) or ".",
                                      OUT_BUBBLE_LOCAL_CSV.format(suffix="_wsci")), index=False)
        return build_bubble_map(dom_draw, tot_draw, us_poly, norms, metrics, out_stem, iv=None,
                                fill=fill, local=local)
    finally:
        _OUTCOME.update(OUTCOME_STYLES["stability"])


if __name__ == "__main__":
    run()
    plt.show()