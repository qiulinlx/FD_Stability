"""
Ecoregion map of WHICH diversity metric has the largest effect on NPP
stability (hue) and HOW LARGE that effect is (colour intensity).

Input: per-observation effects from the joint (mutually adjusted) model, long
format -- one row per plot x diversity metric with columns
    PID, diversity_metric, adjusted_effect, lon, lat

Steps
  1. For every plot and metric take |adjusted_effect| (the effects are centred
     partial effects, so their signed mean is ~0; magnitude is what carries
     information).
  2. Assign plots to WWF Ecoregions2017 polygons (point-in-polygon; plots just
     outside any polygon are snapped to the nearest ecoregion within
     NEAREST_MAX_M).
  3. Within each ecoregion take the mean |effect| of each metric. The metric with
     the largest mean sets the HUE; that winning mean sets the INTENSITY
     (white -> full hue), on one shared scale so intensity is comparable across
     hues. Magnitudes are directly comparable across metrics because every
     adjusted effect is a contribution to the same response from the same joint
     model, in the same units.

Legend: a ranking panel beside the colour key shows each metric's NATIONAL mean
  |effect| (all plots), so metrics that never win an ecoregion still appear and
  readers can see how the winners compare with the full field.

Caveats
  * Where two metrics are nearly tied the winner is fragile. The summary CSV
    reports `margin` (winner / runner-up mean); the console prints how many
    ecoregions have margin < 1.10.
  * Ecoregions with fewer than MIN_PLOTS_PER_ECOREGION plots are hatched (their
    means are too noisy to assign a winner); ecoregions with no plots are grey.
  * Pale shades of different hues can look alike to colour-blind readers; the
    legend lists every metric (including ones that win no ecoregion) at every
    intensity, and METRIC_COLORS can be swapped.

STATES_PATH (recommended): any US states polygon file (Census cb_2018_us_state_*
or Natural Earth admin-1). Ecoregions2017 polygons are global, so without it
border-straddling ecoregions are drawn in full, including their Canadian /
Mexican parts. With it, the map is clipped to the contiguous US and outlined.

Outputs: <OUT_STEM>.png (600 dpi), <OUT_STEM>.pdf (vector, editable text) and
<OUT_STEM>_summary.csv.
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
from matplotlib.ticker import FuncFormatter, MaxNLocator
from shapely.geometry import box

# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------
EFFECTS_CSV = "results/mutually_adjusted_effects_per_observation_with_coords.csv"
ECOREGIONS_PATH = "data/Ecoregions/Ecoregions2017.shp"
ECO_NAME_COL = "ECO_NAME"
STATES_PATH = None            # e.g. "data/states/cb_2018_us_state_20m.shp"

# Hue per metric (keys must match the diversity_metric values in the CSV;
# these are the colours used for the SHAP figures)
METRIC_COLORS = {
    "Rao's Q": "#4910D8", # red
    "Functional Evenness": "#9975BA", # purple
    "Shannon": "#7DDFE7", # brown
    "Species richness": "#13D2B9", # olive
    }
INTENSITY_FLOOR = 0.18        # weakest effects still show 18% of the hue (stays distinct from grey/white)

MIN_PLOTS_PER_ECOREGION = 20
NEAREST_MAX_M = 5000          # snap unmatched plots to an ecoregion this close (m); None = don't
MAP_CRS = "EPSG:5070"         # CONUS Albers equal-area
SIMPLIFY_M = 500              # simplify polygons for DRAWING only (joins use full detail)
FIG_WIDTH_IN = 7.0            # ~178 mm, double-column width
DPI = 600
OUT_STEM = "results/dominant_diversity_metric_ecoregions"


# ---------------------------------------------------------------
# Data
# ---------------------------------------------------------------
def load_plot_effects(csv_path):
    """Wide table: one row per plot, lon/lat plus |adjusted_effect| for every metric."""
    long = pd.read_csv(csv_path, usecols=["PID", "diversity_metric", "adjusted_effect", "lon", "lat"])
    wide = long.pivot(index="PID", columns="diversity_metric", values="adjusted_effect").abs()

    n_all = len(wide)
    wide = wide.dropna()  # only plots with an effect for EVERY metric are comparable
    if len(wide) < n_all:
        print(f"[load] dropped {n_all - len(wide)} plots missing an effect for one or more metrics")

    unknown = [m for m in wide.columns if m not in METRIC_COLORS]
    if unknown:
        raise KeyError(f"No colour defined in METRIC_COLORS for: {unknown}")

    coords = long.drop_duplicates("PID").set_index("PID")[["lon", "lat"]]
    plots = coords.loc[wide.index].join(wide).reset_index()
    print(f"[load] {len(plots)} plots, metrics: {list(wide.columns)}")
    return plots, list(wide.columns)


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
          f"(<= {nearest_max_m} m), {n_lost} unassigned and dropped (e.g. AK/HI plots, "
          f"which lie outside the contiguous-US ecoregions)")
    return j.dropna(subset=["eco_id"])


def summarize(joined, eco, metrics, min_plots=MIN_PLOTS_PER_ECOREGION):
    """Per ecoregion: mean |effect| per metric, the winning metric, its magnitude and its margin."""
    grp = joined.groupby("eco_id")
    mean_abs = grp[metrics].mean().add_prefix("mag__")
    n_plots = grp.size().rename("n_plots")
    eco = eco.merge(pd.concat([n_plots, mean_abs], axis=1), left_on="eco_id",
                    right_index=True, how="left")
    eco["n_plots"] = eco["n_plots"].fillna(0).astype(int)

    mags = eco[[f"mag__{m}" for m in metrics]].to_numpy(dtype=float)
    has = ~np.isnan(mags).all(axis=1)
    order = np.argsort(-np.nan_to_num(mags, nan=-np.inf), axis=1)
    top, second = order[:, 0], order[:, 1]
    rows = np.arange(len(eco))

    eco["winner"] = [metrics[k] if h else None for k, h in zip(top, has)]
    eco["value"] = np.where(has, mags[rows, top], np.nan)
    eco["runner_up"] = [metrics[k] if h else None for k, h in zip(second, has)]
    eco["margin"] = np.where(has, mags[rows, top] / mags[rows, second], np.nan)
    eco["no_plots"] = eco["n_plots"] == 0
    eco["low_n"] = (eco["n_plots"] > 0) & (eco["n_plots"] < min_plots)
    return eco


# ---------------------------------------------------------------
# Colour: hue = metric, intensity = magnitude
# ---------------------------------------------------------------
def blend(base_hex, t, floor=INTENSITY_FLOOR):
    """White -> base colour; t in [0, 1] is the normalised magnitude."""
    base = np.array(mcolors.to_rgb(base_hex))
    a = floor + (1 - floor) * np.clip(t, 0, 1)
    return tuple(1 - a * (1 - base))


# ---------------------------------------------------------------
# Figure
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


def draw_map(eco, us_poly, metrics, min_plots=MIN_PLOTS_PER_ECOREGION, out_stem=OUT_STEM,
             national_means=None):
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": [_pick_font()],
        "pdf.fonttype": 42, "ps.fonttype": 42, "hatch.linewidth": 0.3,
    })

    colored = eco[~eco.no_plots & ~eco.low_n].copy()
    vals = colored["value"]
    exp = int(np.floor(np.log10(vals.max())))
    scale = 10.0 ** exp
    # Magnitude limits snapped outward to "nice" tick values so both ends of the scale are labelled
    ticks = MaxNLocator(5, steps=[1, 2, 2.5, 5, 10]).tick_values(vals.min() / scale, vals.max() / scale)
    vmin, vmax = ticks[0] * scale, ticks[-1] * scale
    norm = mcolors.Normalize(vmin, vmax)

    colored["fill"] = [mcolors.to_hex(blend(METRIC_COLORS[m], norm(v)))
                       for m, v in zip(colored["winner"], colored["value"])]

    # Legend rows: EVERY metric in the data (in METRIC_COLORS order), even ones that
    # never win an ecoregion, so the key is the same regardless of the result
    legend_metrics = [m for m in METRIC_COLORS if m in metrics]
    n_rows = len(legend_metrics)

    xmin, ymin, xmax, ymax = us_poly.bounds if us_poly is not None else eco.total_bounds
    pad = 0.01 * (xmax - xmin)
    xmin, xmax, ymin, ymax = xmin - pad, xmax + pad, ymin - pad, ymax + pad
    width, height = xmax - xmin, ymax - ymin

    map_h_in = FIG_WIDTH_IN * height / width
    strip_in = 0.80 + 0.15 * n_rows       # strip under the map that holds the legend
    fig_h_in = map_h_in + strip_in
    fig = plt.figure(figsize=(FIG_WIDTH_IN, fig_h_in))
    ax = fig.add_axes([0, strip_in / fig_h_in, 1, map_h_in / fig_h_in])
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_axis_off()

    if eco.no_plots.any():
        eco[eco.no_plots].plot(ax=ax, facecolor="#e6e6e6", edgecolor="#9a9a9a", linewidth=0.2)
    if eco.low_n.any():
        eco[eco.low_n].plot(ax=ax, facecolor="#f7f7f7", edgecolor="#808080",
                            linewidth=0.2, hatch="//////")
    colored.plot(ax=ax, color=colored["fill"], edgecolor="#5a5a5a", linewidth=0.2)
    if us_poly is not None:
        gpd.GeoSeries([us_poly], crs=eco.crs).boundary.plot(ax=ax, color="black", linewidth=0.5)
    _scale_bar(ax, xmax, ymin, width, height)

    # Legend beneath the map: one gradient row per metric, shared magnitude axis on top
    row_h = 0.15 / fig_h_in
    lax = fig.add_axes([0.20, 0.10 / fig_h_in, 0.30, n_rows * row_h])
    t = np.linspace(0, 1, 256)
    img = np.array([[blend(METRIC_COLORS[m], tt) for tt in t] for m in legend_metrics])
    lax.imshow(img, aspect="auto", extent=(vmin / scale, vmax / scale, n_rows, 0), interpolation="nearest")
    for r in range(1, n_rows):
        lax.axhline(r, color="white", lw=1.2)
    lax.set_yticks(np.arange(n_rows) + 0.5)
    lax.set_yticklabels(legend_metrics, fontsize=7)
    lax.set_xticks(ticks)
    lax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    lax.xaxis.tick_top()
    lax.xaxis.set_label_position("top")
    lax.set_xlabel(f"Dominant metric (hue) and its mean\neffect magnitude (intensity, "
                   f"\u00d710$^{{{exp}}}$)", fontsize=7.5, labelpad=4, loc="left", linespacing=1.2)
    lax.tick_params(axis="x", length=2, width=0.4, labelsize=7, pad=1.5)
    lax.tick_params(axis="y", length=0, pad=3)
    for sp in lax.spines.values():
        sp.set_linewidth(0.4)

    # Ranking panel: national mean |effect| for EVERY metric (same rows as the gradients, same
    # units), so metrics that never win an ecoregion still show where they rank overall
    if national_means is not None:
        bax = fig.add_axes([0.58, 0.10 / fig_h_in, 0.20, n_rows * row_h])
        vals_nat = np.array([national_means[m] for m in legend_metrics]) / scale
        ypos = np.arange(n_rows) + 0.5
        bax.barh(ypos, vals_nat, height=0.72, color=[METRIC_COLORS[m] for m in legend_metrics])
        bax.set_ylim(n_rows, 0)
        bax.set_xlim(0, vals_nat.max() * 1.22)
        for y, v in zip(ypos, vals_nat):
            bax.text(v + vals_nat.max() * 0.03, y, f"{v:.2f}", va="center", ha="left", fontsize=6.5)
        bax.set_yticks([])
        bax.xaxis.set_major_locator(MaxNLocator(3))
        bax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        bax.xaxis.tick_top()
        bax.xaxis.set_label_position("top")
        bax.set_xlabel(f"National mean |effect|\n(all plots, \u00d710$^{{{exp}}}$)",
                       fontsize=7.5, labelpad=4, loc="left", linespacing=1.2)
        bax.tick_params(axis="x", length=2, width=0.4, labelsize=7, pad=1.5)
        for name_, sp in bax.spines.items():
            sp.set_visible(name_ == "left")
            sp.set_linewidth(0.4)
    key_x = 0.83 if national_means is not None else 0.66

    handles = [Patch(facecolor="#f7f7f7", edgecolor="#808080", hatch="//////", linewidth=0.4,
                     label=f"< {min_plots} plots")]
    if eco.no_plots.any():
        handles.append(Patch(facecolor="#e6e6e6", edgecolor="#9a9a9a", linewidth=0.4, label="No plots"))
    fig.legend(handles=handles, loc="lower left",
               bbox_to_anchor=(key_x, 0.10 / fig_h_in), frameon=False, fontsize=7,
               handlelength=1.6, handleheight=1.0, borderaxespad=0)

    os.makedirs(os.path.dirname(out_stem) or ".", exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_stem}.{ext}", dpi=DPI, bbox_inches=None, facecolor="white")
    print(f"[figure] saved {out_stem}.png / .pdf")
    return fig


# ---------------------------------------------------------------
# Run
# ---------------------------------------------------------------
def run(effects_csv=EFFECTS_CSV, ecoregions_path=ECOREGIONS_PATH, states_path=STATES_PATH,
        min_plots=MIN_PLOTS_PER_ECOREGION, out_stem=OUT_STEM):
    plots, metrics = load_plot_effects(effects_csv)
    eco = read_ecoregions(ecoregions_path, plots)
    joined = assign_ecoregions(plots, eco)
    eco = summarize(joined, eco, metrics, min_plots)

    ok = eco[~eco.no_plots & ~eco.low_n]
    print("[result] ecoregions won per metric (ecoregions with >= "
          f"{min_plots} plots):\n" + ok["winner"].value_counts().to_string())
    close = int((ok["margin"] < 1.10).sum())
    print(f"[result] {close} of {len(ok)} ecoregions have a winner within 10% of the runner-up "
          f"(margin < 1.10) -- treat those winners as fragile")

    # Simplify for drawing only (assignment above used the full-detail polygons)
    eco_draw = eco.copy()
    eco_draw["geometry"] = eco.geometry.simplify(SIMPLIFY_M, preserve_topology=True)

    us_poly = None
    if states_path:
        us_poly = load_us_outline(states_path)
        eco_draw = gpd.clip(eco_draw, us_poly)
        eco_draw = eco_draw[~eco_draw.geometry.is_empty]
    else:
        print("[map] no STATES_PATH: showing only ecoregions that contain plots, unclipped "
              "(border-straddling ecoregions will extend beyond the US)")
        eco_draw = eco_draw[~eco_draw.no_plots]

    national_means = plots[metrics].mean()
    print("[result] national mean |effect| (all plots), x1e-3:\n"
          + (national_means * 1e3).round(3).sort_values(ascending=False).to_string())

    fig = draw_map(eco_draw, us_poly, metrics, min_plots, out_stem, national_means=national_means)

    keep = [c for c in (ECO_NAME_COL,) if c in eco.columns] + \
           ["n_plots", "winner", "value", "runner_up", "margin", "low_n"] + \
           [f"mag__{m}" for m in metrics]
    eco[keep].sort_values("value", ascending=False).to_csv(f"{out_stem}_summary.csv", index=False)
    print(f"[summary] saved {out_stem}_summary.csv")
    return eco, fig


if __name__ == "__main__":
    run()
    plt.show()