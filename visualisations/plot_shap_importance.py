import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
INPUT_CSV = "results/shap_values/shap_feature_importance.csv"
OUTPUT_PNG = "results2/shap_feature_importance.png"

TOP_N = 9

# ---------------------------------------------------------------

df = pd.read_csv(INPUT_CSV)

# Comments corrected to match the actual hex colors (the originals
# said "red"/"purple"/"brown"/"olive" but didn't match these values --
# #4910d8 and #9975ba are both violet/purple shades, #7ddfe7 is a pale
# cyan, #13d2b9 is teal, not brown/olive). Kept the hex values
# themselves unchanged since presumably they were picked deliberately
# to match other figures in this pipeline's METRIC_COLORS convention.
METRIC_COLORS = {
    'Raos_Q': "#4910d8",              # deep violet
    'Functional_Evenness': "#9975ba",  # muted purple
    'Shannon Diversity': "#7ddfe7",              # pale cyan
    "Species Richness": "#13d2b9",     # teal
}

df["feature"] = df["feature"].replace({
    "CHELSA_exBIO_GrowingSeasonLength": "Growing Season Length",
    "SG_CEC_015cm": "Soil CEC",
    "EarthEnvTopoMed_Eastness": "Eastness",
    "SG_Soil_pH_H2O_015cm": "Soil pH",
})

# Drop npp targets, Species Richness (much larger SHAP scale, swamps the
# other four targets' bars), and the percent_conifer feature (dominates
# the scale and swamps every other feature's bar into near-invisibility)
df = df[~df["target"].str.contains("npp", case=False)]
df = df[df["target"] != "Simpson's Index"]
df = df[df["feature"] != "percent_conifer"]

targets = df["target"].unique().tolist()

# Any target not explicitly in METRIC_COLORS gets a safe gray fallback
# instead of crashing with a KeyError -- this will only trigger if a
# new target shows up in the CSV that hasn't been added to the dict
# above, and it's better to get an obviously-gray bar flagging that
# than a hard failure or a silently wrong color from a colormap.
missing_targets = [t for t in targets if t not in METRIC_COLORS]


if missing_targets:
    print(f"WARNING: no entry in METRIC_COLORS for: {missing_targets} -- "
          f"falling back to gray for these; add them to the dict above if they should "
          f"have their own color")

target_colors = {t: METRIC_COLORS.get(t, "#999999") for t in targets}

# Order features by their average importance across targets, descending,
# so the chart reads left-to-right from most to least important overall
feature_order = (
    df.groupby("feature")["mean_abs_shap"]
    .mean()
    .sort_values(ascending=False)
    .index.tolist()
)

# doesn't change which features count as "most important."
feature_order = feature_order[:TOP_N]


 
# Pivot to feature x target grid for easy grouped-bar plotting
pivot_raw = df.pivot(index="feature", columns="target", values="mean_abs_shap").loc[feature_order]
n_missing_cells = pivot_raw.isna().sum().sum()
if n_missing_cells:
    print(f"WARNING: {n_missing_cells} feature x target cells had no SHAP value "
          f"(filled with 0 for plotting) -- check pivot_raw.isna() if this is unexpected")
pivot = pivot_raw.fillna(0)

n_features = len(feature_order)
n_targets = len(targets)
x = np.arange(n_features)
bar_width = 0.8 / n_targets

fig, ax = plt.subplots(figsize=(22, 4.5))

for i, target in enumerate(targets):
    offset = (i - (n_targets - 1) / 2) * bar_width
    ax.bar(x + offset, pivot[target], width=bar_width, label=target, color=target_colors[target])

ax.set_xticks(x)
ax.set_xticklabels(feature_order, rotation=90, fontsize=9)
ax.set_ylabel("Mean |SHAP value|", fontsize=10)
ax.set_title("SHAP Feature Importance by Environmental Variable, Grouped by Target\n"
             "(percent_conifer and Species Richness excluded)", fontsize=12)
ax.legend(title="Target", fontsize=9)

fig.tight_layout()
os.makedirs(os.path.dirname(OUTPUT_PNG), exist_ok=True)
fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
print(f"Saved to {OUTPUT_PNG}")