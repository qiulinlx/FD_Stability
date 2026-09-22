import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
INPUT_CSV = "results/shap_values/shap_feature_importance.csv"
OUTPUT_PNG ="results2/shap_feature_importance.png"
# ---------------------------------------------------------------

df = pd.read_csv(INPUT_CSV)

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
df = df[df["target"] != "Species Richness"]
df = df[df["feature"] != "percent_conifer"]

targets = df["target"].unique().tolist()

# Order features by their average importance across targets, descending,
# so the chart reads left-to-right from most to least important overall
feature_order = (
    df.groupby("feature")["mean_abs_shap"]
    .mean()
    .sort_values(ascending=False)
    .index.tolist()
)

# Pivot to feature x target grid for easy grouped-bar plotting
pivot = df.pivot(index="feature", columns="target", values="mean_abs_shap").loc[feature_order]

n_features = len(feature_order)
n_targets = len(targets)
x = np.arange(n_features)
bar_width = 0.8 / n_targets

fig, ax = plt.subplots(figsize=(22, 4.5))

colors = plt.cm.tab10(np.linspace(0, 1, n_targets))
for i, target in enumerate(targets):
    offset = (i - (n_targets - 1) / 2) * bar_width
    ax.bar(x + offset, pivot[target], width=bar_width, label=target, color=colors[i])

ax.set_xticks(x)
ax.set_xticklabels(feature_order, rotation=90, fontsize=9)
ax.set_ylabel("Mean |SHAP value|", fontsize=10)
ax.set_title("SHAP Feature Importance by Environmental Variable, Grouped by Target\n(percent_conifer and Species Richness excluded)", fontsize=12)
ax.legend(title="Target", fontsize=9)

fig.tight_layout()
fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
print(f"Saved to {OUTPUT_PNG}")