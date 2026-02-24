from shapely.geometry import shape
import json
import pandas as pd
from scipy.signal import detrend
import matplotlib.pyplot as plt


# Assign a color to each forest type
forest_colors = {
    'Temperate conifer forests': 'lightgreen',
    'Temperate grasslands': 'lightblue',
    'Xeric shrublands': 'wheat',
    'Mediterranean woodlands': 'orange',
    'Tropical coniferous forests': 'darkgreen'
}

managed= {
    1.0: 'blue',
    0.0: 'red'
}

ownership = {
    'national_forest': 'darkblue',
    'other': 'lightgray',
    'state': 'lightgreen',
    'blm': 'orange',
    'national_park': 'darkgreen',
    'local': 'purple',
    'other_federal': 'brown',
    'fish_wildlife': 'pink',
    'dod': 'gray',
    'other_forest_service': 'lightblue',
    'national_grassland': 'yellow'
}


def autocorr_pid(group, max_lag=1):
    group = group.sort_values('year')
    acfs = [group['Npp'].autocorr(lag=1)]
    return pd.Series(acfs, index=[f'ACF_lag{lag}' for lag in range(1, max_lag+1)])

def detrend_pid(group):
    # Sort by year
    group = group.sort_values('year')
    # Detrend NPP
    group['NPP_detrended'] = detrend(group['Npp'].values)
    return group


df = pd.read_parquet("data/dataset.parquet")
PID_df=pd.read_csv('data/lookup/PID_location_all.csv')
df_joined=pd.read_csv('data/processed/PID_npp.csv')

# df_joined = df_joined.groupby('PID').apply(detrend_pid).reset_index(drop=True) No longer detrending

df_joined.drop(columns=(['Unnamed: 0']), inplace=True)

autocorr_df = df_joined.groupby('PID').apply(autocorr_pid).reset_index()
df= autocorr_df[["PID", "ACF_lag1"]]

df.rename(columns={"ACF_lag1": "TAC_NPP"}, inplace=True)
# Plot 1

plt.figure(figsize=(6,4))
plt.hist(df["TAC_NPP"], bins=40)
plt.xlabel("Temporal Autocorrelation (TAC_NPP)")
plt.ylabel("Count")
plt.title("Distribution of Temporal Autocorrelation")
plt.show()

# Plot 2 -------------------------------

total_df = df.merge(PID_df, on='PID', how='inner')

plt.figure(figsize=(12,5))
plt.scatter(total_df["lon"], total_df["lat"], c=total_df["TAC_NPP"], s=1, cmap= 'PiYG')
plt.colorbar(label="TAC_NPP")
plt.title("Spatial Pattern of Temporal Autocorrelation")
plt.show()

# Plot 3 
plt.figure(figsize=(12,5))

for ftype, color in forest_colors.items():
    subset = total_df[total_df["biome"] == ftype]
    plt.scatter(
        subset["lon"], 
        subset["lat"], 
        color=color, 
        alpha=0.5,  # transparency
        s=30, 
        label=f"Forest {ftype}"
    )

# First, scatter points colored by TAC_NPP
sc = plt.scatter(
    total_df["lon"], 
    total_df["lat"], 
    c=total_df["TAC_NPP"], 
    s=0.5, 
    cmap='PiYG'
)
plt.colorbar(sc, label="TAC_NPP")

plt.title("Spatial Pattern of Temporal Autocorrelation with Managed vs Unmanaged")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend(
    title="Forest Type",
    bbox_to_anchor=(1.2, 1),  # x=1.05 moves it right outside axes
    loc='upper left',           # anchors the legend's upper-left corner
    borderaxespad=0            # padding between axes and legend
)
plt.show()