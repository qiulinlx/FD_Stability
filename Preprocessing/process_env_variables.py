import os
import numpy as np
import pandas as pd

import geopandas as gpd

from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import fcluster

def visualise_groups(linkage_matrix, df):
    # Plot dendrogram with bigger labels
    plt.figure(figsize=(70, 10))
    dendrogram(
        linkage_matrix,
        labels=df.columns,
        leaf_rotation=45,      # rotate x-axis labels
        leaf_font_size=12      # increase font size
    )
    plt.title('Feature Clustering Dendrogram', fontsize=16)
    plt.show()

def select_representative_variable(linkage_matrix, corr_matrix, threshold=0.5):
    groups = fcluster(linkage_matrix, t=threshold, criterion='distance')

    # Map features to their groups
    feature_groups = pd.DataFrame({'feature': joined.columns, 'group': groups})
    print(feature_groups)

    # Compute absolute correlation matrix
    corr_matrix = corr_matrix.abs()

    representative_features = []

    for group in feature_groups['group'].unique():
        # Features in this group
        group_features = feature_groups[feature_groups['group'] == group]['feature']
        
        # Subset correlation matrix for this group
        sub_corr = corr_matrix.loc[group_features, group_features]
        
        # Compute average correlation of each feature with others in the group
        avg_corr = sub_corr.mean()
        
        # Pick feature with highest average correlation
        rep_feature = avg_corr.idxmax()
        representative_features.append(rep_feature)

    return representative_features


env_data= pd.read_csv('data/processed/CA_env.csv', index_col=0)
PID_locations=pd.read_csv('data/lookup/PID_location.csv', index_col=0)

gdf1 = gpd.GeoDataFrame(
    env_data,
    geometry=gpd.points_from_xy(env_data.lon, env_data.lat),
    crs="EPSG:4326"
)

gdf2 = gpd.GeoDataFrame(
    PID_locations,
    geometry=gpd.points_from_xy(PID_locations.lon, PID_locations.lat),
    crs="EPSG:4326"
)

joined = gpd.sjoin_nearest(gdf1, gdf2, how="left", distance_col="dist_m")

# Calculating correlations between environmental variables
corr_matrix = joined.corr()
distance_matrix = 1 - np.abs(corr_matrix) # Distance matrix
linkage_matrix = linkage(distance_matrix, method='average')

r_features=select_representative_variable(linkage_matrix, corr_matrix)

df = joined[r_features]

df.to_csv('CA_env_PID_matched.csv', index=False)