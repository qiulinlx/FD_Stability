import pandas as pd
import pandas as pd
import xarray as xr
import numpy as np
from sklearn.neighbors import BallTree

PID_df= pd.read_csv('data/lookup/PID_location_all.csv')

PID_df.drop_duplicates(subset=['PID'], inplace=True)


url = "http://thredds.northwestknowledge.net:8080/thredds/dodsC/agg_terraclimate_pet_1950_CurrentYear_GLOBE.nc"

ds = xr.open_dataset(url)  # lazy-loads, doesn't download everything at once

# Subset by time and (optionally) space before pulling data over the network
pet = ds['pet'].sel(
    time=slice("2006-01-01", "2024-12-31"),
    lat=slice(85, 5),      # TerraClimate lat is usually descending, so max first
    lon=slice(-170, -50)
)

df = pet.to_dataframe().reset_index()
df.dropna(subset=['pet'], inplace=True)

pet_std = (
    df.groupby(["lat", "lon"], as_index=False)["pet"]
      .std()
      .rename(columns={"pet": "pet_std"})
)

# BallTree with haversine requires radians, and expects [lat, lon] order
tree = BallTree(np.radians(pet_std[['lat', 'lon']].values), metric='haversine')

query_coords = np.radians(PID_df[['lat', 'lon']].values)
dist, idx = tree.query(query_coords, k=1)

EARTH_RADIUS_KM = 6371.0
dist_km = dist.flatten() * EARTH_RADIUS_KM

# Build matched result
matched = pet_std.iloc[idx.flatten()].reset_index(drop=True)
matched['dist_km'] = dist_km

# Apply 4 km cutoff -> NaN out anything too far
mask = matched['dist_km'] > 4
matched.loc[mask, matched.columns.difference(['dist_km'])] = np.nan

# Combine with original df2 for a full result
result = pd.concat([PID_df.reset_index(drop=True), matched], axis=1)

result[['PID','pet_std']].to_csv('data/processed/PID_with_pet.csv', index=False)

