import pandas as pd
import generate_fd as gfd
import rasterio
import pyarrow as pa
import pyarrow.ipc as ipc
import os 
import numpy as np
import warnings

warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

if __name__ == "__main__":

    variables= ["lat", "lon",
    "CHELSA_BIO_Annual_Mean_Temperature",
    "CHELSA_BIO_Annual_Precipitation",
    "CHELSA_BIO_Precipitation_Seasonality", 
    "CrowtherLab_SoilMoisture_intraAnnualSD_downsampled10km",
    "SG_SOC_Content_015cm",
    "EsaCci_BurntAreasProbability", 
    "EarthEnvTopoMed_Elevation", 
    "EarthEnvTopoMed_Slope",
    "SG_Depth_to_bedrock", 
    "EarthEnvTopoMed_Northness"
    ]

    env_df= pd.read_csv('data/Composite.csv')
    PID_df=pd.read_csv('data/lookup/PID_location_all.csv')
    table = pd.read_csv('data/processed/species_abundance_all.csv')
    traits=pd.read_csv("data/lookup/traitMatrix.csv")
    # tif_path = "data/Tiff/MODIS_CA_2015.tif"  


    # Preparing Composite Data ------------------------------------------------

    env_df = env_df[variables]

    env_df[['lat', 'lon']] = env_df[['lat', 'lon']].round(4)
    

    PID_df[['lat', 'lon']] = PID_df[['lat', 'lon']].round(4)
    env_df = env_df.merge(PID_df, on=['lat', 'lon'], how='left')
    env_df.dropna(subset=['PID'], inplace=True)  
    env_df.drop(columns=['lat', 'lon'], inplace=True)
    
    # Preparing Funcional Diversity Data ------------------------------------------------

    table=table[["PID", "accepted_bin"]]

    pivot = pd.crosstab(table["PID"], table["accepted_bin"])

    pivot.columns.name = None       # remove the "Type" name
    pivot = pivot.reset_index() 

    pivot = pivot.set_index("PID")

    fd_df=gfd.generate_functional_diversity_metrics(pivot, traits)

    fd_df.dropna(subset=['Raos_Q'], inplace=True)

    # #Loading in Remote sensing data ------------------------------------------------

    # with rasterio.open(tif_path) as src:
    #     transform = src.transform
    #     crs = src.crs
    #     band_names=src.descriptions
    #     data = src.read()  


    # # Get row/col indices
    # bands, rows, cols = data.shape
    # row_idx, col_idx = np.meshgrid(
    #     np.arange(rows),
    #     np.arange(cols),
    #     indexing='ij'
    # )
    # # Convert pixel indices → coordinates
    # xs, ys = rasterio.transform.xy(transform, row_idx, col_idx)

    # # Flatten
    # lons = np.array(xs).flatten()
    # lats = np.array(ys).flatten()

    # # Build DataFrame
    # rs_df = pd.DataFrame({
    #     'lat': lats,
    #     'lon': lons
    # })

    # # Add band values
    # for i in range(bands):
    #     rs_df[band_names[i]] = data[i].flatten()

    # PID_df[['lat', 'lon']] = PID_df[['lat', 'lon']].round(3)
    # rs_df[['lat', 'lon']]=rs_df[['lat', 'lon']].round(3)

    # rs_df = rs_df.merge(PID_df, on=['lat', 'lon'], how='left')
    # rs_df.dropna(inplace=True)

    # Merging FD and Environmental Data ------------------------------------------------
    total_df = fd_df.merge(env_df, on='PID', how='inner')
    
    # total_df = total_df.merge(rs_df, on='PID', how='inner')

    total_df.to_csv('data/processed/dataset.csv', index=False)