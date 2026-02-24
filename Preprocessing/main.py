import pandas as pd
import generate_metrics as gm
import rasterio
import pyarrow as pa
import pyarrow.ipc as ipc
import os 
import numpy as np
import warnings
from process_arrow import load_arrow

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
    table = load_arrow('data/FIA_states/FIA_CA.arrow')
    traits=pd.read_csv("data/lookup/traitMatrix.csv")


    # Preparing Composite Data ------------------------------------------------

    env_df = env_df[variables]

    env_df[['lat', 'lon']] = env_df[['lat', 'lon']].round(4)
    

    PID_df[['lat', 'lon']] = PID_df[['lat', 'lon']].round(4)
    env_df = env_df.merge(PID_df, on=['lat', 'lon'], how='left')
    env_df.dropna(subset=['PID'], inplace=True)  
    env_df.drop(columns=['lat', 'lon'], inplace=True)
    
    # Preparing Funcional Diversity Data ------------------------------------------------
    folder= "data/FIA_states"
    i=1
    for file in os.listdir(folder):
        if file.endswith(".arrow"):
            filepath= os.path.join(folder, file)

            try:
                reader = ipc.RecordBatchFileReader(filepath)
                table= load_arrow(filepath)

            except pa.ArrowInvalid:
                table = pd.read_feather(filepath)

        print((f"Processing {file}"))

        pivot = pd.crosstab(table["PID"], table["accepted_bin"])

        pivot.columns.name = None       # remove the "Type" name
        pivot = pivot.reset_index() 

        pivot = pivot.set_index("PID")

        fd_df=gm.generate_functional_diversity_metrics(pivot, traits)

        sd_df=gm.generate_species_diversity_metrics(pivot)

        # Merging FD and Environmental Data ------------------------------------------------
        total_df = fd_df.merge(env_df, on='PID', how='inner').merge(sd_df, on='PID', how='inner')
        
        # total_df = total_df.merge(rs_df, on='PID', how='inner')

        total_df.to_csv(f'data/joined/dataset{i}.csv', index=False)
        i+=1
