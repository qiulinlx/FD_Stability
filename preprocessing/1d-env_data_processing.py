import pandas as pd
import warnings
import sys

from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

'''

RUN PROCESS ARROW FIRST
We run this to:
    pull data from the Composite file
'''


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

    env_df= pd.read_csv('data/raw/Composite.csv')
    PID_df=pd.read_csv('data/lookup/PID_location_all.csv')

    PID_df["managed"] = PID_df["managed"].fillna(-1)
    PID_df["ownership"] = PID_df["ownership"].fillna("No Data")
    PID_df["biome"] = PID_df["biome"].fillna("No Data")


    PID_df.drop_duplicates(subset=['PID'], inplace=True)

    # Preparing Composite Data ------------------------------------------------

    env_df = env_df[variables]

    env_df[['lat', 'lon']] = env_df[['lat', 'lon']].round(4)
    

    PID_df[['lat', 'lon']] = PID_df[['lat', 'lon']].round(4)
    env_df = env_df.merge(PID_df, on=['lat', 'lon'], how='left')
    env_df.dropna(subset=['PID'], inplace=True)  
    env_df.drop(columns=['lat', 'lon'], inplace=True)


    env_df.to_csv('data/processed/env_data.csv', index=False)