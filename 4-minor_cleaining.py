import pandas as pd
import numpy as np

df= pd.read_csv("data/final/final_dataset.csv")
df1=pd.read_csv("data/processed/Aridity.csv")


df = df.rename(columns={'CHELSA_BIO_Annual_Mean_Temperature': 'Annual Temp', 'CHELSA_BIO_Annual_Precipitation': 'Annual Precipitation',
       'CHELSA_BIO_Precipitation_Seasonality': 'Precipitation Seasonality',
       'CrowtherLab_SoilMoisture_intraAnnualSD_downsampled10km': 'Soil Moisture',
       'SG_SOC_Content_015cm': 'SOC Content', 'EsaCci_BurntAreasProbability': 'Burnt Areas Probability',
       'EarthEnvTopoMed_Elevation': 'Elevation', 'EarthEnvTopoMed_Slope': 'Slope',
       'SG_Depth_to_bedrock': 'Depth to Bedrock', 'EarthEnvTopoMed_Northness': 'Northness'})

df = df.loc[:, ~df.columns.str.contains('_y')]

df = df.rename(columns={'lon_x': 'lon', 'lat_x': 'lat',  
                             'managed_x': 'managed', 'ownership_x': 'ownership',
                            'biome_x': 'biome',  'TPA_UNADJ_x': 'TPA_UNADJ'})

df['WSCI'] = np.where(df['WSCI'] < -5, 0, df['WSCI'])

df.to_csv('data/final/final_dataset.csv', index=False)

df=df.merge(df1[['PID','aridity_delta']], on='PID', how='left')
df.drop_duplicates(subset=['PID'], inplace=True)

df.to_csv("data/final/final_dataset_with_aridity.csv", index=False)

# df=pd.read_csv("data/processed/Aridity.csv")

# new_df=df[['PID','aridity_delta']] 
# new_df.drop_duplicates(inplace=True)
# new_df['aridity_delta'] = np.where(new_df['aridity_delta'] < -2, -2, new_df['aridity_delta'])

# new_df.to_csv("data/processed/aridity_delta.csv", index=False)