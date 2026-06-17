import pandas as pd

df= pd.read_csv("data/final/final_dataset.csv")


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

df.to_csv('data/final/final_dataset.csv', index=False)