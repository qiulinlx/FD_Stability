import pandas as pd
import numpy as np

dataset_name = "data/final/final_dataset.csv"

additinal_data =  'data/processed/PID_land_cover.csv'
# "data/processed/aridity.csv"
# 'data/processed/PID_with_pet.csv' pet_std
# "data/treecover/treecover_PID.csv"

df= pd.read_csv(dataset_name)
df1=pd.read_csv(additinal_data)

# df=df.merge(df1[['PID','treecover2000']], on='PID', how='left')
df=df.merge(df1[['PID','land_cover_value']], on='PID', how='left')

df.drop_duplicates(subset=['PID'], inplace=True)

df.to_csv(dataset_name, index=False)
