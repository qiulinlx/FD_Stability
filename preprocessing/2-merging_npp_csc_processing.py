import pandas as pd
import numpy as np

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.data_utils import data_preprocessing


'''
We run this file to merge datasets for modelling
'''

csc_df= pd.read_csv('data/processed/PID_location_WSCI.csv')
env_df=pd.read_csv('data/processed/env_data.csv')
npp_df=pd.read_csv('data/processed/PID_npp_volatility.csv')
df = pd.read_parquet("data/final/dataset_ba.parquet")



additinal_df1 = pd.read_csv('data/processed/PID_land_cover.csv')
additional_df3 = pd.read_csv('data/processed/PID_with_pet.csv')
additional_df4 = pd.read_csv('data/treecover/treecover_PID.csv')


df=df.merge(csc_df[['PID','WSCI']], on='PID', how='left')
df=df.merge(env_df, on='PID', how='left')
df=df.merge(npp_df, on='PID', how='left')

df=df.merge(additinal_df1[['PID','land_cover_value']], on='PID', how='left')
df=df.merge(additional_df3[['PID','pet_std']], on='PID', how='left')
df=df.merge(additional_df4[['PID','treecover2000']], on='PID', how='left')

df.drop(columns=["source_file", 'biome'], inplace=True)
df = df.rename(columns={'BIOME_NAME': 'biome'})

df.to_csv("data/final/final_dataset_ba_v2.csv", index=False)