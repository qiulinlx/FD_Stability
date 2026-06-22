import pandas as pd
import numpy as np
import utils.analysis_functions as af
from scipy.signal import detrend
from utils.data_utils import data_preprocessing
from sklearn.preprocessing import LabelEncoder


'''
We run this file to get cleaned datasets for modelling
'''

PID_df=pd.read_csv('data/lookup/PID_location_all2.csv')
csc_df= pd.read_csv('data/WSCI/PID_location_WSCI.csv')
DBH_df=pd.read_csv('data/processed/PID_GCDBH.csv')
npp_df=pd.read_csv('data/processed/PID_npp.csv')
df = pd.read_parquet("data/final/dataset.parquet")

PID_df, csc_df, npp_df = af.cleaning(PID_df, csc_df, npp_df)

csc_df.drop_duplicates(subset=['PID'], inplace=True)

npp_df.rename(columns={'NPP_kgC_m2_yr': "Npp"}, inplace=True)

grouped = npp_df.groupby('PID').apply(lambda x: x.sort_values('year')['Npp'].to_numpy())
grouped = grouped[grouped.apply(lambda x: ~np.isnan(x).any())]

volatility=[]
PID_list=[]
mean=[]

for pid, arr in grouped.items():
    if arr.shape[0] > 5:
        if (arr == 0).sum() < 4: 
            v , s = af.compute_volatility(arr)
            volatility.append(v)
            PID_list.append(pid)
            mean.append(s)

npp_df = pd.DataFrame({'PID': PID_list, 'transformed npp': volatility})

y_df=npp_df.merge(csc_df,  on='PID', how='inner')

# le = LabelEncoder()
# df['biome'] = le.fit_transform(df['biome'])

# le = LabelEncoder()
# df['ownership'] = le.fit_transform(df['ownership'])
df.drop(columns=["source_file", 'biome'], inplace=True)
df = df.rename(columns={'BIOME_NAME': 'biome'})
df=df.merge(y_df, on='PID', how='inner')

df.to_csv("data/final/final_dataset.csv", index=False)