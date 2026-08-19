import pandas as pd
import numpy as np
from scipy.signal import detrend

'''
We run this file to get cleaned datasets for modelling
'''

def cleaning(PID_df, csc_df, npp_df):

    # PID_df["managed"] = PID_df["managed"].fillna(-1)
    # PID_df["ownership"] = PID_df["ownership"].fillna("No Data")
    # PID_df["biome"] = PID_df["biome"].fillna("No Data")
    if 'lon' in PID_df.columns and 'lat' in PID_df.columns:
        PID_df = PID_df[PID_df["lon"] <= 0]
        PID_df = PID_df[PID_df["lat"] > 22]

    csc_df.rename(columns={'PID_left': "PID"}, inplace= True)

    csc_df=csc_df.merge(PID_df, on='PID', how='inner')

    return PID_df, csc_df, npp_df 

def compute_volatility(arr): 
    '''
    Inverse CoV
    '''
    s=arr.mean()

    arr= pd.Series(detrend(arr*10))
    
    v= (s/arr.std())
    return v , s

df1=pd.read_csv("data/raw/NPP_PIDs_1.csv")
df2=pd.read_csv("data/raw/NPP_PIDs_2.csv")

npp_df=pd.concat([df1, df2], ignore_index=True)

npp_df.rename(columns={'NPP_kgC_m2_yr': "Npp"}, inplace=True)

grouped = npp_df.groupby('PID').apply(lambda x: x.sort_values('year')['Npp'].to_numpy())
grouped = grouped[grouped.apply(lambda x: ~np.isnan(x).any())]

volatility=[]
PID_list=[]
mean=[]

for pid, arr in grouped.items():
    if arr.shape[0] > 5:
        if (arr == 0).sum() < 4: 
            v , s = compute_volatility(arr)
            volatility.append(v)
            PID_list.append(pid)
            mean.append(s)

npp_df = pd.DataFrame({'PID': PID_list, 'transformed npp': volatility, 'mean': mean})

npp_df.to_csv('data/processed/PID_npp_volatility.csv', index=False)