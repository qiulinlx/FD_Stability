import pandas as pd
import numpy as np
import utils.analysis_functions as af
from scipy.signal import detrend
from utils.data_utils import data_preprocessing

'''
We run this file to get cleaned datasets for modelling
'''

PID_df=pd.read_csv('data/lookup/PID_location_all.csv')
csc_df= pd.read_csv('data/processed/PID_location_WSCI.csv')
npp_df=pd.read_csv('data/processed/PID_npp.csv')
df = pd.read_parquet("data/processed/dataset1.parquet")


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

# csc_df['csc'] = (csc_df['csc']-min(csc_df['csc']))/(max(csc_df['csc'])-min(csc_df['csc']))
# csc_df["BHAGE"] = csc_df["BHAGE"].fillna("0.0")
# csc_df=csc_df[['PID', 'csc']]

# y=['transformed npp', 'csc'] 

sd_df, fd_df=data_preprocessing(df, npp_df, csc_df)

fd_df.drop_duplicates(subset=['PID'], inplace=True)
sd_df.drop_duplicates(subset=['PID'], inplace=True)


fd_df.drop_duplicates(inplace=True)
sd_df.drop_duplicates(inplace=True)
sd_df.to_csv('data/processed/sd_df.csv', index=False)
fd_df.to_csv('data/processed/fd_df.csv', index=False)