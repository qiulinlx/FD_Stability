from shapely.geometry import shape
import geopandas as gpd
import json
import pandas as pd
from scipy.signal import detrend

def detrend_pid(group):
    # Sort by year
    group = group.sort_values('year')
    # Detrend NPP
    group['NPP_detrended'] = detrend(group['Npp'].values)
    return group

def autocorr_pid(group, max_lag=9):
    group = group.sort_values('year')
    acfs = [group['NPP_detrended'].autocorr(lag=lag) for lag in range(1, max_lag+1)]
    return pd.Series(acfs, index=[f'ACF_lag{lag}' for lag in range(1, max_lag+1)])

df_joined=pd.read_csv('data/processed/PID_npp.csv')

df_joined = df_joined.groupby('PID').apply(detrend_pid).reset_index(drop=True)
df_joined.drop(columns=(['Unnamed: 0']), inplace=True)

autocorr_df = df_joined.groupby('PID').apply(autocorr_pid).reset_index()

autocorr_df["mean"] = autocorr_df.drop(columns=["PID"]).mean(axis=1)
df= autocorr_df[["PID", "mean"]]
df = df.rename(columns={"mean": "TAC_NPP"})
df.to_csv('npp_processed.csv')