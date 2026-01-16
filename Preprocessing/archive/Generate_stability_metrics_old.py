import numpy as np
import pandas as pd
from scipy.stats import kurtosis
from scipy.fft import fft

df=pd.read_csv('data/EVI_PID.csv')
pid=pd.read_csv('data/lookup/PID_location.csv')

AR1=[]
Temp_kurt=[]
pids=[]
spectral_sd_list=[]
evi_sd=[]

def add_to_list(list, value):
    if np.isnan(value):
        list.append(np.nan)
    else:
        list.append(value)


for pid in pid['PID']:

    sub = df[df['PID'] == pid]

    if sub.empty:
        AR1.append(np.nan)
        pids.append(pid)
        Temp_kurt.append(np.nan)
        evi_sd.append(np.nan)
        spectral_sd_list.append(np.nan)
        continue

    sub = sub.dropna(axis=1)
    sub.drop(columns=['PID'], inplace=True)

    dates=sub.columns.tolist()
    evi=np.array(sub)
  
    sub1=pd.Series(evi.flatten(), index=pd.to_datetime(dates))
    spectral_sd = np.std(fft(evi[0])[1:])
    sd = np.std(evi[0])

    lag_cor =sub1.autocorr()  
    k = kurtosis(sub1, fisher=True, bias=False)

    evi_sd.append(sd)
    pids.append(pid)
    Temp_kurt.append(k)
    AR1.append(lag_cor)
    spectral_sd_list.append(spectral_sd)

df=pd.DataFrame({'PID': pids, 'AR1': AR1, 'Kurtosis': Temp_kurt, 'Temp Spectral SD':spectral_sd_list, 'EVI SD': evi_sd})
df.to_csv('data/temp_stability_metrics.csv', index=False)