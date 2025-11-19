import numpy as np
import pandas as pd
from scipy.stats import kurtosis

df=pd.read_csv('data/EVI_PID.csv')
pid=pd.read_csv('data/lookup/PID_location.csv')

AR1=[]
Temp_kurt=[]
pids=[]

for pid in pid['PID']:

    sub = df[df['PID'] == pid]

    if sub.empty:
        AR1.append(np.nan)
        pids.append(pid)
        Temp_kurt.append(np.nan)
        continue

    sub = sub.dropna(axis=1)
    sub.drop(columns=['PID'], inplace=True)

    dates=sub.columns.tolist()
    evi=np.array(sub)
    sub1=pd.Series(evi.flatten(), index=pd.to_datetime(dates))

    lag_cor =sub1.autocorr()  
    k = kurtosis(sub1, fisher=True, bias=False)

    pids.append(pid)
    Temp_kurt.append(k)
    AR1.append(lag_cor)


df=pd.DataFrame({'PID': pids, 'AR1': AR1, 'Kurtosis': Temp_kurt})
df.to_csv('data/temp_stability_metrics.csv', index=False)