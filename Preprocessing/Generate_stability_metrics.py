import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import kurtosis

import matplotlib.pyplot as plt


df=pd.read_csv('data/EVI_PID.csv')

pid=pd.read_csv('data/lookup/PID_location.csv')

df1=df.copy()
df1.index=df1['PID']
df1=df1.drop(columns=['PID'])
df1['row_mean'] = df1.mean(axis=1, skipna=True)
df1=df1[["row_mean"]]

s_df=df1.merge()

# Temporal EVI Metrics

pid = df['PID'].sample(1).iloc[0]
sub = df[df['PID'] == pid]

sub = sub.dropna(axis=1)
sub.drop(columns=['PID'], inplace=True)

dates=sub.columns.tolist()
evi=np.array(sub)
sub1=pd.Series(evi.flatten(), index=pd.to_datetime(dates))
pearson =sub1.autocorr()  

k = kurtosis(sub1, fisher=True, bias=False)
print(k)


plt.figure(figsize=(20,5))
plt.plot(dates, evi.flatten())
plt.title(f"EVI Time Series for PID: {pid} with AR(1)={pearson}", fontsize=16)
plt.show()