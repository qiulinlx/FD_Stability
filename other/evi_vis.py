import numpy as np
import pandas as pd
from scipy.stats import kurtosis
import matplotlib.pyplot as plt


df=pd.read_csv('data/EVI_PID.csv')
pid=pd.read_csv('data/lookup/PID_location.csv')

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