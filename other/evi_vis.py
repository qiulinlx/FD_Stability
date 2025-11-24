import numpy as np
import pandas as pd
from scipy.stats import kurtosis
import matplotlib.pyplot as plt
from scipy.fft import fft

df=pd.read_csv('data/EVI_PID.csv')
pid=pd.read_csv('data/lookup/PID_location.csv')

# Temporal EVI Metrics

pid = df['PID'].sample(1).iloc[0]
sub = df[df['PID'] == pid]

sub = sub.dropna(axis=1)
sub.drop(columns=['PID'], inplace=True)

dates=sub.columns.tolist()
evi=np.array(sub)
evi=evi.flatten()

sub1=pd.Series(evi, index=pd.to_datetime(dates))

pearson =sub1.autocorr()  
k = kurtosis(sub1, fisher=True, bias=False)

plt.figure(figsize=(20,5))
plt.plot(dates, evi)
plt.title(f"EVI Time Series for PID: {pid} with AR(1)={pearson} and kurtosis={k}", fontsize=16)

plt.figure(figsize=(15,8))
plt.plot(dates[1:], fft(evi)[1:])
plt.title(f" FFT of EVI for PID: {pid} with std={np.std(fft(evi)[1:])} ", fontsize=16)


plt.show()