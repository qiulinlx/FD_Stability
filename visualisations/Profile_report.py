import pandas as pd
from pandas_profiling import ProfileReport
from utils.utils import truncate_after_n_underscores

x=pd.read_csv("data/processed/FD_tree.csv")
y=pd.read_csv('data/stability_metrics.csv')

df = pd.merge(x, y, left_on="PID", right_on="PID")
df = df.dropna()
df.drop(columns=['PID',	'Unnamed: 0', 'n_years', 'slope', 'intercept'], inplace=True)
df.rename(columns={'residual_sd': 'Stability'}, inplace=True)
df.fillna(0, inplace=True)
profile= ProfileReport(df, title="Linking FD Traits to Stability")

profile.to_file("evi_stability_profile1.html")