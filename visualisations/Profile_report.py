import pandas as pd
from pandas_profiling import ProfileReport
from utils.utils import truncate_after_n_underscores

x=pd.read_csv('data/FD_Metrics_California.csv')
y=pd.read_csv('data/stability_metrics.csv')



y["PID"] = y["PID"].apply(truncate_after_n_underscores)


df = pd.merge(x, y, left_on="PID", right_on="PID")
df.drop(columns=['PID','Functional_Richness', 'n_years', 'slope', 'intercept'], inplace=True)
df.rename(columns={'residual_sd': 'Stability'}, inplace=True)
df.fillna(0, inplace=True)

profile= ProfileReport(df, title="Linking FD Traits to Stability")

profile.to_file("evi_stability_profile.html")