import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

x=pd.read_csv('data/FD_Metrics_California.csv')
y=pd.read_csv('data/stability_metrics.csv')

def truncate_after_4_underscores(s: str) -> str:
    """
    Truncate a string after the fourth underscore.
    Using this to standardize plot IDs (PIDs).

    Args:
        s (str): Input string.

    Returns:
        str: String containing only the first four underscore-separated segments.
    """
    parts = s.split("_")
    return "_".join(parts[:4])

y["PID"] = y["PID"].apply(truncate_after_4_underscores)

df = pd.merge(x, y, left_on="PID", right_on="PID")
df.drop(columns=['PID','Functional_Richness', 'n_years', 'slope', 'intercept'], inplace=True)
df.rename(columns={'residual_sd': 'Stability'}, inplace=True)
df.fillna(0, inplace=True)

y=df.pop('Stability')
X=df


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=11)

regr = RandomForestRegressor(random_state=0)

regr.fit(X_train, y_train)

y_pred = regr.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print(f"R^2: {r2:.4f}, \n RMSE: {rmse:.4f}, \n MAE: {mae:.4f}")