import numpy as np
import pandas as pd

def gini_coefficient(dbh_values):
    """
    Calculate the Gini coefficient of DBH values.
    
    Parameters:
        dbh_values: array-like of DBH measurements (any unit, must be positive)
    
    Returns:
        Gini coefficient (float between 0 and 1)
    """
    dbh = np.array(dbh_values, dtype=float)
    
    if len(dbh) < 2:
        raise ValueError("Need at least 2 trees")
    if np.any(dbh <= 0):
        raise ValueError("All DBH values must be positive")
    
    # Sort ascending
    dbh = np.sort(dbh)
    n = len(dbh)
    
    # Standard Lorenz-curve formulation
    # GC = (2 * sum(i * x_i)) / (n * sum(x_i)) - (n + 1) / n
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * dbh) / (n * np.sum(dbh))) - (n + 1) / n

PID_df=pd.read_csv('data/lookup/PID_location_all.csv')

gc=[]

Pids=list(set(PID_df['PID']))

for pid in Pids.copy():
    plot=PID_df.loc[PID_df['PID'] == pid]
    dbh_values=plot['DIA'].values

    if len(dbh_values) > 2:
        gini=gini_coefficient(dbh_values)
        gc.append(gini)
    else: 
        Pids.remove(pid)

df=pd.DataFrame({'PID':Pids,'GC':gc})
df.to_csv('data/processed/PID_GCDBH.csv',index=False)