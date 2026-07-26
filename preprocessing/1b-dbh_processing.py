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

PID_df.dropna(subset=['DIA'],inplace=True)
PID_df = PID_df[PID_df['DIA'] != 0]

pid_df = PID_df.groupby('PID')['DIA'].apply(list).reset_index()
pid_df.columns = ['PID', 'DIA']

pid_df = pid_df[pid_df['DIA'].apply(len) >= 2]

gc=[]
pid=[]
mx=[]
i=0
for idx, row in pid_df.iterrows():
    
    dbh = np.array(row['DIA'])    
    if len(dbh) > 2:
        gini= gini_coefficient(dbh)
        mx.append(max(dbh))
        gc.append(gini)
        pid.append(row['PID'])
        i+=1
    else: 
        print('did not compute')

df=pd.DataFrame({'PID':pid,'GC':gc,'Max_DBH':mx})
df.to_csv('data/processed/PID_GCDBH.csv',index=False)