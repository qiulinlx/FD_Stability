import pandas as pd
from scipy.signal import detrend

def autocorr_pid(group, max_lag=1):
    group = group.sort_values('year')
    acfs = [group['Npp'].autocorr(lag=1)]
    return pd.Series(acfs, index=[f'ACF_lag{lag}' for lag in range(1, max_lag+1)])


def detrend_pid(group):

    """
    Manual Detrending
    
    """
    # Sort by year
    group = group.sort_values('year')
    # Detrend NPP
    group['NPP_detrended'] = detrend(group['Npp'].values)
    return group

#check all code from here

def histogram(df, column, title):
    plt.figure(figsize=(6,4))
    plt.hist(df[column], bins=40)
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.title(title)
    plt.show()


def spatial_scatter(df, x_col, y_col, color_col, title):
    plt.figure(figsize=(12,5))
    plt.scatter(df[x_col], df[y_col], c=df[color_col], s=1, cmap='RdYlGn', norm=norm)
    plt.colorbar(label=color_col)
    plt.title(title)
    plt.show()

def ecoregion_plot(df, ecoregion_col, value_col, title):
    plt.figure(figsize=(12,5))
    df.groupby(ecoregion_col)[value_col].mean().plot(kind='bar')
    plt.xlabel(ecoregion_col)
    plt.ylabel(f'Average {value_col}')
    plt.title(title)
    plt.xticks(rotation=45)
    plt.show()