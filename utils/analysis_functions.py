import pandas as pd
from scipy.signal import detrend
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from arch import arch_model

import matplotlib.colors as mcolors

# Assign a color to each forest type
forest_colors = {
    'Temperate conifer forests': 'lightgreen',
    'Temperate grasslands': 'lightblue',
    'Xeric shrublands': 'wheat',
    'Mediterranean woodlands': 'orange',
    'Tropical coniferous forests': 'darkgreen',
    'No Data': 'gray'
}

managed= {
    1.0: 'blue',
    0.0: 'red',
    -1.0: 'gray'
}

ownership = {
    'national_forest': 'darkblue',
    'other': 'lightgray',
    'state': 'lightgreen',
    'blm': 'orange',
    'national_park': 'darkgreen',
    'local': 'purple',
    'other_federal': 'brown',
    'fish_wildlife': 'pink',
    'dod': 'gray',
    'other_forest_service': 'lightblue',
    'national_grassland': 'yellow',
    'No Data': 'gray'

}

def arch_coeff(series):
    series = series.dropna()
    if len(series) < 5:
        return None
    # series = series * 10
    am = arch_model(series, vol='ARCH', p=1)
    res = am.fit(disp='off')
    
    return res.params['alpha[1]']

def compute_volatility(arr): 
    arr= arr*10
    arr = pd.Series(detrend(arr))
    v=arr.rolling(window=2).std()
    v.dropna(inplace=True)
    v=v.mean()
    return v 

def ols_ar1(group, npp_col="Npp"):
    group = group.sort_values("year")

    series = pd.to_numeric(group[npp_col], errors="coerce").dropna()

    # need at least 3 points
    if len(series) < 3:
        return pd.Series({"transformed npp": np.nan})

    X = series.shift(1).iloc[1:]
    y = series.iloc[1:]

    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()
    phi = model.params.iloc[1]

    return pd.Series({"transformed npp": phi})

def compute_acf_lags(group, npp_col="Npp", max_lag=3):
    group = group.sort_values("year")
    series = pd.to_numeric(group[npp_col], errors="coerce").dropna()

    # Need enough observations
    if len(series) < (max_lag + 2):
        return pd.Series({f"acf_lag{i}": np.nan for i in range(1, max_lag+1)})

    try:
        acf_vals = acf(series, nlags=max_lag, fft=False)

        # skip lag 0 (always 1)
        return pd.Series({
            f"acf_lag{i}": acf_vals[i] for i in range(1, max_lag+1)
        })

    except Exception:
        return pd.Series({f"acf_lag{i}": np.nan for i in range(1, max_lag+1)})


def cleaning(PID_df, csc_df, npp_df):
    npp_df.drop(columns=(['Unnamed: 0']), inplace=True)

    PID_df["managed"] = PID_df["managed"].fillna(-1)
    PID_df["ownership"] = PID_df["ownership"].fillna("No Data")
    PID_df["biome"] = PID_df["biome"].fillna("No Data")
    if 'lon' in PID_df.columns and 'lat' in PID_df.columns:
        PID_df = PID_df[PID_df["lon"] <= 0]
        PID_df = PID_df[PID_df["lat"] > 22]

    csc_df.rename(columns={'PID_left': "PID"}, inplace= True)
    csc_df.drop(columns=['Unnamed: 0'], inplace=True)

    csc_df=csc_df.merge(PID_df, on='PID', how='inner')

    return PID_df, csc_df, npp_df 

def autocorr_pid(group,n_lag=1):

    "fix so we only go"
    group = group.sort_values('year')
    acfs = [group['Npp'].autocorr(lag=n_lag)]
    return pd.Series(acfs, index=['transformed npp'])

def compute_residuals(group):
    X = group["year"].values.reshape(-1,1)
    y = group["Npp"].values
    
    model = LinearRegression().fit(X, y)
    y_hat = model.predict(X)
    
    group["transformed npp"] = y - y_hat
    return group

def detrend_pid(group):

    """
    Manual Detrending
    
    """
    # Sort by year
    group = group.sort_values('year')
    # Detrend NPP
    group['Npp'] = detrend(group['Npp'].values)
    return group

#check all code from here

def histogram(df, column, title):
    plt.figure(figsize=(6,4))
    plt.hist(df[column], bins=40)
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.title(title)
    plt.show()


def spatial_scatter(x_col, y_col, color_col, title, cmap, vcenter=None,):

    norm = None
    
    if vcenter is not None:
        
        norm = mcolors.TwoSlopeNorm(
            vmin=color_col.min(),
            vcenter=vcenter,
            vmax=color_col.max()
        )

    plt.figure(figsize=(12,5))
    plt.scatter(x_col, y_col, c=color_col, s=1, cmap=cmap, norm=norm)
    plt.colorbar()
    plt.title(title)
    plt.show()

def ecoregion_plot(gdf, na_ecoregions, value_col):
    
    ax = na_ecoregions.plot(
        color=na_ecoregions['COLOR_BIO'],  # or use a column like 'ECO_NAME'
        edgecolor='grey',
        figsize=(12,8)
    )

    gdf.plot(
        ax=ax,
        column=value_col,
        cmap='PiYG',
        markersize=0.5,
        alpha=0.2)

    # Zoom to North America
    ax.set_xlim(-140, -60)  # longitude
    ax.set_ylim(20, 53)      # latitude

    plt.show()


    plt.figure(figsize=(20,10))


def spatial_scatter_sp(
    df,
    x_col,
    y_col,
    color_col,
    title,
    add=None,          # dict: {category_value: color}
    add_col=None,
    cmap = 'Greens'      # column to group by
):
    plt.figure(figsize=(8,6))

    if add is not None and add_col is not None:
        for ftype, color in add.items():
            subset = df[df[add_col] == ftype]

            plt.scatter(
                subset[x_col],
                subset[y_col],
                color=color,
                alpha=0.5,
                s=30,
                label=f"{ftype}"
            )

    sc = plt.scatter(
        df[x_col],
        df[y_col],
        c=df[color_col],
        s=1,
        cmap=cmap
    )

    plt.colorbar(sc, label=color_col)


    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)

    plt.tight_layout()
    plt.show()