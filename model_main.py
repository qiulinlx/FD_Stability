import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt

def create_sub_df(df, list):
    """
    Docstring for create_sub_df
    
    :param df: Description
    :param list: List of elevemtns to drop
    """

    sdf=df.copy()
    sdf.drop(columns=list, inplace=True)
    sdf=sdf.merge(y_df, on='PID')
    sdf.drop(columns=['year', "source_file"], inplace=True)
    sdf.dropna(axis=0, inplace=True)

    return sdf

def evaluate_rf(X_test, y_test, feature_names: list, importance: bool):

    y_pred = regr.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred, multioutput='raw_values')

    r2 = r2_score(y_test, y_pred, multioutput='raw_values')
    if importance: 
        
        importances = regr.feature_importances_

        feature_imp_df = pd.DataFrame({'Feature': fd_x.columns, 'Gini Importance': importances})
        fd_i = feature_imp_df.loc[feature_imp_df['Feature'].isin(fd_list)].sum()
        fd_i['Gini Importance']
        new_row = {'Feature': 'Functional Diversity', 'Gini Importance': fd_i['Gini Importance']}
        feature_imp_df = pd.concat([feature_imp_df, pd.DataFrame([new_row])], ignore_index=True).sort_values('Gini Importance', ascending=False)
        print(feature_imp_df)
        
        plt.figure(figsize=(8, 4))
        plt.barh(feature_names, importances, color='skyblue')
        plt.xlabel('Gini Importance')
        plt.title('Feature Importance - Gini Importance')
        plt.gca().invert_yaxis()
        plt.savefig(f'results/rf_importances{np.random.randint(1,100)}.png')

    with open("results/experiments.txt", "w") as f:
        f.write(f" \n R2 output: {r2}, MAE output: {mae} \n")

df = pd.read_parquet("dataset.parquet")
y_df= pd.read_csv('data/MODIS_data/MOD17_Annual_NPP_per_PID.csv')
csc_df= pd.read_csv('data/processed/PID_csc_variability.csv')

# Preprocessing 

sd_list=['Species Richness', 'Shannon Diversity', "Simpson's Index", "Shannon Equitabiltiy Index"]
fd_list=['Raos_Q', 'Functional_Evenness', "Functional_Dispersion", "Functional_Divergences", "Mean Pairwise D"]
y=['npp', 'csc']


y_df=y_df.merge(csc_df,  on='PID', how='inner')
y_df.drop(columns=['system:index', 'stdDev', '.geo', 'y', 'x'], inplace=True)
y_df = y_df[y_df["csc"] != -3.4028235e+38]
y_df.rename(columns={"mean": "npp"}, inplace=True)


fd_df= create_sub_df(df, sd_list)
sd_df=create_sub_df(df, fd_list)


fd_x=fd_df.drop(columns=y+['PID'])
y = np.column_stack([fd_df['npp'], fd_df['csc']])

sd_x=sd_df.drop(columns=y+['PID'])
sy = np.column_stack([sd_df['npp'], sd_df['csc']])

X_train, X_test, y_train, y_test = train_test_split(fd_x, y, test_size=0.2, random_state=42, stratify=y)

regr = RandomForestRegressor(random_state=0)
regr.fit(X_train, y_train)

evaluate_rf(X_test, y_test, regr, feature_names=fd_x.columns, importance= True)