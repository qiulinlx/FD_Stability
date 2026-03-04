import pandas as pd

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
import numpy as np

import matplotlib.pyplot as plt


def data_preprocessing(df, npp_df, csc_df):
    """
    Cleaning and merging datasets for ML models
    """

    y_df=npp_df.merge(csc_df,  on='PID', how='inner')
    y_df.rename(columns={"mean": "npp"}, inplace=True)
    y_df.drop(columns=['Unnamed: 0_x', 'Unnamed: 0_y'], inplace=True)


    fd_df=df.copy()
    fd_df.drop(columns=['Species Richness', 'Shannon Diversity', "Simpson's Index", "Shannon Equitabiltiy Index", "source_file",  "Functional_Dispersion", "Functional_Divergences"], inplace=True)
    fd_df=fd_df.merge(y_df, on='PID')

    fd_df=fd_df.dropna()
    sd_df=df.copy()
    sd_df.drop(columns=['Raos_Q', 'Functional_Evenness', "Functional_Dispersion", "Functional_Divergences", "Mean Pairwise D", "Shannon Equitabiltiy Index", "source_file"], inplace=True)
    sd_df=sd_df.merge(y_df, on='PID')
    sd_df.dropna(axis=0, inplace=True)

    return sd_df, fd_df


def evaluate_rf(X_test, y_test, regr, feature_names: list, importance: bool, div_type: str):

    y_pred = regr.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred, multioutput='raw_values')

    r2 = r2_score(y_test, y_pred, multioutput='raw_values')
    if importance: 
        importances = regr.feature_importances_

        if div_type == "f":
            f_list= ['Raos_Q', 'Functional_Evenness', "Mean Pairwise D"]
            a= 'Functional Diversity'
        else:
            f_list=['Species Richness', 'Shannon Diversity', "Simpson's Index"]
            a='Species Diversity'

        feature_imp_df = pd.DataFrame({'Feature': feature_names, 'Gini Importance': importances})
        fd_i = feature_imp_df.loc[feature_imp_df['Feature'].isin(f_list)].sum()
        fd_i['Gini Importance']
        new_row = {'Feature': a, 'Gini Importance': fd_i['Gini Importance']}
        feature_imp_df = pd.concat([feature_imp_df, pd.DataFrame([new_row])], ignore_index=True).sort_values('Gini Importance', ascending=False)
        print(feature_imp_df)
        
        plt.figure(figsize=(10, 4))
        plt.barh(feature_names, importances, color='skyblue')
        plt.xlabel('Gini Importance')
        plt.title('Feature Importance - Gini Importance')
        plt.gca().invert_yaxis()
        plt.savefig(f'results/rf_importances{np.random.randint(1,100)}.png')

    with open("results/experiments.txt", "a") as f:
        f.write(f" {a}:  \n R2 output: {r2}, MAE output: {mae} \n")
