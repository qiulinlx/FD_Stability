import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
import os
from pathlib import Path
import shutil

def merge_files(csv_dir, out_file):

    dtype_map = {
        "value": "float64"
    }
    dfs = []
    for csv in csv_dir.glob("*.csv"):
        df = pd.read_csv(csv, dtype=dtype_map)
        df["source_file"] = csv.name   # optional but VERY useful
        dfs.append(df)

    stacked = pd.concat(dfs, ignore_index=True)

    stacked.to_parquet(out_file, engine="pyarrow", compression="snappy")
    shutil.rmtree(csv_dir)  


def data_preprocessing(df, npp_df, csc_df):
    """
    Cleaning and merging datasets for ML models
    """

    y_df=npp_df.merge(csc_df,  on='PID', how='inner')
    y_df.rename(columns={"mean": "npp"}, inplace=True)

    print(y_df.shape)
    # y_df.drop(columns=['Unnamed: 0_x', 'Unnamed: 0_y'], inplace=True)

    # df = pd.get_dummies(df, columns=['biome', 'ownership'])
    # df = df.replace({True: 1, False: 0})
    le = LabelEncoder()
    df['biome'] = le.fit_transform(df['biome'])

    le = LabelEncoder()
    df['ownership'] = le.fit_transform(df['ownership'])

    fd_df=df.copy()
    fd_df.drop(columns=['Species Richness', 'Shannon Diversity', "Simpson's Index", "Shannon Equitabiltiy Index", "source_file",  "Functional_Dispersion", "Functional_Divergences"], inplace=True)
    fd_df=fd_df.merge(y_df, on='PID', how='inner')

    fd_df = fd_df.drop(columns=[col for col in fd_df.columns if "_y" in col])
    fd_df.columns = fd_df.columns.str.replace('_x$', '', regex=True)
    fd_df.dropna(subset=['Functional_Evenness', "Mean Pairwise D"], inplace=True)

    sd_df=df.copy()
    sd_df.drop(columns=['Raos_Q', 'Functional_Evenness', "Functional_Dispersion", "Functional_Divergences", "Mean Pairwise D", "Shannon Equitabiltiy Index", "source_file"], inplace=True)
    sd_df=sd_df.merge(y_df, on='PID', how='inner')
    sd_df.dropna(subset=['Species Richness', 'Shannon Diversity', "Simpson's Index"], inplace=True)

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
        
        mask = feature_imp_df['Feature'].isin(f_list)

        # Sum their importance
        group_sum = feature_imp_df.loc[mask, 'Gini Importance'].sum()

        # Remove those rows
        feature_imp_df = feature_imp_df.loc[~mask]

        # Add new aggregated row
        new_row = pd.DataFrame([{
            'Feature': a,  # your new grouped feature name
            'Gini Importance': group_sum
        }])

        feature_imp_df = pd.concat([feature_imp_df, new_row], ignore_index=True)

                
        
        feature_imp_df = feature_imp_df.sort_values('Gini Importance', ascending=False)

        plt.figure(figsize=(20, 10))

        plt.barh(
            feature_imp_df['Feature'],
            feature_imp_df['Gini Importance']
        )

        plt.xlabel('Gini Importance')
        plt.title('Feature Importance - Gini Importance')

        # Most important at the top
        plt.gca().invert_yaxis()

        plt.savefig(f'results/rf_importances{np.random.randint(1,100)}.png')

    with open("results/experiments.txt", "a") as f:
        f.write(f" {a}:  \n R2 output: {r2}, MAE output: {mae} \n")
csv_dir = Path("data/.joined")
out_file = "dataset.parquet"