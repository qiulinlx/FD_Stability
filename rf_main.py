import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from utils.data_utils import data_preprocessing
from utils.data_utils import evaluate_rf
import utils.analysis_functions as af

if __name__ == "__main__":

    TEST_SIZE = 0.2
    RANDOM_KEY = 21

    df = pd.read_parquet("data/dataset.parquet")
    npp_df=pd.read_csv('data/processed/PID_npp.csv')
    csc_df= pd.read_csv('data/processed/PID_csc_upsampled.csv')

    PID_df, csc_df, npp_df = af.cleaning(df, csc_df, npp_df)

    npp_df=pd.read_csv('data/processed/npp_heterosced.csv')

    # Temporal Conditional Heteroscedasticity

    # autocorr_df = (
    #     npp_df.groupby("PID")["Npp"]
    #     .apply(af.arch_coeff)
    #     .reset_index(name="transformed npp")
    # )
    # npp_df= autocorr_df[["PID", "transformed npp"]]

    csc_df['csc'] = (csc_df['csc']-min(csc_df['csc']))/(max(csc_df['csc'])-min(csc_df['csc']))
    csc_df["BHAGE"] = csc_df["BHAGE"].fillna("0.0")
    csc_df=csc_df[['PID', 'csc']]

    y=['transformed npp', 'csc'] 

    sd_df, fd_df=data_preprocessing(df, npp_df, csc_df)

    fd_x=fd_df.drop(columns=y+['PID'])
    fy = np.column_stack([fd_df['transformed npp'], fd_df['csc']])

    sd_x = sd_df.drop(columns=y + ['PID'])
    sy = np.column_stack([sd_df['transformed npp'], sd_df['csc']])

    fX_train, fX_test, fy_train, fy_test = train_test_split(fd_x, fy, test_size=TEST_SIZE, random_state=RANDOM_KEY, stratify=fy)
    sX_train, sX_test, sy_train, sy_test = train_test_split(sd_x, sy, test_size=TEST_SIZE, random_state=RANDOM_KEY, stratify=sy)

    fd_reg = RandomForestRegressor(random_state=RANDOM_KEY, max_depth=5, n_jobs=4)
    fd_reg.fit(fX_train, fy_train)


    sd_reg = RandomForestRegressor(random_state=RANDOM_KEY, max_depth=5, n_jobs=4)
    sd_reg.fit(sX_train, sy_train)


    evaluate_rf(fX_test, fy_test, fd_reg, feature_names=fd_x.columns, importance= True, div_type= 'f')
    evaluate_rf(sX_test, sy_test, sd_reg, feature_names=sd_x.columns, importance= True,  div_type= 's')