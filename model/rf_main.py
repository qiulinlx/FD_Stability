import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from utils.data_utils import data_preprocessing
from utils.data_utils import evaluate_rf


if __name__ == "__main__":

    TEST_SIZE = 0.2
    RANDOM_KEY = 42

    df = pd.read_parquet("data/dataset.parquet")
    y_df= pd.read_csv('npp_processed.csv')
    csc_df= pd.read_csv('data/processed/PID_csc_upsampled.csv')
    
    csc_df.rename(columns={'PID_left': "PID"}, inplace= True)

    y=['TAC_NPP', 'csc'] 

    sd_df, fd_df=data_preprocessing(df, y_df, csc_df)


    fd_x=fd_df.drop(columns=y+['PID'])
    fy = np.column_stack([fd_df['TAC_NPP'], fd_df['csc']])

    sd_x = sd_df.drop(columns=y + ['PID'])
    sy = np.column_stack([sd_df['TAC_NPP'], sd_df['csc']])

    fX_train, fX_test, fy_train, fy_test = train_test_split(fd_x, fy, test_size=TEST_SIZE, random_state=RANDOM_KEY, stratify=fy)
    sX_train, sX_test, sy_train, sy_test = train_test_split(sd_x, sy, test_size=TEST_SIZE, random_state=RANDOM_KEY, stratify=sy)

    fd_reg = RandomForestRegressor(random_state=RANDOM_KEY)
    fd_reg.fit(fX_train, fy_train)


    sd_reg = RandomForestRegressor(random_state=RANDOM_KEY)
    sd_reg.fit(sX_train, sy_train)


    evaluate_rf(fX_test, fy_test, fd_reg, feature_names=fd_x.columns, importance= True)
    evaluate_rf(sX_test, sy_test, sd_reg, feature_names=sd_x.columns, importance= True)