import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from utils.data_utils import data_preprocessing
from utils.data_utils import evaluate_rf
import utils.analysis_functions as af
import utils.cross_validation as cval

from shapely.geometry import box



if __name__ == "__main__":

    TEST_SIZE = 0.2
    RANDOM_KEY = 42

   
    df= pd.read_csv("data/final/fd_df.csv")
    PID_loc= pd.read_csv("data/lookup/PID_location_all.csv")

    ecoregions=cval.process_ecoregion("data/Ecoregions/Ecoregions2017.shp")

    df=df.merge(PID_loc, on="PID", how="left")
    df.dropna(subset=["lat", "lon"], inplace=True)

    df = df.drop(columns=[col for col in df.columns if "_y" in col])
    df.columns = df.columns.str.replace('_x$', '', regex=True)

    df=cval.assign_spatial_groups(df, grid_size=1.0)

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:4326"  # WGS84
    )

    train, test= cval.ecoregion_cross_validation(gdf, ecoregions, TEST_SIZE, BATCH_SIZE)

    train = train.loc[:, :'csc']
    test = test.loc[:, :'csc']

    y=['transformed npp', 'csc'] 

    X_train=train.drop(columns=y+['PID']).values
    y_train = np.column_stack([train['transformed npp'], train['csc']])

    X_test=test.drop(columns=y+['PID']).values
    y_test = np.column_stack([test['transformed npp'], test['csc']])


    fd_reg = RandomForestRegressor(random_state=RANDOM_KEY, n_jobs=4)
    fd_reg.fit(fX_train, fy_train)


    sd_reg = RandomForestRegressor(random_state=RANDOM_KEY, n_jobs=4)
    sd_reg.fit(sX_train, sy_train)


    evaluate_rf(fX_test, fy_test, fd_reg, feature_names=fd_x.columns, importance= True, div_type= 'f')
    evaluate_rf(sX_test, sy_test, sd_reg, feature_names=sd_x.columns, importance= True,  div_type= 's')