import pandas as pd
import geopandas as gpd
from sklearn.ensemble import RandomForestRegressor
import utils.cross_validation as cval
import random 
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
import model_utils as mu
import shap
import numpy as np

def quick_eval(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred, multioutput='raw_values')
    r2 = r2_score(y_test, y_pred, multioutput='raw_values')
    return mae, r2

def get_mean_shap(model, X_test):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test, approximate=True)

    shap_exp = shap.Explanation(
        values=shap_values,
        base_values=np.repeat(explainer.expected_value, len(X_test)),
        data=X_test.values,
        feature_names=list(X_test.columns))
    mean_shap = np.abs(shap_exp.values).mean(axis=0)
    return mean_shap

if __name__ == "__main__":
    TEST_SIZE = 0.3
    BATCH_SIZE=16

    fshap_df=pd.DataFrame()
    sshap_df=pd.DataFrame()
    
    result_df=pd.DataFrame(columns=['random_key', 'sd_mae', 'sd_r2', 'fd_mae', 'fd_r2'])

    random_list =[]
    for i in range(10):

        RANDOM_KEY = random.randint(0, 1000)
        random_list.append(RANDOM_KEY)

        fd_df= pd.read_csv("data/final/fd_df.csv")
        sd_df = pd.read_csv("data/final/sd_df.csv")

        ecoregions=cval.process_ecoregion("data/Ecoregions/Ecoregions2017.shp")

        ecoregions=ecoregions[['ECO_NAME', 'geometry']]

        fd_df = fd_df.loc[:, ~fd_df.columns.str.contains(r'\.1')]
        sd_df = sd_df.loc[:, ~sd_df.columns.str.contains(r'\.1')]

        fd_df.drop(columns=['Mean Pairwise D'], inplace=True)
        fd_df.dropna(subset=['TPA_UNADJ'], inplace=True) # Maybe exclude

        sd_df.dropna(subset=['TPA_UNADJ'], inplace=True) # Maybe exclude

        sd_df.columns = sd_df.columns.str.replace('_x$', '', regex=True)
        sd_df.columns = sd_df.columns.str.replace('_y$', '', regex=True)

        sd_df = sd_df.loc[:, ~sd_df.columns.duplicated()]

        sX_train, sy_train, sX_test, sy_test = mu.data_preprocessing(sd_df, RANDOM_KEY)
        fX_train, fy_train, fX_test, fy_test = mu.data_preprocessing(fd_df, RANDOM_KEY)

        sd_reg=RandomForestRegressor(
        min_samples_split=3,
        min_samples_leaf=10,
        max_features='sqrt',
        random_state=RANDOM_KEY,
        n_jobs=-1
        )
        sd_reg.fit(sX_train, sy_train)

        fd_reg=RandomForestRegressor(
        min_samples_split=3,
        min_samples_leaf=10,
        max_features='sqrt',
        random_state=RANDOM_KEY,
        n_jobs=-1
        )
        fd_reg.fit(fX_train, fy_train)

        smae, sr2 = quick_eval(sd_reg, sX_test, sy_test)  
        fmae, fr2 = quick_eval(fd_reg, fX_test, fy_test)

        result_df.loc[len(result_df)] = [RANDOM_KEY, smae, sr2, fmae, fr2]

        # Done training, now to get the SHAP Values 

        smean_shap = get_mean_shap(sd_reg, sX_test)
        fmean_shap = get_mean_shap(fd_reg, fX_test)

        fshap_df.loc[len(fshap_df)] = fmean_shap 
        sshap_df.loc[len(sshap_df)] = smean_shap 
