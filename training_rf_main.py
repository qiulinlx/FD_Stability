import pandas as pd
import geopandas as gpd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

import utils.cross_validation as cval
import random 
import matplotlib.pyplot as plt

import utils.model_utils as mu
import utils.eval_utils as ev
import shap
import numpy as np
from sklearn.inspection import partial_dependence

def hyperparameter_tuning(df, RANDOM_KEY):
    X_train, y_train, _, _ = mu.data_preprocessing(df, RANDOM_KEY)
    model=RandomForestRegressor(
            random_state=RANDOM_KEY)

    search = RandomizedSearchCV(
            model,
            param_distributions=param_grid,
            n_iter=10,           # number of random combinations to try
            cv=4,                 # 5-fold cross validation
            scoring='r2',         # or 'neg_mean_squared_error'
            n_jobs=4,            # use all cores
            verbose=2,
            random_state=RANDOM_KEY
        )
    
    search.fit(X_train, y_train)

    model = search.best_estimator_
    model__hyperparam_df = pd.DataFrame(search.cv_results_)

    return model, model__hyperparam_df


if __name__ == "__main__":
    TEST_SIZE = 0.2
    BATCH_SIZE=16

    # ── Parameter grid ──
    param_grid = {
        'n_estimators':      [100, 200, 300, 500],
        'max_depth':         [5, 10, 20, 30, 40],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf':  [1, 2, 4],
        'max_features':      ['sqrt'],
        'bootstrap':         [True, False]
    }

    fd_df= pd.read_csv("data/final/fd_df.csv")
    sd_df = pd.read_csv("data/final/sd_df.csv")

    ecoregions=cval.process_ecoregion("data/Ecoregions/Ecoregions2017.shp")

    ecoregions=ecoregions[['ECO_NAME', 'geometry']]

    fd_df = fd_df.loc[:, ~fd_df.columns.str.contains(r'\.1')]
    sd_df = sd_df.loc[:, ~sd_df.columns.str.contains(r'\.1')]

    fd_df.drop(columns=['Mean Pairwise D'], inplace=True)
    fd_df.dropna(subset=['TPA_UNADJ'], inplace=True) # Maybe exclude

    sd_df.dropna(subset=['TPA_UNADJ'], inplace=True) # Maybe exclude
    sd_df = sd_df[sd_df["WSCI"] >= -1000]
    fd_df = fd_df[fd_df["WSCI"] >= -1000]


    sd_df.columns = sd_df.columns.str.replace('_x$', '', regex=True)
    sd_df.columns = sd_df.columns.str.replace('_y$', '', regex=True)

    sd_df = sd_df.loc[:, ~sd_df.columns.duplicated()]


    shared_pids = set(sd_df["PID"]).intersection(fd_df["PID"])

    # Filter both dataframes to only shared PIDs
    sd_df = sd_df[sd_df["PID"].isin(shared_pids)]
    fd_df = fd_df[fd_df["PID"].isin(shared_pids)]

    for i in range(10):

        RANDOM_KEY = random.randint(0, 1000)
        print(RANDOM_KEY) # For different splits 

        sX_train, sy_train, sX_test, sy_test = mu.data_preprocessing(sd_df, RANDOM_KEY)
        fX_train, fy_train, fX_test, fy_test = mu.data_preprocessing(fd_df, RANDOM_KEY)

        sd_reg=RandomForestRegressor(
            random_state=RANDOM_KEY)

        # sd_reg.fit(sX_train, sy_train)


        search = RandomizedSearchCV(
            sd_reg,
            param_distributions=param_grid,
            n_iter=20,           # number of random combinations to try
            cv=5,                 # 5-fold cross validation
            scoring='r2',         # or 'neg_mean_squared_error'
            n_jobs=4,            # use all cores
            verbose=2,
            random_state=RANDOM_KEY
        )

        search.fit(sX_train, sy_train)

        print("Best sd params:  ", search.best_params_)
        print("Best sd CV score:", search.best_score_)

        # ── Refit with best params ──
        sd_reg = search.best_estimator_
        sd__hyperparam_df = pd.DataFrame(search.cv_results_)

        fd_reg=RandomForestRegressor(
            random_state=RANDOM_KEY,
        )

        search = RandomizedSearchCV(
            fd_reg,
            param_distributions=param_grid,
            n_iter=20,           # number of random combinations to try
            cv=5,                 # 5-fold cross validation
            scoring='r2',         # or 'neg_mean_squared_error'
            n_jobs=4,            # use all cores
            verbose=2,
            random_state=RANDOM_KEY
        )

        search.fit(fX_train, fy_train)

        print("Best fd params:  ", search.best_params_)
        print("Best fd score:", search.best_score_)

        # ── Refit with best params ──
        fd_reg = search.best_estimator_
        fd__hyperparam_df = pd.DataFrame(search.cv_results_)

        smae, sr2 = ev.quick_eval(sd_reg, sX_test, sy_test)  
        fmae, fr2 = ev.quick_eval(fd_reg, fX_test, fy_test)

        print(f"SD MAE: {smae}, SD R2: {sr2}")
        print(f"FD MAE: {fmae}, FD R2: {fr2}")

        fd__hyperparam_df.to_csv(f"results/fd_hyperparam_tuning_results{RANDOM_KEY}.csv", index=False)
        sd__hyperparam_df.to_csv(f"results/sd_hyperparam_tuning_results{RANDOM_KEY}.csv", index=False)

        # # Done training, now to get the SHAP Values 

        # fshap_df, fexplainer = ev.SHAP_plots(fd_reg, fX_test,
        #                                     colour="#4b72f1", 
        #                                     plot_name=f"fd_shap_plot_{RANDOM_KEY}.png"
        #                                     )   
        
        # fshap_values = fexplainer.shap_values(fX_test, approximate=True)

        # diversity_vars_f = ['Raos_Q', 'Functional_Evenness', 'Species Richness', 
        #                     'WSCI', r"Gini Coefficient$_{DBH}$", r'Maximum $_{DBH}$'
        #                     ]
        # diversity_vars_s = ["Simpson's Index", 'Shannon Diversity', 'Species Richness', 
        #                     'WSCI', r"Gini Coefficient$_{DBH}$", r'Maximum $_{DBH}$'
        #                     ]

        # ev.plot_pdp(diversity_vars_f, diversity_vars_s, 
        #             fd_reg, sd_reg, fX_test, sX_test,
        #               f"pdp_plot_{RANDOM_KEY}.png")
