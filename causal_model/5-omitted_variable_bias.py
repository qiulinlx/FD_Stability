"""
OVB sensitivity analysis for the partially linear DML model, from scratch.
Chernozhukov, Cinelli, Newey, Sharma & Syrgkanis (2022), "Long Story Short".

Model:   Y = theta * D + g0(X) + e
Inputs:  y, d       original outcome and treatment
         l_hat      cross-fitted (out-of-fold) prediction of E[Y | X]
         m_hat      cross-fitted (out-of-fold) prediction of E[D | X]
"""
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import pandas as pd
from sklearn.model_selection import KFold                                  # NEW
from xgboost import XGBRegressor                                           # NEW


def dml_sensitivity(y, d, l_hat, m_hat, cf_y=0.03, cf_d=0.03, rho=1.0, level=0.95):
    y, d, l_hat, m_hat = map(np.asarray, (y, d, l_hat, m_hat))
    n = len(y)

    u = y - l_hat                     # outcome residual
    v = d - m_hat                     # treatment residual

    # --- 1. DML point estimate and its influence function (partialling-out score)
    theta = np.sum(u * v) / np.sum(v * v)
    J = np.mean(v * v)
    psi_theta = (u - theta * v) * v / J          # influence function of theta

    # --- 2. Sensitivity elements
    eps = u - theta * v                           # Y - g(D, X)
    sigma2 = np.mean(eps ** 2)                    # E[(Y - g)^2]
    alpha = v / J                                 # Riesz representer
    nu2 = np.mean(alpha ** 2)                     # = 1 / E[v^2]
    S = np.sqrt(sigma2 * nu2)

    psi_sigma2 = eps ** 2 - sigma2                # influence functions of the
    psi_nu2 = nu2 - (v ** 2) * nu2 ** 2           # two sensitivity elements

    # --- 3. Bias bound and its influence function (delta method)
    C = np.abs(rho) * np.sqrt(cf_y * cf_d / (1 - cf_d))
    max_bias = C * S
    psi_bias = C / (2 * S) * (sigma2 * psi_nu2 + nu2 * psi_sigma2)

    theta_lo, theta_hi = theta - max_bias, theta + max_bias
    se_lo = np.sqrt(np.mean((psi_theta - psi_bias) ** 2) / n)
    se_hi = np.sqrt(np.mean((psi_theta + psi_bias) ** 2) / n)
    se = np.sqrt(np.mean(psi_theta ** 2) / n)

    z = norm.ppf(level)                           # one-sided, as in DoubleML
    ci_lo, ci_hi = theta_lo - z * se_lo, theta_hi + z * se_hi

    # --- 4. Robustness values (cf_y = cf_d = rv)
    # RV: point estimate driven to zero. Closed form from rv^2/(1-rv) = (theta/(rho*S))^2
    c = (theta / (np.abs(rho) * S)) ** 2
    rv = (-c + np.sqrt(c ** 2 + 4 * c)) / 2
    rv = min(rv, 1.0)

    # RVa: confidence bound reaches zero (solved numerically)
    def ci_edge(r):
        Cr = np.abs(rho) * np.sqrt(r * r / (1 - r))
        b = Cr * S
        pb = Cr / (2 * S) * (sigma2 * psi_nu2 + nu2 * psi_sigma2)
        if theta > 0:
            return (theta - b) - z * np.sqrt(np.mean((psi_theta - pb) ** 2) / n)
        return (theta + b) + z * np.sqrt(np.mean((psi_theta + pb) ** 2) / n)

    if (theta > 0 and theta - z * se <= 0) or (theta < 0 and theta + z * se >= 0):
        rva = 0.0                                 # already insignificant
    else:
        rva = brentq(ci_edge, 1e-10, 1 - 1e-10)

    return {
        "theta": theta, "se": se,
        "sigma2": sigma2, "nu2": nu2, "S": S, "max_bias": max_bias,
        "theta_lower": theta_lo, "theta_upper": theta_hi,
        "ci_lower": ci_lo, "ci_upper": ci_hi,
        "rv": rv, "rva": rva,
    }


def benchmark(y, d, l_hat, m_hat, l_hat_short, m_hat_short):
    """
    Implied strength of an observed covariate group Z, by refitting the nuisances
    WITHOUT Z (l_hat_short, m_hat_short) and comparing with the full model.
    """
    long_ = dml_sensitivity(y, d, l_hat, m_hat, cf_y=0, cf_d=0)
    short = dml_sensitivity(y, d, l_hat_short, m_hat_short, cf_y=0, cf_d=0)

    cf_y = (short["sigma2"] - long_["sigma2"]) / short["sigma2"]
    cf_d = (long_["nu2"] - short["nu2"]) / long_["nu2"]
    delta = short["theta"] - long_["theta"]
    denom = np.sqrt((short["sigma2"] - long_["sigma2"]) * (long_["nu2"] - short["nu2"]))
    rho = np.clip(delta / denom, -1, 1) if denom > 0 else 0.0
    return {"cf_y": cf_y, "cf_d": cf_d, "rho": rho, "delta_theta": delta}


def crossfit(X, target, folds):                                            # NEW
    pred = np.empty(len(target))
    for tr, te in folds:
        model = XGBRegressor(**xgb_params).fit(X.iloc[tr], target.iloc[tr])
        pred[te] = model.predict(X.iloc[te])
    return pred

# Data Cleaning and Preprocessing
fd_df = pd.read_csv("data/final/final_dataset_ba_v2.csv")
pred_df=pd.read_csv('results/step_1_residuals4_rarity.csv')
PID_df= pd.read_csv('data/lookup/PID_location_v3.csv')

fd_df=fd_df.merge(PID_df[['PID', 'STDAGE']], on='PID', how='left')

fd_df.drop(columns=['Unnamed: 0', 'managed', 'ownership', 'DIA', 'TPA_UNADJ','Functional_Richness', 'Shannon Equitabiltiy Index', 'transformed npp'], inplace=True)
fd_df.dropna(subset=['std npp', 'mean', 'Raos_Q', 'Functional_Evenness', 'Soil Moisture',
                      'Species Richness', 'Shannon Diversity', "Simpson's Index", "pet_std"], inplace=True)

fd_df = fd_df[fd_df["treecover2000"] > 10]
fd_df.drop_duplicates(subset=['PID'], inplace=True)

fd_df.rename(columns={'mean':'mean npp', 'pet_std': "PET sd", 'land_cover_value': "Land Cover", 'STDAGE': "Stand Age", 'CHELSA_exBIO_GrowingSeasonLength': "Growing Season Length"}, inplace=True)


fd_df['std npp']=np.log1p(fd_df['std npp'])
fd_df['mean npp']=np.log1p(fd_df['mean npp'])

fd_df = fd_df[fd_df["WSCI"] != 0]

fd_df = fd_df[fd_df["disturbance_value"] != -2147483648]

diversity_vars = ["Species Richness", "Shannon Diversity", "Raos_Q", "Functional_Evenness"]

# NEW: must match the covariates and XGBoost settings used in step 1
exclude = ['PID', 'std npp', "Simpson's Index", 'WSCI', 'treecover2000'] + diversity_vars
env_cols = [c for c in fd_df.columns if c not in exclude]

xgb_params = dict(n_estimators=300, max_depth=4, learning_rate=0.05)
benchmark_groups = {
    'climate & water': ['Annual Temp', 'Annual Precipitation', 'Precipitation Seasonality',
                        'Growing Season Length', 'CGIAR_Aridity_Index', 'PET sd', 'Soil Moisture'],
    'soil':            ['SOC Content', 'SG_CEC_015cm', 'SG_Soil_pH_H2O_015cm', 'Depth to Bedrock'],
    'topography':      ['Elevation', 'Slope', 'Northness', 'EarthEnvTopoMed_Eastness'],
    'disturbance & stand history': ['Burnt Areas Probability', 'disturbance_value', 'Stand Age'],
    'land cover':      ['Land Cover'],
    'mean npp':        ['mean npp'],
}

df_t = pred_df[pred_df["target"] == "std npp"]
df_d = pred_df[pred_df["target"].isin(diversity_vars)]
y = fd_df[['PID', 'std npp']]

results = {}
bench_rows = []                                                            # NEW

l_hat = df_t[['PID', 'y_pred']].rename(columns={'y_pred': 'l_hat'})

for diversity in diversity_vars:
    d = fd_df[['PID', diversity]]
    m_hat = (df_d.loc[df_d['target'] == diversity, ['PID', 'y_pred']]
                 .rename(columns={'y_pred': 'm_hat'}))

    big_df = (y.merge(d, on='PID', how='inner')
               .merge(l_hat, on='PID', how='inner')
               .merge(m_hat, on='PID', how='inner')
               .dropna()
               .merge(fd_df[['PID'] + env_cols], on='PID', how='left')   # NEW
               .reset_index(drop=True))                                   # NEW

    assert big_df['PID'].is_unique, f"duplicate PIDs after merge for {diversity}"

    results[diversity] = dml_sensitivity(
        big_df['std npp'],
        big_df[diversity],
        big_df['l_hat'],
        big_df['m_hat'],
    )

    # ---------------- benchmarking ----------------
    yv, dv = big_df['std npp'], big_df[diversity]
    X = fd_df[env_cols]

    folds = list(KFold(5, shuffle=True, random_state=42).split(big_df))
    l_long, m_long = crossfit(X, yv, folds), crossfit(X, dv, folds)

    for name, cols in benchmark_groups.items():
        X_short = X.drop(columns=cols)
        b = benchmark(yv, dv, l_long, m_long,
                      crossfit(X_short, yv, folds), crossfit(X_short, dv, folds))
        s = dml_sensitivity(yv, dv, big_df['l_hat'], big_df['m_hat'],
                            cf_y=float(np.clip(b['cf_y'], 0, 0.99)),
                            cf_d=float(np.clip(b['cf_d'], 0, 0.99)), rho=1.0)
        bench_rows.append({'diversity': diversity, 'benchmark': name, **b,
                           **{k: s[k] for k in ['theta_lower', 'theta_upper',
                                                'ci_lower', 'ci_upper']}})
    # ---------------------------------------------------

results_df = pd.DataFrame(results).T
print(results_df[['theta', 'se', 'theta_lower', 'theta_upper', 'ci_lower', 'ci_upper', 'rv', 'rva']])

results_df.to_csv("results/ovb_sensitivity_results.csv", index=True)

bench_df = pd.DataFrame(bench_rows).set_index(['diversity', 'benchmark'])  # NEW

bench_df.to_csv("results/ovb_sensitivity_benchmark.csv", index=True)
print(bench_df.round(5))                                                   # NEW