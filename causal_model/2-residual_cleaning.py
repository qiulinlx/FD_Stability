import pandas as pd
from functools import reduce

res_df = pd.read_csv('results/step_1_residuals.csv')
res_df['residual'] = res_df['y_actual'] - res_df['y_pred']

diversity_vars = ["Species Richness", "Shannon Diversity", "Raos_Q", "Simpson's Index", "Functional_Evenness"]

# map target name -> desired residual column name
target_map = {
    "std npp": "residual_std_npp",
    "mean npp": "residual_mean",
    "WSCI": "residual_wsci",
    "Species Richness": "residual_species_richness",
    "Shannon Diversity": "residual_shannon",
    "Simpson's Index": "residual_simpsons",
    "Raos_Q": "residual_raos_q",
    "Functional_Evenness": "residual_functional_evenness",
}

results = {}
for target, col_name in target_map.items():
    sub = res_df[res_df['target'] == target].copy()
    sub[col_name] = sub['residual']
    results[target] = sub

# unpack if you want to keep your old variable names
env_sd_results = results["std npp"]
env_mean_results = results["mean npp"]
wsci_results = results["WSCI"]
species_richness_results = results["Species Richness"]
shannon_results = results["Shannon Diversity"]
simpsons_results = results["Simpson's Index"]
raos_q_results = results["Raos_Q"]
functional_evenness_results = results["Functional_Evenness"]

# keep only PID and the target-specific residual column (no seed anymore — gridded CV)
dfs_to_merge = []
for target, col_name in target_map.items():
    sub = results[target][['PID', col_name]]
    dfs_to_merge.append(sub)

merged = reduce(
    lambda left, right: pd.merge(left, right, on=['PID'], how='outer'),
    dfs_to_merge
)

merged.to_csv('results/step_2_residuals_merged2.csv', index=False)