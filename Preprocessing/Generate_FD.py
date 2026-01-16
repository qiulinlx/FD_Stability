import os
import numpy as np
import pandas as pd
import utils.fdiv as fd
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)


def generate_functional_diversity_metrics(df, traits):
    species= df.columns.tolist()

    traits = traits[traits["Species"].isin(species)]

    cols_to_log = traits.columns.difference(['Species'])
    traits[cols_to_log] = traits[cols_to_log].apply(np.log)  #Apply some sort of normalisation transformation

    #FRich = fd.Functional_Richness(df, traits)
    FEve_df=fd.Functional_Evenness(df, traits, Relative_abundance=True)
    FDis = fd.Functional_Dispersion(df, traits)
    FDiv = fd.Functional_Divergence(df, traits)
    RQ_df = fd.Raos_Q(df, traits)

    df =RQ_df.merge(FEve_df, on="PID").merge(FDis, on="PID").merge(FDiv, on="PID")
    return df

