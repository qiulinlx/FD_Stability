import os
import numpy as np
import pandas as pd
import utils.fdiv as fd
import warnings
import utils.sdiv as sd

warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

def generate_functional_diversity_metrics(df:pd.DataFrame, traits:pd.DataFrame)->pd.DataFrame:
    species= df.columns.tolist()

    traits = traits[traits["Species"].isin(species)]

    cols_to_log = traits.columns.difference(['Species'])
    traits[cols_to_log] = traits[cols_to_log].apply(np.log)  #Apply some sort of normalisation transformation

    Fmpd = fd.MPD(df, traits)
    FEve_df=fd.Functional_Evenness(df, traits, Relative_abundance=True)
    FDis = fd.Functional_Dispersion(df, traits)
    FDiv = fd.Functional_Divergence(df, traits)
    RQ_df = fd.Raos_Q(df, traits)

    df =RQ_df.merge(FEve_df, on="PID").merge(FDis, on="PID").merge(FDiv, on="PID").merge(Fmpd, on="PID")
    return df


def generate_species_diversity_metrics(df):

    sp_df=sd.species_richness(df)
    shannon_df=sd.shannon_diversity(df)
    simpsons_df=sd.simpsons_index(df)
    seq_df=sd.shannon_equitability(df)

    df = sp_df.merge(shannon_df, on="PID").merge(simpsons_df, on="PID").merge(seq_df, on="PID")
    return df

