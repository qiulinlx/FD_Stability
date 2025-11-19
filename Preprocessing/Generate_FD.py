import os
import numpy as np
import pandas as pd
import utils.fdiv as fd

from utils.abundance_utils import Relative_Abundance
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

df=pd.read_csv("example/bird/bird_traits.csv")
traits=pd.read_csv("example/bird/bird_location.csv")


df = df.set_index("PID")

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

df.to_csv("test.csv")