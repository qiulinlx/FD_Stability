import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import os 
from scipy.stats import zscore
import numpy as np
import warnings

def load_arrow(filename: str) -> pd.DataFrame:
    with pa.memory_map(filename, "r") as source:
        reader = ipc.RecordBatchFileReader(source)
        table = reader.read_all()
        table= table.to_pandas()
    return table

df_trait=pd.read_csv('data/rueda_traits_cleaned.csv')
df=pd.read_csv('data/processed/species_abundance_all.csv')

df['count'] = 0.005454 * (df['DIA'] ** 2) * df['TPA_UNADJ']
df.dropna(subset=['count'], inplace=True)

long_df=df

# 2. Drop absences
long_df = long_df[long_df['count'] > 0]

# 3. Select trait columns to use (exclude identifier/text columns)
exclude_cols = ['accepted_bin', 'family', 'species']
trait_cols = [c for c in df_trait.columns if c not in exclude_cols]

# 4. Merge in trait values on accepted_bin
merged = long_df.merge(
    df_trait[['accepted_bin'] + trait_cols],
    on='accepted_bin',
    how='left'
)

trait_cols.remove('dispersal')
trait_cols.remove('leaf_cn')

merged[trait_cols] = merged[trait_cols].apply(zscore, nan_policy='omit')

# 5. Unweighted variance per plot
variance_per_pid = merged.groupby('PID')[trait_cols].var(skipna=True)


trait_variance_index = variance_per_pid.mean(axis=1, skipna=True)

trait_variance_index.to_csv('data/processed/syndrome_varianece_per_pid1.csv', index=True)