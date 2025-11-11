
import os
import re
import warnings

import pandas as pd
import numpy as np

from utils.utils import truncate_after_n_underscores
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

"""
Script to process FIA ecological trait data:
1. Reads CSV of traits.
2. Truncates plot IDs to the first four underscore-separated parts.
3. Drops unnecessary columns.
4. Splits the dataset into sub-datasets by PID.
5. Writes each sub-dataset to a Parquet file with safe filenames.
"""

df = pd.read_csv("data/processed/CA_functional_traits.csv")

df["PID"] = df["PID"].apply(truncate_after_n_underscores)

df.drop(columns=['LON', 'LAT', 'accepted_bin'], inplace=True)

cols_to_log = df.columns.difference(['PID', 'species'])
df[cols_to_log] = df[cols_to_log].apply(np.log)

groups = {name: g for name, g in df.groupby("PID")}

# Create output folder for sub-datasets
out_dir = "county_data"
os.makedirs(out_dir, exist_ok=True)

# Save each PID group as a separate Parquet file
for name, g in df.groupby("PID"):
    # Make a filename safe by replacing non-alphanumeric characters
    safe_name = re.sub(r"[^A-Za-z0-9_\-]", "_", str(name))

    path = os.path.join(out_dir, f"{safe_name}.parquet")
    g.to_parquet(path, index=False)
