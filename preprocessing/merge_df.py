import pandas as pd
from pathlib import Path

'''
TODO:

Make this a function and add it to the main.py for preprocessing

'''

csv_dir = Path("data/joined")
out_file = "dataset.parquet"
dtype_map = {
    "value": "float64"
}
dfs = []
for csv in csv_dir.glob("*.csv"):
    df = pd.read_csv(csv, dtype=dtype_map)
    df["source_file"] = csv.name   # optional but VERY useful
    dfs.append(df)

stacked = pd.concat(dfs, ignore_index=True)

stacked.to_parquet(out_file, engine="pyarrow", compression="snappy")