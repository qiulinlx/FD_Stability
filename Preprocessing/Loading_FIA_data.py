import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import pandas as pd

"""
Load and merge Functional Trait data with FIA data (Currently just for location)"""

def load_arrow(filename: str) -> pd.DataFrame:
    with pa.memory_map(filename, "r") as source:
        reader = ipc.RecordBatchFileReader(source)
        table = reader.read_all()
        table= table.to_pandas()
    return table

table = load_arrow("data/raw/FIA_CA.arrow")

table=table[["PID", "accepted_bin"]]

pivot = pd.crosstab(table["PID"], table["accepted_bin"])

# pivot = pivot.sample(n=20, replace=False, random_state=42)  # For Generating subsample 

pivot.to_csv("tree_locations.csv")

