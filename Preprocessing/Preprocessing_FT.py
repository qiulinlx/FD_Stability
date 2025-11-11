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
ft_df= pd.read_csv("data/lookup/traitMatrix.csv")
ft_df.columns.values[0] = "species"

table = table[table["firstYear"] == True]


'''Dropping unneeded columns that have null or nonunique values'''

cols_to_drop = [
    "PLT_CN.y",
    "STATE_ABBRV",
    "STDAGE",
    "FORINDCD",
    "TRTCD1_P2A",
    "BHAGE",
    "TOTAGE",
    "MORTCD",
    "CPOSCD",
    "CLIGHTCD",
    "AGENTCD",
    "STATECD",
    "MACRO_BREAKPOINT_DIA",
    "hasPlot",
    "goodDesign",
    "USE",
    "STATE", 
    "llId",
    "STANDING_DEAD_CD",
    "damage", 
    "PREV_TREEID", 
    "REMPER"
    
]

regulatory_to_drop = [
"UNITCD",
"COUNTYCD",
"PLOT",
"PLT_CN.x",
"SPCD",
"SUBP"
]

eco_vars = [
    'PID',
    "LON", 
    "LAT",
    "accepted_bin"       
]

# eco_vars = [
#     'PID',
#     "LON", 
#     "LAT",
#     "CARBON_AG",       # Aboveground carbon
#     "biomass",         # Total biomass
#     "VOLCFGRS",        # Volume of growing stock
#     "sizeClass",       # Structural diversity (size class)
#     "canopy_pos",      # Vertical canopy position
#     "biome",           # Biome classification 
#     "FORTYPCD",        # Forest type (composition)
#     "height",             # Tree diameter
#     "managed",
#     "accepted_bin"       
# ]

df = table.drop(columns=cols_to_drop)

df = df.drop(columns=regulatory_to_drop)

eco_df = df[eco_vars]

merged_df = pd.merge(eco_df, ft_df, left_on="accepted_bin", right_on="species")
merged_df.to_csv("CA_functional_traits.csv", index=False)