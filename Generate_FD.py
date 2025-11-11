import os
import numpy as np
import pandas as pd
import utils.fdiv as fd

from utils.abundance_utils import Relative_Abundance

import warnings
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)


# Folder containing all Parquet files
folder = "data/county_data"

pid=[]
F_Eve=[]
F_Div=[]
F_Dis=[]
Rao_Q=[]
# F_Rich=[]

# List all Parquet files
all_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".parquet")]

j=0

# Loop through each file
for i, file_path in enumerate(all_files, 1):
    print(f"Processing file {i}/{len(all_files)}: {file_path}")

    # Load the Parquet file and generate abundance column
    df = pd.read_parquet(file_path)


    df['abundances'] = df.groupby(list(df.columns)).transform('size')

    df.drop_duplicates(inplace=True)
    abundance_df = pd.DataFrame()   
    abundance_df['species'] = df.pop('species')
    abundance_df['abundances'] = df.pop('abundances')

    # Skip files with less than 5 rows
    if len(abundance_df) < 2:
        print(f"Skipping {file_path} (only {len(df)} rows)")
        j+=1
        del df
        continue

    pid.append(df['PID'][0])
    df.drop(columns=['PID'], inplace=True)
    
    # Generate FD metrics
    trait_array=np.array(df)
    abundance_df = Relative_Abundance(abundance_df, "abundances")

    r_ab = abundance_df['Relative_Abundances'].tolist()
    # r, hull = fd.Functional_Richness(trait_array)
    # F_Rich.append(r)
    F_Eve.append( fd.Functional_Evenness(trait_array, r_ab))
    F_Div.append( fd.Functional_Divergence(trait_array, r_ab))
    Rao_Q.append(fd.Raos_Q(trait_array, r_ab))
    F_Dis.append(fd.Functional_Dispersion(trait_array, r_ab))


    del df

# Create DataFrame
FD_df = pd.DataFrame({
    'PID': pid,
    'Functional_Evenness': F_Eve,
    'Functional_Divergence': F_Div,
    'Rao_Quadratic_Entropy': Rao_Q,
    'Functional_Dispersion': F_Dis
})

# FD_df = pd.DataFrame({
#     'PID': pid,
#     'Functional_Richness': F_Rich,
#     'Functional_Evenness': F_Eve,
#     'Functional_Divergence': F_Div,
#     'Rao_Quadratic_Entropy': Rao_Q,
#     'Functional_Dispersion': F_Dis
# })




print(f"{j} PIDs not used")
FD_df.to_csv("FD_Metrics_California.csv", index=False)