import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import os 
import warnings
from sklearn.preprocessing import StandardScaler
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from preprocessing.process_arrow import load_arrow
from pathlib import Path
import utils.generate_metrics as gm
from utils.data_utils import merge_files
import warnings

warnings.filterwarnings("ignore")

'''

RUN PROCESS ARROW FIRST
We run this to:
    pull data from the Composite file
    load in the FIA data 
    generate functional and species diversity metrics
    merge them all together into a singular dataset

'''


if __name__ == "__main__":

    os.makedirs("data/.joined", exist_ok=True)

    PID_df=pd.read_csv('data/lookup/PID_location_all.csv')
    traits=pd.read_csv("data/lookup/traitMatrix1.csv")

    cols_to_log = traits.columns.difference(['Species'])
    traits[cols_to_log] = traits[cols_to_log].apply(np.log)
    scaler = StandardScaler()
    traits[cols_to_log] = scaler.fit_transform(traits[cols_to_log])
    PID_df["managed"] = PID_df["managed"].fillna(-1)
    PID_df["ownership"] = PID_df["ownership"].fillna("No Data")
    PID_df["biome"] = PID_df["biome"].fillna("No Data")

    PID_df.drop_duplicates(subset=['PID'], inplace=True)

    species_list = traits['Species'].tolist()

    ba=True
    stem=False

    # Preparing Funcional Diversity Data ------------------------------------------------
    folder= "data/FIA_states"
    i=1
    for file in os.listdir(folder):
        if file.endswith(".arrow"):
            filepath= os.path.join(folder, file)

            try:
                reader = ipc.RecordBatchFileReader(filepath)
                table= load_arrow(filepath)

            except pa.ArrowInvalid:
                table = pd.read_feather(filepath)

        print((f"Processing {file}"))

        table = table[table['lastYear']==True]
        table = table[table['DESIGNCD'].isin([1, 111, 112, 113, 116, 117, 311, 312, 501, 502, 503, 504, 505, 506])]
        table = table[table['status'].isin(['live'])]
        table = table[table['cdMult'].isin([0.0])]
        table['BA_per_acre'] = 0.005454 * (table['DIA'] ** 2) * table['TPA_UNADJ']

        if ba: 
            pivot = table.pivot_table(
                index='PID',
                columns='accepted_bin',
                values='BA_per_acre',
                aggfunc='sum',
                fill_value=0
            )

        if stem: 
            pivot = table.pivot_table(
                index='PID',
                columns='accepted_bin',
                values='TPA_UNADJ',
                aggfunc='sum',
                fill_value=0
            )

        fd_df=gm.generate_functional_diversity_metrics(pivot, traits)

        # Keep only species present in trait table
        missing_species = pivot.columns.difference(species_list)

        print(f"Dropping {len(missing_species)} species missing from traits:")
        print(missing_species.tolist())

        pivot = pivot.drop(columns=missing_species)
        sd_df=gm.generate_species_diversity_metrics(pivot)
        
        # Merging FD and Environmental Data ------------------------------------------------
        total_df = fd_df.merge(sd_df, on='PID', how='inner')
        total_df.to_csv(f'data/.joined/dataset{i}.csv', index=False)
        i+=1

    csv_dir = Path("data/.joined")
    out_file = "data/final/dataset_ba.parquet"
    merge_files(csv_dir, out_file)