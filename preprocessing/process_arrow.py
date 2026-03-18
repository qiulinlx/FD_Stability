import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import os 
import numpy as np
import warnings

warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

def load_arrow(filename: str) -> pd.DataFrame:
    with pa.memory_map(filename, "r") as source:
        reader = ipc.RecordBatchFileReader(source)
        table = reader.read_all()
        table= table.to_pandas()
    return table


if __name__ == "__main__":

    folder= "data/FIA_states"
    PID_locations = 'data/lookup/PID_location_all.csv'
    PID_abundances = 'data/processed/species_abundance_all.csv'


    df= pd.DataFrame()

    for file in os.listdir(folder):
        if file.endswith(".arrow"):
            filepath= os.path.join(folder, file)
            
            try:
                reader = ipc.RecordBatchFileReader(filepath)
                state_df= load_arrow(filepath)

            except pa.ArrowInvalid:
                state_df = pd.read_feather(filepath)
        df = pd.concat([df, state_df], axis=0, ignore_index=True)

    df1=df[['PID', "accepted_bin"]].copy()
    df1.dropna(inplace=True)

    df2= df[['PID', 'LAT', 'LON', 'BHAGE','managed', 'ownership', 'biome']].copy()
    df2.dropna(subset=['LAT', 'LON'], inplace=True)

    df2 = df2.rename(columns={
    'LAT': 'lat',
    'LON': 'lon'})

    df1.to_csv(PID_abundances, index=False)
    df2.to_csv(PID_locations, index=False)