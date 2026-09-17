import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import os
import shutil
import warnings
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from preprocessing.process_arrow import load_arrow
from utils.data_utils import merge_files

warnings.filterwarnings("ignore")

'''
RUN PROCESS ARROW FIRST

This step pulls plot-level tree data from the Composite/FIA arrow
files and computes RARITY/DOMINANCE metrics only (percentage of rare,
subordinate, and dominant species per plot). It does NOT compute
functional or species diversity metrics -- those are handled
elsewhere in the pipeline.

This version uses RELATIVE ABUNDANCE (relative BA per plot) instead of
raw BA_per_acre, so each species' value represents its share of total
plot-level basal area rather than an absolute quantity.

RARITY DEFINITION:
Within each plot (PID), a species present (relative BA > 0) is
classified by its ABSOLUTE SHARE of that plot's total basal area:
  - "rare"        if relative BA < RARE_THRESHOLD (default 10%)
  - "dominant"    if relative BA >= DOMINANT_THRESHOLD (default 80%)
  - "subordinate" otherwise
This is a direct comparison against fixed thresholds, not a per-plot
rank/quantile -- so at most one species per plot can be "dominant"
under the default 80% threshold, and most plots will have none.
'''

RARE_THRESHOLD = 0.10       # species with relative BA below this share of plot total -> rare
DOMINANT_THRESHOLD = 0.80   # species with relative BA at or above this share of plot total -> dominant


def classify_rarity_dominance(pivot, rare_threshold=RARE_THRESHOLD, dominant_threshold=DOMINANT_THRESHOLD):
    """
    Classify species at each site (PID) as rare, subordinate, or dominant
    based on their relative basal area (share of that PLOT's total BA):
      - "rare"        if present but relative BA < rare_threshold
      - "dominant"    if relative BA >= dominant_threshold
      - "subordinate" otherwise (present, but neither rare nor dominant)

    This is an ABSOLUTE-SHARE definition, not a rank/quantile one: a
    species is "dominant" only if it actually holds >= dominant_threshold
    of the plot's total basal area on its own. Since relative BA values
    within a plot sum to 1, at most one species per plot can ever be
    "dominant" under the default threshold of 0.80 (two species both
    >=80% would sum to over 100%), and most plots will have zero.
    """
    presence = pivot > 0

    rare_mask = presence & pivot.lt(rare_threshold)
    dominant_mask = presence & pivot.ge(dominant_threshold)
    subordinate_mask = presence & ~rare_mask & ~dominant_mask

    n_species_total = presence.sum(axis=1)
    n_rare = rare_mask.sum(axis=1)
    n_dominant = dominant_mask.sum(axis=1)
    n_subordinate = subordinate_mask.sum(axis=1)

    safe_denom = n_species_total.replace(0, np.nan)

    rarity_df = pd.DataFrame({
        'PID': pivot.index,
        'n_species_total': n_species_total.values,
        'n_rare_species': n_rare.values,
        'n_subordinate_species': n_subordinate.values,
        'n_dominant_species': n_dominant.values,
        'pct_rare_species': (n_rare / safe_denom).values,
        'pct_subordinate_species': (n_subordinate / safe_denom).values,
        'pct_dominant_species': (n_dominant / safe_denom).values,
    })

    return rarity_df


if __name__ == "__main__":

    # Clear any files left over from a previous run BEFORE recreating the
    # directory -- otherwise a run over a different set of .arrow files
    # (fewer states, a renamed file, a changed filter) leaves stale
    # datasetN.csv files behind that silently get swept into merge_files()
    # below alongside the new ones.
    shutil.rmtree("data/.joined", ignore_errors=True)
    os.makedirs("data/.joined", exist_ok=True)

    PID_df = pd.read_csv('data/lookup/PID_location_all.csv')
    PID_df["managed"] = PID_df["managed"].fillna(-1)
    PID_df["ownership"] = PID_df["ownership"].fillna("No Data")
    PID_df["biome"] = PID_df["biome"].fillna("No Data")
    PID_df.drop_duplicates(subset=['PID'], inplace=True)

    # Preparing rarity/dominance data ------------------------------------------------
    folder = "data/FIA_states"
    i = 1
    for file in os.listdir(folder):
        if file.endswith(".arrow"):
            filepath = os.path.join(folder, file)

            try:
                table = load_arrow(filepath)
            except pa.ArrowInvalid:
                table = pd.read_feather(filepath)

            print(f"Processing {file}")

            table = table[table['lastYear'] == True]
            table = table[table['DESIGNCD'].isin([1, 111, 112, 113, 116, 117, 311, 312, 501, 502, 503, 504, 505, 506])]
            table = table[table['status'].isin(['live'])]
            table = table[table['cdMult'].isin([0.0])]
            table['BA_per_acre'] = 0.005454 * (table['DIA'] ** 2) * table['TPA_UNADJ']

            pivot = table.pivot_table(
                index='PID',
                columns='accepted_bin',
                values='BA_per_acre',
                aggfunc='sum',
                fill_value=0
            )

            # --- Convert to relative abundance -----------------------------------
            # Each row (PID) is normalized to sum to 1, so each species' value is
            # its share of total plot-level basal area rather than an absolute BA.
            row_totals = pivot.sum(axis=1)

            # Guard against any plots with zero total BA (avoid divide-by-zero)
            nonzero_mask = row_totals > 0
            if (~nonzero_mask).any():
                n_dropped = (~nonzero_mask).sum()
                print(f"Dropping {n_dropped} plots with zero total BA")
                pivot = pivot.loc[nonzero_mask]
                row_totals = row_totals.loc[nonzero_mask]

            pivot = pivot.div(row_totals, axis=0)
            # -----------------------------------------------------------------------

            rarity_df = classify_rarity_dominance(pivot, rare_threshold=RARE_THRESHOLD, dominant_threshold=DOMINANT_THRESHOLD)

            rarity_df.to_csv(f'data/.joined/dataset{i}.csv', index=False)
            i += 1

    csv_dir = Path("data/.joined")
    out_file = "data/final/dataset_relba_10.parquet"
    merge_files(csv_dir, out_file)