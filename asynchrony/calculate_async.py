"""
Asynchrony per FIA plot, using the same files and filters as the functional diversity script.

Differences from the functional diversity script (needed for a time series):
    - no `lastYear` filter: every visit of each plot is kept
    - no `PREV_PLT_CN` filter: keeping it would drop each plot's first visit
    - pivot index is (PID, INVYR) instead of PID

Outputs one row per PID:
    phi_LdM, asynchrony_LdM (Loreau & de Mazancourt 2008), eta_gross (Gross et al. 2014),
    cv_community, cv_species, log_stab_community, log_stab_species, async_term
    (log_stab_community = log_stab_species + async_term)
"""
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

sys.path.append(str(Path(__file__).resolve().parents[1]))
from preprocessing.process_arrow import load_arrow

# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------
FOLDER = "data/FIA_states"
OUTPUT_PATH = "data/processed/asynchrony_per_plot.csv"
DESIGN_CODES = [1, 111, 112, 113, 116, 117, 311, 312, 501, 502, 503, 504, 505, 506]

PID_COL, YEAR_COL, SP_COL, VAL_COL = "PID", "INVYR", "accepted_bin", "BA_per_acre"
MIN_POINTS = 4            # minimum number of visits per plot
DETREND = True            # remove a linear trend per species ...
DETREND_MIN_POINTS = 5    # ... only when there are at least this many visits


# ---------------------------------------------------------------
# Data loading and filtering (same as the functional diversity script)
# ---------------------------------------------------------------
def load_state(filepath):
    try:
        return load_arrow(filepath)
    except pa.ArrowInvalid:
        return pd.read_feather(filepath)


def prepare(table):
    table = table[table["DESIGNCD"].isin(DESIGN_CODES)]
    table = table[table["status"].isin(["live"])]
    table = table[table["cdMult"].isin([0.0])].copy()
    table["BA_per_acre"] = 0.005454 * (table["DIA"] ** 2) * table["TPA_UNADJ"]
    table[YEAR_COL] = pd.to_numeric(table[YEAR_COL], errors="coerce")
    return table.dropna(subset=[YEAR_COL])


# ---------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------
def plot_metrics(X, t):
    """X: visits x species array; t: inventory year of each visit."""
    X = X[:, np.any(X != 0, axis=0)]
    n_t, n_sp = X.shape
    out = {"n_species": n_sp, "n_years": n_t}
    if n_t < MIN_POINTS or n_sp == 0:
        return out

    means = X.mean(axis=0)
    total_mean = means.sum()
    R = X - means
    if DETREND and n_t >= DETREND_MIN_POINTS:
        tc = t - t.mean()
        R = R - np.outer(tc, (tc @ R) / (tc @ tc))

    cov = (R.T @ R) / (n_t - 1)
    sd_i = np.sqrt(np.clip(np.diag(cov), 0, None))
    sd_total = np.sqrt(max(cov.sum(), 0.0))
    phi = (sd_total / sd_i.sum()) ** 2 if sd_i.sum() > 0 else np.nan

    total_R = R.sum(axis=1)
    etas = [np.corrcoef(R[:, i], total_R - R[:, i])[0, 1]
            for i in range(n_sp)
            if n_sp > 1 and sd_i[i] > 0 and (total_R - R[:, i]).std() > 0]

    out.update({
        "phi_LdM": phi,
        "asynchrony_LdM": 1 - phi if np.isfinite(phi) else np.nan,
        "eta_gross": float(np.mean(etas)) if etas else np.nan,
    })
    if total_mean > 0 and sd_total > 0 and np.isfinite(phi) and phi > 0:
        cv_comm, cv_sp = sd_total / total_mean, sd_i.sum() / total_mean
        out.update({
            "cv_community": cv_comm,
            "cv_species": cv_sp,
            "log_stab_community": -np.log(cv_comm),
            "log_stab_species": -np.log(cv_sp),
            "async_term": -0.5 * np.log(phi),
        })
    return out


def asynchrony_for_table(table):
    pivot = table.pivot_table(
        index=[PID_COL, YEAR_COL],
        columns=SP_COL,
        values=VAL_COL,
        aggfunc="sum",
        fill_value=0,
    )
    rows = []
    for pid, block in pivot.groupby(level=PID_COL):
        block = block.droplevel(PID_COL).sort_index()
        m = plot_metrics(block.to_numpy(dtype=float), block.index.to_numpy(dtype=float))
        m[PID_COL] = pid
        rows.append(m)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------
# Run
# ---------------------------------------------------------------
if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    results = []
    for file in sorted(os.listdir(FOLDER)):
        if not file.endswith(".arrow"):
            continue
        print(f"Processing {file}")
        table = prepare(load_state(os.path.join(FOLDER, file)))
        if table.empty:
            continue
        res = asynchrony_for_table(table)
        res["source_file"] = file
        results.append(res)

    asynchrony = pd.concat(results, ignore_index=True)
    asynchrony = asynchrony[[PID_COL] + [c for c in asynchrony.columns if c != PID_COL]]
    asynchrony.dropna(subset=['asynchrony_LdM'], inplace=True)

    print("\nPlots by number of visits:")
    print(asynchrony["n_years"].value_counts().sort_index().to_string())
    print(f"Plots with metrics (>= {MIN_POINTS} visits): "
          f"{asynchrony['phi_LdM'].notna().sum()} of {len(asynchrony)}")

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    asynchrony.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved to {OUTPUT_PATH}")