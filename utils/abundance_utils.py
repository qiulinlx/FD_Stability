
import numpy as np
import pandas as pd


def Relative_Abundance(df: pd.DataFrame, abundances_col: str) -> pd.DataFrame:
    """
    Compute relative abundances (%) for a given abundance column.

    Args:
        df: pandas DataFrame with species abundances
        abundances_col: column name for absolute abundances

    Returns:
        df: DataFrame with new column "Relative_Abundances"
    """
    total = df[abundances_col].sum()
    df["Relative_Abundances"] = df[abundances_col] / total * 100
    return df

def normalise_abundance(abundances: list) -> np.ndarray:
    """
    Normalize abundances to sum to 1.

    Args:
        abundances: list or array of abundances

    Returns:
        normalized_abundances: np.ndarray of normalized abundances
    """
    abundances = np.array(abundances)
    normalized_abundances = abundances / abundances.sum()
    return normalized_abundances