import pandas as pd
import numpy as np

from scipy.spatial.distance import pdist, squareform

from sklearn.preprocessing import StandardScaler, MinMaxScaler


# --- Abundance calculation functions ---


def calculate_relative_abundance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute relative abundances (%) for a given abundance column.

    Args:
        df: pandas DataFrame with species abundances
            each contains the absolute abundances

    Returns:
        df: new dataframe containing relative abundances
    """
    row_sum = df.sum(axis=1)
    df_relative_frequency = df.div(row_sum, axis=0)

    return df_relative_frequency


def normalize_abundance(abundances: list) -> np.ndarray:
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


# --- Trait standardization ---


def standardize_trait_matrix_z_score(trait_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize the trait matrix to have mean 0 and variance 1 for each trait.

    Args:
        trait_matrix (pd.DataFrame): DataFrame with species as rows and traits as columns.

    Returns:
        pd.DataFrame: Standardized trait matrix.
    """

    scaler = StandardScaler()
    standardized_traits = pd.DataFrame(
        scaler.fit_transform(trait_matrix),
        index=trait_matrix.index,
        columns=trait_matrix.columns,
    )

    return standardized_traits


def standardize_trait_matrix_min_max(trait_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize the trait matrix to have values between 0 and 1 for each trait.

    Args:
        trait_matrix (pd.DataFrame): DataFrame with species as rows and traits as columns.

    Returns:
        pd.DataFrame: Standardized trait matrix.
    """

    scaler = MinMaxScaler()
    standardized_traits = pd.DataFrame(
        scaler.fit_transform(trait_matrix),
        index=trait_matrix.index,
        columns=trait_matrix.columns,
    )

    return standardized_traits


def standardize_trait_matrix(
    trait_matrix: pd.DataFrame, method: str = None
) -> pd.DataFrame:
    """
    Wrapper function to standardize the trait matrix using the specified method.

    Args:
        trait_matrix (pd.DataFrame): DataFrame with species as rows and traits as columns.
        method (str, default=None): Method for standardization.
            - "z_score": standardize traits to mean=0 and var=1.
            - "min_max": standardize traits to range [0, 1].
            - None: do not standardize traits.

    Returns:
        pd.DataFrame: Standardized trait matrix.
    """
    if method is None:
        return trait_matrix.copy()

    if method == "z_score":
        return standardize_trait_matrix_z_score(trait_matrix)
    elif method == "min_max":
        return standardize_trait_matrix_min_max(trait_matrix)
    else:
        raise ValueError("Invalid method. Choose 'z_score', 'min_max' or None.")


# --- Distance calculation functions ---


def euclidean_distance(
    traits: pd.DataFrame, metric: str = "euclidean", standardize_method: str = None
) -> pd.DataFrame:
    """Compute the pairwise distance matrix for a given trait matrix.

    Args:
        traits (pd.DataFrame): DataFrame with species as rows and traits as columns
        metric (str, default="euclidean"): distance metric to use
        standardize_method (str, default=None): Method for standardizing traits before computing distances

    Returns:
        pd.DataFrame: pairwise (Species x Species) distance matrix
    """

    traits = standardize_trait_matrix(traits, method=standardize_method)

    return pd.DataFrame(
        squareform(pdist(traits, metric=metric)),
        index=traits.index,
        columns=traits.index,
    )
