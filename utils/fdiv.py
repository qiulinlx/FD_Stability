import numpy as np
import pandas as pd

from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist, squareform

from sklearn.preprocessing import StandardScaler

import networkx as nx
from abundance_utils import Relative_Abundance, normalise_abundance


from scipy.spatial import Delaunay


"""
TODO: 
Add argument:
Euclidean vs Gower
Abundance Weighting for Centroids and more


Functional richness (FRic),
Functional volume intersections (FRic_intersect),
Functional divergence (FDiv),
Functional evenness (FEve),
Functional dispersion (FDis)

Rao's Quadratic Entropy (RaoQ)

"""


def Functional_Richness(traits: np.ndarray) -> float:
    """
    Compute Functional Richness (FRic) as the volume of the convex hull
    in standardized trait space.

    Args:
        traits: np.ndarray of shape (S, T) where S = species, T = traits

    Returns:
        FRic (float) or None if not enough species to form a convex hull
    """
    # Standardize traits (mean=0, std=1)
    traits_scaled = StandardScaler().fit_transform(traits)

    n_species, n_traits = traits_scaled.shape
    if n_species <= n_traits:
        print("Not enough species to form a convex hull.")
        return None

    hull = ConvexHull(traits_scaled)
    FRic = hull.volume
    return FRic, hull


def Functional_Evenness(trait_array: np.ndarray, rel_ab: list) -> float:
    """
    Compute Functional Evenness (FEve) using the MST approach (Villéger et al., 2008).

    Args:
        trait_array: np.ndarray of shape (S, T)
        rel_ab: list or array of relative abundances (same length as S)

    Returns:
        FEve (float)
    """
    dist_matrix = squareform(pdist(trait_array, metric='euclidean'))
    rel_ab = np.array(rel_ab)
    S = len(rel_ab)

    # Construct graph and compute Minimum Spanning Tree
    G = nx.from_numpy_array(dist_matrix)
    mst = nx.minimum_spanning_tree(G)
    edges = list(mst.edges(data=True))

    # Compute weighted branch lengths
    EW_list = []
    for i, j, data in edges:
        EW = data['weight'] / (rel_ab[i] + rel_ab[j])
        EW_list.append(EW)

    EW_array = np.array(EW_list)
    PEW = EW_array / np.sum(EW_array)

    # Functional Evenness formula
    FEve = np.sum(np.minimum(PEW, 1 / (S - 1))) / (1 - 1 / (S - 1))
    return FEve


def Functional_Divergence(trait_array: np.ndarray, rel_ab: list) -> float:
    """
    Compute Functional Divergence (FDiv).

    Args:
        trait_array: np.ndarray of shape (S, T)
        abundances: list or array of relative abundances (should sum to 1)

    Returns:
        FDiv (float)
    """
    
    rel_ab = normalise_abundance(rel_ab)

    # Compute community centroid
    centroid = np.mean(trait_array, axis=0) 
    
    # Distances to centroid
    distances = np.linalg.norm(trait_array - centroid, axis=1)
    dG= np.mean(distances)


    delta_d= np.sum(rel_ab - (distances - dG))
    abs_delta_d = np.sum(rel_ab - np.abs(distances - dG))

    FDiv = (delta_d + dG) / (abs_delta_d + dG)
    
    return FDiv


def functional_dispersion(traits: np.ndarray, abundances: list, weighted: bool = True) -> float:
    """
    Compute Functional Dispersion (FDis) for a community.

    FDis measures the spread of species in trait space.
    Can be computed as abundance-weighted or unweighted.

    Args:
        traits (np.ndarray): Trait matrix of shape (S, T), where S = species, T = traits.
        abundances (list or np.ndarray): Species abundances (any scale; will be normalized if weighted).
        weighted (bool): If True, compute abundance-weighted FDis. If False, compute unweighted FDis.

    Returns:
        float: Functional Dispersion (FDis)
    """
    traits = np.array(traits)
    
    if weighted:
        abundances = np.array(abundances)
        abundances = abundances / abundances.sum()  # Normalize relative abundances
        centroid = np.sum(traits * abundances[:, None], axis=0)  # Abundance-weighted centroid
    else:
        centroid = np.mean(traits, axis=0)  # Unweighted centroid

    # Distances from centroid
    distances = np.linalg.norm(traits - centroid, axis=1)

    # Compute FDis
    FDis = np.sum(distances * abundances) if weighted else np.mean(distances)




def Raos_Q(trait_array: np.ndarray, rel_ab: list) -> float:
    """
    Compute Rao's Quadratic Entropy (RaoQ) from a trait distance matrix.

    Args:
        trait_array: np.ndarray of shape (S, T)
        rel_ab: list or array of relative abundances (should sum to 1)

    Returns:
        RaoQ (float)
    """
    dist_matrix = squareform(pdist(trait_array, metric='euclidean'))
    
    rel_ab = normalise_abundance(rel_ab)

    # Compute abundance weight matrix
    weight_matrix = np.outer(rel_ab, rel_ab)

    # Optional: set diagonal to zero
    np.fill_diagonal(weight_matrix, 0)

    # Rao's Q = sum(p_i * p_j * d_ij)
    RaoQ = np.sum(weight_matrix * dist_matrix)
    return RaoQ
