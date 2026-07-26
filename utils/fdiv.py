import numpy as np
import pandas as pd

from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist, squareform

from sklearn.preprocessing import StandardScaler

import networkx as nx
# from utils.abundance_utils import Relative_Abundance, normalise_abundance


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
def Functional_Richness(sp_loc:pd.DataFrame, traits: np.ndarray) -> pd.DataFrame:
    """
    Compute Functional Richness (FRic) as the volume of the convex hull
    in standardized trait space.

    Args:
        sp_loc: Pivot table of Plot IDs and Species
        traits: np.ndarray of shape (S, T) where S = species, T = traits

    Returns:
        FRic (float) or None if not enough species to form a convex hull
    """
    pID = []
    Frich = []

    species_PID = sp_loc.apply(lambda row: row.index[row != 0].tolist(), axis=1)

    for i, species in enumerate(species_PID):
        pid = species_PID.index[i]
        
        # Select traits for species
        traits_sub = traits[traits["Species"].isin(species)].copy()
        traits_sub.drop(columns=['Species'], inplace=True)

        n_species, n_traits = np.array(traits_sub).shape

        if n_species <= n_traits  or n_species < 3:
            # FEve undefined for <2 species
            Frich.append(np.nan)
            pID.append(pid)
            
            continue
            continue  # will automatically go to the next iteration
        # Scale traits
        traits_scaled = StandardScaler().fit_transform(traits_sub)

        n_species, n_traits = traits_scaled.shape

        hull = ConvexHull(traits_scaled)
        FRic = hull.volume
        Frich.append(FRic)
        pID.append(pid)

    
    FRich_df= pd.DataFrame({"PID": pID, "Functional_Richness": Frich})

    return FRich_df


def Frich_Intersect(hull1, hull2, n_samples: int = 100000) -> float:
    """
    GPT GENERATED REQUIRES MORE VERIFICATION
    Compute approximate Functional Volume Intersection (FRic_intersect) 
    between two communities using convex hulls.

    Args:
        hull1, hull2: scipy.spatial.ConvexHull objects for each community
        n_samples: number of Monte Carlo samples for approximation

    Returns:
        FRic_intersect: float between 0 and 1
    """
    # Combine points to get bounding box
    all_points = np.vstack([hull1.points, hull2.points])
    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)

    # Generate random points in bounding box
    samples = np.random.uniform(mins, maxs, size=(n_samples, hull1.points.shape[1]))

    # Helper: check if points are inside a convex hull
    def in_hull(points, hull):
        delaunay = Delaunay(hull.points[hull.vertices])
        return delaunay.find_simplex(points) >= 0

    inside1 = in_hull(samples, hull1)
    inside2 = in_hull(samples, hull2)

    # Intersection fraction
    intersection_fraction = np.sum(inside1 & inside2) / n_samples
    union_fraction = np.sum(inside1 | inside2) / n_samples

    FRic_intersect = intersection_fraction / union_fraction if union_fraction > 0 else 0
    return FRic_intersect

def Functional_Evenness(sp_loc: pd.DataFrame, traits: pd.DataFrame, Relative_abundance: bool = False) -> pd.DataFrame:
    """
    Compute Functional Evenness (FEve) using the MST approach (Villéger et al., 2008)
    for presence/absence or relative abundance data.
    Args:
        sp_loc: Pivot table of Plot IDs and Species
        traits: dataframe of functional traits (rows=species, columns=traits), must have column "Species"
        Relative_abundance: If True, uses relative abundance weighting
    Returns:
        FEve_df: DataFrame with PID and Functional Evenness
    """
    pID = []
    FEven = []

    # Get species present in each PID
    species_PID = sp_loc.apply(lambda row: row.index[row != 0].tolist(), axis=1)

    for pid, species in zip(species_PID.index, species_PID):

        # Subset, align traits to species order, drop species with missing traits
        traits_sub = traits[traits["Species"].isin(species)].copy()
        traits_sub = traits_sub.set_index("Species").reindex(species).dropna()
        species = traits_sub.index.tolist()
        S = len(species)

        if S < 2:
            FEven.append(np.nan)
            pID.append(pid)
            continue

        if S == 2:
            if Relative_abundance:
                ab = sp_loc.loc[[pid], species]
                total = ab.sum(axis=1)
                if (total == 0).any():
                    raise ValueError(f"Plot {pid} has zero total abundance")
                ab = ab.div(total, axis=0).values.flatten().astype(float)
                FEven.append(1 - np.abs(ab[0] - ab[1]))
            else:
                FEven.append(1.0)
            pID.append(pid)
            continue

        # Distance matrix (traits_sub already indexed by Species)
        dist_matrix = squareform(pdist(traits_sub.values, metric='euclidean'))

        # Minimum Spanning Tree
        G = nx.from_numpy_array(dist_matrix)
        mst = nx.minimum_spanning_tree(G)

        # Weighted branch lengths
        EW_list = []
        if Relative_abundance:
            ab = sp_loc.loc[[pid], species]  # enforce species order
            total = ab.sum(axis=1)
            if (total == 0).any():
                raise ValueError(f"Plot {pid} has zero total abundance")
            ab = ab.div(total, axis=0).values.flatten().astype(float)

            for u, v, data in mst.edges(data=True):
                EW_list.append(data['weight'] / (ab[u] + ab[v]))
        else:
            for u, v, data in mst.edges(data=True):
                EW_list.append(data['weight'] / 2)

        EW_array = np.array(EW_list)
        PEW = EW_array / EW_array.sum()
        FEve_val = (np.sum(np.minimum(PEW, 1 / (S - 1))) - (1 / (S - 1))) / (1 - 1 / (S - 1))

        FEven.append(FEve_val)
        pID.append(pid)

    FEve_df = pd.DataFrame({"PID": pID, "Functional_Evenness": FEven})
    return FEve_df

def Functional_Divergence(sp_loc:pd.DataFrame, traits: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Functional Divergence (FDiv).

    Args:
        sp_loc: Pivot table of Plot IDs and Species
        traits: DataFrame of species traits

    Returns:
        FDiv (float)
    """
    
    pID = []
    FDivergence = []

    # Get species present in each PID
    species_PID = sp_loc.apply(lambda row: row.index[row != 0].tolist(), axis=1)

    for pid, species in zip(species_PID.index,species_PID):
        
        S = len(species)
        
        if S < 3:
            # FEve undefined for <2 species
            FDivergence.append(np.nan)
            pID.append(pid)
            
            continue

        ab= sp_loc[sp_loc.index == pid]
        # Relative abundancesp    
        ab = ab[[c for c in species if c in ab.columns]]

        ab = ab.loc[:, ab.columns.isin(traits["Species"])] # Check that species are in both Dfs

        ab = ab.div(ab.sum(axis=1), axis=0)
        ab=np.array(ab)[0]

        # Subset traits for present species
        trait_array = traits[traits["Species"].isin(species)].copy()
        trait_array.drop(columns=['Species'], inplace=True)


        # Compute community centroid
        centroid = np.array(np.mean(trait_array, axis=0))
        trait_array=np.array(trait_array)
        distances = np.linalg.norm(trait_array - centroid, axis=1) 
        dG= np.mean(distances)

        # Distances to centroid
        delta_d= np.sum(ab * (distances - dG))

        abs_delta_d = np.sum(ab * np.abs(distances - dG))

        FDiv = (delta_d + dG) / (abs_delta_d + dG)
        FDivergence.append(FDiv)
        pID.append(pid)

    FDiv_df = pd.DataFrame({"PID": pID, "Functional_Divergences": FDivergence})
    
    return FDiv_df


def Functional_Dispersion(sp_loc:pd.DataFrame, traits: np.ndarray, weighted: bool=False) -> pd.DataFrame:
    """
    Compute Functional Dispersion (FDis) for a community.

    FDis measures the spread of species in trait space.
    Can be computed as abundance-weighted or unweighted.

    Args:
        sp_loc: Pivot table of Plot IDs and Species
        traits (np.ndarray): Trait matrix of shape (S, T), where S = species, T = traits.
        weighted (bool): If True, compute abundance-weighted FDis. If False, compute unweighted FDis.

    Returns:
        float: Functional Dispersion (FDis)
    """
    
    pID = []
    FDispersion = []

    # Get species present in each PID
    species_PID = sp_loc.apply(lambda row: row.index[row != 0].tolist(), axis=1)

    for pid, species in zip(species_PID.index,species_PID):
   
        S = len(species)

        if S < 3:
            # FEve undefined for <2 species
            FDispersion.append(np.nan)
            pID.append(pid)
            
            continue

        # Subset traits for present species
        traits_sub = traits[traits["Species"].isin(species)].copy()
        traits_sub.drop(columns=['Species'], inplace=True)

        if weighted:

            ab= sp_loc[sp_loc.index == pid]
            ab = ab[[c for c in species if c in ab.columns]]
            ab = ab.div(ab.sum(axis=1), axis=0)
            ab=np.array(ab)[0]

            centroid = np.sum(traits_sub * ab[:, None], axis=0)  # Abundance-weighted centroid

        else:
            centroid = np.mean(traits_sub, axis=0)  # Unweighted centroid

        # Distances from centroid
        distances = np.linalg.norm(traits_sub - centroid, axis=1)

        # Compute FDis
        FDis = np.sum(distances * ab) if weighted else np.mean(distances)
        FDispersion.append(FDis)
        pID.append(pid)
    
    FDis_df = pd.DataFrame({"PID": pID, "Functional_Dispersion": FDispersion})

    return FDis_df



def Raos_Q(sp_loc: pd.DataFrame, traits: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Rao's Quadratic Entropy (RaoQ) from a trait distance matrix.
    Args:
        sp_loc: Pivot table of Plot IDs and Species
        traits: DataFrame of functional traits, must have column "Species"
    Returns:
        RQ_df: DataFrame with PID and Raos_Q
    """
    pID = []
    RaosQ = []

    species_PID = sp_loc.apply(lambda row: row.index[row != 0].tolist(), axis=1)

    for pid, species in zip(species_PID.index, species_PID):

        # Align traits to species order, drop species with missing traits
        traits_sub = traits[traits["Species"].isin(species)].copy()
        traits_sub = traits_sub.set_index("Species").reindex(species).dropna()
        species = traits_sub.index.tolist()
        S = len(species)

        if S < 2:
            RaosQ.append(np.nan)
            pID.append(pid)
            continue

        # Distance matrix
        dist_matrix = squareform(pdist(traits_sub.values, metric='euclidean'))

        # Relative abundance — enforce species order
        ab = sp_loc.loc[[pid], species]
        total = ab.sum(axis=1)
        if (total == 0).any():
            raise ValueError(f"Plot {pid} has zero total abundance")
        ab = ab.div(total, axis=0).values.flatten().astype(float)

        # Rao's Q = sum(p_i * p_j * d_ij)
        weight_matrix = np.outer(ab, ab)
        RaoQ_val = np.sum(weight_matrix * dist_matrix)

        RaosQ.append(RaoQ_val)
        pID.append(pid)

    RQ_df = pd.DataFrame({"PID": pID, "Raos_Q": RaosQ})
    return RQ_df

def MPD(sp_loc:pd.DataFrame, traits: np.ndarray) -> pd.DataFrame():
    pID = []
    mpd_list = []

    species_PID = sp_loc.apply(lambda row: row.index[row != 0].tolist(), axis=1)

    for pid, species in zip(species_PID.index,species_PID):

        S = len(species)

        if S < 3:
            # Metric undefined for <2 species
            mpd_list.append(np.nan)
            pID.append(pid)
            
            continue
        
        traits_sub = traits[traits["Species"].isin(species)].copy()
        traits_sub.drop(columns=['Species'], inplace=True)

        # pairwise distances
        distances = pdist(traits_sub.values, metric="euclidean")

        # MPD
        mpd = distances.mean()
        mpd_list.append(mpd)
        pID.append(pid)

    return pd.DataFrame({"PID": pID, "Mean Pairwise D": mpd_list})
