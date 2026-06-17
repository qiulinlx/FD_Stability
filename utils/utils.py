import json
"""
Additional Functions that is used for generation etc. 
"""

def parse_geo_string(geo_str):
    '''
    Select longitude and latitute values from a string.
    '''
    # Fix doubled quotes
    fixed = geo_str.replace('""', '"').strip('"')
    
    # Convert to dict
    d = json.loads(fixed)

    # Extract lon / lat
    lon, lat = d["coordinates"]
    return lon, lat


def truncate_after_n_underscores(s: str, n: int = 4) -> str:
    """
    Truncate a string after the fourth underscore.
    Using this to standardize plot IDs (PIDs).

    Args:
        s (str): Input string.
        N (int): number of _ before truncation. Defaults to 4 
    Returns:
        str: String containing only the first four underscore-separated segments.
    """
    parts = s.split("_")
    return "_".join(parts[:n])