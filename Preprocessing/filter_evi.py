import os
import glob
import rasterio
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

"""
Extracts EVI values from a folder of GeoTIFF rasters for only a set of point
locations. This script produces a CSV containing EVI values
for each (lat, lon) point.

INPUTS
------
1) PID_location.csv
   Must contain:
       lat, lon   (WGS84 coordinates)

2) Folder of GeoTIFFs:
   Each GeoTIFF represents an EVI map at a unique time (e.g., July each year)
"""

def load_points(csv_path: str, lat_col="lat", lon_col="lon") -> gpd.GeoDataFrame:
    """
    Load point locations (lat, lon) into a GeoDataFrame.

    Args:
        csv_path (str): Path to CSV with lat/lon columns.
        lat_col (str): Name of the latitude column.
        lon_col (str): Name of the longitude column.

    Returns:
        gpd.GeoDataFrame: Points in EPSG:4326.
    """
    df = pd.read_csv(csv_path)
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs="EPSG:4326"
    )
    return gdf

def extract_evi(points_gdf: gpd.GeoDataFrame, tif_folder: str) -> pd.DataFrame:
    """
    Extract raster values for each point across all GeoTIFFs.

    Args:
        points_gdf (gpd.GeoDataFrame): Point geometries (EPSG:4326).
        tif_folder (str): Directory containing GeoTIFFs.

    Returns:
        pd.DataFrame: Table of filename, lat, lon, evi.
    """
    tif_list = sorted(glob.glob(os.path.join(tif_folder, "*.tif")))
    if not tif_list:
        raise FileNotFoundError(f"No .tif files found in {tif_folder}")

    all_records = []

    for tif_path in tif_list:
        print(f"Processing: {os.path.basename(tif_path)}")

        with rasterio.open(tif_path) as src:
            # Reproject if CRS differs
            if points_gdf.crs != src.crs:
                pts = points_gdf.to_crs(src.crs)
            else:
                pts = points_gdf.copy()

            raster_band = src.read(1)

            values = []
            for geom in pts.geometry:
                try:
                    row, col = src.index(geom.x, geom.y)
                    val = raster_band[row, col]
                except:
                    val = None
                values.append(val)

        temp_df = pd.DataFrame({
            "filename": os.path.basename(tif_path),
            "lat": points_gdf.geometry.y,
            "lon": points_gdf.geometry.x,
            "evi": values
        })
        all_records.append(temp_df)

    return pd.concat(all_records, ignore_index=True)


if __name__ == "__main__":

    points_csv = "PID_location.csv"
    tif_folder = "data/GEE_EVI_July"

    # Load point locations
    points = load_points(points_csv)

    # Extract EVI
    results = extract_evi(points, tif_folder)
    
    results['filename'] = results['filename'].replace({
        "California_July_EVI_2013.tif": 2013,
        "California_July_EVI_2014.tif": 2014, 
        "California_July_EVI_2015.tif": 2015,
        "California_July_EVI_2016.tif": 2016, 
        "California_July_EVI_2017.tif": 2017,
        "California_July_EVI_2018.tif": 2018, 
        "California_July_EVI_2019.tif": 2019,
        "California_July_EVI_2020.tif": 2020,
        "California_July_EVI_2021.tif": 2021,
        "California_July_EVI_2022.tif": 2022
    })

    results=results.rename(columns={'filename': "year"}, errors="raise")

    # Save
    out_path = "evi_time.csv"
    results.to_csv(out_path, index=False)

    print(f"Saved: {out_path}")