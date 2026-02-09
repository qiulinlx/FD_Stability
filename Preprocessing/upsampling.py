import rasterio
from rasterio.enums import Resampling

# Open the original canopy entropy raster
with rasterio.open('data/global_forest_csc/global_forest_csc.tif') as src:
    # Compute new shape (4x finer)
    new_shape = (src.height * 2, src.width * 4)
    
    # Resample data
    data_resampled = src.read(
        1,  # first band
        out_shape=new_shape,
        resampling=Resampling.bilinear  # use 'nearest', 'bilinear', or 'cubic'
    )
    
    # Update the transform (pixel size changes)
    transform = src.transform * src.transform.scale(
        (src.width / data_resampled.shape[1]),
        (src.height / data_resampled.shape[0])
    )
    
    # Update metadata
    profile = src.profile
    profile.update(height=data_resampled.shape[0],
                   width=data_resampled.shape[1],
                   transform=transform)
    
    # Save new 4x finer raster
    with rasterio.open('data/global_forest_csc/global_forest_csc_upsampled.tif', 'w', **profile) as dst:
        dst.write(data_resampled, 1)
