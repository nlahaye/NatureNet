import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage

import os

def build_ocean_mask(input_mask_path, water_value=1, connectivity=8):
    """
    Build a boolean mask of ocean-connected water cells.

    Assumes water pixels have value == water_value.
    Ocean is defined as water connected to the raster boundary.
    """
    with rasterio.open(input_mask_path) as src:
        band = src.read(1)

        water = (band == water_value)

        if connectivity == 8:
            structure = np.array([[1, 1, 1],
                                  [1, 1, 1],
                                  [1, 1, 1]], dtype=np.uint8)
        else:
            structure = np.array([[0, 1, 0],
                                  [1, 1, 1],
                                  [0, 1, 0]], dtype=np.uint8)

        labels, _ = ndimage.label(water, structure=structure)

        boundary_labels = set()
        boundary_labels.update(np.unique(labels[0, :]))
        boundary_labels.update(np.unique(labels[-1, :]))
        boundary_labels.update(np.unique(labels[:, 0]))
        boundary_labels.update(np.unique(labels[:, -1]))
        boundary_labels.discard(0)

        ocean_mask = np.isin(labels, list(boundary_labels))

        profile = src.profile.copy()
        profile.update(
                driver="GTiff",
                count=1,
                dtype=rasterio.uint8,
                compress="lzw",
                nodata=0
        )
 
        output_mask_path = os.path.splitext(input_mask_path)[0] + ".ocean.tif"
        with rasterio.open(output_mask_path, "w", **profile) as dst:
            dst.write(ocean_mask, 1)

 
#build_ocean_mask("/data/nlahaye/remoteSensing/watermask_2025.whale.2.tif")
build_ocean_mask("/data/nlahaye/remoteSensing/watermask_2025.shark.2.tif")


