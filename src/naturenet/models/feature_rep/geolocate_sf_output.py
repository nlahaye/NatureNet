import math
import numpy as np
import rasterio
from rasterio.transform import from_origin
from osgeo import gdal
import numpy as np
import glob
import cv2

min_lon = 16.6
max_lon = 17.0
min_lat = 48.1
max_lat = 48.2

x_res = 0.0009000090001
y_res = 0.0009000090001

for i in range(61):
    fname = glob.glob("/data/nlahaye/output/Learnergy/COP_BOAR_DEMO/cop_env_boar_" +\
         str(i) + ".zarr.clust.data_*clusters.no_geo.tif")[0] 
 
    array = gdal.Open(fname).ReadAsArray()

    if array.ndim == 2:
        array = array[np.newaxis, :, :] 


    bands, height, width = array.shape

    expected_width = int(round((max_lon - min_lon) / x_res))
    expected_height = int(round((max_lat - min_lat) / y_res))

    if width != expected_width or height != expected_height:
        array = cv2.resize(np.squeeze(array), (expected_height, expected_width), interpolation=cv2.INTER_NEAREST)
        array = array[np.newaxis, :, :]

    transform = from_origin(min_lon, max_lat, x_res, y_res)
    dtype = array.dtype

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": bands,
        "dtype": dtype,
        "crs": "EPSG:4326",
        "transform": transform,
    }

    profile["nodata"] = -1
 

    output_tif = fname.replace("no_geo", "full_geo")

    with rasterio.open(output_tif, "w", **profile) as dst:
        dst.write(array.astype(dtype))

    print(f"Wrote {output_tif} with CRS EPSG:4326")

