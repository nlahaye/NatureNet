import os
import zarr
import numpy as np

 
data_arr = zarr.load("/data/nlahaye/NatureNet_Env/Copernicus/cop_env_whales.zarr")

for i in range(data_arr.shape[0]):
    zarr.save_array("/data/nlahaye/NatureNet_Env/Copernicus/cop_env_whales_" + str(i) + ".zarr", np.squeeze(data_arr[i,:,:,:]))
 

