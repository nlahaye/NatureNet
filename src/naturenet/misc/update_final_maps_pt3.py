
import re
#re_w = "2017CA-Bmu-\d+"
import os
import pickle
import zarr
import numpy as np
import glob
fles = glob.glob("/data/nlahaye/NatureNet/Hammerhead_Out_v1/final_env_maps_*.pkl")
#"/data/nlahaye/NatureNet/Blue_Whale_v1/final_env_maps_2017CA-Bmu-*.pkl")
final_dict = {}
out_fle = "/data/nlahaye/NatureNet/Hammerhead_Out_v1/final_env_maps_hammerhead_v1.pkl"
#"/data/nlahaye/NatureNet_Env/output_whale/final_env_maps_whale_v1.pkl"

for fi in range(len(fles)):
     del_keys = []
     print(fles[fi])
     with open(fles[fi], "rb") as f:
         abstract_grid = pickle.load(f)
     for key in abstract_grid.keys():
         print(fles[fi], key)
         final_dict[key] = abstract_grid[key]

with open(out_fle, 'wb') as f:
    pickle.dump(final_dict, f, protocol=pickle.HIGHEST_PROTOCOL)


