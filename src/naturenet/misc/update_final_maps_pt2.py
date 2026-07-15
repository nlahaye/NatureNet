
import re
re_w = "2017CA-Bmu-\d+"
import os
import pickle
import zarr
import numpy as np
import glob
fles = glob.glob("/data/nlahaye/NatureNet/Blue_Whale_v1/final_env_maps_2017CA-Bmu-*.pkl")
for fi in range(len(fles)):
     del_keys = []
     print(fles[fi])
     with open(fles[fi], "rb") as f:
         abstract_grid = pickle.load(f)
     mtch = re.search(re_w, fles[fi])
     mtch_str = mtch.group()
     for key in abstract_grid.keys():
         print(key, mtch_str)
         if key != mtch_str:
             del_keys.append(key)
     for key in del_keys:
         del abstract_grid[key]
     for key in abstract_grid.keys():
        for i in range(len(abstract_grid[key])):
            abstract_grid[key][i] = np.array(abstract_grid[key][i]).astype(np.int16)
     with open(fles[fi], 'wb') as f:
         pickle.dump(abstract_grid, f, protocol=pickle.HIGHEST_PROTOCOL)


