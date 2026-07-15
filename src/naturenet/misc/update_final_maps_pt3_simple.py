
import re
#re_w = "2017CA-Bmu-\d+"
import os
import pickle
import zarr
import numpy as np
import glob
#fles = glob.glob("/data/nlahaye/NatureNet/Hammerhead_Out_v1/final_env_maps_*.pkl")
fles = glob.glob("/data/nlahaye/NatureNet/Blue_Whale_v1/final_env_maps*simple_1_*Bmu*pkl")
final_dict = {}
#out_fle = "/data/nlahaye/NatureNet/Hammerhead_Out_v1/final_env_maps_hammerhead_v1.pkl"
out_fle = "/data/nlahaye/NatureNet_Env/output_whale/final_env_maps_whale_v1_simple.pkl"

df_dir = "/data/nlahaye/NatureNet/Blue_Whale_v1/"
df_uid = "whale_v1"
out_dir = "/data/nlahaye/NatureNet/Blue_Whale_v1/"

with open(os.path.join(df_dir, df_uid + '_dfs.pkl'), "rb") as f:
    movement_dfs = pickle.load(f)
 
for key in movement_dfs.keys():
    dels = []
    with open(os.path.join(out_dir, "final_env_maps_simple_1_" + key + ".pkl"), "rb") as f:
        abstract_grid = pickle.load(f)

    for key2 in abstract_grid.keys():
        if key2 != key:
            dels.append(key2)
            continue

        for i in range(len(abstract_grid[key])):
             print(key, np.min(abstract_grid[key][i]), np.max(abstract_grid[key][i]))
        final_dict[key] = abstract_grid[key]
 

    for key2 in dels:
        del abstract_grid[key2]

    with open(os.path.join(out_dir, "final_env_maps_simple_1_" + key + ".pkl"), "wb") as f:
        pickle.dump(abstract_grid, f, protocol=pickle.HIGHEST_PROTOCOL)


with open(out_fle, 'wb') as f:
    pickle.dump(final_dict, f, protocol=pickle.HIGHEST_PROTOCOL)


