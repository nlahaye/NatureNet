import pickle
import os
import numpy as np



fname = "/mnt/data/NatureNet_Env/output/env_maps_simple_env60e19c41-25b8-4dba-8508-9428be3c9325.pkl"
 

scenes_per_uid = None
with open(fname, 'rb') as f:
    scenes_per_uid = pickle.load(f)
 


for uid in scenes_per_uid:
    scenes = scenes_per_uid[uid]

    new_fname = "env_map_" + uid + ".pkl"
    with open(new_fname, 'wb') as f:
        pickle.dump(scenes, f, protocol=pickle.HIGHEST_PROTOCOL)



