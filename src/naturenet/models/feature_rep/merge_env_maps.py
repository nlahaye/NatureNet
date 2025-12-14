

from sit_fuse.utils import read_yaml

import pickle
import os
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("-y", "--yaml", help="YAML file for fusion info.")
args = parser.parse_args()
 
yml_conf = read_yaml(args.yaml)

primary_env_map_fname = os.path.join(yml_conf["primary_env_map"])
with open(primary_env_map_fname, 'rb') as f:
    primary_scenes_per_uid = pickle.load(f)
 


scenes_per_uids = []
for i in range(len(yml_conf["env_maps"])):
    with open(yml_conf["env_maps"][i], 'rb') as f
        scenes_per_uids.append(pickle.load(f))



scenes_per_uid[uid]


for i in range(len(scenes_per_uids)):
    for uid in scenes_per_uids[i]
        if uid not in primary_env_map:
            primary_env_map[uid] = scenes_per_uids[i][uid]
        else:
            scenes_per_uid = scenes_per_uids[i][uid]
            for j in range(len(scenes_per_uid)):
                scenes_per_df = scenes_per_uid[j]
                for k in range(len(scenes_per_df)):
                    if len(primary_env_map[uid][j][k]) == 0:
                        primary_env_map[uid][j][k] = scenes_per_df[k]
                    else:
                        for m in range(len(scenes_per_df[k])):
                            if not hasattr(primary_env_map[uid][j][k][m], "shape"):
                                primary_env_map[uid][j][k][m] = scenes_per_df[k][m]

 

with open(primary_env_map_fname, 'wb') as f:
    pickle.dump(primary_env_map, f, protocol=pickle.HIGHEST_PROTOCOL)


