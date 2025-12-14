

from sit_fuse.utils import read_yaml

import pickle
import os
import numpy as np

from pprint import pprint

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


#"/data/nlahaye/NatureNet/Hammerhead_Out_v1/hammerhead_v1_dfs.pkl"
movement_df_fname = yml_conf["movement_df"]
with open(movement_df_fname, 'rb') as f
    movement_dfs = pickle.load(f)


complete = []
missing = []
for uid in primary_env_map.keys():
    for j in range(len(primary_env_map[uid])):
        for k in range(len(primary_env_map[uid][k])):
            if len(primary_env_map[uid][j][k] == 0:
                missing.append(movement_df[uid][j][k]["date"])
            else:
                missing = False
                for m in range(len(primary_env_map[uid][j][k])):
                    if not hasattr(primary_env_map[uid][j][k][m], "shape"):
                        missing.append(movement_df[uid][j][k]["date"])
                        missing = True
                if missing == False:
                    complete.append(movement_df[uid][j][k]["date"])

pprint(missing)
pprint(complete)


for uid in primary_env_map.keys():
    fname = os.path.join(os.path.dirname(primary_env_map_fname, "env_map_" + uid + ".pkl")

    with open(fname, 'wb') as f:
        pickle.dump(primary_env_map[uid], f, protocol=pickle.HIGHEST_PROTOCOL)



for i in range(len(yml_conf["env_maps"])):
    print("rm", yml_conf["env_maps"][i])
