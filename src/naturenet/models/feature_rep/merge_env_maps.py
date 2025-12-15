

from sit_fuse.utils import read_yaml

import argparse
import pickle
import os
import numpy as np

from pprint import pprint

parser = argparse.ArgumentParser()
parser.add_argument("-y", "--yaml", help="YAML file for fusion info.")
args = parser.parse_args()
 
yml_conf = read_yaml(args.yaml)


#"/data/nlahaye/NatureNet/Hammerhead_Out_v1/hammerhead_v1_dfs.pkl"
movement_df_fname = yml_conf["movement_df"]
with open(movement_df_fname, 'rb') as f:
    movement_dfs = pickle.load(f)
  
for pm in range(len(yml_conf["primary_env_maps"])):
    primary_env_map = None
    primary_env_map_fname = os.path.join(yml_conf["primary_env_maps"][pm])
    with open(primary_env_map_fname, 'rb') as f:
        primary_env_map = pickle.load(f)
 
 

    scenes_per_uids = []
    for i in range(len(yml_conf["env_maps"])):
        with open(yml_conf["env_maps"][i], 'rb') as f:
            scenes_per_uids.append(pickle.load(f))


    for i in range(len(scenes_per_uids)):
        for uid in scenes_per_uids[i]:
            if True:
                scenes_per_uid = scenes_per_uids[i][uid]
                for j in range(len(scenes_per_uid)):
                    scenes_per_df = scenes_per_uid[j]
                    if len(primary_env_map) == j:
                        print("ADDING SCENES", uid, j)
                        primary_env_map.append(scenes_per_df)
                        continue
                    elif len(primary_env_map[j]) == 0:
                        print("UPDATING SCENES", uid, j)
                        primary_env_map[j] = scenes_per_df
                        continue
                    else:
                        for k in range(len(scenes_per_df)):
                            print(len(primary_env_map[j]), k)
                            if len(primary_env_map[j]) == k:
                                primary_env_map[j].append(scenes_per_df[k])
                                print("ADDING SCENE", uid, j, k)
                                continue
                            elif len(primary_env_map[j][k]) == 0:
                                primary_env_map[j][k] = scenes_per_df[k]
                                print("UPDATING SCENE", uid, j, k)
                                continue
                            else:
                                good = True
                                for m in range(len(scenes_per_df[k])):
                                    if not hasattr(primary_env_map[j][k][m], "shape"):
                                        print("UPDATING SCENE EMBED", uid, j, k, m)
                                        primary_env_map[j][k][m] = scenes_per_df[k][m]
                                        good = False
                                        continue
                                if good:
                                    print("GOOD", uid, j, k)



    if "uid" in yml_conf and len(yml_conf["uid"]) > pm:
        uid =  yml_conf["uid"][pm]
    else:
        uid = yml_conf["primary_env_maps"][pm][-12:-4]
    print("UID", yml_conf["primary_env_maps"][pm], uid)
    complete = []
    missing = []
    for j in range(len(primary_env_map)):
            if j >= len(movement_dfs[uid]):
                continue
            for k in range(len(primary_env_map[j])):
                if len(primary_env_map[j][k]) == 0:
                    missing.append(movement_dfs[uid][j].iloc[k]["date"])
                else:
                    missed = False
                    for m in range(len(primary_env_map[j][k])):
                        if not hasattr(primary_env_map[j][k][m], "shape"):
                            missing.append(movement_dfs[uid][j].iloc[k]["date"])
                            missed = True
                    if missed == False:
                        complete.append(movement_dfs[uid][j].iloc[k]["date"])

    pprint(missing)
    pprint(complete)


    if "uid" in yml_conf and len(yml_conf["uid"]) > pm:
        fname = os.path.join(os.path.dirname(yml_conf["primary_env_maps"][pm]), "env_map_" + yml_conf["uid"][pm] + ".pkl")
    else
        fname = os.path.join(yml_conf["primary_env_maps"][pm])
 
    with open(fname, 'wb') as f:
        pickle.dump(primary_env_map, f, protocol=pickle.HIGHEST_PROTOCOL)



for i in range(len(yml_conf["env_maps"])):
    print("rm", yml_conf["env_maps"][i])
