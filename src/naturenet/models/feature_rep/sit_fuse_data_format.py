
import copy
import os
import zarr
import glob
import pickle
from osgeo import gdal
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def gen_scene_list(yml_conf):
    times = []
    scenes = []


    n_days = yml_conf["n_days"]
    start_time = yml_conf["start_time"]
    glob_start = yml_conf["glob_start"]
    glob_end = yml_conf["glob_end"]
    df_uid = yml_conf["df_uid"]
    df_dir = yml_conf["df_dir"]
    out_dir = yml_conf["out_dir"]

    tme = datetime.strptime(start_time,"%Y-%m-%d")
    for i in range(n_days):
        fname = glob.glob(glob_start + str(i) + glob_end)[0]
        dat = gdal.Open(fname).ReadAsArray() 
        scenes.append(dat)
        times.append(tme)
        tme = tme + timedelta(days=1)

    

    scenes_per_uid = {}
    movement_dfs = None
    print(os.path.join(df_dir, df_uid + '_dfs.pkl'))
    with open(os.path.join(df_dir, df_uid + '_dfs.pkl'), "rb") as f:
        movement_dfs = pd.read_pickle(f)
 
    for uid in movement_dfs:
        if "__aug" in uid:
            continue

        run_uid_dist = df_uid + "_" + uid
        #with open(os.path.join(df_dir, run_uid_dist + "_distances.pkl"), "rb") as f:
        #    distances = pickle.load(f)


        print("Matching movement tracks to scenes and adding track-specific environment info for track", uid)
        movement_dfs_uid = movement_dfs[uid]

        if uid not in scenes_per_uid:
            scenes_per_uid[uid] = []


        print(len(movement_dfs_uid))
        for dind in range(len(movement_dfs_uid)):
            movement_df = movement_dfs_uid[dind]
            print(len(movement_df))
            #distance_grids = distances[dind]
            act_index = 0
            scene_ind = 0

            current_date = None
            current_date_cntr = 0

            if len(scenes_per_uid[uid]) < dind+1:
                scenes_per_df = []
            else:
                scenes_per_df = scenes_per_uid[uid]

            for index, row in movement_df.iterrows():
                #act_index = 0
                #if len(row["date"]) > 10:
                #    row["date"] = row["date"][:10]
                print(row['timestamp'])
                row['timestamp'] = row['timestamp'].replace(" ", "T").replace("+00:00", "Z")
                df_st_str = datetime.strptime(row['timestamp'],"%Y-%m-%dT%H:%M:%SZ")  #row['date'][0:10],"%Y-%m-%d")
                st_str = times[scene_ind] #prelim_scene_map["times"][scene_ind]
 
                print(df_st_str, st_str, len(movement_df), scene_ind, act_index, uid)
                while st_str.date() < df_st_str.date() and scene_ind < len(times)-1: #len(prelim_scene_map["times"])-1:
                    scene_ind = scene_ind + 1
                    st_str = times[scene_ind] #prelim_scene_map["times"][scene_ind]
                    print(df_st_str, st_str, len(movement_df), uid, dind, scene_ind, act_index)
                if current_date is None:
                    current_date = copy.deepcopy(df_st_str)
                else:
                    if current_date.date() == df_st_str.date(): #year == df_st_str.year and current_date.day== df_st_str.day and current_date.month== df_st_str.month:
                        current_date_cntr = current_date_cntr + 1
                        #act_index = act_index + 1
                    else:
                        current_date = copy.deepcopy(df_st_str)
                        current_date_cntr = 0
                
                #if st_str - df_st_str > timedelta(hours=15) or df_st_str - st_str > timedelta(hours=15):
                #    if len(scenes_per_df) < act_index + 1:
                #        scenes_per_df.append([])
                #    act_index = act_index + 1
                #    continue
                #if scene_ind >= len(times): #len(prelim_scene_map["times"]):
                #    act_index = act_index + 1
                #    continue

                #distance_grid = distance_grids[act_index]
                #resample_shape = (scenes["combined_features"][scene_ind][0].shape[1], scenes["combined_features"][scene_ind][0].shape[0])
                #distance_grid = cv2.resize(distance_grid, resample_shape, interpolation=cv2.INTER_CUBIC)
                #distance_grid = np.reshape(distance_grid, (resample_shape[1], resample_shape[0], 1,1,1))
              
                print(st_str, df_st_str, scene_ind)
 
                if len(scenes_per_df) < act_index + 1:
                    print("Appending new scene in ", uid, "at", act_index, "from", scene_ind)
                    scenes_per_df.append(scenes[scene_ind])
                else:
                    print("Inserting scene in ", uid, "at", act_index, "from", scene_ind)
                    scenes_per_df[act_index] = scene


                act_index = act_index + 1

            #print("Intermediate Scene Size", scenes["combined_features"][scene_ind][-1].shape, len(scenes["combined_features"][scene_ind]))

            if len(scenes_per_uid[uid]) < dind+1:
                print("Appending new DF for", uid, "at", dind)
                scenes_per_uid[uid].append(scenes_per_df)
            else:
                print("Inserting DF for", uid, "at", dind)
                scenes_per_uid[uid] = scenes_per_df

            pkl_file = os.path.join(out_dir, "final_env_maps_" + uid + ".pkl")
            with open(pkl_file, 'wb') as f:
                pickle.dump(scenes_per_uid, f, protocol=pickle.HIGHEST_PROTOCOL)

gen_scene_list()


