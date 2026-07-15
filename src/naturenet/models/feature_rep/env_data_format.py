import argparse
import copy
import os
import zarr
import glob
import pickle
from osgeo import gdal
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from sit_fuse.utils import read_yaml

import re
import cv2

from pprint import pprint


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


    mask_path = yml_conf["mask_path"]
    land_value = yml_conf["land_value"]

    output_uid = yml_conf["output_uid"] #"final_env_maps_"
 
    initial_water_mask = gdal.Open(mask_path).ReadAsArray()
    resized_water_mask = None
    land_inds = None

    tme = datetime.strptime(start_time,"%Y-%m-%d")
    for i in range(n_days):
        fname = glob.glob(glob_start + str(i) + glob_end)[0]
        if "tif" in fname:
            dat = gdal.Open(fname).ReadAsArray() 
        elif "zarr" in fname:
            dat = zarr.load(fname)

        if resized_water_mask is None:
            print("SHAPING WATER MASK", dat.shape)
            dat_shp = (dat.shape[1], dat.shape[0])
            if dat.ndim > 2:
                dat_shp = (dat.shape[2], dat.shape[1])
            resized_water_mask = cv2.resize(initial_water_mask, dat_shp, interpolation=cv2.INTER_CUBIC)
            land_inds = np.where(resized_water_mask == land_value)
            
        if dat.ndim > 2:
            for j in range(dat.shape[0]):
                tmp = dat[j]
                tmp[land_inds] = -1
                dat[j] = tmp
        else:
            dat[land_inds] = -1

        print("SCENE", i, "LOADED")        
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
            strng = re.sub("__aug\d+","", uid)

            print(strng, uid)
            if not os.path.exists(os.path.join(out_dir, output_uid + uid + ".pkl")):
                os.symlink(os.path.join(out_dir, output_uid + strng + ".pkl"), os.path.join(out_dir, output_uid + uid + ".pkl"))
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

            pkl_file = os.path.join(out_dir, output_uid + uid + ".pkl")
            with open(pkl_file, 'wb') as f:
                pickle.dump({uid: scenes_per_uid[uid]}, f, protocol=pickle.HIGHEST_PROTOCOL)



def compress_clusters(yml_conf):

    times = []
    scenes = []


    n_days = yml_conf["n_days"]
    start_time = yml_conf["start_time"]
    glob_start = yml_conf["glob_start"]
    glob_end = yml_conf["glob_end"]
    df_uid = yml_conf["df_uid"]
    df_dir = yml_conf["df_dir"]
    out_dir = yml_conf["out_dir"]

    output_uid = yml_conf["output_uid"] #"final_env_maps_"

    final_env_maps = []
    lst = None

    tme = datetime.strptime(start_time,"%Y-%m-%d")

    for i in range(n_days):
        fname = glob.glob(glob_start + str(i) + glob_end)[0]
        dat = gdal.Open(fname).ReadAsArray()
        scenes.append(dat)
        times.append(tme)
        tme = tme + timedelta(days=1)

        inds = np.where(dat <= 0.0)
        dat = dat*1000
        dat[inds] = -1

        dat = np.round(dat, decimals=0, out=None).astype(np.int32)
        scenes.append(fname)

        unq = np.unique(dat)
        if lst is None:
            lst = unq
        else:
            lst = np.concatenate((lst, unq), axis=0)
        lst = np.sort(np.unique(lst))


    print("UNIQUE", lst)
    mapper = {}
    print(lst.shape)
    for i in range(lst.shape[0]):
        mapper[float(lst[i])] = i

    mapper[float(-1.0)] = 0.0
    mapper[float(0.0)] = 0.0

    print("MAPPER")
    pprint(mapper)
    with open(os.path.join(out_dir, output_uid + "_whale_class_mapper.pkl"), "wb") as f:
        pickle.dump(mapper, f, protocol=pickle.HIGHEST_PROTOCOL)


    #for j in range(len(scenes)):
    #    dat = gdal.Open(scenes[j])
    #    for key in mapper.keys():
    #        dat[np.where(dat == float(key))] = mapper[key]
    #    print(dat.min(), dat.max())    


    #with open(os.path.join(out_dir, output_uid + "_whale_class_mapper.pkl"), "rb") as f:
    #    mapper = pd.read_pickle(f)

    with open(os.path.join(df_dir, df_uid + '_dfs.pkl'), "rb") as f:
        movement_dfs = pd.read_pickle(f)

    for key in movement_dfs.keys():
        if "__aug" in key:
            continue

        #if "Bmu-008" not in key and "Bmu-01" not in key:
        #    continue

        with open(os.path.join(out_dir, output_uid + key + ".pkl"), "rb") as f:
            abstract_grid = pickle.load(f)

        movement_sub_df = movement_dfs[key]
        
        abstract_grid_sub = abstract_grid[key]

        processed = 0
        for i in range(len(movement_sub_df)):
            #for j in range(len(abstract_grid_sub[i])):
            abstract_grid_sub[i] = np.array(abstract_grid_sub[i])
            print("HERE1", abstract_grid_sub[i].min(), abstract_grid_sub[i].max(), abstract_grid_sub[i].mean(), abstract_grid_sub[i].shape)
            #if abstract_grid_sub[i].max() > 100 and abstract_grid_sub[i].max() < 1000:
            #    processed = processed + 1
            #    continue
            inds = np.where(abstract_grid_sub[i] <= 0.0)
            #if abstract_grid_sub[i].max() >= 1000:
            #    abstract_grid_sub[i] = abstract_grid_sub[i]/1000
            #else:
            print("HERE UNIQUE", abstract_grid_sub[i].min(), abstract_grid_sub[i].max(), abstract_grid_sub[i].mean(), np.unique(abstract_grid_sub[i]))

            abstract_grid_sub[i] = abstract_grid_sub[i]*1000
            abstract_grid_sub[i][inds] = -1
            abstract_grid_sub[i] = np.round(abstract_grid_sub[i], decimals=0, out=None).astype(np.int32)

            print("HERE UNIQUE", abstract_grid_sub[i].min(), abstract_grid_sub[i].max(), abstract_grid_sub[i].mean(), np.unique(abstract_grid_sub[i]))
            for key2 in mapper.keys():
                print(key, key2, mapper[key2], i)
                abstract_grid_sub[i][np.where(abstract_grid_sub[i] == float(key2))] = mapper[key2]
            print("HERE3", abstract_grid_sub[i].min(), abstract_grid_sub[i].max(), i)
            abstract_grid_sub[i] = abstract_grid_sub[i].astype(np.int16)
        #print(np.min(abstract_grid_sub), np.max(abstract_grid_sub))   
        if processed >= len(movement_sub_df):
            continue



        abstract_grid[key] = abstract_grid_sub

        pkl_file = os.path.join(out_dir, output_uid + key + ".pkl")
        with open(pkl_file, 'wb') as f:
            pickle.dump(abstract_grid, f, protocol=pickle.HIGHEST_PROTOCOL)



def merge_compressed_maps(yml_conf):

    merge_glob = os.path.join(yml_conf["out_dir"], yml_conf["output_uid"] + "*.pkl")
    fles = glob.glob(merge_glob)
    out_fle = os.path.join(yml_conf["out_dir"], yml_conf["output_uid"] + yml_conf["df_uid"] + ".pkl")

    final_dict = {}

    for fi in range(len(fles)):
         if "__aug" in fles[fi]:
             continue
         print(fles[fi])
         with open(fles[fi], "rb") as f:
             abstract_grid = pickle.load(f)
         for key in abstract_grid.keys():
             print(fles[fi], key)
             final_dict[key] = abstract_grid[key]

    with open(out_fle, 'wb') as f:
        pickle.dump(final_dict, f, protocol=pickle.HIGHEST_PROTOCOL)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for fusion info.")
    args = parser.parse_args()
  
    yml_conf = read_yaml(args.yaml)

    #gen_scene_list(yml_conf)
 
    if yml_conf["sit_fuse_output"]:
        compress_clusters(yml_conf)

    merge_compressed_maps(yml_conf)

