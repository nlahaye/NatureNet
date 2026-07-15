



import os
import argparse
import math
import pickle
import sparse
import pandas as pd
from sit_fuse.utils import read_yaml
import numpy as np

from naturenet.environment.grid_utils import Grid, compute_transition_probabilities, split_movement_streams, compute_transition_probs_abstracted_env, lat_lon_to_grid, split_pre_split_streams, streamline_columns



def run_movement_track_preprocess(yml_conf):


    df_uid = yml_conf["df_run_uid"]
    df_dir = yml_conf["df_dir"]

    run_uid = yml_conf["run_uid"]
    out_dir = yml_conf["out_dir"]
    skip_uids = yml_conf["skip_uids"]

    df_fname = os.path.join(df_dir, df_uid + '_dfs.pkl')

    #with open(os.path.join(df_dir, df_uid + '_dfs.pkl'), "rb") as f:
    #    movement_dfs = pickle.load(f)

    if "pre_split" in yml_conf and yml_conf["pre_split"]:
        ids = []
        dfs_by_id = []
   
        for i in range(len(yml_conf["pre_split_movement_csvs"])):
            dfs_by_id.append(streamline_columns(pd.read_csv(yml_conf["pre_split_movement_csvs"][i])))
            ids.append((os.path.splitext(os.path.basename(yml_conf["pre_split_movement_csvs"][i]))[0]).replace(" ", "_"))
  
        movement_dfs = split_pre_split_streams(dfs_by_id, ids, yml_conf["out_dir"], yml_conf["run_uid"])

    elif os.path.exists(df_fname):
        
        movement_dfs = s = None
        with open(df_fname, "rb") as f:
            movement_dfs = pd.read_pickle(f)
    else:
        movement_df = pd.read_csv(yml_conf["movement_csv"])
        movement_dfs = split_movement_streams(movement_df, yml_conf["out_dir"], yml_conf["run_uid"])
    points = None
    total_count = {} 
    for key in movement_dfs.keys():

        run_uid = yml_conf["run_uid"] + "_" + key
        tp_fname = os.path.join(yml_conf["out_dir"], run_uid + "_trans_prob.npz")
        print("HERE", tp_fname, os.path.exists(tp_fname))
        if os.path.exists(tp_fname) or key in skip_uids: 
            continue

        movement_sub_df = movement_dfs[key]

        if "lon_bounds" in yml_conf:
            min_lon = yml_conf["lon_bounds"][0]
            max_lon = yml_conf["lon_bounds"][1]
            min_lat = yml_conf["lat_bounds"][0]
            max_lat = yml_conf["lat_bounds"][1]
        else:
            min_lon = 1000
            min_lat = 1000
            max_lon = -1000
            max_lat = -1000
            for i in range(len(movement_sub_df)):
                min_lon = min(min_lon, movement_sub_df[i]["longitude"].min())    #["lon"].min())
                min_lat = min(min_lat, movement_sub_df[i]["latitude"].min())   #["lat"].min())
                max_lon = max(max_lon, movement_sub_df[i]["longitude"].max())   #["lon"].max())
                max_lat = max(max_lat, movement_sub_df[i]["latitude"].max())     #["lat"].max())
        grid = Grid(min_lon, min_lat, max_lon, max_lat, yml_conf["grid_res_lon"], yml_conf["grid_res_lat"])

        single_unit_coords_full = []
        actions_full = []
        distances_full = []       
 
        for i in range(len(movement_sub_df)):
            print(key, i)
            total_count, points, trans_prob, actions, single_unit_coords, distances = compute_transition_probabilities(movement_sub_df[i], grid, total_count = total_count, points = points)

            single_unit_coords_full.append(single_unit_coords) 
            actions_full.append(actions)
            distances_full.append(distances)

        run_uid = yml_conf["run_uid"] + "_" + key
        sparse.save_npz(os.path.join(yml_conf["out_dir"], run_uid + "_trans_prob"), trans_prob)

        with open(os.path.join(yml_conf["out_dir"], run_uid + "_grid.pkl"), "wb") as f:
             pickle.dump(grid, f, protocol=pickle.HIGHEST_PROTOCOL) 

        with open(os.path.join(yml_conf["out_dir"], run_uid + "_total_count.pkl"), "wb") as f:
             pickle.dump(total_count, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(yml_conf["out_dir"], run_uid + "_points.pkl"), "wb") as f:
             pickle.dump(points, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(yml_conf["out_dir"], run_uid + "_actions.pkl"), "wb") as f:
             pickle.dump(actions_full, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(yml_conf["out_dir"], run_uid + "_single_unit_coords.pkl"), "wb") as f:
             pickle.dump(single_unit_coords_full, f, protocol=pickle.HIGHEST_PROTOCOL)      

        #with open(os.path.join(yml_conf["out_dir"], run_uid + "_distances.pkl"), "wb") as f:
        #     pickle.dump(distances_full, f, protocol=pickle.HIGHEST_PROTOCOL)



    sparse.save_npz(os.path.join(yml_conf["out_dir"], yml_conf["run_uid"] +  "_trans_prob"), trans_prob)

    with open(os.path.join(yml_conf["out_dir"],  yml_conf["run_uid"] +  "_total_count.pkl"), "wb") as f:
        pickle.dump(total_count, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(yml_conf["out_dir"],  yml_conf["run_uid"] +  "_points.pkl"), "wb") as f:
        pickle.dump(points, f, protocol=pickle.HIGHEST_PROTOCOL)


def gen_grid_point_paths(yml_conf):

    movement_dfs = None
    grid = None

    df_uid = yml_conf["df_run_uid"]
    df_dir = yml_conf["df_dir"]

    run_uid = yml_conf["run_uid"]
    out_dir = yml_conf["out_dir"]

    with open(os.path.join(df_dir, df_uid + '_dfs.pkl'), "rb") as f:
        movement_dfs = pickle.load(f)

    for key in movement_dfs.keys():
        grid = None
        
        paths = []
        df_uid = yml_conf["df_run_uid"] + "_" + key    
        with open(os.path.join(df_dir, df_uid + "_grid.pkl"), "rb") as f:
            grid = pickle.load(f)

        movement_sub_df = movement_dfs[key]
        for i in range(len(movement_sub_df)):
            sub_df = movement_sub_df[i]
            path = []
            for index, row in movement_sub_df[i].iterrows():
                y,x = lat_lon_to_grid(grid, row["latitude"], row["longitude"]) #row["lat"], row["lon"])
                print(y, x, grid.grid_res_lon_deg, grid.grid_res_lat_deg, grid.lon_size, grid.lat_size, row["latitude"], row["longitude"], "HERE GRIDDING")
                print(grid.min_lon, grid.min_lat, grid.max_lon, grid.max_lat, grid.lon_tiles, grid.lat_tiles)

                path.append({"y":y,"x":x})
            print(path, len(movement_sub_df[i]), i, key)
            paths.append(path)

        with open(os.path.join(df_dir, yml_conf["df_run_uid"] + "_" + str(key) + "_grid_paths.pkl"), "wb") as f:
            pickle.dump(paths, f, protocol=pickle.HIGHEST_PROTOCOL) 
         
 
def reprocess_env_stats_abstract(yml_conf):

    movement_dfs = None
    grid = None
    abstract_grid = None

    df_uid = yml_conf["df_run_uid"]
    df_dir = yml_conf["df_dir"]

    run_uid = yml_conf["run_uid"]
    out_dir = yml_conf["out_dir"]



    with open(os.path.join(df_dir, df_uid + '_dfs.pkl'), "rb") as f:
        movement_dfs = pd.read_pickle(f)

    with open(os.path.join(out_dir, "final_env_maps_" + run_uid + ".pkl"), "rb") as f: #TODO - _simple* vs not simple
        abstract_grid = pickle.load(f)


    n_clusters = yml_conf["n_clusters"]

    total_count = {}
    points = None
    for key in movement_dfs.keys():
        if key not in abstract_grid:
            continue
        grid = None
        actions = None
 
        df_uid = yml_conf["df_run_uid"] + "_" + key
        with open(os.path.join(df_dir, df_uid + "_grid.pkl"), "rb") as f:
            grid = pickle.load(f)
 
        with open(os.path.join(df_dir, df_uid + "_actions.pkl"), "rb") as f:
            actions = pickle.load(f)

        abstract_grid_sub = abstract_grid[key]
        movement_sub_df = movement_dfs[key]

        #total_count = {}
        #points = None
        states_full = []

        print(len(movement_sub_df))
        for i in range(len(movement_sub_df)):
            abstract_grid_sub[i] = np.array(abstract_grid_sub[i])
            abstract_grid_sub[i][np.where(abstract_grid_sub[i] < 0)] = 0
            total_count, points, trans_prob, states = compute_transition_probs_abstracted_env(movement_sub_df[i], abstract_grid_sub[i], actions[i], grid, n_clusters, points = points, total_count = total_count)
            states_full.append(states)

        with open(os.path.join(df_dir, yml_conf["df_run_uid"] + "_" + str(key) + "_states_abstract_env.pkl"), "wb") as f:
            pickle.dump(states_full, f, protocol=pickle.HIGHEST_PROTOCOL)

    sparse.save_npz(os.path.join(df_dir, yml_conf["df_run_uid"] + "_trans_prob_abstract_env"), trans_prob)

    with open(os.path.join(df_dir, yml_conf["df_run_uid"] + "_total_count_abstract_env.pkl"), "wb") as f:
         pickle.dump(total_count, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(df_dir, yml_conf["df_run_uid"] + "_points_abstract_env.pkl"), "wb") as f:
         pickle.dump(points, f, protocol=pickle.HIGHEST_PROTOCOL)

 

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for fusion info.")
    args = parser.parse_args()

    #Translate config to dictionary 
    yml_conf = read_yaml(args.yaml)

 
    if yml_conf["init_preprocess"]:
        run_movement_track_preprocess(yml_conf)
    else:
       reprocess_env_stats_abstract(yml_conf)
        








