




import os
import argparse
import math
import pickle
import sparse
import pandas as pd
from sit_fuse.utils import read_yaml

from naturenet.environment.grid_utils import Grid, compute_transition_probabilities, split_movement_streams



def run_movement_track_preprocess(yml_conf):

    movement_df = pd.read_csv(yml_conf["movement_csv"])

    #grid = Grid(math.floor(movement_df["lon"].min()), math.floor(movement_df["lat"].min()), \
    #        math.ceil(movement_df["lon"].max()),  math.ceil(movement_df["lat"].max()), yml_conf["grid_res_lon"], yml_conf["grid_res_lat"])

    movement_dfs = split_movement_streams(movement_df, yml_conf["out_dir"], yml_conf["run_uid"])
 
    for key in movement_dfs.keys():

        movement_sub_df = movement_dfs[key]
        print(len(movement_sub_df), movement_sub_df[0].info)

        total_count = {}
        min_lon = 1000
        min_lat = 1000
        max_lon = -1000
        max_lat = -1000
        for i in range(len(movement_sub_df)):
            min_lon = min(min_lon, movement_sub_df[i]["lon"].min())
            min_lat = min(min_lat, movement_sub_df[i]["lat"].min())
            max_lon = max(max_lon, movement_sub_df[i]["lon"].max())
            max_lat = max(max_lat, movement_sub_df[i]["lat"].max())
        grid = Grid(min_lon, min_lat, max_lon, max_lat, yml_conf["grid_res_lon"], yml_conf["grid_res_lat"])

        total_count = {}
        points = None
        for i in range(len(movement_sub_df)):
            checked, total_count, points, trans_prob = compute_transition_probabilities(movement_sub_df[i], grid, checked = {}, total_count = total_count, points = points)

        run_uid = yml_conf["run_uid"] + "_" + key
        print(run_uid)
        sparse.save_npz(os.path.join(yml_conf["out_dir"], run_uid + "_grid.pkl"), trans_prob)

        with open(os.path.join(yml_conf["out_dir"], run_uid + "_grid.pkl"), "wb") as f:
             pickle.dump(grid, f, protocol=pickle.HIGHEST_PROTOCOL) 

        with open(os.path.join(yml_conf["out_dir"], run_uid + "_total_count.pkl"), "wb") as f:
             pickle.dump(total_count, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(yml_conf["out_dir"], run_uid + "_points.pkl"), "wb") as f:
                     pickle.dump(points, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for fusion info.")
    args = parser.parse_args()

    #Translate config to dictionary 
    yml_conf = read_yaml(args.yaml)

    run_movement_track_preprocess(yml_conf)









