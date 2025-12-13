
from naturenet.models.feature_rep.rs_feature_reduce import MultiSourceRSFeatureReduc
from naturenet.models.feature_rep.rs_feature_net import RSFeatureNet

from sit_fuse.pipelines.inference.inference_utils import run_embed_gen_from_scene_arr
from sit_fuse.preprocessing.colocate_and_resample import resample_scene
from sit_fuse.datasets.dataset_utils import get_scenes
from sit_fuse.utils import read_yaml

from sklearn.cluster import MiniBatchKMeans

from collections.abc import Iterable

import torch
import copy
import joblib
import os
import yaml
import argparse
import datetime
import numpy as np
import pickle
import math 
import cv2

from pprint import pprint

def run_normalization_stats(yml_conf, stats = {}):
    for key in yml_conf["instruments"].keys():
        data = None
        for i in range(len(yml_conf["instruments"][key]["filenames"])):
            data_config = read_yaml(yml_conf["instruments"][key]["data_config"])
            dat_tmp, _, _ = get_scenes(data_config, yml_conf["instruments"][key]["filenames"][i])
            dat_tmp = np.array(dat_tmp)        
    
            #if dat_tmp.ndim < 3:
            #    np.expand_dims(dat_tmp, axis=2)

            if data is None:
                data = dat_tmp
            else:
                data = np.concatenate((data, dat_tmp), axis=0)    

        inds = np.where(data < -99990.0)
        data[inds] = np.nan 
        mean = np.nanmean(data, axis=(0,1,2)) 
        std = np.nanstd(data, axis=(0,1,2))

        if key not in stats:
            stats[key] = {}
            stats[key]["mean"] = mean
            stats[key]["std"] = std
        else:
            if isinstance(stats[key], list):
                stats[key] = stats[key][0]
            elif isinstance(stats[key], np.ndarray):
                stats[key] = dict(stats[key])
            stats[key]["mean"] = mean
            stats[key]["std"] = std

    return stats


def resample_or_fuse_scene(scene, init_location, resample_config):

    resample_config["low_res"]["filenames"] = scene
    resample_config["low_res"]["geo_filenames"] = scene
    

    resample_config["return_products"] = True
    output, location = resample_scene(scene, init_location, resample_config)
    return output, location

def gen_grid_info(location, data, final_grid_res_deg, per_channel_stats, compute_final_grid_info = False):

    print("Computing grid scaling and resampling needs")
    #Calculate the current grid resolution (assume even spacing in both direction)
    current_grid_res_deg = abs(location[0,1,0] - location[0,0,0])

    #Number of pixels at current resolution in each tile of final grid
    final_grid_steps = int(math.ceil((final_grid_res_deg / current_grid_res_deg)))

    #Compute differential between current scene size and requirements to evenly fit into final grid size
    final_width = location.shape[1]
    width_mod = int(math.ceil(location.shape[1] % final_grid_steps))
    if width_mod > 0.0:
        final_width = final_width + (final_grid_steps - width_mod)

    final_height = location.shape[0]
    height_mod = int(math.ceil(location.shape[0] / final_grid_steps))
    if height_mod > 0.0:
        final_height = final_height + (final_grid_steps - height_mod)    

    print("Resizing features to account for minor grid size discrepancies")
    #Account for small discrepancies as to not throw off tiling, etc
    if final_width > location.shape[1] or final_height > location.shape[0]:
        final_loc = np.zeros((final_height, final_width, 2))
        for i in range(location.shape[2]):
            final_loc[:,:,i] = cv2.resize(location[:,:,i], (final_width, final_height), interpolation=cv2.INTER_CUBIC)
        del location
        final_data = np.zeros((data.shape[0],final_height, final_width))
        for i in range(data.shape[0]):
            final_data[i,:,:]  = cv2.resize(data[i,:,:], (final_width, final_height), interpolation=cv2.INTER_NEAREST) #Small change, if any - accounts for fill values better w/ NN - may change logic here later
        del data
    else:
        final_loc = location
        final_data = data

    #Compute final grid size - only need to do this once

    final_grid_coords = None
    if compute_final_grid_info:
        print("Computing final grid layout")
        final_grid_width = int(final_width/final_grid_steps)
        final_grid_height = int(final_height/final_grid_steps)
  
        final_grid_coords = np.zeros((final_grid_height, final_grid_width, 2))

        current_j = 0
        for i in range(0,final_grid_width,final_grid_steps):
            center_col = int(i + (final_grid_steps/2))

            for j in range(0,final_grid_height,final_grid_steps):
                center_row = int(j + (final_grid_steps/2))

                center_lon = final_loc[j,center_col,0] + (current_grid_res_deg*(final_grid_steps/2.0))
                center_lat = final_loc[center_row,i,1] + (current_grid_res_deg*(final_grid_steps/2.0))
             
                final_grid_coords[j,i,0] = center_lon
                final_grid_coords[j,i,1] = center_lat

    return final_data, final_loc, final_grid_steps, final_grid_coords





def get_datetime_info(yml_conf):

    #For now, assume uniform time distribution across inputs, taken from input config
    #TODO - automate and deepen logic here

    dts = yml_conf["datetimes"]
    ret_dts = []
  
    for i in range(len(dts)):
        ret_dts.append(datetime.datetime.strptime(dts[i], "%d%m%YT%H:%M:%SZ"))

    return ret_dts


def gen_prelim_scene_map(yml_conf, scene_count, per_channel_stats):
    
    prelim_scene_map = {}    #prelim info
    for i in range(scene_count):
        for instrument in yml_conf["instruments"]:

            resample_config = read_yaml(yml_conf["instruments"][instrument]["resample_config"])
            data_config = read_yaml(yml_conf["instruments"][instrument]["data_config"])

            fnames = yml_conf["instruments"][instrument]["filenames"][i]
            if not isinstance(fnames, list):
                fnames = [fnames]

            print("Getting scene", instrument, i)
            #Get data
            scene, init_shape, init_location = get_scenes(data_config, fnames, stats = per_channel_stats[instrument])
            for scn in range(len(scene)):
                if scene[scn].ndim == 2:
                    scene[scn] = np.expand_dims(scene[scn], 2) #Data preprocessing keeps channel dim at back of dims
 

            #For now all scene sets should be of size 1, and in the future stitching will enforce this in other cases
            scene = scene[0]
            init_location = init_location[0]

            print("Resampling", instrument, i)
            #Resample over grid - TODO I will need to add stitching prior to this for inputs in future iterations
            #Process will be stitch, resample, impute
            data, location = resample_or_fuse_scene(scene, init_location,  resample_config)
            del scene

            print("Post-resample data imputation", instrument, i)
            #Impute across fill values for now - may want to change approach later on #TODO - revisit
            for ch in range(data.shape[0]):
                subd = data[ch]
                inds = np.where(subd < -99990)
                subd[inds] = per_channel_stats[instrument]["mean"][ch]
                data[ch] = subd

            print("Generating grid info", instrument, i)
            if instrument not in prelim_scene_map:
                prelim_scene_map[instrument] = {"scenes" : []} 
            #Generate inital grid info per-scene
            if "final_grid" not in prelim_scene_map:
                #Can ignore resampled lat/lon for now - may be useful later
                final_data, final_loc, final_grid_steps, final_grid_coords = gen_grid_info(location, data, yml_conf["final_grid_res_deg"], per_channel_stats, compute_final_grid_info = True) 
                #tile size will vary relative to native resolution
                prelim_scene_map["final_grid"] = final_grid_coords
                prelim_scene_map[instrument]["tile_size"] = final_grid_steps
            else:
                final_data, final_loc, final_grid_steps, _ = gen_grid_info(location, data, yml_conf["final_grid_res_deg"], per_channel_stats, compute_final_grid_info = False)
                prelim_scene_map[instrument]["tile_size"] = final_grid_steps
            prelim_scene_map[instrument]["scenes"].append(final_data)

 
            print("Associating times to grid", instrument, i)
            if "times" not in prelim_scene_map:
                #Get time info connected to scene
                prelim_scene_map["times"] = get_datetime_info(yml_conf)
    return prelim_scene_map
    

def run_surface_feature_connect(yml_conf, scenes_per_uid={}):
    

    out_dir = yml_conf["out_dir"]
    run_uid = yml_conf["run_uid"]

    pkl_file = os.path.join(out_dir, "input_per_channel_stats.pkl")
    #TODO movement_dfs = yml_conf["movement_dfs"]

    if os.path.exists(pkl_file):
        with open(pkl_file, 'rb') as f:
            per_channel_stats = pickle.load(f)

        for key in per_channel_stats:
            if isinstance(per_channel_stats[key], list):
                per_channel_stats[key] = per_channel_stats[key][0]       
  
        if yml_conf["update_stats"]:
            print("Generating normalization stats")
            per_channel_stats = run_normalization_stats(yml_conf, per_channel_stats)
            with open(pkl_file, 'wb') as f:
                pickle.dump(per_channel_stats, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            print("Loaded normalization stats") 
    else:
        print("Generating normalization stats")
        per_channel_stats = run_normalization_stats(yml_conf)
        with open(pkl_file, 'wb') as f:
            pickle.dump(per_channel_stats, f, protocol=pickle.HIGHEST_PROTOCOL)

    rsfns, _ = load_models(out_dir, run_uid, per_channel_stats)
    if rsfns is None:
        rsfns = {}


    scenes ={}
    grids = {}
    grid_size = None


    #TODO get final shape via initial grids

    print("Computing total number of feature channels")
    scene_count = -1
    total_chans = 0
    for key in yml_conf["instruments"].keys():
        if scene_count < 1:
            scene_count = len(yml_conf["instruments"][key]["filenames"])
        if yml_conf["instruments"][key]["multi_chan"]:
            total_chans += 5
        else:
            total_chans += 1

    msrffr = None
    actual_total_chans = 0

    print("Generating preliminary environment scene info")
    prelim_scene_map = gen_prelim_scene_map(yml_conf, scene_count, per_channel_stats)

    #TODO parallelize
    for i in range(scene_count):
        for instrument in yml_conf["instruments"]:
            #For reflectance or backscatter datasets
            print("Generating features", instrument, i)
            if "encoder_conf" in yml_conf["instruments"][instrument]:
                encoder_conf_fpath = yml_conf["instruments"][instrument]["encoder_conf"]
                encoder_conf = read_yaml(encoder_conf_fpath)
                embed, _ = run_embed_gen_from_scene_arr(encoder_conf, [prelim_scene_map[instrument]["scenes"][i]], \
                    [prelim_scene_map[instrument]["scenes"][i].shape[0:2]], gen_image_shaped = True)
            else:
                embed = prelim_scene_map[instrument]["scenes"][i]

            if embed.ndim < 4: #Likely no sample dimension
                while embed.ndim < 4:
                    embed = np.expand_dims(embed, 0)
                
            print("Extracting multi-scale coarser features", i)
            n_chans = 1
            if isinstance(per_channel_stats[instrument]["mean"], Iterable):
                n_chans = len(per_channel_stats[instrument]["mean"])
            if instrument not in rsfns: #Assuming N_Samples X N_Channels X YDIM X XDIM
                rsfns[instrument] = RSFeatureNet(n_chans, prelim_scene_map[instrument]["tile_size"], \
                    per_channel_stats[instrument]["mean"], per_channel_stats[instrument]["std"])
                scenes[instrument] = []
                #if "combined" not in scenes:
                #    scenes["combined"] = []
                #    scenes["combined_final"] = []

            if i == 0:
                if torch.cuda.is_available():
                    rsfns[instrument] = rsfns[instrument].cuda()

            x1, x2, x3, grid_size  = rsfns[instrument](embed)

            if x1.ndim == 3:
                x1 = x1.reshape((grid_size[0], grid_size[1], 1, x1.shape[1], x1.shape[2]))
                x2 = x2.reshape((grid_size[0], grid_size[1], 1, x2.shape[1], x2.shape[2]))
                x3 = x3.reshape((grid_size[0], grid_size[1], 1, x3.shape[1], x3.shape[2]))
            elif x1.ndim == 4:
                x1 = x1.reshape((grid_size[0], grid_size[1], x1.shape[1], x1.shape[2], x1.shape[3]))
                x2 = x2.reshape((grid_size[0], grid_size[1], x2.shape[1], x2.shape[2], x2.shape[3]))
                x3 = x3.reshape((grid_size[0], grid_size[1], x3.shape[1], x3.shape[2], x3.shape[3]))

            if instrument not in grids:
                grids[instrument] = []

            if instrument not in scenes:
                scenes[instrument] = []

            grids[instrument].append(grid_size)
            scenes[instrument].append([x1, x2, x3])

            if i == 0:
                actual_total_chans += rsfns[instrument].out_chans
            #if len(scenes["combined"]) < (i+1):
            #    scenes["combined"].append(embed)
            #else:
            #    scenes["combined"][i] = np.concatenate((scenes["combined"][i], embed), axis=1)


    for instrument in yml_conf["instruments"]:
        del prelim_scene_map[instrument]["scenes"][i]
        

    print("Concatenating features per-scene")
    for scn in range(scene_count):
        if "combined_features" not in scenes:
            scenes["combined_features"] = []
        for instrument in grids:
           if len(scenes["combined_features"]) < scn+1:
               scenes["combined_features"].append(scenes[instrument][scn])
               for scn_sub in range(len(scenes[instrument][scn])):
                   scenes["combined_features"][scn][scn_sub] = scenes["combined_features"][scn][scn_sub].numpy()
           else:
               for scn_sub in range(len(scenes[instrument][scn])):
                   scenes["combined_features"][scn][scn_sub] = np.concatenate((scenes["combined_features"][scn][scn_sub], \
                       scenes[instrument][scn][scn_sub].numpy()), axis=2)

    
    print("Adding per-agent grid distances to feature set")
    movement_dfs = None
    df_uid = yml_conf["df_run_uid"]
    df_dir = yml_conf["df_dir"]

    with open(os.path.join(df_dir, df_uid + '_dfs.pkl'), "rb") as f:
        movement_dfs = pickle.load(f)

    for uid in movement_dfs:

        run_uid_dist = df_uid + "_" + uid
        with open(os.path.join(df_dir, run_uid_dist + "_distances.pkl"), "rb") as f:
            distances = pickle.load(f)


        print("Matching movement tracks to scenes and adding track-specific environment info for track", uid)
        movement_dfs_uid = movement_dfs[uid]
 
        if uid not in scenes_per_uid:
            scenes_per_uid[uid] = []

        for dind in range(len(movement_dfs_uid)):
            movement_df = movement_dfs_uid[dind]
            distance_grids = distances[dind]
            act_index = 0
            scene_ind = 0
        
            if len(scenes_per_uid[uid]) < dind+1:
                scenes_per_df = []
            else:
                scenes_per_df = scenes_per_uid[uid]

            for index, row in movement_df.iterrows():
                if len(row["date"]) > 10:
                    row["date"] = row["date"][:10]
                df_st_str = datetime.datetime.strptime(row['date'][0:10],"%Y-%m-%d")
                st_str = prelim_scene_map["times"][scene_ind]
                while st_str < df_st_str and scene_ind < len(prelim_scene_map["times"])-1:
                    scene_ind = scene_ind + 1
                    st_str = prelim_scene_map["times"][scene_ind]
                if st_str - df_st_str > datetime.timedelta(hours=15) or df_st_str - st_str > datetime.timedelta(hours=15):
                    if len(scenes_per_df) < act_index + 1:
                        scenes_per_df.append([])
                        continue
                if scene_ind >= len(prelim_scene_map["times"]):
                    break
                distance_grid = distance_grids[act_index]
                resample_shape = (scenes["combined_features"][scene_ind][0].shape[0], scenes["combined_features"][scene_ind][0].shape[1])
                distance_grid = cv2.resize(distance_grid, resample_shape, interpolation=cv2.INTER_CUBIC)
                distance_grid = np.reshape(distance_grid, (resample_shape[0], resample_shape[1], 1,1,1))
                new_scene = np.concatenate((scenes["combined_features"][scene_ind][-1], distance_grid), axis=2)

                print("HERE IN SCENE", uid, dind, scene_ind, act_index, df_st_str, st_str)

                new_full_scene = copy.deepcopy(scenes["combined_features"][scene_ind])
                new_full_scene[-1] = new_scene
 
                if len(scenes_per_df) < act_index + 1:
                    print("Appending new scene in ", uid, "at", act_index, "from", scene_ind)
                    scenes_per_df.append(new_full_scene)
                else:
                    print("Inserting scene in ", uid, "at", act_index, "from", scene_ind)
                    scenes_per_df[act_index] = new_full_scene

                for tmpp in range(len(scenes_per_df[act_index])):
                    if scenes_per_df[act_index] is not None and len(scenes_per_df[act_index]) > 0:
                        print("IN SCENE SHAPE", tmpp, new_scene.shape, len(scenes_per_df[act_index][tmpp]))

                act_index = act_index + 1

            print("Intermediate Scene Size", scenes["combined_features"][scene_ind][-1].shape, len(scenes["combined_features"][scene_ind]))

            if len(scenes_per_uid[uid]) < dind+1:
                print("Appending new DF for", uid, "at", dind)
                scenes_per_uid[uid].append(scenes_per_df)
            else:
                print("Inserting DF for", uid, "at", dind)
                scenes_per_uid[uid] = scenes_per_df

    return prelim_scene_map, grids, rsfns, scenes_per_uid


def run_surface_feature_connect_final(yml_conf, scenes_per_uid, clustering = None):

    msrffr = None
    _, msrffr = load_models(yml_conf["out_dir"], yml_conf["run_uid"], None) 

    print("Generating final feature combo")
    final_scenes_per_uid = {}

    counts = {}

    #TODO device management for encoders - later.
    device_check = False
    for uid in scenes_per_uid:

        counts[uid] = {"no_data" : 0, "data" : 0, "total" : 0}

        df_uid = yml_conf["df_run_uid"] + "_" + uid
        df_dir = yml_conf["df_dir"]
        with open(os.path.join(df_dir, df_uid + "_grid.pkl"), "rb") as f:
            final_grid = pickle.load(f) 

        final_scenes_per_uid[uid] = []
        for uid_ind in range(len(scenes_per_uid[uid])):
            final_scenes_per_uid[uid].append([])
            #if actual_total_chans > 10:
            if uid_ind == 0 and msrffr is None:
                msrffr = MultiSourceRSFeatureReduc(actual_total_chans) #TODO - incorporate. Currently not needed
 
            #if not device_check and torch.cuda.is_available(): 
            #   msrffr = msrffr.cuda()
            #   device_check = True

            #UID x Movement Stream x Time Index
            for scn_ind in range(len(scenes_per_uid[uid][uid_ind])):
                final_scenes_per_uid[uid][uid_ind].append([])
                tmp = None 
                for feat_ind in range(len(scenes_per_uid[uid][uid_ind][scn_ind])):

                    if scenes_per_uid[uid][uid_ind][scn_ind][feat_ind].ndim == 5:
                        scenes_per_uid[uid][uid_ind][scn_ind][feat_ind] = np.mean(scenes_per_uid[uid][uid_ind][scn_ind][feat_ind], axis=(-2,-1))

                    if tmp is None:
                        tmp = scenes_per_uid[uid][uid_ind][scn_ind][feat_ind]
                    else:
                        tmp = np.concatenate((tmp, scenes_per_uid[uid][uid_ind][scn_ind][feat_ind]), axis=2)
 

                tmp2 = None 
                if tmp is not None: 
                    tmp2 = np.zeros((final_grid.lat_tiles, final_grid.lon_tiles, tmp.shape[2]))
                    for chn in range(tmp2.shape[2]):
                        tmp2[:,:,chn] = cv2.resize(tmp[:,:,chn], (final_grid.lon_tiles, final_grid.lat_tiles), interpolation=cv2.INTER_CUBIC)
                    del tmp
                 
                if tmp2 is None:
                    counts[uid]["no_data"] = counts[uid]["no_data"] + 1
                else:
                    counts[uid]["data"] = counts[uid]["data"] + 1
                counts[uid]["total"] = counts[uid]["total"] + 1

                final_scenes_per_uid[uid][uid_ind][scn_ind] = tmp2

    pprint(counts)                
       
    print("Generating final simplified (via clustering) scene representation")
    if clustering is None:
        print("Running cluster training")
        #Sample first 10 samples from each stream
        training_grids = None
        for uid in final_scenes_per_uid:
            for uid_ind in range(len(final_scenes_per_uid[uid])):
                final_ind = min(10, len(final_scenes_per_uid[uid][uid_ind]))
                grid_sub = final_scenes_per_uid[uid][uid_ind][:final_ind]
                extend = False
                for sub_ind in range(final_ind):
                    if grid_sub[sub_ind] is None:
                        continue
                    extend = True
                    grid_sub[sub_ind] = np.reshape(grid_sub[sub_ind],\
                        shape=(grid_sub[sub_ind].shape[0]*grid_sub[sub_ind].shape[1],\
                        grid_sub[sub_ind].shape[2]))
                    if extend:
                        if training_grids is None:
                            training_grids = grid_sub[sub_ind]
                        else:
                            training_grids = np.concatenate((training_grids, grid_sub[sub_ind]), axis=0)
        clustering = MiniBatchKMeans(n_clusters=yml_conf["n_clusters"], max_iter=500, batch_size=32) #TODO parameterize via config
        clustering.fit(training_grids)

    for uid in final_scenes_per_uid:
        for uid_ind in range(len(final_scenes_per_uid[uid])):
            for scene_ind in range(len(final_scenes_per_uid[uid][uid_ind])):
                grid_tmp = final_scenes_per_uid[uid][uid_ind][scene_ind]
                if grid_tmp is None:
                    continue
                init_shape = grid_tmp.shape
                grid_tmp = grid_tmp.reshape((grid_tmp.shape[0]*grid_tmp.shape[1], grid_tmp.shape[2]))
                grid_tmp = clustering.predict(grid_tmp)
                grid_tmp = grid_tmp.reshape((init_shape[0], init_shape[1]))
                final_scenes_per_uid[uid][uid_ind][scene_ind] = grid_tmp

    return msrffr, final_scenes_per_uid, clustering
  

def save_models(out_dir, run_uid, rsfns=None, msrffr=None):

    if rsfns is not None:
        model_dict = {}
        for instrument in rsfns.keys():
            state_dict = rsfns[instrument].state_dict()
            in_chans = rsfns[instrument].in_chans
            tile_size = rsfns[instrument].tile_size

            model_dict[instrument] = {"weights" : state_dict, \
                "in_chans" : in_chans, "tile_size" : tile_size}
        torch.save(model_dict, os.path.join(out_dir, run_uid + "rsfns.ckpt"))

    if msrffr is not None:
        model_dict = {"weights" : msrffr.state_dict(), "in_chans" : msrffr.in_chans}
        torch.save(model_dict, os.path.join(out_dir, run_uid + "msrffr.ckpt"))

def load_models(out_dir, run_uid, stats=None):

    rsfns = None

    fname = os.path.join(out_dir, run_uid + "rsfns.ckpt")
    if os.path.exists(fname) and stats is not None:
        rsfns_init = torch.load(fname)

        rsfns = {}
        for instrument in rsfns_init.keys():
            rsfns[instrument] = RSFeatureNet(rsfns_init[instrument]["in_chans"],\
                stats[instrument]["mean"], stats[instrument]["std"])
            rsfns[instrument].load_state_dict(rsfns_init["in_chans"])
            rsfns[instrument].eval()

    msrffr = None

    fname = os.path.join(out_dir, run_uid + "msrffr.ckpt")
    if os.path.exists(fname):
        msrffr_init = torch.load(fname)

        msrffr = MultiSourceRSFeatureReduc(msrffr_init["in_chans"]) 

        msrffr.load_state_dict(msrffr_init["weights"])
        msrffr.eval()

    return rsfns, msrffr
    
def build_and_save_envs(yml_fpath):

    #Translate config to dictionary 
    yml_conf = read_yaml(yml_fpath)


    scenes_per_uid = {}
    if "scenes_per_uid" in yml_conf and yml_conf["scenes_per_uid"]:
        pkl_file = os.path.join(yml_conf["out_dir"], "env_maps_" + yml_conf["run_uid"] + ".pkl")
        with open(pkl_file, 'rb') as f:
            scenes_per_uid = pickle.load(f)
 

    rsfns = None
    msrffr = None 
    if yml_conf["run_env_gen"]:
        print("Generating surface features")
        prelim_scene_map, grids, rsfns, scenes_per_uid  = run_surface_feature_connect(yml_conf, scenes_per_uid)

        pkl_file = os.path.join(yml_conf["out_dir"], "env_maps_" + yml_conf["run_uid"] + ".pkl")
        with open(pkl_file, 'wb') as f:
            pickle.dump(scenes_per_uid, f, protocol=pickle.HIGHEST_PROTOCOL)

    if yml_conf["run_final_env_gen"]:

        save_cluster = True
        clustering = None
        if os.path.exists(os.path.join(yml_conf["out_dir"], "clust_env.joblib")):
            clustering = joblib.load(os.path.join(yml_conf["out_dir"], "clust_env.joblib"))

        msrffr, final_scenes_per_uid, clustering = run_surface_feature_connect_final(yml_conf, scenes_per_uid, clustering)
 
        pkl_file = os.path.join(yml_conf["out_dir"], "final_env_maps_" + yml_conf["run_uid"] + ".pkl")
        with open(pkl_file, 'wb') as f:
            pickle.dump(final_scenes_per_uid, f, protocol=pickle.HIGHEST_PROTOCOL)    
      
        if save_cluster:
            #Save clustering model
            clust_fname = os.path.join(yml_conf["out_dir"], "clust_env.joblib")
            joblib.dump(clustering, clust_fname)


    #TODO no need to re-save models - check fu

    print("Saving model weights")
    save_models(yml_conf["out_dir"], yml_conf["run_uid"], rsfns, msrffr)

 
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for fusion info.")
    args = parser.parse_args()

    build_and_save_envs(args.yaml)





