
from naturenet.models.feature_rep.rs_feature_reduce import MultiSourceRSFeatureReduc

from sit_fuse.pipelines.inference.inference_utils import run_embed_gen_from_scene_arr
from sit_fuse.preprocessing.colocate_and_resample import resample_scene
from sit_fuse.datasets.dataset_utils import get_scenes
from sit_fuse.utils import read_yaml

import os
import yaml
import argparse
from datetime import datetime
import numpy as np

def run_normalization_stats(yml_conf, stats = {}):
    data = None
    for key in yml_conf["instruments"].keys():
        for i in range(len(yml_conf["instruments"][key]["filenames"])):
            data_config = read_yaml(yml_conf["instruments"][key]["data_config"])
            dat_tmp, _, _ = get_scenes(data_config, yml_conf["instruments"][key]["filenames"][i])
            dat_tmp = np.array(dat_tmp)            
 
            if dat_tmp.ndim < 4: #Likely no channel or sample dimension
                if dat_tmp.ndim == 3:
                    dat_tmp = np.expand_dims(dat_tmp, 1)
                else:
                    while dat_tmp.ndim < 4:
                        dat_tmp = np.expand_dims(dat_tmp, 0)
            if data is None:
                data = dat_tmp
            else:
                data = np.concatenate((data, dat_tmp), axis=0)    

        inds = np.where(data < -99990.0)
        data[inds] = np.nan 
        mean = np.nanmean(data, axis=(0,2,3)) 
        std = np.nanstd(data, axis=(0,2,3))

        if key not in stats:
            stats[key] = {"mean" : mean, "std" : std}
        else:
            stats[key]["mean"] = mean
            stats[key]["std"] = std

    return stats


def resample_or_fuse_scene(scene, init_location, resample_config):

    resample_config["low_res"]["filenames"] = scene
    resample_config["low_res"]["geo_filenames"] = scene
    
    print("HERE", scene)

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
        for i in range(0,final_width,final_grid_steps):
            center_col = int(i + (final_grid_steps/2))

            for j in range(0,final_height,final_grid_steps):
                center_row = int(j + (final_grid_steps/2))

                center_lon = location[j,center_col,0] + (current_grid_res_deg*(final_grid_steps/2.0))
                center_lat = location[center_row,i,1] + (current_grid_res_deg*(final_grid_steps/2.0))
              
                final_grid_coords[j,i,0] = center_lon
                final_grid_coords[j,i,1] = center_lat

    return final_data, final_loc, final_grid_steps, final_grid_coords





def get_datetime_info(yml_conf):

    #For now, assume uniform time distribution across inputs, taken from input config
    #TODO - automate and deepen logic here

    dts = yml_conf["datetimes"]
    ret_dts = []
  
    for i in range(len(dts)):
        ret_dts.append(datetime.strptime(dts[i], "%d%m%YT%H:%M:%SZ"))



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
 

            print(len(scene), len(init_location))
            #For now all scene sets should be of size 1, and in the future stitching will enforce this in other cases
            scene = scene[0]
            init_location = init_location[0]

            print("Resampling", instrument, i)
            #Resample over grid - TODO I will need to add stitching prior to this for inputs in future iterations
            #Process will be stitch, resample, impute
            data, location = resample_or_fuse_scene(scene, init_location,  resample_config)
            del scene

            print(len(data))

            print("Post-resample data imputation", instrument, i)
            #Impute across fill values for now - may want to change approach later on #TODO - revisit
            for ch in range(len(data[0])):
                subd = data[ch]
                inds = np.where(subd < -99990)
                subd[inds] = per_channel_stats[instrument]["mean"]
                data[ch] = subd


            print("Generating grid info", instrument, i)
            if instrument not in prelim_scene_map:
                prelim_scene_map[instrument] = {"scenes" : []} 
            #Generate inital grid info per-scene
            if final_grid not in prelim_scene_map:
                #Can ignore resampled lat/lon for now - may be useful later
                final_data, final_loc, final_grid_steps, final_grid_coords = gen_grid_info(location, data, yml_conf["final_grid_res_deg"], per_channel_stats, compute_final_grid_info = True) 
                #tile size will vary relative to native resolution
                prelim_scene_map["final_grid"] = final_grid_coords
                prelim_scene_map[instrument]["tile_size"] = final_grid_steps
            else:
                final_data, final_loc, final_grid_steps, _ = gen_grid_info(location, data, yml_conf["final_grid_res_deg"], per_channel_stats, compute_final_grid_info = False)
            prelim_scene_map[instrument].append(final_data)

 
            print("Associating times to grid", instrument, i)
            if "times" not in prelim_scene_map:
                #Get time info connected to scene
                prelim_scene_map["times"] = get_datetime_info(yml_conf)
    return prelim_scene_map
    

def run_surface_feature_connect(yml_conf):

    npz_file = os.path.join(yml_conf["out_dir"], "input_per_channel_stats.npz")
    if os.path.exists(npz_file):
        per_channel_stats = dict(np.load(npz_file, allow_pickle=True))
        
        if yml_conf["update_stats"]:
            print("Generating normalization stats")
            per_channel_stats = run_normalization_stats(yml_conf, per_channel_stats)
            np.savez(npz_file, **per_channel_stats)
        else:
            print("Loaded normalization stats") 
    else:
        print("Generating normalization stats")
        per_channel_stats = run_normalization_stats(yml_conf)
        np.savez(npz_file, **per_channel_stats)

    rsfns = {}

    scenes ={}
    grids = {}
    grid_size = None


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
                embed, _ = run_embed_gen_from_scene_arr(encoder_conf, [prelim_scene_map[instrument]["scenes"][i]], [prelim_scene_map[instrument]["scenes"][i].shape[0:2]], gen_image_shaped = True)
            else:
                embed = prelim_scene_map[instrument]["scenes"][i]

            if embed.ndim < 4: #Likely no channel or sample dimension
                if embed.ndim == 3:
                    embed = np.expand_dims(embed, 1)
                else:
                    while embed.ndim < 4:
                        embed = np.expand_dims(embed, 0)
                
            print("Extracting multi-scale coarser features", i)
            if instrument not in rsfns: #Assuming N_Samples X N_Channels X YDIM X XDIM
                rsfns[instrument] = RSFeatureNet(embed.shape[1], prelim_scene_map[instrument]["tile_size"], per_channel_stats[instrument]["mean"], per_channel_stats[instrument]["std"])
                scenes[instrument] = []
                if "combined" not in scenes:
                    scenes["combined"] = []
                    scenes["combined_final"] = []
            x1, x2, x3, grid_sz = rsfns[instrument](embed)
            if grid_size is None:
                grid_size = grid_sz
 
            if x1.ndim == 3:
                x1 = x1.reshape((grid_size[0], grid_size[1], x1.shape[1], x1.shape[2],1))
                x2 = x2.reshape((grid_size[0], grid_size[1], x2.shape[1], x2.shape[2],1))
                x3 = x3.reshape((grid_size[0], grid_size[1], x3.shape[1], x3.shape[2],1))
            elif x1.ndim == 3:
                x1 = x1.reshape((grid_size[0], grid_size[1], x1.shape[1], x1.shape[2], x1.shape[3]))
                x2 = x2.reshape((grid_size[0], grid_size[1], x2.shape[1], x2.shape[2], x2.shape[3]))
                x3 = x3.reshape((grid_size[0], grid_size[1], x3.shape[1], x3.shape[2], x3.shape[3]))

            grids[instrument].append(grid_size)
            scenes[instrument].append([x1, x2, x3])
            if i == 0:
                actual_total_chans += rsfns[instrument].out_chans
            if len(scenes["combined"]) < (i+1):
                scenes["combined"].append(embed)
            else:
                scenes["combined"][i] = np.concatenate((scenes["combined"][i], embed), axis=4)

 
    print("Generating final feature combo")
    for i in range(scene_count):    
        if actual_total_chans > 10:
            if i == 0:
                msrffr = MultiSourceRSFeatureReduc(actual_total_chans, stats[instrument]["mean"], stats[instrument]["std"])
            if len(scenes["combined_final"])  < (i+1): 
                scenes["combined_final"].append(msrffr(scenes["combined"][i]))
            else:
                scenes["combined_final"][i] = np.concatenate((scenes["combined_final"][i], msrffr(scenes["combined"][i])),  axis=4)
        else:
            if len(scenes["combined_final"])  < (i+1):
                scenes["combined_final"].append(scenes["combined"][i])
            else:
                scenes["combined_final"][i] = np.concatenate((scenes["combined_final"][i], scenes["combined_final"][i]),  axis=4)


    return prelim_scene_map, scenes, grids, rsfns, msrffr

    
def build_and_save_envs(yml_fpath):

    #Translate config to dictionary 
    yml_conf = read_yaml(yml_fpath)
          
    print("Generating surface features")
    prelim_scene_map, scenes, grids, rsfns, msrffr = run_surface_feature_connect(yml_conf)

    print("Saving surface features")
    if yml_conf["save_prelim_scene_maps"]:
        npz_file = os.path.join(yml_conf["out_dir"], "prelim_scene_maps_" + yml_conf["run_uid"] + ".npz")
        np.savez(npz_file, **prelim_scene_map)

    if yml_conf["save_intermediate_maps"]:
        npz_file = os.path.join(yml_conf["out_dir"], "intermediate_state_maps_" + yml_conf["run_uid"] + ".npz")
        np.savez(npz_file, **scenes)

    final_data = {"environment" : scenes["combined_final"], "grid" : prelim_scene_map["final_grid"], "times" : prelim_scene_map["times"]} 
    npz_file = os.path.join(yml_conf["out_dir"], "final_env_maps_" + yml_conf["run_uid"] + ".npz")
    np.savez(npz_file, **final_data)


    print("Saving model weights")
    weights = {}
    for inst in rsfns:
        weights[inst] = rsfns[inst].state_dict()
    if msrffr is not None:
        weights["msrffr"] =  msrffr.state_dict()

 
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for fusion info.")
    args = parser.parse_args()

    build_and_save_envs(args.yaml)





