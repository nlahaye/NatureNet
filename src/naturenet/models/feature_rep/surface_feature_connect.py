
from naturenet.models.feature_rep.rs_feature_reduce import MultiSourceRSFeatureReduc

from sit_fuse.pipelines.inference.inference_utils import run_embed_gen
from sit_fuse.utils import read_yaml
import os
import yaml

def run_surface_feature_connect(yml_fpath):

    #Translate config to dictionary 
    yml_conf = read_yaml(yml_fpath)
    #Run 
    run_basic_inference_geolocation(yml_conf)
    rsfns = {}

    scenes ={}

    scene_count = -1
    for key in yml_conf["instruments"].keys():
        if scene_count < 1:
            scene_count = len(yml_conf["instruments"][key]["filenames"])
        if yml_conf["instruments"][key]["multi_chan"]:
            total_chans += 5
        else:
            total_chans += 1

    msrffr = None
    actual_total_chans = 0
    #TODO parallelize
    for i in range(scene_count):
        for instrument in yml_conf["instruments"]:
            encoder_conf_fpath = yml_conf["instruments"][instrument]["encoder_conf"]
            encoder_conf = read_yaml(encoder_conf_fpath)
            embed, _ = run_embed_gen(encoder_conf, yml_conf["instruments"][instrument]["filenames"][i], gen_image_shaped = True)
            graph_max_scale = 1000 #TODO
            if instrument not in rsfns:
                rsfns[instrument] = RSFeatureNet(embed.shape[1], graph_max_scale)
                scenes[instrument] = []
                if "combined" not in scenes:
                    scenes[instrument]["combined"] = None
            scenes[instrument].append(rsfns[instrument](embed))
            if i == 0:
                actual_total_chans += rsfns[instrument].out_chans
            if len(scenes["combined"]) < 1:
                scenes["combined"] = embed
            else:
                scenes["combined"] = np.concatenate((scenes["combined"], embed), axis=1)
        if actual_total_chans > 10:
            if i == 0:
                msrffr = MultiSourceRSFeatureReduc(actual_total_chans)
                scenes["combined_final"] = msrffr(scenes["combined"])
            else:
                scenes["combined_final"] = np.concatenate((scenes["combined_final"], msrffr(scenes["combined"])),  axis=1)
        else:
            if i == 0:
                scenes["combined_final"] = scenes["combined"]
            else:
                scenes["combined_final"] = np.concatenate((scenes["combined_final"], scenes["combined_final"]),  axis=1)

    return scenes, rsfns, msrffr

    

        

