
import zarr
import numpy as np
import os
import copy
from datetime import datetime
import re
from sit_fuse.pipelines.inference.inference_utils import run_embed_gen_from_scene_arr
from sit_fuse.preprocessing.colocate_and_resample import resample_scene
from sit_fuse.datasets.dataset_utils import get_scenes
from sit_fuse.utils import read_gtiff_generic, read_gtiff_generic_geo, read_yaml


def resample_or_fuse_scene(scene, init_location, resample_config):

    resample_config["low_res"]["filenames"] = scene
    resample_config["low_res"]["geo_filenames"] = scene


    resample_config["return_products"] = True
    output, location = resample_scene(scene, init_location, resample_config)
    return output, location

def gen_copernicus_scenes_map(): #yml_conf,  per_channel_stats):


    data_dict = {


        "0.0009000090001" : [
            {"fnames": ["/data/nlahaye/NatureNet_Env/Copernicus_Ter/Results/EMC_BUILT/EMC_BUILT_GREENNESS_E2022_GLOBE_R2025A_54009_10_V1_0_R4_C20.tif.clipped.tif"], "key":"built_g"},
            {"fnames": ["/data/nlahaye/NatureNet_Env/Copernicus_Ter/Results/EMC_BUILT/EMC_BUILT_S_E2022_GLOBE_R2025A_54009_10_V1_0_R4_C20.tif.clipped.tif"], "key":"built_s"},
            #{"fnames": ["/data/nlahaye/NatureNet_Env/Copernicus_Ter/Results/TCD/CLMS_HRLVLCC_TCD_S2021_R10m_E48N28_03035_V01_R00.tif.clipped.tif"], "key":"tcd"},
            {"fnames": ["/data/nlahaye/NatureNet_Env/Copernicus_Ter/Results/CLCplus_RASTER_2021_010m_03035/CLCplus_RASTER_2021_010m_03035/CLCplus_RASTER_2021_010m_03035.tif.clipped.tif"], "key":"clc"}
        ], 

        #"0.000665206898853" : [
        #    {"fnames": [ "/data/nlahaye/NatureNet_Env/Copernicus_Ter/Results/GFSC/GFSC_20210901_GF.tif",
        #                 "/data/nlahaye/NatureNet_Env/Copernicus_Ter/Results/GFSC/GFSC_20210902_GF.tif",
        #                 "/data/nlahaye/NatureNet_Env/Copernicus_Ter/Results/GFSC/GFSC_20210903_GF.tif",
        #                 "/data/nlahaye/NatureNet_Env/Copernicus_Ter/Results/GFSC/GFSC_20210904_GF.tif",
        #                 "/data/nlahaye/NatureNet_Env/Copernicus_Ter/Results/GFSC/GFSC_20210905_GF.tif",
        #                 "/data/nlahaye/NatureNet_Env/Copernicus_Ter/Results/GFSC/GFSC_20210906_GF.tif",
        #                 "/data/nlahaye/NatureNet_Env/Copernicus_Ter/Results/GFSC/GFSC_20210907_GF.tif",
        #                 "/data/nlahaye/NatureNet_Env/Copernicus_Ter/Results/GFSC/GFSC_20210908_GF.tif",
        #                 "/data/nlahaye/NatureNet_Env/Copernicus_Ter/Results/GFSC/GFSC_20210909_GF.tif",
        #                 "/data/nlahaye/NatureNet_Env/Copernicus_Ter/Results/GFSC/GFSC_20210910_GF.tif",
        #                 "/data/nlahaye/NatureNet_Env/Copernicus_Ter/Results/GFSC/GFSC_20210911_GF.tif",
        #                 "/data/nlahaye/NatureNet_Env/Copernicus_Ter/Results/GFSC/GFSC_20210912_GF.tif",
        #                 "/data/nlahaye/NatureNet_Env/Copernicus_Ter/Results/GFSC/GFSC_20210913_GF.tif",
        #                 "/data/nlahaye/NatureNet_Env/Copernicus_Ter/Results/GFSC/GFSC_20210914_GF.tif",
        #                 "/data/nlahaye/NatureNet_Env/Copernicus_Ter/Results/GFSC/GFSC_20210915_GF.tif",
        #                 "/data/nlahaye/NatureNet_Env/Copernicus_Ter/Results/GFSC/GFSC_20210916_GF.tif",
        #                 "/data/nlahaye/NatureNet_Env/Copernicus_Ter/Results/GFSC/GFSC_20210917_GF.tif",
        #                 "/data/nlahaye/NatureNet_Env/Copernicus_Ter/Results/GFSC/GFSC_20210918_GF.tif",
        #                 "/data/nlahaye/NatureNet_Env/Copernicus_Ter/Results/GFSC/GFSC_20210919_GF.tif",
        #                 "/data/nlahaye/NatureNet_Env/Copernicus_Ter/Results/GFSC/GFSC_20210920_GF.tif",
        #                 "/data/nlahaye/NatureNet_Env/Copernicus_Ter/Results/GFSC/GFSC_20210921_GF.tif",
        #                 "/data/nlahaye/NatureNet_Env/Copernicus_Ter/Results/GFSC/GFSC_20210922_GF.tif",
        #                 "/data/nlahaye/NatureNet_Env/Copernicus_Ter/Results/GFSC/GFSC_20210923_GF.tif",
        #                 "/data/nlahaye/NatureNet_Env/Copernicus_Ter/Results/GFSC/GFSC_20210924_GF.tif",
        #                 "/data/nlahaye/NatureNet_Env/Copernicus_Ter/Results/GFSC/GFSC_20210925_GF.tif",
        #                 "/data/nlahaye/NatureNet_Env/Copernicus_Ter/Results/GFSC/GFSC_20210926_GF.tif",
        #                 "/data/nlahaye/NatureNet_Env/Copernicus_Ter/Results/GFSC/GFSC_20210927_GF.tif",
        #                 "/data/nlahaye/NatureNet_Env/Copernicus_Ter/Results/GFSC/GFSC_20210928_GF.tif",
        #                 "/data/nlahaye/NatureNet_Env/Copernicus_Ter/Results/GFSC/GFSC_20210929_GF.tif",
        #                 "/data/nlahaye/NatureNet_Env/Copernicus_Ter/Results/GFSC/GFSC_20210930_GF.tif",
        #                 "/data/nlahaye/NatureNet_Env/Copernicus_Ter/Results/GFSC/GFSC_20211001_GF.tif",
        #                 "/data/nlahaye/NatureNet_Env/Copernicus_Ter/Results/GFSC/GFSC_20211002_GF.tif",
        #                 "/data/nlahaye/NatureNet_Env/Copernicus_Ter/Results/GFSC/GFSC_20211003_GF.tif",
        #                 "/data/nlahaye/NatureNet_Env/Copernicus_Ter/Results/GFSC/GFSC_20211004_GF.tif",
        #                 "/data/nlahaye/NatureNet_Env/Copernicus_Ter/Results/GFSC/GFSC_20211005_GF.tif",
        #                 "/data/nlahaye/NatureNet_Env/Copernicus_Ter/Results/GFSC/GFSC_20211006_GF.tif",
        #                 "/data/nlahaye/NatureNet_Env/Copernicus_Ter/Results/GFSC/GFSC_20211007_GF.tif",
        #                 "/data/nlahaye/NatureNet_Env/Copernicus_Ter/Results/GFSC/GFSC_20211008_GF.tif",
        #                 "/data/nlahaye/NatureNet_Env/Copernicus_Ter/Results/GFSC/GFSC_20211009_GF.tif",
        #                 "/data/nlahaye/NatureNet_Env/Copernicus_Ter/Results/GFSC/GFSC_20211010_GF.tif",
        #                 "/data/nlahaye/NatureNet_Env/Copernicus_Ter/Results/GFSC/GFSC_20211011_GF.tif",
        #                 "/data/nlahaye/NatureNet_Env/Copernicus_Ter/Results/GFSC/GFSC_20211012_GF.tif",
        #                 "/data/nlahaye/NatureNet_Env/Copernicus_Ter/Results/GFSC/GFSC_20211013_GF.tif",
        #                 "/data/nlahaye/NatureNet_Env/Copernicus_Ter/Results/GFSC/GFSC_20211014_GF.tif",
        #                 "/data/nlahaye/NatureNet_Env/Copernicus_Ter/Results/GFSC/GFSC_20211015_GF.tif",
        #                 "/data/nlahaye/NatureNet_Env/Copernicus_Ter/Results/GFSC/GFSC_20211016_GF.tif",
        #                 "/data/nlahaye/NatureNet_Env/Copernicus_Ter/Results/GFSC/GFSC_20211017_GF.tif",
        #                 "/data/nlahaye/NatureNet_Env/Copernicus_Ter/Results/GFSC/GFSC_20211018_GF.tif",
        #                 "/data/nlahaye/NatureNet_Env/Copernicus_Ter/Results/GFSC/GFSC_20211019_GF.tif",
        #                 "/data/nlahaye/NatureNet_Env/Copernicus_Ter/Results/GFSC/GFSC_20211020_GF.tif",
        #                 "/data/nlahaye/NatureNet_Env/Copernicus_Ter/Results/GFSC/GFSC_20211021_GF.tif",
        #                 "/data/nlahaye/NatureNet_Env/Copernicus_Ter/Results/GFSC/GFSC_20211022_GF.tif",
        #                 "/data/nlahaye/NatureNet_Env/Copernicus_Ter/Results/GFSC/GFSC_20211023_GF.tif",
        #                 "/data/nlahaye/NatureNet_Env/Copernicus_Ter/Results/GFSC/GFSC_20211024_GF.tif",
        #                 "/data/nlahaye/NatureNet_Env/Copernicus_Ter/Results/GFSC/GFSC_20211025_GF.tif",
        #                 "/data/nlahaye/NatureNet_Env/Copernicus_Ter/Results/GFSC/GFSC_20211026_GF.tif",
        #                 "/data/nlahaye/NatureNet_Env/Copernicus_Ter/Results/GFSC/GFSC_20211027_GF.tif",
        #                 "/data/nlahaye/NatureNet_Env/Copernicus_Ter/Results/GFSC/GFSC_20211028_GF.tif",
        #                 "/data/nlahaye/NatureNet_Env/Copernicus_Ter/Results/GFSC/GFSC_20211029_GF.tif",
        #                 "/data/nlahaye/NatureNet_Env/Copernicus_Ter/Results/GFSC/GFSC_20211030_GF.tif",
        #                 "/data/nlahaye/NatureNet_Env/Copernicus_Ter/Results/GFSC/GFSC_20211031_GF.tif"],
        #     "date_re": "(\d{8})",
        #     "strptime_fmt": "%Y%m%d", "key":"gfsc"}
        #],

        "0.002700027" : [
            {"fnames": ["/data/nlahaye/NatureNet_Env/Copernicus_Ter/Results/HLS_L30/HLS_SCENE.tif.clipped.tif"], "key":"hls"}
        ],

        "0.009000090001" : [
            {"fnames": ["/data/nlahaye/NatureNet_Env/Copernicus_Ter/Results/GHS_POP/GHS_POP_E2020_GLOBE_R2023A_4326_3ss_V1_0_R5_C20.tif.clipped.tif"], "key":"pop"}
        ], 
      
 
       #"0.003852536424896" : [
       #     {"fnames": ["/data/nlahaye/NatureNet_Env/Copernicus_Ter/Results/NDVI/NDVI_202109.tif", "/data/nlahaye/NatureNet_Env/Copernicus_Ter/Results/NDVI/NDVI_202110.tif"],
       #      "date_re": "(\d{6})",
       #      "strptime_fmt": "%Y%m", "key":"ndvi"} 
       #],

 
        "0.09000090001" : [
            {"fnames": ["/data/nlahaye/NatureNet_Env/Copernicus_Ter/Results/GHS_SMOD/GHS_SMOD_E2020_GLOBE_R2023A_54009_1000_V2_0_R4_C20.tif.clipped.tif"], "key":"smod"},
        ],


    }

    ##For each resolution
    ##For each file
    ##Read vars - take top depth
    #split by time
    ##stack channels

    ##Get lat/lon once
    ##longr, latgr = np.meshgrid(data1[1], data1[0])
    ##data1 = np.array([longr, latgr]).astype(np.float32)

    #Get times

    ##Read mask
    ##Resample
    ##Mask land

    
    resample_config = read_yaml("../../config/environment/cop_ter_resample_config.yaml")

    fill = -999999
 
    #data[:,inds[0], inds[1]] = fill

    final_dict = {}

    dat_final = {}
    lonlat = None
    for key in data_dict.keys():
        print("KEY", key)
        dat1 = None
        lonlat = None
        for fn_dict in data_dict[key]:
            dat = None
            tm = []
            for f in range(len(fn_dict["fnames"])):
 
                if "date_re" in fn_dict:
                    mtch = re.search(fn_dict["date_re"], fn_dict["fnames"][f])
                    tm_str = mtch.group(1)
                    tm_dt = datetime.strptime(tm_str, fn_dict["strptime_fmt"]).isoformat() 
                    tm.append(tm_dt)
                else:
                    tm.append(None)
            
                #Some tiles are slightly differently sized - have to resample prior to stacking instead of vice versa
                print(fn_dict["fnames"][f])
                if dat is None:
                    dat = read_gtiff_generic(fn_dict["fnames"][f])
                    if "HLS" in fn_dict["fnames"][f]:
                        dat[np.where((dat < -99) | (dat > 32767))] =  -999999.0
                    elif "TCD" in fn_dict["fnames"][f]:
                        dat[np.where(dat > 250)] =  -999999.0
                    if dat.ndim < 3:
                        dat = np.expand_dims(dat, 0)
                    lonlat = read_gtiff_generic_geo(fn_dict["fnames"][f])
                    if "TCD" in fn_dict["fnames"][f] or "EMC_BUILT" in fn_dict["fnames"][f] or "HLS" in fn_dict["fnames"][f] or "SMOD" in fn_dict["fnames"][f]:
                        resample_config["low_res"]["data"]["geo_lat_index"] = 1
                        resample_config["low_res"]["data"]["geo_lon_index"] = 0
                    else:
                        resample_config["low_res"]["data"]["geo_lat_index"] = 0
                        resample_config["low_res"]["data"]["geo_lon_index"] = 1
                    #print(lonlat)
                    dat, location = resample_or_fuse_scene(dat, lonlat,  resample_config)
                else:
                    tmp = read_gtiff_generic(fn_dict["fnames"][f])
                    #tmp[np.where((tmp < -99)  | (tmp > 32767))] =  -999999.0
                    if tmp.ndim < 3:
                        tmp = np.expand_dims(tmp, 0)
                    if "HLS" in fn_dict["fnames"][f]:
                        tmp[np.where((tmp < -99) | (tmp > 32767))] =  -999999.0
                    elif "TCD" in fn_dict["fnames"][f]:
                        tmp[np.where(tmp > 250)] =  -999999.0
                    lonlat = read_gtiff_generic_geo(fn_dict["fnames"][f])
                    #print(lonlat)
                    if "TCD" in fn_dict["fnames"][f] or "EMC_BUILT" in fn_dict["fnames"][f] or "HLS" in fn_dict["fnames"][f] or "SMOD" in fn_dict["fnames"][f]:
                        resample_config["low_res"]["data"]["geo_lat_index"] = 1
                        resample_config["low_res"]["data"]["geo_lon_index"] = 0
                    else:
                        resample_config["low_res"]["data"]["geo_lat_index"] = 0
                        resample_config["low_res"]["data"]["geo_lon_index"] = 1
                    tmp, _ = resample_or_fuse_scene(tmp, lonlat, resample_config)
                    print(dat.shape, tmp.shape)
                    dat = np.concatenate((dat, tmp), axis=0)
 

            #T x Lat x Lon
            print(dat.shape, lonlat.shape, "HERE HERE")

            del lonlat
            lonlat = None

            #if dat.ndim < 4:
            #    dat = np.expand_dims(dat, axis=0)  #T x C x Lat x Lon
            print("HERE", fn_dict["key"])
            dat_final[fn_dict["key"]] = {"data" : copy.deepcopy(dat), "t" : tm}

            print("HERE", dat.min(), dat.max(), dat.mean(), dat.std())
            del dat, location
   
    np.savez("/data/nlahaye/NatureNet_Env/Copernicus_Ter/cop_env_boar.npz", dat_final)
    #zarr.save("/data/nlahaye/NatureNet_Env/Copernicus/cop_env_boar.zarr", dat_final)

    
    single_scene_keys = ["hls", "pop", "clc", "smod", "built_g", "built_s"] #tcd
    month_scene_keys = [] #"ndvi"]
    #main_key = "gfsc"
    stacked_scenes = []
    month_ind = 0

    stacked_scene = None
    for ky1 in range(len(single_scene_keys)):
        ky = single_scene_keys[ky1]
        if stacked_scene is None:
                stacked_scene = dat_final[ky]["data"]
        else:
                stacked_scene = np.concatenate((stacked_scene, dat_final[ky]["data"]), axis=0)

    for i in range(stacked_scene.shape[0]):
        print(stacked_scene[i].min(), stacked_scene[i].mean(), stacked_scene[i].max(), i)

    print(stacked_scene.shape)
    for i in range(62):

        """
        for i in range(len(dat_final[main_key]["data"])):
        stacked_scene = dat_final[main_key]["data"][i]
        tm = dat_final[main_key]["t"][i]
        for ky1 in range(len(single_scene_keys)):
            ky = single_scene_keys[ky1]
            stacked_scene = np.concatenate((stacked_scene, dat_final[ky]["data"][0]), axis=0)
        for ky1 in range(len(month_scene_keys)):
            ky = single_scene_keys[ky1]
            if month_ind < len(dat_final[ky]["data"]) -1:
                while month_ind < len(dat_final[ky]["data"]) -1 and dat_final[ky]["t"][month_ind] < tm:
                    month_ind = month_ind + 1
                if dat_final[ky]["t"][month_ind] > tm:
                    month_ind = month_ind - 1
            stacked_scene = np.concatenate((stacked_scene, dat_final[ky]["data"][month_ind]), axis=0)
            print(stacked_scene.shape)
        """   

        zarr.save("/data/nlahaye/NatureNet_Env/Copernicus_Ter/env_scenes/cop_env_boar_" + str(i) + ".zarr", stacked_scene)

 


gen_copernicus_scenes_map()


