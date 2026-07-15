import pickle
import os
from datetime import datetime
import zarr
import sys
import numpy as np
from osgeo import gdal
import glob

def update_clusters():
    times = []
    scenes = []

    final_env_maps = []

    df_dir = "/data/nlahaye/NatureNet/Blue_Whale_v1/"
    df_uid = "whale_v1"
    out_dir = "/data/nlahaye/NatureNet/Blue_Whale_v1/"

    lst = None
    tme = datetime.strptime("2017-07-10","%Y-%m-%d")
    for i in range(155):
        fname = glob.glob("/data/nlahaye/output/Learnergy/COP_WHALE_DEMO/cop_env_whales_" +\
             str(i) + ".zarr.clust.data_*clusters.no_geo.tif")[0]
        dat = gdal.Open(fname).ReadAsArray()

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
        lst = np.unique(lst)

    mapper = {}
    print(lst.shape)
    for i in range(lst.shape[0]):
        mapper[float(lst[i])] = i

    mapper[float(-1.0)] = -1.0
    mapper[float(0.0)] = 0.0

    
    #for j in range(len(scenes)):
    #    dat = gdal.Open(scenes[j])
    #    for key in mapper.keys():
    #        dat[np.where(dat == float(key))] = mapper[key]
    #    print(dat.min(), dat.max())    


    with open(os.path.join(df_dir, df_uid + '_dfs.pkl'), "rb") as f:
        movement_dfs = pickle.load(f)

    for key in movement_dfs.keys():
        if "2017CA-Bmu-05800" in key or "2017CA-Bmu-058" not in key:
            continue        
        with open(os.path.join(out_dir, "final_env_maps_" + key + ".pkl"), "rb") as f:
            abstract_grid = pickle.load(f)
  
        movement_sub_df = movement_dfs[key]
        abstract_grid_sub = abstract_grid[key]

        processed = 0
        for i in range(len(movement_sub_df)):
            #for j in range(len(abstract_grid_sub[i])):
            abstract_grid_sub[i] = np.array(abstract_grid_sub[i])              
            print("HERE1", abstract_grid_sub[i].min(), abstract_grid_sub[i].max(), abstract_grid_sub[i].mean(), abstract_grid_sub[i].shape)
            if abstract_grid_sub[i].max() > 100 and abstract_grid_sub[i].max() < 1000: 
                processed = processed + 1
                continue
            #sys.exit(0)
            inds = np.where(abstract_grid_sub[i] <= 0.0)
            if abstract_grid_sub[i].max() >= 1000:
                abstract_grid_sub[i] = abstract_grid_sub[i]/1000
            else:
                abstract_grid_sub[i] = abstract_grid_sub[i]*1000
            abstract_grid_sub[i][inds] = -1 
            print("HERE", abstract_grid_sub[i].min(), abstract_grid_sub[i].max(), abstract_grid_sub[i].mean())
            for key2 in mapper.keys():
                print(key, key2, mapper[key2], i)
                abstract_grid_sub[i][np.where(abstract_grid_sub[i] == float(key2))] = mapper[key2]
            print("HERE3", abstract_grid_sub[i].min(), abstract_grid_sub[i].max(), i)             
        #print(np.min(abstract_grid_sub), np.max(abstract_grid_sub))   
        if processed >= len(movement_sub_df): 
            continue
        abstract_grid[key] = abstract_grid_sub 


        pkl_file = os.path.join(out_dir, "final_env_maps_" + key + ".pkl")
        with open(pkl_file, 'wb') as f:
            pickle.dump(abstract_grid, f, protocol=pickle.HIGHEST_PROTOCOL)
        #sys.exit(0)

update_clusters()

