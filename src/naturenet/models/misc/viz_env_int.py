import numpy as np

import matplotlib
matplotlib.use("Agg")

from matplotlib import pyplot as plt
import os
import math
import pickle

import zarr

cmap = "jet"
  
#fname = "/data/nlahaye/NatureNet/Terrestrial_Movement/Boar_Europe/output/final_env_maps_1F5B2F1_(4118).pkl"
fname = "/data/nlahaye/NatureNet_Env/Copernicus_Ter/env_scenes/cop_env_boar_0.zarr"
 
env = None
#with open(fname, 'rb') as f:
#   env = pickle.load(f)

data = zarr.load(fname)
  
uid_key = '1F5B2F1_(4118)'
#TODO open file

#iterate over instrument keys
print(data.shape)

for i in range(data.shape[0]):
    dat_tmp = data[i]
    plt.matshow(np.squeeze(dat_tmp), cmap=cmap, interpolation='none', aspect="equal")
    plt.colorbar()
    plt.show()
    plt.savefig("boar_viz_map_chan_" + str(i) + ".png")
    plt.clf()
    plt.close() 
 

"""
for i in range(len(data)):
    for j in range(len(data[i])):
        dat_tmp = data[i][j]
        print(dat_tmp.shape)
        #dat_tmp = np.squeeze(np.max(dat_tmp, axis=(-1,-2)))
        #print(dat_tmp.shape)
        #for j in range(dat_tmp.shape[2]):
        plt.matshow(np.squeeze(dat_tmp[:,:,j]), cmap=cmap, interpolation='none', aspect="equal")
        plt.show()
        instrument = "coarsening_" + str(i)
        plt.savefig(instrument + "_boar_viz_map_chan_" + str(j) + ".png")
        plt.clf()
        plt.close()
"""
