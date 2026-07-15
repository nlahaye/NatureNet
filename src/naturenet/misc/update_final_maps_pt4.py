
import re
import os
import pickle
import zarr
import numpy as np
import glob
import sys

fles = glob.glob("/data/nlahaye/NatureNet/Hammerhead_Out_v1/final_env_maps_*.pkl")
out_fle = "/data/nlahaye/NatureNet/Hammerhead_Out_v1/final_env_maps_hammerhead_v1.pkl"
 
#fles = glob.glob("/data/nlahaye/NatureNet/Terrestrial_Movement/Boar_Europe/output/final_env_maps_*pkl")
final_dict = {}
#out_fle = "/data/nlahaye/NatureNet/Terrestrial_Movement/Boar_Europe/output/final_env_maps_boar_v1.pkl"

key_re = "env_maps_(.*)\.pkl"

scale = 1#1000.0

unq = []
np.set_printoptions(suppress=True)
for fi in range(len(fles)):
     del_keys = []
     print(fles[fi])
     with open(fles[fi], "rb") as f:
         abstract_grid = pickle.load(f)

     mtch = re.search(key_re, fles[fi])
     use_key = mtch.groups(1)[0]

     print(use_key, fles[fi])

     for key in abstract_grid.keys():
         if key != use_key:
             continue
         print(fles[fi], key, len(abstract_grid[key]))

         for i in range(len(abstract_grid[key])):
             #print(np.unique(abstract_grid[key][i]), np.unique(np.array(abstract_grid[key][i])*1000.0), np.unique(np.array(abstract_grid[key][i])*1000.0).astype(np.int32))
             abstract_grid[key][i] = (np.array(abstract_grid[key][i])*scale).astype(np.int16)
             unq.extend(np.unique(abstract_grid[key][i]))
             print(unq, fles[fi], "UNIQUE")

unq = np.unique(unq)
print(unq, len(unq))
#sys.exit(0)
dct = {}
for i in range(len(unq)):
    print(i)
    dct[i] = unq[i]

for fi in range(len(fles)):
    with open(fles[fi], "rb") as f:
         abstract_grid = pickle.load(f)
    mtch = re.search(key_re, fles[fi])
    use_key = mtch.groups(1)[0]
 
    print(use_key, fles[fi])

    for key in abstract_grid.keys():
        if key != use_key:
            continue
        for i in range(len(abstract_grid[key])):
            abstract_grid[key][i] = (np.array(abstract_grid[key][i])*scale).astype(np.int16)
            for key2 in dct.keys():
                print(key2, dct[key2], dct.keys())
                abstract_grid[key][i][np.where(abstract_grid[key][i] == int(dct[key2]))] = int(key2)
        final_dict[key] = abstract_grid[key] 


print(final_dict.keys())
print(len(unq))
             
 
with open(out_fle, 'wb') as f:
    pickle.dump(final_dict, f, protocol=pickle.HIGHEST_PROTOCOL)


