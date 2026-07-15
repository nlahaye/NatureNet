
import os
import numpy as np
import pickle

init_envs = [
"/data/nlahaye/NatureNet_Env/output/env_map_200368_1.pkl",
"/data/nlahaye/NatureNet_Env/output/env_map_235283_6.pkl",
"/data/nlahaye/NatureNet_Env/output/env_map_222133_1.pkl",
"/data/nlahaye/NatureNet_Env/output/env_map_209020_1.pkl",
"/data/nlahaye/NatureNet_Env/output/env_map_261743_2.pkl",
"/data/nlahaye/NatureNet_Env/output/env_map_183623_1.pkl",
"/data/nlahaye/NatureNet_Env/output/env_map_244607_1.pkl",
"/data/nlahaye/NatureNet_Env/output/env_map_200369_2.pkl",
"/data/nlahaye/NatureNet_Env/output/env_map_244608_2.pkl",
"/data/nlahaye/NatureNet_Env/output/env_map_244608_1.pkl",
"/data/nlahaye/NatureNet_Env/output/env_map_200369_1.pkl",
]

fname = "/data/nlahaye/NatureNet_Env/output_test/env_maps_simple_env7779d556-40b2-4f20-8c12-ae4f0947ab46.pkl"
#uid_key = '235283_6'


data = []
with open(fname, 'rb') as f:
    simple_dat = pickle.load(f)

 
for uid_key in simple_dat.keys():
    if uid_key == '235283_6':
        continue
    simple_dat = simple_dat[uid_key]
    for i in range(len(simple_dat[0][0])):

        print(len(simple_dat), len(simple_dat[0]), len(simple_dat[0][0]), simple_dat[0][0][i].shape)
        data.append(simple_dat[0][0][i][:,:,0:2,:,:])

    print(len(data), data[0].shape, data[1].shape, data[2].shape)
    for i in range(len(init_envs)):
        with open(init_envs[i], 'rb') as f:
            dat = pickle.load(f)
        print(len(dat))
        for j in range(len(dat)):
            for k in range(len(dat[j])):
                print(len(dat[j][k]))
                dat[j][k] = data
        fname_2 = os.path.join(os.path.dirname(fname), os.path.basename(init_envs[i]))
        with open(fname_2, 'wb') as f:
                pickle.dump(dat, f, protocol=pickle.HIGHEST_PROTOCOL)




