import zarr
import numpy as np


base = zarr.load("/data/nlahaye/NatureNet_Env/Copernicus/cop_env_shark_0.zarr")
 
#for i in range(19):
#    base.append(zarr.load("/data/nlahaye/NatureNet_Env/Copernicus/cop_env_shark_" + str(i) + ".zarr"))


for t in range(0, base.shape[0]):
    dat = base[t,:,:,:]
    for i in range(1,398):
        print(t, i, base.shape)
        dat2 = zarr.load("/data/nlahaye/NatureNet_Env/Copernicus/cop_env_shark_" +str(i) + ".zarr")
        print(dat2.shape)
        dat2 = dat2[t,:,:,:]
        print(dat.shape, dat2.shape)
        dat = np.concatenate((dat, dat2), axis=0)  
        print(dat.shape, dat2.shape, "PRINT 2")
        del dat2
        #if dat is None:
        #    dat = base[i][t,:,:,:]
        #else:
        #    print(dat.shape, base[i][t,:,:,:].shape)
        #    dat = np.concatenate((dat, base[i][t,:,:,:]), axis=0)    

    zarr.save_array("/data/nlahaye/NatureNet_Env/Copernicus/cop_env_shark_t" + str(t) + ".zarr", dat)
    del dat




