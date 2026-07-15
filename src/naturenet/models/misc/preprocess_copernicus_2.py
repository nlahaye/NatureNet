
import zarr
import numpy as np
import os
import copy
from sit_fuse.pipelines.inference.inference_utils import run_embed_gen_from_scene_arr
from sit_fuse.preprocessing.colocate_and_resample import resample_scene
from sit_fuse.datasets.dataset_utils import get_scenes
from sit_fuse.utils import read_copernicus_generic, read_vars_copernicus_generic, read_yaml

import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.ndimage import map_coordinates, distance_transform_edt



def fill_missing_bilinear_like(
    arr,
    missing_value=np.nan,
    radius=8,
    n_angles=8,
    step=1.0,
    mode="nearest",
):
    """
    Fast bilinear-like fill for missing pixels on a regular 2D raster grid.

    Strategy
    --------
    1. For each missing pixel, sample the image at several nearby floating-point
       coordinates using scipy.ndimage.map_coordinates(order=1), which performs
       bilinear interpolation in 2D.
    2. Ignore samples that still land on missing areas.
    3. Average valid samples.
    4. Fall back to nearest-neighbor fill for any unresolved pixels.

    Parameters
    ----------
    arr : 2D ndarray
        Input array.
    missing_value : scalar, default np.nan
        Missing-data marker.
    radius : int, default 8
        Max search radius in pixels.
    n_angles : int, default 8
        Number of directions to probe around each missing pixel.
    step : float, default 1.0
        Radial increment in pixels.
    mode : str, default "nearest"
        Boundary mode passed to map_coordinates.

    Returns
    -------
    filled : 2D ndarray
        Filled array.
    """
    arr = np.asarray(arr, dtype=float)

    if np.isnan(missing_value):
        mask_missing = np.isnan(arr)
    else:
        mask_missing = arr == missing_value

    if not np.any(mask_missing):
        return arr.copy()

    filled = arr.copy()

    # Nearest-neighbor fallback prepared once
    valid_mask = ~mask_missing
    _, nearest_idx = distance_transform_edt(mask_missing, return_indices=True)
    nearest_fill = arr[tuple(nearest_idx)]

    rows, cols = arr.shape
    miss_y, miss_x = np.where(mask_missing)

    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    radii = np.arange(step, radius + step, step)

    sample_sum = np.zeros(len(miss_y), dtype=float)
    sample_count = np.zeros(len(miss_y), dtype=float)

    # We only sample from known values, so create a version where missing entries are 0
    # and a separate valid-mask sampler to test whether each bilinear sample is usable.
    arr0 = arr.copy()
    arr0[mask_missing] = 0.0
    valid_float = valid_mask.astype(float)

    for r in radii:
        for theta in angles:
            yq = miss_y + r * np.sin(theta)
            xq = miss_x + r * np.cos(theta)

            inside = (yq >= 0) & (yq <= rows - 1) & (xq >= 0) & (xq <= cols - 1)
            if not np.any(inside):
                continue

            coords = np.vstack([yq[inside], xq[inside]])

            # Bilinear sample of values
            vals = map_coordinates(
                arr0,
                coords,
                order=1,
                mode=mode,
                prefilter=False
            )

            # Bilinear sample of validity mask
            weights = map_coordinates(
                valid_float,
                coords,
                order=1,
                mode=mode,
                prefilter=False
            )

            # Keep samples mostly supported by valid neighbors
            ok = weights > 0.95
            if np.any(ok):
                idx = np.where(inside)[0][ok]
                sample_sum[idx] += vals[ok]
                sample_count[idx] += 1

    resolved = sample_count > 0
    filled[miss_y[resolved], miss_x[resolved]] = (
        sample_sum[resolved] / sample_count[resolved]
    )

    # Fallback for unresolved pixels
    unresolved = ~resolved
    if np.any(unresolved):
        filled[miss_y[unresolved], miss_x[unresolved]] = nearest_fill[
            miss_y[unresolved], miss_x[unresolved]
        ]

    return filled




def resample_or_fuse_scene(scene, init_location, resample_config):

    resample_config["low_res"]["filenames"] = scene
    resample_config["low_res"]["geo_filenames"] = scene


    resample_config["return_products"] = True
    output, location = resample_scene(scene, init_location, resample_config)
    return output, location

def gen_copernicus_scenes_map(): #yml_conf,  per_channel_stats):

#/data/nlahaye/NatureNet_Env/Copernicus/copernicus-data-whales/cmems_mod_glo_bgc_my_0.083deg-lmtl_P1D-i.nc
#/data/nlahaye/NatureNet_Env/Copernicus/copernicus-data-whales/cmems_mod_glo_phy_anfc_0.083deg_static.nc
#/data/nlahaye/NatureNet_Env/Copernicus/copernicus-data-whales/cmems_obs-oc_glo_bgc-plankton_my_l4-gapfree-multi-4km_P1D.nc
#/data/nlahaye/NatureNet_Env/Copernicus/copernicus-data-whales/cmems_mod_glo_bgc_anfc_0.25deg_static.nc
#/data/nlahaye/NatureNet_Env/Copernicus/copernicus-data-whales/cmems_mod_glo_bgc_my_0.083deg-lmtl-Fphy_P1D-i.nc

#/data/nlahaye/NatureNet_Env/Copernicus/copernicus-data-whales/cmems_mod_glo_bgc_my_0.083deg-lmtl_P1D-i.nc
#/data/nlahaye/NatureNet_Env/Copernicus/copernicus-data-whales/cmems_mod_glo_phy_anfc_0.083deg_static.nc
#/data/nlahaye/NatureNet_Env/Copernicus/copernicus-data-whales/cmems_obs-oc_glo_bgc-plankton_my_l4-gapfree-multi-4km_P1D.nc
#/data/nlahaye/NatureNet_Env/Copernicus/copernicus-data-whales/cmems_mod_glo_bgc_anfc_0.25deg_static.nc
#/data/nlahaye/NatureNet_Env/Copernicus/copernicus-data-whales/cmems_mod_glo_bgc_my_0.083deg-lmtl-Fphy_P1D-i.nc


    data_dict = {

        #"0.25" : [
       # 
       # {"fname": "/data/nlahaye/NatureNet_Env/Copernicus/copernicus-data/cmems_mod_glo_bgc-bio_anfc_0.25deg_P1D-m.nc", \
       #     "vars": ["nppv", "o2"], "resolution": 0.25},
       # {"fname": "/data/nlahaye/NatureNet_Env/Copernicus/copernicus-data/cmems_mod_glo_bgc-car_anfc_0.25deg_P1D-m.nc", \
       #     "vars": ["ph", "dissic"], "resolution": 0.25},
       # {"fname": "/data/nlahaye/NatureNet_Env/Copernicus/copernicus-data/cmems_mod_glo_bgc-nut_anfc_0.25deg_P1D-m.nc", \
       #     "vars": ["fe", "no3", "po4", "si"], "resolution": 0.25},
       # {"fname": "/data/nlahaye/NatureNet_Env/Copernicus/copernicus-data/cmems_mod_glo_bgc-pft_anfc_0.25deg_P1D-m.nc", \
       #     "vars": ["phyc"], "resolution": 0.25}],
        "0.083" : [
        {"fname": "/data/nlahaye/NatureNet_Env/Copernicus/copernicus-data-whales/cmems_mod_glo_bgc_my_0.083deg-lmtl-Fphy_P1D-i.nc", \
            "vars": ["T", "U", "V"], "resolution": 0.083},
        {"fname": "/data/nlahaye/NatureNet_Env/Copernicus/copernicus-data-whales/cmems_mod_glo_bgc_my_0.083deg-lmtl_P1D-i.nc", \
             "vars": ["zooc", "npp"], "resolution": 0.083},
        #{"fname": "/data/nlahaye/NatureNet_Env/Copernicus/copernicus-data/cmems_mod_glo_phy_anfc_0.083deg_P1D-m.nc", \
        #     "vars": ["zos"], "resolution": 0.083},
        #{"fname": "/data/nlahaye/NatureNet_Env/Copernicus/copernicus-data/cmems_mod_glo_phy_anfc_0.083deg-sst-anomaly_P1D-m.nc", \
        #     "vars": ["sea_surface_temperature_anomaly"], "resolution": 0.083},
        #{"fname": "/data/nlahaye/NatureNet_Env/Copernicus/copernicus-data/cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m.nc", \
        #     "vars": ["uo", "vo"], "resolution": 0.083},
        #{"fname": "/data/nlahaye/NatureNet_Env/Copernicus/copernicus-data/cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m.nc", \
        #     "vars": ["so"], "resolution": 0.083},
        #{"fname": "/data/nlahaye/NatureNet_Env/Copernicus/copernicus-data/cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m.nc", \
        #     "vars": ["thetao"], "resolution": 0.083},
        #{"fname": "/data/nlahaye/NatureNet_Env/Copernicus/copernicus-data/cmems_mod_glo_phy-wcur_anfc_0.083deg_P1D-m.nc", \
        #     "vars": ["wo"], "resolution": 0.083},
        #{"fname": "/data/nlahaye/NatureNet_Env/Copernicus/copernicus-data/cmems_mod_glo_wav_anfc_0.083deg_PT3H-i.nc", \
        #     "vars": ["VCMX", "VHM0", "VHM0_SW1", "VHM0_WW", "VMDR", "VMDR_SW1", "VMDR_WW", "VMXL", "VPED", "VTM01_SW1", "VTM01_WW", "VTPK"]},
        ],
        "4km" : [
        {"fname": "/data/nlahaye/NatureNet_Env/Copernicus/copernicus-data-whales/cmems_obs-oc_glo_bgc-plankton_my_l4-gapfree-multi-4km_P1D.nc", \
            "vars": ["CHL"], "resolution": 0.03600036}
        ]
    }

 
    mask_fname = "/data/nlahaye/NatureNet_Env/Copernicus/copernicus-data-whales/cmems_mod_glo_phy_anfc_0.083deg_static.nc"

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


    resample_config = read_yaml("../../config/environment/cop_resample_config.yaml")

    #msk = read_copernicus_generic(mask_fname, ["mask"], depth=0)
    #msk_geo = read_vars_copernicus_generic(mask_fname, ["latitude", "longitude"], depth = 0)
    #longr, latgr = np.meshgrid(msk_geo[1], msk_geo[0])
    #msk_lonlat = np.array([longr, latgr]).astype(np.float32)
    #msk = np.expand_dims(np.squeeze(msk[0,0,:,:]), 0)
    #mask, mask_loc = resample_or_fuse_scene(msk, msk_lonlat, resample_config)
    #mask_inds = np.where(mask == 0)

    #fill = -999999
 
    #data[:,inds[0], inds[1]] = fill

    final_dict = {}

    dat_final = None
    lonlat = None

    n_days = 155

    day_ind = 0


    """
    for key in data_dict.keys():
        dat1 = None
        lonlat = None
        for fn_dict in data_dict[key]:
            dat = read_copernicus_generic(fn_dict["fname"], fn_dict["vars"], depth = 0) #C x Time x Lat x Lon
            print("INIT SHAPE", dat.shape)
            if dat.shape[1] > n_days:
                subsample = int(math.ceil(dat.shape[1] / n_days))
                print(subsample, len(list(range(0, dat.shape[1], subsample))))
                ind_sub = list(range(0, dat.shape[1], subsample))
                dat = dat[:, ind_sub, :, :]
            #if dat1 is None:
            #    dat1 = dat 
            #else:
            #    dat1 = np.concatenate((dat1, dat), axis=0)

            if lonlat is None:
                geo = read_vars_copernicus_generic(fn_dict["fname"], ["latitude", "longitude"], depth = 0)
                longr, latgr = np.meshgrid(geo[1], geo[0])
                lonlat = np.array([longr, latgr]).astype(np.float32)

            
            dims = copy.deepcopy(dat.shape)
            print(dims)
            dat = dat.reshape((dims[0]*dims[1], dims[2], dims[3]))
            print(dat.shape, lonlat.shape, "HERE HERE")
            data, location = resample_or_fuse_scene(dat, lonlat,  resample_config)
            print(data.shape, location.shape) 
            data = data.reshape((dims[0], dims[1], data.shape[1], data.shape[2]))

            data = np.swapaxes(data, 0,1) #T x C x Lat x Lon
            print(data.shape, "FINAL SAVE")

            fkey = key + "_whale_" + fn_dict["vars"][0]
            zarr.save("/data/nlahaye/NatureNet_Env/Copernicus/" + fkey + "_subsampled.zarr", data)
            del data
        del lonlat
    """

    dt_start = 4
    n_days = 155
    day_gap = 4

    for d in range(dt_start, n_days+1, day_gap):

        dat = {}
        dat_init = {}
        dt_end = d +  (day_gap)
        for key in data_dict.keys():
            for fn_dict in data_dict[key]:
                fkey = key + "_whale_" + fn_dict["vars"][0]
                dt = zarr.load("/data/nlahaye/NatureNet_Env/Copernicus/" + fkey + "_subsampled.zarr").astype(np.float32)
                print("LOADED", "/data/nlahaye/NatureNet_Env/Copernicus/" + fkey + "_subsampled.zarr")
                print(dt.shape, "HERE", dt.min())
                dt = dt[d:dt_end]
                inds = np.where(np.isinf(dt) | np.isnan(dt))
                dt[inds] = -999999.0
                print(dt.shape, "HERE2", dt.min())
                dat_init[fkey] = dt
                del dt



        for key in data_dict.keys():
                for fn_dict in data_dict[key]:
                    fkey = key + "_whale_" + fn_dict["vars"][0]
                    dt = dat_init[fkey]
                    if dt.ndim > 3:
                        for v in range(dt.shape[1]):
                            for t in range(dt.shape[0]):

                                dt[t,v] = fill_missing_bilinear_like(dt[t,v], -999999.0, 8, 8, 1.0, "nearest")
                                print("COMPLETED", t, v)
                    else:
                        for t in range(dt.shape[0]):
                            dt[t] = fill_missing_bilinear_like(dt[t], -999999.0, 8, 8, 1.0, "nearest")
                            print("COMPLETED", t)

                    dat[fkey] = dt
                    print("FIXED_FILL", "/data/nlahaye/NatureNet_Env/Copernicus/" + fkey + "_subsampled.zarr")
                    plt.imshow(np.squeeze(dt[0,0]))
                    plt.savefig(fkey + ".png")
                    del dt


        for day in range(d, dt_end):
            dat_cur = None
            for key in data_dict.keys():
                for fn_dict in data_dict[key]:

                    #fkey = key + "_" + fn_dict["vars"][0]
                    #dat = zarr.load("/data/nlahaye/NatureNet_Env/Copernicus/" + fkey + "_subsampled.zarr")            
                    day_ind = day - d
                    if dat_cur is None:
                        dat_cur = copy.deepcopy(dat[fkey][day_ind,:,:,:])
                    else:
                        tmp = dat[fkey][day_ind,:,:,:]
                        if len(tmp.shape) < 3:
                            tmp = np.expand_dims(tmp, axis=0)
                        print(dat_cur.shape, tmp.shape)
                        dat_cur = np.concatenate((dat_cur, tmp), axis=0)
                    if len(dat_cur.shape) < 3:
                        dat_cur = np.expand_dims(dat_cur, axis=0)


            print("HERE", dat_cur.min(), dat_cur.max(), dat_cur.mean(), dat_cur.std())
            zarr.save("/data/nlahaye/NatureNet_Env/Copernicus/cop_env_whales_" "t" + str(day) + ".zarr", dat_cur)
            del dat_cur

        del dat
        del dat_init



gen_copernicus_scenes_map()


