





#!/usr/bin/env python3
import numpy as np
import shutil
import copy
from pathlib import Path
import argparse
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import glob
import rioxarray as rxr
import xarray as xr
from rioxarray.merge import merge_arrays
import rasterio
from rasterio.crs import CRS
from rasterio.merge import merge
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds


BBOX_LAT = [48.1, 48.2]
BBOX_LON = [16.6, 17.0]


start_date = datetime.strptime("2021-09-01","%Y-%m-%d")
end_date = datetime.strptime("2021-11-01","%Y-%m-%d")

HLS_FILE = "/data/nlahaye/NatureNet_Env/Copernicus_Ter/Results/HLS_L30/HLS_SCENE.tif"

GFSC_DIR = "/data/nlahaye/NatureNet_Env/Copernicus_Ter/Results/GFSC/"

TCD_PATTERN = "/data/nlahaye/NatureNet_Env/Copernicus_Ter/Results/TCD/CLMS_HRLVLCC_TCD_S2021_R10m_*tif"
TCD_DIR = "/data/nlahaye/NatureNet_Env/Copernicus_Ter/Results/TCD/"

CLC_DIR = "/data/nlahaye/NatureNet_Env/Copernicus_Ter/Results/CLCplus_RASTER_2021_010m_03035/CLCplus_RASTER_2021_010m_03035/"

NDVI_DIR = "/data/nlahaye/NatureNet_Env/Copernicus_Ter/Results/NDVI/"

SWI_DIR = "/data/nlahaye/NatureNet_Env/Copernicus_Ter/Results/SWI/"

BUILT_DIR = "/data/nlahaye/NatureNet_Env/Copernicus_Ter/Results/EMC_BUILT/"

POP_DIR = "/data/nlahaye/NatureNet_Env/Copernicus_Ter/Results/GHS_POP/"

SMOD_DIR = "/data/nlahaye/NatureNet_Env/Copernicus_Ter/Results/GHS_SMOD/"

def run_mosaic():
    ##mosaic_snow()
    ##print("Snow Done")
    mosaic_hls()
    print("HLS Done")
    mosaic_td()
    print("TCD Done")
    #mosaic_clc()
    #print("CLC Done")
    #mosaic_ndvi()
    #print("NDVI Done")
    ##mosaic_swi()
    ##print("SWI Done") #UNUSED
    #mosaic_emc_built()
    #print("EMC_B Done")
    #mosaic_emc_built_g()
    #print("EMC_B_G Done")
    mosaic_pop()
    print("POP Done")
    #mosaic_smod()
    #print("SMOD Done")

def mosaic_hls():
    bbox = [BBOX_LON[0], BBOX_LAT[0], BBOX_LON[1], BBOX_LAT[1]]
    files = [HLS_FILE]
    out_fname = "/data/nlahaye/NatureNet_Env/Copernicus_Ter/Results/HLS_L30/HLS_CLIPPED.tif"
    mosaic_data(files, bbox, out_fname)

def mosaic_snow():

    bbox = [BBOX_LON[0], BBOX_LAT[0], BBOX_LON[1], BBOX_LAT[1]]
    
    date = copy.deepcopy(start_date)
    while date < end_date:
        files = glob.glob(GFSC_DIR + "/*/*" + date.strftime("%Y%m%d") + "*GF.tif")
        if len(files) < 1:
            continue
        out_fname = GFSC_DIR + "/GFSC_" + date.strftime("%Y%m%d") + "_GF.tif"
        mosaic_data(files, bbox, out_fname)

        date = date + timedelta(days=1)


def mosaic_td():

    bbox = [BBOX_LON[0], BBOX_LAT[0], BBOX_LON[1], BBOX_LAT[1]]
    files = glob.glob(TCD_PATTERN)
    out_fname = TCD_DIR + "TCD_2021.tif"
    mosaic_data(files, bbox, out_fname)


def mosaic_clc():

    bbox = [BBOX_LON[0], BBOX_LAT[0], BBOX_LON[1], BBOX_LAT[1]]
    files = glob.glob(CLC_DIR + "*tif")
    out_fname = CLC_DIR + "CLC_2021.tif"
    mosaic_data(files, bbox, out_fname)


def mosaic_emc_built():
    bbox = [BBOX_LON[0], BBOX_LAT[0], BBOX_LON[1], BBOX_LAT[1]]
    files = glob.glob(BUILT_DIR + "*BUILT_S_*tif")
    out_fname = BUILT_DIR + "EMC_BUILT_S_2021.tif"
    mosaic_data(files, bbox, out_fname)

def mosaic_emc_built_g():
    bbox = [BBOX_LON[0], BBOX_LAT[0], BBOX_LON[1], BBOX_LAT[1]]
    files = glob.glob(BUILT_DIR + "*BUILT_GREENNESS_*tif")
    out_fname = BUILT_DIR + "EMC_BUILT_GREENNESS_2021.tif"
    mosaic_data(files, bbox, out_fname)

def mosaic_pop():
    bbox = [BBOX_LON[0], BBOX_LAT[0], BBOX_LON[1], BBOX_LAT[1]]
    files = glob.glob(POP_DIR + "*tif")
    out_fname = POP_DIR + "POP_2021.tif"
    mosaic_data(files, bbox, out_fname)

def mosaic_smod():
    bbox = [BBOX_LON[0], BBOX_LAT[0], BBOX_LON[1], BBOX_LAT[1]]
    files = glob.glob(SMOD_DIR + "*tif")
    out_fname = SMOD_DIR + "SMOD_2021.tif"
    mosaic_data(files, bbox, out_fname)

def mosaic_ndvi():

    bbox = [BBOX_LON[0], BBOX_LAT[0], BBOX_LON[1], BBOX_LAT[1]]
    date = copy.deepcopy(start_date)
    while date < end_date:
        
        files = glob.glob(NDVI_DIR + "NDVI_*" + date.strftime("%Y%m") + "*tif")
        out_fname = NDVI_DIR + "NDVI_" + date.strftime("%Y%m") + ".tif"
        mosaic_data(files, bbox, out_fname)
        date = date + relativedelta(months=1)

def mosaic_swi():

    bbox = [BBOX_LON[0], BBOX_LAT[0], BBOX_LON[1], BBOX_LAT[1]]
    date = copy.deepcopy(start_date)
    while date < end_date:
        files = glob.glob(SWI_DIR + "SWI100*" + date.strftime("%Y%m") + "*tif")
        out_fname = SWI_DIR + "SWI_" + date.strftime("%Y%m") + ".tif"
        print(files)
        mosaic_data(files, bbox, out_fname)
        date = date + relativedelta(months=1) #timedelta(months=1) 


def mosaic_data(inputs, bbox, output, nodata=-999):
    input_paths = [Path(p) for p in inputs]
    out_path = Path(output)
    min_lon, min_lat, max_lon, max_lat = bbox

    last_valid = None

    srcs = []
    for p in input_paths:
        if not p.exists():
            raise FileNotFoundError(f"Missing input: {p}")
        tmp = rxr.open_rasterio(p) 

        tmp = tmp.rio.reproject(CRS.from_string('EPSG:4326'))        

        srcs.append(tmp)
    
    out_crs = CRS.from_string('EPSG:4326')

    bbox_wgs84 = (min_lon, min_lat, max_lon, max_lat)
    #if out_crs.to_string() != "EPSG:4326":
    #    bbox_out = transform_bounds("EPSG:4326", out_crs, *bbox_wgs84, densify_pts=21)
    #else:
    bbox_out = bbox_wgs84


    merged_raster = merge_arrays(dataarrays = srcs, bounds=bbox_out, method="first")
    print(merged_raster.min(), merged_raster.max(), merged_raster.mean(), merged_raster.std())
    merged_raster.rio.to_raster(out_path)

if __name__ == "__main__":
    run_mosaic()


