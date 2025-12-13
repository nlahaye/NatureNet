
import datetime

from sit_fuse.utils import read_yaml

import os

import glob
 
import yaml

#fname = ["/mnt/data/NatureNet_Env/Bath/gebco_2025_sub_ice_n40.0_s20.0_w-100.0_e-60.0_geotiff.tif"]

 
yml_fname = "../../config/environment/basic_env_setup_2.yaml"
  
bath_fglob_begin = "/mnt/data/NatureNet_Env/Bath/gebco_"
bath_fglob_date_str = "%Y"
bath_fglob_end = "_sub_ice_n40.0_s20.0_w-100.0_e-60.0_geotiff.tif"

bath_fglob_end_2 = "_n40.0_s20.0_w-100.0_e-60.0_geotiff.tif" 

sss_fglob_begin = "/mnt/data/NatureNet_Env/SSS/dataset-sss-ssd-rep-daily_"

sss_fglob_date_str = "%Y%m%d"

sst_fglob_begin = "/mnt/data/NatureNet_Env/SST/oisst-avhrr-v02r01."
sst_fglob_date_str = "%Y%m%d" 
 
sdate = "03012024T00:00:00Z"
edate = "31102024T23:59:59Z"
  
coastal_dist_fname = "/mnt/data/NatureNet_Env/Coastal_Dist/GMT_intermediate_coast_distance_01d.tif"
 
date_str_pattern = "%d%m%YT%H:%M:%SZ"

yml_conf = read_yaml(yml_fname)


current_date = datetime.datetime.strptime(sdate, date_str_pattern)
end_date = datetime.datetime.strptime(edate, date_str_pattern)


dates = []
fnames = {
"Bathymetry": [],
"SSS" : [],
"SST" : [],
"Coastal_Dist" : []
}


date_ind = 0

file_ind = 0

while current_date < end_date:
    dates.append(current_date.strftime(date_str_pattern))

    date_bath = current_date.strftime(bath_fglob_date_str)
    bath_fname = glob.glob(bath_fglob_begin + date_bath + bath_fglob_end)
    if len(bath_fname) == 0:
        bath_fname = glob.glob(bath_fglob_begin + date_bath + bath_fglob_end_2)
    bath_fname = bath_fname[0]
    fnames["Bathymetry"].append([bath_fname])

    
    date_sss = current_date.strftime(sss_fglob_date_str)
    sss_fname = glob.glob(sss_fglob_begin + date_sss + "*nc")[0]
    fnames["SSS"].append([sss_fname])

    date_sst = current_date.strftime(sst_fglob_date_str)
    sst_fname = glob.glob(sst_fglob_begin + date_sst + "*nc")[0]
    fnames["SST"].append([sst_fname])

    fnames["Coastal_Dist"].append([coastal_dist_fname])

    current_date = current_date + datetime.timedelta(days=1)

    if date_ind >= 32:


        yml_conf["instruments"]["Bathymetry"]["filenames"] = fnames["Bathymetry"]
        yml_conf["instruments"]["SSS"]["filenames"] = fnames["SSS"]
        yml_conf["instruments"]["SST"]["filenames"] = fnames["SST"]
        yml_conf["instruments"]["Coastal_Dist"]["filenames"] = fnames["Coastal_Dist"]
        yml_conf["datetimes"] = dates


        new_fname = os.path.splitext(yml_fname)[0] + "NEW_SET_" + str(file_ind) + ".yaml"
        print(new_fname)
        with open(new_fname, 'w') as yml:
            yaml.dump(yml_conf, yml)

        print(dates)
   
        print(fnames)

        file_ind = file_ind + 1

        date_ind = 0
        dates = []
        fnames = {
        "Bathymetry": [],
        "SSS" : [],
        "SST" : [],
        "Coastal_Dist" : []
        }

    date_ind = date_ind + 1
