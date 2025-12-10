
import datetime


#fname = ["/opt/dlami/nvme/NatureNet_Env/Bath/gebco_2025_sub_ice_n40.0_s20.0_w-100.0_e-60.0_geotiff.tif"]

  
bath_fglob_begin = "/opt/dlami/nvme/NatureNet_Env/Bath/gebco_"
bath_fglob_date_str = "%Y"
bath_fglob_end = "_sub_ice_n40.0_s20.0_w-100.0_e-60.0_geotiff.tif"
 

sss_fglob_begin = "/opt/dlami/nvme/NatureNet_Env/SSS/dataset-sss-ssd-rep-daily_"

sss_fglob_date_str = "%Y%m%d"

sst_fglob_begin = "/opt/dlami/nvme/NatureNet_Env/SST/oisst-avhrr-v02r01."
sst_fglob_date_str = "%Y%m%d" 
 
sdate = "30082024T00:00:00Z"
edate = "31082024T23:59:59Z"

date_str_pattern = "%d%m%YT%H:%M:%SZ"




current_date = datetime.datetime.strptime(sdate, date_str_pattern)
end_date = datetime.datetime.strptime(edate, date_str_pattern)


dates = []
fnames = {
"Bathymetry": []
"SSS" : []
"SST" : []

}

while current_date < end_date:
    dates.append(current_date.strftime(date_str_pattern))

    date_bath = current_date.strftime(bath_fglob_date_str)
    bath_fname = glob.glob(bath_fglob_begin + date_bath + bath_fglob_end)[0]
    fnames["Bathymetry"].append(bath_fname)

    date_sss = current_date.strftime(sss_fglob_date_str)
    sss_fname = glob.glob(sss_fglob_begin + date_sss + "*nc")[0]
    fnames["SSS"].append(bath_fname)

    date_sst = current_date.strftime(sst_fglob_date_str)
    sst_fname = glob.glob(sst_fglob_begin + date_sst + "*nc")[0]
    fnames["SST"].append(bath_fname)

    current_date = current_date + datetime.timedelta(days=1)



print(dates)

print(fnames)



