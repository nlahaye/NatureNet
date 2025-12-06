
import datetime





fname = ["/opt/dlami/nvme/NatureNet_Env/Bath/gebco_2025_sub_ice_n40.0_s20.0_w-100.0_e-60.0_geotiff.tif"]

sdate = "01062024T00:00:00Z"
edate = "31082024T23:59:59Z"

date_str_pattern = "%d%m%YT%H:%M:%SZ"




current_date = datetime.datetime.strptime(sdate, date_str_pattern)
end_date = datetime.datetime.strptime(edate, date_str_pattern)


dates = []
fnames = []

while current_date < end_date:
    dates.append(current_date.strftime(date_str_pattern))
    current_date = current_date + datetime.timedelta(days=1)
    fnames.append(fname)



print(dates)

print(fnames)



