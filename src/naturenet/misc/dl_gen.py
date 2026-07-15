import datetime


command_begin = "wget https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/v2.1/access/avhrr/"
 

start_date = "20220101"
end_date = "20240601"

start_date = datetime.datetime.strptime(start_date, "%Y%m%d")
end_date =  datetime.datetime.strptime(end_date, "%Y%m%d")


date = start_date
while date < end_date:
    command = command_begin + date.strftime("%Y%m") + "/" + "oisst-avhrr-v02r01." + date.strftime("%Y%m%d") + ".nc"
    print(command)
    date = date + datetime.timedelta(days=1)
    



