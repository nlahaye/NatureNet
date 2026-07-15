

from goespy.Downloader import ABI_Downloader


destination_path = "/data/nlahaye/remoteSensing/GOES_LA/"
bucket = 'noaa-goes18'
year=['2025']
month=['01']
day=['06', '07', '08', '09','10', '11', '12', '13']
hour= [str(x).zfill(2) for x in [0,1, 16, 17, 18, 19, 20, 21, 22, 23]] #[range(18,19)]

product='ABI-L1b-RadC'
channel=["C01", "C02", "C03", "C04", "C05", "C06", "C07", "C08", "C09", "C10", "C11", "C12", "C13", "C14", "C15", "C16"]

Abi = ABI_Downloader(destination_path, bucket,year,month,day,hour,product,channel)





