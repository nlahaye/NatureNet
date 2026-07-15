

from osgeo import gdal



fname = "/data/nlahaye/NatureNet_Env/Copernicus/copernicus-data/shipdensity_global.tif"

out_fname = "/data/nlahaye/NatureNet_Env/Copernicus/copernicus-data/shipdensity_global_norm.tif"

dat = gdal.Open(fname)
arr = dat.ReadAsArray()

stdev = arr.std()
rnge = arr.max() - arr.min()
mn = arr.min()

arr = (arr - mn) / rnge

sizex = arr.shape[1]
sizey = arr.shape[0]

geoTransform = dat.GetGeoTransform()
wkt = dat.GetProjection()
dat.FlushCache()
dat = None

out_ds = gdal.GetDriverByName("GTiff").Create(out_fname, sizex, sizey, 1, gdal.GDT_Float32)

out_ds.SetGeoTransform(geoTransform)
out_ds.SetProjection(wkt)
out_ds.GetRasterBand(1).WriteArray(arr)
out_ds.FlushCache()
out_ds = None
 






