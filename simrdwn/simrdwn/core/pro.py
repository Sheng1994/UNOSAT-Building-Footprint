# coding=utf-8
from osgeo import gdal
from gdalconst import *


img_path = "/workspace/Buildingfootprint/OrgImage/MungunoTown.TIF"
def tifproextract(img_path):
    dataset = gdal.Open(img_path, GA_ReadOnly)
    im_proj = dataset.GetProjection()  # 获取投影信息
    return im_proj.split(",")[-1].split("\"")[1]

a = tifproextract(img_path)

print (a)