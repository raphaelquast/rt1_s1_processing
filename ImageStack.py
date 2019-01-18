# Copyright (c) 2014, Vienna University of Technology (TU Wien), Department
# of Geodesy and Geoinformation (GEO).
# All rights reserved.
# 
# All information contained herein is, and remains the property of Vienna
# University of Technology (TU Wien), Department of Geodesy and Geoinformation
# (GEO). The intellectual and technical concepts contained herein are
# proprietary to Vienna University of Technology (TU Wien), Department of
# Geodesy and Geoinformation (GEO). Dissemination of this information or
# reproduction of this material is forbidden unless prior written permission
# is obtained from Vienna University of Technology (TU Wien), Department of
# Geodesy and Geoinformation (GEO).

import os
from datetime import datetime
import tempfile
import numpy as np
from osgeo import gdal
import xml.etree.ElementTree as ET
from TSDataset import TSDataset


_GDAL_DTYPE = {gdal.GDT_Byte: "Byte",
               gdal.GDT_Int16: "Int16",
               gdal.GDT_Int32: "Int32",
               gdal.GDT_UInt16: "Uint16",
               gdal.GDT_UInt32: "Uint32",
               gdal.GDT_Float32: "Float32",
               gdal.GDT_Float64: "Float64",
               gdal.GDT_CFloat32: "CFloat32",
               gdal.GDT_CFloat64: "CFloat64"}


def get_gdal_dtype_string(gdal_dtype):
    return _GDAL_DTYPE[gdal_dtype]


class VrtImageStackDataset(TSDataset):

    def __init__(self, name, vrt_path, times, nodata=None, autoclean=True, filelist=None):
        self._isvalid = True
        self._vrt_path = vrt_path
        self.filelist = filelist
        self.autoclean = autoclean
        self._vrt_ds = gdal.Open(self._vrt_path, gdal.GA_ReadOnly)
        if self._vrt_ds is None:
            self._isvalid = False
        shape = (self._vrt_ds.RasterCount, self._vrt_ds.RasterYSize,
                 self._vrt_ds.RasterXSize)
        geot = self._vrt_ds.GetGeoTransform()
        proj = self._vrt_ds.GetProjection()
        super(VrtImageStackDataset, self).__init__(name, shape, geot, proj,
                                                   times=times, nodata=nodata)

    def is_valid(self):
        return self._isvalid

    def read_ts(self, col, row, col_size=1, row_size=1):
        ts = self._vrt_ds.ReadAsArray(col, row, col_size, row_size)
        return (self.times(), ts)

    def read_image(self, time, subset=None):
        """
        read an image layer

        Parameters
        ----------
        time : datetime object
            the exact time of the image
        subset : list or tuple
            The subset should be in pixels, like this (xmin, ymin, xmax, ymax)

        Return
        ------
            2D numpy array
        """
        times_match_ind = np.where(np.array([(x == time) for x in self._times]))
        band_idx = times_match_ind[0][0]
        band = self._vrt_ds.GetRasterBand(band_idx)
        if subset:
            arr = band.ReadAsArray(0, 0, band.XSize, band.YSize)
        else:
            arr = band.ReadAsArray(subset[0], subset[1], subset[2], subset[3])
        return arr

    def __del__(self):
        if self.autoclean:
            self._vrt_ds = None
            if os.path.exists(self._vrt_path):
                os.remove(self._vrt_path)


def create_imagestack_dataset(name, filelist, times, nodata):
    tmp_vrt_name = "temp_{}_{}.vrt".format(name, datetime.utcnow().strftime("%Y%m%d%H%M%S"))
    vrt_path = os.path.join(tempfile.gettempdir(), tmp_vrt_name)
    build_vrt(vrt_path, filelist, nodata=nodata)
    return VrtImageStackDataset(name, vrt_path, times, nodata=nodata, filelist=filelist)


def build_vrt(outfile, infilelist, nodata=None):

    first_ds = gdal.Open(infilelist[0])
    proj = first_ds.GetProjection()
    geot = first_ds.GetGeoTransform()
    xsize = first_ds.RasterXSize
    ysize = first_ds.RasterYSize
    dtype = first_ds.GetRasterBand(1).DataType
    first_ds = None

    # =======================================================
    # make vrt dataset
    vrt_root = ET.Element("VRTDataset",
                          attrib={"rasterXSize": str(xsize),
                                  "rasterYSize": str(ysize)})
    # geotransform
    geot_elem = ET.SubElement(vrt_root, "GeoTransform")
    geot_elem.text = ",".join(map(str, geot))
    # spatial reference
    geot_elem = ET.SubElement(vrt_root, "SRS")
    geot_elem.text = proj

    for idx, img in enumerate(infilelist):
        dtype_str = get_gdal_dtype_string(dtype)
        band_elem = ET.SubElement(vrt_root, "VRTRasterBand",
                                  attrib={"dataType": dtype_str,
                                            "band": str(idx + 1)})
        src_elem = ET.SubElement(band_elem, "SimpleSource")
        file_elem = ET.SubElement(src_elem, "SourceFilename",
                                  attrib={"relativetoVRT": "0"})
        file_elem.text = img
        ET.SubElement(src_elem, "SourceBand").text = "1"
        if nodata:
            ET.SubElement(band_elem, "NodataValue").text = str(nodata)

    # write vrt dataset
    tree = ET.ElementTree(vrt_root)
    tree.write(outfile, encoding="utf-8")
    return None

