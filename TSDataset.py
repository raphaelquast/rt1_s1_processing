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

from osgeo import osr


class TSDataset(object):
    def __init__(self, name, shape, geotransform, projection,
                 times=None, nodata=None, description=None):
        """ General dataset

        Parameter
        ---------
        name : string
            dataset name
        shape : list
            dimension of dataset as (bands, rows, cols)
        geotransfrom : list
            georeferencing transform list as gdal geotransfrom
        projection : string
            wkt string of spatial reference system
        times : list of datetime
            sequential datetime of dataset
        nodata : float
            nodata value
        description : string
            description text of dataset

        """
        self._name = name
        self._shape = shape
        self._geotrans = geotransform
        self._projection = projection
        self._times = times
        self._nodata = nodata
        self._desc = description
        self._extent = None
        self._ds_sr = None

    @property
    def name(self):
        return self._name

    @property
    def shape(self):
        return self._shape

    @property
    def geotransform(self):
        return self._geotrans

    @property
    def projection(self):
        return self._projection

    def times(self):
        return self._times

    def nodata(self):
        return self._nodata

    def description(self):
        return self._desc

    @property
    def extent(self):
        if self._extent is None:
            self._extent = (self._geotrans[0], self._geotrans[3] + self._shape[1] * self._geotrans[5],
                            self._geotrans[0] + self.shape[2] * self._geotrans[1], self._geotrans[3])
        return self._extent

    def SpatialReference(self):
        if self._ds_sr is None:
            self._ds_sr = osr.SpatialReference()
            self._ds_sr.ImportFromWkt(self._projection)
        return self._ds_sr

    def xy2rowcol(self, x, y, crs=None):
        if crs:
            geo_sr = osr.SpatialReference()
            geo_sr.ImportFromWkt(crs)
        else:
            # if crs is not set, use latlon as default
            geo_sr = osr.SpatialReference()
            geo_sr.SetWellKnownGeogCS("EPSG:4326")

        ds_sr = self.SpatialReference()
        if not ds_sr.IsSame(geo_sr):
            tx = osr.CoordinateTransformation(geo_sr, ds_sr)
            x, y, _ = tx.TransformPoint(x, y)
        # calc the row and columns
        col = int((x - self.extent[0]) / self._geotrans[1])
        row = int((y - self.extent[3]) / self._geotrans[5])
        if (col < 0 or col >= self._shape[2]
                or row < 0 or row >= self._shape[1]):
            return None
        return (row, col)

    def read_ts(self, row, col, row_size=1, col_size=1):
        # if error occurs, return None
        return None

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
        # if error occurs, return None
        return None

