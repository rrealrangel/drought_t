#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""DROUGHT ASSESSMENT
DESCRIPTION
    Set of scripts to perform a drought hazard quantitative assessment.

AUTHOR
    Roberto A. Real-Rangel (Institute of Engineering UNAM; Mexico)

LICENSE
    GNU General Public License
"""
import math
import sys as _sys

from numpy import nan as _nan
from pathlib2 import Path as _Path
import gdal as _gdal
import ogr as _ogr
import pandas as _pd
import toml as _toml
import xarray as _xr


class Configurations():
    """
    """
    def __init__(self, config_file):
        self.config_file = config_file
        config = _toml.load(config_file)

        for key, value in config.items():
            setattr(self, key, value)


def list_files(parent_dir, ext):
    """List all files in a directory with a specified extension.

    Parameters
        parent_dir: string
            Full path of the directory of which the files are to be listed.
        ext: string or list of strings
            Extension(s) of the files to be listed.
    """
    parent_dir = _Path(parent_dir)
    files_list = []

    if isinstance(ext, str):
        patterns = ['**/*' + ext]

    elif isinstance(ext, list):
        patterns = ['**/*' + i for i in ext]

    for patt in patterns:
        files_list.extend(parent_dir.glob(pattern=patt))

    return([str(i) for i in files_list])


def vmap_area(input_vmap):
    source_ds = _ogr.Open(input_vmap)
    source_layer = source_ds.GetLayer(0)
    feature = source_layer[0]
    geom = feature.GetGeometryRef()
    return(geom.GetArea())


def degdist_2_meters(lat1, lat2, lon1, lon2):
    """
    https://stackoverflow.com/a/11172685/6331477
    """
    R = 6378137  # Radius of earth in meters
    dLat = lat2 * math.pi / 180 - lat1 * math.pi / 180
    dLon = lon2 * math.pi / 180 - lon1 * math.pi / 180
    a = ((math.sin(dLat / 2) * math.sin(dLat / 2)) +
         (math.cos(lat1 * math.pi / 180) *
          math.cos(lat2 * math.pi / 180) *
          math.sin(dLon / 2) *
          math.sin(dLon / 2)))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = R * c
    return(d)


def degbox_2_area(lat, lon, res):
    w = degdist_2_meters(
        lat1=(lat),
        lat2=(lat),
        lon1=(lon - res / 2),
        lon2=(lon + res / 2)
        )
    h = degdist_2_meters(
        lat1=(lat - res / 2),
        lat2=(lat + res / 2),
        lon1=(lon),
        lon2=(lon)
        )
    return(w * h)


def vector2array(basin_vmap, resolution, nodata):
    """
    Parameters:
        basin_vmap : string
        resolution : float
        nodata : float

    Source:
        https://bit.ly/2HxeOng
    """
    # Open the data source and read in the extent
    source_ds = _ogr.Open(basin_vmap)
    source_layer = source_ds.GetLayer(0)
    xmin, xmax, ymin, ymax = source_layer.GetExtent()

    def round_mult(num, mult, to):
        if to == 'up':
            return(int(mult * round(float(num + mult) / mult)))

        elif to == 'down':
            return(int(mult * round(float(num - mult) / mult)))

    xmin = round_mult(num=xmin, mult=resolution, to='down')
    xmax = round_mult(num=xmax, mult=resolution, to='up')
    ymin = round_mult(num=ymin, mult=resolution, to='down')
    ymax = round_mult(num=ymax, mult=resolution, to='up')

    # Create the destination data source
    cols = int((xmax - xmin) / resolution)
    rows = int((ymax - ymin) / resolution)
    output_source = _gdal.GetDriverByName('MEM').Create(
        '', cols, rows, _gdal.GDT_Byte
        )
    output_source.SetGeoTransform((xmin, resolution, 0, ymax, 0, -resolution))
    output_band = output_source.GetRasterBand(1)
    output_band.SetNoDataValue(nodata)

    # Rasterize
    _gdal.RasterizeLayer(output_source, [1], source_layer, burn_values=[1])

    # Read as array
    return(output_band.ReadAsArray().astype(int))


def open_clim_ts(
        input_dir, basin_vmap, resolution, nodata, missthresh, output_varname,
        flowstate
        ):
    """Reads a datacube (t, x, y) and generates a time series.

    Parameters:
        input_dir : string
        basin_vmap : string
        resolution : float
        nodata : float
        missthresh : float
        output_varname : string
        flowstate : string

    Outputs:
    """
    data = _xr.open_mfdataset(paths=(input_dir + '/*.nc4'))
    data = data.rename(name_dict={'prec': output_varname})
    mask = vector2array(basin_vmap, resolution, nodata)
    inregion_cells = (mask == 1).sum()
    min_cells = inregion_cells * missthresh
    basin_area = inregion_cells * (resolution**2 / 1e6)

    if flowstate == 'flow':
        accum = (data[output_varname] * resolution**2).sum(['east', 'north'])
        data_aggregated = accum / (basin_area * 1e6)

    elif flowstate == 'state':
        data_aggregated = data[output_varname].mean(['east', 'north'])

    cells_per_date = data[output_varname].notnull().sum(['east', 'north'])

    time_series = _xr.where(
        cells_per_date < min_cells, _nan, data_aggregated
        ).to_series()

    new_indices = _pd.date_range(
        start=sorted(time_series.index)[0],
        end=sorted(time_series.index)[-1],
        freq='1D'
        )

    # TODO: clip data to the basin.

    return(time_series.reindex(new_indices))


def open_chirps_ts(input_dir, output_varname, res):
    data = _xr.open_mfdataset(paths=(input_dir + '/*.nc'))
    data = data.rename(name_dict={'precip': output_varname})
    data_accum_per_cell = data.copy().observed
    data_accum_per_cell.values = data_accum_per_cell.values * _pd.np.nan

    basin_area = 0
    for lat in data_accum_per_cell.latitude:
        for lon in data_accum_per_cell.longitude:
            cell_area = degbox_2_area(
                lat=lat,
                lon=lon,
                res=res
                )
            data_accum_per_cell.loc[
                {'latitude': lat, 'longitude': lon}
                ].values[:] = (data.observed.loc[
                    {'latitude': lat, 'longitude': lon}
                    ] * cell_area)
            basin_area += cell_area

    basin_precip = (data_accum_per_cell.sum(
        ['latitude', 'longitude']
        ) / basin_area).to_dataframe()['observed']
#    basin_precip.index.freq = _pd.infer_freq(basin_precip.index)
    return(basin_precip)


def open_sflow_ts(input_file, basin_vmap, local_tz):
    # Open data source.
    data = _xr.open_dataset(input_file)
    data = data['disc_filled'].rename('observed').to_series()

    # Add 8 hours to the timestamps (actual measuring time) and
    # convert them from local to UTC
    data.index = data.index + _pd.Timedelta(8, 'h')
    data.index = data.index.tz_localize(local_tz)
    data.index = data.index.tz_convert('UTC')

    # Convert data units from m3 s-1 to mm day-1 m-2.
    basin_area = vmap_area(input_vmap=basin_vmap)
    data = ((data * (24 * 60 * 60) * 1000) / basin_area)
    new_indices = _pd.date_range(
        start=data.index[0],
        end=data.index[-1],
        freq='1D'
        )
    return(data.reindex(new_indices))


def progress_message(current, total, message="- Processing", units=None):
    """Issue a messages of the progress of the process.

    Generates a progress bar in terminal. It works within a for loop,
    computing the progress percentage based on the current item
    number and the total length of the sequence of item to iterate.

    Parameters:
        current : integer
            The last item number computed within the for loop. This
            could be obtained using enumerate() in when calling the for
            loop.
        total : integer
            The total length of the sequence for which the for loop is
            performing the iterations.
        message : string (optional; default = "- Processing")
    """
    if units is not None:
        progress = float(current)/total
        _sys.stdout.write("\r    {} ({:.1f} % of {} processed)".format(
                message, progress * 100, units))

    else:
        progress = float(current)/total
        _sys.stdout.write("\r    {} ({:.1f} % processed)".format(
                message, progress * 100))

    if progress < 1:
        _sys.stdout.flush()

    else:
        _sys.stdout.write('\n')
