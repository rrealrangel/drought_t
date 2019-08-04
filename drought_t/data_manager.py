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


def list_files(input_dir, input_patt):
    """List all files in a directory with a specified extension.

    Parameters
        input_dir: string
            Full path of the directory of which the files are to be listed.
        input_patt: string or list of strings
            Extension(s) of the files to be listed.
    """
    input_dir = _Path(input_dir)
    files_list = []

    if isinstance(input_patt, str):
        patterns = ['**/*' + input_patt]

    elif isinstance(input_patt, list):
        patterns = ['**/*' + i for i in input_patt]

    for patt in patterns:
        files_list.extend(input_dir.glob(pattern=patt))

    return([str(i) for i in files_list])


def vmap_area(input_path):
    source_ds = _ogr.Open(input_path)
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


def degbox_2_area(lat, lon, resolution):
    w = degdist_2_meters(
        lat1=(lat),
        lat2=(lat),
        lon1=(lon - resolution / 2),
        lon2=(lon + resolution / 2)
        )
    h = degdist_2_meters(
        lat1=(lat - resolution / 2),
        lat2=(lat + resolution / 2),
        lon1=(lon),
        lon2=(lon)
        )
    return(w * h)


def vector2array(input_path, resolution, nodata):
    """
    Parameters:
        input_path : string
        resolution : float
        nodata : float

    Source:
        https://bit.ly/2HxeOng
    """
    # Open the data source and read in the extent
    source_ds = _ogr.Open(input_path)
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


def open_precip(
        source, input_data_dir, output_var_name, resolution, time_zone,
        **kwargs
        ):
    """Open the precipitation data source.

    This functions joints the functions open_clim_ts() and
        open_chirps_ts() of erlier versions.

    Parameters:
        source : string
            Options are 'bdcn' and 'chirps'.
        input_data_dir : string
            The path of the local directory that contains the input
            precipitation datasets.
        output_var_name : string
            The name assigned to the precipitation variable in the
            output data array.
        resolution : float
            Spatial resolution of the input data. Future versions will
            get this value from the input data.
        kwargs['input_mask_path'] : string
            Used only if source == 'bdcn'. Path of the file of the
            vector map (Shapefile format) used to mask the input data.
        kwargs['nodata'] : int
            Used only if source == 'bdcn'. Value assigned to empty
            cells.
        kwargs['missthresh'] : float
            Used only if source == 'bdcn'. Minimum ratio of available
            data to perform the area aggregation.
        kwargs['flowstate'] : string
            Used only if source == 'bdcn'. Wether the input data is a
            flow or a state variable. This parameter will be removed
            in future versions.

    # TODO: Optimize this function.
    """
    if source == 'bdcn':
        # TODO: clip data to the basin.
        data = _xr.open_mfdataset(paths=(input_data_dir + '/*.nc4'))
        data = data.rename(name_dict={'prec': output_var_name})
        mask = vector2array(
            input_path=kwargs['input_mask_path'],
            resolution=resolution,
            nodata=kwargs['nodata']
            )
        inregion_cells = (mask == 1).sum()
        min_cells = inregion_cells * kwargs['missthresh']
        basin_area = inregion_cells * (resolution**2 / 1e6)

        if kwargs['flowstate'] == 'flow':
            accum = (data[output_var_name] * resolution**2).sum(
                ['east', 'north']
                )
            data_aggregated = accum / (basin_area * 1e6)

        elif kwargs['flowstate'] == 'state':
            data_aggregated = data[output_var_name].mean(['east', 'north'])

        cells_per_date = data[output_var_name].notnull().sum(['east', 'north'])
        time_series = _xr.where(
            cells_per_date < min_cells, _nan, data_aggregated
            ).to_series()
        new_indices = _pd.date_range(
            start=sorted(time_series.index)[0],
            end=sorted(time_series.index)[-1],
            freq='1D'
            )
        precip = time_series.reindex(new_indices)

    elif source == 'chirps':
        data = _xr.open_mfdataset(paths=(input_data_dir + '/*.nc'))
        data = data.rename(name_dict={'precip': output_var_name})
        data_accum_per_cell = data.copy().observed
        data_accum_per_cell.values = data_accum_per_cell.values * _pd.np.nan

        basin_area = 0
        for lat in data_accum_per_cell.latitude:
            for lon in data_accum_per_cell.longitude:
                cell_area = degbox_2_area(
                    lat=lat,
                    lon=lon,
                    resolution=resolution
                    )
                data_accum_per_cell.loc[
                    {'latitude': lat, 'longitude': lon}
                    ].values[:] = (data.observed.loc[
                        {'latitude': lat, 'longitude': lon}
                        ] * cell_area)
                basin_area += cell_area

        precip = (data_accum_per_cell.sum(
            ['latitude', 'longitude']
            ) / basin_area).to_dataframe()['observed']
        # precip.index.freq = _pd.infer_freq(precip.index)

    precip.index = precip.index.tz_localize(time_zone)
    precip.index = precip.index.tz_convert('UTC')
    precip.index = precip.index.tz_localize(None)
    return(precip)


def open_sflow_ts(input_file, input_mask_path, time_zone):
    # Open data source.
    data = _xr.open_dataset(input_file)
    data = data['disc_filled'].rename('observed').to_series()

    # Add 8 hours to the timestamps (actual measuring time) and
    # convert them from local to UTC
    data.index = data.index + _pd.Timedelta(8, 'h')
    data.index = data.index.tz_localize(time_zone)
    data.index = data.index.tz_convert('UTC')
    data.index = data.index.tz_localize(None)

    # Convert data units from m3 s-1 to mm day-1 m-2.
    basin_area = vmap_area(input_path=input_mask_path)
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
