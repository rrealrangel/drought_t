#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DROUGHT ASSESSMENT
DESCRIPTION
    Set of scripts to perform a drought hazard quantitative assessment.

AUTHOR
    Roberto A. Real-Rangel (Institute of Engineering UNAM; Mexico)

LICENSE
    GNU General Public License
"""
import sys

from numpy import nan
from pathlib2 import Path
import gdal
import ogr
import pandas as pd
import toml as _toml
import xarray as xr


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
    parent_dir = Path(parent_dir)
    files_list = []

    if isinstance(ext, str):
        patterns = ['**/*' + ext]

    elif isinstance(ext, list):
        patterns = ['**/*' + i for i in ext]

    for patt in patterns:
        files_list.extend(parent_dir.glob(pattern=patt))

    return([str(i) for i in files_list])


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
    source_ds = ogr.Open(basin_vmap)
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
    output_source = gdal.GetDriverByName('MEM').Create(
        '', cols, rows, gdal.GDT_Byte
        )
    output_source.SetGeoTransform((xmin, resolution, 0, ymax, 0, -resolution))
    output_band = output_source.GetRasterBand(1)
    output_band.SetNoDataValue(nodata)

    # Rasterize
    gdal.RasterizeLayer(output_source, [1], source_layer, burn_values=[1])

    # Read as array
    return(output_band.ReadAsArray().astype(int))


def clim_time_series(
        input_dir, basin_vmap, resolution, nodata, missthresh, variable,
        flowstate
        ):
    """Reads a datacube (t, x, y) and generates a time series.

    Parameters:
        input_dir : string
        basin_vmap : string
        resolution : float
        nodata : float
        missthresh : float
        variable : string
        flowstate : string

    Outputs:
    """
    paths = list_files(
        parent_dir=input_dir,
        ext=['.nc', '.nc4']
        )

    data = xr.open_mfdataset(paths)
    data = data.rename(name_dict={'__xarray_dataarray_variable__': variable})
    mask = vector2array(basin_vmap, resolution, nodata)
    inregion_cells = (mask == 1).sum()
    min_cells = inregion_cells * missthresh
    basin_area = inregion_cells * (resolution**2 / 1e6)

    if flowstate == 'flow':
        accum = (data[variable] * resolution**2).sum(['east', 'north'])
        data_aggregated = accum / (basin_area * 1e6)

    elif flowstate == 'state':
        data_aggregated = data[variable].mean(['east', 'north'])

    cells_per_date = data[variable].notnull().sum(['east', 'north'])

    time_series = xr.where(
        cells_per_date < min_cells, nan, data_aggregated
        ).to_series()

    return(time_series.reindex(sorted(time_series.index)))


def sflo_time_series(input_file, basin_vmap, resolution, nodata):
    data = xr.open_dataset(input_file)
    data = data['disc_filled'].rename('observed')
    data = data * (24 * 60 * 60) * 1000   # Convert m3/s to mm/d
    mask = vector2array(basin_vmap, resolution, nodata)
    inregion_cells = (mask == 1).sum()
    basin_area = inregion_cells * (resolution**2 / 1e6)  # Prevent overflow.
    time_series = (data / (basin_area * 1e6)).to_series()
    time_series.index = time_series.index + pd.Timedelta(8, 'h')
    return(time_series.reindex(sorted(time_series.index)))


def scale_data(input_data, scale=10):
    """Scale input data following a given method (resample or rolling).

    PARAMETERS
        input_data: pandas.Series
            The input time series in a form of a pandas.Series.
        scale: string or integer (optional)
            Temporal scale to which transform the input time series. It
            is formatted as pandas offset strings (to learn more about
            the offset strings, see https://bit.ly/2C4P9ig). Default is
            '1D'. Note: if method is 'roll', scale must be a fixed
            frequency.
        method: string (optional)
            Defines the method by which the input time series is
            scaled. The options are 'resample' and 'roll'. Default is
            'resample'.
        function: string (optional)
            Method for down/re-sampling. Default is 'sum'. Flux
            variable might use 'sum', while state variables mihgt use
            'mean'.
    """
    scaler = input_data.rolling(window=scale, center=True)
    return(scaler.mean())


def esaccism_daily2annual_storing(
        input_dir, output_dir, xmin, ymin, xmax, ymax
        ):
    daily_paths = list_files(
        parent_dir=input_dir,
        ext=['.nc', '.nc4']
        )

    for year in range(1978, 2019):
        year_paths = [i for i in daily_paths if ('-' + str(year)) in i]

        data = xr.open_mfdataset(
            year_paths,
            autoclose=True
            )

        clip_study_area = data.where(
            (data.lat >= ymin) &
            (data.lat <= ymax) &
            (data.lon >= xmin) &
            (data.lon <= xmax),
            drop=True
            )

        output_name = output_dir + '/' + str(year) + '.nc4'
        clip_study_area.to_netcdf(path=output_name)


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
        sys.stdout.write("\r    {} ({:.1f} % of {} processed)".format(
                message, progress * 100, units))

    else:
        progress = float(current)/total
        sys.stdout.write("\r    {} ({:.1f} % processed)".format(
                message, progress * 100))

    if progress < 1:
        sys.stdout.flush()

    else:
        sys.stdout.write('\n')
