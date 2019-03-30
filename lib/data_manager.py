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
from numpy import nan
from pathlib2 import Path
import gdal
import ogr
import pandas as pd
import toml
import xarray as xr


class Configurations():
    """
    """
    def __init__(self, config_file):
        self.config_file = config_file

        with open(self.config_file, 'rb') as file_content:
            config = toml.load(file_content)

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

    return(files_list)


def read_input(input_file):
    """Retrieve a time series from a .csv file.

    Parameters:
        input_file : string
            The ful path of the input .csv file."""
    return(
        pd.read_csv(
            filepath_or_buffer=input_file,
            index_col='time',
            parse_dates=True
            )
        )


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


def grid_time_series(
        input_dir, basin_vmap, resolution, nodata, missthresh, variable,
        flowstate):
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
    paths = list_files(parent_dir=input_dir, ext=['.nc', '.nc4'])
    data = xr.open_mfdataset(paths)
    data = data.rename(name_dict={data.keys()[0]: variable})
    mask = vector2array(basin_vmap, resolution, nodata)
    inregion_cells = (mask == 1).sum()
    min_cells = inregion_cells * missthresh
    basin_area = inregion_cells * resolution**2

    if flowstate == 'flow':
        accum = (data[variable] * resolution**2).sum(['east', 'north'])
        data_aggregated = accum / basin_area

    elif flowstate == 'state':
        data_aggregated = data[variable].mean(['east', 'north'])

    cells_per_date = data[variable].notnull().sum(['east', 'north'])
    output_dataframe = xr.where(
        cells_per_date < min_cells, nan, data_aggregated
        ).to_dataframe()
    return(output_dataframe.reindex(sorted(output_dataframe.index)))


def sflo_time_series(input_dir, basin_vmap, resolution, nodata):
    # TODO: Generalize this function.
    paths = list_files(parent_dir=input_dir, ext=['.nc', '.nc4'])
    data = xr.open_mfdataset(paths)
    dropvars = [i for i in data.keys() if i != data.keys()[0]]
    data = data.rename(name_dict={data.keys()[0]: 'Streamflow'})
    data = data.drop(labels=dropvars)
    data = data * (24 * 60 * 60) * 1000   # Convert m3/s to mm/d
    mask = vector2array(basin_vmap, resolution, nodata)
    inregion_cells = (mask == 1).sum()
    basin_area = inregion_cells * resolution**2
    output_dataframe = (data[data.keys()[0]] / basin_area).to_dataframe()
    output_dataframe.index = output_dataframe.index + pd.Timedelta(8, 'h')
    return(output_dataframe.reindex(sorted(output_dataframe.index)))


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
