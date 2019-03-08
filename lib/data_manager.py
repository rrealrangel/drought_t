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
import numpy as np
import pandas as pd
import toml


class Configurations():
    """
    """
    def __init__(self, config_file):
        self.config_file = config_file

        with open(self.config_file, 'rb') as file_content:
            config = toml.load(file_content)

        for key, value in config.items():
            setattr(self, key, value)


def read_input(input_file):
    """Retrieve a time series from a .csv file.

    PARAMETERS
        input_file: string
            The ful path of the input .csv file."""
    return(pd.read_csv(
            filepath_or_buffer=input_file,
            index_col='time',
            parse_dates=True))


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
