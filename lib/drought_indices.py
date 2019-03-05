#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 12:13:26 2019

@author: rrealrangel
"""

from lib.threshold_level_method import DroughtIndicator
from scipy.stats import norm
import numpy as np
import pandas as pd


def npsdi(data, a_coef=0.44):
    """Computes the non-parametric univariate standardized index of a
    given climate and land-surface variable (precipitation, soil
    moisture, relative humidity, evapotranspiration, etc.) without
    having to assume the existence of representative parametric
    probability distribution functions.

    Here, the empirical probability of the elements of a time series
    is computed using the plotting position general form:

        P  = (i - a) / (n + 1 - (2 * a))

    where i is the rank of the item, n is the size of the sample
    and a is a coeficient to be proposed. Some values of a found in
    literature are:

        a = 0.000 (Weibull, 1939)
        a = 0.375 (Blom, 1958)
        a = 0.400 (Cunnane, 1978)
        a = 0.440 (Gringorten, 1963)
        a = 0.500 (Hazen, 1913)

    REFERENCES
        Blom, Gunnar (1958). Statistical estimates and transformed
            beta-variables. In: Almquist und Wiksell, pp. 68-75, 143-
            146.
        Cunnane, C. (1978). Unbiased plotting positions - A review. In:
            Journal of Hydrology 37, pp. 205-222. doi: 10.1016/0022-
            1694(79)90120-3.
        Farahmand, A., & AghaKouchak, A. (2015). A generalized
            framework for deriving nonparametric standardized drought
            indicators. Advances in Water Resources, 76, 140–145.
            https://doi.org/10.1016/j.advwatres.2014.11.012
        Gringorten, Irving I. (1963). A plotting rule for extreme
            probability paper. In: Journal of Geophysical Research
            68.3, pp. 813. doi: 10.1029/JZ068i003p00813.
        Hazen, Allen (1913). Storage to be provided impounding
            reservoirs for municipal water supply. In: Proceedings of
            the American Society of Civil Engineers 39.9, pp. 1943-
            2044.
        Weibull, Ernst Hjalmar Waloddi (1939). The Phenomenon of
        Rupture in Solids. Stockholm.

    PARAMETERS
        data: pandas.series
            Time series. It contains as many rows as time steps and as
            many columns as variables.
        a_coef: float (default is 0.44)
            Parameter a in the general formula for plotting possitions.
    RETURNS
        numpy.ndarray
            Sequence of values representing the empirical probability
            of each element of the time series.
    """
    def plotpos(data, a_coef):
        sample_size = data.notnull().sum()
        rank = data.rank(method='dense')
        return((rank - a_coef) / (sample_size + 1 - (2 * a_coef)))

    dates = pd.date_range(start='1980-01-01', end='1980-12-31')
    to_concatenate = []

    for date in dates:
        data_by_date = data[
                (data.index.month == date.month) &
                (data.index.day == date.day)]
        probability = plotpos(data=data_by_date, a_coef=a_coef)
        to_concatenate.append(probability.apply(norm.ppf))

    return(DroughtIndicator(pd.concat(to_concatenate).sort_index()))


def anomaly(data, method='median', threshold=0.1):
    """Returns the variable reference level of drought for a variable
    time series for a given reference used in the standardized drought
    indices.

    PARAMETERS
        data: pandas.series
            Contains the records of the variable analyzed.
        method: string (default is 'median')
            Defines wether to use 'mean' or 'median' as central
            tendency measure as reference of normality.
    """
    dates = pd.date_range(start='1980-01-01', end='1980-12-31')
    threshold = data.copy()

    for date in dates:
        data_by_date = data[
                (data.index.month == date.month) &
                (data.index.day == date.day)]

        if method == 'median':
            threshold[
                (data.index.month == date.month) &
                (data.index.day == date.day)] = float(
                    data_by_date.quantile(q=threshold))

        elif method == 'mean':
            threshold[
                (data.index.month == date.month) &
                (data.index.day == date.day)] = float(
                    data_by_date.mean())

    return(DroughtIndicator(data - threshold))
