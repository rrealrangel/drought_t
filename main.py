#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""Quality control routines. Data management.
    Author
        Roberto A. Real-Rangel (Institute of Engineering UNAM; Mexico)
    License
        GNU General Public License
"""
from scipy.stats import norm
import numpy as np
import pandas as pd
import toml

with open('config.toml', 'rb') as fin:
    config = toml.load(fin)


def read_input(input_file):
    return(pd.read_csv(
            filepath_or_buffer=input_file,
            index_col='time',
            parse_dates=True))


def empirical_probability(input_time_series, a_coef=0.44):
    """Computes the empirical probability of the elements of a time
        series using the plotting position general form as follows:

            P  = (i - a) / (n + 1 - (2 * a))

        where i is the rank of the item, n is the size of the sample
        and a is a coeficient to be proposed. Some values of a found in
        literature are:

            a = 0.000 (Weibull, 1939)
            a = 0.375 (Blom, 1958)
            a = 0.400 (Cunnane, 1978)
            a = 0.440 (Gringorten, 1963)
            a = 0.500 (Hazen, 1913)

        References:
            Blom, Gunnar (1958). Statistical estimates and transformed
                beta-variables. In: Almquist und Wiksell, pp. 68-75,
                143-146.
            Cunnane, C. (1978). Unbiased plotting positions - A review.
                In: Journal of Hydrology 37, pp. 205-222. doi:
                10.1016/0022-1694(79)90120-3.
            Gringorten, Irving I. (1963). A plotting rule for extreme
                probability paper. In: Journal of Geophysical Research
                68.3, pp. 813. doi: 10.1029/JZ068i003p00813.
            Hazen, Allen (1913). Storage to be provided impounding
                reservoirs for municipal water supply. In: Proceedings
                of the American Society of Civil Engineers 39.9, pp.
                1943-2044.
            Weibull, Ernst Hjalmar Waloddi (1939). The Phenomenon of
                Rupture in Solids. Stockholm.

    Parameters
    ----------
        input_time_series : pandas.Dataframe
            Time series. It contains as many rows as time steps and as
            many columns as variables.
        a_coef : float
            Parameter a in the general formula for plotting possitions.
    Returns
    -------
        numpy.ndarray
            Sequence of values representing the empirical probability
            of each element of the time series.
    """
    sample_size = input_time_series.notnull().sum()
    rank = input_time_series.rank(method='dense')
    return((rank - a_coef) / (sample_size + 1 - (2 * a_coef)))


def compute_npsdi(data, time_scale=30):
    """Computes the non-parametric standardized drought index of a
    given variable, with a specified time scale.

    Parameters
        data: pandas.Dataframe
        time_scale: integer (default is 30)
            In days.
    """
    data_scaled = data.rolling(window=time_scale).sum()
    dates = pd.date_range(start='1980-01-01', end='1980-12-31')
    to_concatenate = []

    for date in dates:
        data_by_date = data_scaled[
                (data_scaled.index.month == date.month) &
                (data_scaled.index.day == date.day)]
        probability = empirical_probability(input_time_series=data_by_date)

        if int(probability.notnull().sum()) < 30:
            to_concatenate.append(probability.apply(norm.ppf) * np.nan)

        else:
            to_concatenate.append(probability.apply(norm.ppf))

    return(pd.concat(to_concatenate).sort_index())


def main():
    prec = read_input(config['inputs']['prec_file'])
    disc = read_input(config['inputs']['disc_file'])
    spi = compute_npsdi(data=prec)
    sri = compute_npsdi(data=disc)
