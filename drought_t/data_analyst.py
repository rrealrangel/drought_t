#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 18:27:56 2019

@author: realrangel
"""
import numpy as _np
from scipy import stats as _stats


def smooth_data(input_data, agg_type='mean', agg_scale=7):
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
    scaler = input_data.rolling(
        window=agg_scale,
        center=True
        )

    if agg_type == 'mean':
        return(scaler.mean())

    elif agg_type == 'sum':
        return(scaler.sum())


def trend_is_significant(x, method='m-k', alpha=0.05):
    """Define if data is trended and wether it is significant or not.

    Parameters:
        x: pandas.Series
            Input data (full record available).
        method: string; optional
            Statistical tecnique to detect a trend in the data.
            Currently, it is available only the Mann-Kendall test
            ('m-k'). Default is 'm-k'.

    Output:
        trend: tells the trend (increasing, decreasing or no trend)
        h: True (if trend is present) or False (if trend is absence)
        p: p value of the significance test
        uc: normalized test statistics

    Some parts of this code are based on the function mk_test by Jonny
        Brooks-Bartlett, Michael Schramm, and Kevin Fichter published
        in https://github.com/mps9506/Mann-Kendall-Trend/blob/master
        /mk_test.py.

    References
        Machiwal, D. & Jha, M. K. (2012). Hydrologic time series
            analysis: Theory and practice. New Delhi: Springer.
        Salas, J. (1993). Analysis and modeling of hydrologic time
            series. In D. R. Maidment (Ed.), Handbook of hydrology (pp.
            19.2-19.72). U. S. A.: McGraw-Hill Inc.
    """
    N = len(x.notnull()) - 1

    if method == 'm-k':
        # Apply the Mann-Kendall test.
        def zk(x, j, k):
            sgn = _np.sign(x[j] - x[k])

            if _np.isnan(sgn):
                return(0)

            else:
                return(sgn)

        k = [zk(x, j, k) for k in range(N - 1) for j in range(k + 1, N)]
        S = sum(k)
        gr_data = [
            len([i for i in x if i == j])
            for j in list(set(x[x.duplicated(keep=False)]))
            ]
        V = 1.0 / 18 * (N * (N - 1) * (2 * N + 5) - sum(
            [e * (e - 1) * (2 * e + 5) for e in gr_data]
            ))
        uc = (S + (_np.sign(S) * -1)) / _np.sqrt(V)
        # p = 2 * (1 - _stats.norm.cdf(abs(uc)))  # two tail test
        h = abs(uc) > _stats.norm.ppf(1 - (alpha / 2))

        # if (uc < 0) and h:
        #     trend = 'upwards'

        # elif (uc > 0) and h:
        #     trend = 'downwards'

        # else:
        #     trend = 'no trend'

        return(h)


def detrend_lineal(data):
    x = _np.arange(len(data))
    m, b, r_val, p_val, std_err = _stats.linregress(
        x=x[data.notnull()],
        y=data[data.notnull()]
        )
    detrended = data.copy() - (m * x + b)
    return(detrended)
