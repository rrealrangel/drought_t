# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 12:59:58 2019

@author: r.realrangel
"""
import numpy as np
import pandas as pd
from drought_t import quadratic_lowess as ql


def smoothvar(x, method='lowess', **kwargs):
    """
    Smoothes the raw records of a variable.

    References:
    Cleveland, W. S. (1979). Robust Locally Weighted Regression and
        Smoothing Scatterplots. Journal of the American Statistical
        Association, 9.
    Cleveland, W. S., & Devlin, S. J. (1988). Locally Weighted
        Regression: An Approach to Regression Analysis by Local
        Fitting. Journal of the American Statistical Association,
        83(403), 596–610. Retrieved from http://links.jstor.org
        /sici?sici=0162-1459%28198809%2983%3A403%3C596%3ALWRAAT%3E2.0.
        CO%3B2-Y
    Durre, I., Squires, M. F., Vose, R. S., Applequist, S., & Yin,
        X. (2011). Computational Procedures for the 1981-2010
        Normals: Precipitation, Snowfall, and Snow Depth.

    Parameters
    ----------
    x : pandas.Series
        The time series of the input data.
    window : int, optional
        Size of the window centered on each day of the year used to
        choose the values from which the mean will be computed. By
        default, 29.

    Returns
    -------
    pandas.Series
        The time series of the variable of interest (x) with the same
        length as the input x.
    """
    if method == 'ma':
        window = kwargs['window']
        min_periods_r = kwargs['min_periods_r']
        x_smooth = x.rolling(
            window=window,
            center=True,
            min_periods=int(np.ceil(window * min_periods_r))
            ).mean()

    elif method == 'lowess':
        ftries = kwargs['ftries']
        x_smooth = x.copy()
        oneyear = x[x.index.year == x.index.year[1]]
        ydata = oneyear[np.isfinite(oneyear)]
        xdata = np.arange(1, len(ydata) + 1).astype(dtype=float)
        f = ql.optimal_f(
            x=xdata,
            y=ydata.values,
            ftries=ftries
            )
        ydata_lowess = ydata.copy()
        ydata_lowess[:] = ql.quadratic_lowess(
            x=xdata,
            y=ydata,
            f=f
            )

        for date in ydata_lowess.index:
            x_smooth[
                (x_smooth.index.month == date.month) &
                (x_smooth.index.day == date.day)
                ] = ydata_lowess[date]

    return(x_smooth)


def normedian(
        x, q=0.5, window=29, max_zero=0.0, min_notnull=0.667,
        min_wet_proportion=0.1, min_len=15, smoothing_method='lowess',
        **kwargs
        ):
    """
    Compute 50th percentile of long-term daily records of a variable.

    References:
    Durre, I., Squires, M. F., Vose, R. S., Applequist, S., & Yin,
        X. (2011). Computational Procedures for the 1981-2010
        Normals: Precipitation, Snowfall, and Snow Depth.

    Parameters
    ----------
    x : pandas.Series
        The time series of the input data.
    q : float, optional
        The value of the percentile to usa as threshold level. By
        default, 0.5.
    window : int, optional
        Size of the window centered on each day of the year used to
        choose the values from which the percentile will be computed.
        By default, 29.
    max_zero : float, optional
        Maximum dry value. Every value below or equal is considered to
        be zero. By default, 0.0.
    min_notnull : int, optional
        Minumum fraction of values available within the applicable window
        in a given year to be included in the analysis. It cannot be
        greater than window. By default, 0.667.
    min_wet_proportion : float, optional
        Minimum ratio of wet values of the chosen values to perform
        the computation of the percentile.

    Returns
    -------
    pandas.Series
        The time series of the base value (x0) with the same length as
        the input x.
    """
    # Prepare the output pandas.Series.
    x0 = x.copy() * np.nan

    for date in pd.date_range(start='1981-01-01', end='1981-12-31'):
        data_window = pd.Series()

        # Extract the values in the corresponding date window.
        for year in list(set(x.index.year - 1)):
            window_first = date - pd.Timedelta(
                value=window / 2,
                unit='D'
                )
            window_first = pd.datetime(
                year=year,
                month=window_first.month,
                day=window_first.day
                )
            window_last = window_first + pd.Timedelta(
                value=window,
                unit='D'
                )
            data_window = data_window.append(
                to_append=x[
                    (x.index >= window_first) &
                    (x.index <= window_last)
                    ]
                )

        # Remove February 29 values.
        data_window[
            (data_window.index.month == 2) &
            (data_window.index.day == 29)
            ] = np.nan

        # Remove years with fewer than min_notnull values.
        data_subset = data_window.groupby(
            by=data_window.index.year
            ).filter(lambda group: group.count() > (min_notnull * window))

        # Remove zero-values.
        data_subset_wet = data_subset[data_subset > max_zero]

        # Compute the x0 only if, at least, min_wet_proportion of the chosen
        # values are nonzero.
        if len(data_subset_wet) / float(len(data_subset)) > min_wet_proportion:
            x0[
                (x0.index.month == date.month) &
                (x0.index.day == date.day)
                ] = data_subset_wet.quantile(q=q)

    # Set February 29 values as the mean of their corresponding
    # February 28 and March 1.
    x0[(x0.index.month == 2) & (x0.index.day == 29)] = np.mean([
        x0['1981-02-28'], x0['1981-03-01']
        ])

    # Smooth the resulting x0.
    if smoothing_method == 'lowess':
        ftries = kwargs['ftries']
        x0 = smoothvar(
            x=x0,
            method=smoothing_method,
            ftries=ftries
            )

    elif smoothing_method == 'ma':
        smoothing_window = kwargs['smoothing_window']
        smoothing_min_periods_r = kwargs['smoothing_min_periods_r']
        x0 = smoothvar(
            x=x0,
            method=smoothing_method,
            window=smoothing_window,
            min_periods_r=smoothing_min_periods_r
            )

    # Remove stretches shorter than min_len.
    def remove_shorts(group):
        if 1 < len(group) < min_len:  # ??? <- Just "len(group) < min_len"?
            return(group * np.nan)

        else:
            return(group)

    x0 = x0.groupby(by=x0.isna().cumsum()).transform(remove_shorts)
    return(x0)


def lynehollick(x, a=0.925, reflection=30, passes=3):
    """
    References:
        Ladson, A., Brown, R., Neal, B. & Nathan, R. (2013). A standard
        approach to baseflow separation using the Lyne and Hollick
        filter. Australian Journal of Water Resources, 17(1). https://
        doi.org/10.7158/W12-028.2013.17.1.

    """
    if len(x) < 65:
        return(x * pd.np.nan)

    else:
        x_notna = x[x.notna()]
        x_warm = pd.Series(   # <- Warm up period.
            data=list(reversed(x_notna.iloc[1: reflection + 1].values)),
            index=pd.date_range(
                start=x_notna.index[0] - pd.Timedelta(reflection, 'D'),
                periods=reflection
                )
            )
        x_cool = pd.Series(   # <- Cool down period.
            data=list(reversed(x_notna.iloc[-reflection - 1: -1].values)),
            index=pd.date_range(
                start=x_notna.index[-1] + pd.Timedelta(1, 'D'),
                periods=reflection
                )
            )
        q = pd.concat(objs=[x_warm, x_notna, x_cool])
        qf = q.copy() * pd.np.nan
        qb = q.copy() * pd.np.nan

        for _pass in range(passes):
            if _pass == 0:
                qin = q.copy()

            else:
                qin = qb.copy()

            if (_pass + 1) % 2 == 1:
                it = range(len(q))

            else:
                it = range(len(q) - 1, -1, -1)

            for i in range(len(it)):
                if i == 0:
                    qf[it[i]] = qin[it[i]]

                else:
                    qf[it[i]] = (   # <- Filter
                        (a * qf[it[i - 1]]) +
                        (((1 + a) / 2) * (qin[it[i]] - qin[it[i - 1]]))
                        )

                if qf[it[i]] <= 0:
                    qf[it[i]] = 0

                qb[it[i]] = qin[it[i]] - qf[it[i]]

        return(qb.reindex(index=x.index))
