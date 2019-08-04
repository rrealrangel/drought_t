#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""THRESHOLD LEVEL METHOD FOR DEFINING DROUGHT EVENTS

DESCRIPTION
    The threshold level method derive the drought characteristics
    directly from time series of observed or simulated
    hydrometeorological variables using the predefined threshold level.
    When the variable is below this level, the site is in drought.
    Drought duration, severity and frequency can easily be calculated.

AUTHOR
    Roberto A. Real-Rangel (Institute of Engineering UNAM; Mexico)

LICENSE
    GNU General Public License
"""
import sys as _sys

import numpy as _np
import pandas as _pd

import data_manager as dmgr


def base_value(x, window=29, min_notnull=20, min_nonzero=0.1):
    """
    Compute the base value (x0) that will cut the sequence x into runs.
    The base value (x0) cuts the series of interest in many places and
    the relationship of the value x0 to all other values of x serves as
    the basis for the following definitions of runs (Yevjevitch, 1967):
        * excess period duration: distance between the successive
            upcross and downcross;
        * drought duration: distance between the successive downcross
            and upcross;
        * excess magnitude: sum of positive deviations, between the
            successive upcross and downcross; and
        * drought magnitude: sum of negative deviations, between the
            successive downcross and upcross.

    Here, x0 is defined as median of nonzero daily observations,
    computed following the procedure of Durre et al. (2011).

    References:
    Durre, I., Squires, M. F., Vose, R. S., Applequist, S., & Yin,
        X. (2011). Computational Procedures for the 1981-2010
        Normals: Precipitation, Snowfall, and Snow Depth.
    Yevjevich, V. (1967). An objective approach to definitions and
        investigations of continental hydrologic droughts.
        Hydrology Papers 23, (23), 25–25. https://doi.org/10.1016
        /0022-1694(69)90110-3.

    Parameters
    ----------
    x : pandas.Series
        The time series of the input data.
    window : int, optional
        Size of the window centered on each day of the year used to
        choose the values from which the median will be computed. By
        default, 29.
    min_notnull : int, optional
        Minumum number of values available within the applicable window
        in a given year to be included in the analysis. It cannot be
        greater than window. By default, 20.
    min_nonzero : float, optional
        Minimum ratio of nonzero values of the chosen values to perform
        the computation of the median.

    Returns
    -------
    pandas.Series
        The time series of the base value (x0) with the same length as
        the input x.

    Raises
    ------
    ValueError
        When min_notnull is greater than window.
    """
    # Check if min_notnull is lower than window.
    if min_notnull > window:
        raise(ValueError("min_notnull value is greater than window."))

    # Prepare the output pandas.Series.
    x0 = x.copy() * _np.nan

    for date in _pd.date_range(start='1981-01-01', end='1981-12-31'):
        data_window = _pd.Series()

        # Extract the values in the corresponding date window.
        for year in list(set(x.index.year - 1)):
            window_first = date - _pd.Timedelta(window / 2, 'D') - 1
            window_first = _pd.datetime(
                year=year,
                month=window_first.month,
                day=window_first.day
                )
            window_last = window_first + _pd.Timedelta(window, 'D')
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
            ] = _np.nan

        # Remove years with fewer than min_notnull values.
        data_subset = data_window.groupby(
            by=data_window.index.year
            ).filter(lambda group: group.count() > min_notnull)

        # Remove zero-values.
        data_subset_nonzero = data_subset[data_subset > 0]

        # Compute the x0 only if, at least, min_nonzero of the chosen
        # values are nonzero.
        if len(data_subset_nonzero) / float(len(data_subset)) > min_nonzero:
            x0[
                (x0.index.month == date.month) &
                (x0.index.day == date.day)
                ] = data_subset_nonzero.median()

    # Set February 29 values as the mean of their corresponding
    # February 28 and March 1.
    x0[(x0.index.month == 2) & (x0.index.day == 29)] = _np.mean([
        x0['1981-02-28'], x0['1981-03-01']
        ])

    # Smooth the resulting x0.
    x0 = x0.rolling(
        window=window,
        center=True,
        min_periods=(window / 2)
        ).mean()

    return(x0)


def anomaly(x, x0, window=29):
    """
    Compute the anomalies in a series x relative to a base value x0.

    Parameters
    ----------
    x : pandas.Series
        Time series of the input data.
    x0 : pandas.Series
        Time series of the base value (x0).
    window : integer, optional
        Size of the window centered on each day of the year used to
        aggregate the input x values and to choose the values from
        which the median will be computed. By default, 29.

    Output
    ------
    pandas.Series
    """
    x_aggregated = x.rolling(
        window=window,
        center=True,
        min_periods=(window / 2)
        ).mean()
    return(x_aggregated - x0)


def _sign_wo_zero(value):
    """
    Sign of a value. This function applies numpy.sign(), with the
    difference that 0 is interpedted as negative values (-1).

    Parameters
    ----------
    value : float
        Input value to which define the sign.

    Output
    ------
    integer
        The sign of the input value. -1, if value is negative or zero;
        and 1, is value is positive.
    """
    if _np.sign(value) == 0:
        return(1)

    else:
        return(_np.sign(value))


def _sign_grouper(anomalies):
    """
    References:
    Tallaksen, L. M., Madsen, H., & Clausen, B. (1997). On the
        definition and modelling of streamflow drought duration
        and deficit volume. Hydrological Sciences Journal,
        42(1), 15–33. https://doi.org/10.1080/
        02626669709492003.
    Tallaksen, L. M., & van Lanen, H. A. J. (Eds.). (2004).
        Hydrological Drought: Processes and Estimation Methods
        for Streamflow and Groundwater. Elsevier Inc. Retrieved
        from http://europeandroughtcentre.com/resources/
        hydrological-drought-1st-edition/

    Parameters
    ----------
    anomalies : pandas.Series
        Series of anomalies.

    Output
    ------
    pandas.SeriesGroupBy
    """
    sign = _np.sign(anomalies)
    sign[sign == 0] = 1
    runs = (sign != sign.shift(1)).astype(int).cumsum()
    runs[anomalies.isnull()] = _np.nan
    runs = runs - (runs.min() - 1)
    return(anomalies.groupby(runs))


def get_runs(anomalies):
    """
    Parameters
    ----------
    anomalies : pandas.Series
        Series of anomalies.

    Output
    ------
    dict
        Runs identified in the input series of anomalies.
    """
    return({
        name: _sign_grouper(anomalies=anomalies).get_group(name=name)
        for name in _sign_grouper(anomalies=anomalies).indices
        })


def pool_runs(runs, pooling_method=None, show_positives=False, **kwargs):
    """
    Parameters
    ----------
    runs : dict
    pooling_method : str, optional
    show_positives : bool, optional

    Output
    ------
        pandas.Series
    """
    if pooling_method is None:
        # Not pooling runs.
        runs_pooled = runs

    elif pooling_method == 'ma':
        # Pooling runs through the moving average method.
        x = kwargs['x']
        ma_window = kwargs['ma_window']
        bv_window = kwargs['bv_window']
        bv_min_notnoull = kwargs['bv_min_notnoull']
        bv_min_nonzero = kwargs['bv_min_nonzero']

        # Compute runs for the transformed input data.
        ma_x = x.rolling(
            window=ma_window,
            center=True,
            min_periods=ma_window
            ).mean()
        ma_x0 = base_value(
            x=ma_x,
            window=bv_window,
            min_notnull=bv_min_notnoull,
            min_nonzero=bv_min_nonzero
            )
        ma_anomalies = anomaly(
            x=ma_x,
            x0=ma_x0,
            window=bv_window
            )
        ma_runs = get_runs(ma_anomalies)
        runs_pooled = {}
        counter = 1
        counter2 = 0

        for ma_num, ma_run in ma_runs.items():
            candidates_to_pool = []

            for num, run in runs.items():
                if run.index[0] <= ma_run.index[-1]:
                    for date in ma_run.index:
                        if date in run.index:
                            candidates_to_pool.append(run)
                            break

                    else:
                        pass

            runs_to_pool = [
                run
                for run in candidates_to_pool
                if _sign_wo_zero(run.sum()) == _sign_wo_zero(ma_run.sum())
                ]

            if len(runs_to_pool) > 0:
                pooled = _pd.concat(objs=runs_to_pool)

                if counter == 1:
                    runs_pooled[counter] = _pd.concat(
                        objs=runs_to_pool
                        )

                    counter += 1

                elif pooled.index[0] != runs_pooled[counter - 1].index[0]:
                    runs_pooled[counter] = _pd.concat(
                        objs=runs_to_pool
                        )

                    counter += 1

            counter2 += 1
            dmgr.progress_message(
                current=(counter2),
                total=len(ma_runs),
                message="- Pooling runs",
                units='runs'
                )

    elif pooling_method == 'sp':
        # Sequent peak algorithm method.
        pass

    elif pooling_method == 'ic':
        # Inter-event time and volume criterion method.
        pass

    if show_positives:
        return(_pd.Series(
            data=runs_pooled,
            name='anomaly'
            ))

    else:
        return(_pd.Series(
            data={
                num: run
                for num, run in runs_pooled.items()
                if run.sum() < 0
                },
            name='anomaly'
            ))


def runs_onset(runs):
    """Extract the onset of each run.

    Parameters
    ----------
    runs : pandas.Series
        The runs time series (as obtained with get_runs())

    Output
    ------
    pandas.Series
    """
    return(_pd.Series(
        data={
            num: run.index[0]
            for num, run in runs.items()
            },
        name='onset_date'
        ))


def runs_end(runs):
    return(_pd.Series(
        data={
            num: run.index[-1]
            for num, run in runs.items()
            },
        name='end_date'
        ))


def runs_sum(runs):
    """
    Computes the sum of all deviations between successive downcrosses
    and upcrosses. Indicates the deficiency of water in the variable
    analyzed or the severity of drought.

    References:
    Yevjevich, V. (1967). An objective approach to definitions and
        investigations of continental hydrologic droughts.
        Hydrology Papers 23, (23), 25–25. https://doi.org/10.1016
        /0022-1694(69)90110-3

    Parameters
    ----------
    runs : pandas.Series
        The runs time series (as obtained with get_runs())

    Output
    ------
    pandas.Series
    """
    return(_pd.Series(
        data={index: runs[index].sum() for index in runs.index},
        name='length'
        ))


def runs_length(runs):
    """
    Computes the distance between successive downcrosses and upcrosses.
    The run-length represents the duration of a drought.

    References:
        Yevjevich, V. (1967). An objective approach to definitions and
            investigations of continental hydrologic droughts.
            Hydrology Papers 23, (23), 25–25. https://doi.org/10.1016
            /0022-1694(69)90110-3

    Parameters
    ----------
    runs : pandas.Series
        The runs time series (as obtained with get_runs())

    Output
    ------
    pandas.Series
    """
    return(_pd.Series(
        data={index: runs[index].count() for index in runs.index},
        name='length'
        ))


def sum_length_ratio(runs):
    """
    Parameters
    ----------
    runs : pandas.Series
        The runs time series (as obtained with get_runs())

    Output
    ------
    pandas.Series
    """
    return(runs_sum(runs) / runs_length(runs))


def run_peak(runs):
    """
    Returns the maximum absolute value in the run.

    Parameters
    ----------
    runs : pandas.Series
        The runs time series (as obtained with get_runs())

    Output
    ------
    pandas.Series
    """
    return(_pd.Series(
        data={index: runs[index].min() for index in runs.index},
        name='peak'
        ))


def runs_cumsum(runs):
    """
    Parameters
    ----------
    runs : pandas.Series
        The runs time series (as obtained with get_runs())

    Output
    ------
    pandas.Series
    """
    return(_sign_grouper(
        _pd.concat(objs=[v for k, v in runs.items()])
        ).cumsum())
