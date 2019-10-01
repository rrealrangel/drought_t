#!/usr/bin/env python
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
import numpy as np
import pandas as pd

from drought_t import data_manager as dmgr
from drought_t import quadratic_lowess as ql


def smooth_variable(x, method='lowess', **kwargs):
    """
    Computes a moving average to the raw records of a variable (e. g.,
    precipitation, streamflow, etc.) to make comparable to the
    reference level computed with threshold_level().

    References:
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
        x_smooth = x.copy()
        oneyear = x[x.index.year == x.index.year[1]]
        ydata = oneyear[np.isfinite(oneyear)]
        xdata = np.arange(1, len(ydata) + 1).astype(dtype=float)
        f = ql.optimal_f(
            x=xdata,
            y=ydata.values
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


def threshold_level(
        x, q=0.5, window=29, max_zero=0.0, min_notnull=0.667,
        min_wet_proportion=0.1, min_len=15
        ):
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

    Here, x0 is defined as a percentile (q) of wet daily
    observations, computed following the procedure of Durre et al.
    (2011).

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
    x0 = smooth_variable(x0, method='lowess')

    # Remove stretches of percentiles shorter than min_len.
    def remove_minor(group):
        if 1 < len(group) < min_len:  # Shouldnt it be "len(group) < min_len"?
            return(group * np.nan)

        else:
            return(group)

    x0 = x0.groupby(by=x0.isna().cumsum()).transform(remove_minor)
    return(x0)


def _sign_grouper(anomalies):
    """Groups the values of a series according to their sign.

    Parameters
    ----------
    anomalies : pandas.Series
        Series of anomalies.

    Output
    ------
    pandas.SeriesGroupBy
    """
    sign = np.sign(anomalies)
    sign[sign == 0] = 1
    runs = (sign != sign.shift(1)).astype(int).cumsum()
    runs[anomalies.isnull()] = np.nan
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


def pool_runs(runs, pooling_method='None', **kwargs):
    """
    Parameters
    ----------
    x : pandas.Series
        Time series of the variable analyzed.
    x0 : pandas.Series
        Time series of the threshold level for the variable x.
    pooling_method : str, optional
        Pooling method to apply. For not applying any pooling, enter
        'none' (default). The other options are: the moving average
        method ('ma'), and the inter-event time and volume criterion
        method ('ic').
    kwargs : keyword arguments
        runs_ma: the runs derived from a smoothed time series of the
            variable of interest. Only used if pooling_method == 'ma'.
        tc: the inter-event critic duration (in days). Only used if
            pooling_method == 'ic'.
        pc: the critical ratio. Only used if pooling_method == 'ic'.

    Output
    ------
        numpy.array
    """
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
        if np.sign(value) == 0:
            return(1)

        else:
            return(np.sign(value))

# =============================================================================
# Do not pool events.
# =============================================================================
    if pooling_method == 'None':
        # Not pooling runs.
        runs_pooled = runs

# =============================================================================
# Pooling runs through the moving average (MA) method.
# =============================================================================
    elif pooling_method == 'ma':
        runs_ma = kwargs['runs_ma']
        runs_pooled = {}
        counter = 1
        counter2 = 0

        for i_ma in sorted(runs_ma.keys()):
            run_ma = runs_ma[i_ma]
            candidates_to_pool = []

            for i in sorted(runs.keys()):
                run = runs[i]
                if run.index[0] <= run_ma.index[-1]:
                    for date in run_ma.index:
                        if date in run.index:
                            candidates_to_pool.append(run)
                            break

                    else:
                        pass

            runs_same_sign_to_pool = [
                i
                for i in candidates_to_pool
                if _sign_wo_zero(i.sum()) == _sign_wo_zero(run_ma.sum())
                ]

            if len(runs_same_sign_to_pool) > 0:
                runs_to_pool = [
                    i
                    for i in runs.values()
                    if ((i.index[0] >= runs_same_sign_to_pool[0].index[0]) &
                        (i.index[0] <= runs_same_sign_to_pool[-1].index[-1]))
                    ]

                pooled = pd.concat(objs=runs_to_pool)

                if counter == 1:
                    runs_pooled[counter] = pd.concat(
                        objs=runs_to_pool
                        )

                    counter += 1

                elif pooled.index[0] != runs_pooled[counter - 1].index[0]:
                    runs_pooled[counter] = pd.concat(
                        objs=runs_to_pool
                        )

                    counter += 1

            counter2 += 1
            dmgr.progress_message(
                current=(counter2),
                total=len(runs_ma),
                message="- Pooling runs",
                units='runs'
                )

        runs_pooled_tagged = {}

        for k, run in runs_pooled.items():
            if np.sign(run.sum()) == -1:
                peak_day = run.index[run == run.min()]

            else:
                peak_day = run.index[run == run.max()]

            if len(peak_day) > 1:
                peak_day = peak_day[int(np.ceil((len(peak_day) / 2.) - 1))]

            else:
                peak_day = peak_day[0]

            runs_pooled_tagged[peak_day] = runs_pooled[k]

        runs_concat = pd.concat(objs=[v for v in runs_pooled.values()])
        runs_to_add = [
            run
            for run in runs.values()
            if run.index[0] not in runs_concat.index
            ]

        for run in runs_to_add:
            if np.sign(run.sum()) == -1:
                peak_day = run.index[run == run.min()]

            else:
                peak_day = run.index[run == run.max()]

            if len(peak_day) > 1:
                peak_day = peak_day[int(np.ceil((len(peak_day) / 2.) - 1))]

            else:
                peak_day = peak_day[0]

            runs_pooled_tagged[peak_day] = run

        runs_pooled = runs_pooled_tagged

# =============================================================================
# Pooling runs through the inter-event time and volume criterion (IC) method.
# =============================================================================
    elif pooling_method == 'ic':
        tc = kwargs['tc']
        pc = kwargs['pc']
        runs_pooled = {}
        runs_key = 0

        while runs_key < len(runs):
            run_pooled = runs[sorted(runs.keys())[runs_key]]

            if np.sign(run_pooled.sum()) == -1:
                if runs_key < len(runs) - 2:
                    run_inter = runs[sorted(runs.keys())[runs_key + 1]]

                    # Test the pooling conditioning factors and pool the runs.
                    while ((run_inter.count() <= tc) &
                           (abs(run_inter.sum() / run_pooled.sum()) <= pc) &
                           (runs_key < len(runs) - 2)):
                        run_to_pool = runs[sorted(runs.keys())[runs_key + 2]]

                        if (((run_inter.index[0] -
                             run_pooled.index[-1]).days == 1) &
                            ((run_to_pool.index[0] -
                              run_inter.index[-1]).days == 1)):
                            run_pooled = pd.concat(
                                [run_pooled, run_inter, run_to_pool]
                                )
                            runs_key += 2

                            if runs_key < len(runs) - 2:
                                run_inter = (
                                    runs[sorted(runs.keys())[runs_key + 1]]
                                    )

                        else:
                            break

                peak_day = run_pooled.index[run_pooled == run_pooled.min()]

                if len(peak_day) > 1:
                    peak_day = peak_day[
                        int(np.ceil((len(peak_day) / 2.) - 1))
                        ]

                else:
                    peak_day = peak_day[0]

                runs_pooled[peak_day] = run_pooled
                runs_key += 1

            else:
                runs_key += 1

    return({num: run for num, run in runs_pooled.items()})


def remove_minors(runs, len_c, sum_c):
    """ Remove minor droughts.
    """
    len_min = runs_length(runs=runs).mean() * len_c
    sum_min = abs(runs_sum(runs=runs).mean()) * sum_c
    return({
        k: v for k, v in runs.items()
        if (len(v) >= len_min) and (abs(v.sum()) >= sum_min)
        })


def runs_onset(runs):
    """Extract the date of onset of each run.

    Parameters
    ----------
    runs : pandas.Series
        The runs time series (as obtained with get_runs())

    Output
    ------
    pandas.Series
    """
    return(pd.Series(
        data={
            num: run.index[0]
            for num, run in runs.items()
            },
        name='onset_date'
        ))


def runs_end(runs):
    """Extract the date of termination of each run.

    Parameters
    ----------
    runs : pandas.Series
        The runs time series (as obtained with get_runs())

    Output
    ------
    pandas.Series
    """
    return(pd.Series(
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
    return(pd.Series(
        data={k: v.sum() for k, v in runs.items()},
        name='sum'
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
    return(pd.Series(
        data={k: len(v) for k, v in runs.items()},
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
    return(pd.Series(
        data={k: v.min() for k, v in runs.items()},
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
        pd.concat(objs=[v for k, v in runs.items()])
        ).cumsum())
