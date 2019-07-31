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

REFERENCES
    Tallaksen, L. M., & van Lanen, H. A. J. (Eds.). (2004).
        Hydrological Drought: Processes and Estimation Methods for
        Streamflow and Groundwater. Elsevier Inc. Retrieved from
        http://europeandroughtcentre.com/resources/hydrological-
        drought-1st-edition/
    van Loon, A. F. (2013). On the Propagation of Drought: How climate
        and catchment characteristics influence hydrological drought
        development and recovery.
    Yevjevich, V. (1967). An objective approach to definitions and
        investigations of continental hydrologic droughts. Hydrology
        Papers 23, (23), 25–25. https://doi.org/10.1016/0022-
        1694(69)90110-3"""

import sys as _sys

from PyEMD import EMD as _EMD
from scipy import signal as _signal
import numpy as _np
import pandas as _pd

import lib.data_manager as dmgr


def _sign_wo_zero(value):
    if _np.sign(value) == 0:
        return(1)

    else:
        return(_np.sign(value))


def reference_value(data, detrend='linear', ref_level=0.5):
    # Compute the reference level.
    dates = _pd.date_range(
        start='1981-01-01',
        end='1981-12-31',
        freq=data.index.freqstr
        )

    reflev = data.copy() * _np.nan

    for date in dates:
        data_subts = data[
            (data.index.month == date.month) &
            (data.index.day == date.day)
            ]

        if detrend == 'linear':
            data_subts_detrended = data_subts.copy()

            data_subts_detrended[data_subts_detrended.notnull()] = (
                _signal.detrend(
                    data=data_subts[data_subts.notnull()],
                    axis=0,
                    type='linear'
                    )
                )

            trend_diff = (data_subts - data_subts_detrended)

            if ref_level == 'mean':
                reference = float(
                    data_subts_detrended.mean()
                    )

                reflev_part = trend_diff + reference
                reflev_part[reflev_part < 0] = 0  # Impossible neg values.

            else:
                reference = float(
                    data_subts_detrended.quantile(q=ref_level)
                    )

                reflev_part = trend_diff + reference
                reflev_part[reflev_part < 0] = 0  # Impossible neg values.

        elif detrend == 'emd':
            data_subts_detrended = data_subts.copy()
            emd = _EMD()

            data_subts_detrended[data_subts_detrended.notnull()] = emd(
                data_subts[data_subts.notnull()].values
                )

        elif detrend is False:
            if ref_level == 'mean':
                reflev_part = float(data_subts.mean())

            else:
                reflev_part = float(data_subts.quantile(q=ref_level))

        reflev[
            (reflev.index.month == date.month) &
            (reflev.index.day == date.day)
            ] = reflev_part

    return(reflev)


def _sign_grouper(anomalies):
    """
    PARAMETERS
        pooling_method: string or None
            Defines the pool method to be used for grouping values
            into runs.

    REFERENCE
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
    """
    sign = _np.sign(anomalies)
    sign[sign == 0] = 1
    runs = (sign != sign.shift(1)).astype(int).cumsum()
    runs[anomalies.isnull()] = _np.nan
    runs = runs - (runs.min() - 1)
    return(anomalies.groupby(runs))


def anomaly(data, detrend='linear', ref_level=0.5):
    """Returns the deviation of the indicator values from
    reference level.

    PARAMETERS
    """
    # Temporal aggregation of the drought indicator.
    return(data - reference_value(
        data=data,
        detrend=detrend,
        ref_level=ref_level
        ))


def runs_sum(runs):
    """Computes the sum of all deviations between successive
    downcrosses and upcrosses. Indicates the deficiency of water
    in the variable analyzed or the severity of drought.

    PARAMETERS

    REFERENCE
        Yevjevich, V. (1967). An objective approach to definitions
        and investigations of continental hydrologic droughts.
        Hydrology Papers 23, (23), 25–25. https://doi.org/10.1016
        /0022-1694(69)90110-3
    """
    return(_pd.Series(
        data={index: runs[index].sum() for index in runs.index},
        name='length'
        ))


def get_runs(
        data, pooling_method=None, detrend='linear', ref_level=0.5,
        show_positives=False, window=None
        ):

    anomalies = anomaly(
        data=data,
        detrend=detrend,
        ref_level=ref_level
        )

    runs = {
        name: _sign_grouper(anomalies=anomalies).get_group(name=name)
        for name in _sign_grouper(anomalies=anomalies).indices
        }

    if pooling_method is None:
        # Not pooling runs.
        runs_pooled = runs

    elif pooling_method == 'ma':
        # Pooling runs through the moving average method.
        if window is None:
            _sys.exit("Specify a value for the moving averaging window.")

        else:
            data_ma = data.rolling(
                window=window,
                center=True
                ).mean()

            anomalies_ma = anomaly(
                data=data_ma,
                detrend=detrend,
                ref_level=ref_level
                )

            grouper_ma = _sign_grouper(anomalies=anomalies_ma)

            runs_ma = {
                name: grouper_ma.get_group(name=name)
                for name in grouper_ma.indices
                }

            runs_pooled = {}
            counter = 1
            counter2 = 0

            for num_ma, run_ma in runs_ma.iteritems():
                candidates_to_pool = []

                for num, run in runs.iteritems():
                    if run.index[0] <= run_ma.index[-1]:
                        for date in run_ma.index:
                            if date in run.index:
                                candidates_to_pool.append(run)
                                break

                        else:
                            pass

                runs_to_pool = [
                    run
                    for run in candidates_to_pool
                    if _sign_wo_zero(run.sum()) == _sign_wo_zero(run_ma.sum())
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
                    total=len(runs_ma),
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
        runs_output = _pd.Series(
            data=runs_pooled,
            name='anomaly'
            )

        onsets_output = _pd.Series(
            data={
                num: run.index[0]
                for num, run in runs_pooled.iteritems()
                },
            name='onset_date'
            )

        ends_output = _pd.Series(
            data={
                num: run.index[-1]
                for num, run in runs_pooled.iteritems()
                },
            name='end_date'
            )

    else:
        runs_output = _pd.Series(
            data={
                num: run
                for num, run in runs_pooled.iteritems()
                if run.sum() < 0
                },
            name='anomaly'
            )

        onsets_output = _pd.Series(
            data={
                num: run.index[0]
                for num, run in runs_pooled.iteritems()
                if run.sum() < 0
                },
            name='onset_date'
            )

        ends_output = _pd.Series(
            data={
                num: run.index[-1]
                for num, run in runs_pooled.iteritems()
                if run.sum() < 0
                },
            name='end_date'
            )

    return(runs_output, onsets_output, ends_output)


def runs_length(runs):
    """Computes the distance between successive downcrosses and
    upcrosses. The run-length represents the duration of a drought.

    Parameters:
        runs : pandas.Series
            A series of runs where each value is the time series of the
            run.

    References:
        Yevjevich, V. (1967). An objective approach to definitions
        and investigations of continental hydrologic droughts.
        Hydrology Papers 23, (23), 25–25. https://doi.org/10.1016
        /0022-1694(69)90110-3
    """
    return(_pd.Series(
        data={index: runs[index].count() for index in runs.index},
        name='length'
        ))


def sum_length_ratio(show_positives=True, detrend='linear', ref_level=0.5):
    return(
        runs_sum(
            show_positives=show_positives,
            detrend=detrend,
            ref_level=ref_level
            ) /
        runs_length(
            show_positives=show_positives,
            detrend=detrend,
            ref_level=ref_level
            )
        )


def run_peak(runs):
    """Returns the maximum absolute value in the run.

    Parameters:
        runs : pandas.Series
            A series of runs where each value is the time series of the
            run.
    """
    return(_pd.Series(
        data={index: runs[index].min() for index in runs.index},
        name='peak'
        ))


def runs_cumsum(
        anomalies, pooling_method, show_positives=True, detrend='linear',
        ref_level=0.5
        ):
    return(
        _sign_grouper(
            anomalies=anomalies,
            pooling_method=pooling_method
            ).cumsum()
        )
