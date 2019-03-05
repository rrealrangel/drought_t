#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""THRESHOLD LEVEL METHOD FOR DEFINING DROUGHT EVENTS

DESCRIPTION
    This is the most frequently applied quantitative method where it is
    essential to defining a threshold, Q_0, below which the river flow
    is considered as a drought (also referred to as a low flow spell).

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
    van Loon, A. F. (2013). On the propagation of drought: how climate
        and catchment characteristics influence hydrological drought
        development and recovery.
    Yevjevich, V. (1967). An objective approach to definitions and
        investigations of continental hydrologic droughts. Hydrology
        Papers 23, (23), 25–25. https://doi.org/10.1016/0022-
        1694(69)90110-3"""

import numpy as np


class DroughtIndicator:
    """A time series containing some drought indicator which can be
    used in reference level analysis approach (runs theory)."""
    def __init__(self, data):
        self.data = data
        self.values = data.values

    def group_runs(self, pool_method, ma_window=10):
        """
        PARAMETERS
            pool_method: string or None
                Defines the pool method to be used for grouping values
                into runs.
            ma_window: integer (default is 10)
                Is applied only if pool_method = 'ma'. Defines the size
                of the moving average window.

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
        if pool_method is None:
            sign = np.sign(self.data)
            sign[sign == 0] = 1
            runs = (sign != sign.shift(1)).astype(int).cumsum()
            runs[self.data.isnull()] = np.nan
            runs = runs - (runs.min() - 1)
            main_key = runs.keys().values[0]
            return(self.data.groupby(runs[main_key]))

        elif pool_method == 'ma':
            # Moving average method.
            # TODO: Make 'window' a **kwarg (or *arg?).
            pooled_data = self.data.rolling(window=10).mean()
            sign = np.sign(pooled_data)
            pooled_runs = (sign != sign.shift(1)).astype(int).cumsum()
            pooled_runs[pooled_data.isnull()] = np.nan
            pooled_runs = pooled_runs - (pooled_runs.min() - 1)
            main_key = pooled_runs.keys().values[0]
            return(pooled_data.groupby(pooled_runs[main_key]))

        elif pool_method == 'sp':
            # Sequent peak algorithm method.
            pass

        elif pool_method == 'ic':
            # Inter-event time and volume criterion method.
            pass

    def get_runs(self, pool_method='ma', positive=True):
        """Returns the pandas.Series for each run identified in the input data.
        """
        print("This action might take several seconds."
              " Please wait while processing.")
        runs = {name: self.group_runs(pool_method).get_group(name=name) for
                name in self.group_runs(pool_method).indices}

        if positive:
            return(runs)
        else:
            return({key: runs[key] for key in runs.keys() if
                    runs[key].values.sum() < 0})

    def get_runs_sum(self, pool_method='ma', positive=True):
        """Computes the sum of all deviations between successive
        downcrosses and upcrosses. Indicates the deficiency of water
        in the variable analyzed or the severity of drought.

        PARAMETERS
            positive: boolean (default is False)
                If True, also returns the sum of positive deviations.

        REFERENCE
            Yevjevich, V. (1967). An objective approach to definitions
            and investigations of continental hydrologic droughts.
            Hydrology Papers 23, (23), 25–25. https://doi.org/10.1016
            /0022-1694(69)90110-3
        """
        runs_sums = self.group_runs(pool_method).sum()
        main_key = runs_sums.keys().values[0]

        if positive:
            return(runs_sums)

        else:
            return(runs_sums.loc[runs_sums[main_key] < 0])

    def get_runs_length(self, pool_method='ma', positive=True):
        """Computes the distance between successive downcrosses and
        upcrosses. The run-length represents the duration of a drought.

        PARAMETERS
            positive: boolean (default is False)
                If True, also returns the distance between successive
                upcrosses and downcrosses (periods of excess of water).

        REFERENCE
            Yevjevich, V. (1967). An objective approach to definitions
            and investigations of continental hydrologic droughts.
            Hydrology Papers 23, (23), 25–25. https://doi.org/10.1016
            /0022-1694(69)90110-3
        """
        runs_sums = self.group_runs(pool_method).sum()
        runs_lengths = self.group_runs(pool_method).count()
        main_key = runs_lengths.keys().values[0]

        if positive:
            return(runs_lengths)

        else:
            return(runs_lengths.loc[runs_sums[main_key] < 0])

    def get_sum_length_ratio(self, pool_method='ma', positive=True):
        return(self.get_runs_sum(pool_method, positive) /
               self.get_runs_length(pool_method, positive))
