#!/usr/bin/env python2
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

import numpy as np
import pandas as pd


def group_runs(deviations, pooling_method):
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
    if pooling_method in [None, 'ma']:
        sign = np.sign(deviations)
        sign[sign == 0] = 1
        runs = (sign != sign.shift(1)).astype(int).cumsum()
        runs[deviations.isnull()] = np.nan
        runs = runs - (runs.min() - 1)
        main_key = runs.keys().values[0]
        return(deviations.groupby(runs[main_key]))

    elif pooling_method == 'sp':
        # Sequent peak algorithm method.
        pass

    elif pooling_method == 'ic':
        # Inter-event time and volume criterion method.
        pass


class DroughtIndicator:
    """A time series containing some drought index which can be
    used in reference level analysis approach (runs theory)."""
    def __init__(self, indicator, pooling_method='ma', threshold='median'):
        """
        PARAMETERS
        """
        self.indicator = indicator
        self.pooling_method = pooling_method
        self.threshold = threshold

    def reference_level(self):
        # Compute the reference level.
        dates = pd.date_range(
                start='1981-01-01',
                end='1981-12-31',
                freq=self.indicator.index.freqstr)
        reflev = self.indicator.copy()

        for date in dates:
            indicator_parcial = self.indicator[
                    (self.indicator.index.month == date.month) &
                    (self.indicator.index.day == date.day)]

            if self.threshold == 'median':
                reflev[
                    (reflev.index.month == date.month) &
                    (reflev.index.day == date.day)] = float(
                        indicator_parcial.quantile(q=0.5))

            elif self.threshold == 'mean':
                reflev[
                    (reflev.index.month == date.month) &
                    (reflev.index.day == date.day)] = float(
                        indicator_parcial.mean())

        return(reflev)

    def deviation(self):
        """Returns the deviation of the indicator values from
        reference level.

        PARAMETERS
        """
        # Temporal aggregation of the drought indicator.
        return(self.indicator - self.reference_level())

    def get_runs(self, show_positives=True):
        """Returns the pandas.Series for each run identified in the
        input index.
            show_positives: boolean (optional)
                If True, also returns the distance between successive
                upcrosses and downcrosses (periods of excess of water).
                Default is True.
        """
        print("This action might take several seconds."
              " Please wait while processing.")
        grouper = group_runs(
                deviations=self.deviation(),
                pooling_method=self.pooling_method)
        runs = {name: grouper.get_group(name=name) for name in grouper.indices}

        if show_positives:
            return(runs)
        else:
            return({key: runs[key] for key in runs.keys() if
                    runs[key].values.sum() < 0})

    def get_runs_sum(self, show_positives=True):
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
        grouper = group_runs(
                deviations=self.deviation(),
                pooling_method=self.pooling_method)
        runs_sums = grouper.sum()
        main_key = runs_sums.keys().values[0]

        if show_positives:
            return(runs_sums)

        else:
            return(runs_sums.loc[runs_sums[main_key] < 0])

    def get_runs_length(self, show_positives=True):
        """Computes the distance between successive downcrosses and
        upcrosses. The run-length represents the duration of a drought.

        PARAMETERS

        REFERENCE
            Yevjevich, V. (1967). An objective approach to definitions
            and investigations of continental hydrologic droughts.
            Hydrology Papers 23, (23), 25–25. https://doi.org/10.1016
            /0022-1694(69)90110-3
        """
        grouper = group_runs(
                deviations=self.deviation(),
                pooling_method=self.pooling_method)
        runs_sums = grouper.sum()
        runs_lengths = grouper.count()
        main_key = runs_lengths.keys().values[0]

        if show_positives:
            return(runs_lengths)

        else:
            return(runs_lengths.loc[runs_sums[main_key] < 0])

    def get_sum_length_ratio(self):
        return(self.get_runs_sum() / self.get_runs_length())
