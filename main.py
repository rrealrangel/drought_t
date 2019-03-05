#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""DROUGHT ASSESSMENT
DESCRIPTION
    Set of scripts to perform a drought hazard quantitative assessment.

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
        1694(69)90110-3
"""

import lib.drought_indices as indices
import numpy as np
import pandas as pd
import toml

with open('config.toml', 'rb') as fin:
    config = toml.load(fin)


def read_input(input_file):
    """Retrieve a time series from a .csv file.

    PARAMETERS
        input_file: string
            The ful path of the input .csv file."""
    return(pd.read_csv(
            filepath_or_buffer=input_file,
            index_col='time',
            parse_dates=True))


prec = read_input(config['inputs']['prec_file'])
disc = read_input(config['inputs']['disc_file'])
disc[prec.isnull().values] = np.nan
prec[disc.isnull().values] = np.nan
prec_sdi = indices.npsdi(data=prec.rolling(window=1).sum())
disc_sdi = indices.npsdi(data=disc.rolling(window=1).sum())
prec_anm = indices.anomaly(data=prec.rolling(window=1).sum(), method='median')
disc_anm = indices.anomaly(data=disc.rolling(window=1).sum(), method='median')
