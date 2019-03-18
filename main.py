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

import numpy as np

import lib.data_manager as dmgr
from lib.threshold_level_method import DroughtIndicator

config = dmgr.Configurations('config.toml')
prec = dmgr.read_input(config.gral_prec_file)
disc = dmgr.read_input(config.gral_disc_file)
disc[prec.isnull().values] = np.nan
prec[disc.isnull().values] = np.nan

if config.pooling_method == 'ma':
    prec = dmgr.scale_data(input_data=prec, scale=config.agg_scale)
    disc = dmgr.scale_data(input_data=disc, scale=config.agg_scale)

prec = DroughtIndicator(
        indicator=prec,
        pooling_method=config.pooling_method,
        threshold=config.threshold)
disc = DroughtIndicator(
        indicator=disc,
        pooling_method=config.pooling_method,
        threshold=config.threshold)
