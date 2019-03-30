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
import lib.threshold_level_method as tlm

config = dmgr.Configurations('config.toml')

# Retrieve variables data.
# TODO: Make a unique function for any variable.
prec_raw = dmgr.grid_time_series(
    input_dir=config.gral['var1_dir'],
    basin_vmap=config.gral['basin_vmap'],
    resolution=config.grid['res'],
    nodata=config.grid['nodata'],
    missthresh=config.grid['missthresh'],
    variable='prec',
    flowstate='flow'
    )
sflo_raw = dmgr.sflo_time_series(
    input_dir=config.gral['var2_dir'],
    basin_vmap=config.gral['basin_vmap'],
    resolution=config.grid['res'],
    nodata=config.grid['nodata']
    )
tmax_raw = dmgr.grid_time_series(
    input_dir=config.gral['var3_dir'],
    basin_vmap=config.gral['basin_vmap'],
    resolution=config.grid['res'],
    nodata=config.grid['nodata'],
    missthresh=config.grid['missthresh'],
    variable='tmax',
    flowstate='state'
    )

# Trim datasets to set a common time period.
first = max([sflo_raw.index.min(), prec_raw.index.min(), tmax_raw.index.min()])
last = min([sflo_raw.index.max(), prec_raw.index.max(), tmax_raw.index.max()])
prec_raw = prec_raw[(prec_raw.index >= first) & (prec_raw.index <= last)]
sflo_raw = sflo_raw[(sflo_raw.index >= first) & (sflo_raw.index <= last)]
tmax_raw = tmax_raw[(tmax_raw.index >= first) & (tmax_raw.index <= last)]
sflo_raw[prec_raw.isnull().values] = np.nan
prec_raw[sflo_raw.isnull().values] = np.nan

# TODO: Put the following 'if' into the DroughtIndicator class.   #nextversion
if config.drought['pooling_method'] == 'ma':
    prec = dmgr.scale_data(
        input_data=prec_raw,
        scale=config.drought['agg_scale']
        )
    sflo = dmgr.scale_data(
        input_data=sflo_raw,
        scale=config.drought['agg_scale']
        )

prec = tlm.DroughtIndicator(
    indicator=prec,
    pooling_method=config.drought['pooling_method'],
    threshold=config.drought['threshold']
    )
sflo = tlm.DroughtIndicator(
    indicator=sflo,
    pooling_method=config.drought['pooling_method'],
    threshold=config.drought['threshold']
    )
