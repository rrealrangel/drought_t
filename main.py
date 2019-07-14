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
import os

import lib.data_manager as dmgr
import lib.plot_time_series as plt
import lib.threshold_level_method as tlm

config = dmgr.Configurations(
    '/home/realrangel/MEGA/projects/multiannual/04-doctorado/analysis'
    '/02_drought_analysis/15014/config.toml'
    )

# TODO: Make a unique function for any variable.
# Retrieve variables precipitation data.
if config.vars['prec']['input_dir'] is not False:
    prec_raw = dmgr.clim_time_series(
        input_dir=config.vars['prec']['input_dir'],
        basin_vmap=config.vars['basin_vmap'],
        resolution=config.grid['res'],
        nodata=config.grid['nodata'],
        missthresh=config.grid['missthresh'],
        variable='observed',
        flowstate='flow'
        )

else:
    prec_raw = None

# Retrieve stream flow data.
if config.vars['sflo']['input_file'] is not False:
    sflo_raw = dmgr.sflo_time_series(
        input_file=config.vars['sflo']['input_file'],
        basin_vmap=config.vars['basin_vmap'],
        resolution=config.grid['res'],
        nodata=config.grid['nodata']
        )

else:
    sflo_raw = None

# Retrieve soil moisture data.
if config.vars['smoi']['input_dir'] is not False:
    smoi_raw = dmgr.smoi_time_series(
        input_dir=config.vars['smoi']['input_dir'],
        basin_vmap=config.vars['basin_vmap'],
        resolution=config.grid['res'],
        nodata=config.grid['nodata']
        )

else:
    smoi_raw = None

# Retrieve temperature data.
if config.vars['tmea']['input_dir'] is not False:
    tmea_raw = dmgr.clim_time_series(
        input_dir=config.vars['tmea']['input_dir'],
        basin_vmap=config.vars['basin_vmap'],
        resolution=config.grid['res'],
        nodata=config.grid['nodata'],
        missthresh=config.grid['missthresh'],
        variable='observed',
        flowstate='state'
        )

else:
    tmea_raw = None

# Aggregate data.
if prec_raw is not None:
    prec = dmgr.scale_data(
        input_data=prec_raw,
        scale=config.vars['prec']['agg_scale']
        )

if sflo_raw is not None:
    sflo = dmgr.scale_data(
        input_data=sflo_raw,
        scale=config.vars['sflo']['agg_scale']
        )

if smoi_raw is not None:
    smoi = dmgr.scale_data(
        input_data=smoi_raw,
        scale=config.vars['smoi']['agg_scale']
        )

if tmea_raw is not None:
    tmea = dmgr.scale_data(
        input_data=tmea_raw,
        scale=config.vars['tmea']['agg_scale']
        )

# Precipitation deficit
precdef_run, precdef_onset, precdef_end = tlm.get_runs(
    data=prec,
    pooling_method=config.drought['pooling_method'],
    detrend=config.vars['detrend'],
    ref_level=config.drought['ref_level'],
    show_positives=config.drought['show_positives'],
    window=config.drought['ma_window']
    )

precdef_magnitude = tlm.runs_sum(runs=precdef_run)


precdef_run_large = precdef_run[
    precdef_magnitude <= precdef_magnitude.quantile(q=0.1)
    ]

precdef_peak = tlm.run_peak(
    runs=precdef_run
    )

precdef_peak_large = tlm.run_peak(
    runs=precdef_run_large
    )

precdef_onset_large = precdef_onset[
    precdef_magnitude <= precdef_magnitude.quantile(q=0.1)
    ]

precdef_end_large = precdef_end[
    precdef_magnitude <= precdef_magnitude.quantile(q=0.1)
    ]

precdef_magnitude_large = precdef_magnitude[
    precdef_magnitude <= precdef_magnitude.quantile(q=0.1)
    ]

# Stream flow deficit
sflodef_run, sflodef_onset, sflodef_end = tlm.get_runs(
    data=sflo,
    pooling_method=config.drought['pooling_method'],
    detrend=config.vars['detrend'],
    ref_level=config.drought['ref_level'],
    show_positives=config.drought['show_positives'],
    window=config.drought['ma_window']
    )

sflodef_magnitude = tlm.runs_sum(runs=sflodef_run)

sflodef_run_large = sflodef_run[
    sflodef_magnitude <= sflodef_magnitude.quantile(q=0.1)
    ]

sflodef_peak = tlm.run_peak(
    runs=sflodef_run
    )

sflodef_peak_large = tlm.run_peak(
    runs=sflodef_run_large
    )

sflodef_onset_large = sflodef_onset[
    sflodef_magnitude <= sflodef_magnitude.quantile(q=0.1)
    ]

sflodef_end_large = sflodef_end[
    sflodef_magnitude <= sflodef_magnitude.quantile(q=0.1)
    ]

sflodef_magnitude_large = sflodef_magnitude[
    sflodef_magnitude <= sflodef_magnitude.quantile(q=0.1)
    ]

if config.plot['plot_time_series']:
    if not os.path.isdir(config.plot['output_dir']):
        os.mkdir(config.plot['output_dir'])

    series1_ref = tlm.reference_value(data=prec)
    series2_ref = tlm.reference_value(data=sflo)

    first_year = max([
        i.index.min()
        for i in [prec_raw, sflo_raw, tmea_raw]
        if i is not None
        ]).year

    last_year = min([
        i.index.max()
        for i in [prec_raw, sflo_raw, tmea_raw]
        if i is not None
        ]).year + 1

    for date in range(first_year, last_year):
        output_file = (
            config.plot['output_dir'] + '/drought_' + str(date) + '-' +
            str(date + 1)[-2:] + '.png'
            )

        plt.plot_ts(
            series1=prec,
            series2=sflo,
            start_date=(str(date) + '-' + config.plot['plot_start']),
            end_date=(str(date + 1) + '-' + config.plot['plot_end']),
            output_file=output_file,
            series1_ref=series1_ref,
            series2_ref=series2_ref
            )
