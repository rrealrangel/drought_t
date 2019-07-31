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
import os

import lib.data_analyst as dnlst
import lib.data_manager as dmgr
import lib.plot_time_series as plots
import lib.threshold_level_method as tlm

config = dmgr.Configurations(
    'C:/Users/rreal/Mega/projects/multiannual/04-doctorado/analysis'
    '/02_drought_analysis/15014/drought_t.toml'
    )

if config.vars['precip']['input_dir'] is not False:
    # Retrieve the precipitation (precip) data.
#    precip_raw = dmgr.open_clim_ts(
#        input_dir=config.vars['precip']['input_dir'],
#        basin_vmap_epsg4326=config.vars['basin_vmap_epsg4326'],
#        basin_vmap_epsg6372=config.vars['basin_vmap_epsg6372'],
#        resolution=config.grid['res'],
#        nodata=config.grid['nodata'],
#        missthresh=config.grid['missthresh'],
#        output_varname='observed',
#        flowstate='flow'
#        )

    precip_raw = dmgr.open_chirps_ts(
        input_dir=config.vars['precip']['input_dir'],
        output_varname='observed',
        res=0.05
        )

#    # Detrend the data
#    precip_detrended = dnlst.detrend_data(
#        data=precip_raw)

    # Aggregate the data.
    precip = dnlst.smooth_data(
        input_data=precip_raw,
        agg_type=config.vars['precip']['smoothing_type'],
        agg_scale=config.vars['precip']['smoothing_scale']
        )


if config.vars['sflow']['input_file'] is not False:
    # Retrieve the streamflow data.
    sflow_raw = dmgr.open_sflow_ts(
        input_file=config.vars['sflow']['input_file'],
        basin_vmap=config.vars['basin_vmap_epsg6372'],
#        resolution=config.grid['res'],
        nodata=config.grid['nodata']
        )

#    # Detrend the data
#    precip_detrended = dnlst.detrend_data(
#        data=sflow_raw)

    # Aggregate the data.
    sflow = dnlst.scale_data(
        input_data=sflow_raw,
        scale=config.vars['sflow']['agg_scale']
        )


# Precipitation deficit
precip_anomaly_run, precip_anomaly_onset, precip_anomaly_end = tlm.get_runs(
    data=precip,
    pooling_method=config.drought['pooling_method'],
    detrend=config.vars['detrend'],
    ref_level=config.drought['ref_level'],
    show_positives=config.drought['show_positives'],
    window=config.drought['ma_window']
    )

precip_anomaly_magnitude = tlm.runs_sum(runs=precip_anomaly_run)


precip_anomaly_run_l = precip_anomaly_run[
    precip_anomaly_magnitude <= precip_anomaly_magnitude.quantile(q=0.1)
    ]

precip_anomaly_peak = tlm.run_peak(
    runs=precip_anomaly_run
    )

precip_anomaly_peak_l = tlm.run_peak(
    runs=precip_anomaly_run_l
    )

precip_anomaly_onset_l = precip_anomaly_onset[
    precip_anomaly_magnitude <= precip_anomaly_magnitude.quantile(q=0.1)
    ]

precip_anomaly_end_l = precip_anomaly_end[
    precip_anomaly_magnitude <= precip_anomaly_magnitude.quantile(q=0.1)
    ]

precip_anomaly_magnitude_l = precip_anomaly_magnitude[
    precip_anomaly_magnitude <= precip_anomaly_magnitude.quantile(q=0.1)
    ]

# Stream flow deficit
sflow_anomaly_run, sflow_anomaly_onset, sflow_anomaly_end = tlm.get_runs(
    data=sflow,
    pooling_method=config.drought['pooling_method'],
    detrend=config.vars['detrend'],
    ref_level=config.drought['ref_level'],
    show_positives=config.drought['show_positives'],
    window=config.drought['ma_window']
    )

sflow_anomaly_magnitude = tlm.runs_sum(runs=sflow_anomaly_run)

sflow_anomaly_run_l = sflow_anomaly_run[
    sflow_anomaly_magnitude <= sflow_anomaly_magnitude.quantile(q=0.1)
    ]

sflow_anomaly_peak = tlm.run_peak(
    runs=sflow_anomaly_run
    )

sflow_anomaly_peak_l = tlm.run_peak(
    runs=sflow_anomaly_run_l
    )

sflow_anomaly_onset_l = sflow_anomaly_onset[
    sflow_anomaly_magnitude <= sflow_anomaly_magnitude.quantile(q=0.1)
    ]

sflow_anomaly_end_l = sflow_anomaly_end[
    sflow_anomaly_magnitude <= sflow_anomaly_magnitude.quantile(q=0.1)
    ]

sflow_anomaly_magnitude_l = sflow_anomaly_magnitude[
    sflow_anomaly_magnitude <= sflow_anomaly_magnitude.quantile(q=0.1)
    ]

if config.plot['plot_time_series']:
    if not os.path.isdir(config.plot['output_dir']):
        os.mkdir(config.plot['output_dir'])

    series1_ref = tlm.reference_value(data=precip)
    series2_ref = tlm.reference_value(data=sflow)

    first_year = max([
        i.index.min()
        for i in [precip, sflow]
        if i is not None
        ]).year

    last_year = min([
        i.index.max()
        for i in [precip, sflow]
        if i is not None
        ]).year + 1

    for year in range(first_year, last_year):
        output_file = (
            config.plot['output_dir'] + '/drought_' + str(year) + '-' +
            str(year + 1)[-2:] + '.png'
            )

        plots.plot_ts(
            series1=precip,
            series2=sflow,
            start_date=(str(year) + '-' + config.plot['plot_start']),
            end_date=(str(year + 1) + '-' + config.plot['plot_end']),
            output_file=output_file,
            series1_ref=series1_ref,
            series2_ref=series2_ref
            )
