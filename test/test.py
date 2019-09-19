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

import drought_t.data_analyst as dnlst
import drought_t.data_manager as dmgr
import drought_t.plot_time_series as plots
import drought_t.threshold_level_method as tlm

config = dmgr.Configurations('test.toml')

if config.vars['precip']['input_dir'] is not False:
    # Retrieve the precipitation (precip) data.
    precip_raw = dmgr.open_precip(
        source=config.vars['precip']['source'],
        input_data_dir=config.vars['precip']['input_dir'],
        output_var_name=config.vars['precip']['output_var_name'],
        resolution=config.vars['precip']['input_data_res'],
        time_zone=config.vars['precip']['time_zone']
        )

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
        input_mask_path=config.vars['basin_vmap_epsg6372'],
        time_zone=config.vars['sflow']['time_zone']
        )

    # Aggregate the data.
    sflow = dnlst.smooth_data(
        input_data=sflow_raw,
        agg_type=config.vars['sflow']['smoothing_type'],
        agg_scale=config.vars['sflow']['smoothing_scale']
        )

# Precipitation deficit
metdr = tlm.get_runs(
    data=precip,
    pooling_method=config.drought['pooling_method'],
    detrend=config.vars['detrend'],
    thresh=config.drought['ref_level'],
    show_positives=config.drought['show_positives'],
    window=config.drought['ma_window']
    )

metdr_magnitude = tlm.runs_sum(runs=metdr)

metdr_large = metdr[
    metdr_magnitude <= metdr_magnitude.quantile(
        q=config.drought['large_runs']
        )
    ]

metdr_large_peak = tlm.run_peak(
    runs=metdr_large
    )

metdr_large_onset = tlm.runs_onset(metdr_large)

metdr_large_end = tlm.runs_end(metdr_large)

metdr_large_magnitude = metdr_magnitude[
    metdr_magnitude <= metdr_magnitude.quantile(
        q=config.drought['large_runs']
        )
    ]

# Streamflow deficit
hiddr = tlm.get_runs(
    data=sflow,
    pooling_method=config.drought['pooling_method'],
    detrend=config.vars['detrend'],
    thresh=config.drought['ref_level'],
    show_positives=config.drought['show_positives'],
    window=config.drought['ma_window']
    )

hiddr_magnitude = tlm.runs_sum(runs=hiddr)

hiddr_large = hiddr[
    hiddr_magnitude <= hiddr_magnitude.quantile(
        q=config.drought['large_runs']
        )
    ]

hiddr_large_peak = tlm.run_peak(
    runs=hiddr_large
    )

hiddr_large_onset = tlm.runs_onset(hiddr_large)

hiddr_large_end = tlm.runs_end(hiddr_large)

hiddr_large_magnitude = hiddr_magnitude[
    hiddr_magnitude <= hiddr_magnitude.quantile(
        q=config.drought['large_runs']
        )
    ]

if config.plot['plot_time_series']:
    if not os.path.isdir(config.plot['output_dir']):
        os.mkdir(config.plot['output_dir'])

    series1_ref = precip - tlm.anomaly(
        data=precip,
        detrend='linear',
        thresh=0.5
        )
    series2_ref = sflow - tlm.anomaly(
        data=sflow,
        detrend='linear',
        thresh=0.5
        )

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
