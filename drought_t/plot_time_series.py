# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 12:49:05 2019

@author: r.realrangel
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np


def plot_ts(
        series1, series2, start_date, end_date, output_file, series1_ref=None,
        series2_ref=None):
    # Precipitation deficit.
    # - Define variables.
    obs1 = series1[start_date: end_date]
    ref1 = series1_ref[start_date: end_date]
    dry = obs1.copy()
    dry[obs1 > ref1] = np.nan
    plt.figure()

    # - Define plot figure settings.
    fig, axes = plt.subplots(2, 1)

    axes[0].plot(
        obs1,
        color='k',
        linewidth=0.5
        )

    axes[0].plot(
        ref1,
        color='k',
        linestyle='--',
        linewidth=0.25
        )

    axes[0].fill_between(
        x=dry.index,
        y1=dry.values,
        y2=ref1.values,
        color='red'
        )

    axes[0].set_xlim([start_date, end_date])
    axes[0].set_ylim([0, 15])
    axes[0].set_ylabel('Precipitation (mm)')
    axes[0].xaxis.set_ticks([])

    # Stream flow deficit.
    # - Define variables.
    obs2 = series2[start_date: end_date]
    ref2 = series2_ref[start_date: end_date]
    dry = obs2.copy()
    dry[obs2 > ref2] = np.nan

    # Define plot figure settings.
    axes[1].plot(
        obs2,
        color='k',
        linewidth=0.5,
        label='Observed'
        )
    axes[1].plot(
        ref2,
        color='k',
        linestyle='--',
        linewidth=0.25,
        label='Ref. level'
        )
    axes[1].fill_between(
        x=dry.index,
        y1=dry.values,
        y2=ref2.values,
        color='red',
        label='Deficit'
        )

    axes[1].set_xlim([start_date, end_date])
    axes[1].set_ylim([0, 15])
    axes[1].set_ylabel('Streamflow (mm)')
    axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    axes[1].legend(
        frameon=False,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.175),
        ncol=3
        )

    fig.savefig(
        fname=output_file,
        dpi=300,
        bbox_inches='tight',
        pad_inches=0.05
        )
