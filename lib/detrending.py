# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 13:25:44 2019

@author: r.realrangel
"""
from scipy import signal


def linear_draft(data, ref=0.5):
    detrended = data.copy()

    detrended[detrended.notnull()] = (
        signal.detrend(
            data=data[data.notnull()],
            axis=0,
            type='linear'
            )
        )

    trend_diff = (data - detrended)

    if ref == 'mean':
        reference = float(detrended.mean())
        reference_level = trend_diff + reference
        reference_level[reference_level < 0] = 0  # Impossible neg values.

    else:
        reference = float(detrended.quantile(q=ref))
        reference_level = trend_diff + reference
        reference_level[reference_level < 0] = 0  # Impossible neg values.

    return()