# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 14:16:17 2019

@author: r.realrangel
"""
import numpy as np
from drought_t import threshold_level_method as tlm


def pooling_par(x, x0, pooling_method, tc_list, pc_list):
    # Moving average (MA) method.
    if pooling_method == 'ma':
        runnum = []
        runlen = []
        runsum = []
        window_list = range(1, 30)

        for window in window_list:
            droughts = tlm.pool_runs(
                x=x,
                x0=x0,
                pooling_method=pooling_method,
                window=window
                )
            runnum.append(len(droughts))
            runlen.append(tlm.runs_length(
                runs=droughts
                ).mean())
            runsum.append(tlm.runs_sum(
                runs=droughts
                ).mean())

    # Inter-event time and volume criterion (IC) method
    if pooling_method == 'ic':
        runnum = np.ndarray(shape=(len(tc_list), len(pc_list))) * np.nan
        runlen = np.ndarray(shape=(len(tc_list), len(pc_list))) * np.nan
        runsum = np.ndarray(shape=(len(tc_list), len(pc_list))) * np.nan

        for i, tc in enumerate(tc_list):
            for j, pc in enumerate(pc_list):
                if ((i == 0) and (j != 0)) or ((i != 0) and (j == 0)):
                    continue

                else:
                    droughts = tlm.pool_runs(
                        x=x,
                        x0=x0,
                        pooling_method=pooling_method,
                        tc=tc,
                        pc=pc
                        )

                    # Remove minor droughts.
                    len_min = tlm.runs_length(runs=droughts).mean() * 0.1
                    sum_min = tlm.runs_sum(runs=droughts).mean() * 0.1
                    droughts = {
                        k: v for k, v in droughts.items()
                        if (len(v) >= len_min) and (v.sum() <= sum_min)
                        }

                    # Define the reference values (not pooled).
                    if (i == 0) and (j == 0):
                        ref_ev_count = len(droughts)
                        ref_duration = tlm.runs_length(
                            runs=droughts
                            ).mean()
                        ref_defaccum = tlm.runs_sum(
                            runs=droughts
                            ).mean() * -1

                # Results of the loop.
                runnum[i][j] = len(droughts) / float(ref_ev_count)
                runlen[i][j] = (
                    tlm.runs_length(runs=droughts).mean()
                    ) / ref_duration
                runsum[i][j] = (
                    tlm.runs_sum(runs=droughts).mean() * -1
                    ) / ref_defaccum
                print(
                    "tc= {:.2f}, "
                    "pc= {:.2f}, "
                    "num= {:.2f}, "
                    "len={:.2f}, "
                    "sum= {:.2f}".format(
                        tc, pc,
                        runnum[i][j],
                        runlen[i][j],
                        runsum[i][j]
                        )
                    )

    return(runnum, runlen, runsum)
