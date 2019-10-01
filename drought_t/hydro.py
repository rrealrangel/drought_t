# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 12:59:58 2019

@author: r.realrangel
"""
import pandas as pd


def baseflow_lh(x, a=0.925, reflection=30, passes=3):
    """
    References:
        Ladson, A., Brown, R., Neal, B. & Nathan, R. (2013). A standard
        approach to baseflow separation using the Lyne and Hollick
        filter. Australian Journal of Water Resources, 17(1). https://
        doi.org/10.7158/W12-028.2013.17.1.

    """
    if len(x) < 65:
        return(x * pd.np.nan)

    else:
        x_notna = x[x.notna()]
        x_warm = pd.Series(   # <- Warm up period.
            data=list(reversed(x_notna.iloc[1: reflection + 1].values)),
            index=pd.date_range(
                start=x_notna.index[0] - pd.Timedelta(reflection, 'D'),
                periods=reflection
                )
            )
        x_cool = pd.Series(   # <- Cool down period.
            data=list(reversed(x_notna.iloc[-reflection - 1: -1].values)),
            index=pd.date_range(
                start=x_notna.index[-1] + pd.Timedelta(1, 'D'),
                periods=reflection
                )
            )
        q = pd.concat(objs=[x_warm, x_notna, x_cool])
        qf = q.copy() * pd.np.nan
        qb = q.copy() * pd.np.nan

        for _pass in range(passes):
            if _pass == 0:
                qin = q.copy()

            else:
                qin = qb.copy()

            if (_pass + 1) % 2 == 1:
                it = range(len(q))

            else:
                it = range(len(q) - 1, -1, -1)

            for i in range(len(it)):
                if i == 0:
                    qf[it[i]] = qin[it[i]]

                else:
                    qf[it[i]] = (   # <- Filter
                        (a * qf[it[i - 1]]) +
                        (((1 + a) / 2) * (qin[it[i]] - qin[it[i - 1]]))
                        )

                if qf[it[i]] <= 0:
                    qf[it[i]] = 0

                qb[it[i]] = qin[it[i]] - qf[it[i]]

        return(qb.reindex(index=x.index))
