# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 12:59:58 2019

@author: r.realrangel
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from drought_t import data_manager as dmgr
from scipy import stats
from sklearn.model_selection import KFold
# from sklearn.model_selection import train_test_split


def lowess(data, poly=2):
    def optimal_f(exog, endog, it=2, xval_folds=3):
        """
        """
#        exog_trn, exog_tst, endog_trn, endog_tst = train_test_split(
#            exog, endog, test_size=0.9, random_state=1
#            )
        kf = KFold(
            n_splits=xval_folds,
            shuffle=True,
            random_state=1
            )
        crossval = {}
        oneday = 1 / (len(exog) * ((xval_folds - 1) / xval_folds))
        f_tries = np.arange(
            start=oneday,
            stop=(oneday * 31),
            step=(oneday * 1)
            )

        for i, f in enumerate(f_tries):
            mae = []

            for trn_fold, tst_fold in kf.split(endog):
                # Split the sample in training and testing subsets.
                exog_tst = exog[tst_fold].copy()
                endog_tst = endog[tst_fold].copy()
                exog_trn = exog[trn_fold].copy()
                endog_trn = endog[trn_fold].copy()

                # Apply the model.
                data_model = sm.nonparametric.lowess(
                    endog=endog_trn,
                    exog=exog_trn,
                    frac=f,
                    it=2,
                    missing='raise'
                    )
                endog_trn_model = data_model[:, 1].copy()

                # Test the results.
                endog_tst_model = np.interp(
                    x=exog_tst,
                    xp=exog_trn,
                    fp=endog_trn_model,
                    left=np.nan,
                    right=np.nan
                    )
                endog_tst = endog_tst[np.isfinite(endog_tst_model)]
                endog_tst_model = endog_tst_model[np.isfinite(endog_tst_model)]
                mae.append(np.mean(np.abs(endog_tst - endog_tst_model)))

            dmgr.progress_message(
                current=(i + 1),
                total=len(f_tries),
                message="- Looking for the optimal f value",
                units=None
                )

            crossval[f] = np.mean(mae)

        return(min(crossval.keys(), key=(lambda k: crossval[k])))

    data_x = np.array(list(range(len(data))))
    data_y = data.values
    data_x = data_x[np.isfinite(data_y)]
    data_y = data_y[np.isfinite(data_y)]
    data_x = data_x ** poly
    f = optimal_f(
        exog=data_x,
        endog=data_y,
        it=2,
        xval_folds=3
        )
    data_model = sm.nonparametric.lowess(
        endog=data_y,
        exog=data_x,
        frac=f,
        it=2,
        missing='raise'
        )
    data_lowess = data.copy()
    data_lowess[data_lowess.notna()] = data_model[:, 1]
    return(data_lowess)


def maverage(x, window, min_periods_r):
    """
    Smoothes the raw records of a variable.

    Parameters
    ----------
    x : pandas.Series
        The time series of the input data.
    window : int, optional
        Size of the window centered on each day of the year used to
        choose the values from which the mean will be computed. By
        default, 29.

    Returns
    -------
    pandas.Series
        The time series of the variable of interest (x) with the same
        length as the input x.
    """
    return(x.rolling(
        window=window,
        center=True,
        min_periods=int(np.ceil(window * min_periods_r))
        ).mean())


def smoothvar(x, **parameters):
    if parameters['method'] == 'ma':
        return(maverage(
            x=x,
            window=parameters['window'],
            min_periods_r=parameters['min_periods_r']))
    elif parameters['method'] == 'lowess':
        return(lowess(
            data=x,
            poly=parameters['poly']))


def normedian(
        x, q=0.5, window=29, max_zero=0.0, min_notnull=0.667,
        min_wet_proportion=0.1, min_len=15, smooth=False, smoothpar=None
        ):
    """
    Compute 50th percentile of long-term daily records of a variable.

    References:
    Durre, I., Squires, M. F., Vose, R. S., Applequist, S., & Yin,
        X. (2011). Computational Procedures for the 1981-2010
        Normals: Precipitation, Snowfall, and Snow Depth.

    Parameters
    ----------
    x : pandas.Series
        The time series of the input data.
    q : float, optional
        The value of the percentile to usa as threshold level. By
        default, 0.5.
    window : int, optional
        Size of the window centered on each day of the year used to
        choose the values from which the percentile will be computed.
        By default, 29.
    max_zero : float, optional
        Maximum dry value. Every value below or equal is considered to
        be zero. By default, 0.0.
    min_notnull : int, optional
        Minumum fraction of values available within the applicable window
        in a given year to be included in the analysis. It cannot be
        greater than window. By default, 0.667.
    min_wet_proportion : float, optional
        Minimum ratio of wet values of the chosen values to perform
        the computation of the percentile.

    Returns
    -------
    pandas.Series
        The time series of the base value (x0) with the same length as
        the input x.
    """
    # Prepare the output pandas.Series.
    x0 = x.copy() * np.nan

    for date in pd.date_range(start='1981-01-01', end='1981-12-31'):
        data_window = pd.Series()

        # Extract the values in the corresponding date window.
        for year in list(set(x.index.year - 1)):
            window_first = date - pd.Timedelta(
                value=window / 2,
                unit='D'
                )
            window_first = pd.datetime(
                year=year,
                month=window_first.month,
                day=window_first.day
                )
            window_last = window_first + pd.Timedelta(
                value=window,
                unit='D'
                )
            data_window = data_window.append(
                to_append=x[
                    (x.index >= window_first) &
                    (x.index <= window_last)
                    ]
                )

        # Remove February 29 values.
        data_window[
            (data_window.index.month == 2) &
            (data_window.index.day == 29)
            ] = np.nan

        # Remove years with fewer than min_notnull values.
        data_subset = data_window.groupby(
            by=data_window.index.year
            ).filter(lambda group: group.count() > (min_notnull * window))

        # Remove zero-values.
        data_subset_wet = data_subset[data_subset > max_zero]

        # Compute the x0 only if, at least, min_wet_proportion of the chosen
        # values are nonzero.
        if len(data_subset_wet) / float(len(data_subset)) > min_wet_proportion:
            x0[
                (x0.index.month == date.month) &
                (x0.index.day == date.day)
                ] = data_subset_wet.quantile(q=q)

    # Set February 29 values as the mean of their corresponding
    # February 28 and March 1.
    x0[(x0.index.month == 2) & (x0.index.day == 29)] = np.mean([
        x0['1981-02-28'], x0['1981-03-01']
        ])

    x0.interpolate(
        method='linear',
        limit=3,
        inplace=True,
        limit_direction='both',
        limit_area='inside'
        )

    # Smooth the resulting x0.
    if smooth:
        x0 = smoothvar(
            x=x0,
            **smoothpar
            )
#    if smoothing_method == 'lowess':
#        poly = smoothpar['poly']
#        x0 = lowess(
#            data=x0,
#            poly=poly
#            )
#
#    elif smoothing_method == 'ma':
#        smoothing_window = smoothpar['smoothing_window']
#        smoothing_min_periods_r = smoothpar['smoothing_min_periods_r']
#        x0 = maverage(
#            x=x0,
#            window=smoothing_window,
#            min_periods_r=smoothing_min_periods_r
#            )

    # Remove stretches shorter than min_len.
    def remove_shorts(group):
        if 1 < len(group) < min_len:  # ??? <- Just "len(group) < min_len"?
            return(group * np.nan)

        else:
            return(group)

    x0 = x0.groupby(by=x0.isna().cumsum()).transform(remove_shorts)
    return(x0)


def lynehollick(x, a=0.925, reflection=30, passes=3):
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


def rank2v(x, y):
    """
    Reference:
        Yue, S., Ouarda, T. B. M. J., Bobée, B., Legendre, P. &
            Bruneau, P. (1999). The Gumbel mixed model for flood
            frequency analysis. Journal of Hydrology, 226(1–2), 88–100
            https://doi.org/10.1016/S0022-1694(99)00168-7.
    """
    xy = np.array([x.values, y.values]).T
    rank = np.array([
        len(np.where(np.all(xy[:] <= xy[i], axis=1))[0])
        for i in range(len(xy))
        ])
    return(pd.Series(
        data=rank,
        index=x.index,
        name='rank'
        ))



def plotpos(rank, a=0.44):
    """
    Reference:
        Gringorten, I. I. (1963). A plotting rule for extreme
            probability paper. Journal of Geophysical Research, 68(3),
            813–813. https://doi.org/10.1029/JZ068i003p00813
    """
    empP = (rank - a) / (len(rank) + (1 - (2 * a)))
    empP.name = 'P'
    return(empP)
