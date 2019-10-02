# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 09:29:58 2019

@author: r.realrangel
"""
from sklearn.model_selection import KFold as _KFold
import numpy as _np
from sklearn.preprocessing import PolynomialFeatures as _PolyFeat
from scipy import linalg as _linalg

# =============================================================================
# https://gist.github.com/agramfort/850437
# https://olamyy.github.io/olamyy.github.io/blog/2018/01/30/locally-weighted-regression/
# Lowess smoother: Robust locally weighted regression.
# The lowess function fits a nonparametric regression curve to a scatterplot.
# The arrays x and y contain an equal number of elements; each pair
# (x[i], y[i]) defines a data point in the scatterplot. The function returns
# the estimated (smooth) values of y.
# The smoothing span is given by f. A larger value for f will result in a
# smoother curve. The number of robustifying iterations is given by it. The
# function will run faster with a smaller number of iterations.
# =============================================================================


def optimal_f(x, y, ftries=41, it=2, xval_folds=3):
    """
    """
    kf = _KFold(
        n_splits=xval_folds,
        shuffle=True,
        random_state=1
        )
    f_tries = _np.linspace(
        start=0,
        stop=1,
        num=ftries
        )
    f_tries = f_tries[f_tries != 0]
    crossval = {}

    for f in f_tries:
        mae = []

        for _, tst_fold in kf.split(y):
            # Testing subsets.
            x_tst = x[tst_fold]
            y_tst = y[tst_fold]

            # Training subsets.
            trn_fold = [j for j in range(len(x)) if j not in tst_fold]
            x_trn = x[trn_fold]
            y_trn = y[trn_fold]
            n = len(x_trn)
            y_trn_model = _np.zeros(n)
            r = int(_np.ceil(f * n))
            h = []

            for i in range(n):
                dist = _np.unique(_np.sort(_np.abs(x_trn - x_trn[i])))

                if len(dist) <= r:
                    h.append(dist[-1])

                else:
                    h.append(dist[r])

            w = _np.clip(
                a=_np.abs((x_trn[:, None] - x_trn[None, :]) / h),
                a_min=0.0,
                a_max=1.0
                )
            w = (1 - w ** 3) ** 3
            delta = _np.ones(n)
            x_trn = _np.expand_dims(x_trn, 1)
            poly = _PolyFeat(degree=2)
            x_trn_poly = poly.fit_transform(x_trn)

            for iteration in range(it):
                for i in range(n):
                    weights = delta * w[:, i]

                    if _np.sum(weights) == 0:
                        pass

                    else:
                        coeff = _np.array([
                            [_np.sum(weights),
                             _np.sum(weights * x_trn_poly[:, 2])],
                            [_np.sum(weights * x_trn_poly[:, 2]),
                             _np.sum(weights * x_trn_poly[:, 2] *
                                     x_trn_poly[:, 2])]
                            ])
                        ordin = _np.array(
                            [_np.sum(weights * y_trn),
                             _np.sum(weights * y_trn * x_trn_poly[:, 2])]
                            )
                        beta = _linalg.solve(
                            a=coeff,
                            b=ordin
                            )
                        y_trn_model[i] = (
                            beta[0] + beta[1] * x_trn_poly[:, 2][i]
                            )

                residuals = y_trn - y_trn_model
                s = _np.median(_np.abs(residuals))
                delta = _np.clip(residuals / (6.0 * s), -1, 1)
                delta = (1 - delta ** 2) ** 2

                if _np.nansum(delta) == 0:
                    break

            # Test the results.
            y_tst_model = _np.interp(
                x=x_tst,
                xp=x_trn[:, 0],
                fp=y_trn_model,
                left=_np.nan,
                right=_np.nan
                )
            mae.append(
                _np.nansum(_np.abs(y_tst - y_tst_model)) /
                _np.sum(_np.isfinite(y_tst_model))
                )

        crossval[f] = _np.mean(mae)

    return(min(crossval.keys(), key=(lambda k: crossval[k])))


def quadratic_lowess(x, y, f=0.6667, it=2):
    """
    """
    n = len(x)
    y_model = _np.zeros(n)
    r = int(_np.ceil(f * n))
    h = []

    for i in range(n):
        dist = _np.unique(_np.sort(_np.abs(x - x[i])))

        if len(dist) <= r:
            h.append(dist[-1])

        else:
            h.append(dist[r])

    w = _np.clip(
        a=_np.abs((x[:, None] - x[None, :]) / h),
        a_min=0.0,
        a_max=1.0
        )
    w = (1 - w ** 3) ** 3
    delta = _np.ones(n)
    x = _np.expand_dims(x, 1)
    poly = _PolyFeat(degree=2)
    x_poly = poly.fit_transform(x)

    for iteration in range(it):
        for i in range(n):
            weights = delta * w[:, i]

            if _np.sum(weights) == 0:
                pass

            else:
                coeff = _np.array([
                    [_np.sum(weights),
                     _np.sum(weights * x_poly[:, 2])],
                    [_np.sum(weights * x_poly[:, 2]),
                     _np.sum(weights * x_poly[:, 2] *
                             x_poly[:, 2])]
                    ])
                ordin = _np.array(
                    [_np.sum(weights * y),
                     _np.sum(weights * y * x_poly[:, 2])]
                    )
                beta = _linalg.solve(
                    a=coeff,
                    b=ordin
                    )
                y_model[i] = (
                    beta[0] + beta[1] * x_poly[:, 2][i]
                    )

        residuals = y - y_model
        s = _np.median(_np.abs(residuals))
        delta = _np.clip(residuals / (6.0 * s), -1, 1)
        delta = (1 - delta ** 2) ** 2

        if _np.nansum(delta) == 0:
            break

    return(y_model)
