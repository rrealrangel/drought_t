# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 09:29:58 2019

@author: r.realrangel
"""
from sklearn.model_selection import KFold
import numpy as np
from sklearn.preprocessing import PolynomialFeatures as PolyFeat
from scipy import linalg

# =============================================================================
# https://gist.github.com/agramfort/850437
# https://olamyy.github.io/olamyy.github.io/blog/2018/01/30
# /locally-weighted-regression/
# Lowess smoother: Robust locally weighted regression.
# The lowess function fits a nonparametric regression curve to a
# scatterplot. The arrays exog and endog contain an equal number of
# elements; each pair (exog[i], endog[i]) defines a data point in the
# scatterplot. The function returns the estimated (smooth) values of
# endog. The smoothing span is given by f. A larger value for f will
# result in a smoother curve. The number of robustifying iterations is
# given by it. The function will run faster with a smaller number of
# iterations.
# =============================================================================


def optimal_f(exog, endog, ftries=41, it=2, xval_folds=3):
    """
    """
    kf = KFold(
        n_splits=xval_folds,
        shuffle=True,
        random_state=1
        )
    f_tries = np.linspace(
        start=0,
        stop=1,
        num=ftries
        )
    f_tries = f_tries[f_tries != 0]
    crossval = {}

    for f in f_tries:
        mae = []

        for _, tst_fold in kf.split(endog):
            # Testing subsets.
            x_tst = exog[tst_fold]
            y_tst = endog[tst_fold]

            # Training subsets.
            trn_fold = [j for j in range(len(exog)) if j not in tst_fold]
            x_trn = exog[trn_fold]
            y_trn = endog[trn_fold]
            n = len(x_trn)
            y_trn_model = np.zeros(n)
            r = int(np.ceil(f * n))
            h = []

            for i in range(n):
                dist = np.unique(np.sort(np.abs(x_trn - x_trn[i])))

                if len(dist) <= r:
                    h.append(dist[-1])

                else:
                    h.append(dist[r])

            w = np.clip(
                a=np.abs((x_trn[:, None] - x_trn[None, :]) / h),
                a_min=0.0,
                a_max=1.0
                )
            w = (1 - w ** 3) ** 3
            delta = np.ones(n)
            x_trn = np.expand_dims(x_trn, 1)
            poly = PolyFeat(degree=2)
            x_trn_poly = poly.fit_transform(x_trn)

            for iteration in range(it):
                for i in range(n):
                    weights = delta * w[:, i]

                    if np.sum(weights) == 0:
                        pass

                    else:
                        coeff = np.array([
                            [np.sum(weights),
                             np.sum(weights * x_trn_poly[:, 2])],
                            [np.sum(weights * x_trn_poly[:, 2]),
                             np.sum(weights * x_trn_poly[:, 2] *
                                    x_trn_poly[:, 2])]
                            ])
                        ordin = np.array(
                            [np.sum(weights * y_trn),
                             np.sum(weights * y_trn * x_trn_poly[:, 2])]
                            )
                        beta = linalg.solve(
                            a=coeff,
                            b=ordin
                            )
                        y_trn_model[i] = (
                            beta[0] + beta[1] * x_trn_poly[:, 2][i]
                            )

                residuals = y_trn - y_trn_model
                s = np.median(np.abs(residuals))
                delta = np.clip(residuals / (6.0 * s), -1, 1)
                delta = (1 - delta ** 2) ** 2

                if np.nansum(delta) == 0:
                    break

            # Test the results.
            y_tst_model = np.interp(
                exog=x_tst,
                xp=x_trn[:, 0],
                fp=y_trn_model,
                left=np.nan,
                right=np.nan
                )
            mae.append(
                np.nansum(np.abs(y_tst - y_tst_model)) /
                np.sum(np.isfinite(y_tst_model))
                )

        crossval[f] = np.mean(mae)

    return(min(crossval.keys(), key=(lambda k: crossval[k])))


def quadratic_lowess(exog, endog, f=0.6667, it=2):
    """
    """
    n = len(exog)
    y_model = np.zeros(n)
    r = int(np.ceil(f * n))
    h = []

    for i in range(n):
        dist = np.unique(np.sort(np.abs(exog - exog[i])))

        if len(dist) <= r:
            h.append(dist[-1])

        else:
            h.append(dist[r])

    w = np.clip(
        a=np.abs((exog[:, None] - exog[None, :]) / h),
        a_min=0.0,
        a_max=1.0
        )
    w = (1 - w ** 3) ** 3
    delta = np.ones(n)
    exog = np.expand_dims(exog, 1)
    poly = PolyFeat(degree=2)
    x_poly = poly.fit_transform(exog)

    for iteration in range(it):
        for i in range(n):
            weights = delta * w[:, i]

            if np.sum(weights) == 0:
                pass

            else:
                coeff = np.array([
                    [np.sum(weights),
                     np.sum(weights * x_poly[:, 2])],
                    [np.sum(weights * x_poly[:, 2]),
                     np.sum(weights * x_poly[:, 2] *
                            x_poly[:, 2])]
                    ])
                ordin = np.array(
                    [np.sum(weights * endog),
                     np.sum(weights * endog * x_poly[:, 2])]
                    )
                beta = linalg.solve(
                    a=coeff,
                    b=ordin
                    )
                y_model[i] = (
                    beta[0] + beta[1] * x_poly[:, 2][i]
                    )

        residuals = endog - y_model
        s = np.median(np.abs(residuals))
        delta = np.clip(residuals / (6.0 * s), -1, 1)
        delta = (1 - delta ** 2) ** 2

        if np.nansum(delta) == 0:
            break

    return(y_model)
