"""Two-way fixed-effects regressions over (ensemble, warming bin) cells, shared by the within-ensemble side analyses.

The cells are a DataFrame with columns model, bin, the response (default A) and the regressors. Ensemble effects remove
every fixed difference between models, bin effects whatever all models share at a given warming, so a coefficient is
identified only by ensembles that change faster or slower than the others at the same warming.
"""

import numpy as np
import pandas as pd


def twoway(df, regs, y="A"):
    """OLS of y on regs with ensemble and bin dummies; coefficients, CR1 standard errors clustered by ensemble, and
    the number of clusters."""
    E = pd.get_dummies(df.model, drop_first=False).values.astype(float)
    B = pd.get_dummies(df["bin"], drop_first=True).values.astype(float)
    X = np.column_stack([df[regs].values, E, B])
    yy = df[y].values
    beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
    res = yy - X @ beta
    XtX_inv = np.linalg.pinv(X.T @ X)
    meat = np.zeros((X.shape[1], X.shape[1]))
    groups = df.model.values
    for gname in np.unique(groups):
        idx = groups == gname
        u = X[idx].T @ res[idx]
        meat += np.outer(u, u)
    G, N, P_ = len(np.unique(groups)), len(yy), X.shape[1]
    V = XtX_inv @ meat @ XtX_inv * G / (G - 1) * (N - 1) / (N - P_)
    return beta[: len(regs)], np.sqrt(np.diag(V)[: len(regs)]), G


def permute(df, reg, rng, n=2000, y="A"):
    """Two-sided permutation p of the coefficient of one regressor: whole trajectories of the regressor are reassigned
    among ensembles (ensemble i receives ensemble j's values bin by bin), which keeps each trajectory's smoothness."""
    b0 = twoway(df, [reg], y)[0][0]
    models = df.model.unique()
    table = df.pivot(index="model", columns="bin", values=reg)
    null = []
    for _ in range(n):
        perm = dict(zip(models, rng.permutation(models)))
        v = np.array([table.at[perm[mo], b] if b in table.columns else np.nan for mo, b in zip(df.model, df["bin"])])
        d = df.assign(**{reg: v}).dropna(subset=[reg])
        null.append(twoway(d, [reg], y)[0][0])
    null = np.array(null)
    return b0, np.mean(np.abs(null) >= abs(b0))


def demean(df, v):
    """The variable with ensemble and bin means removed (alternating projections for an unbalanced panel)."""
    x = df[v].values.astype(float).copy()
    for _ in range(50):
        x -= df.assign(_x=x).groupby("model")._x.transform("mean").values
        x -= df.assign(_x=x).groupby("bin")._x.transform("mean").values
    return x
