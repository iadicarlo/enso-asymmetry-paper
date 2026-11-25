"""Two-level bootstrap and permutation tests for cross-model relations."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def model_medians(mem: pd.DataFrame, cols) -> pd.DataFrame:
    return mem.groupby("model")[list(cols)].median()


def two_level_bootstrap(mem: pd.DataFrame, x: str, y: str, n_boot: int = 2000, seed: int = 0) -> dict:
    """Resample models with replacement, then members within each model.

    Returns the point estimates (per-model medians), bootstrap percentile
    intervals for Pearson r, Spearman rho and the OLS slope, and a
    permutation p-value for r obtained by shuffling y across models.
    """
    rng = np.random.default_rng(seed)
    groups = {m: g[[x, y]].dropna().values for m, g in mem.groupby("model")}
    groups = {m: v for m, v in groups.items() if len(v) >= 3}
    models = list(groups)
    n = len(models)
    med = np.array([np.median(groups[m], axis=0) for m in models])
    r0 = stats.pearsonr(med[:, 0], med[:, 1])
    rho0 = stats.spearmanr(med[:, 0], med[:, 1])
    slope0 = stats.linregress(med[:, 0], med[:, 1]).slope
    rs, rhos, slopes = [], [], []
    for _ in range(n_boot):
        pick = rng.integers(0, n, n)
        pts = []
        for i in pick:
            v = groups[models[i]]
            idx = rng.integers(0, len(v), len(v))
            pts.append(np.median(v[idx], axis=0))
        pts = np.array(pts)
        if np.std(pts[:, 0]) == 0 or np.std(pts[:, 1]) == 0:
            continue
        rs.append(stats.pearsonr(pts[:, 0], pts[:, 1])[0])
        rhos.append(stats.spearmanr(pts[:, 0], pts[:, 1])[0])
        slopes.append(stats.linregress(pts[:, 0], pts[:, 1]).slope)
    perm = []
    for _ in range(5000):
        perm.append(stats.pearsonr(med[:, 0], rng.permutation(med[:, 1]))[0])
    perm = np.array(perm)
    p_perm = float(np.mean(np.abs(perm) >= abs(r0[0])))
    return dict(
        x=x,
        y=y,
        n=n,
        r=r0[0],
        p_r=r0[1],
        rho=rho0[0],
        p_rho=rho0[1],
        slope=slope0,
        r_lo=np.percentile(rs, 2.5),
        r_hi=np.percentile(rs, 97.5),
        rho_lo=np.percentile(rhos, 2.5),
        rho_hi=np.percentile(rhos, 97.5),
        slope_lo=np.percentile(slopes, 2.5),
        slope_hi=np.percentile(slopes, 97.5),
        p_perm=p_perm,
        frac_r_positive=float(np.mean(np.array(rs) > 0)),
    )


def model_bootstrap_median(values: np.ndarray, n_boot: int = 5000, seed: int = 0):
    """Bootstrap interval (2.5, 97.5) for the multi-model median (equal weight per model)."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    rng = np.random.default_rng(seed)
    b = np.median(v[rng.integers(0, v.size, (n_boot, v.size))], axis=1)
    return float(np.median(v)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))
