"""ENSO asymmetry and per-side statistics on GWI-binned monthly series."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .io import Member, PHASE_CODES

EN_ALL = (1, 2, 3)
LN_ALL = (-1, -2, -3)
EN_STRONG = (2, 3)
LN_STRONG = (-2, -3)


def bin_mask(m: Member, lo: float, hi: float) -> np.ndarray:
    g = m.gwi
    return np.isfinite(g) & np.isfinite(m.nino34) & (g >= lo) & (g < hi)


def contributions(x: np.ndarray, ph: np.ndarray) -> dict:
    """Exact decomposition of mean(x^2) into phase contributions.

    C_k = mean(x^2 * 1_k), so sum over k of C_k equals the variance of the
    (zero-mean, standardised) index.  Also returns frequency f_k and mean
    squared amplitude within phase.
    """
    n = x.size
    out = {"var": float(np.mean(x**2)), "n": n}
    for name, code in PHASE_CODES.items():
        sel = ph == code
        nk = int(sel.sum())
        out[f"C_{name}"] = float(np.sum(x[sel] ** 2) / n) if nk else 0.0
        out[f"f_{name}"] = nk / n
    for side, codes in (("EN", EN_ALL), ("LN", LN_ALL), ("EN_strong", EN_STRONG), ("LN_strong", LN_STRONG)):
        sel = np.isin(ph, codes)
        nk = int(sel.sum())
        out[f"C_{side}"] = float(np.sum(x[sel] ** 2) / n) if nk else 0.0
        out[f"f_{side}"] = nk / n
        out[f"x2_{side}"] = float(np.mean(x[sel] ** 2)) if nk else np.nan
    for tag in ("", "_strong"):
        ce, cl = out[f"C_EN{tag}"], out[f"C_LN{tag}"]
        out[f"A{tag}"] = (ce - cl) / (ce + cl) if (ce + cl) > 0 else np.nan
    return out


def event_rates(ph: np.ndarray) -> dict:
    """Events per year by tier, counting runs of identical non-zero phase codes."""
    out = {}
    n_yr = ph.size / 12.0
    for name, code in PHASE_CODES.items():
        if code == 0:
            continue
        sel = (ph == code).astype(int)
        starts = int(np.sum(np.diff(np.concatenate([[0], sel])) == 1))
        out[f"rate_{name}"] = starts / n_yr
    out["rate_ext_EN"] = out["rate_extreme_el_nino"]
    out["rate_ext_LN"] = out["rate_extreme_la_nina"]
    out["rate_str_EN"] = out["rate_strong_el_nino"]
    out["rate_str_LN"] = out["rate_strong_la_nina"]
    return out


def _stats_on_mask(m: Member, sel: np.ndarray) -> dict:
    d = contributions(m.nino34[sel], m.phase[sel])
    d.update(event_rates(m.phase[sel]))
    d["gwi_mean"] = float(np.mean(m.gwi[sel]))
    n3 = m.nino3[sel]
    d["skew_n3"] = float(stats.skew(n3[np.isfinite(n3)])) if np.isfinite(n3).sum() > 24 else np.nan
    d["skew_n34"] = float(stats.skew(m.nino34[sel]))
    return d


def member_period_stats(m: Member, y0: int, y1: int, min_months: int = 120) -> dict | None:
    """Statistics over a fixed calendar period (inclusive years)."""
    yrs = pd.DatetimeIndex(m.time).year
    sel = np.isfinite(m.nino34) & (yrs >= y0) & (yrs <= y1)
    if sel.sum() < min_months:
        return None
    d = _stats_on_mask(m, sel)
    ann = m.annual.loc[(m.annual.index >= y0) & (m.annual.index <= y1)]
    for col in ann.columns:
        d[col] = float(np.nanmean(ann[col].values)) if ann[col].notna().sum() >= 5 else np.nan
    if "warm_pool_sst" in ann and "cold_tongue_sst" in ann:
        d["wp_minus_ct"] = d["warm_pool_sst"] - d["cold_tongue_sst"]
    return d


def member_bin_stats(m: Member, lo: float, hi: float, min_months: int = 120) -> dict | None:
    sel = bin_mask(m, lo, hi)
    if sel.sum() < min_months:
        return None
    d = _stats_on_mask(m, sel)
    # annual mean-state averages over the years whose mean GWI falls in the bin
    yrs = pd.DatetimeIndex(m.time).year
    g_year = pd.Series(m.gwi, index=yrs).groupby(level=0).mean()
    in_bin = g_year[(g_year >= lo) & (g_year < hi)].index
    ann = m.annual.reindex(in_bin)
    for col in ann.columns:
        d[col] = float(np.nanmean(ann[col].values)) if ann[col].notna().sum() >= 5 else np.nan
    if "warm_pool_sst" in ann and "cold_tongue_sst" in ann:
        d["wp_minus_ct"] = d["warm_pool_sst"] - d["cold_tongue_sst"]
    return d


def model_delta_table(members: list[Member], ref, warm, periods: bool = False) -> pd.DataFrame:
    """Per-member (ref, warm, delta) rows for one model.

    With ``periods=True`` the reference and warm states are fixed calendar
    periods (ref = (y0, y1), warm = (y0, y1)) instead of GWI bins.
    """
    rows = []
    for m in members:
        if periods:
            r = member_period_stats(m, *ref)
            w = member_period_stats(m, *warm)
        else:
            r = member_bin_stats(m, *ref)
            w = member_bin_stats(m, *warm)
        if r is None or w is None:
            continue
        row = {"model": m.model, "rid": m.rid, "scenario": m.scenario}
        for k in r:
            if k == "n":
                continue
            row[f"{k}_ref"] = r[k]
            row[f"{k}_warm"] = w[k]
            row[f"d_{k}"] = w[k] - r[k]
        row["dgwi"] = w["gwi_mean"] - r["gwi_mean"]
        rows.append(row)
    return pd.DataFrame(rows)


def gwl_trajectory(
    members: list[Member], centres: np.ndarray, width: float = 1.0, min_months: int = 120
) -> pd.DataFrame:
    """Per-member statistics in overlapping GWI bins centred on ``centres``."""
    rows = []
    for m in members:
        for c in centres:
            lo, hi = c - width / 2, c + width / 2
            d = member_bin_stats(m, lo, hi, min_months)
            if d is None:
                continue
            d.update(model=m.model, rid=m.rid, lo=lo, hi=hi, mid=0.5 * (lo + hi))
            rows.append(d)
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame, cols, by="model") -> pd.DataFrame:
    """Per-model median and standard error of the median across members."""

    def agg(g):
        out = {"n_members": len(g)}
        for c in cols:
            v = g[c].dropna().values
            out[c] = np.median(v) if v.size else np.nan
            out[f"{c}_se"] = 1.253 * np.std(v, ddof=1) / np.sqrt(v.size) if v.size > 1 else np.nan
            out[f"{c}_p10"] = np.percentile(v, 10) if v.size else np.nan
            out[f"{c}_p90"] = np.percentile(v, 90) if v.size else np.nan
        return pd.Series(out)

    return df.groupby(by).apply(agg, include_groups=False)
