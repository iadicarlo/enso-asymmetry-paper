"""ENSO classification of one member or one observed record, in the two conventions this project knows.

``noaa``       relative Nino indices (box minus the concurrent 20S to 20N tropical mean), 30-year rolling
               climatology, per-calendar-month standardisation on 1980 to 2014, NOAA's episode rule with
               cumulative intensity tiers (src/ensoasym/noaa_detector.py, vendored from the ENSOscope platform).
``gwi_legacy`` the paper's previous convention: regression on the smoothed GWI, monthly rule with gap bridging,
               December to February phase lock and banded tiers, as stored in the lorenz metrics files.

The convention is chosen once, through the environment variable ENSO_CLASSIFICATION (default ``noaa``), so that
every table and figure script runs unchanged under either.
"""

from __future__ import annotations

import contextlib
import functools
import io
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from . import events as ev
from . import noaa_detector as nd

CLASSIFICATION = os.environ.get("ENSO_CLASSIFICATION", "noaa").lower()
# The rolling climatology cannot be centred in the last 15 years of a record, so a record that ends during rapid
# warming keeps a residual drift there (up to +0.16 sd in the relative Nino-3.4 of models with El Nino-like relative
# warming, which inflates the El Nino share of the warm state). "regress" removes it by regressing the relative
# anomaly of each member (or dataset) on its own smoothed warming index before standardisation; "none" is the
# platform's operational form.
DRIFT = os.environ.get("ENSO_NOAA_DRIFT", "regress").lower()
# Rainfall anomaly of the models: "gwi" (default) reads <model>_extra_vars.nc, whose Nino-3 rainfall was regressed on
# the warming index before deseasonalising on lorenz; "none" reads <model>_extra_vars_prroll.nc, the rolling
# climatology only (the platform's operational form), a sensitivity variant for the supplement.
PR_DETREND = os.environ.get("ENSO_PR_DETREND", "gwi").lower()
EXTRA_SUFFIX = "_extra_vars_prroll.nc" if PR_DETREND == "none" else "_extra_vars.nc"
BASELINE = (1980, 2014)
KGM2S_TO_MMDAY = 86400.0
PHASE_CODES = {"extreme_el_nino": 3, "strong_el_nino": 2, "moderate_el_nino": 1, "neutral": 0,
               "moderate_la_nina": -1, "strong_la_nina": -2, "extreme_la_nina": -3}


def rolling_anomaly(values, time, window_years=30):
    """The 30-year rolling climatology of events.py, vectorised: returns the anomaly, in the series' units.

    Same rule as ``events.remove_seasonal_cycle_rolling_with_std``: for every year the climatology of each calendar
    month is the mean over the years from y - 15 to y + 15, the window shifted to stay inside the record at both
    ends, missing values skipped. ``values`` may be 1-D (time) or 2-D (members, time); the result has the same
    shape. Identical to the loop version to floating precision, about a thousand times faster.
    """
    x = np.asarray(values, float)
    squeeze = x.ndim == 1
    x = np.atleast_2d(x)
    t = pd.DatetimeIndex(time)
    years, months = t.year.values, t.month.values
    y0, y1 = int(years.min()), int(years.max())
    n_years = y1 - y0 + 1
    grid = np.full((x.shape[0], n_years, 12), np.nan)
    grid[:, years - y0, months - 1] = x
    present = np.zeros((n_years, 12), bool)
    present[years - y0, months - 1] = True
    half = window_years // 2
    min_periods = min(10, window_years // 3)
    clim = np.full_like(grid, np.nan)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for i in range(n_years):
            ys, ye = y0 + i - half, y0 + i + half
            if ys < y0:
                ys, ye = y0, min(y0 + window_years - 1, y1)
            elif ye > y1:
                ye, ys = y1, max(y1 - window_years + 1, y0)
            sl = slice(ys - y0, ye - y0 + 1)
            ok = present[sl].sum(axis=0) >= min_periods
            c = np.nanmean(grid[:, sl, :], axis=1)
            c[:, ~ok] = np.nan
            clim[:, i, :] = c
    anom = (grid - clim)[:, years - y0, months - 1]
    return anom[0] if squeeze else anom


def regress_on_gwi(a, gwi):
    """Residual of an OLS regression of an anomaly series on the smoothed warming index (drift removal)."""
    a = np.asarray(a, float); g = np.asarray(gwi, float)
    ok = np.isfinite(a) & np.isfinite(g)
    if ok.sum() < 24:
        return a
    beta, alpha = np.polyfit(g[ok], a[ok], 1)
    return a - (alpha + beta * np.where(np.isfinite(g), g, np.nanmean(g)))


def standardise_monthly(x, time, baseline=BASELINE):
    """Divide each calendar month by its population standard deviation over the baseline years."""
    t = pd.DatetimeIndex(time)
    mon, yr = t.month.values, t.year.values
    out = np.full(np.shape(x), np.nan)
    for m in range(1, 13):
        sel = mon == m
        s = np.std(x[sel & (yr >= baseline[0]) & (yr <= baseline[1])])
        out[sel] = x[sel] / s if s > 0 else np.nan
    return out


def noaa_phase(a34_degc, z3, z4, pr_mm, time):
    """Monthly phase codes from NOAA's rule: episodes on the smoothed relative Nino-3.4, tiers on Nino-3 and Nino-4."""
    da = lambda a: xr.DataArray(np.nan_to_num(np.asarray(a, float)), dims=("time",), coords={"time": time})  # noqa: E731
    with contextlib.redirect_stdout(io.StringIO()):
        evs = nd.detect_enso_events_noaa(da(a34_degc), da(z3), da(pr_mm), da(z4), verbose=False)
    ph = np.zeros(len(a34_degc), int)
    for name, c in PHASE_CODES.items():
        if c == 0:
            continue
        for e in evs.get(name, []):
            ph[int(e["start_idx"]) : int(e["end_idx"])] = c
    return ph, evs


@functools.lru_cache(maxsize=64)
def _open(path: str):
    p = Path(path)
    return xr.open_dataset(p) if p.exists() else None


def _noaa_series_table(data_dir: Path, model: str, scenario: str, time):
    """Classify every member of one (model, scenario) file once and cache the result on disk.

    The detector costs about a second per member, and the loader is called by several scripts, so the standardised
    indices and phase codes are kept in data/lorenz/noaa_cache/<model>_<scenario>.npz, invalidated when any input
    file changes.
    """
    box_p = data_dir / scenario / f"{model}_box_anoms.nc"
    trop_p = data_dir / "trop_mean_global" / f"{model}_{scenario}_trop_mean_global.nc"
    ex_p = data_dir / scenario / f"{model}{EXTRA_SUFFIX}"
    met_p = data_dir / scenario / f"{model}_metrics.nc"
    if not box_p.exists() or not trop_p.exists():
        return {}
    stamp = "|".join(f"{q.name}:{q.stat().st_mtime_ns}" for q in (box_p, trop_p, ex_p, met_p) if q.exists()) + f"|drift={DRIFT}|pr={PR_DETREND}"
    cache_p = data_dir / "noaa_cache" / f"{model}_{scenario}.npz"
    if cache_p.exists():
        z = np.load(cache_p, allow_pickle=False)
        if str(z["stamp"]) == stamp and z["n_time"] == len(time):
            rids = [str(r) for r in z["rids"]]
            return {r: dict(nino34=z["nino34"][i], nino3=z["nino3"][i], phase=z["phase"][i]) for i, r in enumerate(rids)}
    rb, tt, ee = xr.open_dataset(box_p), xr.open_dataset(trop_p), (xr.open_dataset(ex_p) if ex_p.exists() else None)
    gw = None
    if DRIFT == "regress":
        mm = xr.open_dataset(met_p)
        gw = {str(r): np.atleast_2d(mm["gwi"].values)[i].astype(float) for i, r in enumerate(mm.ens.values)}
        mm.close()
    tindex = pd.DatetimeIndex(time).to_period("M").to_timestamp()
    ens = [str(e) for e in rb.ens.values]
    t_ens = [str(e) for e in tt.ens.values]
    e_ens = [str(e) for e in ee.ens.values] if ee is not None else []
    trop_all = tt["trop_mean_sst"].assign_coords(time=pd.DatetimeIndex(tt.time.values).to_period("M").to_timestamp()).reindex(time=tindex)
    out = {}
    if rb.sizes["time"] == len(time):
        keep = [rid for rid in ens if rid in t_ens and np.isnan(trop_all.sel(ens=rid).values).mean() <= 0.05]
        trop_anoms = dict(zip(keep, rolling_anomaly(np.stack([trop_all.sel(ens=r).values for r in keep]), time))) if keep else {}
        for k, rid in enumerate(ens):
            if rid not in trop_anoms:
                continue
            trop_anom = trop_anoms[rid]
            a34 = rb["nino34_anom"].values[k] - trop_anom
            a3 = rb["nino3_anom"].values[k] - trop_anom
            a4 = rb["nino4_anom"].values[k] - trop_anom
            if gw is not None:
                if rid not in gw or len(gw[rid]) != len(time):
                    continue
                a34, a3, a4 = (regress_on_gwi(a, gw[rid]) for a in (a34, a3, a4))
            z34, z3, z4 = (standardise_monthly(np.nan_to_num(a), time) for a in (a34, a3, a4))
            pr = np.zeros(len(time))
            if rid in e_ens:
                pr = np.nan_to_num(ee["nino3_pr_anom"].values[e_ens.index(rid)] * KGM2S_TO_MMDAY)
            ph, _ = noaa_phase(a34, z3, z4, pr, time)
            out[rid] = dict(nino34=np.nan_to_num(z34), nino3=np.nan_to_num(z3), phase=ph)
    for d in (rb, tt, ee):
        if d is not None:
            d.close()
    if out:
        cache_p.parent.mkdir(parents=True, exist_ok=True)
        rids = list(out)
        np.savez(cache_p, stamp=stamp, n_time=len(time), rids=np.array(rids),
                 nino34=np.stack([out[r]["nino34"] for r in rids]), nino3=np.stack([out[r]["nino3"] for r in rids]),
                 phase=np.stack([out[r]["phase"] for r in rids]))
    return out


_TABLE_CACHE = {}


def noaa_member_series(data_dir: Path, model: str, scenario: str, rid: str, time):
    """Standardised relative Nino-3.4 and Nino-3 and the phase codes of one member, or None if an input is missing."""
    key = (str(data_dir), model, scenario, len(time))
    if key not in _TABLE_CACHE:
        _TABLE_CACHE[key] = _noaa_series_table(data_dir, model, scenario, time)
    return _TABLE_CACHE[key].get(rid)
