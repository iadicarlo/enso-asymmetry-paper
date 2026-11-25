"""The convective threshold of Johnson and Kosaka (2016, Fig. 2d): helpers shared by the side scripts.

December to March means of OLR (or rainfall) against relative SST (box SST minus the 20S to 20N mean) in the eastern
equatorial Pacific (5S to 5N, 160W to 120W), fitted with a hinge: flat below a threshold, linear above it.
"""

import numpy as np
import pandas as pd
import xarray as xr

from .io import ROOT

OBS = ROOT / "data" / "obs"
BOX = (200, 240)  # longitudes (degrees east) of the eastern box, 5S to 5N


def hinge(x, y, sign):
    """y = a for x < T, a + b (x - T) above; sign -1 for OLR (falls), +1 for rainfall (rises). Returns T, a, b, rmse."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 20:
        return np.nan, np.nan, np.nan, np.nan
    best = (np.inf, np.nan, np.nan, np.nan)
    for T in np.quantile(x, np.linspace(0.05, 0.95, 91)):
        X = np.column_stack([np.ones_like(x), np.clip(x - T, 0, None)])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        if np.sign(coef[1]) != sign:
            continue
        r = np.mean((y - X @ coef) ** 2)
        if r < best[0]:
            best = (r, T, coef[0], coef[1])
    return best[1], best[2], best[3], np.sqrt(best[0])


def hinge_boot(x, y, sign, n=500, seed=1):
    """5th and 95th percentiles of the threshold over bootstrap resamples of the points."""
    rng = np.random.default_rng(seed)
    t = [hinge(x[i], y[i], sign)[0] for i in (rng.integers(0, len(x), len(x)) for _ in range(n))]
    return np.nanpercentile(t, 5), np.nanpercentile(t, 95)


def mkeys(t):
    """Month numbers, year * 12 + month - 1."""
    t = pd.DatetimeIndex(t)
    return (t.year * 12 + t.month - 1).values


def djfm(keys, v):
    """December to March means, labelled by the January year (complete winters only)."""
    keys, v = np.asarray(keys), np.asarray(v, dtype=float)
    yr, mon = keys // 12, keys % 12
    win = np.where(mon == 11, yr + 1, yr)
    sel = np.isin(mon, [11, 0, 1, 2])
    s = pd.Series(v[sel]).groupby(win[sel]).agg(["mean", "count"])
    return s["mean"][s["count"] == 4]


def opn(f):
    """An observed file on the archive grid; cdo leaves the source grid in some, which is dropped."""
    d = xr.open_dataset(OBS / f, drop_variables=["time_bnds", "time_bounds", "lat_bnds", "lon_bnds", "nbnds", "bnds"])
    extra = {k: k.split("_")[0] for k in d.dims if k.startswith(("lat_", "lon_"))}
    return d.drop_dims([v for v in extra.values() if v in d.dims]).rename(extra)


def box_mean(da, lo=BOX[0], hi=BOX[1]):
    d = da.sel(lat=slice(-5, 5), lon=slice(lo, hi))
    return d.weighted(np.cos(np.deg2rad(d.lat))).mean(["lat", "lon"])


TIERS = {"extreme_el_nino": 3, "strong_el_nino": 2, "moderate_el_nino": 1, "neutral": 0,
         "moderate_la_nina": -1, "strong_la_nina": -2, "extreme_la_nina": -3}


def paper_winter_tiers(sst_set="ERSSTv6"):
    """Tier of each observed winter under the paper's classification (NOAA rule on the relative Nino-3.4 index,
    results/tables/obs_events_noaa.csv): the strongest tier among its December to March months, provided at least two of
    the four months share its sign, else neutral."""
    ev = pd.read_csv(ROOT / "results" / "tables" / "obs_events_noaa.csv")
    ev = ev[(ev.sst == sst_set) & (ev.method == "noaa_relative")]
    months = pd.period_range("1950-01", "2026-12", freq="M")
    code = pd.Series(0.0, index=months)
    for r in ev.itertuples():
        code[pd.Period(r.start, "M"):pd.Period(r.end, "M")] = TIERS[r.tier]
    k = np.array([p.year * 12 + p.month - 1 for p in months])
    yr, mon = k // 12, k % 12
    win, sel = np.where(mon == 11, yr + 1, yr), np.isin(mon, [11, 0, 1, 2])

    def one(c):
        top = c.iloc[np.argmax(np.abs(c.values))]
        return top if (np.sign(c) == np.sign(top)).sum() >= 2 else 0.0

    return pd.Series(code.values[sel]).groupby(win[sel]).apply(one)


def paper_index(sst_set="ERSSTv6"):
    """The paper's standardised relative Nino-3.4 index (30-year rolling seasonal cycle, 1980 to 2014 standard deviation)."""
    import contextlib
    import io

    from . import events as ev

    rb = xr.open_dataset(OBS / f"relative_boxes_{sst_set}.nc")
    with contextlib.redirect_stdout(io.StringIO()):
        z = ev.remove_seasonal_cycle_rolling_with_std(rb["nino34_rel"].sel(time=slice("1950", "2026")), window_years=30,
                                                      rolling_std=False, baseline_period=(1980, 2014))["standardized"]
    return djfm(mkeys(z.time.values), z.values)


def observed(sst_set="ERSSTv6", years=(1980, 2025)):
    """Observed winters in the eastern box: relative SST (sst_set minus its 20S to 20N mean, all longitudes), NOAA OLR
    CDR v3 and GPCP rainfall; plus the paper's relative Nino-3.4 and Nino-3 SST, its standardised Nino-3.4 index and
    the paper's tier of the winter."""
    f = {"ERSSTv6": "ersstv6_g025_tropics.nc", "COBE2": "cobe2_g025_tropics.nc"}[sst_set]
    sst = box_mean(opn(f)["sst"])
    rb = xr.open_dataset(OBS / f"relative_boxes_{sst_set}.nc")
    tm = rb["trop"]
    rs = (pd.Series(sst.values, index=mkeys(sst.time.values)) - pd.Series(tm.values, index=mkeys(tm.time.values))).dropna()
    olr = box_mean(opn("olr_cdr_v3_g025_tropics.nc")["olr"])
    pr = box_mean(opn("gpcp_g025_tropics.nc")["precip"])
    df = pd.DataFrame({"rsst": djfm(rs.index.values, rs.values), "olr": djfm(mkeys(olr.time.values), olr.values),
                       "pr": djfm(mkeys(pr.time.values), pr.values),
                       "rel34": djfm(mkeys(rb.time.values), rb["nino34_rel"].values),
                       "rel3": djfm(mkeys(rb.time.values), rb["nino3_rel"].values), "z34": paper_index(sst_set)})
    df["tier"] = paper_winter_tiers(sst_set).reindex(df.index).fillna(0)
    return df.loc[years[0]:years[1]].dropna(subset=["rsst", "olr", "pr"])


def brokenstick(x, y, n_t=150, q=(0.05, 0.95)):
    """Continuous two-segment regression y = a + b1 min(x - T, 0) + b2 max(x - T, 0) (segmented, or broken-stick,
    regression), T by profile least squares on a grid between the q quantiles of x. Returns dict(T, a, b1, b2, sse, n,
    aic_gain) where aic_gain is the AIC of a single straight line minus that of the broken line (positive: the break is
    supported; one extra parameter counted for T)."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x)[ok], np.asarray(y)[ok]
    n = len(x)
    if n < 20:
        return dict(T=np.nan, a=np.nan, b1=np.nan, b2=np.nan, sse=np.nan, n=n, aic_gain=np.nan)
    best = None
    for T in np.linspace(*np.quantile(x, q), n_t):
        X = np.column_stack([np.ones(n), np.minimum(x - T, 0), np.maximum(x - T, 0)])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        sse = float(np.sum((y - X @ coef) ** 2))
        if best is None or sse < best[0]:
            best = (sse, T, coef)
    sse, T, (a, b1, b2) = best
    lin = np.polyfit(x, y, 1)
    sse1 = float(np.sum((y - np.polyval(lin, x)) ** 2))
    aic_gain = (n * np.log(sse1 / n) + 2 * 2) - (n * np.log(sse / n) + 2 * 4)
    return dict(T=T, a=a, b1=b1, b2=b2, sse=sse, n=n, aic_gain=aic_gain)


def brokenstick_curve(p, xx):
    return p["a"] + p["b1"] * np.minimum(xx - p["T"], 0) + p["b2"] * np.maximum(xx - p["T"], 0)


def brokenstick_boot(df, xcol, ycol, group=None, n=200, seed=1, n_t=80):
    """Bootstrap of the broken-stick fit: resampling whole groups (members, or years) with replacement when group is given,
    else rows. Returns the list of parameter dicts."""
    rng = np.random.default_rng(seed)
    out = []
    if group is not None:
        parts = {k: v for k, v in df.groupby(group)}
        keys = list(parts)
    for _ in range(n):
        if group is not None:
            s = pd.concat([parts[k] for k in rng.choice(keys, len(keys), replace=True)])
        else:
            s = df.iloc[rng.integers(0, len(df), len(df))]
        out.append(brokenstick(s[xcol].values, s[ycol].values, n_t=n_t))
    return out
