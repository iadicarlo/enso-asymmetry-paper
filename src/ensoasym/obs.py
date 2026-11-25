"""Observed indices computed with the model pipeline's functions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from . import events as ev
from .io import CLASSIFICATION, ROOT, PHASE_CODES

OBS = ROOT / "data" / "obs"
_ALL_SST_SETS = ("ERSSTv6", "COBE2", "ERSSTv5", "HadISST")
# under the relative-index classification a product needs its global-field box file (data/obs/relative_boxes_<set>.nc)
SST_SETS = tuple(s for s in _ALL_SST_SETS if CLASSIFICATION != "noaa" or (ROOT / "data" / "obs" / f"relative_boxes_{s}.nc").exists())
PR_SETS = ("GPCP", "CMAP")
EVENT_KW = dict(
    min_duration=5,
    extreme_el_nino_sst_std_threshold=1.75,
    extreme_el_nino_pr_threshold=5.0,
    strong_el_nino_sst_lower=1.25,
    strong_el_nino_sst_upper=1.75,
    extreme_la_nina_sst_std_threshold=-1.75,
    strong_la_nina_sst_lower=-1.75,
    strong_la_nina_sst_upper=-1.25,
    extreme_min_consecutive=3,
)


def month_start(da):
    return da.assign_coords(time=pd.DatetimeIndex(da.time.values).to_period("M").to_timestamp())


def berkeley_gwi() -> xr.DataArray:
    """Berkeley Earth global mean anomaly relative to 1850-1900, degree-4 polynomial smoothed."""
    rows = []
    for line in open(OBS / "berkley_temperature_anom.txt", encoding="ascii", errors="ignore"):
        p = line.split()
        if not p or line.startswith("%") or len(p) < 3:
            continue
        try:
            y, m, a = int(p[0]), int(p[1]), float(p[2])
        except ValueError:
            continue
        if 1700 <= y <= 2200 and 1 <= m <= 12:
            rows.append((y, m, a))
    df = (
        pd.DataFrame(rows, columns=["year", "month", "anomaly"])
        .drop_duplicates(["year", "month"])
        .sort_values(["year", "month"])
    )
    gwi = xr.DataArray(
        df.anomaly.values, dims=["time"], coords={"time": pd.to_datetime(df[["year", "month"]].assign(day=1))}
    )
    gwi = gwi - gwi.sel(time=slice("1850", "1900")).mean()
    return month_start(ev.lowpass_filter_gwi(gwi, polynomial_degree=4))


def detrend_on_gwi(series: xr.DataArray, gwi: xr.DataArray) -> xr.DataArray:
    """Residual of an OLS regression of the series on the smoothed GWI (1-D version
    of the pipeline's grid-point regression; OLS is linear, so box means commute)."""
    y = series.values.astype(float)
    x = gwi.values.astype(float)
    ok = np.isfinite(x) & np.isfinite(y)
    beta, alpha = np.polyfit(x[ok], y[ok], 1)
    return series - (alpha + beta * gwi)


def obs_indices(sst_set: str, pr_set: str, start="1980-01-01", end="2025-12-01") -> dict:
    # the record ends with December 2025 so that every calendar year entering the annual regressions is complete
    """Standardised Niño indices, phase indicator and mean-state boxes for one dataset pair."""
    gwi = berkeley_gwi().sel(time=slice(start, end))
    b = month_start(xr.open_dataset(OBS / f"boxes_{sst_set}.nc")).sel(time=slice(start, end))
    pr = month_start(xr.open_dataset(OBS / f"boxes_{pr_set}.nc")["nino3"]).sel(time=slice(start, end))
    b, pr, gwi = xr.align(b, pr, gwi, join="inner")
    pr_anom = ev.remove_seasonal_cycle_rolling_with_std(pr, window_years=30)["anomaly"]
    if CLASSIFICATION == "noaa":
        # the models' rainfall anomaly is regressed on the warming index before the extreme gate and the rainfall
        # responses; treat the observed record the same way (Berkeley GWI)
        from .classification import DRIFT, PR_DETREND, regress_on_gwi

        if DRIFT == "regress" and PR_DETREND != "none":
            pr_anom = pr_anom.copy(data=regress_on_gwi(pr_anom.values, gwi.reindex(time=pr_anom.time).values))
    if CLASSIFICATION == "noaa":
        # relative indices from the global fields (data/obs/relative_boxes_<set>.nc), NOAA's rule
        # same steps as classification._noaa_series_table for the models: rolling anomaly of the relative box,
        # residual drift regressed on the warming index (ENSO_NOAA_DRIFT), per-month standardisation, NOAA's rule
        from .classification import DRIFT, noaa_phase, regress_on_gwi, rolling_anomaly, standardise_monthly

        rel = month_start(xr.open_dataset(OBS / f"relative_boxes_{sst_set}.nc")).sel(time=slice(start, end))
        rel, pr_anom, gwi_r = xr.align(rel, pr_anom, gwi, join="inner")
        t = rel.time.values
        an = {}
        for k in ("nino34", "nino3", "nino4"):
            a = rolling_anomaly(rel[f"{k}_rel"].values, t)
            an[k] = regress_on_gwi(a, gwi_r.values) if DRIFT == "regress" else a
        z = {k: standardise_monthly(np.nan_to_num(a), t) for k, a in an.items()}
        n34f = xr.DataArray(np.nan_to_num(z["nino34"]), dims=("time",), coords={"time": t})
        n3 = xr.DataArray(np.nan_to_num(z["nino3"]), dims=("time",), coords={"time": t})
        ph, evs = noaa_phase(an["nino34"], z["nino3"], z["nino4"], pr_anom.values, t)
    else:
        idx = {}
        for k in ("nino34", "nino3", "nino4"):
            det = detrend_on_gwi(b[k], gwi)
            idx[k] = ev.remove_seasonal_cycle_rolling_with_std(
                det, window_years=30, rolling_std=False, baseline_period=(1980, 2014)
            )["standardized"]
        n34, n3, n4, pr_anom = xr.align(idx["nino34"], idx["nino3"], idx["nino4"], pr_anom, join="inner")
        n34f = ev.fill_small_enso_gaps(n34, phase="el_nino", gap_threshold_months=2)
        n34f = ev.fill_small_enso_gaps(n34f, phase="la_nina", gap_threshold_months=2)
        evs = ev.detect_enso_events(nino34_anom=n34f, nino3_sst_std=n3, nino3_pr=pr_anom, nino4_sst_std=n4, **EVENT_KW)
        evs = ev.merge_events_by_gap(evs, nino34_anom=n34f, gap_months_threshold=3, min_duration=5)
        ph = np.zeros(n34f.size, int)
        for name, c in PHASE_CODES.items():
            if c == 0:
                continue
            for e in evs.get(name, []):
                ph[int(e["start_idx"]) : int(e["end_idx"])] = c
    ann = b.groupby("time.year").mean()
    ann["sst_gradient"] = ann["west"] - ann["east"]
    ann["wp_minus_ct"] = ann["warm_pool"] - ann["cold_tongue"]
    g_year = gwi.groupby("time.year").mean()
    return dict(
        sst=sst_set,
        pr=pr_set,
        time=n34f.time.values,
        nino34=n34f.values,
        nino3=n3.values,
        phase=ph,
        events=evs,
        annual=ann.to_dataframe(),
        gwi_year=g_year.to_series(),
        gwi=gwi,
    )
