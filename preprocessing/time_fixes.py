"""Repair of a known time-axis defect of the MMLEAv2 archive, applied where the pipeline opens raw files.

The MMLEAv2 E3SMv1 files (E3SM-1-0, r*i2p2f1; every variable, Amon and Omon) carry a time axis one month early:
files named 185001-210012 have time bounds from 1849-12 to 2100-11. The monthly values are right; only the labels
are off. Against the ESGF E3SM-1-0 files the tropical-mean SST correlates 0.998 at a one-month lag and 0.96 at lag
0 (analysis/checks/planton_boxes_check.py, 2026-09-28).

month_offset() compares a file's time axis with the YYYYMM-YYYYMM span in its name. A file counts as one month
early only when its first and last months are both one month before the span and it has exactly the number of
months the span implies, so a scenario file that merely repeats December 2014 is not touched. fix_month_offset()
moves every time step and the time bounds of such a file one month later; shifted_copy() does the same for tools
that read the files themselves (cdo), writing a shifted copy with cdo shifttime. Every other disagreement between
name and axis is left alone.

Not repaired, because nothing in the pipeline reads them: the 6S to 6N E3SMv1 subsets
(*_lat-6.0_6.0_lon110.0_285.0.nc) carry the same shifted data with the first record dropped (labels 1850-01 to
2100-11, 3011 months), which cannot be told apart from a file that lacks its last month. Every paper product reads
the 20S to 20N subsets (config.LAT_MIN, LAT_MAX).

Copy of scripts/analysis/enso_future/time_fixes.py on lorenz.
"""

import logging
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

log = logging.getLogger(__name__)
_SPAN = re.compile(r"_(\d{4})(\d{2})-(\d{4})(\d{2})(?=[_.])")


def _ym(t) -> int:
    if isinstance(t, np.datetime64):
        t = pd.Timestamp(t)
    return t.year * 12 + t.month - 1


def _plus_one_month(values):
    v = np.asarray(values)
    if np.issubdtype(v.dtype, np.datetime64):
        return (pd.DatetimeIndex(v.ravel()) + pd.DateOffset(months=1)).values.reshape(v.shape)
    out = [t.replace(year=t.year + (t.month == 12), month=t.month % 12 + 1, day=min(t.day, 28)) for t in v.ravel()]
    return np.array(out, dtype=object).reshape(v.shape)


def month_offset(times, path) -> int:
    """-1 when the axis is exactly one month earlier than the span in the file name, else 0."""
    m = _SPAN.search(Path(path).name)
    if m is None or len(times) == 0 or isinstance(times[0], (float, np.floating)):
        return 0
    start = int(m.group(1)) * 12 + int(m.group(2)) - 1
    end = int(m.group(3)) * 12 + int(m.group(4)) - 1
    first, last = _ym(times[0]), _ym(times[-1])
    n = len({_ym(t) for t in times})  # distinct months: some files repeat their historical part
    if first == start - 1 and last == end - 1 and n == end - start + 1:
        return -1
    return 0


def fix_month_offset(ds: xr.Dataset, path) -> xr.Dataset:
    """Move the time axis (and bounds) of a one-month-early file one month later; other files unchanged."""
    if "time" not in ds.coords or month_offset(ds["time"].values, path) == 0:
        return ds
    log.warning(f"  {Path(path).name}: time axis one month early, moved one month later (time_fixes)")
    attrs, enc = ds["time"].attrs.copy(), ds["time"].encoding.copy()
    ds = ds.assign_coords(time=("time", _plus_one_month(ds["time"].values), attrs))
    ds["time"].encoding = enc
    for b in {attrs.get("bounds"), "time_bnds", "time_bounds"} - {None}:
        if b in ds:
            ds[b] = (ds[b].dims, _plus_one_month(ds[b].values), ds[b].attrs)
    return ds


def needs_shift(path) -> bool:
    with xr.open_dataset(path, use_cftime=True) as ds:
        return "time" in ds.coords and month_offset(ds["time"].values, path) == -1


def shifted_copy(path, tmpdir, cdo="cdo") -> Path:
    """The file itself, or a copy one month later (cdo shifttime) when its axis is one month early."""
    path = Path(path)
    if not needs_shift(path):
        return path
    out = Path(tmpdir) / f"shifted_{path.name}"
    if not out.exists():
        subprocess.run([cdo, "-s", "-O", "shifttime,1month", str(path), str(out)], check=True, capture_output=True)
        log.warning(f"  {path.name}: time axis one month early, cdo shifttime copy used (time_fixes)")
    return out
