"""Yann Planton's box averages (data/planton_boxes/raw_2026-09-28), matched to the paper's members.

Files are named project_model_family_experiment_NNNmembers_tim_<var>_val_<region>.nc with dimensions (member, time);
the member labels are in the member coordinate's "members" attribute. Checked in analysis/checks/planton_boxes_check.py.
"""

import re

import numpy as np
import pandas as pd
import xarray as xr

from .io import ROOT

RAW = ROOT / "data" / "planton_boxes" / "raw_2026-09-28"
# our ensemble -> Yann's model and variant family (a dict when the family changes between experiments). Only the
# ensembles whose runs are the paper's (tropical-mean SST anomalies correlate at 0.998 or more member by member).
SOURCES = {
    "access_lens": ("ACCESS-ESM1-5", "i1p1f1"), "canesm5_lens": ("CanESM5", "i1p2f1"), "csiro_mk36_lens": ("CSIRO-Mk3-6-0", "i1p1"),
    "e3sm_v1_lens": ("E3SM-1-0", "i2p2f1"), "giss-e2": ("GISS-E2-1-G", "i1p1f2"), "ipsl_cm6a_lens": ("IPSL-CM6A-LR", "i1p1f1"),
    "miroc6_lens": ("MIROC6", "i1p1f1"), "miroc_esm2l_lens": ("MIROC-ES2L", "i1p1f2"), "mpi_lens_cmip6": ("MPI-ESM1-2-LR", "i1p1f1"),
    "mpi_lens": ("MPI-ESM", {"historical": "i1850p3", "rcp85": "i2005p3"}), "ukesm1-0-ll_lens": ("UKESM1-0-LL", "i1p1f2"),
}


def rnum(label):
    return int(re.match(r"r0*(\d+)", str(label)).group(1))


def load(model, family, exp, var, region="n30e"):
    """One variable and region as {member label: pd.Series indexed by year*12 + month - 1}, or {} when absent."""
    fam = family[exp] if isinstance(family, dict) else family
    f = sorted(RAW.glob(f"*_{model}_{fam}_{exp}_*members_tim_{var}_val_{region}.nc"))
    if not f:
        return {}
    ds = xr.open_dataset(f[0], decode_times=xr.coders.CFDatetimeCoder(use_cftime=True))
    da = ds[f"tim_{var}_val_{region}"]
    labels = str(ds.member.attrs.get("members", "")).split()
    key = np.array([t.year * 12 + t.month - 1 for t in da.time.values])
    out = {lab: pd.Series(da.values[i].astype(float), index=key) for i, lab in enumerate(labels)}
    ds.close()
    return out


def member_series(rid, scenario, by_exp):
    """Historical plus the member's own scenario, matched on the label (CMIP6) or the realisation number (CMIP5)."""
    parts = []
    for exp in ("historical", scenario):
        d = by_exp.get(exp, {})
        hit = d.get(rid) if "f" in rid else next((s for lab, s in d.items() if rnum(lab) == rnum(rid)), None)
        if hit is not None:
            parts.append(hit)
    if not parts:
        return None
    s = pd.concat(parts)
    return s[~s.index.duplicated(keep="first")].sort_index()


def has_scenario(rid, scenario, by_exp):
    return any(rnum(lab) == rnum(rid) for lab in (by_exp.get(scenario) or {}))


def month_keys(time):
    t = pd.DatetimeIndex(time)
    return t.year.values * 12 + t.month.values - 1
