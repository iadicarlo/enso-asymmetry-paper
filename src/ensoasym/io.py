"""Data access for the SMILE metrics produced on lorenz.

The metrics files live in ``data/lorenz/<scenario>/<model>_metrics.nc`` with
dimensions (ens, time) for monthly series and (ens, year) for annual mean-state
series.  Members are pooled across scenarios: for each realisation the scenario
with the deepest warm-bin coverage is kept (ssp585 preferred on ties).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "lorenz"
FIG_DIR = ROOT / "figures"

import os

# "noaa": relative index and NOAA's episode rule (the platform method); "gwi_legacy": the stored lorenz classification
CLASSIFICATION = os.environ.get("ENSO_CLASSIFICATION", "noaa").lower()
SCENARIOS = ("ssp585", "ssp370", "rcp85")
SCENARIO_PREF = {"ssp585": 0, "ssp370": 1, "rcp85": 2}

MODELS = [
    "access_lens",
    "canesm2_lens",
    "canesm5_lens",
    "cesm1_lens",
    "cesm2_lens",
    "csiro_mk36_lens",
    "e3sm_v1_lens",
    "e3sm_v2_lens",
    "ec-earth3_lens",
    "gfdl_cm3_lens",
    "gfdl_esm2m_lens",
    "gfdl_spear_lens",
    "giss-e2",
    "ipsl_cm6a_lens",
    "miroc6_lens",
    "miroc_esm2l_lens",
    "mpi_lens",
    "mpi_lens_cmip6",
    "ukesm1-0-ll_lens",
]

PRETTY = {
    "access_lens": "ACCESS-ESM1-5",
    "canesm2_lens": "CanESM2",
    "canesm5_lens": "CanESM5",
    "cesm1_lens": "CESM1",
    "cesm2_lens": "CESM2",
    "csiro_mk36_lens": "CSIRO-Mk3.6",
    "e3sm_v1_lens": "E3SMv1",
    "e3sm_v2_lens": "E3SMv2",
    "ec-earth3_lens": "EC-Earth3",
    "gfdl_cm3_lens": "GFDL-CM3",
    "gfdl_esm2m_lens": "GFDL-ESM2M",
    "gfdl_spear_lens": "GFDL-SPEAR",
    "giss-e2": "GISS-E2.1-G",
    "ipsl_cm6a_lens": "IPSL-CM6A",
    "miroc6_lens": "MIROC6",
    "miroc_esm2l_lens": "MIROC-ES2L",
    "mpi_lens": "MPI-GE",
    "mpi_lens_cmip6": "MPI-GE-CMIP6",
    "ukesm1-0-ll_lens": "UKESM1",
}

# GWI bins (degrees C of global warming above the 1850-1900 baseline)
REF = (0.0, 1.0)
WARM = (3.0, 99.0)
MIN_BIN_MONTHS = 120
MIN_MEMBERS = int(os.environ.get("ENSO_MIN_MEMBERS", 9))  # ensembles with fewer usable members are not used (GISS-E2.1-G has 9, EC-Earth3 8)

PHASE_CODES = {
    "extreme_el_nino": 3,
    "strong_el_nino": 2,
    "moderate_el_nino": 1,
    "neutral": 0,
    "moderate_la_nina": -1,
    "strong_la_nina": -2,
    "extreme_la_nina": -3,
}


@dataclass
class Member:
    model: str
    rid: str
    scenario: str
    nino34: np.ndarray  # standardised Niño-3.4 anomaly, monthly
    nino3: np.ndarray  # standardised Niño-3 anomaly, monthly
    phase: np.ndarray  # int8 phase code, monthly
    gwi: np.ndarray  # smoothed global warming index, monthly
    time: np.ndarray  # datetime64 monthly
    annual: pd.DataFrame  # index year: mean-state series (annual means)


def _annual_frame(ds_metrics: xr.Dataset, ds_nonlin: xr.Dataset | None, k: int) -> pd.DataFrame:
    years = ds_metrics["year"].values.astype(int)
    out = {"year": years}
    for v in ("meanstate_sst_gradient", "meanstate_zos_gradient", "meanstate_z20_tilt"):
        if v in ds_metrics:
            out[v.replace("meanstate_", "")] = np.atleast_2d(ds_metrics[v].values)[k]
    if ds_nonlin is not None:
        yrs2 = ds_nonlin["year"].values.astype(int)
        for v in (
            "meanstate_cold_tongue_sst",
            "meanstate_warm_pool_sst",
            "meanstate_ctex",
            "meanstate_pr_nino3",
            "meanstate_tauu_central_box",
        ):
            if v in ds_nonlin:
                s = pd.Series(np.atleast_2d(ds_nonlin[v].values)[k], index=yrs2)
                out[v.replace("meanstate_", "")] = s.reindex(years).values
    return pd.DataFrame(out).set_index("year")


def load_members(model: str, ref=REF, warm=WARM, min_bin=MIN_BIN_MONTHS) -> list[Member]:
    """Deduplicated member roster for one model, pooled across scenarios."""
    cands: dict[str, list] = {}
    handles = []
    for scen in SCENARIOS:
        p = DATA / scen / f"{model}_metrics.nc"
        if not p.exists():
            continue
        ds = xr.open_dataset(p)
        pn = DATA / scen / f"{model}_nonlinear_indices.nc"
        dn = xr.open_dataset(pn) if pn.exists() else None
        handles += [ds] + ([dn] if dn is not None else [])
        ens = [str(e) for e in ds["ens"].values]
        gwi = np.atleast_2d(ds["gwi"].values)
        nino = np.atleast_2d(ds["nino34_std"].values)
        for k, rid in enumerate(ens):
            g = gwi[k]
            ok = np.isfinite(g)
            n_ref = int((ok & (g >= ref[0]) & (g < ref[1])).sum())
            n_warm = int((ok & (g >= warm[0]) & (g < warm[1])).sum())
            if n_ref < min_bin or n_warm < min_bin:
                continue
            x = nino[k]
            if np.nanstd(x) < 1e-6:
                continue
            cands.setdefault(rid, []).append((n_warm, SCENARIO_PREF[scen], scen, ds, dn, k))
    members = []
    for rid, opts in cands.items():
        opts.sort(key=lambda o: (-o[0], o[1]))
        _, _, scen, ds, dn, k = opts[0]
        nonlin_k = k
        if dn is not None:
            ens_n = [str(e) for e in dn["ens"].values]
            nonlin_k = ens_n.index(rid) if rid in ens_n else None
        annual = (
            _annual_frame(ds, dn if nonlin_k is not None else None, k if nonlin_k is None else nonlin_k)
            if nonlin_k is not None
            else _annual_frame(ds, None, k)
        )
        if CLASSIFICATION == "noaa":
            # relative index and NOAA's rule, computed here from the box anomalies and the global tropical mean
            from .classification import noaa_member_series

            series = noaa_member_series(DATA, model, scen, rid, ds["time"].values)
            if series is None:
                continue
        else:
            series = dict(
                nino34=np.atleast_2d(ds["nino34_std"].values)[k].astype(float),
                nino3=np.atleast_2d(ds["nino3_std"].values)[k].astype(float)
                if "nino3_std" in ds
                else np.full(ds.sizes["time"], np.nan),
                phase=np.atleast_2d(ds["phase_indicator"].values)[k].astype(int),
            )
        members.append(
            Member(
                model=model,
                rid=rid,
                scenario=scen,
                nino34=series["nino34"],
                nino3=series["nino3"],
                phase=series["phase"],
                gwi=np.atleast_2d(ds["gwi"].values)[k].astype(float),
                time=ds["time"].values,
                annual=annual,
            )
        )
    for h in handles:
        h.close()
    members.sort(key=lambda m: m.rid)
    return members


def load_all(models=MODELS) -> dict[str, list[Member]]:
    return {m: load_members(m) for m in models}


def read_table(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / name)
