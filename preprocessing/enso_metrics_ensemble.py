#!/usr/bin/env python3
"""
ENSO Metrics for Multi‑Model Large Ensembles

Processes monthly tropical data from mmleav2_tropics_pipeline (which may have been
subsetted to a specific latitude‑longitude box), computes ENSO event statistics,
variance, skewness, and mean‑state indices, and saves per‑model results.

Usage:
    # For equatorial Pacific subset (e.g., 6°S–6°N, 120°E–80°W):
    python enso_metrics_ensemble.py --lat-range -6 6 --lon-range 120 280 --sst-var tos --n-jobs 4

    # For full tropics (30°S–30°N, global):
    python enso_metrics_ensemble.py --lat-range -30 30 --lon-range 0 360 --sst-var tos --n-jobs 4
"""

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import xarray as xr
import joblib
from tqdm import tqdm

# Shared config (paths, model list, extract_realization, extract_scenario)
_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))
from config import (  # noqa: E402
    TROPICS_DIR as TROPICS_DATA_DIR,
    GWI_CACHE_DIR,
    RESULTS_DIR as OUTPUT_DIR,
    LAT_MIN as _LAT_MIN_DEFAULT,
    LAT_MAX as _LAT_MAX_DEFAULT,
    LON_MIN as _LON_MIN_DEFAULT,
    LON_MAX as _LON_MAX_DEFAULT,
    TIME_START,
    TIME_END,
    extract_realization as _extract_realization_from_config,
    extract_scenario as _extract_scenario_from_config,
)

# Ensure functions.py is found
analysis_dir = Path(__file__).resolve().parents[1]
if str(analysis_dir) not in sys.path:
    sys.path.insert(0, str(analysis_dir))
import functions as fns
import time_fixes  # MMLEAv2 E3SMv1 time axis one month early (2026-09-28)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BASELINE = (1980, 2014)           # fixed baseline for standardisation

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def get_model_files(model: str, var: str,
                    lat_min: float, lat_max: float,
                    lon_min: float, lon_max: float) -> List[Path]:
    """Return list of local NetCDF files for a given model, variable and subset box.

    Excludes piControl / 1pctCO2 / abrupt-4xCO2 / lastmillennium files so
    forced-run analyses do not accidentally ingest them.  piControl is
    especially problematic because its time axis is often "hours since
    0101-01-16" (year 101), which forces xarray into cftime mode and
    cannot be compared with the datetime64 axes of hist / ssp runs in
    _merge_var_files.
    """
    var_dir = TROPICS_DATA_DIR / model / var
    if not var_dir.exists():
        return []
    pattern = f"*_lat{lat_min}_{lat_max}_lon{lon_min}_{lon_max}.nc"
    excluded_tokens = ("piControl", "1pctCO2", "abrupt-4xCO2",
                        "lastmillennium", "pi-control", "picontrol",
                        # DAMIP single-forcing experiments (hist-GHG, hist-aer,
                        # hist-nat, hist-sol, hist-volc, hist-stratO3) — only
                        # MIROC6 ships them in the MMLEAv2 tree; without this
                        # filter _dedup_files picks them over the proper
                        # historical run because they span 1850-2020.
                        "hist-ghg", "hist-aer", "hist-nat",
                        "hist-sol", "hist-volc", "hist-strato3",
                        "hist-totalo3", "hist-co2", "hist-bmb",
                        "hist-1950hc", "amip")
    return sorted(
        f for f in var_dir.glob(pattern)
        if not any(tok.lower() in f.name.lower() for tok in excluded_tokens)
    )


# Realization ID and scenario extraction — imported from config.py (single source of truth)
extract_realization = _extract_realization_from_config
extract_scenario    = _extract_scenario_from_config


def get_realizations_by_scenario(model: str, sst_var: str,
                                  lat_min: float, lat_max: float,
                                  lon_min: float, lon_max: float
                                  ) -> Dict[str, List[str]]:
    """
    Return {scenario: [realization_id, ...]} for a model.

    Files that are historical-only are grouped under 'historical'.
    Files that contain a future scenario (ssp370, ssp585, rcp85, …) are grouped
    under that scenario label.  Each (scenario, realization) pair will be
    processed as one independent run.
    """
    files = get_model_files(model, sst_var, lat_min, lat_max, lon_min, lon_max)
    # Fallback: if no tos files found, try ts (e.g. e3sm_v2_lens)
    if not files and sst_var != "ts":
        files = get_model_files(model, "ts", lat_min, lat_max, lon_min, lon_max)
    by_scenario: Dict[str, set] = {}
    for f in files:
        real = extract_realization(f.name)
        scen = extract_scenario(f.name)
        by_scenario.setdefault(scen, set()).add(real)
    return {s: sorted(r) for s, r in by_scenario.items()}


def load_gwi(model: str, realization: str) -> xr.DataArray:
    """Load pre-computed smoothed GWI for a realization. Raises FileNotFoundError if missing."""
    gwi_file = GWI_CACHE_DIR / model / realization / "gwi.nc"
    if not gwi_file.exists():
        raise FileNotFoundError(f"GWI file missing: {gwi_file}")
    ds = xr.open_dataset(gwi_file)
    gwi = ds["gwi_smoothed"]
    return gwi


def _merge_var_files(ds_list: list) -> xr.Dataset:
    """
    Merge a list of xr.Datasets (one per file) for the same variable/member.

    Sorts by time, concatenates, then drops duplicate time steps that appear
    at the hist/future boundary (e.g. 2014-12 appearing in both files).

    combine_attrs='drop' is used at the xr.concat level to avoid attribute
    conflicts, but we explicitly preserve each data variable's 'units' and
    'standard_name' attrs from the FIRST (earliest-time) dataset so
    downstream sanity checks (e.g. cm→m detection for CESM2 ZOS) can still
    see the original CF metadata.
    """
    if len(ds_list) == 1:
        return ds_list[0]
    ds_list_sorted = sorted(ds_list, key=lambda d: d.time.values[0])
    # Preserve units / standard_name from first dataset
    preserved = {}
    for var in ds_list_sorted[0].data_vars:
        a = {}
        for k in ("units", "standard_name", "long_name"):
            if k in ds_list_sorted[0][var].attrs:
                a[k] = ds_list_sorted[0][var].attrs[k]
        if a:
            preserved[var] = a
    combined = xr.concat(ds_list_sorted, dim='time', combine_attrs='drop')
    combined = combined.sortby('time')
    # Drop duplicate time steps. Using xr.drop_duplicates preserves the
    # DatetimeIndex/CFTimeIndex backing — the previous np.unique + isel path
    # stripped it and broke downstream .resample() calls.
    combined = combined.drop_duplicates('time')
    # Re-attach preserved attrs
    for var, attrs in preserved.items():
        if var in combined.data_vars:
            combined[var].attrs.update(attrs)
    return combined

def compute_nino_indices(ts: xr.DataArray) -> Dict[str, xr.DataArray]:
    """Compute Nino3.4, Nino3 SST, Nino4 SST from detrended SST."""
    nino34 = ts.sel(lat=slice(-5,5), lon=slice(190,240)).mean(["lat","lon"])
    nino3_sst = ts.sel(lat=slice(-5,5), lon=slice(210,270)).mean(["lat","lon"])
    nino4_sst = ts.sel(lat=slice(-5,5), lon=slice(160,210)).mean(["lat","lon"])
    return {"nino34": nino34, "nino3_sst": nino3_sst, "nino4_sst": nino4_sst}


def compute_nino34_rsst(sst_raw: xr.DataArray) -> xr.DataArray:
    """Niño-3.4 RSST per Izumo 2020 / van Oldenborgh 2021 / Izumo 2024 §2b.

    RSST = SST(Niño-3.4 box: 5°S-5°N, 170°W-120°W)
           − SST(tropical mean: 20°S-20°N, all available longitudes).

    Uses RAW (not GWI-detrended) SST so the tropical-mean subtraction
    does the detrending by construction — matching the paper's method.
    Requires the input tropics box to cover at least ±20° in latitude;
    if the data is narrower, returns a best-effort value using the
    available latitudes, which will be less effective at removing the
    forced warming but is still a valid RSST-style quantity.

    Returns a monthly DataArray on sst_raw.time.
    """
    # Use full lat range present (data should be ±20° per current pipeline);
    # if only ±6° is available, that is still a tropical average.
    lat_min_avail = float(sst_raw.lat.min())
    lat_max_avail = float(sst_raw.lat.max())
    trop_lat_lo = max(-20.0, lat_min_avail)
    trop_lat_hi = min( 20.0, lat_max_avail)
    trop_mean = sst_raw.sel(
        lat=slice(trop_lat_lo, trop_lat_hi)
    ).mean(["lat", "lon"])
    nino34_sst = sst_raw.sel(
        lat=slice(-5, 5), lon=slice(190, 240)
    ).mean(["lat", "lon"])
    rsst = nino34_sst - trop_mean
    rsst.name = "nino34_rsst"
    rsst.attrs["description"] = (
        "Niño-3.4 Relative SST = SST(5°S-5°N,170°W-120°W) − "
        "SST(%.1f°-%.1f° lat, all lon)" % (trop_lat_lo, trop_lat_hi)
    )
    rsst.attrs["lat_band_used"] = f"{trop_lat_lo}..{trop_lat_hi}"
    return rsst

def compute_mean_state(
    ts: xr.DataArray,
    tauu: Optional[xr.DataArray],
    z20: Optional[xr.DataArray] = None,
    zos: Optional[xr.DataArray] = None,
) -> Dict[str, xr.DataArray]:
    """Compute equatorial-Pacific mean-state indices.

    Six metrics (all at 5°S–5°N), all computed on RAW monthly fields
    (no seasonal-cycle removal — these are climatology-level quantities):

    1. sst_gradient      — SST(W 110-180°E) − SST(E 180-280°E)
                           Watanabe et al. 2021 (Nat Clim Change).
    2. z20_tilt          — Z20(W 120-180°E) − Z20(E 220-280°E)   [Cai 2018 boxes]
                           Thermocline slope; Cai et al. 2018 (Nature).
    3. z20_mean          — spatial mean Z20 over 110-280°E
                           Bulk thermocline depth; Yeh et al. 2018, Cai et al. 2021.
    4. zos_gradient      — ZOS(W 110-180°E) − ZOS(E 180-280°E)
                           Heede et al. 2021, Seager et al. 2019.
    5. tauu_central      — τx mean over 170°E-230°E               [Cai 2018 box]
                           Walker-cell strength; Cai et al. 2018 (Nature).
    6. tauu_gradient     — τx(W 110-180°E) − τx(E 180-280°E)
                           Heede et al. 2021, Seager et al. 2022.

    Data availability:
      Z20 unavailable on MMLEAv2 for: e3sm_v2, gfdl_esm2m, gfdl_spear, miroc_esm2l.
      ZOS unavailable on MMLEAv2 for: canesm2, csiro_mk36, gfdl_cm3, gfdl_spear,
        giss-e2, miroc_esm2l.
      τx unavailable on MMLEAv2 for: access, gfdl_esm2m, gfdl_spear.
    """
    results: Dict[str, xr.DataArray] = {}

    # 1. SST zonal gradient (Watanabe 2021 boxes)
    sst_w = ts.sel(lat=slice(-5, 5), lon=slice(110, 180)).mean(["lat", "lon"])
    sst_e = ts.sel(lat=slice(-5, 5), lon=slice(180, 280)).mean(["lat", "lon"])
    sst_grad = sst_w - sst_e
    sst_grad.name = "sst_gradient"
    results["sst_gradient"] = sst_grad

    # 2-3. Z20 metrics (Cai 2018 boxes for the tilt: W 120-180, E 220-280)
    if z20 is not None:
        z20_w    = z20.sel(lat=slice(-5, 5), lon=slice(120, 180)).mean(["lat", "lon"])
        z20_e    = z20.sel(lat=slice(-5, 5), lon=slice(220, 280)).mean(["lat", "lon"])
        z20_tilt = z20_w - z20_e
        z20_tilt.name = "z20_tilt"
        results["z20_tilt"] = z20_tilt

        z20_mean = z20.sel(lat=slice(-5, 5), lon=slice(110, 280)).mean(["lat", "lon"])
        z20_mean.name = "z20_mean"
        results["z20_mean"] = z20_mean

        # 3b. h* — thermocline sharpness proxy (Iwakiri & Watanabe 2019,
        # JGR-Oceans; Cai et al. 2018 Fig 2b; DiNezio & Deser 2014).
        # Iwakiri's full h* (thickness of the 16-22°C isothermal layer) needs
        # 3D θ which MMLEAv2 does not provide. The accepted 2D proxy is the
        # negative of the eastern-Pacific Z20 depth over the Niño3 box: a
        # shallower eastern thermocline compresses the vertical T gradient
        # in the surface mixed layer and sharpens the T_sub response to
        # thermocline vertical displacements, which scales linearly with
        # σ(Niño3.4) and skew(Niño3.4) in the Timmermann 2003 low-order
        # nonlinear ENSO model (see Iwakiri 2019 Fig 7).
        z20_nino3 = z20.sel(lat=slice(-5, 5),
                            lon=slice(210, 270)).mean(["lat", "lon"])
        h_star = -z20_nino3
        h_star.name = "h_star"
        results["h_star"] = h_star

    # 4. ZOS zonal gradient (Heede 2021, same boxes as SST gradient)
    if zos is not None:
        zos_w    = zos.sel(lat=slice(-5, 5), lon=slice(110, 180)).mean(["lat", "lon"])
        zos_e    = zos.sel(lat=slice(-5, 5), lon=slice(180, 280)).mean(["lat", "lon"])
        zos_grad = zos_w - zos_e
        zos_grad.name = "zos_gradient"
        results["zos_gradient"] = zos_grad

    # 4b. Izumo recharge index h_eq+sw (Izumo & Colin 2022; Izumo et al. 2024 JCLI).
    # Slow recharge mode of the Recharge Delayed Oscillator (RDO),
    # approximately independent of the fast tilt mode once the southwest
    # Pacific box is included.  Two boxes (asymmetric N–S because the
    # southern STC convergence is ~2× the northern):
    #   eq : 5°S–5°N, 120°E–80°W   (equatorial band)
    #   sw : 15°–5°S, 120°E–170°W  (southwest Pacific, SPCZ region)
    # Uses ZOS if available (direct SLA analogue; Izumo's preferred form)
    # and separately Z20 (thermocline-depth analogue per Izumo's own
    # comparisons in their 2022 GRL).  h_eq_sw ≈ h_ind_eq_sw (regression
    # coefficient on T_E small ~0.30 when sw box is included; full T_E
    # residualisation is applied downstream in fit_izumo_rdo.py).
    #
    # Requires ≥±15° latitude coverage on the input archive.  See
    # config.LAT_MIN / LAT_MAX; if the archive is cropped tighter the
    # southwest box returns NaN and the RDO fit downstream will refuse to
    # run (fail-loud NaN audit in apply_rdo_fit.py).
    for var_name, var_field in (("h_eq_sw_zos", zos),
                                  ("h_eq_sw_z20", z20)):
        if var_field is None:
            continue
        h_eq = var_field.sel(lat=slice(-5, 5),
                              lon=slice(120, 280)).mean(["lat", "lon"])
        h_sw = var_field.sel(lat=slice(-15, -5),
                              lon=slice(120, 190)).mean(["lat", "lon"])
        # Equal-area weighted mean of the two boxes (Izumo 2022 Fig 2h).
        h_eq_sw = (h_eq + h_sw) / 2.0
        h_eq_sw.name = var_name
        results[var_name] = h_eq_sw

    # 5-6. τx metrics (Cai 2018 box for central: 170-230°E)
    if tauu is not None:
        tauu_central = tauu.sel(lat=slice(-5, 5), lon=slice(170, 230)).mean(["lat", "lon"])
        tauu_central.name = "tauu_central"
        results["tauu_central"] = tauu_central

        tauu_w    = tauu.sel(lat=slice(-5, 5), lon=slice(110, 180)).mean(["lat", "lon"])
        tauu_e    = tauu.sel(lat=slice(-5, 5), lon=slice(180, 280)).mean(["lat", "lon"])
        tauu_grad = tauu_w - tauu_e
        tauu_grad.name = "tauu_gradient"
        results["tauu_gradient"] = tauu_grad

    return results


def compute_bjerknes_feedback_rolling(
    tauu: xr.DataArray,
    z20: xr.DataArray,
    window_years: int = 30,
) -> Optional[xr.DataArray]:
    """Deseasonalised monthly τ_x and H_tilt anomalies for β_H computation.

    Returns the atomic inputs needed for the Bjerknes feedback regression —
    not β_H itself.  β_H is now computed downstream on per-GWI-bin pooled
    samples (see viz layer `_compute_beta_h_pooled`), avoiding the noise
    of short 30-year rolling-window regressions.

    Following Cai et al. (2018):
      τ_x box:  5°S–5°N, 170°E–230°E  (170°E–130°W)
      H_W box:  5°S–5°N, 120°E–180°
      H_E box:  5°S–5°N, 220°E–280°E  (140°W–80°W)
      H_tilt    = H_E − H_W   (Cai 2018 convention; positive under
                                normal trade-wind regime when east
                                thermocline is deeper than west NOT —
                                with trades, H_E < H_W so H_tilt < 0).
      β_H = cov(H_tilt, τ_x) / var(τ_x)
      Sign: τ_x eastward-positive; trade winds (easterly, negative τ_x)
      give H_E < H_W so H_tilt is negative.  A strengthening of trade
      winds deepens west / shoals east → Δτ_x < 0 → ΔH_tilt < 0 →
      cov > 0 → **β_H > 0** for physical models.

    Deseasonalisation uses the same rolling-climatology +
    baseline-period-std convention as Niño3.4
    (see functions.remove_seasonal_cycle_rolling_with_std with
    window_years=30, rolling_std=False, baseline_period=(1980,2014)),
    so τ_x, H_tilt and Niño3.4 are all on the same anomaly timeline.

    Returns {"tauu_cai_anom": DataArray(time), "h_tilt_cai_anom": DataArray(time)}
    aligned on common monthly time axis.  Returns None if inputs missing.
    Argument ``window_years`` is kept for backward compatibility but is
    only used as the rolling-climatology window in the deseasonalisation.
    """
    if tauu is None or z20 is None:
        return None

    # Box averages (Cai 2018 exact boxes)
    tx = tauu.sel(lat=slice(-5, 5), lon=slice(170, 230)).mean(["lat", "lon"])
    hw = z20.sel(lat=slice(-5, 5), lon=slice(120, 180)).mean(["lat", "lon"])
    he = z20.sel(lat=slice(-5, 5), lon=slice(220, 280)).mean(["lat", "lon"])
    # Cai 2018 convention: H_tilt = H_E − H_W
    h_tilt = he - hw

    # Align τ_x and H_tilt on common monthly time axis
    if "time" not in tx.dims or "time" not in h_tilt.dims:
        return None
    try:
        y_t = np.array([(t.year * 100 + t.month) for t in tx["time"].values])
        y_z = np.array([(t.year * 100 + t.month) for t in h_tilt["time"].values])
    except AttributeError:
        ttv = tx["time"].values
        hhv = h_tilt["time"].values
        y_t = (ttv.astype("datetime64[Y]").astype(int) + 1970) * 100 + \
              (ttv.astype("datetime64[M]").astype(int) % 12 + 1)
        y_z = (hhv.astype("datetime64[Y]").astype(int) + 1970) * 100 + \
              (hhv.astype("datetime64[M]").astype(int) % 12 + 1)
    common = np.intersect1d(y_t, y_z)
    if common.size < 12 * 30:
        return None
    i_t = np.where(np.isin(y_t, common))[0]
    i_z = np.where(np.isin(y_z, common))[0]
    tx_al = tx.isel(time=i_t)
    h_al  = h_tilt.isel(time=i_z)
    # Snap both onto a single time coordinate (τ_x's) so downstream
    # reindexing works.
    h_al  = h_al.assign_coords(time=tx_al["time"].values)

    # Deseasonalise with the same pipeline as Niño3.4: rolling 30-yr
    # climatology (``remove_seasonal_cycle_rolling_with_std``) with
    # ``rolling_std=False`` — we read the ``anomaly`` field (raw −
    # rolling-climatology), NOT the ``standardized`` field.  We do NOT
    # divide by std because β_H needs dimensional anomalies (τ_x in Pa,
    # H_tilt in metres) to return the correct regression units.
    tx_res = fns.remove_seasonal_cycle_rolling_with_std(
        tx_al, window_years=window_years, rolling_std=False,
        baseline_period=(1980, 2014))
    h_res = fns.remove_seasonal_cycle_rolling_with_std(
        h_al, window_years=window_years, rolling_std=False,
        baseline_period=(1980, 2014))
    return {
        "tauu_cai_anom":   tx_res["anomaly"],
        "h_tilt_cai_anom": h_res["anomaly"],
    }


def _files_for_member_scenario(all_files: List[Path], realization: str,
                               scenario: str) -> List[Path]:
    """
    Filter file list to those belonging to a given realization and scenario.

    For a combined hist+future file (scenario != 'historical'), we want:
      - any file whose name contains the realization AND contains the scenario token
      - plus any historical file for that realization (the hist segment)

    For a historical-only run we want:
      - only files whose name does NOT contain any ssp/rcp token
        (or that explicitly contain 'historical' but not a future label)
    """
    future_tokens = re.compile(r'ssp\d+|rcp\d+', re.IGNORECASE)

    result = []
    for f in all_files:
        if realization not in f.name:
            continue
        file_scen = extract_scenario(f.name)
        if scenario == "historical":
            # Accept only files with no future token
            if not future_tokens.search(f.name):
                result.append(f)
        else:
            # Accept historical file for this member (hist segment)
            # AND the specific future scenario file
            if file_scen == "historical" or file_scen == scenario:
                result.append(f)
    return sorted(result)


def _has_hist_continuity(files: List[Path], scenario: str) -> bool:
    """
    True if the file list contains coverage of the historical segment for the
    requested (future) scenario.

    A historical segment is considered present if either
      (a) at least one file in the list has extract_scenario == 'historical'
          (separate hist file), or
      (b) at least one file has a filename indicating a concatenated
          historical+future file (e.g. 'historical_ssp370', 'histssp585').

    For scenario == 'historical', any file qualifies.

    Rationale: the MMLEAv2 pipeline deseasonalises, detects events, and
    anchors the baseline period (1980-2014) on the hist segment. A
    future-only file for a given (model, var, realization) cannot be used
    — it produces an anomaly computed against a 30-yr rolling climatology
    that slides entirely inside the warming signal, which biases Niño3.4
    thresholds, event counts, and β_H.
    """
    if not files:
        return False
    if scenario == "historical":
        return True
    concat_re = re.compile(r'(historical[_-]?ssp|historical[_-]?rcp|histssp|histrcp)',
                           re.IGNORECASE)
    for f in files:
        if extract_scenario(f.name) == "historical":
            return True
        if concat_re.search(f.name):
            return True
    return False


_SIBLING_TIME_DONORS = ("tauu", "pr", "tos", "ts", "psl", "tas")


def _borrow_time_from_sibling(broken_file: Path, target_len: int) -> Optional[np.ndarray]:
    """Return a valid time array borrowed from a sibling variable's file for the
    same model and realization.

    Used as a rescue path when a file's own time coord is corrupted (e.g.
    e3sm_v2 zos on MMLEAv2 has the historical portion zeroed out in the source
    archive). We look in the sibling-variable folders of the same model for a
    file whose filename matches on everything except the variable name and the
    frequency table (Amon ↔ Omon), load its time axis, and return it if the
    length matches. Returns None if no suitable donor is found.
    """
    # broken_file: .../<model>/<var>/<prefix>_<var>_<source>_<experiments>_<realization>_<grid>_<times>_lat...nc
    # We'll glob for candidates by replacing the variable directory.
    model_dir = broken_file.parent.parent     # .../<model>/
    realization_match = re.search(r"_(r\d+i\d+p\d+f\d+)_", broken_file.name)
    if not realization_match:
        return None
    realization = realization_match.group(1)

    for donor_var in _SIBLING_TIME_DONORS:
        donor_dir = model_dir / donor_var
        if not donor_dir.is_dir():
            continue
        for donor_file in sorted(donor_dir.glob(f"*_{realization}_*.nc")):
            try:
                donor_ds = xr.open_dataset(donor_file)
                if ("time" in donor_ds.coords
                        and not np.issubdtype(donor_ds["time"].dtype, np.floating)
                        and len(donor_ds["time"]) == target_len):
                    t = donor_ds["time"].values.copy()
                    donor_ds.close()
                    log.info(f"    borrowed time axis from {donor_var}/{donor_file.name}")
                    return t
                donor_ds.close()
            except Exception:
                continue
    return None


def _load_and_merge_var(files: List[Path]) -> Optional[xr.Dataset]:
    """Load one xr.Dataset per file, merge with sort+dedup.

    If a file's time coord is raw numeric (e.g. E3SMv2 zos on MMLEAv2 uses
    "day as %Y%m%d.%f" with historical values zeroed out upstream), try to
    borrow a valid time axis from a sibling variable (tauu/pr/tos/…) of the
    same model + realization. Only bails out if no sibling donor is found.
    """
    ds_list = []
    for f in files:
        ds = xr.open_dataset(f)
        # Detect corrupt/undecoded numeric time and try to repair it.
        if "time" in ds.coords and np.issubdtype(ds["time"].dtype, np.floating):
            donor_time = _borrow_time_from_sibling(f, len(ds["time"]))
            if donor_time is None:
                log.warning(f"  skipping {f.name}: non-decodable time axis "
                            f"(units='{ds['time'].attrs.get('units','?')}'), "
                            f"no sibling donor found")
                return None
            ds = ds.assign_coords(time=("time", donor_time))
        ds = time_fixes.fix_month_offset(ds, f)
        # Drop extra non-(time,lat,lon) dimensions (e.g. st_ocean depth in gfdl_esm2m)
        for d in list(ds.dims):
            if d not in ("time", "lat", "lon"):
                ds = ds.isel({d: 0}, drop=True)
        ds_list.append(ds)
    if not ds_list:
        return None
    return _merge_var_files(ds_list)


def process_member(model: str, realization: str, scenario: str,
                   sst_var: str, pr_var: str, tauu_var: str,
                   lat_min: float, lat_max: float,
                   lon_min: float, lon_max: float) -> Optional[xr.Dataset]:
    """
    Process one (realization, scenario) pair: merge hist+future files for SST,
    PR, tauu, compute ENSO indices, events, and rolling metrics.
    """
    log.info(f"Processing {model} {realization} [{scenario}]")

    all_sst   = get_model_files(model, sst_var,  lat_min, lat_max, lon_min, lon_max)
    all_pr    = get_model_files(model, pr_var,   lat_min, lat_max, lon_min, lon_max)
    all_tauu  = get_model_files(model, tauu_var, lat_min, lat_max, lon_min, lon_max)

    sst_files  = _files_for_member_scenario(all_sst,  realization, scenario)
    pr_files   = _files_for_member_scenario(all_pr,   realization, scenario)
    tauu_files = _files_for_member_scenario(all_tauu, realization, scenario)

    log.info(f"  SST files  ({len(sst_files)}): {[f.name for f in sst_files]}")
    log.info(f"  PR  files  ({len(pr_files)}): {[f.name for f in pr_files]}")
    log.info(f"  Tauu files ({len(tauu_files)}): {[f.name for f in tauu_files]}")

    if not sst_files and sst_var != "ts":
        # Fallback: try surface temperature (ts) for models that lack tos (e.g. e3sm_v2_lens)
        all_sst = get_model_files(model, "ts", lat_min, lat_max, lon_min, lon_max)
        sst_files = _files_for_member_scenario(all_sst, realization, scenario)
        if sst_files:
            log.info(f"  Using 'ts' as SST fallback for {model} {realization}")
    if not sst_files:
        log.warning(f"No {sst_var} files for {model} {realization} [{scenario}]")
        return None
    if not pr_files:
        log.warning(f"No {pr_var} files for {model} {realization} [{scenario}] – skipping")
        return None

    # ── Enforce historical-segment continuity for MANDATORY variables ────────
    # A future-only member (SSP/RCP file with no matching hist file of same
    # realization id) biases Niño3.4 thresholds, event counts, and baseline
    # β_H. SST and PR anchor the entire ENSO-detection pipeline, so skip the
    # member if either is future-only.
    if scenario != "historical":
        if not _has_hist_continuity(sst_files, scenario):
            log.warning(f"  {model} {realization} [{scenario}]: "
                        f"SST has no historical segment – skipping member")
            return None
        if not _has_hist_continuity(pr_files, scenario):
            log.warning(f"  {model} {realization} [{scenario}]: "
                        f"PR has no historical segment – skipping member")
            return None

    # ========== Load and merge SST ==========
    sst_ds = _load_and_merge_var(sst_files)
    if sst_ds is None or len(sst_ds.time) == 0:
        log.warning(f"  Empty SST dataset for {model} {realization} [{scenario}] – skipping")
        return None
    log.info(f"  SST time range: {sst_ds.time.values[0]} to {sst_ds.time.values[-1]}")

    sst = None
    for name in [sst_var, 'tos', 'ts', 'sst', 'thetao']:
        if name in sst_ds:
            sst = sst_ds[name]
            break
    if sst is None:
        log.warning(f"No SST variable found – skipping")
        return None
    if hasattr(sst, "units") and sst.units == "K":
        sst = sst - 273.15

    # ========== Load and merge precipitation ==========
    pr_ds = _load_and_merge_var(pr_files)
    if pr_ds is None or len(pr_ds.time) == 0:
        log.warning(f"  Empty PR dataset for {model} {realization} [{scenario}] – skipping")
        return None
    log.info(f"  PR  time range: {pr_ds.time.values[0]} to {pr_ds.time.values[-1]}")

    precc = pr_ds.get("PRECC", None)
    precl = pr_ds.get("PRECL", None)
    if precc is not None and precl is not None:
        pr = (precc + precl) * 86400 * 1000
    else:
        pr = pr_ds.get("pr", None)
        if pr is None:
            log.warning(f"No precipitation variable found – skipping")
            return None
        if hasattr(pr, "units") and pr.units in ["kg m-2 s-1", "kg m**-2 s**-1"]:
            pr = pr * 86400 * 1000

    # ========== Load and merge tauu (optional) ==========
    # Per-variable hist continuity: future-only tauu/z20/zos would produce
    # deseasonalised anomalies anchored on 2015+ climatology only — unusable
    # for β_H / mean-state comparisons. Drop silently and metrics downstream
    # treat the variable as absent for this member.
    tauu = None
    if tauu_files and _has_hist_continuity(tauu_files, scenario):
        tauu_ds = _load_and_merge_var(tauu_files)
        if tauu_ds is not None:
            tauu = tauu_ds.get("tauu", None)
            if tauu is not None:
                log.info(f"  Tauu time range: {tauu_ds.time.values[0]} to {tauu_ds.time.values[-1]}")
                # Auto-detect sign convention on the HISTORICAL period only
                # (1850-2014).  Historical climatological zonal wind stress
                # in the central tropical Pacific (170°-230°E, 5°S-5°N) is
                # unambiguously easterly (negative) by physics.  Using the
                # full record would be theoretically polluted by any
                # warming-induced Walker slowdown, so we restrict to the
                # historical segment where easterlies unambiguously dominate.
                # A positive historical climatology means the archive stored
                # τ_x with flipped sign (observed for CESM1-LE, CESM2-LE,
                # GFDL-CM3, E3SMv2 in MMLEAv2 despite some files declaring
                # standard_name "surface_downward_eastward_stress").
                try:
                    _hist = tauu.sel(time=slice("1850-01-01", "2014-12-31"))
                except Exception:
                    _hist = tauu
                _cent_mean = float(
                    _hist.sel(lat=slice(-5, 5),
                               lon=slice(170, 230)).mean().values
                )
                if np.isfinite(_cent_mean) and _cent_mean > 0:
                    log.warning(
                        f"  {model} {realization}: τ_x historical (1850-2014) "
                        f"central-Pacific mean = +{_cent_mean:.4f} N/m² "
                        "(positive — non-physical easterlies). Flipping sign "
                        "to restore CMIP convention."
                    )
                    tauu = -tauu
                    tauu.attrs["sign_flip_applied"] = (
                        "auto-corrected: archive stored with reversed "
                        "convention (original 1850-2014 central-Pacific "
                        f"climatological mean was +{_cent_mean:.4f} N/m²)"
                    )
    elif tauu_files:
        log.warning(f"  {model} {realization} [{scenario}]: tauu has no "
                    f"historical segment – τ_x / β_H will be absent")

    # ========== Load and merge z20 (optional — thermocline depth) ==========
    z20 = None
    all_z20 = get_model_files(model, "z20", lat_min, lat_max, lon_min, lon_max)
    z20_files = _files_for_member_scenario(all_z20, realization, scenario)
    if z20_files and _has_hist_continuity(z20_files, scenario):
        z20_ds = _load_and_merge_var(z20_files)
        if z20_ds is not None:
            z20 = z20_ds.get("z20", None)
            if z20 is not None:
                log.info(f"  Z20 time range: {z20_ds.time.values[0]} to {z20_ds.time.values[-1]}")
    elif z20_files:
        log.warning(f"  {model} {realization} [{scenario}]: z20 has no "
                    f"historical segment – H_tilt / β_H will be absent")
    else:
        log.debug(f"  No z20 files for {model} {realization} — z20 metrics will be absent")

    # ========== Load and merge zos (optional — sea-surface height) ==========
    zos = None
    all_zos = get_model_files(model, "zos", lat_min, lat_max, lon_min, lon_max)
    zos_files = _files_for_member_scenario(all_zos, realization, scenario)
    if zos_files and _has_hist_continuity(zos_files, scenario):
        zos_ds = _load_and_merge_var(zos_files)
        if zos_ds is not None:
            zos = zos_ds.get("zos", None)
            if zos is not None:
                log.info(f"  ZOS time range: {zos_ds.time.values[0]} to {zos_ds.time.values[-1]}")
                # Unit normalisation: most CMIP models store ZOS in metres,
                # but CESM2-LE stores it in CENTIMETRES (the file attribute
                # says units='centimeter') — giving absolute values ~56 cm
                # and a W-E gradient of ~32 cm, which looks like "32 m" and
                # pollutes the ensemble comparison.  Detect via the units
                # attribute and convert to metres for cross-model consistency.
                _units = str(zos.attrs.get("units", "m")).strip().lower()
                if _units in ("cm", "centimeter", "centimeters"):
                    log.info(f"  ZOS units = {_units!r} → converting to metres")
                    zos = zos * 0.01
                    zos.attrs["units"] = "m"
                    zos.attrs["unit_conversion_applied"] = (
                        f"cm → m (factor 0.01); original units attribute: {_units}"
                    )
                elif _units in ("mm", "millimeter", "millimeters"):
                    log.info(f"  ZOS units = {_units!r} → converting to metres")
                    zos = zos * 0.001
                    zos.attrs["units"] = "m"
                    zos.attrs["unit_conversion_applied"] = (
                        f"mm → m (factor 0.001); original units attribute: {_units}"
                    )
                elif _units not in ("m", "metre", "metres", "meter", "meters",
                                       "m/s", ""):
                    log.warning(
                        f"  {model} {realization}: unrecognised ZOS units "
                        f"{_units!r} — leaving unchanged, cross-model "
                        "comparisons may be invalid."
                    )
    elif zos_files:
        log.warning(f"  {model} {realization} [{scenario}]: zos has no "
                    f"historical segment – zos_gradient will be absent")
    else:
        log.debug(f"  No zos files for {model} {realization} — zos_gradient will be absent")

    # ========== Load GWI — hard failure if missing ==========
    try:
        gwi_smooth = load_gwi(model, realization)
    except FileNotFoundError:
        log.warning(f"GWI missing for {model} {realization} – skipping (run preprocess_gwi_ensemble.py first)")
        return None

    # Align GWI to SST time axis by year+month integer key.
    # Avoids calendar-type comparison errors (cftime vs pd.Timestamp)
    # that arise when using reindex with tolerance=pd.Timedelta across mixed calendars.
    sst_ym = sst.time.dt.year.values * 100 + sst.time.dt.month.values
    gwi_ym = gwi_smooth.time.dt.year.values * 100 + gwi_smooth.time.dt.month.values
    ym_to_idx = {ym: i for i, ym in enumerate(gwi_ym)}
    indices = [ym_to_idx.get(ym, None) for ym in sst_ym]
    n_nan = sum(i is None for i in indices)
    if n_nan > 0:
        # Some SST months lie outside the GWI window (1850-01 to 2100-12).
        # This is a benign calendar-boundary issue: e.g. E3SM-1-0 files are
        # tagged 1849-12-16 for their first timestamp even though the span
        # is 1850-01-..-2100-12 per the filename.  Drop the out-of-window
        # SST months (typically 1 boundary month) instead of failing the
        # whole member.  If the fraction is substantial (>2%), skip as
        # something is genuinely wrong.
        drop_frac = n_nan / len(sst_ym)
        if drop_frac > 0.02:
            log.warning(f"  GWI alignment: {n_nan}/{len(sst_ym)} ({drop_frac:.1%}) "
                        f"SST months not found in GWI for {model} {realization} "
                        f"— skipping (fraction > 2%)")
            return None
        log.info(f"  GWI alignment: dropping {n_nan}/{len(sst_ym)} boundary "
                  f"month(s) for {model} {realization} (outside GWI window)")
        keep_mask = np.array([i is not None for i in indices])
        sst = sst.isel(time=keep_mask)
        sst_ym = sst.time.dt.year.values * 100 + sst.time.dt.month.values
        indices = [ym_to_idx[ym] for ym in sst_ym]
    gwi_aligned = xr.DataArray(
        gwi_smooth.values[indices],
        coords={"time": sst.time},
        dims=["time"],
    )

    # Detrend SST
    sst_detrended = fns.remove_gwi_residuals_vectorized(sst, gwi_aligned)

    # Detrend precipitation – must succeed
    if not pr.time.equals(sst.time):
        log.info(f"  Aligning precipitation time axis to SST")
        pr = pr.reindex(time=sst.time, method="nearest")
    try:
        pr_detrended = fns.remove_gwi_residuals_vectorized(pr, gwi_aligned)
    except Exception as e:
        log.warning(f"Precipitation detrending failed for {model} {realization}: {e} – skipping")
        return None

    # Compute Nino indices
    nino = compute_nino_indices(sst_detrended)
    nino34_raw = nino["nino34"]
    nino3_sst_raw = nino["nino3_sst"]
    nino4_sst_raw = nino["nino4_sst"]

    # Skip members whose data doesn't overlap the baseline period (e.g. ssp-only members)
    sst_years = sst.time.dt.year.values
    if sst_years.max() < BASELINE[0] or sst_years.min() > BASELINE[1]:
        log.warning(f"  {model} {realization} [{scenario}]: data {sst_years.min()}-{sst_years.max()} does not overlap baseline {BASELINE} – skipping")
        return None

    # Remove seasonal cycle and standardize
    nino34_res = fns.remove_seasonal_cycle_rolling_with_std(
        nino34_raw, window_years=30, rolling_std=False, baseline_period=BASELINE
    )
    nino34_std = nino34_res["standardized"]

    nino3_res = fns.remove_seasonal_cycle_rolling_with_std(
        nino3_sst_raw, window_years=30, rolling_std=False, baseline_period=BASELINE
    )
    nino3_std = nino3_res["standardized"]

    nino4_res = fns.remove_seasonal_cycle_rolling_with_std(
        nino4_sst_raw, window_years=30, rolling_std=False, baseline_period=BASELINE
    )
    nino4_std = nino4_res["standardized"]

    # Niño-3.4 RSST (Izumo 2024 §2b) — built from RAW SST so the tropical-
    # mean subtraction does the detrending by construction.  The Izumo RDO
    # expects T_E = RSST, not SST anomaly.  We store it alongside
    # nino34_std so downstream code can choose.  Deseasonalised (not
    # standardised — kept in °C) so it is directly plug-and-play for the
    # RDO multivariate regression.
    try:
        nino34_rsst_raw = compute_nino34_rsst(sst)
        nino34_rsst_res = fns.remove_seasonal_cycle_rolling_with_std(
            nino34_rsst_raw, window_years=30, rolling_std=False,
            baseline_period=BASELINE,
        )
        nino34_rsst_anom = nino34_rsst_res["anomaly"]
        nino34_rsst_std  = nino34_rsst_res["standardized"]
    except Exception as e:
        log.warning(f"  {model} {realization} [{scenario}]: RSST computation "
                     f"failed ({e}) — storing NaN")
        nino34_rsst_anom = xr.full_like(nino34_std, np.nan)
        nino34_rsst_std  = xr.full_like(nino34_std, np.nan)

    # Compute precipitation anomaly – must succeed
    try:
        nino3_pr = pr_detrended.sel(lat=slice(-5,5), lon=slice(210,270)).mean(["lat","lon"])
        nino3_pr_anom = fns.remove_seasonal_cycle_rolling_with_std(
            nino3_pr, window_years=30, rolling_std=False, baseline_period=BASELINE
        )["anomaly"]
    except Exception as e:
        log.warning(f"Precipitation anomaly calculation failed for {model} {realization}: {e} – skipping")
        return None

    # Fill gaps in Nino3.4
    # gap_threshold_months=2: treat ≥3-month sustained dips below |σ|<0.5
    # as genuine terminations.  See preprocess_detect_events_observations.py
    # for the 2014-16 rationale.
    nino34_filled = fns.fill_small_enso_gaps(nino34_std, phase='el_nino', gap_threshold_months=2)
    nino34_filled = fns.fill_small_enso_gaps(nino34_filled, phase='la_nina', gap_threshold_months=2)

    if nino34_filled.isnull().all():
        log.warning(f"No valid Nino3.4 data for {model} {realization}, skipping")
        return None

    # Detect ENSO events
    try:
        events = fns.detect_enso_events(
            nino34_anom=nino34_filled,
            nino3_sst_std=nino3_std,
            nino3_pr=nino3_pr_anom,
            nino4_sst_std=nino4_std,
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
    except TypeError as e:
        if "unsupported operand type(s) for -: 'int' and 'NoneType'" in str(e):
            log.warning(f"Skipping {model} {realization} due to internal event_start None error")
            return None
        else:
            raise

    events = fns.merge_events_by_gap(events, nino34_anom=nino34_filled, gap_months_threshold=3, min_duration=5)

    phase_keys = [
        'extreme_el_nino', 'strong_el_nino', 'moderate_el_nino',
        'extreme_la_nina', 'strong_la_nina', 'moderate_la_nina'
    ]

    # Build phase indicator array over full time axis (integer index → phase label)
    # Used for variance decomposition C_k = mean(X_t² · 1_k(t))
    # Derive both _nino34_years and phase_indicator from the same .values call
    # so their lengths are guaranteed identical.
    _nino34_vals  = nino34_filled.values
    _nino34_years = nino34_filled.time.dt.year.values
    n_times = len(_nino34_vals)
    if len(_nino34_years) != n_times:
        # Defensive: force consistency (should never happen, but guards against
        # lazy-load edge cases where time and data axes resolve differently)
        _nino34_years = _nino34_years[:n_times]

    phase_indicator = np.full(n_times, 'neutral', dtype=object)
    for phase in phase_keys:
        for e in events.get(phase, []):
            s_idx = int(e['start_idx'])
            e_idx = min(int(e['end_idx']), n_times)  # clamp to array length
            phase_indicator[s_idx:e_idx] = phase

    # Mean‑state indices from raw SST (not detrended), annual means
    mean_state = compute_mean_state(sst, tauu, z20=z20, zos=zos)

    # Bjerknes feedback inputs: return deseasonalised monthly τ_x and
    # H_tilt anomalies on Cai 2018 boxes (170-230°E / 120-180°E / 220-280°E)
    # with H_tilt = H_E − H_W.  β_H itself is computed downstream as a
    # per-GWI-bin pooled regression (no 30-yr rolling window).
    bjerknes_inputs = compute_bjerknes_feedback_rolling(tauu, z20,
                                                         window_years=30)

    # ── Save full monthly time series (no rolling here) ──────────────────────
    # Rolling windows are computed in the figure script with a configurable width.
    # Phase indicator: encode as integer for compact NetCDF storage.
    _PHASE_INT = {
        'neutral':          0,
        'moderate_el_nino': 1,
        'strong_el_nino':   2,
        'extreme_el_nino':  3,
        'moderate_la_nina': -1,
        'strong_la_nina':   -2,
        'extreme_la_nina':  -3,
    }
    phase_int = np.array([_PHASE_INT.get(p, 0) for p in phase_indicator], dtype=np.int8)

    # Time coordinate for monthly variables
    time_coord = nino34_filled.time
    nino_years = np.unique(time_coord.dt.year.values)

    # Build annual-mean mean-state indices, aligned to nino_years
    # Each variable is resampled then reindexed to the common year range so
    # they all share one 'year' dimension (avoids conflicting sizes error).
    # Iterate over whatever keys compute_mean_state actually produced — different
    # models return different subsets depending on variable availability.
    ms_annual: dict[str, np.ndarray] = {}
    # Helper: extract calendar year from any time value (np.datetime64, cftime, etc.)
    # Neither .dt nor resample/groupby can be relied upon because certain merge paths
    # strip the DatetimeIndex/CFTimeIndex backing from the coord.
    def _years_of(t_vals: np.ndarray) -> np.ndarray:
        if np.issubdtype(t_vals.dtype, np.datetime64):
            return t_vals.astype("datetime64[Y]").astype(int) + 1970
        # cftime or other object dtype — extract .year attribute per element
        return np.array([int(v.year) for v in t_vals], dtype=int)

    for key in ("sst_gradient", "z20_tilt", "z20_mean", "h_star",
                "h_eq_sw_zos", "h_eq_sw_z20",
                "zos_gradient", "tauu_central", "tauu_gradient"):
        if key not in mean_state or mean_state[key] is None:
            continue
        arr = mean_state[key]
        t_years = _years_of(arr["time"].values)
        vals_arr = arr.values
        # Annual mean via numpy grouping (avoids pandas DatetimeIndex requirement)
        ann_years_unique, inverse = np.unique(t_years, return_inverse=True)
        ann_sum   = np.zeros(len(ann_years_unique), dtype=np.float64)
        ann_count = np.zeros(len(ann_years_unique), dtype=np.int64)
        finite = np.isfinite(vals_arr)
        np.add.at(ann_sum,   inverse[finite], vals_arr[finite])
        np.add.at(ann_count, inverse[finite], 1)
        with np.errstate(invalid="ignore"):
            ann_mean = np.where(ann_count > 0, ann_sum / ann_count, np.nan)
        # Reindex to nino_years; fill missing with NaN
        year_to_idx = {int(y): i for i, y in enumerate(ann_years_unique)}
        vals = np.full(len(nino_years), np.nan)
        for j, yr in enumerate(nino_years):
            i = year_to_idx.get(int(yr))
            if i is not None:
                vals[j] = ann_mean[i]
        ms_annual[key] = vals

    ds_vars = {
        "nino34_std":      (["time"], _nino34_vals),
        "nino3_std":       (["time"], nino3_std.reindex(time=time_coord, method="nearest").values),
        "nino4_std":       (["time"], nino4_std.reindex(time=time_coord, method="nearest").values),
        # Niño-3.4 RSST (Izumo 2020 / van Oldenborgh 2021 / Izumo 2024 §2b).
        # Stored as BOTH:
        #   - anomaly  in °C, for use as T_E in the RDO multivariate regression
        #   - standardised, for side-by-side diagnostics with nino34_std
        "nino34_rsst_anom": (["time"],
                             nino34_rsst_anom.reindex(time=time_coord,
                                                      method="nearest").values),
        "nino34_rsst_std":  (["time"],
                             nino34_rsst_std.reindex(time=time_coord,
                                                     method="nearest").values),
        "phase_indicator": (["time"], phase_int),
        "gwi":             (["time"], gwi_aligned.values),
    }
    # Annual mean-state fields (year-indexed, backward compatible)
    for key, vals in ms_annual.items():
        ds_vars[f"meanstate_{key}"] = (["year"], vals)

    # Bjerknes-feedback input anomalies: monthly τ_x(170-230°E) and
    # H_tilt = H_E − H_W (Cai 2018), deseasonalised via
    # remove_seasonal_cycle_rolling_with_std.  β_H itself is computed
    # downstream as a pooled regression per (member, GWI-bin).
    if bjerknes_inputs is not None:
        for src_key, dst_name in (
                ("tauu_cai_anom",   "tauu_cai_anom"),
                ("h_tilt_cai_anom", "h_tilt_cai_anom")):
            src = bjerknes_inputs[src_key]
            try:
                aligned = src.reindex(time=time_coord, method="nearest",
                                       tolerance="31D")
                ds_vars[dst_name] = (["time"], aligned.values)
            except Exception:
                try:
                    src_ym = np.array([t.year * 100 + t.month
                                        for t in src["time"].values])
                    dst_ym = np.array([t.year * 100 + t.month
                                        for t in time_coord.values])
                except AttributeError:
                    st = src["time"].values
                    dt = time_coord.values
                    src_ym = (st.astype("datetime64[Y]").astype(int) + 1970) * 100 + \
                             (st.astype("datetime64[M]").astype(int) % 12 + 1)
                    dst_ym = (dt.astype("datetime64[Y]").astype(int) + 1970) * 100 + \
                             (dt.astype("datetime64[M]").astype(int) % 12 + 1)
                src_lookup = {int(ym): i for i, ym in enumerate(src_ym)}
                vals = np.full(len(dst_ym), np.nan, dtype=np.float64)
                for j, ym in enumerate(dst_ym):
                    i = src_lookup.get(int(ym))
                    if i is not None:
                        vals[j] = float(src.values[i])
                ds_vars[dst_name] = (["time"], vals)

    # Monthly mean-state fields (time-indexed) — allows downstream DJF/MAM/...
    # seasonal analysis without reopening raw tropics files. Aligned to the
    # same time_coord as nino34_std via nearest-neighbor reindexing.
    for key in ("sst_gradient", "z20_tilt", "z20_mean", "h_star",
                "h_eq_sw_zos", "h_eq_sw_z20",
                "zos_gradient", "tauu_central", "tauu_gradient"):
        if key not in mean_state or mean_state[key] is None:
            continue
        ms_monthly = mean_state[key]
        try:
            aligned = ms_monthly.reindex(time=time_coord, method="nearest",
                                         tolerance="31D")
        except Exception:
            # Fall back to manual year-month matching
            try:
                src_ym = np.array([t.year * 100 + t.month
                                   for t in ms_monthly["time"].values])
                dst_ym = np.array([t.year * 100 + t.month
                                   for t in time_coord.values])
            except AttributeError:
                st = ms_monthly["time"].values
                dt = time_coord.values
                src_ym = (st.astype("datetime64[Y]").astype(int) + 1970) * 100 + \
                         (st.astype("datetime64[M]").astype(int) % 12 + 1)
                dst_ym = (dt.astype("datetime64[Y]").astype(int) + 1970) * 100 + \
                         (dt.astype("datetime64[M]").astype(int) % 12 + 1)
            src_lookup = {int(ym): i for i, ym in enumerate(src_ym)}
            vals_monthly = np.full(len(dst_ym), np.nan, dtype=np.float64)
            for j, ym in enumerate(dst_ym):
                i = src_lookup.get(int(ym))
                if i is not None:
                    vals_monthly[j] = float(ms_monthly.values[i])
            ds_vars[f"meanstate_monthly_{key}"] = (["time"], vals_monthly)
            continue
        ds_vars[f"meanstate_monthly_{key}"] = (["time"], aligned.values)

    # ── Recharge-index anomaly (h_eq_sw_*) for RDO fitting ──────────────────
    # The raw meanstate_monthly_h_eq_sw_* is the absolute spatial average of
    # Z20 depth (~150 m) or ZOS (~0.1 m), dominated by the mean + seasonal
    # cycle + long-term warming trend.  Izumo's RDO expects h as a
    # deseasonalised anomaly on interannual timescales.  Apply the same
    # deseasonalisation that nino34/tauu/h_tilt already use, so downstream
    # RDO fits see a clean h anomaly.
    for key in ("h_eq_sw_zos", "h_eq_sw_z20"):
        if key not in mean_state or mean_state[key] is None:
            continue
        raw = mean_state[key]
        try:
            res = fns.remove_seasonal_cycle_rolling_with_std(
                raw, window_years=30, rolling_std=False,
                baseline_period=BASELINE,
            )
            anom = res["anomaly"]
            aligned = anom.reindex(time=time_coord, method="nearest",
                                   tolerance="31D")
            ds_vars[f"meanstate_monthly_{key}_anom"] = (
                ["time"], aligned.values)
        except Exception as e:
            log.warning(f"  {model} {realization}: deseasonalisation of "
                        f"{key} failed ({e}) — skipping anom variant")

    ds_metrics = xr.Dataset(
        ds_vars,
        coords={
            "time": time_coord.values,
            "year": nino_years,
        }
    )
    ds_metrics.attrs["realization"] = realization

    # Build per-event rows for composite output
    # GWI is smoothed (polynomial) so any month within the event gives the same
    # warming level; use the DJF season the event is phase-locked to.
    gwi_ym = gwi_aligned.time.dt.year.values * 100 + gwi_aligned.time.dt.month.values
    gwi_val = gwi_aligned.values
    gwi_lookup = {int(ym): float(v) for ym, v in zip(gwi_ym, gwi_val)}

    event_rows = []
    for phase in phase_keys:
        for e in events.get(phase, []):
            peak_date = e.get("peak_date", "")
            try:
                peak_yr  = int(str(peak_date)[:4])
                peak_mon = int(str(peak_date)[5:7])
            except Exception:
                continue
            # D(year0)JF(year+1): Dec of year0, Jan+Feb of year+1
            # year0 is the calendar year of December in the DJF season
            # For peak in Dec → year0 = peak_yr; for Jan/Feb → year0 = peak_yr-1
            if peak_mon == 12:
                djf_year0 = peak_yr
            else:
                djf_year0 = peak_yr - 1
            djf_months = [djf_year0 * 100 + 12,
                          (djf_year0 + 1) * 100 + 1,
                          (djf_year0 + 1) * 100 + 2]
            gwi_djf_vals = [gwi_lookup[ym] for ym in djf_months if ym in gwi_lookup]
            gwi_djf = float(np.mean(gwi_djf_vals)) if gwi_djf_vals else np.nan
            event_rows.append({
                "model": model,
                "scenario": scenario,
                "member": realization,
                "phase": phase,
                "start_date": e.get("start_date", ""),
                "end_date": e.get("end_date", ""),
                "peak_date": peak_date,
                "duration": e.get("duration", np.nan),
                "start_idx": e.get("start_idx", np.nan),
                "end_idx": e.get("end_idx", np.nan),
                "djf_year0": djf_year0,
                "gwi_djf": gwi_djf,
            })

    return ds_metrics, event_rows

def stack_members(model: str, scenario: str, member_results: List[xr.Dataset]) -> xr.Dataset:
    """Stack per-member raw time series along a new 'ens' dimension.

    All variables with a 'time' dimension are aligned to the union of all
    member time axes (NaN-filled where a member has no data); variables with
    a 'year' dimension are aligned to their union similarly.

    Calendar mismatches (cftime vs pandas Timestamp) are resolved by
    normalising every member's time axis to a pandas DatetimeIndex keyed on
    year*100+month before stacking.

    Returns an xr.Dataset with dims (ens, time) and (ens, year), where the
    'ens' coordinate contains the realization strings.
    """
    if not member_results:
        return None

    def _ym_keys(ds):
        t = ds["time"]
        return t.dt.year.values * 100 + t.dt.month.values

    def _to_pandas_time(ds, common_ym):
        """Select common months and normalise time axis to pandas DatetimeIndex."""
        ym = _ym_keys(ds)
        mask = np.isin(ym, common_ym)
        ds_sel = ds.isel(time=mask)
        t = ds_sel["time"]
        yr  = t.dt.year.values
        mon = t.dt.month.values
        try:
            day = t.dt.day.values
        except Exception:
            day = np.ones(len(t), dtype=int)
        # Always use day=01 so that ref_times (also day=01) matches exactly
        pd_times = pd.to_datetime(
            [f"{y:04d}-{m:02d}-01" for y, m in zip(yr, mon)]
        )
        return ds_sel.assign_coords(time=pd_times)

    # Build variable schema as UNION across all members — different members
    # can have different mean-state fields depending on which variables (tauu,
    # zos, z20) were available for that realization.
    time_vars_set: set[str] = set()
    year_vars_set: set[str] = set()
    for r in member_results:
        for v in r.data_vars:
            dims = r[v].dims
            if "time" in dims:
                time_vars_set.add(v)
            elif "year" in dims:
                year_vars_set.add(v)
    time_vars = sorted(time_vars_set)
    year_vars = sorted(year_vars_set)

    # ── Build common time axis (union — keep all months present in any member) ─
    # Use intersection so that every member has full data on the common axis;
    # members that start later (e.g. ssp-only) are handled by skipping them if
    # they have no overlap — they still get included for the months they do have.
    # We use UNION here and NaN-fill missing months below.
    all_ym = set()
    for r in member_results:
        all_ym.update(_ym_keys(r).tolist())
    common_ym = np.array(sorted(all_ym), dtype=np.int64)

    # Build a reference pandas DatetimeIndex from common_ym
    ref_times = pd.to_datetime([
        f"{ym // 100:04d}-{ym % 100:02d}-01" for ym in common_ym
    ])

    # ── Stack time-indexed variables ──────────────────────────────────────────
    realizations = [r.attrs.get("realization", f"r{i}") for i, r in enumerate(member_results)]
    aligned_t = []
    for r in member_results:
        r_norm = _to_pandas_time(r, _ym_keys(r))  # just normalise calendar
        # Select only the time_vars this member actually has — then reindex onto
        # the full common time axis (NaN-fill missing months). For time_vars
        # absent from this member, add a NaN-filled placeholder.
        present = [v for v in time_vars if v in r_norm.data_vars]
        r_reindexed = r_norm[present].reindex(time=ref_times)
        for v in time_vars:
            if v not in r_reindexed.data_vars:
                r_reindexed[v] = (("time",),
                                  np.full(len(ref_times), np.nan, dtype=np.float64))
        aligned_t.append(r_reindexed)
    combined_t = xr.concat(aligned_t, dim=xr.DataArray(realizations, dims="ens", name="ens"))

    # ── Stack year-indexed variables ──────────────────────────────────────────
    combined_y = None
    if year_vars:
        all_years = set()
        for r in member_results:
            if "year" in r.coords:
                all_years.update(r["year"].values.tolist())
        common_years = np.array(sorted(all_years), dtype=np.int64)

        aligned_y = []
        for r in member_results:
            if "year" not in r.coords:
                # Create NaN-filled placeholder across full union schema
                stub = xr.Dataset(
                    {v: (["year"], np.full(len(common_years), np.nan)) for v in year_vars},
                    coords={"year": common_years},
                )
                aligned_y.append(stub)
            else:
                present = [v for v in year_vars if v in r.data_vars]
                r_y = r[present].reindex(year=common_years)
                # NaN-fill any variable this member was missing
                for v in year_vars:
                    if v not in r_y.data_vars:
                        r_y[v] = (("year",),
                                  np.full(len(common_years), np.nan, dtype=np.float64))
                aligned_y.append(r_y)
        combined_y = xr.concat(aligned_y, dim=xr.DataArray(realizations, dims="ens", name="ens"))

    # ── Merge and set attributes ──────────────────────────────────────────────
    ds_model = combined_t
    if combined_y is not None:
        ds_model = ds_model.merge(combined_y)
    ds_model.attrs["model"]     = model
    ds_model.attrs["scenario"]  = scenario
    ds_model.attrs["n_members"] = len(member_results)
    return ds_model

# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="ENSO metrics for multi-model large ensembles.")
    parser.add_argument("--models", nargs="+", default="all", help="Models to process (default: all)")
    parser.add_argument("--sst-var", default="tos", help="SST variable name (default: tos)")
    parser.add_argument("--pr-var", default="pr", help="Precipitation variable name (default: pr)")
    parser.add_argument("--tauu-var", default="tauu", help="Zonal wind stress variable name (default: tauu)")
    parser.add_argument("--lat-range", nargs=2, type=float,
                        default=(_LAT_MIN_DEFAULT, _LAT_MAX_DEFAULT),
                        help=f"Latitude range (default: {_LAT_MIN_DEFAULT} {_LAT_MAX_DEFAULT} from config.py)")
    parser.add_argument("--lon-range", nargs=2, type=float,
                        default=(_LON_MIN_DEFAULT, _LON_MAX_DEFAULT),
                        help=f"Longitude range (default: {_LON_MIN_DEFAULT} {_LON_MAX_DEFAULT} from config.py)")
    parser.add_argument("--output", default=str(OUTPUT_DIR), help="Output root directory")
    parser.add_argument("--n-jobs", type=int, default=1, help="Number of parallel jobs")
    args = parser.parse_args()

    if args.models == "all":
        models = [p.name for p in TROPICS_DATA_DIR.iterdir() if p.is_dir()]
    else:
        models = args.models

    lat_min, lat_max = args.lat_range
    lon_min, lon_max = args.lon_range
    output_root = Path(args.output)

    # Collect all (model, scenario) results for a cross-scenario combined file
    all_ds: List[xr.Dataset] = []

    for model in tqdm(models, desc="Models"):
        reals_by_scenario = get_realizations_by_scenario(
            model, args.sst_var, lat_min, lat_max, lon_min, lon_max
        )
        if not reals_by_scenario:
            log.warning(f"No realizations found for {model} – skipping")
            continue

        # Drop pure-historical entries: hist data is already merged into each
        # future scenario run (1850-2100). Processing historical alone gives
        # only 1850-2014 and covers no meaningful GWI warming levels.
        future_scenarios = {s: r for s, r in reals_by_scenario.items() if s != "historical"}
        if not future_scenarios:
            log.warning(f"  {model}: no future scenarios found – skipping")
            continue

        log.info(f"{model}: scenarios found: {list(future_scenarios.keys())}")

        for scenario, realizations in future_scenarios.items():
            log.info(f"  {model} [{scenario}]: {len(realizations)} realizations")

            scen_dir = output_root / scenario
            model_output = scen_dir / f"{model}_metrics.nc"

            raw_results = joblib.Parallel(n_jobs=args.n_jobs)(
                joblib.delayed(process_member)(
                    model, real, scenario,
                    args.sst_var, args.pr_var, args.tauu_var,
                    lat_min, lat_max, lon_min, lon_max,
                )
                for real in realizations
            )
            raw_results = [r for r in raw_results if r is not None]
            if not raw_results:
                log.warning(f"  No valid members for {model} [{scenario}]")
                continue

            member_results = [r[0] for r in raw_results]
            all_event_rows = [row for r in raw_results for row in r[1]]

            ds_model = stack_members(model, scenario, member_results)
            if ds_model is None:
                continue

            # Save per-model per-scenario metrics NetCDF
            scen_dir.mkdir(parents=True, exist_ok=True)
            ds_model.to_netcdf(model_output)
            log.info(f"  Saved {model} [{scenario}] → {model_output}")

            # Save per-model per-scenario events CSV for composites
            if all_event_rows:
                events_df = pd.DataFrame(all_event_rows)
                events_csv = scen_dir / f"{model}_events.csv"
                events_df.to_csv(events_csv, index=False)
                log.info(f"  Saved events CSV ({len(events_df)} rows) → {events_csv}")

            all_ds.append(ds_model)

    if not all_ds:
        log.warning("No model results produced.")


if __name__ == "__main__":
    main()