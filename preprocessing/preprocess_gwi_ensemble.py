#!/usr/bin/env python3
"""
Preprocess GWI for all models and members, combining historical and future files.

Usage:
    # THREDDS/RDA (default; slow — use when Casper unavailable)
    python preprocess_gwi_ensemble.py --models all --n-jobs 4

    # Casper fast path: run cdo -fldmean remotely, rsync ~30 KB/member
    python preprocess_gwi_ensemble.py --source casper --models cesm1_lens gfdl_cm3_lens giss-e2
"""

import argparse
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from collections import defaultdict

import socket
import xarray as xr
import requests
import numpy as np
import joblib
from tqdm import tqdm
from html.parser import HTMLParser

# Casper paths — mirror casper_pipeline.py so the same SSH ControlMaster is reused
CASPER_HOST = "casper"
CASPER_BASE_D651   = "/glade/campaign/collections/gdex/data/d651039"
CASPER_BASE_NMAHER = "/glade/campaign/cgd/cas/nmaher"
_CASPER_SCRATCH    = "/path/to/scratch/gwi_fldmean"

# Hard socket timeout so stalled OPeNDAP connections don't hang forever.
# xr.open_dataset has no built-in timeout; this sets it at the OS level.
socket.setdefaulttimeout(120)

# Shared config (paths, model list, extract_realization)
_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))
from config import (  # noqa: E402
    GWI_CACHE_DIR, ALL_MODELS, extract_realization,
    THREDDS_CATALOG_BASE, THREDDS_DODS_BASE, RDA_DATASET_ID,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

def list_nc_files_in_catalog(catalog_html_url, session):
    """Return list of OPeNDAP URLs for .nc files in a catalog."""
    urls = []
    try:
        resp = session.get(catalog_html_url, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        log.warning(f"    catalog fetch failed: {catalog_html_url} ({e})")
        return urls

    class LinkParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.links = []
        def handle_starttag(self, tag, attrs):
            if tag == "a":
                for name, val in attrs:
                    if name == "href" and val and f"dataset=files/{RDA_DATASET_ID}" in val:
                        self.links.append(val)
    parser = LinkParser()
    parser.feed(resp.text)

    for href in parser.links:
        if "dataset=" not in href:
            continue
        dataset_path = href.split("dataset=")[1]  # files/{RDA_DATASET_ID}/.../filename.nc
        if not dataset_path.endswith(".nc"):
            continue
        dods_url = f"https://thredds.rda.ucar.edu/thredds/dodsC/{dataset_path}"
        urls.append(dods_url)
    return urls

def get_tas_urls(model, session):
    """Return list of OPeNDAP URLs for global tas for a given model."""
    catalog_url = f"{THREDDS_CATALOG_BASE}/{model}/Amon/tas/catalog.html"
    return list_nc_files_in_catalog(catalog_url, session)


# ════════════════════════════════════════════════════════════════════════════
# Casper fast path
# ════════════════════════════════════════════════════════════════════════════
# For the models whose RDA tas files have broken baselines (CESM1/GFDL-CM3:
# 1920-start; GISS-E2 f1 variants: short-hist truncation), it's much faster
# to pull tas from Casper and do the global area-weighted mean remotely with
# CDO's -fldmean operator.  CDO uses native grid-cell area weights (no manual
# cos(lat) weighting needed).  Remote output is a tiny (~30 KB) timeseries.


# Per-(model) Casper source for tas.  d651 has clean merged hist_rcp85 files
# for CESM1/GFDL-CM3 and clean separate hist+ssp585 files for GISS-E2, so we
# prefer d651 for these 3 to avoid nmaher's messy naming.  For models where
# d651 has no tas (e3sm_v1_lens, gfdl_esm2m_lens), fall back to nmaher.
_TAS_SOURCE: dict[str, str] = {
    "access_lens":       "d651",
    "canesm2_lens":      "d651",
    "canesm5_lens":      "d651",
    "cesm1_lens":        "d651",
    "cesm2_lens":        "d651",
    "csiro_mk36_lens":   "d651",
    "e3sm_v1_lens":      "d651",
    "e3sm_v2_lens":      "d651",
    "ec-earth3_lens":    "d651",
    "gfdl_cm3_lens":     "d651",
    "gfdl_esm2m_lens":   "d651",
    "gfdl_spear_lens":   "d651",
    "giss-e2":           "d651",
    "ipsl_cm6a_lens":    "d651",
    "miroc6_lens":       "d651",
    "miroc_esm2l_lens":  "d651",
    "mpi_lens":          "d651",
    "mpi_lens_cmip6":    "d651",
    "ukesm1-0-ll_lens":  "d651",
}


def _casper_tas_base(model: str) -> str:
    tag = _TAS_SOURCE.get(model, "d651")
    if tag == "nmaher":
        return f"{CASPER_BASE_NMAHER}/{model}/Amon/tas"
    return f"{CASPER_BASE_D651}/{model}/Amon/tas"


def list_casper_tas(model: str) -> list[str]:
    """List full Casper paths to tas .nc files for a model."""
    base = _casper_tas_base(model)
    cmd = ["ssh", CASPER_HOST, f"ls {base}/*.nc 2>/dev/null"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=60, check=True)
    except subprocess.CalledProcessError:
        return []
    return [ln for ln in out.stdout.strip().split("\n") if ln.endswith(".nc")]


def remote_fldmean_fetch(remote_src: str, local_dest: Path,
                         max_retries: int = 3) -> bool:
    """Run cdo -fldmean on Casper, rsync the tiny result back.

    -fldmean uses native grid-cell area weights (no manual cos(lat) needed).
    -subc,273.15 converts K→°C so we don't need to load the raw file locally
    just to check units.
    """
    local_dest.parent.mkdir(parents=True, exist_ok=True)
    basename = Path(remote_src).name
    remote_out = f"{_CASPER_SCRATCH}/{basename}"

    # module load cdo needs a login shell on Casper.  Run the whole recipe
    # as a single ssh, piping the commands through bash -l.
    script = (
        f"mkdir -p {_CASPER_SCRATCH} && "
        f"module load cdo > /dev/null 2>&1 && "
        f"cdo -s -O -fldmean -subc,273.15 -selname,tas "
        f"{remote_src} {remote_out}.tmp && "
        f"mv {remote_out}.tmp {remote_out}"
    )
    ssh_cmd = ["ssh", CASPER_HOST, "bash", "-l"]

    for attempt in range(max_retries):
        try:
            subprocess.run(ssh_cmd, input=script,
                           capture_output=True, text=True,
                           timeout=180, check=True)
            break
        except subprocess.CalledProcessError as e:
            log.warning(f"  remote fldmean attempt {attempt+1} failed: "
                        f"{e.stderr.strip()[:200]}")
            if attempt == max_retries - 1:
                return False
            time.sleep(2 ** attempt)
        except subprocess.TimeoutExpired:
            log.warning(f"  remote fldmean attempt {attempt+1} timed out")
            if attempt == max_retries - 1:
                return False

    tmp = local_dest.with_suffix(local_dest.suffix + ".part")
    rsync_cmd = [
        "rsync", "-tz", "--inplace", "--partial",
        f"{CASPER_HOST}:{remote_out}", str(tmp),
    ]
    try:
        subprocess.run(rsync_cmd, check=True, capture_output=True,
                       text=True, timeout=120)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        log.warning(f"  rsync fldmean result failed: {e}")
        return False
    tmp.rename(local_dest)

    # Best-effort cleanup of remote scratch
    try:
        subprocess.run(["ssh", CASPER_HOST, f"rm -f {remote_out}"],
                       check=False, capture_output=True, timeout=30)
    except Exception:
        pass
    return True


def compute_gwi_from_local(nc_path: Path) -> xr.DataArray:
    """Load a local fldmean-reduced tas file and return a clean time series."""
    ds = xr.open_dataset(nc_path, decode_times=True, decode_timedelta=True)
    import importlib.util as _iu  # time_fixes sits next to this file
    _spec = _iu.spec_from_file_location("time_fixes", str(Path(__file__).resolve().parent / "time_fixes.py"))
    _tf = _iu.module_from_spec(_spec); _spec.loader.exec_module(_tf)
    ds = _tf.fix_month_offset(ds, nc_path)
    tas = ds["tas"].squeeze(drop=True)
    # fldmean collapses lat/lon to length-1; drop those degenerate axes
    for d in ("lat", "lon", "latitude", "longitude"):
        if d in tas.dims:
            tas = tas.squeeze(d, drop=True)
    if np.issubdtype(tas.dtype, np.floating) is False:
        tas = tas.astype(float)
    # subc,273.15 was already applied remotely (°C)
    idx = tas.indexes.get("time")
    if idx is not None and hasattr(idx, "to_datetimeindex"):
        tas = tas.assign_coords(time=idx.to_datetimeindex())
    return tas.rename("gwi")


def process_realization_casper(model: str, realization: str,
                                remote_files: list[str],
                                local_scratch: Path) -> None:
    """Casper-fast equivalent of process_realization.

    1. cdo -fldmean on Casper for each remote file → ~30 KB local .nc
    2. Load, concat, dedupe, interpolate small NaN gaps
    3. Polynomial-fit + baseline shift (same code as THREDDS path)
    4. Save to GWI_CACHE_DIR/model/realization/gwi.nc
    """
    out_file = GWI_CACHE_DIR / model / realization / "gwi.nc"
    if out_file.exists():
        log.info(f"GWI already exists for {model} {realization}, skipping.")
        return

    gwi_list = []
    for remote_src in remote_files:
        local_reduced = local_scratch / model / Path(remote_src).name
        if not local_reduced.exists():
            ok = remote_fldmean_fetch(remote_src, local_reduced)
            if not ok:
                log.warning(f"  fldmean failed: {remote_src}")
                continue
        try:
            gwi_list.append(compute_gwi_from_local(local_reduced))
        except Exception as e:
            log.warning(f"  load-reduced failed: {local_reduced} ({e})")

    if not gwi_list:
        log.warning(f"No usable data for {model} {realization}")
        return

    if len(gwi_list) == 1:
        gwi_combined = gwi_list[0].load().astype(float)
    else:
        try:
            gwi_combined = _merge_gwi_timeseries(gwi_list).load()
        except Exception as e:
            log.error(f"Failed to merge {model} {realization}: {e}")
            return

    if gwi_combined.isnull().all():
        log.warning(f"All-NaN after merge for {model} {realization}")
        return
    if gwi_combined.isnull().any():
        gwi_combined = gwi_combined.interpolate_na(
            dim="time", method="linear", limit=5)
        if gwi_combined.isnull().any():
            gwi_combined = gwi_combined.ffill("time").bfill("time")

    try:
        gwi_smooth = _extrapolate_poly_to_standard_grid(gwi_combined)
    except Exception as e:
        log.error(f"Poly-fit failed for {model} {realization}: {e}")
        return

    t0 = str(gwi_combined.time.values[0])[:10]
    t1 = str(gwi_combined.time.values[-1])[:10]
    log.info(f"  {model} {realization}: data {t0}..{t1}, saved.")

    out_file.parent.mkdir(parents=True, exist_ok=True)
    temp_file = out_file.with_suffix(".tmp")
    xr.Dataset({"gwi_smoothed": gwi_smooth}).to_netcdf(temp_file)
    temp_file.rename(out_file)


def compute_gwi_from_url(url, retries=3, timeout_s=120):
    """Open tas file via OPeNDAP, compute global area‑weighted mean, return GWI.

    Uses a per-attempt wall-clock timeout (via concurrent.futures) to avoid
    hanging indefinitely on stalled OPeNDAP connections.
    """
    import concurrent.futures

    def _fetch():
        ds = xr.open_dataset(
            url,
            engine="netcdf4",
            decode_times=True,
            decode_timedelta=True,
            chunks={"time": 12}
        )
        tas = ds["tas"]
        if hasattr(tas, "units") and tas.units == "K":
            tas = tas - 273.15
        weights = np.cos(np.deg2rad(tas.lat))
        weights_normalized = weights / weights.sum()
        gwi = tas.weighted(
            xr.DataArray(weights_normalized, dims=["lat"], coords={"lat": tas.lat})
        ).mean(["lat", "lon"])
        # Convert cftime index to standard datetime so float comparisons work
        if hasattr(gwi.indexes.get("time", None), "to_datetimeindex"):
            gwi["time"] = gwi.indexes["time"].to_datetimeindex()
        return gwi

    for attempt in range(retries):
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(_fetch)
                return future.result(timeout=timeout_s)
        except concurrent.futures.TimeoutError:
            log.warning(f"Attempt {attempt+1} timed out ({timeout_s}s) for {url}")
        except Exception as e:
            log.warning(f"Attempt {attempt+1} failed for {url}: {e}")
        if attempt < retries - 1:
            time.sleep(2 ** attempt)
    raise RuntimeError(f"All {retries} attempts failed for {url}")

def _merge_gwi_timeseries(gwi_list):
    """Sort by time, concat, deduplicate overlapping timestamps."""
    # Normalise all time axes to numpy datetime64 so comparisons work across
    # mixed calendar types (cftime.DatetimeNoLeap, datetime64, etc.)
    normalised = []
    for g in gwi_list:
        idx = g.indexes.get("time")
        if idx is not None and hasattr(idx, "to_datetimeindex"):
            g = g.assign_coords(time=idx.to_datetimeindex())
        normalised.append(g.sortby("time"))
    combined = xr.concat(normalised, dim="time", combine_attrs="drop")
    combined = combined.astype(float).sortby("time")
    _, unique_idx = np.unique(combined.time.values, return_index=True)
    return combined.isel(time=unique_idx)


def _extrapolate_poly_to_standard_grid(gwi_raw):
    """
    Fit a degree-4 polynomial to the available GWI data, evaluate it on a
    fixed 1850-01 to 2100-12 monthly grid, and subtract the 1850-1900 mean
    so that the baseline period is 0 regardless of when the model data starts.

    Returns an xr.DataArray on the full 1850-2100 monthly time axis with the
    1850-1900 mean set to 0.
    """
    import pandas as pd  # needed for pd.date_range output time axis
    from config import TIME_START, TIME_END  # noqa: E402 (import inside fn to avoid circular)

    # Full standard grid defined by config TIME_START / TIME_END
    full_times = pd.date_range(TIME_START, TIME_END, freq="MS")
    n_full = len(full_times)

    # Fit polynomial on the data that exists (any start date)
    gwi_vals = gwi_raw.astype(float).values
    valid = ~np.isnan(gwi_vals)
    if valid.sum() < 5:
        raise ValueError("Too few valid data points for polynomial fit.")

    # Map the model's time axis onto the full grid index space for fitting.
    # Use xarray .dt accessor so this works with any calendar (cftime, numpy, etc.)
    years  = gwi_raw.time.dt.year.values
    months = gwi_raw.time.dt.month.values
    # Integer month offset of each model time step relative to 1850-01
    month_offset_per_step = (years - 1850) * 12 + (months - 1)

    # Indices on the 1850-2100 grid corresponding to the model's time steps
    model_idx = month_offset_per_step

    # Normalise over the full 1850-2100 range for numerical stability
    norm_mean = (n_full - 1) / 2.0
    norm_std = (n_full - 1) / 2.0

    model_idx_norm = (model_idx - norm_mean) / norm_std
    full_idx_norm = (np.arange(n_full) - norm_mean) / norm_std

    # Choose polynomial degree based on data span:
    # - Full span (hist+future, ~250 yr): degree 4 captures acceleration well
    # - Partial span (<100 yr, e.g. hist-only or ssp-only): degree 1 (linear)
    #   to avoid overfitting and extrapolation blow-up
    data_years_span = (model_idx[valid].max() - model_idx[valid].min()) / 12
    poly_deg = 4 if data_years_span >= 100 else 1

    coeffs = np.polyfit(model_idx_norm[valid], gwi_vals[valid], poly_deg)

    # Evaluate within data range; flat-cap outside (no extrapolation beyond data)
    data_idx_min = model_idx_norm[valid].min()
    data_idx_max = model_idx_norm[valid].max()

    clipped = np.clip(full_idx_norm, data_idx_min, data_idx_max)
    gwi_full_vals = np.polyval(coeffs, clipped)

    # Subtract baseline mean.  Preferred baseline is 1850-1900 (first 600
    # months) when the model covers it.  For models whose historical
    # segment starts later (CESM1-LE and GFDL-CM3 start 1920, i.e. month
    # index 840), use 1920-1949 as a fallback baseline — standard IPCC AR6
    # Ch3 convention for CMIP5 short-hist models.  Detection: if the
    # earliest month with **real** (non-extrapolated) data starts after
    # 1850-01, there is no 1850-1900 data to anchor the baseline on, so
    # use 1920-1949 months [840..1200).
    first_real_month_offset = int(model_idx[valid].min())
    if first_real_month_offset <= 0:
        # Real data available in 1850 onwards → use 1850-1900
        baseline_idx = slice(0, 600)
        baseline_desc = "1850-1900"
    else:
        # Fallback: use 1920-1949 (months 840..1200)
        baseline_idx = slice(840, 1200)
        baseline_desc = "1920-1949 (model hist-short fallback)"
    baseline_mean = gwi_full_vals[baseline_idx].mean()
    gwi_full_vals -= baseline_mean

    gwi_full = xr.DataArray(
        gwi_full_vals,
        coords={"time": full_times},
        dims=["time"],
        attrs={
            "description": "GWI polynomial fit extrapolated to 1850-2100, baseline 1850-1900=0",
            "method": "degree-4 polynomial fit to available data, evaluated on full grid",
            "baseline": "1850-1900 mean = 0",
        },
    )
    return gwi_full


def process_realization(model, realization, urls, session):
    """Process a single realization: combine files, compute and save smoothed GWI."""
    out_file = GWI_CACHE_DIR / model / realization / "gwi.nc"
    if out_file.exists():
        log.info(f"GWI already exists for {model} {realization}, skipping.")
        return

    gwi_list = []
    for url in urls:
        try:
            gwi = compute_gwi_from_url(url)
            if not np.issubdtype(gwi.dtype, np.number):
                gwi = gwi.astype(float)
            gwi_list.append(gwi)
        except Exception as e:
            log.warning(f"Skipping {url} due to error: {e}")
            continue

    if not gwi_list:
        log.warning(f"No valid data for {model} {realization}, skipping.")
        return

    if len(gwi_list) == 1:
        gwi_combined = gwi_list[0].load().astype(float)
    else:
        try:
            gwi_combined = _merge_gwi_timeseries(gwi_list).load()
        except Exception as e:
            log.error(f"Failed to merge GWI files for {model} {realization}: {e}")
            return

    if gwi_combined.isnull().all():
        log.warning(f"No valid data for {model} {realization}, skipping.")
        return

    # Interpolate any interior NaNs (small gaps only)
    if gwi_combined.isnull().any():
        log.info(f"  Interpolating NaNs for {model} {realization}")
        gwi_combined = gwi_combined.interpolate_na(dim='time', method='linear', limit=5)
        if gwi_combined.isnull().any():
            gwi_combined = gwi_combined.ffill('time').bfill('time')

    # Fit degree-4 polynomial, extrapolate to full 1850-2100 grid, set baseline
    try:
        gwi_smooth = _extrapolate_poly_to_standard_grid(gwi_combined)
    except Exception as e:
        log.error(f"Failed to compute GWI polynomial for {model} {realization}: {e}")
        return

    t0 = str(gwi_combined.time.values[0])[:10]
    t1 = str(gwi_combined.time.values[-1])[:10]
    log.info(f"  Data range: {t0} to {t1}; GWI(1850-1900) baseline set to 0")

    # Save atomically
    out_file.parent.mkdir(parents=True, exist_ok=True)
    temp_file = out_file.with_suffix(".tmp")
    ds_out = xr.Dataset({"gwi_smoothed": gwi_smooth})
    ds_out.to_netcdf(temp_file)
    temp_file.rename(out_file)
    log.info(f"Saved GWI for {model} {realization}")

def main():
    parser = argparse.ArgumentParser(description="Precompute GWI for all models and members.")
    parser.add_argument("--models", nargs="+", default=ALL_MODELS, help="Models to process")
    parser.add_argument("--n-jobs", type=int, default=1, help="Number of parallel jobs")
    parser.add_argument("--source", choices=["thredds", "casper"], default="thredds",
                        help="thredds = RDA OPeNDAP (default); casper = remote CDO fldmean on NCAR Casper")
    args = parser.parse_args()

    if args.models == ["all"]:
        args.models = ALL_MODELS

    if args.source == "casper":
        scratch = Path(tempfile.gettempdir()) / f"gwi_fldmean_{os.getuid()}"
        scratch.mkdir(parents=True, exist_ok=True)

        tasks = []
        for model in args.models:
            files = list_casper_tas(model)
            if not files:
                log.warning(f"No Casper tas files for {model} (base={_casper_tas_base(model)})")
                continue
            groups = defaultdict(list)
            for f in files:
                real = extract_realization(Path(f).name)
                groups[real].append(f)
            log.info(f"Model {model}: {len(groups)} realizations on Casper")
            for real, files_group in groups.items():
                tasks.append((model, real, files_group))

        log.info(f"Total Casper tasks: {len(tasks)}")
        joblib.Parallel(n_jobs=args.n_jobs)(
            joblib.delayed(process_realization_casper)(model, real, files_group, scratch)
            for model, real, files_group in tqdm(tasks, desc="Casper GWI")
        )
        log.info("GWI preprocessing completed (Casper).")
        return

    # Default: THREDDS/RDA path
    session = requests.Session()
    session.headers["User-Agent"] = "mmleav2-gwi/1.0"

    tasks = []
    for model in args.models:
        urls = get_tas_urls(model, session)
        if not urls:
            log.warning(f"No tas files found for {model}")
            continue

        # Group URLs by realization
        groups = defaultdict(list)
        for url in urls:
            filename = url.split("/")[-1]
            real = extract_realization(filename)
            groups[real].append(url)

        log.info(f"Model {model} has {len(groups)} realizations")
        for real, urls_group in groups.items():
            tasks.append((model, real, urls_group))

    # Process in parallel
    joblib.Parallel(n_jobs=args.n_jobs)(
        joblib.delayed(process_realization)(model, real, urls_group, session)
        for model, real, urls_group in tqdm(tasks, desc="Processing realizations")
    )
    log.info("GWI preprocessing completed.")

if __name__ == "__main__":
    main()