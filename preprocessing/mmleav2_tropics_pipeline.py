#!/usr/bin/env python3
"""
MMLEAv2 Tropical Subset Pipeline — OPeNDAP edition
=====================================================
Crawls the live THREDDS catalog at:
  https://thredds.rda.ucar.edu/thredds/catalog/files/d651039/

Opens every matching NetCDF file *remotely* over OPeNDAP (no download),
subsets to a user-defined latitude-longitude box, writes one small output
file per member, then moves on. Nothing beyond the final subsetted files is
ever written to disk.

Requirements:
  pip install xarray netCDF4 numpy requests

Usage examples:
  # Test with 1 model, 1 variable, equatorial Pacific only:
  python mmleav2_opendap.py --models cesm2_lens --vars pr --lat-range -6 6 --lon-range 120 280 --dry-run

  # All atm monthly for one model, full tropics (default):
  python mmleav2_opendap.py --models cesm2_lens --vars pr tas psl ts tauu tauv zg500 evspsbl

  # All models, all vars, equatorial Pacific only:
  python mmleav2_opendap.py --lat-range -6 6 --lon-range 120 280

  # Resume (skips files whose output already exists):
  python mmleav2_opendap.py --models cesm2_lens --vars pr
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import requests

try:
    import xarray as xr
    import numpy as np
    import dask
    import joblib
except ImportError:
    sys.exit("ERROR: install dependencies first:\n  pip install xarray netCDF4 numpy requests joblib")

# Shared config (paths, model list, VAR_FREQ_MAP, canonical box)
_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))
from config import (
    TROPICS_DIR as _OUTPUT_DIR_DEFAULT,
    LOGS_DIR,
    ALL_MODELS,
    VAR_FREQ_MAP,
    ALL_VARS,
    LAT_MIN as _LAT_MIN_DEFAULT,
    LAT_MAX as _LAT_MAX_DEFAULT,
    LON_MIN as _LON_MIN_DEFAULT,
    LON_MAX as _LON_MAX_DEFAULT,
    TIME_START,
    TIME_END,
    THREDDS_CATALOG_BASE,
    THREDDS_DODS_BASE,
    RDA_DATASET_ID,
)

# ─── logging ──────────────────────────────────────────────────────────────────
import datetime as _dt
_log_file = LOGS_DIR / f"mmleav2_pipeline_{_dt.datetime.now():%Y%m%d_%H%M%S}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(_log_file),
    ],
)
log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

OUTPUT_DIR = _OUTPUT_DIR_DEFAULT


# ═══════════════════════════════════════════════════════════════════════════════
# THREDDS CATALOG CRAWLER
# ═══════════════════════════════════════════════════════════════════════════════

def list_nc_files_in_catalog(catalog_html_url: str, session: requests.Session) -> list[str]:
    """
    Given a THREDDS catalog.html URL for a leaf directory (one that contains .nc files),
    return a list of OPeNDAP dodsC URLs for all .nc files found there.
    """
    urls = []
    resp = None
    for attempt in range(3):  # the RDA THREDDS server can take 40 s or more on a cold catalog
        try:
            resp = session.get(catalog_html_url, timeout=180)
            resp.raise_for_status()
            break
        except Exception as e:
            log.warning(f"    catalog fetch failed (attempt {attempt + 1}): {catalog_html_url}  ({e})")
            resp = None
            time.sleep(30)
    if resp is None:
        return urls

    # Parse dataset links from the HTML table.
    from html.parser import HTMLParser

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


def get_file_urls(model: str, var: str, freq: str,
                  session: requests.Session) -> list[str]:
    """Return all OPeNDAP URLs for a given model/var/freq combination."""
    catalog_url = f"{THREDDS_CATALOG_BASE}/{model}/{freq}/{var}/catalog.html"
    urls = list_nc_files_in_catalog(catalog_url, session)
    if urls:
        log.info(f"    {model}/{freq}/{var}: {len(urls)} file(s) found")
    else:
        log.debug(f"    {model}/{freq}/{var}: no files (model may lack this var)")
    return urls


# ═══════════════════════════════════════════════════════════════════════════════
# SUBSETTING & SAVING
# ═══════════════════════════════════════════════════════════════════════════════

def normalise_dims(ds: xr.Dataset) -> xr.Dataset:
    """Rename latitude/longitude/nav_lat/nav_lon variants to lat/lon."""
    renames = {}
    for old, new in [("latitude", "lat"), ("longitude", "lon"),
                     ("nav_lat", "lat"), ("nav_lon", "lon")]:
        if old in ds.dims and new not in ds.dims:
            renames[old] = new
    if renames:
        ds = ds.rename(renames)
    return ds


def subset_box(ds: xr.Dataset, lat_min: float, lat_max: float,
               lon_min: float, lon_max: float) -> xr.Dataset:
    """Subset dataset to given lat/lon box and time range."""
    ds = normalise_dims(ds)

    if "time" in ds.dims:
        try:
            ds = ds.sel(time=slice(TIME_START, TIME_END))
        except Exception:
            pass  # some files have non-standard time; keep as-is

    # Latitude subset
    if "lat" in ds.dims:
        # Ensure lat is in decreasing order? Actually slice works with bounds.
        ds = ds.sel(lat=slice(lat_min, lat_max))
    elif "lat" in ds.coords:
        # curvilinear grids: mask instead of slice
        mask = (ds["lat"] >= lat_min) & (ds["lat"] <= lat_max)
        ds = ds.where(mask, drop=True)

    # Longitude subset: convert to 0-360 if needed
    lon0 = lon_min if lon_min >= 0 else lon_min + 360
    lon1 = lon_max if lon_max >= 0 else lon_max + 360
    if "lon" in ds.dims:
        if lon0 < lon1:
            ds = ds.sel(lon=slice(lon0, lon1))
        else:
            # wrap-around (should not happen for Pacific range)
            ds = ds.sel(lon=slice(lon0, 360)).isel(lon=...).combine_first(
                ds.sel(lon=slice(0, lon1)))
    elif "lon" in ds.coords:
        mask = (ds["lon"] >= lon0) & (ds["lon"] <= lon1)
        ds = ds.where(mask, drop=True)

    return ds


def save_output(ds: xr.Dataset, out_path: Path):
    """Write compressed NetCDF4 atomically (temp file then rename)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = out_path.with_suffix(".tmp")
    enc = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
    ds.to_netcdf(temp_path, encoding=enc)
    temp_path.rename(out_path)
    size_mb = out_path.stat().st_size / 1e6
    log.info(f"    ✓ saved  {out_path.name}  ({size_mb:.1f} MB)")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PROCESSING LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def process_one_file(dods_url: str, model: str, var: str,
                     lat_min: float, lat_max: float,
                     lon_min: float, lon_max: float) -> bool:
    """Open a single file over OPeNDAP, subset to box, save output."""
    filename = dods_url.split("/")[-1]
    # Build output path including range info
    out_path = OUTPUT_DIR / model / var / (
        filename.replace(".nc", "") + f"_lat{lat_min}_{lat_max}_lon{lon_min}_{lon_max}.nc"
    )

    if out_path.exists():
        log.info(f"    skip (output exists): {out_path.name}")
        return True

    log.info(f"    opening: {filename}")
    max_retries = 3
    for attempt in range(max_retries):
        try:
            ds = xr.open_dataset(
                dods_url,
                engine="netcdf4",
                decode_times=True,
                decode_timedelta=True,
                chunks={"time": 24}
            )
            ds_sub = subset_box(ds, lat_min, lat_max, lon_min, lon_max)
            with dask.config.set(scheduler="synchronous"):
                ds_sub.load()
            ds.close()
            save_output(ds_sub, out_path)
            return True
        except Exception as e:
            log.warning(f"Attempt {attempt+1} for {filename} failed: {e}")
            if attempt == max_retries - 1:
                log.error(f"    ✗ failed {filename} after {max_retries} attempts")
                return False
            time.sleep(2 ** attempt)


def _process_one_file_safe(url, model, var, lat_min, lat_max, lon_min, lon_max):
    """Wrapper for joblib: returns (url, ok)."""
    ok = process_one_file(url, model, var, lat_min, lat_max, lon_min, lon_max)
    return (url, ok)


def run(models: list[str], vars_wanted: list[str],
        lat_min: float, lat_max: float,
        lon_min: float, lon_max: float,
        dry_run: bool = False,
        n_jobs: int = 1):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "mmleav2-tropical-pipeline/1.0"

    total_files = 0
    processed   = 0
    failed      = 0

    for model in models:
        log.info(f"\n══ Model: {model} ══")
        for var in vars_wanted:
            freq = VAR_FREQ_MAP.get(var)
            if freq is None:
                log.warning(f"  Unknown variable '{var}', skipping")
                continue

            log.info(f"  Variable: {var}  ({freq})")
            urls = get_file_urls(model, var, freq, session)
            if not urls:
                continue

            total_files += len(urls)
            if dry_run:
                for u in urls:
                    log.info(f"    [DRY-RUN] would process: {u.split('/')[-1]}")
                continue

            if n_jobs == 1:
                for url in urls:
                    ok = process_one_file(url, model, var, lat_min, lat_max, lon_min, lon_max)
                    processed += int(ok)
                    failed    += int(not ok)
                    time.sleep(0.2)
            else:
                results = joblib.Parallel(n_jobs=n_jobs, backend="loky")(  # netCDF-C is not thread safe: one process per file
                    joblib.delayed(_process_one_file_safe)(
                        url, model, var, lat_min, lat_max, lon_min, lon_max
                    )
                    for url in urls
                )
                for _, ok in results:
                    processed += int(ok)
                    failed    += int(not ok)

    log.info(
        f"\n{'DRY-RUN ' if dry_run else ''}Done. "
        f"Files found: {total_files} | "
        f"Processed: {processed} | "
        f"Failed: {failed}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    global OUTPUT_DIR
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--models", nargs="+", default=ALL_MODELS, metavar="MODEL",
        help=f"Models to process (default: all {len(ALL_MODELS)}).\n"
             f"Available: {' '.join(ALL_MODELS)}"
    )
    p.add_argument(
        "--vars", nargs="+", default=ALL_VARS, metavar="VAR",
        help=f"Variables (default: all). Available: {' '.join(ALL_VARS)}"
    )
    p.add_argument(
        "--lat-range", nargs=2, type=float, metavar=("LAT_MIN", "LAT_MAX"),
        default=(_LAT_MIN_DEFAULT, _LAT_MAX_DEFAULT),
        help=f"Latitude range (default: {_LAT_MIN_DEFAULT} {_LAT_MAX_DEFAULT} from config.py)."
    )
    p.add_argument(
        "--lon-range", nargs=2, type=float, metavar=("LON_MIN", "LON_MAX"),
        default=(_LON_MIN_DEFAULT, _LON_MAX_DEFAULT),
        help=f"Longitude range 0-360 (default: {_LON_MIN_DEFAULT} {_LON_MAX_DEFAULT} from config.py)."
    )
    p.add_argument(
        "--outdir", default=str(OUTPUT_DIR),
        help=f"Output directory (default: {OUTPUT_DIR})"
    )
    p.add_argument(
        "--n-jobs", type=int, default=1,
        help="Parallel download workers (default: 1; use 4–8 for faster downloads)"
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Crawl the catalog and list files without processing anything"
    )
    args = p.parse_args()

    if args.models == ["all"]:
        args.models = ALL_MODELS

    OUTPUT_DIR = Path(args.outdir)

    lat_min, lat_max = args.lat_range
    # Normalize longitude to 0-360 for subsetting (user can give -80 → 280)
    lon_min = args.lon_range[0] if args.lon_range[0] >= 0 else args.lon_range[0] + 360
    lon_max = args.lon_range[1] if args.lon_range[1] >= 0 else args.lon_range[1] + 360

    bad_models = [m for m in args.models if m not in ALL_MODELS]
    bad_vars   = [v for v in args.vars   if v not in ALL_VARS]
    if bad_models:
        sys.exit(f"Unknown model(s): {bad_models}\nAvailable: {ALL_MODELS}")
    if bad_vars:
        sys.exit(f"Unknown variable(s): {bad_vars}\nAvailable: {ALL_VARS}")

    log.info(f"Models      : {args.models}")
    log.info(f"Vars        : {args.vars}")
    log.info(f"Lat range   : {lat_min:.1f} to {lat_max:.1f} degrees")
    log.info(f"Lon range   : {lon_min:.1f} to {lon_max:.1f} degrees")
    log.info(f"Output      : {OUTPUT_DIR}")
    log.info(f"N-jobs      : {args.n_jobs}")
    log.info(f"Dry-run     : {args.dry_run}")

    run(args.models, args.vars, lat_min, lat_max, lon_min, lon_max,
        dry_run=args.dry_run, n_jobs=args.n_jobs)


if __name__ == "__main__":
    main()