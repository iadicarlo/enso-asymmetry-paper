"""Global tropical-mean SST per member: the cosine-weighted ocean mean over 20S to 20N and all longitudes, monthly.

This is the quantity the relative Nino index subtracts (L'Heureux et al. 2024). The local MMLEAv2 subsets only cover
110E to 285E, so the band is read from the global files on RDA THREDDS through OPeNDAP, one file at a time, the
way preprocess_gwi_ensemble.py reads tas, or (default) reduced on Casper with cdo and rsynced back as in the GWI
job's Casper route, which needs an open ssh ControlMaster session to Casper. Single-forcing and unforced
experiments are excluded by name.

Output: results/enso_metrics/trop_mean_global/<model>_<scenario>_trop_mean_global.nc, dims (ens, time), degC,
historical and the scenario concatenated per realization, matching the layout of the metrics files.

    python preprocess_trop_mean_tos.py --models cesm2_lens mpi_lens --n-jobs 4
"""

import argparse
import concurrent.futures
import logging
import re
import shlex
import subprocess
import sys
import time
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ALL_MODELS, extract_realization, extract_scenario  # noqa: E402

ROOT = Path("/path/to/enso_impacts")
OUT = ROOT / "results/enso_metrics/trop_mean_global"
OUT.mkdir(parents=True, exist_ok=True)
RDA_DATASET_ID = "d651039"
CATALOG_BASE = f"https://thredds.rda.ucar.edu/thredds/catalog/files/{RDA_DATASET_ID}"
EXCLUDE = ("picontrol", "pi-control", "1pctco2", "abrupt-4xco2", "lastmillennium", "hist-ghg", "hist-aer", "hist-nat",
           "hist-sol", "hist-volc", "hist-strato3", "hist-totalo3", "hist-co2", "hist-bmb", "hist-1950hc", "amip")
BAND = (-20.0, 20.0)
CASPER_HOST = "casper"                                   # ssh alias on lorenz; needs an open ControlMaster session (2FA)
CASPER_BASES = ("/glade/campaign/collections/gdex/data/d651039", "/glade/campaign/cgd/cas/nmaher")
CASPER_SCRATCH = "/path/to/scratch/trop_mean_tos"
MASK_SOURCE = f"{CASPER_BASES[0]}/access_lens/Omon/tos/tos_Omon_ACCESS-ESM1-5_historical_r10i1p1f1_g025_185001-201412.nc"
LOCAL_SCRATCH = ROOT / "data" / "trop_mean_reduced"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import time_fixes  # noqa: E402  MMLEAv2 E3SMv1 time axis one month early (2026-09-28)
log = logging.getLogger("trop_mean")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def list_nc_files(catalog_url, session):
    """OPeNDAP URLs of the .nc files in one THREDDS catalog page (same parser as the GWI job)."""
    try:
        resp = session.get(catalog_url, timeout=60)
        resp.raise_for_status()
    except Exception as e:
        log.warning(f"catalog fetch failed: {catalog_url} ({e})")
        return []

    class P(HTMLParser):
        def __init__(self):
            super().__init__(); self.links = []

        def handle_starttag(self, tag, attrs):
            if tag == "a":
                for k, v in attrs:
                    if k == "href" and v and f"dataset=files/{RDA_DATASET_ID}" in v:
                        self.links.append(v)

    p = P(); p.feed(resp.text)
    urls = []
    for href in p.links:
        path = href.split("dataset=")[1]
        if path.endswith(".nc") and not any(t in path.lower() for t in EXCLUDE):
            urls.append(f"https://thredds.rda.ucar.edu/thredds/dodsC/{path}")
    return sorted(set(urls))


def band_mean_from_url(url, retries=3, timeout_s=900):
    """Monthly cosine-weighted ocean mean of the 20S to 20N band of one file, in degC."""

    def _fetch():
        ds = time_fixes.fix_month_offset(xr.open_dataset(url, engine="netcdf4", decode_times=True, decode_timedelta=True), url)
        var = "tos" if "tos" in ds else ("ts" if "ts" in ds else None)
        if var is None:
            raise RuntimeError(f"no tos/ts in {url}")
        da = ds[var]
        latn = "lat" if "lat" in da.coords else ("latitude" if "latitude" in da.coords else None)
        lonn = "lon" if "lon" in da.coords else ("longitude" if "longitude" in da.coords else None)
        if latn and da[latn].ndim == 1:
            lat = da[latn]
            sub = da.sel({latn: slice(BAND[1], BAND[0]) if float(lat[0]) > float(lat[-1]) else slice(BAND[0], BAND[1])})
            w = np.cos(np.deg2rad(sub[latn]))
            out = sub.weighted(w).mean([latn, lonn], skipna=True).load()
        else:                                            # curvilinear grid: mask on the 2D latitude
            lat2 = da["lat"] if "lat" in da.coords else da["latitude"]
            m = (lat2 >= BAND[0]) & (lat2 <= BAND[1])
            w = np.cos(np.deg2rad(lat2)).where(m)
            dims = [d for d in da.dims if d != "time"]
            out = (da * w).sum(dims, skipna=True) / w.where(np.isfinite(da)).sum(dims)
            out = out.load()
        if float(out.max()) > 200:
            out = out - 273.15
        idx = out.indexes.get("time")
        if idx is not None and hasattr(idx, "to_datetimeindex"):
            out["time"] = idx.to_datetimeindex()
        return out.astype("float32")

    for attempt in range(retries):
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                return ex.submit(_fetch).result(timeout=timeout_s)
        except concurrent.futures.TimeoutError:
            log.warning(f"attempt {attempt + 1} timed out for {url}")
        except Exception as e:
            log.warning(f"attempt {attempt + 1} failed for {url}: {str(e)[:120]}")
        time.sleep(3 * 2**attempt)
    raise RuntimeError(f"all attempts failed for {url}")


def merge(series):
    """Concatenate segments, historical first, and keep the first value at duplicated months."""
    series = [s for s in series if s is not None and s.size]
    if not series:
        return None
    s = xr.concat(series, "time")
    s = s.assign_coords(time=pd.DatetimeIndex(s.time.values).to_period("M").to_timestamp())
    _, first = np.unique(s.time.values, return_index=True)
    return s.isel(time=np.sort(first)).sortby("time")


def casper_list(model):
    """Casper paths of the model's global tos (else ts) files, DAMIP and unforced runs excluded."""
    for base in CASPER_BASES:
        for var in ("Omon/tos", "Amon/ts"):
            r = subprocess.run(["ssh", CASPER_HOST, f"ls {base}/{model}/{var}/*.nc 2>/dev/null"], capture_output=True, text=True, timeout=120)
            files = [ln for ln in r.stdout.split("\n") if ln.endswith(".nc") and not any(t in ln.lower() for t in EXCLUDE)]
            if files:
                return files
    return []


PBS_ACCOUNT = "P93300324"
PBS_QUEUE = "casper"


def casper_submit(model, files):
    """Submit one PBS job on Casper that reduces every file of the model with cdo in parallel; returns the job id.

    fldmean over the 20S to 20N band uses the grid-cell areas and skips missing values, so it is the
    cosine-weighted ocean mean the relative index needs. Files already reduced on Casper are skipped by the
    job itself, files already on lorenz are not submitted at all, so the whole step is resumable.
    """
    local_dir = LOCAL_SCRATCH / model
    local_dir.mkdir(parents=True, exist_ok=True)
    todo = [f for f in files if not (local_dir / Path(f).name).exists()]
    if not todo:
        return None
    remote_dir = f"{CASPER_SCRATCH}/{model}"
    listing = "\n".join(todo) + "\n"
    job = f"""#!/bin/bash
#PBS -N tropmean_{model}
#PBS -A {PBS_ACCOUNT}
#PBS -l walltime=02:00:00
#PBS -l select=1:ncpus=4:mem=16GB
#PBS -q {PBS_QUEUE}
#PBS -j oe
#PBS -o {remote_dir}/pbs.log
module load cdo > /dev/null 2>&1
cd {remote_dir}
# ocean mask on the common 2.5 degree grid, from a tos file (missing over land): ts files carry land skin
# temperature, which must not enter the tropical mean (platform rule: mask land before the tropical mean)
mask="{CASPER_SCRATCH}/ocean_mask_g025.nc"
# classic netCDF and no HDF5 file locking: four cdo processes read the mask at once on the parallel file system
[ -s "$mask" ] || cdo -s -O -f nc -setrtoc,-1e30,1e30,1 -seltimestep,1 -selname,tos "{MASK_SOURCE}" "$mask"
export mask
export HDF5_USE_FILE_LOCKING=FALSE
reduce() {{
    src="$1"; out="{remote_dir}/$(basename "$src")"
    case "$src" in *tos*) var=tos; pre="" ;; *) var=ts; pre="-ifthen $mask" ;; esac
    [ -s "$out" ] && return 0
    cdo -s -O -fldmean -sellonlatbox,0,360,{BAND[0]},{BAND[1]} $pre -selname,$var "$src" "$out.tmp" && mv -f "$out.tmp" "$out"
}}
export -f reduce
export remote_dir="{remote_dir}"
xargs -a {remote_dir}/files.txt -P 4 -I{{}} bash -c 'reduce "$@"' _ {{}}
touch {remote_dir}/.done
"""
    script = f"mkdir -p {remote_dir} && cat > {remote_dir}/files.txt <<'EOF_LIST'\n{listing}EOF_LIST\ncat > {remote_dir}/job.pbs <<'EOF_JOB'\n{job}EOF_JOB\nrm -f {remote_dir}/.done; qsub {remote_dir}/job.pbs"
    r = subprocess.run(["ssh", CASPER_HOST, "bash", "-l"], input=script, capture_output=True, text=True, timeout=300)
    jobid = r.stdout.strip().split("\n")[-1] if r.returncode == 0 and r.stdout.strip() else None
    log.info(f"{model}: {len(todo)} files submitted to PBS as {jobid}" if jobid else f"{model}: qsub failed: {r.stderr[-200:]}")
    return jobid


def casper_wait(jobids, poll_s=60, max_s=3 * 3600):
    """Block until every submitted job has written its .done marker (or has left the queue after being seen).

    qstat can lag a few seconds behind qsub, so a job must have been seen in the queue at least once before its
    absence counts as completion; the marker is the primary signal. A job never seen and without a marker after
    ten minutes is reported and dropped.
    """
    t0 = time.time()
    pending = {m: j.split(".")[0] for m, j in jobids.items() if j}
    seen = set()
    while pending and time.time() - t0 < max_s:
        cmd = ("qstat -u $USER 2>/dev/null | awk '{print $1}'; echo ---; for d in "
               + " ".join(f"{CASPER_SCRATCH}/{m}" for m in pending) + "; do [ -e $d/.done ] && basename $d; done")
        r = subprocess.run(["ssh", CASPER_HOST, cmd], capture_output=True, text=True, timeout=120)
        q, _, done = r.stdout.partition("---")
        alive = set(x.split(".")[0] for x in q.split())
        finished = set(done.split())
        for m, jid in list(pending.items()):
            if jid in alive:
                seen.add(m)
            if m in finished or (m in seen and jid not in alive):
                del pending[m]
            elif m not in seen and time.time() - t0 > 600:
                log.warning(f"{m}: job {jid} never appeared in the queue and left no marker; giving up on it")
                del pending[m]
        if pending:
            log.info(f"waiting on {len(pending)} PBS jobs: {sorted(pending)}")
            time.sleep(poll_s)
    if pending:
        log.warning(f"gave up waiting on {sorted(pending)}")


def casper_collect(model, files):
    """Pull the reduced files of a model back to lorenz; returns {source path: local path} for those present."""
    local_dir = LOCAL_SCRATCH / model
    local_dir.mkdir(parents=True, exist_ok=True)
    remote_dir = f"{CASPER_SCRATCH}/{model}"
    r2 = subprocess.run(["rsync", "-rtz", "--partial", "--exclude", "*.tmp", f"{CASPER_HOST}:{remote_dir}/", f"{local_dir}/"], capture_output=True, text=True, timeout=1800)
    if r2.returncode != 0:
        log.warning(f"{model}: rsync returned {r2.returncode}: {r2.stderr[-200:]}")
    have = [f for f in files if (local_dir / Path(f).name).exists()]
    log.info(f"{model}: {len(have)} of {len(files)} files reduced")
    return {f: local_dir / Path(f).name for f in have}


def band_mean_from_local(path):
    ds = time_fixes.fix_month_offset(xr.open_dataset(path, decode_times=True, decode_timedelta=True), path)
    var = "tos" if "tos" in ds else "ts"
    out = ds[var].squeeze(drop=True).load()
    if float(out.max()) > 200:
        out = out - 273.15
    idx = out.indexes.get("time")
    if idx is not None and hasattr(idx, "to_datetimeindex"):
        out["time"] = idx.to_datetimeindex()
    return out.astype("float32")


def process_model(model, n_jobs, source="casper", files=None):
    if source == "casper":
        files = files or casper_list(model)
        if not files:
            log.error(f"{model}: no tos or ts files on Casper"); return
        reduced = casper_collect(model, files)
        urls = list(reduced)
        reader = lambda u: band_mean_from_local(reduced[u])  # noqa: E731
    else:
        session = requests.Session()
        urls = list_nc_files(f"{CATALOG_BASE}/{model}/Omon/tos/catalog.html", session)
        if not urls:
            urls = list_nc_files(f"{CATALOG_BASE}/{model}/Amon/ts/catalog.html", session)
        if not urls:
            log.error(f"{model}: no tos or ts files on RDA"); return
        reader = band_mean_from_url
    groups = {}
    for u in urls:
        name = Path(u).name
        groups.setdefault((extract_scenario(name), extract_realization(name)), []).append(u)
    scenarios = sorted({s for s, _ in groups if s != "historical"})
    log.info(f"{model}: {len(urls)} files, scenarios {scenarios}, realizations {len({r for _, r in groups})}")
    for scen in scenarios:
        out_file = OUT / f"{model}_{scen}_trop_mean_global.nc"
        if out_file.exists():
            log.info(f"{model} {scen}: exists, skipping"); continue
        rids = sorted(r for s, r in groups if s == scen)

        def one(rid):
            try:
                segs = [reader(u) for u in groups.get(("historical", rid), [])] + [reader(u) for u in groups[(scen, rid)]]
                return rid, merge(segs)
            except Exception as e:                       # one bad member must not sink the model
                log.warning(f"{model} {scen} {rid}: skipped ({str(e)[:120]})")
                return rid, None

        results = Parallel(n_jobs=n_jobs)(delayed(one)(r) for r in rids)
        results = [(r, s) for r, s in results if s is not None]
        if not results:
            log.error(f"{model} {scen}: nothing usable"); continue
        times = sorted(set().union(*[set(s.time.values) for _, s in results]))
        arr = np.full((len(results), len(times)), np.nan, "float32")
        tindex = pd.DatetimeIndex(times)
        for i, (_, s) in enumerate(results):
            arr[i, tindex.get_indexer(pd.DatetimeIndex(s.time.values))] = s.values
        ds = xr.Dataset({"trop_mean_sst": (("ens", "time"), arr)}, coords={"ens": [r for r, _ in results], "time": tindex},
                        attrs={"model": model, "scenario": scen, "definition": "cosine-weighted ocean mean of monthly SST over 20S to 20N, all longitudes, degC (ts files masked to ocean with the g025 tos land-sea mask)",
                               "source": f"{'Casper GLADE copy of RDA ' + RDA_DATASET_ID + ' reduced with cdo fldmean under PBS' if source == 'casper' else 'RDA ' + RDA_DATASET_ID + ' via OPeNDAP'}, historical plus {scen} per realization, DAMIP and piControl excluded",
                               "script": "preprocess_trop_mean_tos.py"})
        tmp = out_file.with_suffix(".tmp.nc"); ds.to_netcdf(tmp); tmp.rename(out_file)
        log.info(f"{model} {scen}: wrote {out_file.name} ({len(results)} members, {len(times)} months)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=ALL_MODELS)
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--source", choices=("casper", "rda"), default="casper")
    a = ap.parse_args()
    if a.source == "casper":
        # phase 1: one PBS job per model on Casper; phase 2: wait; phase 3: pull back and merge on lorenz
        files = {m: casper_list(m) for m in a.models}
        jobs = {m: casper_submit(m, f) for m, f in files.items() if f}
        casper_wait(jobs)
        for m in a.models:
            process_model(m, a.n_jobs, "casper", files.get(m))
    else:
        for m in a.models:
            process_model(m, a.n_jobs, a.source)
    log.info("done")
