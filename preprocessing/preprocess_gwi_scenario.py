"""Per-member, per-scenario global warming index from the model's own global-mean tas, reduced on Casper under PBS.

Why this exists. preprocess_gwi_ensemble.py caches one curve per (model, realization) and builds it from every
tas file that carries the realization label: for a member that exists in two scenarios (ACCESS-ESM1-5, CanESM5,
GISS-E2.1-G, IPSL-CM6A-LR, MIROC-ES2L, MPI-ESM1-2-LR, UKESM1-0-LL) the 2015 to 2100 part is a month-by-month
mixture of the SSP3-7.0 and SSP5-8.5 series, smoothed by the polynomial fit. The metrics of the SSP5-8.5 series
were then binned with that blended curve. This script writes data/gwi_cache/<model>/<rid>/gwi_<scenario>.nc,
one curve per scenario from the historical file(s) plus that scenario's file only, single-forcing and unforced
experiments excluded by name, and compares every curve with the cached gwi.nc so that only the ensembles whose
curves moved need their products rebuilt.

Casper does the field means (cdo -fldmean on the native grid, one PBS job per model, account P93300324, queue
casper); lorenz only submits, waits, rsyncs and fits polynomials. Needs an open ssh ControlMaster to Casper.

    python preprocess_gwi_scenario.py --models access_lens canesm5_lens          # default: all models
    python preprocess_gwi_scenario.py --no-submit                                # fit and compare only
"""

import argparse
import logging
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import preprocess_gwi_ensemble as pge  # noqa: E402  (polynomial fit and file reader)
from config import ALL_MODELS, GWI_CACHE_DIR, extract_realization, extract_scenario  # noqa: E402

ROOT = Path("/path/to/enso_impacts")
EXCLUDE = ("picontrol", "pi-control", "1pctco2", "abrupt-4xco2", "lastmillennium", "hist-ghg", "hist-aer", "hist-nat",
           "hist-sol", "hist-volc", "hist-strato3", "hist-totalo3", "hist-co2", "hist-bmb", "hist-1950hc", "amip",
           "esm-ssp")
CASPER_HOST = "casper"
CASPER_BASE = "/glade/campaign/collections/gdex/data/d651039"
CASPER_SCRATCH = "/path/to/scratch/gwi_tas_scenario"
LOCAL_SCRATCH = ROOT / "data" / "gwi_tas_reduced"
PBS_ACCOUNT = "P93300324"
PBS_QUEUE = "casper"
REPORT = ROOT / "results" / "gwi_scenario_check.csv"
log = logging.getLogger("gwi_scenario")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def casper_list(model):
    r = subprocess.run(["ssh", CASPER_HOST, f"ls {CASPER_BASE}/{model}/Amon/tas/*.nc 2>/dev/null"],
                       capture_output=True, text=True, timeout=120)
    return [ln for ln in r.stdout.split("\n") if ln.endswith(".nc") and not any(t in ln.lower() for t in EXCLUDE)]


def casper_submit(model, files):
    """One PBS job per model: global cdo fldmean of tas (degC) for every file not yet on lorenz."""
    local_dir = LOCAL_SCRATCH / model
    local_dir.mkdir(parents=True, exist_ok=True)
    todo = [f for f in files if not (local_dir / Path(f).name).exists()]
    if not todo:
        return None
    remote_dir = f"{CASPER_SCRATCH}/{model}"
    job = f"""#!/bin/bash
#PBS -N gwitas_{model}
#PBS -A {PBS_ACCOUNT}
#PBS -l walltime=02:00:00
#PBS -l select=1:ncpus=4:mem=16GB
#PBS -q {PBS_QUEUE}
#PBS -j oe
#PBS -o {remote_dir}/pbs.log
module load cdo > /dev/null 2>&1
cd {remote_dir}
reduce() {{
    src="$1"; out="{remote_dir}/$(basename "$src")"
    [ -s "$out" ] && return 0
    cdo -s -O -fldmean -subc,273.15 -selname,tas "$src" "$out.tmp" && mv -f "$out.tmp" "$out"
}}
export -f reduce
export remote_dir="{remote_dir}"
xargs -a {remote_dir}/files.txt -P 4 -I{{}} bash -c 'reduce "$@"' _ {{}}
touch {remote_dir}/.done
"""
    listing = "\n".join(todo) + "\n"
    script = (f"mkdir -p {remote_dir} && cat > {remote_dir}/files.txt <<'EOF_LIST'\n{listing}EOF_LIST\n"
              f"cat > {remote_dir}/job.pbs <<'EOF_JOB'\n{job}EOF_JOB\nrm -f {remote_dir}/.done; qsub {remote_dir}/job.pbs")
    r = subprocess.run(["ssh", CASPER_HOST, "bash", "-l"], input=script, capture_output=True, text=True, timeout=300)
    jobid = r.stdout.strip().split("\n")[-1] if r.returncode == 0 and r.stdout.strip() else None
    log.info(f"{model}: {len(todo)} tas files submitted as {jobid}" if jobid else f"{model}: qsub failed: {r.stderr[-200:]}")
    return jobid


def casper_wait(jobids, poll_s=60, max_s=3 * 3600):
    t0 = time.time()
    pending = {m: j for m, j in jobids.items() if j}
    while pending and time.time() - t0 < max_s:
        r = subprocess.run(["ssh", CASPER_HOST, "qstat -u $USER 2>/dev/null | awk '{print $1}'"],
                           capture_output=True, text=True, timeout=120)
        alive = set(x.split(".")[0] for x in r.stdout.split())
        pending = {m: j for m, j in pending.items() if j.split(".")[0] in alive}
        if pending:
            log.info(f"waiting on {len(pending)} PBS jobs: {sorted(pending)}")
            time.sleep(poll_s)
    if pending:
        log.warning(f"gave up waiting on {sorted(pending)}")


def casper_collect(model):
    local_dir = LOCAL_SCRATCH / model
    local_dir.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["rsync", "-rtz", "--partial", "--exclude", "*.tmp", "--exclude", "*.log", "--exclude", "*.pbs",
                        "--exclude", "files.txt", f"{CASPER_HOST}:{CASPER_SCRATCH}/{model}/", f"{local_dir}/"],
                       capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        log.warning(f"{model}: rsync returned {r.returncode}: {r.stderr[-200:]}")
    return sorted(local_dir.glob("*.nc"))


# The polynomial fit sets the 1850 to 1900 mean to zero when the record covers it, else the mean of the fitted
# curve over 1920 to 1949 (flat-capped before the first data month, so a record starting in 1950 is zeroed at its
# 1950 level). To keep every ensemble's index relative to 1850 to 1900, add the observed warming between the two
# baselines: Berkeley Earth annual global mean, 1920 to 1949 minus 1850 to 1900 = 0.168 K, 1945 to 1954 minus
# 1850 to 1900 = 0.216 K (data/obs/berkley_temperature_anom.txt in the enso_asymmetry repo, computed 2026-09-14).
BASELINE_OFFSET_K = {"1920-1949": 0.168, "1945-1954": 0.216}


def _to_1850_1900_baseline(smooth, raw):
    first_year = int(raw.time.dt.year.values[~np.isnan(raw.values)].min())
    if first_year <= 1850:
        smooth.attrs["baseline"] = "1850-1900 mean = 0"
        return smooth
    key = "1920-1949" if first_year <= 1949 else "1945-1954"
    off = BASELINE_OFFSET_K[key]
    out = smooth + off
    out.attrs = dict(smooth.attrs, baseline=f"1850-1900 mean = 0, via the fitted {key} level plus the observed "
                                             f"{key} minus 1850-1900 warming of {off} K (Berkeley Earth)",
                     baseline_offset_K=off, first_data_year=first_year)
    return out


def build(model, files):
    """Fit one curve per (realization, scenario) from the reduced files and compare with the cached gwi.nc."""
    rows = []
    by_rid = defaultdict(list)
    for f in files:
        if any(t in f.name.lower() for t in EXCLUDE):
            continue
        by_rid[extract_realization(f.name)].append(f)
    for rid, fl in sorted(by_rid.items()):
        scen_files = defaultdict(list)
        hist = []
        for f in fl:
            s = extract_scenario(f.name)
            (hist if s == "historical" else scen_files[s]).append(f)
        if not scen_files:                       # historical only: nothing to bin against, skip
            log.info(f"{model} {rid}: historical only, no scenario curve written")
            continue
        old_p = GWI_CACHE_DIR / model / rid / "gwi.nc"
        old = xr.open_dataset(old_p)["gwi_smoothed"] if old_p.exists() else None
        for s, sf in scen_files.items():
            series = [pge.compute_gwi_from_local(p) for p in hist + sf]
            raw = series[0].load().astype(float) if len(series) == 1 else pge._merge_gwi_timeseries(series).load()
            if raw.isnull().any():
                raw = raw.interpolate_na(dim="time", method="linear", limit=5).ffill("time").bfill("time")
            smooth = pge._extrapolate_poly_to_standard_grid(raw)
            smooth.attrs["source"] = "; ".join(p.name for p in hist + sf)
            smooth.attrs["scenario"] = s
            smooth = _to_1850_1900_baseline(smooth, raw)
            out = GWI_CACHE_DIR / model / rid / f"gwi_{s}.nc"
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_suffix(".tmp.nc")
            xr.Dataset({"gwi_smoothed": smooth}).to_netcdf(tmp)
            tmp.rename(out)
            row = dict(model=model, rid=rid, scenario=s, n_hist_files=len(hist), n_scen_files=len(sf),
                       new_2090s=float(smooth.sel(time=slice("2090", "2099")).mean()),
                       n_months_above_3K=int((smooth.values >= 3).sum()))
            if old is not None:
                d = smooth.values - old.values
                row.update(old_2090s=float(old.sel(time=slice("2090", "2099")).mean()), max_abs_diff=float(np.nanmax(np.abs(d))),
                           old_months_above_3K=int((old.values >= 3).sum()))
            rows.append(row)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=ALL_MODELS)
    ap.add_argument("--no-submit", action="store_true", help="skip Casper, fit from the reduced files already on lorenz")
    a = ap.parse_args()
    if not a.no_submit:
        files = {m: casper_list(m) for m in a.models}
        jobs = {m: casper_submit(m, f) for m, f in files.items() if f}
        casper_wait(jobs)
        for m in a.models:
            casper_collect(m)
    rows = []
    for m in a.models:
        rows += build(m, sorted((LOCAL_SCRATCH / m).glob("*.nc")))
    df = pd.DataFrame(rows)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(REPORT, index=False, float_format="%.4f")
    if len(df):
        summ = df.groupby("model").agg(members=("rid", "nunique"), curves=("rid", "size"),
                                       max_abs_diff=("max_abs_diff", "max") if "max_abs_diff" in df else ("rid", "size"))
        log.info("per-model summary (max_abs_diff against the cached gwi.nc, K):\n" + summ.to_string())
    log.info("done")
