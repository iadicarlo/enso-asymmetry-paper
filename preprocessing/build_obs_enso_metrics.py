#!/usr/bin/env python3
"""Build obs-side metrics.nc in the same schema as per-model metrics.nc.

Takes the output of ``preprocess_detect_events_observations.py``:

  results/enso_nonstationarity/observations/regional_timeseries_observations.nc
  results/enso_nonstationarity/observations/gwi_timeseries_observations.nc
  results/enso_nonstationarity/observations/enso_events_observations.csv

and produces a single NetCDF that looks (to the rolling-C_k decomposition
downstream) like a 1-member ensemble.  Variables:

  nino34_std(time)      — COBE-SST2 Niño3.4, GWI-detrended, deseasonalised,
                          standardised with baseline (1980, 2014) — same
                          convention as models.
  phase_indicator(time) — 7-category int8 encoded with _PHASE_INT from
                          enso_metrics_ensemble, built by broadcasting the
                          event catalog over each event's [start_idx, end_idx).
  gwi(time)             — Berkeley GWI, lowpass-polynomial-smoothed, baselined
                          on 1850-1900 per the observational pipeline.

Output: ``data/processed/reanalysis/obs_enso_metrics_1979_2025.nc``

Downstream: figE's ``_compute_contributions_one_member`` can consume this as
a one-member "ensemble" and produce rolling A(t) on the obs record.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OBS_DIR   = PROJECT_ROOT / "results" / "enso_nonstationarity" / "observations"
OUT_FILE  = PROJECT_ROOT / "data" / "processed" / "reanalysis" / "obs_enso_metrics_1979_2025.nc"
LOG_DIR   = PROJECT_ROOT / "logs" / "analysis"

# Same mapping as enso_metrics_ensemble.py lines 665-673.  Keep in sync.
_PHASE_INT = {
    "neutral":          0,
    "moderate_el_nino": 1,
    "strong_el_nino":   2,
    "extreme_el_nino":  3,
    "moderate_la_nina": -1,
    "strong_la_nina":   -2,
    "extreme_la_nina":  -3,
}


def build_logger() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"build_obs_enso_metrics_{ts}.log"
    logger = logging.getLogger("build_obs_enso_metrics")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh = logging.FileHandler(log_path); fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(sh)
    logger.propagate = False
    logger.info("Log: %s", log_path)
    return logger


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    logger = build_logger()

    if OUT_FILE.exists() and not args.force:
        logger.warning("%s exists — use --force to overwrite", OUT_FILE)
        return 0

    # ── Load the preprocessed regional time series + GWI + event catalog ──
    ts_path = OBS_DIR / "regional_timeseries_observations.nc"
    gwi_path = OBS_DIR / "gwi_timeseries_observations.nc"
    ev_path  = OBS_DIR / "enso_events_observations.csv"
    for p in (ts_path, gwi_path, ev_path):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} missing — run scripts/analysis/enso_future/"
                "preprocess_detect_events_observations.py first.")

    ts_ds  = xr.open_dataset(ts_path)
    gwi_ds = xr.open_dataset(gwi_path)
    events = pd.read_csv(ev_path)
    logger.info("loaded: %d months, %d events", ts_ds.time.size, len(events))

    # Alignment check — ts_ds and gwi_ds should share the same time axis.
    if not np.array_equal(ts_ds.time.values, gwi_ds.time.values):
        raise ValueError("ts and gwi time axes disagree")

    time_vals = ts_ds.time.values
    n_times   = len(time_vals)

    # ── Build phase_indicator(time) from the event catalog ──────────────
    # Matches enso_metrics_ensemble.py lines 645-674: neutral initialised,
    # each event writes its phase code into [start_idx, end_idx).
    phase_indicator = ["neutral"] * n_times
    for _, row in events.iterrows():
        phase = row["phase"]
        s = int(row["start_idx"])
        e = min(int(row["end_idx"]), n_times)
        if phase not in _PHASE_INT:
            logger.warning("unknown phase %r (skipping)", phase)
            continue
        for k in range(s, e):
            phase_indicator[k] = phase
    phase_int = np.array([_PHASE_INT[p] for p in phase_indicator], dtype=np.int8)

    # ── Assemble in the "one-member ensemble" schema ───────────────────
    out = xr.Dataset(
        data_vars={
            "nino34_std":      (["time"], ts_ds["nino34_std"].values),
            "phase_indicator": (["time"], phase_int),
            "gwi":             (["time"], gwi_ds["gwi"].values),
        },
        coords={"time": time_vals},
        attrs={
            "description":  "Obs ENSO metrics in per-member schema for "
                            "rolling variance decomposition downstream.",
            "source_sst":   "COBE-SST2 (GWI-detrended via Berkeley 4th-deg polynomial)",
            "source_pr":    "GPCP v2.3 (Niño3 box, baseline (1980,2014))",
            "source_gwi":   "Berkeley Earth monthly anomaly (1850-1900 baseline) "
                            "→ lowpass_filter_gwi(polynomial_degree=4)",
            "pipeline":     "scripts/analysis/enso_future/"
                            "preprocess_detect_events_observations.py",
            "grid":         "Niño boxes averaged from native COBE / GPCP grids",
            "time_range":   f"{pd.Timestamp(time_vals[0]).date()} to "
                            f"{pd.Timestamp(time_vals[-1]).date()}",
            "created":      dt.datetime.utcnow().isoformat() + "Z",
        },
    )

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    out.to_netcdf(OUT_FILE)
    logger.info("→ %s", OUT_FILE)

    # Quick sanity summary
    logger.info("phase breakdown (months):")
    for p, code in _PHASE_INT.items():
        n = int((phase_int == code).sum())
        logger.info("  %-20s (code %+d) : %4d months", p, code, n)
    logger.info("nino34_std range: %+.2f .. %+.2f",
                float(np.nanmin(out.nino34_std.values)),
                float(np.nanmax(out.nino34_std.values)))
    logger.info("gwi range:        %+.2f .. %+.2f",
                float(np.nanmin(out.gwi.values)),
                float(np.nanmax(out.gwi.values)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
