#!/usr/bin/env python3
"""Compute obs-benchmark mean-state and β_H values from preprocessed reanalysis.

Inputs (from preprocess_obs_reanalysis.py, all on MMLEAv2 g025 grid):
  data/processed/reanalysis/obs_tropical_pacific/ts_1979_2025_g025.nc    (COBE-SST2)
  data/processed/reanalysis/obs_tropical_pacific/tauu_1979_2025_g025.nc  (ERA5 avg_iews)
  data/processed/reanalysis/obs_tropical_pacific/zos_1979_2025_g025.nc   (ORAS5 sossheig)
  data/processed/reanalysis/obs_tropical_pacific/z20_1979_2025_g025.nc   (ORAS5 so20chgt)

Pipeline:
  1. Run compute_mean_state() from enso_metrics_ensemble — same boxes as
     models.  Produces monthly time series for SST grad, Z20 tilt, Z20 mean,
     ZOS grad, τ_x central, τ_x gradient.
  2. Annualise per calendar year (same np.add.at groupby as models).
  3. Run compute_bjerknes_feedback_rolling() for β_H with the 30-yr window
     giving rolling-window centres 1994-2010 (47-yr record).
  4. Save:
      data/processed/reanalysis/obs_metrics_1979_2025_g025.nc
         meanstate_<key> (year)
         meanstate_monthly_<key> (time)
         bjerknes_beta (year)

Downstream use: figE/F/G and optional figC/D obs overlays read this file.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from pathlib import Path

import numpy as np
import xarray as xr

# Import shared compute_mean_state / compute_bjerknes_feedback_rolling from
# the model pipeline so obs gets exactly the same box definitions and β_H
# methodology.  Path added to sys.path at module level to cover `-m` and
# direct-script runs equivalently.
_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from enso_metrics_ensemble import (   # noqa: E402
    compute_mean_state,
    compute_bjerknes_feedback_rolling,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OBS_DIR = PROJECT_ROOT / "data" / "processed" / "reanalysis" / "obs_tropical_pacific"
OUT_FILE = PROJECT_ROOT / "data" / "processed" / "reanalysis" / "obs_metrics_1979_2025_g025.nc"
LOG_DIR = PROJECT_ROOT / "logs" / "analysis"


def build_logger() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"compute_obs_metrics_{ts}.log"
    logger = logging.getLogger("compute_obs_metrics")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh = logging.FileHandler(log_path); fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(sh)
    logger.propagate = False
    logger.info("Log: %s", log_path)
    return logger


def annualise_by_calendar_year(arr: xr.DataArray) -> xr.DataArray:
    """Annual mean per calendar year — matches the model pipeline exactly.

    Uses np.add.at grouping rather than xarray .groupby() because the model
    code takes the same route (to avoid pandas-DatetimeIndex dependency in
    the merged-tropics path).  Empty years return NaN.
    """
    t_vals = arr["time"].values
    if np.issubdtype(t_vals.dtype, np.datetime64):
        years = t_vals.astype("datetime64[Y]").astype(int) + 1970
    else:
        years = np.array([int(v.year) for v in t_vals], dtype=int)

    vals = arr.values
    ann_years, inverse = np.unique(years, return_inverse=True)
    ann_sum   = np.zeros(len(ann_years), dtype=np.float64)
    ann_count = np.zeros(len(ann_years), dtype=np.int64)
    finite = np.isfinite(vals)
    np.add.at(ann_sum,   inverse[finite], vals[finite])
    np.add.at(ann_count, inverse[finite], 1)
    with np.errstate(invalid="ignore"):
        ann = np.where(ann_count > 0, ann_sum / ann_count, np.nan)

    return xr.DataArray(ann, dims=["year"], coords={"year": ann_years},
                        name=arr.name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing output file.")
    args = parser.parse_args()

    logger = build_logger()

    if OUT_FILE.exists() and not args.force:
        logger.warning("%s exists — use --force to overwrite", OUT_FILE)
        return 0

    # Load all four preprocessed obs variables
    ts   = xr.open_dataset(OBS_DIR / "ts_1979_2025_g025.nc"  )["ts"]
    tauu = xr.open_dataset(OBS_DIR / "tauu_1979_2025_g025.nc")["tauu"]
    zos  = xr.open_dataset(OBS_DIR / "zos_1979_2025_g025.nc" )["zos"]
    z20  = xr.open_dataset(OBS_DIR / "z20_1979_2025_g025.nc" )["z20"]
    logger.info("loaded: ts=%s  tauu=%s  zos=%s  z20=%s",
                ts.shape, tauu.shape, zos.shape, z20.shape)

    # Mean-state metrics via the shared model function — returns a dict
    # {key: monthly DataArray}.  Expected keys: sst_gradient, z20_tilt,
    # z20_mean, zos_gradient, tauu_central, tauu_gradient.
    ms = compute_mean_state(ts, tauu, z20=z20, zos=zos)
    logger.info("mean-state metrics: %s", sorted(ms.keys()))

    # β_H inputs: deseasonalised monthly τ_x + H_tilt anomalies on Cai 2018
    # boxes with H_tilt = H_E − H_W.  β_H itself is computed downstream
    # as a pooled regression (no 30-yr rolling window).
    bh_inputs = compute_bjerknes_feedback_rolling(tauu, z20, window_years=30)
    if bh_inputs is not None:
        logger.info("β_H inputs: tauu_cai_anom=%d months, h_tilt_cai_anom=%d",
                     bh_inputs["tauu_cai_anom"].size,
                     bh_inputs["h_tilt_cai_anom"].size)
    else:
        logger.warning("β_H inputs returned None")

    # Assemble output dataset
    data_vars: dict[str, tuple] = {}

    # Annualised mean-state (year-indexed)
    for key, arr in ms.items():
        ann = annualise_by_calendar_year(arr)
        data_vars[f"meanstate_{key}"] = (["year"], ann.values)
        # Store monthly too for downstream seasonal analyses (DJF subsetting
        # etc.) — mirrors the model metrics.nc schema.
        data_vars[f"meanstate_monthly_{key}"] = (["time"], arr.values)

    # Store obs τ_x + H_tilt anomalies on the obs time axis
    ref_ms = list(ms.values())[0]
    obs_time = ref_ms["time"].values
    if bh_inputs is not None:
        for src_key, dst_name in (
                ("tauu_cai_anom",   "tauu_cai_anom"),
                ("h_tilt_cai_anom", "h_tilt_cai_anom")):
            src = bh_inputs[src_key]
            try:
                aligned = src.reindex(time=obs_time, method="nearest",
                                       tolerance="31D")
                data_vars[dst_name] = (["time"], aligned.values)
            except Exception:
                # Month-based fallback (time coords may differ by hours)
                try:
                    src_ym = np.array([t.year * 100 + t.month
                                        for t in src["time"].values])
                    dst_ym = np.array([t.year * 100 + t.month
                                        for t in obs_time])
                except AttributeError:
                    st = src["time"].values
                    dt = obs_time
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
                data_vars[dst_name] = (["time"], vals)

    # Coord arrays
    years = annualise_by_calendar_year(ref_ms)["year"].values
    time_vals = obs_time

    out = xr.Dataset(
        data_vars=data_vars,
        coords={"year": years, "time": time_vals},
        attrs={
            "description": "Obs benchmark mean-state + Bjerknes β_H, 1979-2025",
            "source_sst":  "COBE-SST2",
            "source_tauu": "ERA5 mean_eastward_turbulent_surface_stress",
            "source_zos":  "ORAS5 sea_surface_height",
            "source_z20":  "ORAS5 depth_of_20_c_isotherm",
            "target_grid": "MMLEAv2 g025 (2.5° regular, lat -18.75..18.75, lon 111.25..283.75)",
            "created":     dt.datetime.utcnow().isoformat() + "Z",
        },
    )

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    out.to_netcdf(OUT_FILE)
    logger.info("→ %s", OUT_FILE)
    logger.info("   vars: %s", sorted(out.data_vars))
    logger.info("   year range: %d..%d", int(years.min()), int(years.max()))

    # Echo the scalar benchmark numbers for a quick read-out
    logger.info("─── Obs benchmarks (1979-2025 mean) ───")
    for key in ("sst_gradient", "z20_tilt", "z20_mean", "h_star",
                "h_eq_sw_zos", "h_eq_sw_z20",
                "zos_gradient", "tauu_central", "tauu_gradient"):
        v = f"meanstate_{key}"
        if v in out.data_vars:
            arr = out[v].values
            logger.info("   %-15s = %+.4g  (%d years)", key,
                        float(np.nanmean(arr)), int(np.isfinite(arr).sum()))
    if "bjerknes_beta" in out.data_vars:
        bh_arr = out["bjerknes_beta"].values
        finite = np.isfinite(bh_arr)
        if finite.any():
            logger.info("   β_H (rolling mean) = %+.4g  (%d centres)",
                        float(np.nanmean(bh_arr[finite])), int(finite.sum()))

    return 0


if __name__ == "__main__":
    sys.exit(main())
