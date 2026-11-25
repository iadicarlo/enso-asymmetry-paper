"""Compute extra per-member monthly variables for the ENSO-SMILEs pipeline.

Skips the expensive parts of ``enso_metrics_ensemble.py`` (event detection,
variance decomposition, mean-state annualisation, GWI loading).  Only
computes the new variables needed for rainfall-variability / atmospheric-
noise diagnostics.

## Stored variables

**Variability-related fields** — **STANDARDISED** by each model's own
baseline (1980–2014) σ, dimensionless.  Convention matches the existing
``nino34_std`` / ``nino3_std`` / ``nino4_std`` in metrics.nc so Δσ of
these is directly cross-model comparable:

  * ``nino3_pr_std``    — Niño-3 precipitation standardised anomaly (dimensionless)
  * ``nino4_pr_std``    — Niño-4 precipitation standardised anomaly (dimensionless)

**Physical-units fields** — **UNSTANDARDISED**, needed for regressions
against SST (atmospheric noise) where units must be preserved:

  * ``nino3_tauu_anom`` — Niño-3 τ_x anomaly (N/m²)
  * ``nino4_tauu_anom`` — Niño-4 τ_x anomaly (N/m²)
  * ``eps_noise_rms``   — atmospheric noise ε_rms(t) from Philip &
                          van Oldenborgh 2009 — regress τ_x(x,y,t) on
                          deseasonalised Niño-3 and Niño-4 SST anomalies;
                          residual spatially RMS'd over 130–280°E, 5°S–5°N.
                          Units: N/m².

SST box-mean anomalies are NOT stored — ``nino3_std`` and ``nino4_std``
are already in metrics.nc and carry the standardised form.

Output: sidecar NetCDF at
    results/enso_metrics/{scenario}/{model}_extra_vars.nc
matching the ``ens``/``time`` schema of the existing ``metrics.nc``.  No
modification of the existing file.

Usage:
    python enso_metrics_extra_vars.py                 # all 19 models × all scenarios
    python enso_metrics_extra_vars.py --models cesm2_lens
    python enso_metrics_extra_vars.py --models cesm2_lens ipsl_cm6a_lens --n-jobs 8

The main SMILEs figures can then xr.merge the two files.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import functions as fns  # type: ignore
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ALL_MODELS, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX

# Re-use a subset of the loader logic from enso_metrics_ensemble.py.
from enso_metrics_ensemble import (
    get_model_files, _files_for_member_scenario, _load_and_merge_var,
    TROPICS_DATA_DIR, OUTPUT_DIR, BASELINE,
)

_log = logging.getLogger("enso_metrics_extra_vars")

_NINO3_BOX = dict(lat=slice(-5, 5), lon=slice(210, 270))
_NINO4_BOX = dict(lat=slice(-5, 5), lon=slice(160, 210))
_NOISE_BOX = dict(lat=slice(-5, 5), lon=slice(130, 280))


def _deseas_box(field: xr.DataArray, box: dict,
                 return_key: str = "anomaly") -> xr.DataArray:
    """Box-mean then deseasonalise (30-yr rolling, baseline=BASELINE).

    return_key='anomaly' → deseasonalised in source units (mm/day, K, N/m²).
    return_key='standardized' → also divided by baseline-period σ → dimensionless.
    """
    bm = field.sel(**box).mean(["lat", "lon"])
    return fns.remove_seasonal_cycle_rolling_with_std(
        bm, window_years=30, rolling_std=False, baseline_period=BASELINE
    )[return_key]


def _atmos_noise_rms(tauu: xr.DataArray,
                      nino3_sst_anom: xr.DataArray,
                      nino4_sst_anom: xr.DataArray) -> xr.DataArray:
    """Philip & van Oldenborgh 2009 atmospheric-noise ε_rms(t).

    Regress deseasonalised 2D τ_x(x,y,t) on Niño-3 and Niño-4 SST anomalies
    over the historical baseline (BASELINE).  Residual ε(x,y,t) is spatially
    averaged (RMS) over 130–280°E, 5°S–5°N to give a monthly scalar noise
    magnitude.  Returns NaN series if tauu is missing.
    """
    if tauu is None:
        n = len(nino3_sst_anom.time)
        return xr.DataArray(np.full(n, np.nan),
                             coords={"time": nino3_sst_anom.time}, dims=["time"])
    tauu_box = tauu.sel(**_NOISE_BOX)
    # 2D monthly anomaly relative to baseline-period climatology
    clim = tauu_box.sel(time=slice(str(BASELINE[0]),
                                    str(BASELINE[1]))).groupby("time.month").mean("time")
    tauu_anom = tauu_box.groupby("time.month") - clim
    t_tauu = tauu_anom.time.values
    bl_mask = (tauu_anom.time.dt.year >= BASELINE[0]) & \
              (tauu_anom.time.dt.year <= BASELINE[1])

    def _pattern(T):
        T_al = T.reindex(time=t_tauu, method="nearest", tolerance="31D").values
        tauu_b = tauu_anom.values[bl_mask.values]
        T_b    = T_al[bl_mask.values]
        if T_b.size < 60 or np.all(~np.isfinite(T_b)):
            return np.full(tauu_anom.shape[1:], np.nan)
        T_c  = T_b - np.nanmean(T_b)
        varT = np.nanvar(T_c, ddof=1)
        if varT <= 0:
            return np.full(tauu_anom.shape[1:], np.nan)
        # Broadcast-friendly: tauu_b shape (T, lat, lon), T_c shape (T,)
        tauu_c = tauu_b - np.nanmean(tauu_b, axis=0, keepdims=True)
        cov = np.nanmean(tauu_c * T_c[:, None, None], axis=0)
        return cov / varT

    A1 = _pattern(nino3_sst_anom)
    A2 = _pattern(nino4_sst_anom)
    T1 = nino3_sst_anom.reindex(time=t_tauu, method="nearest",
                                 tolerance="31D").values
    T2 = nino4_sst_anom.reindex(time=t_tauu, method="nearest",
                                 tolerance="31D").values
    eps = (tauu_anom.values
            - A1[None, :, :] * T1[:, None, None]
            - A2[None, :, :] * T2[:, None, None])
    eps_rms_t = np.sqrt(np.nanmean(eps ** 2, axis=(1, 2)))
    return xr.DataArray(eps_rms_t, coords={"time": t_tauu}, dims=["time"])


def _ensure_pd_datetime_coord(da: xr.DataArray) -> xr.DataArray:
    """Convert cftime time coordinate to pandas/numpy datetime64 if needed.

    Some MMLEAv2 SMILEs (cesm1_lens, e3sm_v2_lens, gfdl_cm3_lens use noleap
    calendar) return cftime.DatetimeNoLeap coords. Mixing cftime with
    pandas Timestamp later in the pipeline (GWI alignment, seasonal-cycle
    .sel(time=slice(...))) raises 'different calendars' or 'timedelta vs
    str' comparison errors.

    Fix: convert cftime → numpy datetime64 via string roundtrip (drops day
    granularity beyond month, which is fine for monthly data).  Keeps any
    already-datetime64 coord unchanged.
    """
    if "time" not in da.coords:
        return da
    t0 = da.time.values[0]
    # cftime objects expose .calendar / are subclass of cftime.datetime.
    # Detect by attribute (cftime not always importable).
    if hasattr(t0, "calendar") or t0.__class__.__module__.startswith("cftime"):
        import pandas as _pd
        new_t = _pd.to_datetime(
            [f"{t.year:04d}-{t.month:02d}-{t.day:02d}" for t in da.time.values]
        )
        return da.assign_coords(time=new_t.values)
    return da


def _load_raw_field(model: str, realization: str, scenario: str,
                     var: str, lat_min: float, lat_max: float,
                     lon_min: float, lon_max: float
                     ) -> xr.DataArray | None:
    """Load raw 2D field for this (model, realisation, scenario). Returns the
    DataArray merged across hist+future file segments, or None if missing.

    Re-uses ``enso_metrics_ensemble.get_model_files`` + ``_files_for_member_scenario``
    + ``_load_and_merge_var`` — same filter chain that the full pipeline uses.

    Variable fallbacks (matching enso_metrics_ensemble.py:113-114):
      tos → ts (e3sm_v2_lens lacks `tos`, has surface `ts`)
    Future runs on any model with the same convention will work.

    cftime → pd.datetime conversion applied at return so downstream
    operations (GWI alignment, seasonal-cycle slicing) work for noleap
    calendars (cesm1_lens, e3sm_v2_lens, gfdl_cm3_lens).
    """
    # Variable fallback chain: try the requested var first, then known
    # alternatives.  Matches upstream enso_metrics_ensemble.py logic so
    # the extra_vars sidecar uses the same physical fields as the main
    # metrics build.
    if var == "tos":
        var_chain = ("tos", "ts")
    else:
        var_chain = (var,)

    for v_try in var_chain:
        all_files = get_model_files(model, v_try, lat_min, lat_max, lon_min, lon_max)
        if not all_files:
            continue
        files = _files_for_member_scenario(all_files, realization, scenario)
        if not files:
            continue
        try:
            ds = _load_and_merge_var(files)
            if ds is None or v_try not in ds.data_vars:
                continue
            if v_try != var:
                _log.info(f"  {model} {realization} [{scenario}]: "
                            f"loaded {v_try} as fallback for {var}")
            return _ensure_pd_datetime_coord(ds[v_try])
        except Exception as e:
            _log.warning(f"  {model} {realization} [{scenario}]: "
                          f"load {v_try} failed: {e}")
            continue
    return None


def process_member(model: str, realization: str, scenario: str,
                    sst_var: str, pr_var: str, tauu_var: str,
                    lat_min: float, lat_max: float,
                    lon_min: float, lon_max: float,
                    gwi_da: xr.DataArray | None = None) -> dict | None:
    """Compute the six box-anomalies + ε_rms for one (model, realisation, scenario).
    Returns a dict of (var_name → xr.DataArray on time) or None on failure.

    If ``gwi_da`` is provided (1-D time series for this realisation), precipitation
    is GWI-detrended (per-pixel residuals via remove_gwi_residuals_vectorized)
    BEFORE box-mean and deseasonalisation, matching the convention used by
    preprocess_detect_events.py for CESM2 SMILE precipitation. Output variables
    nino{3,4}_pr_anom (mm/day, GWI-detrended) are the paper-rigorous version.
    The legacy nino{3,4}_pr_std (standardised, NOT GWI-detrended) is retained
    for back-compat with existing downstream consumers.
    """
    sst  = _load_raw_field(model, realization, scenario, sst_var,
                            lat_min, lat_max, lon_min, lon_max)
    if sst is None:
        _log.warning(f"  {model} {realization} [{scenario}]: no SST — skipping")
        return None
    pr   = _load_raw_field(model, realization, scenario, pr_var,
                            lat_min, lat_max, lon_min, lon_max)
    if pr is None:
        _log.warning(f"  {model} {realization} [{scenario}]: no PR — skipping")
        return None
    tauu = _load_raw_field(model, realization, scenario, tauu_var,
                            lat_min, lat_max, lon_min, lon_max)
    # tauu is optional; noise → NaN if missing

    try:
        # GWI-detrended precip path (paper-rigorous; per user 2026-04-30
        # NO FALLBACK to non-detrended pr — fail loudly so we don't silently
        # contaminate the SMILE precip with the GWI trend).  We align GWI to
        # pr.time via year*100+month integer keys (handles cftime/datetime
        # calendar mismatches that broke the previous reindex).
        if gwi_da is None:
            raise RuntimeError("gwi_da is None — refusing to skip GWI-detrend")

        # Robust time alignment: build a GWI DataArray on pr.time using
        # year*100+month integer matching (same trick as _align_to_time
        # but returning a 1-D DataArray suitable for the per-pixel regress).
        def _ym_key(t):
            if hasattr(t, "year") and hasattr(t, "month"):
                return int(t.year) * 100 + int(t.month)
            try:
                import pandas as pd
                pt = pd.Timestamp(t)
                return int(pt.year) * 100 + int(pt.month)
            except Exception:
                return None
        gwi_keys = np.array([_ym_key(t) for t in gwi_da.time.values])
        pr_keys  = np.array([_ym_key(t) for t in pr.time.values])
        # Map each pr month to gwi index
        idx = np.full(pr_keys.size, -1, dtype=np.int64)
        gwi_lookup = {int(k): i for i, k in enumerate(gwi_keys) if k is not None}
        for j, k in enumerate(pr_keys):
            if k is not None:
                idx[j] = gwi_lookup.get(int(k), -1)
        n_miss = int((idx < 0).sum())
        if n_miss > 0:
            raise RuntimeError(f"GWI alignment: {n_miss}/{pr_keys.size} pr "
                                  f"months have no matching GWI month")
        gwi_aligned = xr.DataArray(gwi_da.values[idx],
                                       coords={"time": pr.time.values},
                                       dims=["time"])
        pr_detrended = fns.remove_gwi_residuals_vectorized(pr, gwi_aligned)
        # mm/day GWI-detrended deseasonalised anomaly (paper-rigorous).
        nino3_pr_anom = _deseas_box(pr_detrended, _NINO3_BOX, return_key="anomaly")
        nino4_pr_anom = _deseas_box(pr_detrended, _NINO4_BOX, return_key="anomaly")
        # Legacy standardised (NOT GWI-detrended) — kept for back-compat.
        nino3_pr_std = _deseas_box(pr, _NINO3_BOX, return_key="standardized")
        nino4_pr_std = _deseas_box(pr, _NINO4_BOX, return_key="standardized")
        # τ_x: physical units (N/m²) — needed for atmospheric-noise regression.
        if tauu is not None:
            nino3_tauu_anom = _deseas_box(tauu, _NINO3_BOX)
            nino4_tauu_anom = _deseas_box(tauu, _NINO4_BOX)
        else:
            nino3_tauu_anom = xr.full_like(nino3_pr_std, np.nan)
            nino4_tauu_anom = xr.full_like(nino4_pr_std, np.nan)

        # For atmospheric noise: regress 2D τ_x(x,y,t) on Niño-3/4 SST.
        # Use deseasonalised-but-unstandardised SST (physical K) for
        # regression so the residual ε is in N/m².
        nino3_sst_anom = _deseas_box(sst, _NINO3_BOX)
        nino4_sst_anom = _deseas_box(sst, _NINO4_BOX)
        eps_rms = _atmos_noise_rms(tauu, nino3_sst_anom, nino4_sst_anom)
    except Exception as e:
        _log.warning(f"  {model} {realization} [{scenario}]: box anomalies "
                      f"failed ({e}) — skipping")
        return None

    return dict(
        realization=realization,
        nino3_pr_anom=nino3_pr_anom,   # GWI-detrended + deseas, mm/day (NEW 2026-04-30)
        nino4_pr_anom=nino4_pr_anom,   # GWI-detrended + deseas, mm/day (NEW 2026-04-30)
        nino3_pr_std=nino3_pr_std,     # legacy: deseas + standardised, NOT GWI-detrended
        nino4_pr_std=nino4_pr_std,     # legacy
        nino3_tauu_anom=nino3_tauu_anom,
        nino4_tauu_anom=nino4_tauu_anom,
        eps_noise_rms=eps_rms,
    )


def _align_to_time(da: xr.DataArray, time_coord) -> np.ndarray:
    """Nearest-neighbour reindex onto target time axis; NaN-fill on miss.

    Uses year*100+month integer keys to handle cftime ↔ datetime64 calendar
    mismatches that make xarray.reindex fail silently.
    """
    arr = np.full(len(time_coord), np.nan, dtype=np.float64)
    src_times = da["time"].values
    dst_times = time_coord.values

    def _ym(t):
        """Extract (year, month) from either cftime or numpy.datetime64."""
        if hasattr(t, "year") and hasattr(t, "month"):
            return int(t.year), int(t.month)
        # numpy.datetime64
        try:
            import pandas as pd
            pt = pd.Timestamp(t)
            return int(pt.year), int(pt.month)
        except Exception:
            return None

    src_lookup = {}
    for i, t in enumerate(src_times):
        k = _ym(t)
        if k is not None:
            src_lookup[k[0] * 100 + k[1]] = i

    src_vals = da.values
    for j, t in enumerate(dst_times):
        k = _ym(t)
        if k is None:
            continue
        i = src_lookup.get(k[0] * 100 + k[1])
        if i is not None and i < len(src_vals):
            arr[j] = float(src_vals[i])
    return arr


def build_sidecar_for_model(model: str, scenario: str, results_dir: Path,
                             sst_var: str = "tos", pr_var: str = "pr",
                             tauu_var: str = "tauu",
                             lat_min: float = LAT_MIN, lat_max: float = LAT_MAX,
                             lon_min: float = LON_MIN, lon_max: float = LON_MAX,
                             n_jobs: int = 1) -> None:
    """Build the sidecar NetCDF for one (model, scenario)."""
    metrics_path = results_dir / scenario / f"{model}_metrics.nc"
    if not metrics_path.exists():
        _log.warning(f"{model} [{scenario}]: no metrics.nc — skipping")
        return
    out_path = results_dir / scenario / f"{model}_extra_vars.nc"

    # Open existing metrics.nc to get the ens/time schema + per-realisation GWI
    ref = xr.open_dataset(metrics_path)
    ens_ids = list(ref.coords["ens"].values) if "ens" in ref.coords else []
    time_coord = ref["time"]
    if not ens_ids:
        _log.warning(f"{model} [{scenario}]: no ens coord in metrics.nc — skipping")
        ref.close(); return

    _log.info(f"{model} [{scenario}]: {len(ens_ids)} members → {out_path}")

    # Build per-realisation GWI DataArray for GWI-detrending.
    if "gwi" in ref.variables:
        gwi_arr = ref["gwi"].values  # (ens, time)
    else:
        gwi_arr = None
        _log.warning(f"{model} [{scenario}]: no gwi var in metrics.nc — "
                       f"precip GWI-detrend will be skipped")

    # Convert metrics-time coord (typically pandas datetime64 already, but
    # may be cftime for some saves) to numpy datetime64 to match the
    # _ensure_pd_datetime_coord conversion applied to raw pr/sst loaders.
    def _to_np_datetime(t_arr):
        t0 = t_arr[0] if len(t_arr) else None
        if t0 is not None and (hasattr(t0, "calendar")
                                  or t0.__class__.__module__.startswith("cftime")):
            import pandas as _pd
            return _pd.to_datetime(
                [f"{t.year:04d}-{t.month:02d}-{t.day:02d}" for t in t_arr]
            ).values
        return t_arr
    metrics_time_np = _to_np_datetime(time_coord.values)

    def _gwi_for(idx):
        if gwi_arr is None:
            return None
        return xr.DataArray(gwi_arr[idx],
                              coords={"time": metrics_time_np}, dims=["time"])

    # Process members in parallel — each receives its own GWI DataArray.
    results = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(process_member)(
            model, str(rid), scenario, sst_var, pr_var, tauu_var,
            lat_min, lat_max, lon_min, lon_max,
            gwi_da=_gwi_for(k),
        ) for k, rid in enumerate(ens_ids)
    )

    # Align each to time_coord and stack
    new_vars = ["nino3_pr_anom", "nino4_pr_anom",   # NEW: GWI-detrended mm/day
                "nino3_pr_std",  "nino4_pr_std",    # legacy: standardised
                "nino3_tauu_anom", "nino4_tauu_anom",
                "eps_noise_rms"]
    arrs = {v: np.full((len(ens_ids), len(time_coord)), np.nan,
                        dtype=np.float64) for v in new_vars}
    for k, res in enumerate(results):
        if res is None:
            continue
        for v in new_vars:
            arrs[v][k, :] = _align_to_time(res[v], time_coord)

    out = xr.Dataset(
        {v: (["ens", "time"], arrs[v]) for v in new_vars},
        coords={"ens": ens_ids, "time": time_coord.values},
        attrs=dict(
            source="enso_metrics_extra_vars.py",
            model=model, scenario=scenario,
            parent=str(metrics_path),
        ),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_netcdf(out_path)
    ref.close()
    _log.info(f"{model} [{scenario}]: saved → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default="all")
    parser.add_argument("--scenarios", nargs="+",
                         default=["ssp585", "ssp370", "rcp85"])
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--output", default=str(OUTPUT_DIR))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                         format="%(asctime)s %(levelname)s %(message)s")

    models = ALL_MODELS if args.models == "all" else args.models
    results_dir = Path(args.output)

    for model in models:
        for scenario in args.scenarios:
            build_sidecar_for_model(model, scenario, results_dir,
                                     n_jobs=args.n_jobs)


if __name__ == "__main__":
    main()
