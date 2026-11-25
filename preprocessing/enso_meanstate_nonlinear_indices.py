"""Compute 5 ENSO-nonlinearity indices per model per member per year.

Uses CDO for the heavy box-mean / year-mean / fldmean lifting.  Python
xarray only reads the tiny (time,) intermediate outputs and combines them.

All indices are designed to express a specific physical mechanism for
the non-linear EN/LN asymmetric response to mean-state change:

  Index 1 — CTEx   : cold-tongue SST − 26.5°C (convective threshold)
  Index 2 — WCCA   : cold-tongue/warm-pool convective asymmetry ratio
  Index 3 — RCAR   : Niño-3 / Niño-4 rainfall ratio (convection migration)
  Index 4 — TWCD   : trade × (1 − pr_east/pr_warmpool)  (Bjerknes closure)
  Index 5 — EWRPA  : zonal correlation corr(τ_x(x), pr(x))  (wind-rain alignment)

Output sidecar (per model, per scenario):
    results/enso_metrics/{scenario}/{model}_nonlinear_indices.nc

With 5 annual-mean scalar fields (ens, year):
    meanstate_ctex       [°C]   = cold_tongue_sst − 26.5
    meanstate_wcca       [-]    = (cold_tongue − 26.5) / (warm_pool − 26.5)
    meanstate_rcar       [-]    = pr_nino3 / pr_nino4
    meanstate_twcd       [N/m²] = tauu_central × (1 − pr_nino3/pr_warmpool)
    meanstate_ewrpa      [-]    = corr(tauu(x), pr(x)) at 5°S-5°N

Plus the individual ingredients (useful for debugging + paper supplement):
    meanstate_cold_tongue_sst [°C]
    meanstate_warm_pool_sst   [°C]
    meanstate_pr_nino3        [mm/day]
    meanstate_pr_nino4        [mm/day]
    meanstate_pr_warmpool     [mm/day]
    meanstate_tauu_central    [N/m²]   (for TWCD)

Usage:
    python enso_meanstate_nonlinear_indices.py                    # all 19 models
    python enso_meanstate_nonlinear_indices.py --models cesm2_lens
    python enso_meanstate_nonlinear_indices.py --n-jobs 8
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ALL_MODELS, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX
import time_fixes  # noqa: E402  MMLEAv2 E3SMv1 time axis one month early (2026-09-28)
from enso_metrics_ensemble import (
    get_model_files, _files_for_member_scenario, OUTPUT_DIR,
    extract_scenario,
)

_log = logging.getLogger("nonlinear_indices")


# ── file dedup ─────────────────────────────────────────────────────────

def _dedup_files(files: list[Path]) -> list[Path]:
    """Deduplicate MMLEAv2 alias files: multiple paths for the same
    (realization, scenario) segment with different grids/naming conventions.

    Strategy: group by extracted scenario token (historical / ssp585 / rcp85 /
    …) and pick one representative per group.  Preference order is driven
    by file-size (largest = most complete), tie-broken by deterministic
    sorted-name.  This avoids CDO ``mergetime`` on overlapping time windows
    and ``mermean`` on heterogeneous grids — both seen in the MMLEAv2 tree
    (e.g. GFDL-CM3 has three aliased files with subtly different grids).
    """
    if not files:
        return []
    groups: dict[str, list[Path]] = {}
    for f in files:
        try:
            key = extract_scenario(f.name) or "unknown"
        except Exception:
            key = "unknown"
        groups.setdefault(key, []).append(f)
    picked: list[Path] = []
    for key in sorted(groups):
        cands = sorted(groups[key], key=lambda p: (-p.stat().st_size, p.name))
        picked.append(cands[0])
    return picked


# ── tauu sign convention ───────────────────────────────────────────────
# CMIP convention: tauu = zonal surface wind stress, eastward positive.
# Central-Pacific trade winds blow westward, so a 5°S-5°N, 170°E-130°W
# box mean SHOULD be negative.  Some LENS datasets (notably CESM1-LE,
# CESM2-LE, E3SM-v2, GFDL-CM3) publish tauu with reversed sign.  Audit
# `audit_nonlinear_indices.py` confirms which models flip.  We correct
# per-model so downstream indices (TWCD, EWRPA) are comparable.
_TAUU_SIGN = {
    "cesm1_lens":    -1,
    "cesm2_lens":    -1,
    "e3sm_v2_lens":  -1,
    "gfdl_cm3_lens": -1,
}


# ── Boxes (5°S-5°N throughout) ─────────────────────────────────────────
_COLD_TONGUE = dict(lat=(-5, 5), lon=(220, 280))   # Niño-3+1+2: 140°W-80°W
_WARM_POOL   = dict(lat=(-5, 5), lon=(140, 190))   # 140°E-170°W
_NINO3       = dict(lat=(-5, 5), lon=(210, 270))   # 150°W-90°W
_NINO4       = dict(lat=(-5, 5), lon=(160, 210))   # 160°E-150°W
_TAUU_CENT   = dict(lat=(-5, 5), lon=(170, 230))   # Cai 2018 central box
_EQ_BASIN    = dict(lat=(-5, 5), lon=(140, 280))   # for EWRPA zonal corr

_CONV_THRESHOLD_SST = 26.5   # °C (Johnson & Xie 2010)

CDO = "cdo"


# ── CDO helpers ────────────────────────────────────────────────────────

def _cdo(*args: str, cwd: Path | None = None):
    """Thin wrapper around subprocess.run(cdo, ...) with concise errors."""
    try:
        subprocess.run([CDO, *args], check=True, capture_output=True,
                        cwd=str(cwd) if cwd else None)
    except subprocess.CalledProcessError as e:
        err = e.stderr.decode()[:400] if e.stderr else "(no stderr)"
        raise RuntimeError(f"cdo {args[0]} failed: {err}") from e


def _box_fldmean_yearmean(in_files: list[Path], box: dict,
                            tmpdir: Path, tag: str,
                            varname: str | None = None) -> xr.DataArray | None:
    """mergetime → sellonlatbox → fldmean → yearmean → annual (time,) series.

    If ``varname`` is given, isolate it with ``-selvar`` so ``fldmean`` does
    not touch auxiliary variables on secondary grids (e.g. ``area`` in IPSL-
    CM6A-LR files, ``time_bnds`` as a generic grid in GFDL-CM3).
    """
    if not in_files:
        return None
    merged   = tmpdir / f"{tag}_merged.nc"
    yearmean = tmpdir / f"{tag}_yrm.nc"
    try:
        sorted_files = sorted(str(time_fixes.shifted_copy(f, tmpdir, CDO)) for f in in_files)
        lat0, lat1 = box["lat"]
        lon0, lon1 = box["lon"]
        ops = ["-O", "-yearmean", "-fldmean",
               f"-sellonlatbox,{lon0},{lon1},{lat0},{lat1}"]
        if varname is not None:
            ops.append(f"-selvar,{varname}")
        ops.extend(["-mergetime", *sorted_files, str(yearmean)])
        _cdo(*ops)
    except RuntimeError as e:
        _log.debug(f"  {tag}: {e}")
        return None
    try:
        ds = xr.open_dataset(yearmean, use_cftime=True, decode_timedelta=False)
        # Pick the data var: must depend on time, must not be a bounds var.
        # This correctly skips `area`, `time_bnds`, etc. that some models carry.
        data_vars = [v for v in ds.data_vars
                     if "time" in ds[v].dims
                     and v not in ("time_bnds", "time_bounds",
                                    "lat_bnds", "lon_bnds")]
        if not data_vars:
            ds.close(); return None
        var = data_vars[0]
        arr = ds[var].squeeze(drop=True)
        years = np.array([int(t.year) for t in arr["time"].values], dtype=int)
        vals = np.asarray(arr.values, dtype=np.float64)
        ds.close()
        # Kelvin → Celsius if obvious
        if vals.size and float(np.nanmean(vals)) > 150.0:
            vals = vals - 273.15
        return xr.DataArray(vals, coords={"year": years}, dims=["year"])
    except Exception as e:
        _log.debug(f"  {tag}: read failed {e}")
        return None


def _box_zonal_yearmean(in_files: list[Path], box: dict,
                         tmpdir: Path, tag: str,
                         varname: str | None = None) -> xr.DataArray | None:
    """Zonal profile: mergetime → sellonlatbox → mermean (avg over lat) → yearmean.

    If ``varname`` is given, isolate it with ``-selvar`` first so ``mermean``
    does not trip over auxiliary variables on secondary grids (some MMLEAv2
    files expose a generic grid for ``time_bnds`` that GFDL-CM3's ``pr``
    trips on).  Returns 2D (year, lon) DataArray of annual-mean climatology
    per year.  (CDO's `mermean` averages meridionally across latitude,
    keeping lon.)
    """
    if not in_files:
        return None
    out = tmpdir / f"{tag}_zon_yrm.nc"
    try:
        sorted_files = sorted(str(time_fixes.shifted_copy(f, tmpdir, CDO)) for f in in_files)
        lat0, lat1 = box["lat"]
        lon0, lon1 = box["lon"]
        ops = ["-O", "-yearmean", "-mermean",
               f"-sellonlatbox,{lon0},{lon1},{lat0},{lat1}"]
        if varname is not None:
            ops.append(f"-selvar,{varname}")
        ops.extend(["-mergetime", *sorted_files, str(out)])
        _cdo(*ops)
    except RuntimeError as e:
        _log.debug(f"  {tag}: {e}")
        return None
    try:
        ds = xr.open_dataset(out, use_cftime=True, decode_timedelta=False)
        data_vars = [v for v in ds.data_vars
                     if "time" in ds[v].dims
                     and v not in ("time_bnds", "time_bounds",
                                    "lat_bnds", "lon_bnds")]
        if not data_vars:
            ds.close(); return None
        var = data_vars[0]
        arr = ds[var].squeeze(drop=True)  # (time, lon) expected
        if "time" not in arr.dims:
            ds.close(); return None
        years = np.array([int(t.year) for t in arr["time"].values], dtype=int)
        lon_name = "lon" if "lon" in arr.coords else (
            "longitude" if "longitude" in arr.coords else None)
        if lon_name is None:
            ds.close(); return None
        lons = arr[lon_name].values
        vals = np.asarray(arr.values, dtype=np.float64)
        ds.close()
        da = xr.DataArray(vals, coords={"year": years, "lon": lons},
                           dims=["year", "lon"])
        return da
    except Exception as e:
        _log.debug(f"  {tag}: read zonmean failed {e}")
        return None


# ── per-member computation ─────────────────────────────────────────────

def _sst_files(model, realization, scenario, lat_min, lat_max, lon_min, lon_max):
    """tos with ts fallback; deduplicated against alias copies."""
    all_tos = get_model_files(model, "tos", lat_min, lat_max, lon_min, lon_max)
    files = _dedup_files(_files_for_member_scenario(all_tos, realization, scenario))
    if files:
        return files, "tos"
    all_ts = get_model_files(model, "ts", lat_min, lat_max, lon_min, lon_max)
    return _dedup_files(_files_for_member_scenario(all_ts, realization, scenario)), "ts"


def _var_files(model, realization, scenario, var, lat_min, lat_max, lon_min, lon_max):
    all_f = get_model_files(model, var, lat_min, lat_max, lon_min, lon_max)
    return _dedup_files(_files_for_member_scenario(all_f, realization, scenario))


def _corr_annual(x_zon: xr.DataArray, y_zon: xr.DataArray) -> xr.DataArray:
    """Per-year Pearson correlation across lon between two zonal-mean fields.

    x_zon, y_zon: (year, lon).  Returns (year,).
    """
    # Align on common years & lons
    yr = np.intersect1d(x_zon["year"].values, y_zon["year"].values)
    lo = np.intersect1d(x_zon["lon"].values,   y_zon["lon"].values)
    if len(yr) == 0 or len(lo) < 3:
        return xr.DataArray(np.full(len(yr), np.nan), coords={"year": yr}, dims=["year"])
    xv = x_zon.sel(year=yr, lon=lo).values
    yv = y_zon.sel(year=yr, lon=lo).values
    n = xv.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        xi, yi = xv[i], yv[i]
        m = np.isfinite(xi) & np.isfinite(yi)
        if m.sum() >= 3:
            xc = xi[m] - np.nanmean(xi[m])
            yc = yi[m] - np.nanmean(yi[m])
            denom = np.sqrt(np.nansum(xc**2) * np.nansum(yc**2))
            if denom > 0:
                out[i] = float(np.nansum(xc * yc) / denom)
    return xr.DataArray(out, coords={"year": yr}, dims=["year"])


def process_member(model: str, realization: str, scenario: str,
                    lat_min: float, lat_max: float,
                    lon_min: float, lon_max: float) -> dict | None:
    """Compute all 5 indices + their raw ingredients for one member."""
    sst_files, sst_name = _sst_files(model, realization, scenario,
                                        lat_min, lat_max, lon_min, lon_max)
    pr_files   = _var_files(model, realization, scenario, "pr",
                              lat_min, lat_max, lon_min, lon_max)
    tauu_files = _var_files(model, realization, scenario, "tauu",
                              lat_min, lat_max, lon_min, lon_max)
    if not sst_files or not pr_files:
        _log.warning(f"  {model} {realization} [{scenario}]: missing SST or PR — skip")
        return None

    with tempfile.TemporaryDirectory(prefix=f"nli_{model}_") as td:
        tdir = Path(td)

        # ── SST box means ─────────────────────────────────────────────
        cold = _box_fldmean_yearmean(sst_files, _COLD_TONGUE, tdir, "sstcold",
                                       varname=sst_name)
        warm = _box_fldmean_yearmean(sst_files, _WARM_POOL,   tdir, "sstwarm",
                                       varname=sst_name)

        # ── Precip box means ──────────────────────────────────────────
        pr_n3 = _box_fldmean_yearmean(pr_files, _NINO3,     tdir, "prn3",
                                        varname="pr")
        pr_n4 = _box_fldmean_yearmean(pr_files, _NINO4,     tdir, "prn4",
                                        varname="pr")
        pr_wp = _box_fldmean_yearmean(pr_files, _WARM_POOL, tdir, "prwp",
                                        varname="pr")

        # ── Tauu central box + zonal profile for EWRPA ───────────────
        tauu_c = None
        tauu_zon = None
        pr_zon = None
        if tauu_files:
            tauu_c = _box_fldmean_yearmean(tauu_files, _TAUU_CENT, tdir, "tauuc",
                                             varname="tauu")
            tauu_zon = _box_zonal_yearmean(tauu_files, _EQ_BASIN, tdir, "tauuzon",
                                             varname="tauu")
            # Apply per-model sign correction (some LENS publish westward-+).
            sign = _TAUU_SIGN.get(model, 1)
            if sign != 1 and tauu_c is not None:
                tauu_c = xr.DataArray(sign * tauu_c.values,
                                       coords=tauu_c.coords, dims=tauu_c.dims)
            if sign != 1 and tauu_zon is not None:
                tauu_zon = xr.DataArray(sign * tauu_zon.values,
                                         coords=tauu_zon.coords, dims=tauu_zon.dims)
        if pr_files:
            pr_zon = _box_zonal_yearmean(pr_files, _EQ_BASIN, tdir, "przon",
                                           varname="pr")

    if cold is None or warm is None or pr_n3 is None or pr_n4 is None or pr_wp is None:
        _log.warning(f"  {model} {realization} [{scenario}]: box computation failed — skip")
        return None

    # Convert precip from kg/m²/s → mm/day if needed (×86400)
    def _to_mmday(da):
        if da is None: return None
        v = da.values
        if np.nanmean(v) < 1e-2:  # likely kg/m²/s
            v = v * 86400.0
        return xr.DataArray(v, coords=da.coords, dims=da.dims)
    pr_n3 = _to_mmday(pr_n3); pr_n4 = _to_mmday(pr_n4); pr_wp = _to_mmday(pr_wp)

    # Align years across ingredients
    def _common(*arrs):
        ys = arrs[0]["year"].values
        for a in arrs[1:]:
            if a is None: continue
            ys = np.intersect1d(ys, a["year"].values)
        return ys
    yrs_sst_pr = _common(cold, warm, pr_n3, pr_n4, pr_wp)
    if len(yrs_sst_pr) == 0:
        return None
    def _pick(da):
        if da is None: return None
        return da.sel(year=yrs_sst_pr)
    cold = _pick(cold); warm = _pick(warm)
    pr_n3 = _pick(pr_n3); pr_n4 = _pick(pr_n4); pr_wp = _pick(pr_wp)

    # ── Compute indices ────────────────────────────────────────────────
    # 1) CTEx
    ctex = cold - _CONV_THRESHOLD_SST

    # 2) WCCA = (cold − 26.5) / (warm − 26.5). Warm pool is ≫ 26.5 always.
    wcca_denom = warm - _CONV_THRESHOLD_SST
    wcca_vals = np.where(wcca_denom.values > 0.1,
                          ctex.values / wcca_denom.values, np.nan)
    wcca = xr.DataArray(wcca_vals, coords=ctex.coords, dims=ctex.dims)

    # 3) RCAR = pr_nino3 / pr_nino4
    rcar_vals = np.where(pr_n4.values > 0.01,
                          pr_n3.values / pr_n4.values, np.nan)
    rcar = xr.DataArray(rcar_vals, coords=pr_n3.coords, dims=pr_n3.dims)

    # 4) TWCD = tauu_central × (1 − pr_nino3 / pr_warmpool)
    twcd = None; tauu_c_out = None
    if tauu_c is not None:
        tauu_c_y = tauu_c.sel(year=np.intersect1d(tauu_c["year"].values, yrs_sst_pr))
        pr_wp_y  = pr_wp.sel(year=tauu_c_y["year"].values)
        pr_n3_y  = pr_n3.sel(year=tauu_c_y["year"].values)
        decoup = np.where(pr_wp_y.values > 0.01,
                           1.0 - pr_n3_y.values / pr_wp_y.values, np.nan)
        twcd_vals = tauu_c_y.values * decoup
        twcd = xr.DataArray(twcd_vals, coords=tauu_c_y.coords, dims=tauu_c_y.dims)
        tauu_c_out = tauu_c_y

    # 5) EWRPA = Pearson corr across lon of (tauu_zonal(x), pr_zonal(x)) per year
    ewrpa = None
    if tauu_zon is not None and pr_zon is not None:
        ewrpa = _corr_annual(tauu_zon, pr_zon)

    return dict(
        realization=realization,
        cold=cold, warm=warm,
        pr_n3=pr_n3, pr_n4=pr_n4, pr_wp=pr_wp,
        tauu_c=tauu_c_out,
        ctex=ctex, wcca=wcca, rcar=rcar, twcd=twcd, ewrpa=ewrpa,
    )


# ── Build per-model sidecar ────────────────────────────────────────────

def _align(da, years):
    out = np.full(len(years), np.nan, dtype=np.float64)
    if da is None: return out
    src = {int(y): i for i, y in enumerate(da["year"].values)}
    for j, y in enumerate(years):
        i = src.get(int(y))
        if i is not None: out[j] = float(da.values[i])
    return out


def build_sidecar(model: str, scenario: str, results_dir: Path,
                   lat_min: float = LAT_MIN, lat_max: float = LAT_MAX,
                   lon_min: float = LON_MIN, lon_max: float = LON_MAX,
                   n_jobs: int = 1) -> None:
    metrics_path = results_dir / scenario / f"{model}_metrics.nc"
    if not metrics_path.exists():
        _log.warning(f"{model} [{scenario}]: no metrics.nc — skipping")
        return
    ref = xr.open_dataset(metrics_path)
    ens_ids = list(ref.coords["ens"].values) if "ens" in ref.coords else []
    years = ref["year"].values.astype(int) if "year" in ref.coords else None
    ref.close()
    if not ens_ids or years is None:
        _log.warning(f"{model} [{scenario}]: no ens/year coord — skip")
        return
    _log.info(f"{model} [{scenario}]: {len(ens_ids)} members")

    results = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(process_member)(
            model, str(rid), scenario, lat_min, lat_max, lon_min, lon_max
        ) for rid in ens_ids
    )

    # Stack (ens, year) arrays
    def _stack(key):
        arr = np.full((len(ens_ids), len(years)), np.nan, dtype=np.float64)
        for k, r in enumerate(results):
            if r is None: continue
            arr[k, :] = _align(r.get(key), years)
        return arr

    ds_out = xr.Dataset(
        {
            "meanstate_cold_tongue_sst": (["ens", "year"], _stack("cold")),
            "meanstate_warm_pool_sst":   (["ens", "year"], _stack("warm")),
            "meanstate_pr_nino3":        (["ens", "year"], _stack("pr_n3")),
            "meanstate_pr_nino4":        (["ens", "year"], _stack("pr_n4")),
            "meanstate_pr_warmpool":     (["ens", "year"], _stack("pr_wp")),
            "meanstate_tauu_central_box":(["ens", "year"], _stack("tauu_c")),
            "meanstate_ctex":            (["ens", "year"], _stack("ctex")),
            "meanstate_wcca":            (["ens", "year"], _stack("wcca")),
            "meanstate_rcar":            (["ens", "year"], _stack("rcar")),
            "meanstate_twcd":            (["ens", "year"], _stack("twcd")),
            "meanstate_ewrpa":           (["ens", "year"], _stack("ewrpa")),
        },
        coords={"ens": ens_ids, "year": years},
        attrs=dict(
            source="enso_meanstate_nonlinear_indices.py",
            conv_threshold_degC=_CONV_THRESHOLD_SST,
            cold_tongue_box="5S-5N, 140W-80W (Niño-3+1+2)",
            warm_pool_box  ="5S-5N, 140E-170W",
            nino3_box      ="5S-5N, 150W-90W",
            nino4_box      ="5S-5N, 160E-150W",
            tauu_central_box="5S-5N, 170E-130W (Cai 2018)",
            ewrpa_domain   ="5S-5N, 140E-80W zonal corr(tauu, pr)",
            index_defs="""
CTEx  = cold_tongue_sst − 26.5                (Johnson & Xie 2010)
WCCA  = (cold − 26.5) / (warm − 26.5)         (new; east-west convective asymmetry)
RCAR  = pr_nino3 / pr_nino4                   (new; convection-centre migration)
TWCD  = tauu_central × (1 − pr_nino3/pr_warmpool)  (new; Bjerknes closure)
EWRPA = corr_lon(tauu(x), pr(x)) at 5S-5N       (new; wind-rain alignment)
""".strip(),
            parent=str(metrics_path),
        ),
    )

    out_path = results_dir / scenario / f"{model}_nonlinear_indices.nc"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds_out.to_netcdf(out_path)
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

    if not shutil.which(CDO):
        raise SystemExit("cdo not found on PATH")

    for model in models:
        for scenario in args.scenarios:
            build_sidecar(model, scenario, results_dir, n_jobs=args.n_jobs)


if __name__ == "__main__":
    main()
