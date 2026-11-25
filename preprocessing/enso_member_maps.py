"""Per-member SST maps for every member of the paper roster, with each member's own scenario only.

For each (model, rid, scenario) of the roster (results/enso_metrics/paper_roster.csv, from the paper's
members_delta.csv), the member's SST files are selected and merged with the functions of enso_metrics_ensemble.py
(historical plus the member's own scenario, the same files as the paper's indices), annual means are formed, and
four maps are written on the archive's 2.5 degree grid:

    trend_per_K   regression of annual SST on the member's annual GWI over 1980 to 2025 (K per K)
    trend_per_dec linear trend over 1980 to 2025 (K per decade)
    dsst_per_K    forced-change estimate of the paper: years with GWI above 3 K minus years below 1 K, over the GWI
                  difference (K per K)
    sst_present   mean over years with GWI below 1 K (degC)

plus the centred, cos(lat)-weighted pattern correlation of the relative 1980 to 2025 trend map (minus its mean over
20S to 20N, 120E to 75W) with the observed COBE-SST2 map of results/enso_metrics/fidelity/obs_pattern_COBE2.nc
interpolated to the same grid. Output: results/enso_metrics/fidelity/member_maps.nc (member dimension) and
member_maps_index.csv. The older enso_pattern_maps.py grouped files by realization label only and could mix the
SSP3-7.0 and SSP5-8.5 futures of a member; this script replaces it for the paper.

    python enso_member_maps.py            (SLURM: jobs/run_member_maps.sh)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed

ROOT = Path("/path/to/enso_impacts")
DATA = ROOT / "data/mmleav2_tropics_output"
RES = ROOT / "results/enso_metrics"
OUT = RES / "fidelity"
YEARS = np.arange(1980, 2026)
BOX = (-20.0, 20.0, 110.0, 285.0)

# the member's SST files and their merge are those of the paper's indices (enso_metrics_ensemble.py): same
# exclusions (control, single-forcing), same GISS-E2-1-G name filter, historical plus the member's own scenario only
sys.path.insert(0, str(ROOT / "scripts/analysis/enso_future"))
import enso_metrics_ensemble as eme  # noqa: E402


def load_member(model, rid, scen):
    for var in ("tos", "ts"):
        fl = eme._files_for_member_scenario(eme.get_model_files(model, var, *BOX), rid, scen)
        if fl:
            break
    else:
        return None
    ds = eme._load_and_merge_var(fl)
    if ds is None:
        return None
    da = ds[var].squeeze(drop=True).load()
    if da.lat.values[0] > da.lat.values[-1]:
        da = da.sortby("lat")
    if float(da.mean()) > 200:
        da = da - 273.15
    ann = da.groupby("time.year").mean("time")
    return var, len(fl), ann


def gwi_annual(model, rid, scen):
    with xr.open_dataset(RES / scen / f"{model}_metrics.nc", use_cftime=True) as dm:
        g = dm["gwi"].sel(ens=rid)
        yrs = g["time"].dt.year.values
        return {int(y): float(np.nanmean(g.values[yrs == y])) for y in np.unique(yrs)}


def slope_map(x, y):
    """OLS slope of y (t, lat, lon) on x (t), grid point by grid point."""
    ok = np.isfinite(x)
    x, y = x[ok], y[ok]
    xa = x - x.mean()
    return np.tensordot(xa, y - y.mean(0), axes=(0, 0)) / np.sum(xa**2)


def process(model, rid, scen):
    try:
        got = load_member(model, rid, scen)
        if got is None:
            return dict(model=model, rid=rid, scenario=scen, status="no files")
        var, nfiles, ann = got
        g = gwi_annual(model, rid, scen)
        yrs = ann["year"].values.astype(int)
        gw = np.array([g.get(int(y), np.nan) for y in yrs])
        a = ann.values
        win = np.isin(yrs, YEARS)
        if win.sum() < 40:
            return dict(model=model, rid=rid, scenario=scen, status=f"only {win.sum()} years in 1980-2025")
        t_per_K = slope_map(gw[win], a[win])
        t_per_dec = 10 * slope_map(yrs[win].astype(float), a[win])
        ref, warm = np.isfinite(gw) & (gw < 1.0), np.isfinite(gw) & (gw > 3.0)
        if ref.sum() >= 10 and warm.sum() >= 10:
            d_per_K = (a[warm].mean(0) - a[ref].mean(0)) / float(gw[warm].mean() - gw[ref].mean())
        else:
            d_per_K = np.full(a.shape[1:], np.nan)
        present = a[ref].mean(0) if ref.sum() >= 10 else np.full(a.shape[1:], np.nan)
        return dict(model=model, rid=rid, scenario=scen, status="ok", var=var, nfiles=nfiles,
                    lat=ann.lat.values, lon=ann.lon.values, trend_per_K=t_per_K.astype("float32"),
                    trend_per_dec=t_per_dec.astype("float32"), dsst_per_K=d_per_K.astype("float32"),
                    sst_present=present.astype("float32"))
    except Exception as e:  # noqa: BLE001
        return dict(model=model, rid=rid, scenario=scen, status="error: " + str(e)[:150])


def relative(field, lat, lon):
    box = (lon >= 120) & (lon <= 285)
    w = np.cos(np.deg2rad(lat))[:, None] * np.ones_like(field)
    m = np.isfinite(field) & box[None, :]
    return field - np.sum(field[m] * w[m]) / np.sum(w[m])


def pattern_corr(a, b, lat, lon):
    box = ((lon >= 120) & (lon <= 285))[None, :] & np.ones_like(a, dtype=bool)
    w = np.cos(np.deg2rad(lat))[:, None] * np.ones_like(a)
    m = box & np.isfinite(a) & np.isfinite(b)
    aa, bb, ww = a[m] - np.average(a[m], weights=w[m]), b[m] - np.average(b[m], weights=w[m]), w[m]
    return float(np.sum(ww * aa * bb) / np.sqrt(np.sum(ww * aa**2) * np.sum(ww * bb**2)))


if __name__ == "__main__":
    roster = pd.read_csv(RES / "paper_roster.csv")
    models = sys.argv[1:] or sorted(roster.model.unique())
    roster = roster[roster.model.isin(models)]
    res = Parallel(n_jobs=16, verbose=5)(delayed(process)(r.model, r.rid, r.scenario) for r in roster.itertuples())
    ok = [r for r in res if r["status"] == "ok"]
    bad = [r for r in res if r["status"] != "ok"]
    for r in bad:
        print("FAILED", r["model"], r["rid"], r["scenario"], r["status"])
    lat, lon = ok[0]["lat"], ok[0]["lon"]
    ok = [r for r in ok if r["lat"].shape == lat.shape and r["lon"].shape == lon.shape]
    obs = xr.open_dataset(OUT / "obs_pattern_COBE2.nc")["dsst_per_K"].interp(lat=lat, lon=lon).values
    obs_rel = relative(obs, lat, lon)
    idx = []
    for r in ok:
        idx.append(dict(model=r["model"], rid=r["rid"], scenario=r["scenario"], var=r["var"], nfiles=r["nfiles"],
                        pattern_corr=pattern_corr(relative(r["trend_per_K"], lat, lon), obs_rel, lat, lon)))
    index = pd.DataFrame(idx)
    ds = xr.Dataset(
        {k: (("member", "lat", "lon"), np.stack([r[k] for r in ok])) for k in ("trend_per_K", "trend_per_dec", "dsst_per_K", "sst_present")},
        coords={"member": np.arange(len(ok)), "lat": lat, "lon": lon,
                "model": ("member", index.model.values), "rid": ("member", index.rid.values),
                "scenario": ("member", index.scenario.values)},
        attrs={"script": "enso_member_maps.py", "years": "1980-2025 for the trends",
               "obs": "COBE-SST2 1980-2025 regression on the Berkeley GWI, interpolated to the archive grid"})
    ds["pattern_corr"] = ("member", index.pattern_corr.values)
    ds["obs_trend_per_K"] = (("lat", "lon"), obs.astype("float32"))
    ds.to_netcdf(OUT / "member_maps.nc")
    index.to_csv(OUT / "member_maps_index.csv", index=False)
    print(f"{len(ok)} members written, {len(bad)} failed")
    print(index.groupby("model").pattern_corr.describe().round(2).to_string())
