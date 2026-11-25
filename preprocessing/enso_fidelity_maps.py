"""DJF SST anomaly standard deviation maps and the seasonal cycle of Nino-3.4 anomaly
standard deviation over 1980-2014, per model (ensemble mean over up to 10 members).
Output: results/enso_metrics/fidelity/<model>_fidelity.nc"""
import glob, sys
from pathlib import Path
import numpy as np, xarray as xr
sys.path.insert(0, str(Path(__file__).resolve().parent))
import time_fixes  # noqa: E402  MMLEAv2 E3SMv1 time axis one month early (2026-09-28)
ROOT = Path("/path/to/enso_impacts")
DATA = ROOT / "data/mmleav2_tropics_output"
OUT = ROOT / "results/enso_metrics/fidelity"; OUT.mkdir(exist_ok=True)
MODELS = sys.argv[1:] or ["ukesm1-0-ll_lens"]   # models to (re)build, given on the command line
# Single-forcing (DAMIP) and non-forced experiments share realization labels with the large-ensemble
# members in some MMLEAv2 folders, so they must be excluded by name before any member grouping.
EXCLUDE_TOKENS = ("picontrol", "pi-control", "1pctco2", "abrupt-4xco2", "lastmillennium",
                  "hist-ghg", "hist-aer", "hist-nat", "hist-sol", "hist-volc", "hist-strato3",
                  "hist-totalo3", "hist-co2", "hist-bmb", "hist-1950hc", "amip")


# The giss-e2 folder also holds GISS-E2-1-H, GISS-E2-2-G and GISS-E2-2-H files and three naming styles of the
# GISS-E2-1-G ones; the MMLEAv2 giss-e2 ensemble (RDA d651039) is GISS-E2-1-G only, one file per member.
REQUIRE_TOKENS = {"giss-e2": ("_omon_giss-e2-1-g_", "_amon_giss-e2-1-g_")}


def _keep(path, model=None):
    name = str(path).rsplit("/", 1)[-1].lower()
    if any(tok in name for tok in EXCLUDE_TOKENS):
        return False
    req = REQUIRE_TOKENS.get(model)
    return req is None or any(tok in name for tok in req)


def files(model):
    for var in ("tos", "ts"):
        fl = sorted(f for f in glob.glob(str(DATA / model / var / "*lat-20.0_20.0*.nc")) if _keep(f, model))
        if fl: return var, fl
        fl = sorted(f for f in glob.glob(str(DATA / model / var / "*.nc")) if _keep(f, model))
        if fl: return var, fl
    return None, []
for model in MODELS:
    var, fl = files(model)
    if not fl:
        print(model, "no files"); continue
    maps, cycles, n = [], [], 0
    for f in fl[:10]:
        try:
            ds = time_fixes.fix_month_offset(xr.open_dataset(f, use_cftime=True), f)
            da = ds[var]
            for d in ("z_t", "lev", "depth"):
                if d in da.dims: da = da.isel({d: 0})
            da = da.squeeze(drop=True)
            if da.lat.values[0] > da.lat.values[-1]: da = da.sortby("lat")
            sel = da.sel(time=slice("1980-01-01", "2014-12-30"))
            if sel.time.size < 400: continue
            if float(sel.mean()) > 200: sel = sel - 273.15
            clim = sel.groupby("time.month").mean("time")
            anom = sel.groupby("time.month") - clim
            months = anom["time"].dt.month
            djf = anom.where(months.isin([12, 1, 2]), drop=True)
            maps.append(djf.std("time").values)
            n34 = anom.sel(lat=slice(-5, 5), lon=slice(190, 240)).mean(("lat", "lon"))
            cycles.append(n34.groupby("time.month").std("time").values)
            n += 1
            lat, lon = anom.lat.values, anom.lon.values
            ds.close()
        except Exception as e:
            print(model, Path(f).name, "skipped:", str(e)[:80])
    if n == 0:
        print(model, "no usable members"); continue
    out = xr.Dataset({"djf_sst_std": (("lat", "lon"), np.nanmean(np.array(maps), axis=0)),
                      "nino34_std_cycle": (("month",), np.nanmean(np.array(cycles), axis=0))},
                     coords={"lat": lat, "lon": lon, "month": np.arange(1, 13)},
                     attrs={"model": model, "members": n, "period": "1980-2014", "variable": var})
    out.to_netcdf(OUT / f"{model}_fidelity.nc"); print(model, "members", n, "written", flush=True)
