"""Observed maps on the archive's 2.5 degree grid, for the map figures (Figures 1, 4d, S5 and S7).

Every map of the paper is shown on the common 2.5 by 2.5 degree grid of MMLEAv2 (cell centres 18.75S to 18.75N and
111.25E to 283.75E for the 20S to 20N, 110E to 285E subsets). COBE-SST2 (1 degree) monthly SST is first averaged onto
those cells, area-weighted with the fractional overlap of each 1 degree cell and ignoring land points (a conservative
average), and every observed statistic is then computed from the 2.5 degree SST, exactly as for the models:

    dsst_per_K        regression of annual SST on the annual Berkeley Earth GWI, 1980 to 2025 (K per K)
    sst_present       annual-mean SST, 1980 to 2014
    djf_sst_std       standard deviation of the DJF monthly anomalies (1980 to 2014 monthly climatology removed),
                      1980 to 2014, as enso_fidelity_maps.py for the models
    nino34_std_cycle  standard deviation of the Nino-3.4 anomaly by calendar month, 1980 to 2014

The box indices of the paper are unaffected by the grid: every box edge (5S, 5N, 110E, 160E, 180, 190E, 210E, 220E,
240E, 270E, 280E) falls on a 2.5 degree cell edge, so the area-weighted box means are the same on both grids.

Output: data/obs/maps_COBE2_g025.nc
"""

import sys
import time
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ensoasym.io import ROOT  # noqa: E402
from ensoasym.obs import berkeley_gwi  # noqa: E402

URL = "https://psl.noaa.gov/thredds/dodsC/Datasets/COBE2/sst.mon.mean.nc"
LAT = np.arange(-18.75, 18.76, 2.5)
LON = np.arange(111.25, 283.76, 2.5)


def overlap(src_c, src_w, dst_c, dst_w):
    """Overlap length between source and destination cells along one axis (dst x src)."""
    s0, s1 = src_c - src_w / 2, src_c + src_w / 2
    d0, d1 = dst_c - dst_w / 2, dst_c + dst_w / 2
    return np.clip(np.minimum(d1[:, None], s1[None, :]) - np.maximum(d0[:, None], s0[None, :]), 0, None)


def to_g025(da):
    """Conservative area-weighted average of a (time, lat, lon) 1 degree field onto the 2.5 degree cells."""
    lat, lon = da.lat.values, da.lon.values
    wy = overlap(lat, 1.0, LAT, 2.5) * np.cos(np.deg2rad(lat))[None, :]
    wx = overlap(lon, 1.0, LON, 2.5)
    v = da.values
    ok = np.isfinite(v)
    num = np.einsum("yi,tij,xj->tyx", wy, np.where(ok, v, 0.0), wx)
    den = np.einsum("yi,tij,xj->tyx", wy, ok.astype(float), wx)
    # a 2.5 degree cell is ocean when at least half of its area is ocean in the 1 degree data
    full = np.einsum("yi,xj->yx", wy, wx)
    out = np.where(den >= 0.5 * full[None], num / np.where(den > 0, den, np.nan), np.nan)
    return xr.DataArray(out, dims=("time", "lat", "lon"), coords={"time": da.time.values, "lat": LAT, "lon": LON})


def read_year(ds, y):
    """One year at a time: a single 46-year OPeNDAP request silently returns zeros from the PSL server."""
    for attempt in range(4):
        try:
            da = ds["sst"].sel(time=slice(f"{y}-01-01", f"{y}-12-31"), lat=slice(20, -20), lon=slice(110, 285)).load()
            if float(np.nanmax(np.abs(da))) > 0:
                return da
            print(y, "attempt", attempt, "returned zeros")
        except Exception as e:  # noqa: BLE001
            print(y, "attempt", attempt, "failed:", str(e)[:80])
        time.sleep(10)
    sys.exit(f"COBE-SST2 {y} could not be read")


with xr.open_dataset(URL) as ds:
    da = xr.concat([read_year(ds, y) for y in range(1980, 2026)], dim="time").sortby("lat")
assert da.sizes["time"] == 552 and 20 < float(np.nanmean(da)) < 30, "unexpected COBE-SST2 input"
sst = to_g025(da)

ann = sst.groupby("time.year").mean("time")
gy = berkeley_gwi().sel(time=slice("1980-01-01", "2025-12-31")).groupby("time.year").mean().to_series()
g = gy.reindex(ann["year"].values).values
ok = np.isfinite(g)
ga = g[ok] - g[ok].mean()
y = ann.values[ok]
dsst_per_K = np.tensordot(ga, y - y.mean(0), axes=(0, 0)) / np.sum(ga**2)
present = ann.sel(year=slice(1980, 2014)).mean("year")

base = sst.sel(time=slice("1980-01-01", "2014-12-31"))
anom = base.groupby("time.month") - base.groupby("time.month").mean("time")
djf = anom.where(anom["time.month"].isin([12, 1, 2]), drop=True)
djf_std = djf.std("time")
n34 = anom.sel(lat=slice(-5, 5), lon=slice(190, 240))
w = np.cos(np.deg2rad(n34.lat))
n34 = n34.weighted(w.fillna(0)).mean(("lat", "lon"))
cycle = n34.groupby("time.month").std("time")

out = xr.Dataset(
    {
        "dsst_per_K": (("lat", "lon"), dsst_per_K.astype("float32")),
        "sst_present": (("lat", "lon"), present.values.astype("float32")),
        "djf_sst_std": (("lat", "lon"), djf_std.values.astype("float32")),
        "nino34_std_cycle": (("month",), cycle.values.astype("float32")),
    },
    coords={"lat": LAT, "lon": LON, "month": np.arange(1, 13)},
    attrs={"source": "COBE-SST2 (PSL), conservatively averaged onto the MMLEAv2 2.5 degree grid",
           "gwi": "Berkeley Earth, degree-4 polynomial", "script": "scripts/build_obs_maps_g025.py"},
)
out.to_netcdf(ROOT / "data/obs/maps_COBE2_g025.nc")
old_n34 = xr.open_dataset(ROOT / "data/obs/fidelity_COBE2.nc").nino34_std_cycle.values
print("saved; Nino-3.4 std cycle, 2.5 degree vs 1 degree product: max difference", float(np.nanmax(np.abs(cycle.values - old_n34))))
