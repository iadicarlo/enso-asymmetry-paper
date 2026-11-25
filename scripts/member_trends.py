"""Every member over the observed period (1980 to 2025), for a like-for-like comparison with observations.

The observed statistics are single 46-year realisations; the model statistics of the paper pool the present-day
state (GWI below 1 K, often a century or more) or compare two warming states, where internal variability averages
out. Following Wills et al. (2022) and Suarez-Gutierrez et al. (2021), each ensemble is compared with observations on
the same years, member by member, and the observed value is ranked within each ensemble separately.

Per member (results/tables/members_1980_2025.csv):
    grad_trend_per_K     regression of the annual zonal SST gradient (110E to 180E minus 180E to 80W, 5S to 5N) on
                         the member's annual GWI
    grad_trend_per_dec   linear trend of the same gradient (K per decade)
    skew_n3              skewness of the paper's standardised relative Nino-3 index
    skew_n3_raw          skewness of the absolute Nino-3 anomaly (30-year rolling seasonal cycle removed)
    A                    asymmetry index of the paper

Per ensemble (results/tables/obs_vs_members.csv): median, 5th and 95th percentiles, number of members at or beyond
the observed value (ERSSTv6 with GPCP), the observed value's rank percentile within the ensemble, and the one-sided
probability of the observed value under a normal fit to the member values.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ensoasym.classification import rolling_anomaly  # noqa: E402
from ensoasym.indices import contributions  # noqa: E402
from ensoasym.io import DATA, MODELS, ROOT, load_members  # noqa: E402
from ensoasym.obs import OBS, month_start  # noqa: E402

TABLES = ROOT / "results" / "tables"
Y0, Y1 = 1980, 2025
YEARS = np.arange(Y0, Y1 + 1)


def skew(x):
    x = x[np.isfinite(x)]
    return float(stats.skew(x)) if x.size > 24 else np.nan


rows = []
for model in MODELS:
    ms = load_members(model)
    if len(ms) < 9:
        continue
    handles = {}
    for m in ms:
        if m.scenario not in handles:
            handles[m.scenario] = (
                xr.open_dataset(DATA / m.scenario / f"{model}_box_anoms.nc"),
                xr.open_dataset(DATA / m.scenario / f"{model}_metrics.nc"),
            )
        box, met = handles[m.scenario]
        t = pd.DatetimeIndex(m.time)
        win = (t.year >= Y0) & (t.year <= Y1)
        grad = m.annual["sst_gradient"].reindex(YEARS).values
        gwi = pd.Series(m.gwi).groupby(t.year).mean().reindex(YEARS).values
        ok = np.isfinite(grad) & np.isfinite(gwi)
        raw = box.nino3_anom.sel(ens=m.rid).values if m.rid in box.ens.values else np.full(t.size, np.nan)
        rows.append(
            dict(
                model=model,
                rid=m.rid,
                scenario=m.scenario,
                n_years=int(ok.sum()),
                grad_trend_per_K=np.polyfit(gwi[ok], grad[ok], 1)[0],
                grad_trend_per_dec=10 * np.polyfit(YEARS[ok], grad[ok], 1)[0],
                skew_n3=skew(m.nino3[win]),
                skew_n3_raw=skew(raw[win]),
                A=contributions(m.nino34[win], m.phase[win])["A"],
            )
        )
    for b, d in handles.values():
        b.close()
        d.close()
mem = pd.DataFrame(rows)
mem.to_csv(TABLES / "members_1980_2025.csv", index=False, float_format="%.5f")

# observed values over the same years (ERSSTv6 with GPCP); the raw Nino-3 anomaly uses the same rolling seasonal
# cycle as the models, on the dataset's record from 1950
om = pd.read_csv(TABLES / "obs_multi.csv")
o6 = om[(om.sst == "ERSSTv6") & (om.pr == "GPCP")].iloc[0]
b = month_start(xr.open_dataset(OBS / "boxes_ERSSTv6.nc")).sel(time=slice("1950-01-01", f"{Y1}-12-01"))
a = rolling_anomaly(b["nino3"].values, b.time.values)
obs = {
    "grad_trend_per_K": o6.sst_gradient_per_K,
    "grad_trend_per_dec": o6.sst_gradient_trend_dec,
    "skew_n3": o6.skew_n3,
    "skew_n3_raw": skew(a[pd.DatetimeIndex(b.time.values).year >= Y0]),
    "A": o6.A,
}

out = []
for stat, ov in obs.items():
    for model, g in mem.groupby("model"):
        v = g[stat].dropna().values
        mu, sd = v.mean(), v.std(ddof=1)
        out.append(
            dict(
                statistic=stat,
                model=model,
                n=v.size,
                observed=ov,
                median=np.median(v),
                p5=np.percentile(v, 5),
                p95=np.percentile(v, 95),
                n_at_or_beyond_obs=int((v >= ov).sum()),
                obs_rank_percentile=100 * np.mean(v < ov),
                p_normal=float(stats.norm.sf((ov - mu) / sd)),
            )
        )
tab = pd.DataFrame(out)
tab.to_csv(TABLES / "obs_vs_members.csv", index=False, float_format="%.4f")
pd.set_option("display.width", 200)
for stat, g in tab.groupby("statistic", sort=False):
    print(
        f"{stat}: observed {g.observed.iloc[0]:.3f}; members at or beyond it {g.n_at_or_beyond_obs.sum()} of {g.n.sum()} "
        f"in {(g.n_at_or_beyond_obs > 0).sum()} ensembles; observed above every member in {(g.obs_rank_percentile == 100).sum()} "
        f"ensembles, above the 95th percentile in {(g.observed > g.p95).sum()}, normal-fit p < 0.05 in {(g.p_normal < 0.05).sum()} "
        f"of {len(g)}; ensemble medians {g['median'].min():.2f} to {g['median'].max():.2f}"
    )
    print("   reaching:", g[g.n_at_or_beyond_obs > 0][["model", "n_at_or_beyond_obs", "n"]].to_string(index=False, header=False).replace("\n", "; "))
