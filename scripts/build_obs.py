"""Observed asymmetry, event rates and warming-pattern trends for several datasets."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ensoasym.io import ROOT  # noqa: E402
from ensoasym.indices import contributions  # noqa: E402
from ensoasym.obs import obs_indices, SST_SETS, PR_SETS, OBS, month_start  # noqa: E402

OUT = ROOT / "results" / "tables"


def block_bootstrap_A(x, ph, time, n=2000, seed=1):
    yrs = pd.DatetimeIndex(time).year
    uy = np.unique(yrs)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        pick = rng.choice(uy, size=uy.size, replace=True)
        sel = np.concatenate([np.where(yrs == p)[0] for p in pick])
        vals.append(contributions(x[sel], ph[sel])["A"])
    return np.percentile(vals, [16, 84]), np.percentile(vals, [2.5, 97.5])


def trend(y, x):
    ok = np.isfinite(x) & np.isfinite(y)
    r = stats.linregress(x[ok], y[ok])
    return r.slope, r.stderr


rows = []
for sst in SST_SETS:
    for pr in PR_SETS:
        o = obs_indices(sst, pr)
        x, ph = o["nino34"], o["phase"]
        ok = np.isfinite(x)
        c = contributions(x[ok], ph[ok])
        (lo16, hi84), (lo2, hi97) = block_bootstrap_A(x[ok], ph[ok], o["time"][ok])
        n_yr = ok.sum() / 12
        ev = o["events"]
        ann = o["annual"]
        yrs = ann.index.values.astype(float)
        g = o["gwi_year"].reindex(ann.index).values
        ref = ann.loc[1980:2014]
        r = dict(
            sst=sst,
            pr=pr,
            years=f"{int(yrs.min())}-{int(yrs.max())}",
            n_months=int(ok.sum()),
            A=c["A"],
            A_lo16=lo16,
            A_hi84=hi84,
            A_lo2=lo2,
            A_hi97=hi97,
            A_strong=c["A_strong"],
            C_EN=c["C_EN"],
            C_LN=c["C_LN"],
            var=c["var"],
            skew_n3=float(stats.skew(o["nino3"][np.isfinite(o["nino3"])])),
            skew_n34=float(stats.skew(x[ok])),
            rate_ext_EN=len(ev["extreme_el_nino"]) / n_yr,
            rate_ext_LN=len(ev["extreme_la_nina"]) / n_yr,
            rate_str_EN=len(ev["strong_el_nino"]) / n_yr,
            rate_str_LN=len(ev["strong_la_nina"]) / n_yr,
            rate_mod_EN=len(ev["moderate_el_nino"]) / n_yr,
            rate_mod_LN=len(ev["moderate_la_nina"]) / n_yr,
            cold_tongue_ref=float(ref["cold_tongue"].mean()),
            warm_pool_ref=float(ref["warm_pool"].mean()),
            pr_nino3_ref=float(
                month_start(xr.open_dataset(OBS / f"boxes_{pr}.nc")["nino3"])
                .sel(time=slice("1980-01-01", "2014-12-31"))
                .mean()
            ),
            sst_gradient_ref=float(ref["sst_gradient"].mean()),
            wp_minus_ct_ref=float(ref["wp_minus_ct"].mean()),
            gwi_span=float(np.nanmax(g) - np.nanmin(g)),
        )
        for q in ("sst_gradient", "wp_minus_ct", "cold_tongue", "warm_pool"):
            s_t, se_t = trend(ann[q].values, yrs)
            s_g, se_g = trend(ann[q].values, g)
            r[f"{q}_trend_dec"] = 10 * s_t
            r[f"{q}_trend_dec_se"] = 10 * se_t
            r[f"{q}_per_K"] = s_g
            r[f"{q}_per_K_se"] = se_g
        rows.append(r)
        print(
            f"{sst:8s} {pr:5s} A={c['A']:+.3f} [{lo16:+.3f},{hi84:+.3f}]  extEN {r['rate_ext_EN']:.3f}/yr extLN {r['rate_ext_LN']:.3f}/yr "
            f"skew3={r['skew_n3']:+.2f}  grad/K={r['sst_gradient_per_K']:+.2f}±{r['sst_gradient_per_K_se']:.2f} "
            f"wpct/K={r['wp_minus_ct_per_K']:+.2f}  CT_ref={r['cold_tongue_ref']:.2f}"
        )
df = pd.DataFrame(rows)
df.to_csv(OUT / "obs_multi.csv", index=False, float_format="%.5f")
# reference: lorenz-produced COBE2+GPCP file
ref = xr.open_dataset(ROOT / "data/obs/obs_enso_metrics_1979_2025.nc")
xr_, phr = ref["nino34_std"].values, ref["phase_indicator"].values.astype(int)
cr = contributions(xr_[np.isfinite(xr_)], phr[np.isfinite(xr_)])
print("lorenz COBE2+GPCP file: A=%.3f C_EN=%.3f C_LN=%.3f" % (cr["A"], cr["C_EN"], cr["C_LN"]))
