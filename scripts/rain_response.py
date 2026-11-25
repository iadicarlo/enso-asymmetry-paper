"""Rainfall response of the eastern Pacific to El Nino and La Nina, by warming state.

For each member and GWI bin (reference below 1 K, warm above 3 K) the mean
Nino-3 rainfall anomaly (mm/day) over El Nino months (P_EN, positive) and over
La Nina months (P_LN, negative) is computed.  The rainfall-response asymmetry
R = P_EN + P_LN is positive when the El Nino rainfall gain exceeds the La Nina
rainfall loss, the signature of the convective threshold.

Normalised responses (added 2026-09-22 after Yann Planton's review): P_EN and P_LN
also depend on how large the SST anomalies of each phase are, so they are divided
by the mean Nino-3 SST anomaly of the same months (T_EN, T_LN, from the 30-year
rolling seasonal cycle, in K): s_EN = P_EN / T_EN and s_LN = P_LN / T_LN, both
positive, in mm/day per K. Their difference sR = s_EN - s_LN is the rainfall
asymmetry per K of SST anomaly. Figure 4 uses these normalised quantities.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ensoasym.classification import EXTRA_SUFFIX  # noqa: E402
from ensoasym.io import ROOT, DATA, REF, WARM, load_members  # noqa: E402
from ensoasym.indices import EN_ALL, LN_ALL, EN_STRONG, LN_STRONG, bin_mask  # noqa: E402
from ensoasym.bootstrap import two_level_bootstrap  # noqa: E402

T = ROOT / "results" / "tables"
mod = pd.read_csv(T / "models_delta.csv", index_col=0)
rows = []
for model in mod.index:
    members = load_members(model)
    cache = {}
    for m in members:
        p = DATA / m.scenario / f"{model}{EXTRA_SUFFIX}"
        if not p.exists():
            continue
        if m.scenario not in cache:
            cache[m.scenario] = (xr.open_dataset(p), xr.open_dataset(DATA / m.scenario / f"{model}_box_anoms.nc"))
        de, bx = cache[m.scenario]
        ens = [str(e) for e in de["ens"].values]
        if m.rid not in ens or m.rid not in [str(e) for e in bx["ens"].values]:
            continue
        pr = de["nino3_pr_anom"].values[ens.index(m.rid)].astype(float) * 86400.0
        sst = bx["nino3_anom"].sel(ens=m.rid).values.astype(float)
        if pr.size != m.nino34.size or sst.size != pr.size:
            continue
        row = dict(model=model, rid=m.rid)
        ok_all = True
        for tag, (lo, hi) in (("ref", REF), ("warm", WARM)):
            sel = bin_mask(m, lo, hi) & np.isfinite(pr) & np.isfinite(sst)
            if sel.sum() < 120:
                ok_all = False
                break
            ph = m.phase[sel]
            prs = pr[sel]
            ts = sst[sel]
            en = np.isin(ph, EN_ALL)
            ln = np.isin(ph, LN_ALL)
            ens_ = np.isin(ph, EN_STRONG)
            lns = np.isin(ph, LN_STRONG)
            row[f"P_EN_{tag}"] = prs[en].mean() if en.sum() > 12 else np.nan
            row[f"P_LN_{tag}"] = prs[ln].mean() if ln.sum() > 12 else np.nan
            row[f"P_ENs_{tag}"] = prs[ens_].mean() if ens_.sum() > 12 else np.nan
            row[f"P_LNs_{tag}"] = prs[lns].mean() if lns.sum() > 12 else np.nan
            row[f"R_{tag}"] = row[f"P_EN_{tag}"] + row[f"P_LN_{tag}"]
            row[f"T_EN_{tag}"] = ts[en].mean() if en.sum() > 12 else np.nan
            row[f"T_LN_{tag}"] = ts[ln].mean() if ln.sum() > 12 else np.nan
            row[f"s_EN_{tag}"] = row[f"P_EN_{tag}"] / row[f"T_EN_{tag}"]
            row[f"s_LN_{tag}"] = row[f"P_LN_{tag}"] / row[f"T_LN_{tag}"]
            row[f"sR_{tag}"] = row[f"s_EN_{tag}"] - row[f"s_LN_{tag}"]
            row[f"Rs_{tag}"] = row[f"P_ENs_{tag}"] + row[f"P_LNs_{tag}"]
            row[f"skewP_{tag}"] = float(stats.skew(prs))
            row[f"sigP_{tag}"] = float(prs.std())
        if not ok_all:
            continue
        for q in ("P_EN", "P_LN", "P_ENs", "P_LNs", "R", "Rs", "skewP", "sigP", "T_EN", "T_LN", "s_EN", "s_LN", "sR"):
            row[f"d_{q}"] = row[f"{q}_warm"] - row[f"{q}_ref"]
        rows.append(row)
    for pair in cache.values():
        for d in pair:
            d.close()
    print(f"{model:18s} members with rainfall: {sum(r['model'] == model for r in rows)}")
df = pd.DataFrame(rows)
df.to_csv(T / "members_rain_response.csv", index=False, float_format="%.4f")
mem = pd.read_csv(T / "members_delta.csv")
j = df.merge(mem[["model", "rid", "d_A", "A_ref", "pr_nino3_ref", "cold_tongue_sst_ref"]], on=["model", "rid"])
g = j.groupby("model").median(numeric_only=True)
print("\nper-model medians (n=%d):" % len(g))
print(
    g[["pr_nino3_ref", "P_EN_ref", "P_LN_ref", "R_ref", "P_EN_warm", "P_LN_warm", "R_warm", "d_R", "d_skewP", "d_A"]]
    .round(2)
    .to_string()
)
print("\ncross-model relations (per-model medians):")
for x, y in [
    ("pr_nino3_ref", "P_LN_ref"),
    ("pr_nino3_ref", "P_EN_ref"),
    ("pr_nino3_ref", "R_ref"),
    ("pr_nino3_ref", "d_R"),
    ("pr_nino3_ref", "d_P_LN"),
    ("pr_nino3_ref", "d_P_EN"),
    ("d_R", "d_A"),
    ("d_skewP", "d_A"),
    ("R_ref", "A_ref"),
    ("d_Rs", "d_A"),
    ("pr_nino3_ref", "d_skewP"),
    ("pr_nino3_ref", "s_EN_ref"),
    ("pr_nino3_ref", "s_LN_ref"),
    ("pr_nino3_ref", "sR_ref"),
    ("sR_ref", "A_ref"),
    ("sR_ref", "d_A"),
    ("pr_nino3_ref", "d_s_EN"),
    ("pr_nino3_ref", "d_s_LN"),
    ("d_sR", "d_A"),
]:
    b = two_level_bootstrap(j, x, y, n_boot=2000)
    print(
        f"  {y:8s} vs {x:14s} r={b['r']:+.2f} [{b['r_lo']:+.2f},{b['r_hi']:+.2f}] rho={b['rho']:+.2f} p_perm={b['p_perm']:.3f}"
    )
