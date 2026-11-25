"""Headline statistics on the fidelity-screened ensembles (cold tongue bias within 1 K)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ensoasym.io import ROOT, PRETTY  # noqa: E402
from ensoasym.bootstrap import two_level_bootstrap, model_bootstrap_median  # noqa: E402

T = ROOT / "results" / "tables"
fid = pd.read_csv(T / "model_fidelity.csv", index_col=0)
mem = pd.read_csv(T / "members_delta.csv")
mem["grad_per_K"] = mem.d_sst_gradient / mem.dgwi
obs = pd.read_csv(T / "obs_multi.csv")
keep = list(fid.index[fid.cold_tongue])
print("screened ensembles (cold tongue bias < 1 K):", len(keep), [PRETTY[m] for m in keep])
sub = mem[mem.model.isin(keep)]
g = sub.groupby("model").median(numeric_only=True)
rows = []
for x, y in [
    ("pr_nino3_ref", "d_A"),
    ("cold_tongue_sst_ref", "d_A"),
    ("pr_nino3_warm", "d_A"),
    ("grad_per_K", "d_A"),
    ("pr_nino3_ref", "d_A_strong"),
    ("cold_tongue_sst_ref", "d_A_strong"),
]:
    b = two_level_bootstrap(sub, x, y, n_boot=1000)
    b["subset"] = "cold_tongue_1K"
    rows.append(b)
    print(
        f"  {y:10s} vs {x:20s} n={b['n']} r={b['r']:+.2f} [{b['r_lo']:+.2f},{b['r_hi']:+.2f}] rho={b['rho']:+.2f} p_perm={b['p_perm']:.3f}"
    )
pd.DataFrame(rows).to_csv(T / "screened_bootstrap.csv", index=False, float_format="%.4f")


def partial(df, x, y, z):
    rx = sm.OLS(df[x], sm.add_constant(df[z])).fit().resid
    ry = sm.OLS(df[y], sm.add_constant(df[z])).fit().resid
    r = stats.pearsonr(rx, ry)[0]
    dof = len(df) - 3  # one controlled variable
    return r, float(2 * stats.t.sf(abs(r) * np.sqrt(dof / (1 - r**2)), dof))


print(
    "  partial: pattern | rainfall r=%+.2f (p=%.2f); rainfall | pattern r=%+.2f (p=%.3f)"
    % (*partial(g, "grad_per_K", "d_A", "pr_nino3_ref"), *partial(g, "pr_nino3_ref", "d_A", "grad_per_K"))
)
mm = []
for q in ("d_C_EN", "d_C_LN", "d_A", "d_var"):
    if q != "d_A":
        pct = 100 * g[q] / g[q.replace("d_", "") + "_ref"]
        med, lo, hi = model_bootstrap_median(pct.values)
        mm.append(dict(quantity=q + "_pct", median=med, lo=lo, hi=hi))
    med, lo, hi = model_bootstrap_median(g[q].values)
    mm.append(dict(quantity=q, median=med, lo=lo, hi=hi))
mm = pd.DataFrame(mm)
mm.to_csv(T / "screened_multimodel_medians.csv", index=False, float_format="%.4f")
print(mm.round(3).to_string(index=False))
out = []
for x, ocol, sets in [
    ("pr_nino3_ref", "pr_nino3_ref", [("ERSSTv6", "GPCP"), ("ERSSTv6", "CMAP")]),
    ("cold_tongue_sst_ref", "cold_tongue_ref", [(s_, "GPCP") for s_ in ("ERSSTv6", "COBE2", "ERSSTv5", "HadISST")]),
]:
    m = sm.OLS(g["d_A"], sm.add_constant(g[x])).fit()
    for sst, pr in sets:
        o = obs[(obs.sst == sst) & (obs.pr == pr)].iloc[0]
        xo = float(o[ocol])
        f68 = m.get_prediction(np.array([[1.0, xo]])).summary_frame(alpha=0.32)
        f95 = m.get_prediction(np.array([[1.0, xo]])).summary_frame(alpha=0.05)
        out.append(
            dict(
                predictor=x,
                sst=sst,
                pr=pr,
                obs_value=xo,
                dA=float(f68["mean"].iloc[0]),
                lo68=float(f68["obs_ci_lower"].iloc[0]),
                hi68=float(f68["obs_ci_upper"].iloc[0]),
                lo95=float(f95["obs_ci_lower"].iloc[0]),
                hi95=float(f95["obs_ci_upper"].iloc[0]),
                slope=float(m.params.iloc[1]),
                r2=float(m.rsquared),
                n=len(g),
                n_below=int((g[x] < xo).sum()),
                n_above=int((g[x] > xo).sum()),
            )
        )
        print(
            f"  constrained dA | {x} = {xo:.2f} ({sst}/{pr}): {out[-1]['dA']:+.3f} [68%: {out[-1]['lo68']:+.3f}, {out[-1]['hi68']:+.3f}] below/above {out[-1]['n_below']}/{out[-1]['n_above']}"
        )
pd.DataFrame(out).to_csv(T / "screened_constrained.csv", index=False, float_format="%.4f")
