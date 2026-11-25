"""Statistics quoted in the text that no other table holds: the cross-model correlation between present-day Nino-3
rainfall and cold tongue SST, and the partial correlations within the screened (cold tongue bias within 1 K) set.
Output: results/tables/extra_stats.csv"""

import sys
from pathlib import Path

import pandas as pd
import statsmodels.api as sm
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ensoasym.plotting import TABLES  # noqa: E402

mod = pd.read_csv(TABLES / "models_delta.csv", index_col=0)
fid = pd.read_csv(TABLES / "model_fidelity.csv", index_col=0)
mem = pd.read_csv(TABLES / "members_delta.csv")
mem["grad_per_K"] = mem.d_sst_gradient / mem.dgwi


def partial_r(df, x, y, z):
    """Correlation between the residuals of x and y after regressing each on z (same method as constrain.py and
    screened.py, so the numbers match Table S3)."""
    rx = sm.OLS(df[x], sm.add_constant(df[z])).fit().resid
    ry = sm.OLS(df[y], sm.add_constant(df[z])).fit().resid
    r = stats.pearsonr(rx, ry)[0]
    df_t = len(df) - 3  # one controlled variable
    return r, float(2 * stats.t.sf(abs(r) * (df_t / (1 - r**2)) ** 0.5, df_t)), len(df)


rows = []
r, p = stats.pearsonr(mod["pr_nino3_ref"], mod["cold_tongue_sst_ref"])
rows.append({"statistic": "r(pr_nino3_ref, cold_tongue_sst_ref), all ensembles", "r": r, "p": p, "n": len(mod)})
keep = list(fid.index[fid["cold_tongue"].astype(bool)])
g = mem[mem.model.isin(keep)].groupby("model").median(numeric_only=True)
r, p, n = partial_r(g, "grad_per_K", "d_A", "pr_nino3_ref")
rows.append({"statistic": "partial r(d_A, grad_per_K | pr_nino3_ref), screened", "r": r, "p": p, "n": n})
r, p, n = partial_r(g, "pr_nino3_ref", "d_A", "grad_per_K")
rows.append({"statistic": "partial r(d_A, pr_nino3_ref | grad_per_K), screened", "r": r, "p": p, "n": n})
out = pd.DataFrame(rows)
out.to_csv(TABLES / "extra_stats.csv", index=False, float_format="%.4f")
print(out.round(3).to_string())

# fixed calendar periods (2070 to 2099 minus 1985 to 2014) against the warming-level design
per = pd.read_csv(TABLES / "members_delta_periods.csv")
per["grad_per_K"] = per["d_sst_gradient"] / per["dgwi"]
gp = per.groupby("model")[["d_A", "grad_per_K"]].median().join(mod[["pr_nino3_ref", "cold_tongue_sst_ref"]])
rows = []
for x in ("pr_nino3_ref", "cold_tongue_sst_ref", "grad_per_K"):
    r, p = stats.pearsonr(gp[x], gp["d_A"])
    rows.append({"predictor": x, "controlling_for": "", "r": r, "p": p, "n": len(gp)})
for x, z in (("grad_per_K", "pr_nino3_ref"), ("pr_nino3_ref", "grad_per_K"), ("grad_per_K", "cold_tongue_sst_ref"),
             ("cold_tongue_sst_ref", "grad_per_K")):
    r, p, n = partial_r(gp, x, "d_A", z)
    rows.append({"predictor": x, "controlling_for": z, "r": r, "p": p, "n": n})
fix = pd.DataFrame(rows)
fix.to_csv(TABLES / "fixed_period_stats.csv", index=False, float_format="%.4f")
print(fix.round(3).to_string())
