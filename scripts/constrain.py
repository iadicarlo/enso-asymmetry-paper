"""Partial correlations and observation-anchored estimates of the asymmetry change."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ensoasym.io import ROOT  # noqa: E402

T = ROOT / "results" / "tables"
mem = pd.read_csv(T / "members_delta.csv")
mem["grad_per_K"] = mem.d_sst_gradient / mem.dgwi
g = mem.groupby("model")[
    ["grad_per_K", "d_A", "pr_nino3_ref", "pr_nino3_warm", "cold_tongue_sst_ref", "skew_n3_ref", "A_ref"]
].median()
obs = pd.read_csv(T / "obs_multi.csv")


def partial_p(r, n, k=1):
    """Two-sided p of a partial correlation with k controlled variables (df = n - 2 - k)."""
    df = n - 2 - k
    return float(2 * stats.t.sf(abs(r) * np.sqrt(df / (1 - r**2)), df))


def partial(df, x, y, z):
    rx = sm.OLS(df[x], sm.add_constant(df[z])).fit().resid
    ry = sm.OLS(df[y], sm.add_constant(df[z])).fit().resid
    r = stats.pearsonr(rx, ry)[0]
    return r, partial_p(r, len(df))


rows = []
for x, z in [
    ("grad_per_K", "pr_nino3_ref"),
    ("pr_nino3_ref", "grad_per_K"),
    ("grad_per_K", "cold_tongue_sst_ref"),
    ("cold_tongue_sst_ref", "grad_per_K"),
    ("pr_nino3_ref", "cold_tongue_sst_ref"),
    ("cold_tongue_sst_ref", "pr_nino3_ref"),
    ("grad_per_K", "skew_n3_ref"),
    ("pr_nino3_ref", "skew_n3_ref"),
    ("grad_per_K", "A_ref"),
    ("pr_nino3_ref", "A_ref"),
]:
    r0, p0 = stats.pearsonr(g[x], g["d_A"])
    r, p = partial(g, x, "d_A", z)
    rows.append(dict(predictor=x, controlling_for=z, r_simple=r0, p_simple=p0, r_partial=r, p_partial=p, n=len(g)))
    print(f"dA ~ {x:20s} | {z:20s}: simple r={r0:+.2f} (p={p0:.3f})  partial r={r:+.2f} (p={p:.3f})")
pd.DataFrame(rows).to_csv(T / "partial_correlations.csv", index=False, float_format="%.4f")
print(
    "corr(grad_per_K, pr_nino3_ref) = %+.2f, corr(grad_per_K, CT_ref) = %+.2f, corr(pr_nino3_ref, CT_ref) = %+.2f"
    % (
        stats.pearsonr(g.grad_per_K, g.pr_nino3_ref)[0],
        stats.pearsonr(g.grad_per_K, g.cold_tongue_sst_ref)[0],
        stats.pearsonr(g.pr_nino3_ref, g.cold_tongue_sst_ref)[0],
    )
)

# observation-anchored estimates: regress dA on the present-day predictor across models, evaluate at the observed value
out = []
for x, obs_col, sets in [
    ("pr_nino3_ref", "pr_nino3_ref", [("ERSSTv6", "GPCP"), ("ERSSTv6", "CMAP")]),
    ("cold_tongue_sst_ref", "cold_tongue_ref", [(s_, "GPCP") for s_ in ("ERSSTv6", "COBE2", "ERSSTv5", "HadISST")]),
]:
    m = sm.OLS(g["d_A"], sm.add_constant(g[x])).fit()
    for sst, pr in sets:
        o = obs[(obs.sst == sst) & (obs.pr == pr)].iloc[0]
        xo = float(o[obs_col])
        pf68 = m.get_prediction(np.array([[1.0, xo]])).summary_frame(alpha=0.32)
        pf95 = m.get_prediction(np.array([[1.0, xo]])).summary_frame(alpha=0.05)
        out.append(
            dict(
                predictor=x,
                sst=sst,
                pr=pr,
                obs_value=xo,
                dA=float(pf68["mean"].iloc[0]),
                lo68=float(pf68["obs_ci_lower"].iloc[0]),
                hi68=float(pf68["obs_ci_upper"].iloc[0]),
                lo95=float(pf95["obs_ci_lower"].iloc[0]),
                hi95=float(pf95["obs_ci_upper"].iloc[0]),
                slope=float(m.params.iloc[1]),
                r2=float(m.rsquared),
                n_colder_or_drier=int((g[x] < xo).sum()),
                n_warmer_or_wetter=int((g[x] > xo).sum()),
            )
        )
        print(
            f"constrained dA | {x} = {xo:.2f} ({sst}/{pr}): {out[-1]['dA']:+.3f} [68%: {out[-1]['lo68']:+.3f}, {out[-1]['hi68']:+.3f}] "
            f"[95%: {out[-1]['lo95']:+.3f}, {out[-1]['hi95']:+.3f}]  models below/above obs: {out[-1]['n_colder_or_drier']}/{out[-1]['n_warmer_or_wetter']}"
        )
# the rainfall asymmetry per K of SST anomaly (scripts/rain_response.py; observed from fig4_mechanism.py), 2026-09-22
rr = pd.read_csv(T / "members_rain_response.csv").groupby("model").sR_ref.median()
orr = pd.read_csv(T / "obs_rain_response.csv").iloc[0]
gs = g.join(rr, how="inner")
m = sm.OLS(gs["d_A"], sm.add_constant(gs["sR_ref"])).fit()
xo = float(orr["sR"])
pf68 = m.get_prediction(np.array([[1.0, xo]])).summary_frame(alpha=0.32)
pf95 = m.get_prediction(np.array([[1.0, xo]])).summary_frame(alpha=0.05)
out.append(
    dict(
        predictor="rain_asymmetry_per_K",
        sst=str(orr["sst"]),
        pr=str(orr["pr"]),
        obs_value=xo,
        dA=float(pf68["mean"].iloc[0]),
        lo68=float(pf68["obs_ci_lower"].iloc[0]),
        hi68=float(pf68["obs_ci_upper"].iloc[0]),
        lo95=float(pf95["obs_ci_lower"].iloc[0]),
        hi95=float(pf95["obs_ci_upper"].iloc[0]),
        slope=float(m.params.iloc[1]),
        r2=float(m.rsquared),
        n_colder_or_drier=int((gs["sR_ref"] < xo).sum()),
        n_warmer_or_wetter=int((gs["sR_ref"] > xo).sum()),
    )
)
print(f"constrained dA | rainfall asymmetry per K = {xo:.2f}: {out[-1]['dA']:+.3f} [68%: {out[-1]['lo68']:+.3f}, {out[-1]['hi68']:+.3f}] "
      f"[95%: {out[-1]['lo95']:+.3f}, {out[-1]['hi95']:+.3f}]  models below/above obs: {out[-1]['n_colder_or_drier']}/{out[-1]['n_warmer_or_wetter']}")
pd.DataFrame(out).to_csv(T / "constrained_estimates.csv", index=False, float_format="%.4f")
