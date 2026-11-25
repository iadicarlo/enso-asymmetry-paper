"""What the large ensembles add: sign agreement of single members, members needed, variance partition,
and the sensitivity of the headline relation to the warming bins.  Outputs tables and Figure S3."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import ultraplot as uplt
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ensoasym import style as S  # noqa: E402
from ensoasym.io import ROOT, FIG_DIR, PRETTY, load_members, MIN_MEMBERS  # noqa: E402
from ensoasym.indices import model_delta_table  # noqa: E402
from ensoasym.plotting import load_tables, rain_color  # noqa: E402

S.apply()
T = ROOT / "results" / "tables"
mod, mem, tr, obs = load_tables()
models = list(mod.index)
rng = np.random.default_rng(0)

# 1. sign agreement of single members with the ensemble median, and members needed for SE < |median|
rows = []
for m in models:
    d = mem[mem.model == m]["d_A"].values
    med = np.median(d)
    agree = np.mean(np.sign(d) == np.sign(med))
    sd = np.std(d, ddof=1)
    # members needed so that the standard error of the mean falls below the magnitude of the forced change
    n_needed = (sd / abs(med)) ** 2 if med != 0 else np.inf
    # subsampling: probability that the median of k members has the right sign
    probs = {}
    for k in (1, 3, 5, 10, 20):
        if k > len(d):
            probs[k] = np.nan
            continue
        idx = rng.integers(0, len(d), (2000, k))
        probs[k] = np.mean(np.sign(np.median(d[idx], axis=1)) == np.sign(med))
    rows.append(
        dict(
            model=m,
            pretty=PRETTY[m],
            n=len(d),
            dA=med,
            sd_members=sd,
            sign_agree_1=agree,
            n_needed=n_needed,
            **{f"p_right_sign_k{k}": v for k, v in probs.items()},
        )
    )
sa = pd.DataFrame(rows).set_index("model")
sa.to_csv(T / "smile_sign_agreement.csv", float_format="%.4f")
print(
    sa[["pretty", "n", "dA", "sd_members", "sign_agree_1", "p_right_sign_k5", "p_right_sign_k10", "n_needed"]]
    .round(2)
    .to_string()
)
print(
    "\nmedian across models: single-member sign agreement %.2f; 10-member %.2f; members needed (SE<|dA|) median %.0f"
    % (sa.sign_agree_1.median(), sa.p_right_sign_k10.median(), sa.n_needed.replace(np.inf, np.nan).median())
)

# 2. variance partition of single-member dA: between models vs within models
d_all = mem["d_A"].values
between = np.var(mem.groupby("model")["d_A"].median().values, ddof=1)
within = mem.groupby("model")["d_A"].var(ddof=1).mean()
print(
    "variance of dA: between-model %.4f, within-model (mean) %.4f, internal share %.0f%%"
    % (between, within, 100 * within / (between + within))
)
# within-model correlation between a member's present-day rainfall and its dA (internal-variability version of the relation)
w = mem.groupby("model").apply(
    lambda g: stats.pearsonr(g.pr_nino3_ref, g.d_A)[0] if len(g) > 5 else np.nan, include_groups=False
)
print("within-model corr(pr_ref, dA): median %.2f, range %.2f to %.2f" % (w.median(), w.min(), w.max()))
pd.Series(
    dict(
        between_var=between,
        within_var=within,
        internal_share=within / (between + within),
        within_corr_median=w.median(),
        within_corr_min=w.min(),
        within_corr_max=w.max(),
    )
).to_csv(T / "smile_variance_partition.csv", header=False)

# 3. sensitivity of the headline relation to the warming bins
sens = []
for ref_hi, warm_lo in ((1.0, 3.0), (1.0, 2.5), (1.0, 3.5), (0.5, 3.0), (1.0, 2.0)):
    per = []
    for m in models:
        mm = load_members(m, ref=(0.0, ref_hi), warm=(warm_lo, 99.0))
        df = model_delta_table(mm, (0.0, ref_hi), (warm_lo, 99.0))
        if len(df) >= MIN_MEMBERS:
            g = df.median(numeric_only=True)
            g["model"] = m
            per.append(g)
    per = pd.DataFrame(per).set_index("model")
    r, p = stats.pearsonr(per.pr_nino3_ref, per.d_A)
    rho = stats.spearmanr(per.pr_nino3_ref, per.d_A)[0]
    r2, p2 = stats.pearsonr(per.cold_tongue_sst_ref, per.d_A)
    sens.append(
        dict(
            ref_bin=f"<{ref_hi} K",
            warm_bin=f">{warm_lo} K",
            n=len(per),
            r_rain=r,
            p_rain=p,
            rho_rain=rho,
            r_ct=r2,
            p_ct=p2,
            dA_median=per.d_A.median(),
        )
    )
    print(
        f"ref <{ref_hi} K, warm >{warm_lo} K: n={len(per)} r(rain)={r:+.2f} (p={p:.3f}) rho={rho:+.2f}  r(CT)={r2:+.2f} (p={p2:.3f})  median dA {per.d_A.median():+.3f}"
    )
pd.DataFrame(sens).to_csv(T / "bin_sensitivity.csv", index=False, float_format="%.4f")

# Figure S3
fig, axs = uplt.subplots(ncols=3, wratios=(1.9, 1, 0.8), refwidth=2.1, refheight=2.6, wspace=(3.6, 6.5), share=False)
ax = axs[0]
order = sa.sort_values("dA").index
y = np.arange(len(order))
for yi, m in zip(y, order):
    d = mem[mem.model == m]["d_A"].values
    c = rain_color(mod.loc[m, "pr_nino3_ref"])
    ax.scatter(d, np.full(d.size, yi) + rng.uniform(-0.22, 0.22, d.size), s=6, color=c, alpha=0.5, lw=0, zorder=2)
    ax.plot([np.percentile(d, 5), np.percentile(d, 95)], [yi, yi], color=S.INK, lw=1.0, zorder=3)
    ax.scatter(sa.loc[m, "dA"], yi, s=30, color="white", edgecolor=S.INK, lw=1.0, zorder=5)
    ax.text(0.33, yi, f"{100 * sa.loc[m, 'sign_agree_1']:.0f}%", va="center", fontsize=6, color=S.INK)
ax.axvline(0, color=S.MUTED, lw=0.6, ls=(0, (3, 2)))
ax.format(
    yticks=y,
    yticklabels=[f"{int(mod.loc[m, 'num'])} {PRETTY[m]} ({int(sa.loc[m, 'n'])})" for m in order],
    xlabel="Asymmetry change Δ$A$ of single members",
    title="Single members versus the ensemble",
    xlim=(-0.42, 0.44),
    ylim=(-0.8, len(order) + 1.6),
    xticks=[-0.4, -0.2, 0, 0.2],
    ytickminor=False,
    xtickminor=False,
)
ax.tick_params(axis="y", length=0, labelsize=6.5)
S.despine(ax, keep=("bottom", "left"))
ax.spines["left"].set_linewidth(0)

ax = axs[1]
ks = np.array([1, 3, 5, 10, 20])
for m in models:
    vals = [sa.loc[m, f"p_right_sign_k{k}"] for k in ks]
    ax.plot(ks, vals, color=rain_color(mod.loc[m, "pr_nino3_ref"]), lw=0.9, alpha=0.8, marker="o", ms=2.5)
ax.axhline(0.5, color=S.MUTED, lw=0.6, ls=(0, (3, 2)))
ax.format(
    xlabel="Members averaged",
    ylabel="Probability of the right sign",
    title="Members needed",
    xscale="log",
    xticks=ks,
    xticklabels=[str(k) for k in ks],
    ylim=(0.4, 1.02),
    xtickminor=False,
    ytickminor=False,
    xformatter="scalar",
)
S.despine(ax)

ax = axs[2]
ax.bar([0, 1], [between, within], color=[S.INK, "0.65"], width=0.6, edgecolor="none")
ax.text(0, between + 0.0005, f"{100 * between / (between + within):.0f}%", ha="center", fontsize=7)
ax.text(1, within + 0.0005, f"{100 * within / (between + within):.0f}%", ha="center", fontsize=7)
ax.format(
    xticks=[0, 1],
    xticklabels=["between", "within"],
    ylabel="Variance of Δ$A$",
    title="Spread",
    xtickminor=False,
    ytickminor=False,
    xlim=(-0.6, 1.6),
)
S.despine(ax)
FIG_DIR.mkdir(exist_ok=True)
fig.save(FIG_DIR / "figS3_smile.png", dpi=300)
fig.save(FIG_DIR / "figS3_smile.pdf")
print("saved figS3")
