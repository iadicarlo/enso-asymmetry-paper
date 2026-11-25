"""Figure 1: detecting forced ENSO asymmetry changes requires large ensembles.

(a) Asymmetry change of every member, the ensemble median and its 5 to 95 percent interval, the residual uncertainty from
internal variability once the ensemble is averaged. Significant medians (interval excluding zero) are red toward El Nino
and blue toward La Nina. (b) For the nine ensembles with a significant change: the fraction of random subsets of k members,
drawn without replacement, whose median differs from zero (same test), against k. Each curve reaches 100 percent at the
full ensemble size; the curves are averaged over three consecutive sizes. The standard error of a median of k members with standard deviation s is 1.253 s / sqrt(k).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import ultraplot as uplt
from matplotlib.lines import Line2D
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ensoasym import style as S  # noqa: E402
from ensoasym.io import FIG_DIR, PRETTY, ROOT  # noqa: E402
from ensoasym.plotting import load_tables  # noqa: E402

S.apply()
uplt.rc.update({"axes.labelsize": 9, "tick.labelsize": 8})
mod, mem, tr, obs = load_tables()
models = list(mod.index)
rng = np.random.default_rng(0)
NDRAW = 20000
GRAY = "#8c8c8c"
ALPHA = 0.10  # two-sided level of the 5 to 95 percent interval


def pvalue(d):
    """Two-sided p-value that the median of each row of d differs from zero."""
    d = np.atleast_2d(d)
    k = d.shape[1]
    se = 1.253 * d.std(axis=1, ddof=1) / np.sqrt(k)
    return 2 * stats.t.sf(np.abs(np.median(d, axis=1)) / se, k - 1)


rows, curves = [], []
for m in models:
    d = mem[mem.model == m]["d_A"].values
    n, med = len(d), np.median(d)
    half = stats.t.ppf(1 - ALPHA / 2, n - 1) * 1.253 * np.std(d, ddof=1) / np.sqrt(n)
    sig = abs(med) > half
    rows.append(dict(model=m, n=n, med=med, sd=np.std(d, ddof=1), half=half, sig=sig, num=int(mod.loc[m, "num"])))
    if sig:
        idx = np.argsort(rng.random((NDRAW, n)), axis=1)  # one set of random orders, so the subsets are nested across k
        ks = np.arange(3, n + 1)
        raw = np.array([100.0 if k == n else 100 * float(np.mean(pvalue(d[idx[:, :k]]) < ALPHA)) for k in ks])
        sm = raw.copy()  # the median of an even and of an odd number of members alternate: average over three consecutive sizes
        sm[1:-1] = 0.25 * raw[:-2] + 0.5 * raw[1:-1] + 0.25 * raw[2:]
        sm = np.maximum.accumulate(sm)
        for k, f0, f1 in zip(ks, raw, sm):
            curves.append(dict(model=m, k=k, frac_raw=f0, frac=f1))
ens = pd.DataFrame(rows).set_index("model").sort_values("num", ascending=False)  # (1) at the top
cur = pd.DataFrame(curves)
ens.to_csv(ROOT / "results/tables/ensemble_interval.csv", float_format="%.4f")
cur.to_csv(ROOT / "results/tables/ensemble_detection_by_members.csv", index=False, float_format="%.2f")
print(ens[["n", "med", "half", "sig"]].round(3).to_string())
print("significant:", int(ens.sig.sum()), "toward LN", int((ens.sig & (ens.med < 0)).sum()), "toward EN", int((ens.sig & (ens.med > 0)).sum()))
need = {}
for m in ens.index[ens.sig]:
    q = cur[cur.model == m].sort_values("k")
    need[m] = {lvl: int(q.k[q.frac >= lvl].iloc[0]) for lvl in (50, 80)}
print(pd.DataFrame(need).T.assign(n=ens.n).to_string())
print("members for 80 percent detection: median", np.median([v[80] for v in need.values()]), "range",
      min(v[80] for v in need.values()), max(v[80] for v in need.values()))


def colour(r):
    return GRAY if not r.sig else (S.EN if r.med > 0 else S.LN)


fig, axs = uplt.subplots(ncols=2, wratios=(1.3, 1), figwidth=7.4, refheight=3.5, wspace=4.0, share=False)
ax = axs[0]
ys = np.arange(len(ens))
for y, (m, r) in zip(ys, ens.iterrows()):
    d = mem[mem.model == m]["d_A"].values
    c = colour(r)
    ax.scatter(d, y + rng.uniform(-0.2, 0.2, len(d)), s=5, color=GRAY, alpha=0.5, lw=0, zorder=2)
    ax.plot([r.med - r.half, r.med + r.half], [y, y], color=c if r.sig else S.INK, lw=2.2, solid_capstyle="butt", zorder=4)
    ax.scatter(r.med, y, s=34, facecolor=c if r.sig else "white", edgecolor=c if r.sig else S.INK, lw=0.9, zorder=5)
ax.axvline(0, color=S.MUTED, lw=0.6, ls=(0, (3, 2)), zorder=1)
ax.format(
    yticks=ys,
    yticklabels=[f"({int(r.num)})  |  {PRETTY[m]}  |  n = {int(r.n)}" for m, r in ens.iterrows()],
    ylim=(-0.7, len(ens) - 0.3),
    xlim=(-0.55, 0.72),
    xlabel="Asymmetry change Δ$A$ (>3 K minus present)",
    title="Forced asymmetry change and internal variability",
    titlesize=9.5,
    ytickminor=False,
    xtickminor=False,
)
ax.tick_params(axis="y", length=0, labelsize=7.2)
S.despine(ax, keep=("bottom", "left"))
ax.spines["left"].set_linewidth(0)
handles = [
    Line2D([], [], marker="o", ls="none", ms=2.6, color=GRAY, alpha=0.7, label="member"),
    Line2D([], [], marker="o", ls="-", lw=2.2, ms=5.5, color=S.INK, mfc="white", mec=S.INK, label="median and 5–95% interval"),
    Line2D([], [], marker="o", ls="-", lw=2.2, ms=5.5, color=S.EN, label="significant, toward El Niño"),
    Line2D([], [], marker="o", ls="-", lw=2.2, ms=5.5, color=S.LN, label="significant, toward La Niña"),
]
fig.legend(handles=handles, loc="b", ncols=2, frame=False, fontsize=7.4, handlelength=2.2, columnspacing=3.0, space=5.5)

ax = axs[1]
ends = []
for m, r in ens[ens.sig].iterrows():
    q = cur[cur.model == m].sort_values("k")
    ax.plot(q.k.values, q.frac.values, color=colour(r), lw=1.3, alpha=0.9, zorder=3)
    ax.scatter([q.k.iloc[-1]], [100], s=16, color=colour(r), edgecolor="white", lw=0.4, zorder=4, clip_on=False)
    ends.append((q.k.iloc[-1], int(r.num), colour(r)))
order = sorted(ens.index[ens.sig], key=lambda m_: need[m_][50])
LEVELS = [86, 68, 50, 32, 77, 59, 41, 23, 68]
for m, lvl in zip(order, LEVELS):  # ensemble number written on its own curve, at staggered heights
    q = cur[cur.model == m].sort_values("k")
    kk = np.interp(lvl, q.frac.values, np.log(q.k.values)) if q.frac.is_monotonic_increasing else np.log(q.k[q.frac >= lvl].iloc[0])
    ax.text(np.exp(kk), lvl, f"{int(ens.loc[m, 'num'])}", ha="center", va="center_baseline", fontsize=7, color=colour(ens.loc[m]), fontweight="bold",
            zorder=6, bbox=dict(boxstyle="circle,pad=0.18", facecolor="white", edgecolor=colour(ens.loc[m]), lw=0.8))
ax.format(
    xscale="log",
    xlim=(2.8, 115),
    ylim=(0, 102),
    xticks=[3, 5, 10, 20, 50, 100],
    xticklabels=["3", "5", "10", "20", "50", "100"],
    yticks=[0, 25, 50, 75, 100],
    xlabel="Members in the subset",
    ylabel="Fraction of subsets with significant Δ$A$ (%)",
    title="Detection probability vs ensemble size",
    titlesize=9.5,
    xtickminor=False,
    ytickminor=False,
)
S.despine(ax)
fig.format(abc="a", abcloc="l", abcsize=10)
fig.save(FIG_DIR / "fig1_large_ensembles.png", dpi=300)
fig.save(FIG_DIR / "fig1_large_ensembles.pdf")
print("saved fig1")
