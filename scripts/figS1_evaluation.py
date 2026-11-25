"""Figure S1: per-ensemble quantities against observations (5 to 95 percent of members). Panels b and c are computed
member by member over the observed years, 1980 to 2025 (scripts/member_trends.py), so that each ensemble's spread is
the spread of single 46-year realisations, like the observed record."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import ultraplot as uplt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ensoasym import style as S  # noqa: E402
from ensoasym.io import FIG_DIR  # noqa: E402
from ensoasym.plotting import load_tables, member_pattern, per_model, label, rain_color, TABLES  # noqa: E402

S.apply()
mod, mem, tr, obs = load_tables()
mem = member_pattern(mem)
models = list(mod.index)
rain = per_model(mem, "pr_nino3_ref").loc[models]
m80 = pd.read_csv(TABLES / "members_1980_2025.csv")
aref = per_model(m80, "A").loc[models]
skew = per_model(m80, "skew_n3").loc[models]
om = pd.read_csv(TABLES / "obs_multi.csv")
fid = pd.read_csv(TABLES / "model_fidelity.csv", index_col=0)
o6 = om[(om.sst == "ERSSTv6") & (om.pr == "GPCP")].iloc[0]
oc = om[(om.sst == "ERSSTv6") & (om.pr == "CMAP")].iloc[0]
omg = om[om.pr == "GPCP"].set_index("sst")
obs_rain, obs_rain_lo, obs_rain_hi = (
    o6.pr_nino3_ref,
    min(o6.pr_nino3_ref, oc.pr_nino3_ref),
    max(o6.pr_nino3_ref, oc.pr_nino3_ref),
)
obs_A, obs_A_lo, obs_A_hi = o6.A, omg["A"].min(), omg["A"].max()
obs_skew, obs_skew_lo, obs_skew_hi = o6.skew_n3, omg["skew_n3"].min(), omg["skew_n3"].max()

order = rain.sort_values("med").index  # driest on top
y = np.arange(len(order))[::-1]
top = len(order) - 1
fig, axs = uplt.subplots(ncols=3, refwidth=1.6, refheight=3.6, wspace=1.6, share=False)


def lollipop(ax, tab, colour_fn, obs_val=None, obs_lo=None, obs_hi=None, obs_text=None):
    if obs_val is not None:
        ax.axvspan(obs_lo, obs_hi, color=S.ACCENT, alpha=0.14, lw=0)
        ax.axvline(obs_val, color=S.ACCENT, lw=1.2, ls=(0, (4, 2)), zorder=3)
    for yi, m in zip(y, order):
        r = tab.loc[m]
        c = colour_fn(r["med"])
        ax.plot([r["p5"], r["p95"]], [yi, yi], color=c, lw=1.3, alpha=0.5, solid_capstyle="round")
        if fid.loc[m, "cold_tongue"]:
            ax.scatter(r["med"], yi, s=36, color=c, edgecolor="white", lw=0.6, zorder=5)
        else:
            ax.scatter(r["med"], yi, s=36, facecolor="white", edgecolor=c, lw=1.2, zorder=5)
    if obs_val is not None:
        ax.text(
            obs_val,
            top + 0.85,
            obs_text,
            color=S.INK,
            ha="center",
            va="center",
            fontsize=6.6,
            fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", pad=1.5),
        )
    ax.format(ylim=(-1.8, top + 1.4), ytickminor=False, xtickminor=False)
    S.despine(ax, keep=("bottom", "left"))
    ax.spines["left"].set_linewidth(0)
    ax.tick_params(axis="y", length=0)


ax = axs[0]
lollipop(
    ax, rain, rain_color, obs_rain, obs_rain_lo, obs_rain_hi, "obs"
)
ax.format(
    xlabel="Annual-mean Niño-3 rainfall,\npresent day (mm/day)", title="Eastern Pacific state", xlim=(0.3, 3.4)
)
ax.set_yticks(
    y, labels=[f"{int(mod.loc[m, 'num'])}  {label(m)}  ({int(rain.loc[m, 'n'])})" for m in order], fontsize=7.5
)
ax.tick_params(axis="y", length=0)

ax = axs[1]
ax.axvline(0, color=S.MUTED, lw=0.6, ls=(0, (3, 2)))
lollipop(ax, skew, lambda v: S.EN if v > 0 else S.LN, obs_skew, obs_skew_lo, obs_skew_hi, "obs")
ax.format(xlabel="Niño-3 skewness,\n1980 to 2025", title="Skewness", xlim=(-0.7, 1.25))
ax.set_yticks(y, labels=[""] * len(y))
ax.tick_params(axis="y", length=0)

ax = axs[2]
ax.axvline(0, color=S.MUTED, lw=0.6, ls=(0, (3, 2)))
lollipop(ax, aref, lambda v: S.EN if v > 0 else S.LN, obs_A, obs_A_lo, obs_A_hi, "obs")
ax.format(xlabel="Asymmetry index $A$,\n1980 to 2025", title="Asymmetry", xlim=(-0.4, 0.4))
ax.set_yticks(y, labels=[""] * len(y))
ax.tick_params(axis="y", length=0)

FIG_DIR.mkdir(exist_ok=True)
fig.save(FIG_DIR / "figS1_evaluation.png", dpi=300)
fig.save(FIG_DIR / "figS1_evaluation.pdf")
print("saved figS1")
