"""Figure 2: what El Nino and La Nina do as the world warms. Panel c is coloured by the change of ENSO amplitude,
the percent change of the summed El Nino and La Nina contributions, C_EN + C_LN."""

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import ultraplot as uplt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ensoasym import style as S  # noqa: E402
from ensoasym.io import FIG_DIR  # noqa: E402
from ensoasym.plotting import (
    load_tables,
    member_pattern,
    per_model,
    numbered_marker,
    open_marker,
    repel_markers,
    TABLES,
)

S.apply()
uplt.rc.update({"axes.labelsize": 9, "tick.labelsize": 8})
mod, mem, tr, obs = load_tables()
mem = member_pattern(mem)
models = list(mod.index)
mmci = pd.read_csv(TABLES / "multimodel_medians.csv").set_index("quantity")
fid = pd.read_csv(TABLES / "model_fidelity.csv", index_col=0)

# percent change of each side's variance contribution, relative to the member's
# own present-day (GWI < 1 K) value, in 1 K bins of global warming every 0.25 K
ref = mem.set_index(["model", "rid"])[["C_EN_ref", "C_LN_ref"]]
tr2 = tr[tr.model.isin(models)].join(ref, on=["model", "rid"], how="inner")
for side in ("EN", "LN"):
    tr2[f"pct_{side}"] = 100 * (tr2[f"C_{side}"] / tr2[f"C_{side}_ref"] - 1)
pm = (
    tr2.groupby(["model", "mid"]).agg(n=("rid", "size"), EN=("pct_EN", "median"), LN=("pct_LN", "median")).reset_index()
)
pm = pm[pm.n >= 5]
mmm = pm.groupby("mid").agg(
    EN=("EN", "median"),
    LN=("LN", "median"),
    EN25=("EN", lambda v: np.percentile(v, 25)),
    EN75=("EN", lambda v: np.percentile(v, 75)),
    LN25=("LN", lambda v: np.percentile(v, 25)),
    LN75=("LN", lambda v: np.percentile(v, 75)),
    nm=("model", "size"),
)
mmm = mmm[mmm.nm >= 8]

fig, axs = uplt.subplots(
    [[1, 2, 3]], wratios=[1, 1, 1.3], figwidth=7.4, refheight=2.25, wspace=(1.6, 6.2), share=False
)

for ax, side, col, light, name in (
    (axs[0], "EN", S.EN, S.EN_LIGHT, "El Niño"),
    (axs[1], "LN", S.LN, S.LN_LIGHT, "La Niña"),
):
    for m in models:
        d = pm[pm.model == m].sort_values("mid")
        if len(d) < 3:
            continue
        ax.plot(d["mid"], d[side], color=light, lw=0.7, alpha=0.6)
    ax.fill_between(mmm.index, mmm[f"{side}25"], mmm[f"{side}75"], color=col, alpha=0.18, lw=0)
    ax.plot(mmm.index, mmm[side], color=col, lw=2.4, solid_capstyle="round")
    ax.axhline(0, color=S.MUTED, lw=0.6, ls=(0, (3, 2)))
    ax.format(
        xlabel="Global warming (K)",
        title=f"{name} variance contribution",
        titlesize=8.8,
        ylim=(-60, 120),
        xlim=(0.5, 4.75),
        xticks=[1, 2, 3, 4],
        xtickminor=False,
        ytickminor=False,
        ylabel="Change from present day (%)" if side == "EN" else "",
    )
    if side == "LN":
        ax.format(yticklabels=[])
    q = mmci.loc[f"d_C_{side}_pct"]
    S.despine(ax)

ax = axs[2]
sub = mem[mem.model.isin(models)].copy()
sub["pEN"] = 100 * sub["d_C_EN"] / sub["C_EN_ref"]
sub["pLN"] = 100 * sub["d_C_LN"] / sub["C_LN_ref"]
pe = per_model(sub, "pEN").loc[models]
pl = per_model(sub, "pLN").loc[models]
sub["pAmp"] = 100 * (sub["d_C_EN"] + sub["d_C_LN"]) / (sub["C_EN_ref"] + sub["C_LN_ref"])
amp = per_model(sub, "pAmp").loc[models]
print("amplitude change (C_EN + C_LN), percent:", amp["med"].round(0).describe().round(1).to_dict())
AMP_NORM = mcolors.Normalize(-100, 100)  # symmetric, so that -50 and +50 sit at the same distance from 0
AMP_CMAP = mcolors.LinearSegmentedColormap.from_list("amplitude", [S.LN, "#f7f7f7", S.EN])  # the colours of panels a and b
lim = (-75, 140)
ax.fill_between(lim, lim, [lim[1], lim[1]], color=S.LN, alpha=0.06, lw=0)
ax.fill_between(lim, [lim[0], lim[0]], lim, color=S.EN, alpha=0.06, lw=0)
ax.plot(lim, lim, color=S.MUTED, lw=0.7, ls=(0, (3, 2)))
ax.axhline(0, color=S.INK, lw=0.6, ls=(0, (1, 2)), zorder=1)
ax.axvline(0, color=S.INK, lw=0.6, ls=(0, (1, 2)), zorder=1)
ax.format(xlim=lim, ylim=lim)
ax.figure.set_size_inches(7.4, 2.75)
xp = repel_markers(ax, pe.loc[models, "med"].values, pl.loc[models, "med"].values, min_pts=13.5)
for (m, (xd, yd)) in zip(models, xp):
    c = AMP_CMAP(AMP_NORM(amp.loc[m, "med"]))
    x0, y0 = pe.loc[m, "med"], pl.loc[m, "med"]
    ax.errorbar(x0, y0, xerr=1.645 * pe.loc[m, "se"], yerr=1.645 * pl.loc[m, "se"], fmt="none", ecolor=S.INK, elinewidth=0.5, alpha=0.5, zorder=3)
    if abs(xd - x0) + abs(yd - y0) > 1e-6:
        ax.plot([x0, xd], [y0, yd], color=S.INK, lw=0.4, zorder=4)
    if fid.loc[m, "cold_tongue"]:
        numbered_marker(ax, xd, yd, int(mod.loc[m, "num"]), c, size=125, fontsize=6.0)
    else:
        open_marker(ax, xd, yd, int(mod.loc[m, "num"]), c, size=130, fontsize=6.0)
ax.format(
    xlim=lim,
    ylim=lim,
    xlabel="El Niño variance change (%)",
    ylabel="La Niña variance change (%)",
    titlesize=8.8,
    title="Above 3 K of warming",
    xticks=[-50, 0, 50, 100],
    yticks=[-50, 0, 50, 100],
    xtickminor=False,
    ytickminor=False,
)
S.despine(ax)
ax.colorbar(
    mpl.cm.ScalarMappable(norm=AMP_NORM, cmap=AMP_CMAP),
    loc="r",
    width=0.1,
    length=0.9,
    space=1.7,
    label="Total ENSO variance change\n(>3 K minus present, %)",
    extend="both",
    ticks=[-100, -50, 0, 50, 100],
    tickminor=False,
)

fig.format(abc="a", abcloc="l", abcsize=9)
FIG_DIR.mkdir(exist_ok=True)
fig.save(FIG_DIR / "fig2_phases_vs_warming.png", dpi=300)
fig.save(FIG_DIR / "fig2_phases_vs_warming.pdf")
print("saved fig2")
