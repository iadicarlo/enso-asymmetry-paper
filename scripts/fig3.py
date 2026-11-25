"""Figure 3: the asymmetry change against the present-day eastern Pacific state and against the forced warming pattern.
The bars are 5 to 95 percent intervals of the ensemble medians, as in Figure 2."""

import sys
from pathlib import Path

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
    rain_color,  # noqa: E402
    rain_mappable,
    fit_line,
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
rain = per_model(mem, "pr_nino3_ref").loc[models]
grad = per_model(mem, "grad_per_K").loc[models]
dA = per_model(mem, "d_A").loc[models]
boot = pd.read_csv(TABLES / "bootstrap.csv")
part = pd.read_csv(TABLES / "partial_correlations.csv")
con = pd.read_csv(TABLES / "constrained_estimates.csv")
om = pd.read_csv(TABLES / "obs_multi.csv")
o6 = om[(om.sst == "ERSSTv6") & (om.pr == "GPCP")].iloc[0]
oc = om[(om.sst == "ERSSTv6") & (om.pr == "CMAP")].iloc[0]
omg = om[om.pr == "GPCP"].set_index("sst")
obs_grad = o6.sst_gradient_per_K
obs_lo = float((omg["sst_gradient_per_K"] - omg["sst_gradient_per_K_se"]).min())
obs_hi = float((omg["sst_gradient_per_K"] + omg["sst_gradient_per_K_se"]).max())


def brow(x, y):
    return boot[(boot.x == x) & (boot.y == y) & (boot.design == "gwi_bins")].iloc[0]


fig, axs = uplt.subplots(ncols=2, figwidth=6.8, refheight=2.9, wspace=6.0, share=False)
fid = pd.read_csv(TABLES / "model_fidelity.csv", index_col=0)
Z90 = 1.645  # 5 to 95 percent interval from the standard error of the median
YLIM = (-0.27, 0.27)


def scatter(ax, xs, xgrid, xlabel, ylabel, title, stat, part_r, box_xy, xlim, given, fixed=()):
    x = xs.loc[models, "med"].values
    y = dA.loc[models, "med"].values
    f = fit_line(x, y, xgrid=xgrid)
    ax.fill_between(f["xgrid"], f["yhat"] - f["band"], f["yhat"] + f["band"], color=S.INK, alpha=0.07, lw=0)
    ax.plot(f["xgrid"], f["yhat"], color=S.INK, lw=1.1)
    ax.axhline(0, color=S.MUTED, lw=0.6, ls=(0, (3, 2)))
    ax.format(xlim=xlim, ylim=YLIM)
    xp = repel_markers(ax, x, y, min_pts=14.0, fixed=fixed)
    for m, (xd, yd) in zip(models, xp):
        c = rain_color(rain.loc[m, "med"])
        x0, y0 = xs.loc[m, "med"], dA.loc[m, "med"]
        ax.errorbar(x0, y0, xerr=Z90 * xs.loc[m, "se"], yerr=Z90 * dA.loc[m, "se"], fmt="none", ecolor=S.INK,
                    elinewidth=0.5, alpha=0.5, zorder=3)
        if abs(xd - x0) + abs(yd - y0) > 1e-6:
            ax.plot([x0, xd], [y0, yd], color=S.INK, lw=0.4, zorder=4)
        (numbered_marker if fid.loc[m, "cold_tongue"] else open_marker)(ax, xd, yd, int(mod.loc[m, "num"]), c, size=135, fontsize=6.2)
    txt = (f"r = {stat['r']:+.2f}, p = {stat['p_perm']:.3f}\npartial r = {part_r.r_partial:+.2f}, p = {part_r.p_partial:.3f}".replace("-", "−")
           + f"\ncontrolling for {given}")
    ax.text(*box_xy, txt, transform=ax.transAxes, fontsize=7.4, fontweight="bold", color=S.INK, ha="right" if box_xy[0] > 0.5 else "left",
            va="top", multialignment="left", zorder=8, linespacing=1.3,
            bbox=dict(facecolor="white", edgecolor=S.INK, lw=0.7, boxstyle="round,pad=0.35"))
    ax.format(xlabel=xlabel, ylabel=ylabel, title=title, xlim=xlim, ylim=YLIM, titlesize=9.5, xtickminor=False, ytickminor=False)
    S.despine(ax)


# (a) present-day Nino-3 rainfall
ax = axs[0]
scatter(
    ax,
    rain,
    np.linspace(0.5, 3.2, 100),
    "Niño-3 present-day rainfall climatology (mm/day)",
    "Asymmetry change Δ$A$ (>3 K minus present)",
    "Present-day Niño-3 rainfall climatology",
    brow("pr_nino3_ref", "d_A"),
    part[(part.predictor == "pr_nino3_ref") & (part.controlling_for == "grad_per_K")].iloc[0],
    (0.97, 0.97),
    (0.45, 3.3),
    "SST gradient",
    fixed=[(o6.pr_nino3_ref, 0.0)],
)
ax.axvspan(
    min(o6.pr_nino3_ref, oc.pr_nino3_ref), max(o6.pr_nino3_ref, oc.pr_nino3_ref), color=S.OBS_BAND, alpha=0.3, lw=0
)
ax.axvline(o6.pr_nino3_ref, color=S.ACCENT, lw=0.6, ls=(0, (4, 2)), zorder=1)
c = con[(con.predictor == "pr_nino3_ref") & (con.pr == "GPCP")].iloc[0]
ax.errorbar(
    c.obs_value,
    c.dA,
    xerr=[[abs(o6.pr_nino3_ref - oc.pr_nino3_ref) / 2 + 0.02]],
    yerr=[[c.dA - c.lo68], [c.hi68 - c.dA]],
    fmt="none",
    ecolor=S.ACCENT,
    capsize=0,
    lw=1.0,
    zorder=2,
)
ax.scatter([c.obs_value], [c.dA], s=125, marker="*", color=S.ACCENT, edgecolor="white", lw=0.5, zorder=9)
ax.text(o6.pr_nino3_ref, YLIM[0] + 0.012, "obs", fontsize=8, fontweight="bold", color=S.ACCENT, ha="center", va="bottom", zorder=9,
        bbox=dict(facecolor="white", edgecolor="none", pad=1.0))
# (b) warming pattern, with the partial correlation
ax = axs[1]
pr = part[(part.predictor == "grad_per_K") & (part.controlling_for == "pr_nino3_ref")].iloc[0]
scatter(
    ax,
    grad,
    np.linspace(-0.36, 0.12, 100),
    "Forced zonal SST-gradient change\n(west minus east; K/K)",
    "",
    "Forced zonal SST-gradient change",
    brow("grad_per_K", "d_A"),
    pr,
    (0.03, 0.97),
    (-0.42, 0.22),
    "Niño-3 rainfall",
)
ax.axvline(0, color=S.MUTED, lw=0.6, ls=(0, (3, 2)))
fig.colorbar(
    rain_mappable(),
    loc="b",
    length=0.42,
    width=0.1,
    extend="max",
    label="Niño-3 rainfall (mm/day)",
    ticks=[0, 1, 2, 3],
    tickminor=False,
)
fig.format(abc="a", abcloc="l", abcsize=9)

FIG_DIR.mkdir(exist_ok=True)
fig.save(FIG_DIR / "fig3_pattern_sets_asymmetry.png", dpi=300)
fig.save(FIG_DIR / "fig3_pattern_sets_asymmetry.pdf")
print("saved fig3")
