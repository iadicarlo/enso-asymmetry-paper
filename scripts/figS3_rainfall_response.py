"""Figure S3: the rainfall response per K of SST anomaly, from the ensembles and observations.

Since 2026-09-22 the rainfall responses are per K of the mean Nino-3 SST anomaly of each phase (s_EN = P_EN / T_EN,
s_LN = P_LN / T_LN, scripts/rain_response.py), as suggested by Yann Planton; the unnormalised version is kept in
figures/archive/fig4_unnormalised_2026-09-22/."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import ultraplot as uplt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ensoasym import style as S  # noqa: E402
from ensoasym.io import FIG_DIR  # noqa: E402
from ensoasym.indices import EN_ALL, LN_ALL  # noqa: E402
from ensoasym.obs import obs_indices  # noqa: E402
from ensoasym.plotting import load_tables, per_model, rain_color, fit_line, numbered_marker, open_marker, repel_markers, rain_mappable, TABLES  # noqa: E402


def fmt_p(p):
    return "p < 0.01" if p < 0.01 else f"p = {p:.2f}"


S.apply()
# titles carry the message, so they read larger than the axis labels and the ticks
uplt.rc.update({"axes.labelsize": 7.2, "tick.labelsize": 6.8})
mod, mem, tr, obs = load_tables()
rr = pd.read_csv(TABLES / "members_rain_response.csv")
j = rr.merge(mem[["model", "rid", "d_A", "A_ref", "pr_nino3_ref"]], on=["model", "rid"])
models = [m for m in mod.index if m in set(j.model)]
rain = per_model(j, "pr_nino3_ref").loc[models]
PEN = per_model(j, "s_EN_ref").loc[models]  # mm/day per K, positive
PLN = per_model(j, "s_LN_ref").loc[models]  # mm/day per K, positive
Rref = per_model(j, "sR_ref").loc[models]
Aref = per_model(j, "A_ref").loc[models]
dPLN = per_model(j, "d_s_LN").loc[models]
dPEN = per_model(j, "d_s_EN").loc[models]
dA = per_model(j, "d_A").loc[models]
fid = pd.read_csv(TABLES / "model_fidelity.csv", index_col=0)

# observed rainfall responses (ERSSTv6 phases, GPCP rainfall)
o = obs_indices("ERSSTv6", "GPCP")
import xarray as xr  # noqa: E402
from ensoasym.obs import OBS, month_start  # noqa: E402
from ensoasym import events as ev  # noqa: E402

pr = month_start(xr.open_dataset(OBS / "boxes_GPCP.nc")["nino3"]).sel(time=slice("1980-01-01", "2025-12-01"))
pr_anom = ev.remove_seasonal_cycle_rolling_with_std(pr, window_years=30)["anomaly"]
pa = pr_anom.reindex(time=o["time"]).values
ph = o["phase"]
# Nino-3 SST anomaly of the same months (ERSSTv6, 30-year rolling seasonal cycle on the record from 1950)
from ensoasym.classification import rolling_anomaly  # noqa: E402

bx = month_start(xr.open_dataset(OBS / "boxes_ERSSTv6.nc")).sel(time=slice("1950-01-01", "2025-12-01"))
ta = pd.Series(rolling_anomaly(bx["nino3"].values, bx.time.values), index=pd.DatetimeIndex(bx.time.values))
ta = ta.reindex(pd.DatetimeIndex(o["time"])).values
raw_PEN = float(np.nanmean(pa[np.isin(ph, EN_ALL)]))
raw_PLN = float(np.nanmean(pa[np.isin(ph, LN_ALL)]))
obs_TEN = float(np.nanmean(ta[np.isin(ph, EN_ALL)]))
obs_TLN = float(np.nanmean(ta[np.isin(ph, LN_ALL)]))
obs_PEN, obs_PLN = raw_PEN / obs_TEN, raw_PLN / obs_TLN  # per K, both positive
yrs = pd.DatetimeIndex(o["time"]).year
uy = np.unique(yrs)
rng = np.random.default_rng(3)
bEN, bLN = [], []
for _ in range(1000):
    pick = rng.choice(uy, size=uy.size, replace=True)
    sel = np.concatenate([np.where(yrs == y)[0] for y in pick])
    pa_s, ph_s, ta_s = pa[sel], ph[sel], ta[sel]
    en_s, ln_s = np.isin(ph_s, EN_ALL), np.isin(ph_s, LN_ALL)
    bEN.append(np.nanmean(pa_s[en_s]) / np.nanmean(ta_s[en_s]))
    bLN.append(np.nanmean(pa_s[ln_s]) / np.nanmean(ta_s[ln_s]))
bEN, bLN = np.array(bEN), np.array(bLN)
se_PEN, se_PLN, se_R = np.nanstd(bEN), np.nanstd(bLN), np.nanstd(bEN - bLN)
om_ = pd.read_csv(TABLES / "obs_multi.csv")
o6_ = om_[(om_.sst == "ERSSTv6") & (om_.pr == "GPCP")].iloc[0]
obs_A_err = [[o6_.A - o6_.A_lo16], [o6_.A_hi84 - o6_.A]]
oc_ = om_[(om_.sst == "ERSSTv6") & (om_.pr == "CMAP")].iloc[0]
obs_rain_err = abs(o6_.pr_nino3_ref - oc_.pr_nino3_ref) / 2 + 0.02
om = pd.read_csv(TABLES / "obs_multi.csv")
o6 = om[(om.sst == "ERSSTv6") & (om.pr == "GPCP")].iloc[0]
obs_rain, obs_A = float(o6.pr_nino3_ref), float(o6.A)
print("obs P_EN %.2f P_LN %.2f (mm/day), T_EN %.2f T_LN %.2f (K), s_EN %.2f s_LN %.2f sR %.2f (mm/day per K), A %.2f"
      % (raw_PEN, raw_PLN, obs_TEN, obs_TLN, obs_PEN, obs_PLN, obs_PEN - obs_PLN, obs_A))
pd.DataFrame(
    [
        {
            "sst": "ERSSTv6",
            "pr": "GPCP",
            "P_EN": raw_PEN,
            "P_LN": raw_PLN,
            "T_EN": obs_TEN,
            "T_LN": obs_TLN,
            "s_EN": obs_PEN,
            "s_EN_se": se_PEN,
            "s_LN": obs_PLN,
            "s_LN_se": se_PLN,
            "sR": obs_PEN - obs_PLN,
            "sR_se": se_R,
            "A": obs_A,
        }
    ]
).to_csv(TABLES / "obs_rain_response.csv", index=False, float_format="%.4f")

fig, axs = uplt.subplots(
    ncols=3,
    refwidth=1.75,
    refheight=1.65,
    wspace=(8.0, 11.0),
    share=False,
    tight=False,
    left=6.0,
    right=7.0,
    bottom=4.5,
    top=2.5,
)

# (a) rainfall response to each phase versus the present-day rainfall
ax = axs[0]
xg = np.linspace(0.5, 3.2, 50)
fe = fit_line(rain.loc[models, "med"].values, PEN.loc[models, "med"].values, xgrid=xg)
fl = fit_line(rain.loc[models, "med"].values, PLN.loc[models, "med"].values, xgrid=xg)
ax.fill_between(xg, fe["yhat"] - fe["band"], fe["yhat"] + fe["band"], color=S.EN, alpha=0.08, lw=0)
ax.fill_between(xg, fl["yhat"] - fl["band"], fl["yhat"] + fl["band"], color=S.LN, alpha=0.08, lw=0)
ax.plot(xg, fe["yhat"], color=S.EN, lw=1.1)
ax.plot(xg, fl["yhat"], color=S.LN, lw=1.1)
ax.errorbar(
    rain.loc[models, "med"],
    PEN.loc[models, "med"],
    xerr=rain.loc[models, "se"],
    yerr=PEN.loc[models, "se"],
    fmt="none",
    ecolor=S.EN,
    elinewidth=0.5,
    alpha=0.6,
    zorder=3,
)
ax.errorbar(
    rain.loc[models, "med"],
    PLN.loc[models, "med"],
    xerr=rain.loc[models, "se"],
    yerr=PLN.loc[models, "se"],
    fmt="none",
    ecolor=S.LN,
    elinewidth=0.5,
    alpha=0.6,
    zorder=3,
)
ax.scatter(
    rain.loc[models, "med"],
    PEN.loc[models, "med"],
    s=32,
    marker="^",
    color=S.EN,
    edgecolor="white",
    lw=0.5,
    zorder=4,
    label=f"El Niño (r = {fe['r']:+.2f})",
)
ax.scatter(
    rain.loc[models, "med"],
    PLN.loc[models, "med"],
    s=32,
    marker="v",
    color=S.LN,
    edgecolor="white",
    lw=0.5,
    zorder=4,
    label=f"La Niña (r = {fl['r']:+.2f})",
)
ax.scatter([obs_rain], [obs_PEN], s=60, marker="^", color=S.EN, edgecolor=S.INK, lw=1.0, zorder=6)
ax.scatter([obs_rain], [obs_PLN], s=60, marker="v", color=S.LN, edgecolor=S.INK, lw=1.0, zorder=6)
ax.scatter([], [], s=60, marker="^", color="white", edgecolor=S.INK, lw=1.0, label="observed")
ax.legend(loc="ul", ncols=1, frame=False, fontsize=5.8, handletextpad=0.3, labelspacing=0.25, borderaxespad=0.15)
ax.format(
    xlabel="Present-day Niño-3 rainfall (mm/day)",
    ylabel="Rainfall response per K of\nNiño-3 SST (mm/day per K)",
    title="Rainfall response to each phase",
    titlesize=8,
    xlim=(0.45, 3.3),
    ylim=(0, 1.9),
    xtickminor=False,
    ytickminor=False,
)
S.despine(ax)

# (b) present-day rainfall asymmetry versus present-day ENSO asymmetry
ax = axs[1]
f = fit_line(Rref.loc[models, "med"].values, Aref.loc[models, "med"].values, xgrid=np.linspace(0.0, 0.7, 50))
ax.fill_between(f["xgrid"], f["yhat"] - f["band"], f["yhat"] + f["band"], color=S.INK, alpha=0.07, lw=0)
ax.plot(f["xgrid"], f["yhat"], color=S.INK, lw=1.1)
ax.axhline(0, color=S.MUTED, lw=0.6, ls=(0, (3, 2)))
ax.axvline(0, color=S.MUTED, lw=0.6, ls=(0, (3, 2)))
ax.format(xlim=(0, 0.7), ylim=(-0.20, 0.30))
FID = pd.read_csv(TABLES / "model_fidelity.csv", index_col=0)
xp = repel_markers(ax, Rref.loc[models, "med"].values, Aref.loc[models, "med"].values, min_pts=10.0, fixed=[(obs_PEN - obs_PLN, obs_A)])
for m, (xd, yd) in zip(models, xp):
    x0, y0 = Rref.loc[m, "med"], Aref.loc[m, "med"]
    ax.errorbar(x0, y0, xerr=Rref.loc[m, "se"], yerr=Aref.loc[m, "se"], fmt="none", ecolor=S.INK, elinewidth=0.4, alpha=0.35, zorder=3)
    if abs(xd - x0) + abs(yd - y0) > 1e-6:
        ax.plot([x0, xd], [y0, yd], color=S.INK, lw=0.4, zorder=4)
    (numbered_marker if FID.loc[m, "cold_tongue"] else open_marker)(ax, xd, yd, int(mod.loc[m, "num"]), rain_color(rain.loc[m, "med"]), size=72, fontsize=5.0)
ax.scatter(obs_PEN - obs_PLN, obs_A, s=120, marker="*", color=S.ACCENT, edgecolor="white", lw=0.5, zorder=7)
from matplotlib.lines import Line2D  # noqa: E402

ax.text(obs_PEN - obs_PLN, obs_A + 0.028, "obs", fontsize=7, fontweight="bold", ha="center", va="bottom", color=S.INK, zorder=8)
ax.text(
    0.03,
    0.96,
    f"n = {f['n']}\nr = {f['r']:+.2f} (p = {f['p']:.3f})",
    transform=ax.transAxes,
    fontsize=6.0,
    ha="left",
    va="top",
    multialignment="left",
    bbox=dict(facecolor="white", edgecolor=S.INK, lw=0.6, boxstyle="square,pad=0.35"),
    zorder=8,
)
ax.colorbar(
    rain_mappable(),
    extend="max",
    loc="r",
    width=0.09,
    space=0.6,
    length=0.9,
    label="Annual-mean Niño-3 rainfall,\npresent day (mm/day)",
    ticks=[0, 1, 2, 3],
    tickminor=False,
)
ax.format(
    xlabel="Rainfall asymmetry per K, $s_{\\mathrm{EN}}-s_{\\mathrm{LN}}$\n(mm/day per K)",
    ylabel="Present-day ENSO asymmetry $A$",
    title="Rainfall and ENSO asymmetries per K",
    titlesize=8,
    ylim=(-0.20, 0.30),
    xtickminor=False,
    ytickminor=False,
)
ax.title.set_fontsize(8)  # ultraplot shrinks a title that overflows a narrow panel; keep the size of the other panels
S.despine(ax)

# (c) change of the La Nina rainfall response with warming versus the present-day rainfall
ax = axs[2]
f2 = fit_line(rain.loc[models, "med"].values, dPLN.loc[models, "med"].values, xgrid=np.linspace(0.5, 3.2, 50))
ax.fill_between(f2["xgrid"], f2["yhat"] - f2["band"], f2["yhat"] + f2["band"], color=S.LN, alpha=0.08, lw=0)
ax.plot(f2["xgrid"], f2["yhat"], color=S.LN, lw=1.1)
f3 = fit_line(rain.loc[models, "med"].values, dPEN.loc[models, "med"].values, xgrid=np.linspace(0.5, 3.2, 50))
ax.fill_between(f3["xgrid"], f3["yhat"] - f3["band"], f3["yhat"] + f3["band"], color=S.EN, alpha=0.08, lw=0)
ax.plot(f3["xgrid"], f3["yhat"], color=S.EN, lw=1.1)
ax.axhline(0, color=S.MUTED, lw=0.6)
ax.errorbar(
    rain.loc[models, "med"],
    dPEN.loc[models, "med"],
    xerr=rain.loc[models, "se"],
    yerr=dPEN.loc[models, "se"],
    fmt="none",
    ecolor=S.EN,
    elinewidth=0.5,
    alpha=0.6,
    zorder=3,
)
ax.errorbar(
    rain.loc[models, "med"],
    dPLN.loc[models, "med"],
    xerr=rain.loc[models, "se"],
    yerr=dPLN.loc[models, "se"],
    fmt="none",
    ecolor=S.LN,
    elinewidth=0.5,
    alpha=0.6,
    zorder=3,
)
ax.scatter(
    rain.loc[models, "med"],
    dPEN.loc[models, "med"],
    s=32,
    marker="^",
    color=S.EN,
    edgecolor="white",
    lw=0.5,
    zorder=4,
    label=f"El Niño (r = {f3['r']:+.2f}, {fmt_p(f3['p'])})",
)
ax.scatter(
    rain.loc[models, "med"],
    dPLN.loc[models, "med"],
    s=32,
    marker="v",
    color=S.LN,
    edgecolor="white",
    lw=0.5,
    zorder=4,
    label=f"La Niña (r = {f2['r']:+.2f}, {fmt_p(f2['p'])})",
)
ax.legend(loc="ul", ncols=1, frame=False, fontsize=5.8, handletextpad=0.3, labelspacing=0.25, borderaxespad=0.15)
ax.format(
    xlabel="Present-day Niño-3 rainfall (mm/day)",
    ylabel="Change of the response per K\n(>3 K minus present, mm/day per K)",
    ylim=(-0.3, 1.45),
    title="Change under warming",
    titlesize=8,
    xlim=(0.45, 3.3),
    xtickminor=False,
    ytickminor=False,
)
S.despine(ax)

FIG_DIR.mkdir(exist_ok=True)
fig.save(FIG_DIR / "figS3_rainfall_response.png", dpi=300)
fig.save(FIG_DIR / "figS3_rainfall_response.pdf")
print("saved fig4 mechanism")
