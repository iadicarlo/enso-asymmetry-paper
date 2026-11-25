"""Figure 4: summary of the paper, built from data. (a) Perspective maps of the composite present-day annual-mean SST of
the six driest and six wettest ensembles (Nino-3 rainfall), with the 28C isotherm and clouds where the equatorial SST
exceeds 28C, and how the two variance contributions change above 3 K of global warming. (b) The 18 ensembles and the
observations on the axis of present-day Nino-3 rainfall. (c) The cross-model relations reported in the paper."""

import sys
from pathlib import Path

import cartopy.feature as cfeature
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely.affinity as saff
import shapely.geometry as sgeom
import xarray as xr
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, PathPatch, Polygon
from matplotlib.path import Path as MPath
from matplotlib.transforms import Affine2D
from scipy import stats
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ensoasym import style as S  # noqa: E402
from ensoasym.io import FIG_DIR, ROOT  # noqa: E402
from ensoasym.plotting import load_tables, numbered_marker, open_marker, rain_color  # noqa: E402

S.apply()
mpl.rcParams.update(
    {"font.size": 6.5, "axes.labelsize": 5.9, "xtick.labelsize": 5.6, "ytick.labelsize": 5.6, "axes.linewidth": 0.5}
)
mod, mem, tr, obs = load_tables()
FID = ROOT / "data/lorenz/fidelity"
order = mod.sort_values("pr_nino3_ref").index
GROUPS = [
    ("dry", list(order[:6]), "Six driest ensembles"),
    ("wet", list(order[-6:]), "Six wettest ensembles"),
]
GRID = xr.open_dataset(ROOT / "data/obs/maps_COBE2_g025.nc")  # the 2.5 degree grid of the archive
LAT, LON = GRID.lat, GRID.lon
X0, X1 = float(LON.min()) - 1.25, float(LON.max()) + 1.25  # the slab is exactly the data box
Y0, Y1 = float(LAT.min()) - 1.25, float(LAT.max()) + 1.25
SH = Affine2D().skew_deg(32, 0)  # oblique view: rows further north are shifted to the right
om = pd.read_csv(ROOT / "results/tables/obs_multi.csv")
o6 = om[(om.sst == "ERSSTv6") & (om.pr == "GPCP")].iloc[0]
OBS_CT, OBS_PR = float(o6.cold_tongue_ref), float(o6.pr_nino3_ref)


def sh(x, y):
    return np.asarray(SH.transform_point((x, y)))


MEMBER_MAPS = xr.open_dataset(FID / "member_maps.nc")  # per-member maps, each member's own scenario


def composite_sst(models):
    """Present-day SST (GWI below 1 K), mean over all members of each ensemble, then over ensembles."""
    arrs = []
    for m in models:
        sel = np.where(MEMBER_MAPS.model.values == m)[0]
        if sel.size:
            arrs.append(MEMBER_MAPS["sst_present"].isel(member=sel).mean("member").interp(lat=LAT, lon=LON).values)
    return np.nanmean(np.array(arrs), axis=0)


def group_stats(models):
    sub = mem[mem.model.isin(models)].copy()
    sub["pEN"] = 100 * sub["d_C_EN"] / sub["C_EN_ref"]
    sub["pLN"] = 100 * sub["d_C_LN"] / sub["C_LN_ref"]
    cols = ["pEN", "pLN", "d_A", "cold_tongue_sst_ref", "cold_tongue_sst_warm", "pr_nino3_ref", "pr_nino3_warm"]
    return sub.groupby("model")[cols].median().median()


def land_path():
    box = sgeom.box(X0, Y0, X1, Y1)
    verts, codes = [], []
    for g in cfeature.NaturalEarthFeature("physical", "land", "50m").geometries():
        for gg in (g, saff.translate(g, xoff=360)):
            c = gg.intersection(box)
            if c.is_empty:
                continue
            polys = (
                [c]
                if isinstance(c, sgeom.Polygon)
                else [p for p in getattr(c, "geoms", []) if isinstance(p, sgeom.Polygon)]
            )
            for p in polys:
                for ring in [p.exterior] + list(p.interiors):
                    xy = np.asarray(ring.coords)
                    if len(xy) < 4:
                        continue
                    verts.extend(xy.tolist())
                    codes.extend([MPath.MOVETO] + [MPath.LINETO] * (len(xy) - 2) + [MPath.CLOSEPOLY])
    return MPath(verts, codes)


LAND = land_path()


def cloud_sprite(seed=0, nx=200, ny=290):
    """A soft cumulonimbus tower (base puffs, a slim tower, an anvil) as an RGBA image, lit from the upper left."""
    rng = np.random.default_rng(seed)
    u = np.linspace(-1, 1, nx)
    v = np.linspace(0, 1, ny)
    U, V = np.meshgrid(u, v)

    def blob(cx, cy, sx, sy):
        return np.exp(-(((U - cx) / sx) ** 2 + ((V - cy) / sy) ** 2))

    d = np.zeros_like(U)
    for cx, cy, sx, sy in [(-0.55, 0.14, 0.36, 0.14), (0.05, 0.16, 0.42, 0.16), (0.6, 0.13, 0.32, 0.13)]:
        d += 1.1 * blob(cx, cy, sx, sy)
    for cy in np.linspace(0.22, 0.7, 7):
        d += 0.9 * blob(0.07 * np.sin(cy * 11), cy, 0.26 - 0.06 * cy, 0.11)
    d += 1.5 * blob(0.1, 0.84, 0.8, 0.1)  # anvil
    d += 0.7 * blob(0.7, 0.88, 0.3, 0.06)
    for cx, cy in [(-0.35, 0.78), (0.3, 0.8), (-0.6, 0.86), (0.0, 0.92)]:  # bulging top
        d += 0.7 * blob(cx, cy, 0.2, 0.08)
    for _ in range(30):
        cx, cy = rng.uniform(-0.75, 0.75), rng.uniform(0.08, 0.92)
        if abs(cx) > 0.32 and 0.32 < cy < 0.72:
            continue
        d += 0.35 * blob(cx, cy, rng.uniform(0.08, 0.16), rng.uniform(0.05, 0.09))
    alpha = gaussian_filter(np.clip((d - 0.6) * 6.0, 0, 1), 0.6)
    dv, du = np.gradient(gaussian_filter(d, 4))
    shade = np.clip(0.92 - 16.0 * dv + 8.0 * du, 0.55, 1.0)  # tops and left flanks lit, undersides in shadow
    rgb = np.dstack([0.58 + 0.42 * shade, 0.60 + 0.40 * shade, 0.65 + 0.35 * shade])
    return np.dstack([np.clip(rgb, 0, 1), alpha])


SPRITES = [cloud_sprite(k) for k in range(3)]


def cloud(ax, x, y, k=0, w=20.0, z=20):
    h = w * 1.45
    ax.add_patch(
        Ellipse((x + 1.5, y - 0.5), 0.8 * w, 0.22 * w, facecolor="0.3", edgecolor="none", alpha=0.18, zorder=z - 1)
    )
    ax.imshow(
        SPRITES[k % 3],
        extent=[x - w / 2, x + w / 2, y - 1.5, y - 1.5 + h],
        origin="lower",
        zorder=z,
        interpolation="bilinear",
        aspect="auto",
    )


def draw_map(ax, sst, box_color, depth=2.4, lat_labels=True):
    T = SH + ax.transData
    corners = [sh(X0, Y0), sh(X1, Y0), sh(X1, Y1), sh(X0, Y1)]
    frame = Polygon(np.array(corners), closed=True, fill=False, edgecolor=S.INK, lw=0.7, zorder=6)
    cf = ax.contourf(
        LON, LAT, sst, levels=np.arange(22, 30.51, 0.5), cmap="RdYlBu_r", extend="both", transform=T, zorder=2
    )
    iso = ax.contour(LON, LAT, sst, levels=[28.0], colors=[S.INK], linewidths=0.9, transform=T, zorder=4)
    land = ax.add_patch(PathPatch(LAND, facecolor="#e6e2da", edgecolor="#4a4a4a", lw=0.3, transform=T, zorder=5))
    ax.add_patch(frame)
    clip = Polygon(np.array(corners), closed=True, transform=ax.transData)
    for art in [cf] if hasattr(cf, "set_clip_path") and not hasattr(cf, "collections") else cf.collections:
        art.set_clip_path(clip)
    for art in [iso] if not hasattr(iso, "collections") else iso.collections:
        art.set_clip_path(clip)
    land.set_clip_path(clip)
    down = np.array([0, -depth])
    c = corners
    ax.add_patch(
        Polygon(
            np.array([c[0], c[1], c[1] + down, c[0] + down]),
            closed=True,
            facecolor="#9a9a9a",
            edgecolor=S.INK,
            lw=0.5,
            zorder=6,
        )
    )
    ax.add_patch(
        Polygon(
            np.array([c[1], c[2], c[2] + down, c[1] + down]),
            closed=True,
            facecolor="#777777",
            edgecolor=S.INK,
            lw=0.5,
            zorder=6,
        )
    )
    for lon, lab in [(120, "120°E"), (180, "180°"), (240, "120°W"), (280, "80°W")]:
        x, y = sh(lon, Y0)
        ax.text(x, y - depth - 1.2, lab, ha="center", va="top", fontsize=5.5, color=S.INK)
    for lat, lab in [(-20, "20°S"), (0, "0°"), (20, "20°N")] if lat_labels else []:
        x, y = sh(X0, lat)
        ax.text(x - 2.0, y, lab, ha="right", va="center", fontsize=5.5, color=S.INK)
    # the equatorial SST decides where the convective clouds stand
    eq = np.nanmean(sst[(LAT.values > -5) & (LAT.values < 5)], axis=0)
    lons = [lon for lon in range(138, 271, 22) if eq[np.argmin(np.abs(LON.values - lon))] > 28.0] or [138]
    for k, lon in enumerate(lons):
        x, y = sh(lon, 3)
        cloud(ax, x, y, k=k)
    cross = np.where((eq[:-1] >= 28.0) & (eq[1:] < 28.0))[0]  # eastern end of the equatorial warm pool
    if cross.size:
        xl, yl = sh(float(LON.values[cross[-1]]) + 3, -7)
        ax.text(xl, yl, "28°C", ha="left", va="top", fontsize=5.6, color=S.INK, zorder=8,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.6, alpha=0.85))
    box = np.array([sh(210, -5), sh(270, -5), sh(270, 5), sh(210, 5)])
    xl, yl = sh(240, -8.2)
    ax.text(xl, yl, "Niño-3", ha="center", va="top", fontsize=5.6, color=S.INK, zorder=8,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.6, alpha=0.85))
    ax.add_patch(Polygon(box, closed=True, fill=False, edgecolor=S.INK, lw=2.3, zorder=9))
    ax.add_patch(Polygon(box, closed=True, fill=False, edgecolor=box_color, lw=1.4, zorder=10))
    ax.set_xlim(X0 - 12, X1 + 26)
    ax.set_ylim(Y0 - depth - 6, Y1 + 16)                     # just enough sky for the cloud tops
    ax.set_aspect("equal", adjustable="box")
    ax.set_anchor("S")
    ax.axis("off")
    return cf



TAB = ROOT / "results/tables"
boot = pd.read_csv(TAB / "bootstrap.csv")
part = pd.read_csv(TAB / "partial_correlations.csv")
con = pd.read_csv(TAB / "constrained_estimates.csv")
c0 = con[(con.predictor == "pr_nino3_ref") & (con.pr == "GPCP")].iloc[0]
amp = {}
for m in mod.index:
    d = xr.open_dataset(FID / f"{m}_fidelity.nc")
    amp[m] = float(np.sqrt((d.nino34_std_cycle**2).mean()))
amp = pd.Series(amp)


def rel(x, y):
    return stats.pearsonr(x, y)


R = dict(
    dA=rel(mod.pr_nino3_ref, mod.d_A),
    grad=rel(mod.pr_nino3_ref, mod.grad_per_K),
    amp=rel(mod.pr_nino3_ref, amp.reindex(mod.index)),
    Aref=rel(mod.pr_nino3_ref, mod.A_ref),
    gdA=rel(mod.grad_per_K, mod.d_A),
)
pg = part[(part.predictor == "grad_per_K") & (part.controlling_for == "pr_nino3_ref")].iloc[0]
print({k: (round(v[0], 2), round(v[1], 3)) for k, v in R.items()})

W, H = 7.4, 6.32
fig = plt.figure(figsize=(W, H))
fid = pd.read_csv(TAB / "model_fidelity.csv", index_col=0)


def box(x, y, w, h):
    """Axes rectangle in inches from the lower left corner of the figure."""
    return [x / W, y / H, w / W, h / H]


# ---- geometry shared by the panels (inches)
MAP_Y, MAP_H = 4.85, 1.18
CARD_Y, CARD_H = 3.79, 0.82
BRACKET_Y = 3.55
AX_X, AX_W = 0.3, 6.8
XL = (0.0, 3.14)
CARDS = {"dry": (0.2, 3.0), "obs": (2.85, 3.91), "wet": (4.4, 7.2)}
dry, wet = [mod.loc[m_, "pr_nino3_ref"] for _, m_, _ in GROUPS]
GCOL = {"dry": rain_color(dry.median()), "wet": rain_color(wet.median()), "obs": (0.55, 0.55, 0.55, 1.0)}


def ax_x(v):
    return AX_X + AX_W * (v - XL[0]) / (XL[1] - XL[0])


back = fig.add_axes([0, 0, 1, 1], zorder=-5)
back.set_xlim(0, W)
back.set_ylim(0, H)
back.axis("off")


def tri(xc, y, c, up=True, w=0.11, h=0.11):
    """Filled triangle standing on a horizontal bar, pointing up or down."""
    tip = y + h if up else y - h
    back.add_patch(Polygon([(xc - w, y), (xc + w, y), (xc, tip)], closed=True, facecolor=c, edgecolor="none", zorder=3))


def wave(points, c):
    """Unfilled line through the points, vertical at each of them, so that it runs down like a wave."""
    v, k = [points[0]], [MPath.MOVETO]
    for (xa, ya), (xb, yb) in zip(points[:-1], points[1:]):
        d = 0.55 * (ya - yb)
        v += [(xa, ya - d), (xb, yb + d), (xb, yb)]
        k += [MPath.CURVE4] * 3
    back.add_patch(PathPatch(MPath(v, k), fill=False, edgecolor=c, lw=1.0, zorder=2))


def column(key, xb0, xb1):
    """Bar under the map with its triangle towards the map, bar over the range of the ensembles in panel b with its triangle
    down, and a wave on each side joining their ends."""
    x0, x1 = CARDS[key]
    c, top = GCOL[key], CARD_Y + CARD_H
    back.plot([x0, x1], [top, top], color=c, lw=1.6, solid_capstyle="butt", zorder=3)
    tri(0.5 * (x0 + x1), top, c, up=True)
    back.plot([xb0, xb1], [BRACKET_Y, BRACKET_Y], color=c, lw=1.6, solid_capstyle="butt", zorder=3)
    tri(0.5 * (xb0 + xb1), BRACKET_Y, c, up=False)
    knee, step, drop = CARD_Y + 0.08, 0.42, 0.22  # each side steps inwards under the bar, then swings to the range bar
    wave([(x0, top), (x0 + step, top - drop), (x0 + step, knee), (xb0, BRACKET_Y)], c)
    wave([(x1, top), (x1 - step, top - drop), (x1 - step, knee), (xb1, BRACKET_Y)], c)


maps, STATS = [], {}
for i, (key, models, title) in enumerate(GROUPS):
    xm = 0.02 + 3.88 * i
    st = group_stats(models)
    axm = fig.add_axes(box(xm, MAP_Y, 3.48, MAP_H))
    cf = draw_map(axm, composite_sst(models), GCOL[key], lat_labels=(i == 0))
    axm.set_title(title, fontsize=8.2, fontweight="bold", loc="center", pad=2, color=S.INK)
    maps.append(axm)

    STATS[key] = st

fig.canvas.draw()  # the map axes now have their final positions

for axm, key in zip(maps, ("dry", "wet")):  # the bars span the width of each map
    T_ = SH + axm.transData
    CARDS[key] = (T_.transform((X0, Y0))[0] / fig.dpi, T_.transform((X1, Y0))[0] / fig.dpi)

for key, _, _ in GROUPS:
    st = STATS[key]
    x0, x1 = CARDS[key]
    vals = [st.pEN, st.pLN]
    left, right = -0.42, 1.5 * max(vals) / 60 + 0.32  # extent of the block in inches from the zero line: phase names, bars, values
    axb = fig.add_axes(box(0.5 * (x0 + x1) - 0.5 * (left + right), CARD_Y + 0.1, 1.5, 0.42))
    axb.patch.set_alpha(0)
    axb.barh([1, 0], vals, height=0.68, color=[S.EN, S.LN], zorder=3)
    for k, v in zip((1, 0), vals):
        axb.text(v + 1.2, k, f"+{v:.0f}%", ha="left", va="center", fontsize=6.6, color=S.INK)
    axb.axvline(0, color=S.INK, lw=0.6, zorder=4)
    axb.set_yticks([1, 0])
    axb.set_yticklabels(["El Niño", "La Niña"])
    axb.set_yticks([], minor=True)
    axb.set_xticks([])
    axb.set_xticks([], minor=True)
    axb.set_xlim(0, 60)
    axb.set_ylim(-0.55, 1.55)
    for side in ("top", "right", "bottom", "left"):
        axb.spines[side].set_visible(False)
    axb.tick_params(axis="y", length=0, labelsize=6.6, pad=2)
    win = "El Niño gains more" if st.d_A > 0 else "La Niña gains more"
    fig.text(0.5 * (x0 + x1) / W, (CARD_Y + CARD_H - 0.09) / H, f"{win}, Δ$A$ = {st.d_A:+.2f}".replace("-", "−"), ha="center", va="top",
             fontsize=7.4, fontweight="bold", color=S.EN if st.d_A > 0 else S.LN)
    print(key, st.round(3).to_dict())



# ---- colour bar: a parallelogram in the gap, parallel to the facing edges of the two maps
axl = maps[0]
Tl = SH + axl.transData
Tr = SH + maps[1].transData
right_of_left = Tl.transform((X1, Y0))[0]
left_of_right = Tr.transform((X0, Y0))[0]
unit = Tl.transform((X1 + 1, Y0))[0] - right_of_left   # display units per degree of the map
gap = (left_of_right - right_of_left) / unit
BW = 6.5
bx0 = X1 + 0.5 * (gap - BW) - 2.2
levels = cf.levels
span = (Y1 - Y0) / (levels[-1] - levels[0])
ylev = lambda v: Y0 + (v - levels[0]) * span  # noqa: E731
for lo, hi in zip(levels[:-1], levels[1:]):
    axl.add_patch(Polygon([(bx0, ylev(lo)), (bx0 + BW, ylev(lo)), (bx0 + BW, ylev(hi)), (bx0, ylev(hi))], closed=True,
                          facecolor=cf.cmap(cf.norm(0.5 * (lo + hi))), edgecolor="none", transform=Tl, clip_on=False, zorder=7))
axl.add_patch(Polygon([(bx0, Y0), (bx0 + BW, Y0), (bx0 + BW, Y1), (bx0, Y1)], closed=True, fill=False, edgecolor=S.INK, lw=0.6,
                      transform=Tl, clip_on=False, zorder=8))
for v in (22, 24, 26, 28, 30):
    axl.plot([bx0 + BW, bx0 + BW + 1.6], [ylev(v)] * 2, color=S.INK, lw=0.5, transform=Tl, clip_on=False, zorder=8)
    axl.text(bx0 + BW + 2.6, ylev(v), str(v), fontsize=5.6, ha="left", va="center", color=S.INK, transform=Tl, clip_on=False, zorder=8)
xtop, ytop = sh(bx0 + 0.5 * BW, Y1)
axl.text(xtop, ytop + 2.2, "SST (°C)", fontsize=6, ha="center", va="bottom", color=S.INK, clip_on=False, zorder=8)

# ---- bars joining each map, its result and its place on the rainfall axis; a rounded box for the observed state
column("dry", ax_x(dry.min()), ax_x(dry.max()))
column("wet", ax_x(wet.min()), ax_x(wet.max()))
xobs = ax_x(OBS_PR)
x0_, x1_ = CARDS["obs"]
back.add_patch(FancyBboxPatch((x0_, CARD_Y), x1_ - x0_, CARD_H - 0.22, boxstyle="round,pad=0,rounding_size=0.1", facecolor=GCOL["obs"], edgecolor="none", alpha=0.2))
back.add_patch(FancyBboxPatch((x0_, CARD_Y), x1_ - x0_, CARD_H - 0.22, boxstyle="round,pad=0,rounding_size=0.1", fill=False, edgecolor=GCOL["obs"], lw=1.0))
tri(xobs, CARD_Y, GCOL["obs"], up=False)
xc = 0.5 * sum(CARDS["obs"])
fig.text(xc / W, (CARD_Y + CARD_H - 0.28) / H, "Observed state", ha="center", va="top", fontsize=7.4, fontweight="bold", color=S.INK)
fig.text(xc / W, (CARD_Y + 0.22) / H, f"neither phase gains\nΔ$A$ = {abs(c0.dA):.2f} ({c0.lo68:+.2f} to {c0.hi68:+.2f})".replace("-", "−").replace(" (", "\n("),
         ha="center", va="center", fontsize=7.2, color=S.INK, linespacing=1.3)

# ---- (b) the ensembles above the rainfall colour bar of Figure 3
axm_ = fig.add_axes(box(AX_X, 2.79, AX_W, 0.56))
axm_.set_xlim(*XL)
axm_.set_ylim(0, 1)
axm_.axis("off")
axm_.patch.set_alpha(0)
od = mod.sort_values("pr_nino3_ref")
ypos, last = {}, [-9.0, -9.0, -9.0]
LEVELS = [0.5, 0.84, 0.16]
for m, v in od.pr_nino3_ref.items():  # stack close neighbours so that every marker stays visible
    free = [k for k in range(3) if v - last[k] > 0.078 and not (k == 0 and -0.09 < v - OBS_PR < 0.22)]
    k = free[0] if free else int(np.argmin(last))
    ypos[m], last[k] = LEVELS[k], v
NUDGE = {"gfdl_spear_lens": -0.02, "canesm5_lens": 0.025}  # two neighbours on either side of the group outline, moved apart by less than a marker
for m in mod.index:
    v = mod.loc[m, "pr_nino3_ref"] + NUDGE.get(m, 0.0)
    (numbered_marker if fid.loc[m, "cold_tongue"] else open_marker)(axm_, v, ypos[m], int(mod.loc[m, "num"]), rain_color(mod.loc[m, "pr_nino3_ref"]), size=105, fontsize=5.8)
for key, models_, _ in GROUPS:
    xs_ = mod.loc[models_, "pr_nino3_ref"]
    axm_.add_patch(FancyBboxPatch((xs_.min() - 0.075, -0.08), xs_.max() - xs_.min() + 0.075 + (0.04 if key == "dry" else 0.075), 1.16, boxstyle="round,pad=0,rounding_size=0.06",
                                  facecolor=(*GCOL[key][:3], 0.10), edgecolor=GCOL[key], lw=1.0, zorder=1, clip_on=False))
axm_.scatter([OBS_PR], [0.5], s=150, marker="*", color=S.ACCENT, edgecolor="white", lw=0.6, zorder=9, clip_on=False)
axm_.text(OBS_PR + 0.05, 0.5, "obs", fontsize=7.5, fontweight="bold", color=S.ACCENT, ha="left", va="center", zorder=9)
for key, models, _ in GROUPS:
    xs = mod.loc[models, "pr_nino3_ref"]
    back.text(ax_x(0.5 * (xs.min() + xs.max())), BRACKET_Y + 0.03, "six driest" if key == "dry" else "six wettest", ha="center", va="bottom", fontsize=6.6, color=S.INK)
axc_ = fig.add_axes(box(AX_X, 2.49, AX_W, 0.12))
xx = np.linspace(0, 3, 300)
axc_.imshow(np.array([[rain_color(v) for v in xx]]), extent=[0, 3, 0, 1], aspect="auto", zorder=1)
axc_.add_patch(Polygon([(3, 0), (3.14, 0.5), (3, 1)], closed=True, facecolor=rain_color(3.0), edgecolor=S.INK, lw=0.5, clip_on=False))
axc_.plot([3, 0, 0, 3], [1, 1, 0, 0], color=S.INK, lw=0.5, clip_on=False, zorder=3)
axc_.set_xlim(*XL)
axc_.set_ylim(0, 1)
axc_.set_yticks([])
axc_.set_yticks([], minor=True)
axc_.set_xticks([0, 0.5, 1, 1.5, 2, 2.5, 3])
axc_.set_xticks([], minor=True)
axc_.tick_params(axis="x", length=2, width=0.5, labelsize=6.4, pad=1.5)
axc_.set_xlabel("Present-day Niño-3 rainfall climatology across ensembles (mm/day)", fontsize=7.6, labelpad=6)
for side in ("top", "right", "left", "bottom"):
    axc_.spines[side].set_visible(False)

# ---- (c) the two candidate predictors of the asymmetry change
CH = 1.72
axn = fig.add_axes(box(0.0, 0.0, W, CH))
axn.set_xlim(0, W)
axn.set_ylim(0, CH)
axn.axis("off")
pr_ = part[(part.predictor == "pr_nino3_ref") & (part.controlling_for == "grad_per_K")].iloc[0]


def rbox(x, y, w, h, txt, fc, ec, fs=7.8, weight="bold", tc=None):
    axn.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.06", facecolor=fc, edgecolor=ec, lw=0.9, zorder=3))
    axn.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs, fontweight=weight, color=tc or S.INK, zorder=4, linespacing=1.2)


PX, PW, PH = 1.3, 1.4, 0.56
TX, TW = 4.95, 1.15
y1, y2 = 0.98, 0.08
GREEN = rain_color(2.3)  # the rainfall colours of Figures 3 and 4b; gray is the predictor that is not significant
rbox(PX, y1, PW, PH, "Present-day\neastern Pacific\nrainfall", (*rain_color(1.6)[:3], 0.3), GREEN, fs=7.6)
rbox(PX, y2, PW, PH, "Forced zonal\nSST-gradient\nchange", "#eeeeee", S.MUTED, fs=7.6)
rbox(TX, 0.22, TW, 1.2, "ENSO\nasymmetry\nchange\nΔ$A$", "white", S.INK, fs=8.0)
xm = 0.5 * (PX + PW + TX)
for y, r_, pr__, sig in ((y1, R["dA"][0], pr_, True), (y2, R["gdA"][0], pg, False)):
    ym = y + PH / 2
    axn.add_patch(FancyArrowPatch((PX + PW + 0.05, ym), (TX - 0.05, ym), arrowstyle="-|>", mutation_scale=10, lw=1.8 if sig else 1.1,
                                  color=S.INK if sig else S.MUTED, ls="-" if sig else (0, (3, 2)), shrinkA=0, shrinkB=0, zorder=2))
    axn.text(xm, ym + 0.2, f"r = {r_:+.2f}".replace("-", "−"), ha="center", va="bottom", fontsize=8.6, fontweight="bold", color=S.INK)
    ptxt = f"partial r = {pr__.r_partial:+.2f}, p = {pr__.p_partial:.2f}" if not sig else f"partial r = {pr__.r_partial:+.2f}, p < 0.05"
    axn.text(xm, ym + 0.05, ptxt.replace("-", "−"), ha="center", va="bottom", fontsize=6.8, color=S.INK)
ym = y1 + PH / 2
axn.add_patch(FancyBboxPatch((xm - 1.0, ym - 0.3), 0.74, 0.2, boxstyle="round,pad=0,rounding_size=0.04", facecolor=GREEN, edgecolor="none", zorder=3))
axn.text(xm - 0.63, ym - 0.2, "Significant", ha="center", va="center", fontsize=6.4, fontweight="bold", color="white", zorder=4)
axn.plot([xm - 0.17, xm - 0.17], [ym - 0.33, ym - 0.07], color=S.MUTED, lw=0.6)
axn.text(xm - 0.08, ym - 0.2, "Drier → El Niño\nWetter → La Niña", ha="left", va="center", fontsize=6.6, color=S.INK, linespacing=1.2)
ym = y2 + PH / 2
axn.add_patch(FancyBboxPatch((xm - 0.78, ym - 0.36), 1.56, 0.29, boxstyle="round,pad=0,rounding_size=0.05", facecolor="#e6e6e6", edgecolor="none", zorder=3))
axn.text(xm, ym - 0.215, "Weak after controlling\nfor rainfall", ha="center", va="center", fontsize=6.4, color=S.INK, zorder=4, linespacing=1.15)

for lab, ttl, y, x in (("a", "Driest and wettest ensembles: mean state and response above 3 K of warming", H - 0.02, 0.02),
                       ("b", "", 3.53, 0.02), ("c", "Which predictor is related to Δ$A$?", 1.89, 0.02)):
    fig.text(x / W, y / H, lab, fontsize=9, fontweight="bold", ha="left", va="top", color=S.INK)
    fig.text((x + 0.17) / W, y / H, ttl, fontsize=9, ha="left", va="top", color=S.INK)
FIG_DIR.mkdir(exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(FIG_DIR / f"fig4_summary.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.03)
print("saved fig4 summary")
