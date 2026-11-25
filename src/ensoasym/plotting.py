"""Shared helpers for the paper figures."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .io import ROOT, PRETTY
from . import style as S

TABLES = ROOT / "results" / "tables"


def load_tables():
    mod = pd.read_csv(TABLES / "models_delta.csv", index_col=0)
    mod = mod.sort_values("pretty")
    mod["num"] = np.arange(1, len(mod) + 1)
    mem = pd.read_csv(TABLES / "members_delta.csv")
    tr = pd.read_csv(TABLES / "gwl_trajectory.csv")
    obs = pd.read_csv(TABLES / "obs_summary.csv", header=None, index_col=0)[1]
    return mod, mem, tr, obs


def member_pattern(mem: pd.DataFrame) -> pd.DataFrame:
    """Per-member zonal gradient change per degree of global warming."""
    mem = mem.copy()
    mem["grad_per_K"] = mem["d_sst_gradient"] / mem["dgwi"]
    mem["wpct_per_K"] = mem["d_wp_minus_ct"] / mem["dgwi"]
    return mem


def per_model(mem: pd.DataFrame, col: str) -> pd.DataFrame:
    g = mem.groupby("model")[col]
    out = pd.DataFrame(
        {
            "med": g.median(),
            "p5": g.quantile(0.05),
            "p10": g.quantile(0.10),
            "p90": g.quantile(0.90),
            "p95": g.quantile(0.95),
            "se": g.apply(lambda v: 1.253 * np.std(v, ddof=1) / np.sqrt(len(v))),
            "n": g.size(),
        }
    )
    return out


def numbered_marker(ax, x, y, num, color, size=110, zorder=5, fontsize=5.6):
    """Circle marker with the model number inside."""
    ax.scatter(x, y, s=size, color=color, edgecolor=S.INK, lw=0.5, zorder=zorder)
    lum = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
    ax.text(
        x,
        y,
        str(num),
        ha="center",
        va="center_baseline",
        fontsize=fontsize,
        zorder=zorder + 1,
        color="white" if lum < 0.55 else S.INK,
        fontweight="bold",
    )


def open_marker(ax, x, y, num, color, size=110, zorder=5, fontsize=5.6):
    """Outlined marker (coloured ring, white centre) with the model number inside."""
    ax.scatter(x, y, s=size, color=color, edgecolor=S.INK, lw=0.5, zorder=zorder)
    ax.scatter(x, y, s=0.58 * size, color="white", edgecolor=S.INK, lw=0.3, zorder=zorder + 0.5)
    ax.text(x, y, str(num), ha="center", va="center_baseline", fontsize=fontsize * 0.92, zorder=zorder + 1, color=S.INK, fontweight="bold")


def repel_markers(ax, x, y, min_pts=11.0, fixed=(), n_iter=400):
    """Plot positions for numbered markers that would overlap. Distances are in points of the final axes, so the axis
    limits and the figure size must be set first. Markers move only as far as needed; `fixed` holds (x, y) pairs that
    must stay clear and do not move. Draw a leader from each returned position to the true one."""
    ax.figure.canvas.draw()
    to_pts = ax.transData.transform
    k = 72.0 / ax.figure.dpi
    pts = to_pts(np.column_stack([x, y])) * k
    orig = pts.copy()
    fix = to_pts(np.atleast_2d(fixed)) * k if len(fixed) else np.empty((0, 2))
    for _ in range(n_iter):
        moved = False
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                v = pts[i] - pts[j]
                d = np.hypot(*v)
                if d < min_pts:
                    v = v / d if d > 1e-6 else np.array([1.0, 0.0])
                    pts[i] += 0.5 * (min_pts - d) * v * 1.02
                    pts[j] -= 0.5 * (min_pts - d) * v * 1.02
                    moved = True
            for f in fix:
                v = pts[i] - f
                d = np.hypot(*v)
                if d < min_pts:
                    v = v / d if d > 1e-6 else np.array([1.0, 0.0])
                    pts[i] += (min_pts - d) * v * 1.02
                    moved = True
        if not moved:
            break
    return ax.transData.inverted().transform(pts / k)


def rain_mappable():
    """Mappable of the rainfall scale (0 to 3 mm/day) for colour bars, to be drawn with extend="max"."""
    import matplotlib as mpl
    import ultraplot as uplt

    return mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0.0, 3.0), cmap=uplt.Colormap("GnBu", left=0.15))


RAIN_CMAP = None


def rain_color(v, vmin=0.0, vmax=3.0):
    """Sequential colour for present-day Niño-3 rainfall (dry pale, wet dark)."""
    import ultraplot as uplt

    cmap = uplt.Colormap("GnBu", left=0.15)
    t = (np.clip(v, vmin, vmax) - vmin) / (vmax - vmin)
    return cmap(t)


def pattern_color(v, vmin=-0.35, vmax=0.35):
    """Map gradient change per K to the diverging pattern colormap."""
    cmap = S.PATTERN_CMAP
    t = (np.clip(v, vmin, vmax) - vmin) / (vmax - vmin)
    return cmap(t)


def fit_line(x, y, xgrid=None):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    res = stats.linregress(x, y)
    rho, prho = stats.spearmanr(x, y)
    if xgrid is None:
        xgrid = np.linspace(x.min(), x.max(), 100)
    yhat = res.intercept + res.slope * xgrid
    n = x.size
    s = np.sqrt(np.sum((y - (res.intercept + res.slope * x)) ** 2) / (n - 2))
    band = stats.t.ppf(0.975, n - 2) * s * np.sqrt(1 / n + (xgrid - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2))
    return dict(r=res.rvalue, p=res.pvalue, rho=rho, prho=prho, slope=res.slope, xgrid=xgrid, yhat=yhat, band=band, n=n)


def stat_text(f):
    return f"r = {f['r']:+.2f} (p = {f['p']:.2f})\nρ = {f['rho']:+.2f} (p = {f['prho']:.2f})\nn = {f['n']}"


def label(model):
    return PRETTY.get(model, model)
