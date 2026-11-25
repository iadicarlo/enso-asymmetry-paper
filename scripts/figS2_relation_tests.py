"""Figure S2: the cross-model relation between the asymmetry change and the present-day eastern Pacific state under other
choices."""

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
    fit_line,
    numbered_marker,
    TABLES,
)

S.apply()
mod, mem, tr, obs = load_tables()
mem = member_pattern(mem)
models = list(mod.index)
grad = per_model(mem, "grad_per_K").loc[models]
boot = pd.read_csv(TABLES / "bootstrap.csv")
selb = pd.read_csv(TABLES / "selection_bootstrap.csv")
om = pd.read_csv(TABLES / "obs_multi.csv")
om = om[om.pr == "GPCP"].set_index("sst")

fig, ax = uplt.subplots(figwidth=6.2, refheight=1.9, share=False)
ax = ax[0]

# forest plot of the cross-model correlation under different choices
scr = pd.read_csv(TABLES / "screened_bootstrap.csv")
rows = [
    (
        "Present-day Niño-3 rainfall (main)",
        boot[(boot.x == "pr_nino3_ref") & (boot.y == "d_A") & (boot.design == "gwi_bins")],
    ),
    ("Present-day Niño-3 rainfall, screened ensembles", scr[(scr.x == "pr_nino3_ref") & (scr.y == "d_A")]),
    ("Present-day cold tongue SST, screened ensembles", scr[(scr.x == "cold_tongue_sst_ref") & (scr.y == "d_A")]),
    (
        "Present-day cold tongue SST",
        boot[(boot.x == "cold_tongue_sst_ref") & (boot.y == "d_A") & (boot.design == "gwi_bins")],
    ),
    (
        "Niño-3 rainfall in the warm state",
        boot[(boot.x == "pr_nino3_warm") & (boot.y == "d_A") & (boot.design == "gwi_bins")],
    ),
]
ys = np.arange(len(rows))[::-1]
for yi, (name, q) in zip(ys, rows):
    if len(q) == 0:  # a subset that fell below the minimum ensemble count is left blank
        ax.text(0, yi, "subset too small", ha="center", va="center", fontsize=6, color=S.MUTED)
        continue
    q = q.iloc[0]
    ax.plot([q.r_lo, q.r_hi], [yi, yi], color=S.INK, lw=1.2)
    ax.scatter(q.r, yi, s=36, color=S.INK, edgecolor=S.INK, lw=0.8, zorder=5)
ax.axvline(0, color=S.MUTED, lw=0.6, ls=(0, (3, 2)))
ax.format(
    yticks=ys,
    yticklabels=[(f"{r[0]}   (n = {int(r[1].iloc[0].n)}, p = {r[1].iloc[0].p_perm:.3f})" if len(r[1]) else f"{r[0]}   (fewer than five ensembles)") for r in rows],
    xlabel="Pearson r with Δ$A$\n(95% two-level bootstrap interval)",
    xlim=(-1.0, 1.0),
        ytickminor=False,
    xtickminor=False,
)
ax.tick_params(axis="y", length=0)
S.despine(ax, keep=("bottom", "left"))
ax.spines["left"].set_linewidth(0)

fig.format(abc=False)
FIG_DIR.mkdir(exist_ok=True)
fig.save(FIG_DIR / "figS2_relation_tests.png", dpi=300)
fig.save(FIG_DIR / "figS2_relation_tests.pdf")
print("saved figS2")
