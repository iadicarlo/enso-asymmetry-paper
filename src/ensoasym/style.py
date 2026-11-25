"""One visual grammar for every figure in the paper (ultraplot)."""

from __future__ import annotations

import ultraplot as uplt

# Phase colours: warm side in red-orange, cold side in blue.  Neutral grey.
EN = "#c8321e"
EN_LIGHT = "#f2a48c"
LN = "#1f4e9c"
LN_LIGHT = "#8db4e2"
NEUTRAL = "#c9c9c9"
INK = "#2b2b2b"
MUTED = "#8a8a8a"
OBS = "#111111"
ACCENT = "#000000"  # observation highlight (black; gold failed the colour-blindness check)
OBS_BAND = "#9e9e9e"  # observation range band
PATTERN_EL = "#d9553c"  # El Nino-like warming (east warms more)
PATTERN_LA = "#3b7dd8"  # La Nina-like warming (east warms less)

# Diverging colormap for "forced warming pattern" colouring of models
PATTERN_CMAP = uplt.Colormap("RdBu_r", name="pattern")

FONT = "Helvetica Neue"


def apply():
    uplt.rc.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [FONT, "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9,
            "legend.fontsize": 7.5,
            "tick.labelsize": 7.5,
            "axes.linewidth": 0.6,
            "tick.width": 0.6,
            "tick.len": 3,
            "grid": False,
            "abc": "a",
            "abc.size": 10,
            "abc.weight": "bold",
            "title.loc": "l",
            "title.weight": "regular",
            "axes.titlepad": 4,
            "subplots.share": False,
            "savefig.dpi": 300,
            "savefig.transparent": False,
            "errorbar.capsize": 0,
            "text.color": INK,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
        }
    )


def despine(ax, keep=("left", "bottom")):
    for side in ("left", "right", "top", "bottom"):
        ax.spines[side].set_visible(side in keep)
