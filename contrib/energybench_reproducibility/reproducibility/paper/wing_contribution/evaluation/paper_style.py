"""Typography shared by Wing's figures, drawn at their final paper width.

The ICLR body uses 10 pt Times. Nimbus Roman is the bundled Times-compatible
family also used by the manuscript's XeTeX build. Its equations retain
Computer Modern, so mathtext uses that family. Matplotlib sizes are points;
exporting at the actual include width avoids shrinking labels in LaTeX.
"""
from pathlib import Path

import matplotlib as mpl
from matplotlib import font_manager

PAPER_WIDTH_IN = 5.5


def apply_paper_style():
    font_dir = Path(__file__).resolve().parents[1] / "fonts"
    for path in sorted(font_dir.glob("NimbusRoman-*.otf")):
        font_manager.fontManager.addfont(str(path))
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Nimbus Roman"],
        "font.cursive": ["Nimbus Roman"],
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.titleweight": "normal",
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.title_fontsize": 8,
        "mathtext.fontset": "cm",
        "mathtext.fallback": "cm",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#555555",
        "axes.linewidth": .7,
        "text.color": "#222222",
        "axes.labelcolor": "#222222",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": None,
    })
