"""Shared style for the paper's figures: one palette, one type setup, one axis treatment.

Every figure script in this directory imports this module so that a sixth figure cannot drift from
the first five. The conventions come from the authors' CatchBench preprint: coral marks the focal
result, mint carries the comparison layer, gray carries context, near-black carries text; no top or
right spines; gray left and bottom spines; no tick marks; type never below 6 pt when the page is set
into the paper's 6.5 inch text width. `make_grounding_effects.py` predates this module and carries
the same constants inline.

Usage:
    import figstyle as fs
    fs.apply()                      # rcParams: sans-serif, TrueType fonts, text colors
    fig, ax = plt.subplots(figsize=(fs.TEXT_WIDTH_IN, 2.4))
    fs.bare(ax)                     # spines and ticks
    fs.panel_title(ax, "(a)", "Smoke detection")
"""
from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt

# Palette (CatchBench). CORAL marks the focal result; MINT carries the comparison layer; GRAY carries
# context and inactive structure.
CORAL = "#ED8D5A"
MINT = "#BFDFD2"
MINT_EDGE = "#8FB7A6"
GRAY = "#999999"
LIGHT_GRAY = "#C9C9C9"
GRID = "#E6E6E6"
NEAR_BLACK = "#1A1A1A"
SUBTITLE = "#666666"

# The paper's text width (TMLR, 6.5 in). Figures are authored at final size; nothing scales later.
TEXT_WIDTH_IN = 6.5

# Type sizes in points, at final size. Nothing printed may fall below 6 pt.
FS_TITLE = 8.0
FS_AXIS = 7.2
FS_TICK = 6.8
FS_SMALL = 6.2

# The six models, in the order every figure and table uses, with the stem each task's response
# files carry: task-<task>/responses-<stem>-<condition>.jsonl.
MODELS = [
    {"label": "claude-opus-4.8", "stem": "claude-opus-4.8", "family": "proprietary"},
    {"label": "claude-opus-5", "stem": "claude-opus-5", "family": "proprietary"},
    {"label": "gemini-3.1-pro", "stem": "gemini-3.1-pro", "family": "proprietary"},
    {"label": "gpt-6-astra", "stem": "gpt-6-astra", "family": "proprietary"},
    {"label": "Qwen3-VL", "stem": "bedrock_qwen.qwen3-vl-235b-a22b", "family": "open-weight"},
    {"label": "Llama 4 Maverick", "stem": "bedrock_us.meta.llama4-maverick-17b-instruct-v1_0", "family": "open-weight"},
]


def apply() -> None:
    """Set the rcParams every figure shares: sans-serif, TrueType (not Type 3), near-black text."""
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "text.color": NEAR_BLACK,
        "axes.labelcolor": NEAR_BLACK,
        "xtick.color": NEAR_BLACK,
        "ytick.color": NEAR_BLACK,
        "axes.edgecolor": GRAY,
        "axes.labelsize": FS_AXIS,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_SMALL,
        "axes.titlesize": FS_TITLE,
    })


def bare(ax, grid: str | None = "x") -> None:
    """No top or right spine, gray left and bottom spines, no tick marks, a light grid behind the data."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRAY)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(axis="both", length=0, labelsize=FS_TICK)
    ax.set_axisbelow(True)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=0.5)
        ax.grid(axis="y" if grid == "x" else "x", visible=False)


def panel_title(ax, tag: str, text: str, x: float = 0.0, y: float = 1.04) -> None:
    """A bold panel label such as '(a) Smoke detection' at the top left of the axes."""
    ax.text(x, y, f"{tag} {text}", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=FS_TITLE, fontweight="bold", color=NEAR_BLACK)


def savefig(fig, out_pdf, out_png, dpi: int = 200) -> None:
    """Write the PDF the paper includes and a PNG for review, both from the same figure object."""
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=dpi)
