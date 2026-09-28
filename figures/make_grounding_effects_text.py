#!/usr/bin/env python
"""Generate the grounding contrasts figure for text-only models (figures/grounding_effects_text.pdf and .png).

This figure displays the paired differences for the nineteen text-only models across the three text tasks:
  - Panel (a): Personnel allocation (ICS-209-PLUS, 300 items)
               Normalized-error difference (grounded minus bare), with incident-clustered
               95% bootstrap intervals (245 clusters). Outliers Llama 3.3 70B (-0.68) and
               Mistral Small 2402 (+1.47) are clipped at the edges with triangles and value labels.
  - Panel (b): Fire danger forecasting (Mesogeos Track A, 386 items)
               AUPRC difference (grounded minus bare), with 1-degree-by-calendar-month
               clustered 95% bootstrap intervals (352 clusters).
  - Panel (c): Fire data tool use (FPA-FOD, 156 items)
               Accuracy difference (tool minus bare), with family-clustered 95% bootstrap
               intervals (12 clusters). Llama 3.1 8B and DeepSeek R1 failed the tool probe
               and show an empty row with a small gray note.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import figstyle as fs

# CatchBench Palette (matching figstyle and make_grounding_effects.py)
COLOR_CORAL = "#ED8D5A"
COLOR_GRAY = "#999999"
COLOR_LIGHT_GRAY = "#E6E6E6"
COLOR_TEXT = "#1A1A1A"
COLOR_SUBTITLE = "#666666"


def load_data(repo_root: Path, json_path: Path | None = None) -> list[dict]:
    """Load text-only model intervals from text_sweep_intervals.json."""
    path = json_path or (repo_root / "analysis" / "text_sweep_intervals.json")
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = []
    for m in data.get("models", []):
        lbl = m["label"]
        stem = m["stem"]
        order = m.get("order", 0)

        # 1. Allocation
        alloc_gmb = m["allocation"]["grounded_minus_bare"]
        a_pt = float(alloc_gmb["point"])
        a_lo = float(alloc_gmb["ci"][0])
        a_hi = float(alloc_gmb["ci"][1])
        a_sig = (a_lo > 0 and a_hi > 0) or (a_lo < 0 and a_hi < 0)
        alloc_diff = (a_pt, a_lo, a_hi, a_sig)

        # 2. Fire danger (Mesogeos)
        meso_gmb = m["mesogeos"]["grounded_minus_bare"]
        m_pt = float(meso_gmb["point"])
        m_lo = float(meso_gmb["ci"][0])
        m_hi = float(meso_gmb["ci"][1])
        m_sig = (m_lo > 0 and m_hi > 0) or (m_lo < 0 and m_hi < 0)
        meso_diff = (m_pt, m_lo, m_hi, m_sig)

        # 3. Tool use
        if m.get("tooluse") is not None:
            tu = m["tooluse"]["tool_minus_bare"]
            t_pt = float(tu["point"])
            t_lo = float(tu["ci"][0])
            t_hi = float(tu["ci"][1])
            t_sig = (t_lo > 0 and t_hi > 0) or (t_lo < 0 and t_hi < 0)
            tool_diff = (t_pt, t_lo, t_hi, t_sig)
        else:
            tool_diff = None

        records.append({
            "model": lbl,
            "stem": stem,
            "order": order,
            "alloc_diff": alloc_diff,
            "meso_diff": meso_diff,
            "tool_diff": tool_diff,
            "raw": m,
        })

    return records


def print_containment_check(records: list[dict], xlim_a: tuple[float, float], xlim_b: tuple[float, float], xlim_c: tuple[float, float]):
    """Programmatically verify that every non-clipped interval lies strictly inside its axis with a margin."""
    print("=" * 115)
    print("PROGRAMMATIC CONTAINMENT CHECK OF INTERVALS AGAINST AXIS LIMITS")
    print(f"Panel (a) Allocation limits: {xlim_a}")
    print(f"Panel (b) Fire danger limits: {xlim_b}")
    print(f"Panel (c) Tool use limits:    {xlim_c}")
    print("=" * 115)

    print("\n--- Panel (a) Allocation Containment (Non-Clipped Models) ---")
    header_a = f"{'Model':<22} | {'Point':<8} | {'[CI Lo, CI Hi]':<20} | {'Margin Left':<12} | {'Margin Right':<12} | {'Status'}"
    print(header_a)
    print("-" * 90)
    for r in records:
        lbl = r["model"]
        pt, lo, hi, _ = r["alloc_diff"]
        if lbl in ("Llama 3.3 70B", "Mistral Small 2402"):
            print(f"{lbl:<22} | {pt:+.4f}  | [{lo:+.4f}, {hi:+.4f}] | {'N/A (Clipped)':<12} | {'N/A (Clipped)':<12} | DELIBERATELY CLIPPED")
            continue
        m_left = lo - xlim_a[0]
        m_right = xlim_a[1] - hi
        ok = m_left > 0 and m_right > 0
        status = "CONTAINED (OK)" if ok else "CUT (FAIL)"
        print(f"{lbl:<22} | {pt:+.4f}  | [{lo:+.4f}, {hi:+.4f}] | {m_left:+12.4f} | {m_right:+12.4f} | {status}")

    print("\n--- Panel (b) Fire Danger Containment (All 19 Models) ---")
    header_b = f"{'Model':<22} | {'Point':<8} | {'[CI Lo, CI Hi]':<20} | {'Margin Left':<12} | {'Margin Right':<12} | {'Status'}"
    print(header_b)
    print("-" * 90)
    for r in records:
        lbl = r["model"]
        pt, lo, hi, _ = r["meso_diff"]
        m_left = lo - xlim_b[0]
        m_right = xlim_b[1] - hi
        ok = m_left > 0 and m_right > 0
        status = "CONTAINED (OK)" if ok else "CUT (FAIL)"
        print(f"{lbl:<22} | {pt:+.4f}  | [{lo:+.4f}, {hi:+.4f}] | {m_left:+12.4f} | {m_right:+12.4f} | {status}")

    print("\n--- Panel (c) Tool Use Containment (17 Evaluated Models) ---")
    header_c = f"{'Model':<22} | {'Point':<8} | {'[CI Lo, CI Hi]':<20} | {'Margin Left':<12} | {'Margin Right':<12} | {'Status'}"
    print(header_c)
    print("-" * 90)
    for r in records:
        lbl = r["model"]
        if r["tool_diff"] is None:
            print(f"{lbl:<22} | {'--':<8} | {'--':<20} | {'--':<12} | {'--':<12} | EMPTY ROW (probe failed)")
            continue
        pt, lo, hi, _ = r["tool_diff"]
        m_left = lo - xlim_c[0]
        m_right = xlim_c[1] - hi
        ok = m_left > 0 and m_right > 0
        status = "CONTAINED (OK)" if ok else "CUT (FAIL)"
        print(f"{lbl:<22} | {pt:+.4f}  | [{lo:+.4f}, {hi:+.4f}] | {m_left:+12.4f} | {m_right:+12.4f} | {status}")
    print("=" * 115)


def _interval(ax, y, pt, lo, hi, col, xlim, marker="o", ms=4.8, lw=1.4, label_dy=0.32):
    """One point with its interval.

    A point beyond the axis limits (Llama 3.3 70B and Mistral Small 2402 in allocation)
    is clipped at the edge, marked with a triangle, and labelled with its value.
    The label is placed at y + label_dy and offset inside the panel so it stays fully inside
    the figure and clear of neighbouring rows and panel titles.
    """
    x0, x1 = xlim
    if pt > x1 or pt < x0:
        edge = x1 if pt > x1 else x0
        ax.plot([max(lo, x0), min(hi, x1)], [y, y], color=col, linewidth=lw, zorder=3, solid_capstyle="butt")
        ax.plot(edge, y, marker=">" if pt > x1 else "<", markersize=ms, color=col, zorder=4, clip_on=False)
        ax.text(
            edge - 0.02 * (x1 - x0) if pt > x1 else edge + 0.02 * (x1 - x0),
            y + label_dy,
            f"{pt:+.2f}".replace("-", "\u2212"),
            fontsize=6.2,
            ha="right" if pt > x1 else "left",
            va="center",
            bbox=dict(facecolor="white", edgecolor="none", pad=0.6),
            zorder=5,
            color=col,
        )
    else:
        capstyle = "round" if lo >= x0 and hi <= x1 else "butt"
        ax.plot([max(lo, x0), min(hi, x1)], [y, y], color=col, linewidth=lw, zorder=3, solid_capstyle=capstyle)
        ax.plot(pt, y, marker=marker, markersize=ms, color=col, zorder=4)


def plot_grounding_effects_text(
    records: list[dict],
    out_pdf: Path | None,
    out_png: Path,
    fig_height: float = 5.4,
    xlim_a: tuple[float, float] = (-0.05, 0.21),
    ticks_a: list[float] = [0.00, 0.08, 0.16],
    xlim_b: tuple[float, float] = (-0.20, 0.10),
    ticks_b: list[float] = [-0.16, -0.08, 0.00, 0.08],
    xlim_c: tuple[float, float] = (-0.06, 1.08),
    ticks_c: list[float] = [0.0, 0.5, 1.0],
):
    """Draw the 3-panel forest plot for the 19 text-only models."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "text.color": COLOR_TEXT,
        "axes.labelcolor": COLOR_TEXT,
        "xtick.color": COLOR_TEXT,
        "ytick.color": COLOR_TEXT,
    })

    n_models = len(records)
    left_margin = 0.23

    fig, axes = plt.subplots(
        1, 3,
        figsize=(fs.TEXT_WIDTH_IN, fig_height),
        sharey=True,
        gridspec_kw={
            "wspace": 0.28,
            "left": left_margin,
            "right": 0.98,
            "top": 0.93,
            "bottom": 0.08,
        }
    )

    y_pos = np.arange(n_models - 1, -1, -1)
    model_names = [r["model"] for r in records]

    def format_ax(ax, x_limits, x_ticks, x_label, title_label):
        ax.set_xlim(x_limits)
        ax.set_xticks(x_ticks)
        ax.set_xlabel(x_label, fontsize=7.2, labelpad=3)
        ax.axvline(0, color=COLOR_GRAY, linestyle="--", linewidth=0.75, zorder=1)
        for y in y_pos:
            ax.axhline(y, color="#F4F4F4", linestyle="-", linewidth=0.6, zorder=0)

        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color(COLOR_GRAY)
            ax.spines[spine].set_linewidth(0.8)

        ax.tick_params(axis="both", length=0, labelsize=6.8)
        ax.tick_params(axis="y", pad=6)
        ax.set_title(title_label, loc="left", fontsize=8.0, fontweight="bold", pad=6)

    # Panel (a): Personnel allocation
    ax_a = axes[0]
    format_ax(
        ax_a,
        x_limits=xlim_a,
        x_ticks=ticks_a,
        x_label="Norm. error diff. (grd. − bare)",
        title_label="(a) Allocation"
    )
    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels(model_names, fontsize=7.2)

    for i, r in enumerate(records):
        y = y_pos[i]
        pt, lo, hi, sig = r["alloc_diff"]
        col = COLOR_CORAL if sig else COLOR_GRAY
        _interval(ax_a, y, pt, lo, hi, col, xlim_a, ms=4.8, lw=1.4, label_dy=0.32)

    # Panel (b): Fire danger
    ax_b = axes[1]
    format_ax(
        ax_b,
        x_limits=xlim_b,
        x_ticks=ticks_b,
        x_label="AUPRC diff. (grd. − bare)",
        title_label="(b) Fire danger"
    )

    for i, r in enumerate(records):
        y = y_pos[i]
        pt, lo, hi, sig = r["meso_diff"]
        col = COLOR_CORAL if sig else COLOR_GRAY
        _interval(ax_b, y, pt, lo, hi, col, xlim_b, ms=4.8, lw=1.4)

    # Panel (c): Tool use
    ax_c = axes[2]
    format_ax(
        ax_c,
        x_limits=xlim_c,
        x_ticks=ticks_c,
        x_label="Accuracy diff. (tool − bare)",
        title_label="(c) Tool use"
    )

    for i, r in enumerate(records):
        y = y_pos[i]
        if r["tool_diff"] is not None:
            pt, lo, hi, sig = r["tool_diff"]
            col = COLOR_CORAL if sig else COLOR_GRAY
            _interval(ax_c, y, pt, lo, hi, col, xlim_c, ms=4.8, lw=1.4)
        else:
            # Empty row with small gray note
            ax_c.text(
                0.51, y, "probe failed",
                color=COLOR_GRAY, fontsize=6.2, fontstyle="italic",
                ha="center", va="center", zorder=4
            )

    ax_a.set_ylim(-0.55, n_models - 0.45)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    if out_pdf is not None:
        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
        print(f"Generated {out_pdf}")
    fig.savefig(out_png, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated {out_png}")


def main():
    parser = argparse.ArgumentParser(description="Generate grounding effects figure for text-only models.")
    parser.add_argument("--repo", type=str, default=str(ROOT),
                        help="Path to repository root.")
    parser.add_argument("--json", type=Path, default=None,
                        help="Path to text_sweep_intervals.json.")
    parser.add_argument("--out-pdf", type=Path, default=None,
                        help="Path for output PDF file.")
    parser.add_argument("--out-png", type=Path, default=None,
                        help="Path for output PNG file.")
    parser.add_argument("--height", type=float, default=5.4,
                        help="Figure height in inches (default: 5.4).")
    parser.add_argument("--xlim-a", type=str, default="-0.05,0.21",
                        help="x-limits for panel (a) as min,max (default: -0.05,0.21).")
    parser.add_argument("--xlim-b", type=str, default="-0.20,0.10",
                        help="x-limits for panel (b) as min,max (default: -0.20,0.10; contains MiniMax M2.5).")
    parser.add_argument("--xlim-c", type=str, default="-0.06,1.08",
                        help="x-limits for panel (c) as min,max (default: -0.06,1.08).")
    args = parser.parse_args()

    repo_root = Path(args.repo).resolve()
    figs_dir = repo_root / "figures"
    out_pdf = args.out_pdf or (figs_dir / "grounding_effects_text.pdf")
    out_png = args.out_png or (figs_dir / "grounding_effects_text.png")

    xlim_a = tuple(float(x.strip()) for x in args.xlim_a.split(","))
    xlim_b = tuple(float(x.strip()) for x in args.xlim_b.split(","))
    xlim_c = tuple(float(x.strip()) for x in args.xlim_c.split(","))

    records = load_data(repo_root, json_path=args.json)
    print_containment_check(records, xlim_a, xlim_b, xlim_c)
    plot_grounding_effects_text(
        records,
        out_pdf,
        out_png,
        fig_height=args.height,
        xlim_a=xlim_a,
        xlim_b=xlim_b,
        xlim_c=xlim_c,
    )


if __name__ == "__main__":
    main()
