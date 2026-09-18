#!/usr/bin/env python
"""Generate the WildFireVQA aerial question-answering figure (figures/wildfirevqa.pdf and .png).

This figure displays the paired aerial question-answering results on WildFireVQA (408 items, 390 FLAME 3 frames):
  - Panel (a): Bare and grounded accuracy per model (pooled over all 408 items) with frame-clustered
               95% bootstrap intervals (20,000 resamples), beside the per-question majority baseline (0.627).
  - Panel (b): Closed-form questions (48 items on types CL1, CMR4, DS7, DS8) against the remaining 360 questions,
               bare and grounded per model, showing the lift on the 48 items answered by construction and
               the small drop on the other 360 items.

Model order (top to bottom):
  1. claude-opus-4.8
  2. claude-opus-5
  3. gemini-3.1-pro
  4. gpt-6-astra
  5. Qwen3-VL
  6. Llama 4 Maverick

Styling conventions (CatchBench submission style):
  - Width: 6.5 in, height <= 2.6 in (2.45 in).
  - Typography: Sans-serif (DejaVu Sans / Arial), TrueType fonts (pdf.fonttype 42), >= 6 pt.
  - Spines: no top/right spines, gray (#999999) left/bottom spines, no tick marks.
  - Palette:
      Coral (#ED8D5A): focal marks (contrasts whose 95% interval excludes zero; lift on closed-form items)
      Gray (#999999 / #C9C9C9): context (contrasts whose 95% interval includes zero; drop on other items)
      Mint (#BFDFD2, edge #8FB7A6): comparison layer (bare condition marks)
      Near-black (#1A1A1A): text
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

# CatchBench Palette
COLOR_CORAL = "#ED8D5A"
COLOR_MINT = "#BFDFD2"
COLOR_MINT_EDGE = "#8FB7A6"
COLOR_GRAY = "#999999"
COLOR_LIGHT_GRAY = "#E6E6E6"
COLOR_TEXT = "#1A1A1A"
COLOR_SUBTITLE = "#666666"

MODELS = [
    "claude-opus-4.8",
    "claude-opus-5",
    "gemini-3.1-pro",
    "gpt-6-astra",
    "Qwen3-VL",
    "Llama 4 Maverick",
]


def load_data(json_path: Path) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_wildfirevqa(data: dict, out_pdf: Path, out_png: Path):
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

    fig, axes = plt.subplots(
        1, 2,
        figsize=(6.5, 2.45),
        sharey=True,
        gridspec_kw={"wspace": 0.22, "left": 0.18, "right": 0.98, "top": 0.83, "bottom": 0.17}
    )

    n_models = len(MODELS)
    # Row 5 is top (claude-opus-4.8), row 0 is bottom (Llama 4 Maverick)
    y_pos = np.arange(n_models - 1, -1, -1)
    model_names = MODELS

    def format_ax(ax, x_limits, x_ticks, x_label, title_label):
        ax.set_xlim(x_limits)
        ax.set_xticks(x_ticks)
        ax.set_xlabel(x_label, fontsize=7.2, labelpad=3)
        for y in y_pos:
            ax.axhline(y, color="#F4F4F4", linestyle="-", linewidth=0.6, zorder=0)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color(COLOR_GRAY)
            ax.spines[spine].set_linewidth(0.8)
        ax.tick_params(axis="both", length=0, labelsize=6.8)
        ax.set_title(title_label, loc="left", fontsize=8.0, fontweight="bold", pad=8)

    # -------------------------------------------------------------
    # Panel (a): Bare and Grounded Accuracy (Pooled, 408 items)
    # -------------------------------------------------------------
    ax_a = axes[0]
    format_ax(
        ax_a,
        x_limits=(0.46, 0.74),
        x_ticks=[0.50, 0.55, 0.60, 0.65, 0.70],
        x_label="Accuracy (pooled, 408 items)",
        title_label="(a) Pooled accuracy"
    )
    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels(model_names, fontsize=7.2)

    # Vertical line at majority baseline 0.627
    maj_acc = data["majority_baseline"]["accuracy"]  # ~0.627
    ax_a.axvline(maj_acc, color=COLOR_GRAY, linestyle="--", linewidth=0.9, zorder=1)
    ax_a.text(
        maj_acc + 0.004, -0.42, f"Majority {maj_acc:.3f}",
        fontsize=6.0, color=COLOR_SUBTITLE, verticalalignment="bottom"
    )

    v_off_a = 0.12
    for i, m in enumerate(MODELS):
        y = y_pos[i]
        m_data = data["models"][m]
        b = m_data["bare"]
        g = m_data["grounded"]
        diff = m_data["grounded_minus_bare"]

        # Contrast status: coral if interval excludes 0, gray otherwise (all include 0 here)
        ci_lo, ci_hi = diff["ci"]
        sig = (ci_lo > 0 and ci_hi > 0) or (ci_lo < 0 and ci_hi < 0)
        col_g = COLOR_CORAL if sig else COLOR_GRAY

        # Bare: mint circle at y - v_off_a
        y_b = y - v_off_a
        ax_a.plot([b["accuracy_ci"][0], b["accuracy_ci"][1]], [y_b, y_b],
                  color=COLOR_MINT_EDGE, linewidth=1.2, zorder=3, solid_capstyle="round")
        ax_a.plot(b["accuracy"], y_b, marker="o", markersize=4.2,
                  markerfacecolor=COLOR_MINT, markeredgecolor=COLOR_MINT_EDGE, markeredgewidth=0.8, zorder=4)

        # Grounded: circle at y + v_off_a
        y_g = y + v_off_a
        ax_a.plot([g["accuracy_ci"][0], g["accuracy_ci"][1]], [y_g, y_g],
                  color=col_g, linewidth=1.2, zorder=3, solid_capstyle="round")
        ax_a.plot(g["accuracy"], y_g, marker="o", markersize=4.2,
                  color=col_g, zorder=4)

        # Connecting vertical guide between bare and grounded points
        ax_a.plot([b["accuracy"], g["accuracy"]], [y_b, y_g],
                  color="#D0D0D0", linestyle=":", linewidth=0.7, zorder=2)

    # Legend for Panel (a)
    legend_elements_a = [
        Line2D([0], [0], marker="o", markersize=4.0, markerfacecolor=COLOR_MINT,
               markeredgecolor=COLOR_MINT_EDGE, markeredgewidth=0.8, linestyle="-",
               color=COLOR_MINT_EDGE, linewidth=1.1, label="Bare"),
        Line2D([0], [0], marker="o", markersize=4.0, color=COLOR_GRAY, linestyle="-",
               linewidth=1.1, label="Grounded"),
    ]
    ax_a.legend(
        handles=legend_elements_a, loc="upper left", frameon=False, fontsize=6.2,
        handletextpad=0.4, handlelength=1.2, borderaxespad=0.3, labelcolor=COLOR_TEXT
    )

    # -------------------------------------------------------------
    # Panel (b): Closed-form 48 vs Other 360 (Bare & Grounded)
    # -------------------------------------------------------------
    ax_b = axes[1]
    format_ax(
        ax_b,
        x_limits=(0.36, 1.02),
        x_ticks=[0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00],
        x_label="Accuracy by question subset",
        title_label="(b) Closed-form 48 vs. other 360"
    )

    # Closed-form rule vertical dashed line at 1.000
    ax_b.axvline(1.0, color=COLOR_MINT_EDGE, linestyle=":", linewidth=0.9, zorder=1)
    ax_b.text(
        0.995, -0.42, "Rule 1.000",
        fontsize=6.0, color=COLOR_SUBTITLE, verticalalignment="bottom", horizontalalignment="right"
    )

    v_off_b = 0.13
    for i, m in enumerate(MODELS):
        y = y_pos[i]
        m_data = data["models"][m]
        b_cf = m_data["bare"]["closed_form_accuracy"]
        g_cf = m_data["grounded"]["closed_form_accuracy"]
        b_oth = m_data["bare"]["other_accuracy"]
        g_oth = m_data["grounded"]["other_accuracy"]

        # Closed-form 48 (top sub-row, y + v_off_b): coral lift
        y_cf = y + v_off_b
        ax_b.annotate(
            "", xy=(g_cf, y_cf), xytext=(b_cf, y_cf),
            arrowprops=dict(arrowstyle="->", color=COLOR_CORAL, lw=1.2, mutation_scale=7),
            zorder=3
        )
        ax_b.plot(b_cf, y_cf, marker="o", markersize=3.6,
                  markerfacecolor=COLOR_MINT, markeredgecolor=COLOR_MINT_EDGE, markeredgewidth=0.7, zorder=4)
        ax_b.plot(g_cf, y_cf, marker="o", markersize=3.8, color=COLOR_CORAL, zorder=4)

        # Other 360 (bottom sub-row, y - v_off_b): gray drop
        y_oth = y - v_off_b
        ax_b.annotate(
            "", xy=(g_oth, y_oth), xytext=(b_oth, y_oth),
            arrowprops=dict(arrowstyle="->", color=COLOR_GRAY, lw=1.0, mutation_scale=6),
            zorder=3
        )
        ax_b.plot(b_oth, y_oth, marker="o", markersize=3.6,
                  markerfacecolor=COLOR_MINT, markeredgecolor=COLOR_MINT_EDGE, markeredgewidth=0.7, zorder=4)
        ax_b.plot(g_oth, y_oth, marker="o", markersize=3.8, color=COLOR_GRAY, zorder=4)

    # Place legend cleanly at lower-left of panel (b) in the empty area x in [0.46, 0.58], y around 1.5-2.2
    legend_elements_b = [
        Line2D([0], [0], marker="o", markersize=3.8, color=COLOR_CORAL, linestyle="-",
               linewidth=1.2, label="Closed-form 48"),
        Line2D([0], [0], marker="o", markersize=3.8, color=COLOR_GRAY, linestyle="-",
               linewidth=1.0, label="Other 360"),
        Line2D([0], [0], marker="o", markersize=3.6, markerfacecolor=COLOR_MINT,
               markeredgecolor=COLOR_MINT_EDGE, markeredgewidth=0.7, linestyle="None",
               label="Bare baseline"),
    ]
    ax_b.legend(
        handles=legend_elements_b, loc="center left", bbox_to_anchor=(0.005, 0.30),
        frameon=False, fontsize=6.0, handletextpad=0.3, handlelength=1.2, borderaxespad=0.0,
        labelcolor=COLOR_TEXT
    )

    # Adjust vertical limits
    ax_a.set_ylim(-0.55, n_models - 0.45)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    fig.savefig(out_png, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated {out_pdf} and {out_png}")


def main():
    parser = argparse.ArgumentParser(description="Generate WildFireVQA figure.")
    parser.add_argument("--input", type=str,
                        default=str(Path(__file__).resolve().parent.parent / "analysis" / "wildfirevqa_paired.json"),
                        help="Path to wildfirevqa_paired.json (written by analysis/wildfirevqa_paired.py in the AI4Fire repository)")
    parser.add_argument("--out-pdf", type=str, default=None,
                        help="Path for output PDF file.")
    parser.add_argument("--out-png", type=str, default=None,
                        help="Path for output PNG file.")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise SystemExit(f"Input data not found at {input_path}")

    repo_root = Path(__file__).resolve().parent.parent
    out_pdf = Path(args.out_pdf).resolve() if args.out_pdf else repo_root / "figures" / "wildfirevqa.pdf"
    out_png = Path(args.out_png).resolve() if args.out_png else repo_root / "figures" / "wildfirevqa.png"

    data = load_data(input_path)
    plot_wildfirevqa(data, out_pdf, out_png)


if __name__ == "__main__":
    main()
