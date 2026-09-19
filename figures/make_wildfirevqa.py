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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import models
from models import add_model_args, resolve_models

# CatchBench Palette
COLOR_CORAL = "#ED8D5A"
COLOR_MINT = "#BFDFD2"
COLOR_MINT_EDGE = "#8FB7A6"
COLOR_GRAY = "#999999"
COLOR_LIGHT_GRAY = "#E6E6E6"
COLOR_TEXT = "#1A1A1A"
COLOR_SUBTITLE = "#666666"


def load_data(json_path: Path) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_wildfirevqa(data: dict, selected_models, out_pdf: Path | None, out_png: Path):
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

    models_dict = data.get("models", {})
    models_to_plot = []
    for m in selected_models:
        if m.label in models_dict:
            models_to_plot.append(m.label)
        elif m.stem in models_dict:
            models_to_plot.append(m.stem)

    if not models_to_plot:
        models_to_plot = list(models_dict.keys())

    n_models = len(models_to_plot)
    fig_height = 2.45 if n_models <= 6 else max(2.45, 0.35 * n_models + 0.6)
    left_margin = 0.18 if n_models <= 6 else 0.22

    fig, axes = plt.subplots(
        1, 2,
        figsize=(6.5, fig_height),
        sharey=True,
        gridspec_kw={"wspace": 0.22, "left": left_margin, "right": 0.98,
                     "top": 0.83 if n_models <= 6 else 0.90,
                     "bottom": 0.17 if n_models <= 6 else 0.10}
    )

    y_pos = np.arange(n_models - 1, -1, -1)
    model_names = models_to_plot

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

    # Panel (a): Bare and Grounded Accuracy (Pooled, 408 items)
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

    maj_acc = data["majority_baseline"]["accuracy"]
    ax_a.axvline(maj_acc, color=COLOR_GRAY, linestyle="--", linewidth=0.9, zorder=1)
    ax_a.text(
        maj_acc + 0.004, -0.42, f"Majority {maj_acc:.3f}",
        fontsize=6.0, color=COLOR_SUBTITLE, verticalalignment="bottom"
    )

    v_off_a = 0.12
    for i, m in enumerate(models_to_plot):
        y = y_pos[i]
        m_data = models_dict[m]
        b = m_data["bare"]
        g = m_data["grounded"]
        diff = m_data["grounded_minus_bare"]

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

    # Panel (b): Closed-form 48 vs Other 360
    ax_b = axes[1]
    format_ax(
        ax_b,
        x_limits=(0.36, 1.02),
        x_ticks=[0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00],
        x_label="Accuracy by question subset",
        title_label="(b) Closed-form 48 vs. other 360"
    )

    ax_b.axvline(1.0, color=COLOR_MINT_EDGE, linestyle=":", linewidth=0.9, zorder=1)
    ax_b.text(
        0.995, -0.42, "Rule 1.000",
        fontsize=6.0, color=COLOR_SUBTITLE, verticalalignment="bottom", horizontalalignment="right"
    )

    v_off_b = 0.13
    for i, m in enumerate(models_to_plot):
        y = y_pos[i]
        m_data = models_dict[m]
        b_cf = m_data["bare"]["closed_form_accuracy"]
        g_cf = m_data["grounded"]["closed_form_accuracy"]
        b_oth = m_data["bare"]["other_accuracy"]
        g_oth = m_data["grounded"]["other_accuracy"]

        # Closed-form 48 (top sub-row, y + v_off_b)
        y_cf = y + v_off_b
        ax_b.annotate(
            "", xy=(g_cf, y_cf), xytext=(b_cf, y_cf),
            arrowprops=dict(arrowstyle="->", color=COLOR_CORAL, lw=1.2, mutation_scale=7),
            zorder=3
        )
        ax_b.plot(b_cf, y_cf, marker="o", markersize=3.6,
                  markerfacecolor=COLOR_MINT, markeredgecolor=COLOR_MINT_EDGE, markeredgewidth=0.7, zorder=4)
        ax_b.plot(g_cf, y_cf, marker="o", markersize=3.8, color=COLOR_CORAL, zorder=4)

        # Other 360 (bottom sub-row, y - v_off_b)
        y_oth = y - v_off_b
        ax_b.annotate(
            "", xy=(g_oth, y_oth), xytext=(b_oth, y_oth),
            arrowprops=dict(arrowstyle="->", color=COLOR_GRAY, lw=1.0, mutation_scale=6),
            zorder=3
        )
        ax_b.plot(b_oth, y_oth, marker="o", markersize=3.6,
                  markerfacecolor=COLOR_MINT, markeredgecolor=COLOR_MINT_EDGE, markeredgewidth=0.7, zorder=4)
        ax_b.plot(g_oth, y_oth, marker="o", markersize=3.8, color=COLOR_GRAY, zorder=4)

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
    parser = argparse.ArgumentParser(description="Generate WildFireVQA figure.")
    parser.add_argument("--input", type=Path,
                        default=Path(__file__).resolve().parent.parent / "analysis" / "wildfirevqa_paired.json",
                        help="Path to wildfirevqa_paired.json")
    parser.add_argument("--out-pdf", type=Path, default=None,
                        help="Path for output PDF file.")
    parser.add_argument("--out-png", type=Path, default=None,
                        help="Path for output PNG file.")
    add_model_args(parser, default_tier="core")
    args = parser.parse_args()

    input_path = args.input.resolve()
    if not input_path.exists():
        raise SystemExit(f"Input data not found at {input_path}")

    repo_root = Path(__file__).resolve().parent.parent
    figs_dir = repo_root / "figures"
    if args.out_pdf:
        out_pdf = args.out_pdf
    elif args.out_png:
        out_pdf = None
    else:
        out_pdf = figs_dir / "wildfirevqa.pdf"
    out_png = args.out_png or (figs_dir / "wildfirevqa.png")

    selected_models = resolve_models(args, task="wildfirevqa", default_tier="core")

    data = load_data(input_path)
    plot_wildfirevqa(data, selected_models, out_pdf, out_png)


if __name__ == "__main__":
    main()
