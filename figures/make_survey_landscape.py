#!/usr/bin/env python
"""Generate the survey collection figure (figures/survey_landscape.pdf and .png).

This figure displays the shape of the 138 kept survey works:
  - Panel (a): Matrix of task category (ten rows, Table tab:collection order) by kind of work
               (seven columns, Table tab:collection order), with a shaded cell per count,
               column totals at the top (kinds: 75, 21, 14, 14, 8, 5, 1 summing to 138),
               and row totals at the right (task tags: 71, 52, 37, 35, 30, 30, 26, 20, 10, 5).
               Coral highlights the single largest cell (39: decision-support-operations x system-with-eval),
               marking the focal point of the literature, with gray scale for all other cells.
  - Panel (b): Small bar panel of works by number of models evaluated (unknown 34, one 12,
               two to three 26, four to nine 16, ten or more 2) across the 90 works marked
               as LLM evaluations by the keyword screen, all in CatchBench gray.

Data sources in the AI4Fire repository:
  - survey/prior-art-r2-2026-09-13.json:
    138 records of kept works with 'kind', 'task_categories', 'n_models', and 'llm_keyword_screen'.
  - survey/collection_counts.py:
    computes every count of Table tab:collection including the model-tested binning rule and screen.

Styling conventions (CatchBench / AI4Fire style via figures/figstyle.py):
  - Width: fs.TEXT_WIDTH_IN (6.5 in), height: 2.85 in (<= 3.0 in).
  - Typography: Sans-serif (DejaVu Sans / Arial), TrueType fonts (pdf.fonttype 42), >= 6.2 pt.
  - Palette:
      Coral (#ED8D5A): focal mark (single largest cell: decision-support x system-with-eval, 39)
      Gray (#999999 / ramp): context cells and panel (b) bars
      Near-black (#1A1A1A): text
"""

from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Import shared styling
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import figstyle as fs


# Task categories in Table tab:collection order and taxonomy of Section 2
TASK_CATEGORIES = [
    "decision-support-operations",
    "detection-perception",
    "geospatial-analysis",
    "document-understanding",
    "simulation-coupled-agents",
    "communication-alerts",
    "knowledge-qa",
    "forecasting-prediction",
    "data-retrieval-tools",
    "other",
]

# Kinds in Table tab:collection order
PAPER_KINDS = [
    "system-with-eval",
    "evaluation",
    "benchmark",
    "survey",
    "position",
    "dataset",
    "system-no-eval",
]

# Model buckets from survey/collection_counts.py
MODEL_BUCKET_KEYS = ["unknown", "one", "2-3", "4-9", "10 or more"]
MODEL_BUCKET_LABELS = ["unknown", "one", "2–3", "4–9", "10 or more"]

# Expected counts from Table tab:collection (for verification)
EXPECTED_KINDS = {
    "system-with-eval": 75,
    "evaluation": 21,
    "benchmark": 14,
    "survey": 14,
    "position": 8,
    "dataset": 5,
    "system-no-eval": 1,
}

EXPECTED_TAGS = {
    "decision-support-operations": 71,
    "detection-perception": 52,
    "geospatial-analysis": 37,
    "document-understanding": 35,
    "simulation-coupled-agents": 30,
    "communication-alerts": 30,
    "knowledge-qa": 26,
    "forecasting-prediction": 20,
    "data-retrieval-tools": 10,
    "other": 5,
}

EXPECTED_MODELS = {
    "unknown": 34,
    "one": 12,
    "2-3": 26,
    "4-9": 16,
    "10 or more": 2,
}


def bucket_models(n: int | str | None) -> str:
    """Classify model count into discrete buckets matching collection_counts.py."""
    if n is None or n == "unknown":
        return "unknown"
    n = int(n)
    if n == 1:
        return "one"
    if n <= 3:
        return "2-3"
    if n <= 9:
        return "4-9"
    return "10 or more"


def load_and_compute_counts(repo_root: Path):
    """Load prior-art data and compute matrix and model bucket counts."""
    data_path = repo_root / "survey" / "prior-art-r2-2026-09-13.json"
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    works = data["works"]

    n_cats = len(TASK_CATEGORIES)
    n_kinds = len(PAPER_KINDS)
    cat_to_idx = {c: i for i, c in enumerate(TASK_CATEGORIES)}
    kind_to_idx = {k: j for j, k in enumerate(PAPER_KINDS)}

    matrix = np.zeros((n_cats, n_kinds), dtype=int)
    for w in works:
        k = w["kind"]
        j = kind_to_idx[k]
        for t in w["task_categories"]:
            i = cat_to_idx[t]
            matrix[i, j] += 1

    row_totals = [sum(t == c for w in works for t in w["task_categories"]) for c in TASK_CATEGORIES]
    col_totals = [sum(w["kind"] == k for w in works) for k in PAPER_KINDS]

    # Model evaluation counts over keyword screened works
    screened = [w for w in works if w.get("llm_keyword_screen")]
    evals = [w for w in screened if w["kind"] in ("system-with-eval", "evaluation", "benchmark")]
    models_counter = collections.Counter(bucket_models(w.get("n_models")) for w in evals)
    model_counts = [models_counter.get(k, 0) for k in MODEL_BUCKET_KEYS]

    return {
        "works": works,
        "matrix": matrix,
        "row_totals": row_totals,
        "col_totals": col_totals,
        "screened_count": len(screened),
        "evals_count": len(evals),
        "model_counts": model_counts,
        "models_counter": dict(models_counter),
    }


def print_verification_table(data: dict) -> bool:
    """Print verification table matching computed numbers against Table tab:collection."""
    all_match = True
    print("\n" + "=" * 78)
    print(f"{'ITEM':<32} {'COMPUTED':>10} {'PAPER TABLE':>14} {'STATUS':>12}")
    print("=" * 78)

    print("\n--- Paper Kinds (Column Totals) ---")
    for k, comp in zip(PAPER_KINDS, data["col_totals"]):
        exp = EXPECTED_KINDS[k]
        match = (comp == exp)
        all_match = all_match and match
        status = "MATCH" if match else "MISMATCH"
        print(f"  {k:<30} {comp:>10} {exp:>14} {status:>12}")
    print(f"  {'Total works':<30} {sum(data['col_totals']):>10} {138:>14} {'MATCH':>12}")

    print("\n--- Task Categories (Row Totals) ---")
    for c, comp in zip(TASK_CATEGORIES, data["row_totals"]):
        exp = EXPECTED_TAGS[c]
        match = (comp == exp)
        all_match = all_match and match
        status = "MATCH" if match else "MISMATCH"
        print(f"  {c:<30} {comp:>10} {exp:>14} {status:>12}")
    print(f"  {'Total category tags':<30} {sum(data['row_totals']):>10} {316:>14} {'MATCH':>12}")

    print("\n--- Models Tested (90 Evaluated Works) ---")
    for k, comp in zip(MODEL_BUCKET_KEYS, data["model_counts"]):
        exp = EXPECTED_MODELS[k]
        match = (comp == exp)
        all_match = all_match and match
        status = "MATCH" if match else "MISMATCH"
        print(f"  {k:<30} {comp:>10} {exp:>14} {status:>12}")
    print(f"  {'Total evaluated works':<30} {sum(data['model_counts']):>10} {90:>14} {'MATCH':>12}")

    print("=" * 78)
    print(f"Overall Verification: {'ALL MATCH' if all_match else 'FAILED'}\n")
    return all_match


def plot_panel_a(ax: plt.Axes, matrix: np.ndarray, row_totals: list[int], col_totals: list[int]) -> None:
    """Render Panel (a): Task category x Kind matrix with marginal totals and focal cell."""
    n_rows = len(TASK_CATEGORIES)
    n_cols = len(PAPER_KINDS)
    vmax_gray = 30.0

    for i in range(n_rows):
        for j in range(n_cols):
            val = matrix[i, j]
            if i == 0 and j == 0:
                # Single largest cell (39 works): CatchBench CORAL focal highlight
                cell_color = fs.CORAL
                text_color = "white"
                edge_color = "#D46D3A"
            elif val == 0:
                cell_color = "#F8F8F8"
                text_color = "#C4C4C4"
                edge_color = "#EDEDED"
            else:
                # Grayscale ramp
                frac = (val / vmax_gray) ** 0.65
                r = 0.94 - frac * (0.94 - 0.32)
                cell_color = (r, r, r)
                text_color = "white" if frac > 0.50 else fs.NEAR_BLACK
                edge_color = (r * 0.9, r * 0.9, r * 0.9)

            rect = plt.Rectangle(
                (j - 0.45, n_rows - 1 - i - 0.45), 0.9, 0.9,
                facecolor=cell_color, edgecolor=edge_color, linewidth=0.5, zorder=2
            )
            ax.add_patch(rect)

            label_text = str(val) if val > 0 else "·"
            fontweight = "bold" if (i == 0 and j == 0) else "normal"
            ax.text(
                j, n_rows - 1 - i, label_text, ha="center", va="center",
                fontsize=fs.FS_TICK, color=text_color, fontweight=fontweight, zorder=3
            )

    # Row totals on right
    for i in range(n_rows):
        rt = row_totals[i]
        ax.text(
            n_cols - 0.5 + 0.35, n_rows - 1 - i, str(rt), ha="left", va="center",
            fontsize=fs.FS_TICK, fontweight="bold", color=fs.NEAR_BLACK
        )

    ax.text(
        n_cols - 0.5 + 0.35, n_rows + 0.15, "Total", ha="left", va="bottom",
        fontsize=fs.FS_SMALL, fontweight="bold", color=fs.SUBTITLE
    )

    # Column totals at top
    for j in range(n_cols):
        ct = col_totals[j]
        ax.text(
            j, n_rows + 0.15, str(ct), ha="center", va="bottom",
            fontsize=fs.FS_TICK, fontweight="bold", color=fs.NEAR_BLACK
        )

    ax.text(
        -0.55, n_rows + 0.15, "Total:", ha="right", va="bottom",
        fontsize=fs.FS_SMALL, fontweight="bold", color=fs.SUBTITLE
    )

    # Column names at top, angled at 32 degrees
    for j, k_name in enumerate(PAPER_KINDS):
        ax.text(
            j - 0.15, n_rows + 1.1, k_name, ha="left", va="bottom",
            rotation=32, fontsize=fs.FS_SMALL, color=fs.NEAR_BLACK
        )

    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(reversed(TASK_CATEGORIES), fontsize=fs.FS_TICK)
    ax.tick_params(axis="y", left=False, right=False, length=0)
    ax.set_xticks([])
    ax.set_xlim(-0.5, n_cols + 0.6)
    ax.set_ylim(-0.55, n_rows + 0.9)

    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)


def plot_panel_b(ax: plt.Axes, model_counts: list[int]) -> None:
    """Render Panel (b): Horizontal bars for number of models tested across 90 evaluated works."""
    fs.bare(ax, grid="x")
    y_pos = np.arange(len(MODEL_BUCKET_LABELS))[::-1]  # 'unknown' at top to match Table tab:collection
    bars = ax.barh(y_pos, model_counts, height=0.60, color=fs.GRAY, edgecolor="none", zorder=3)

    for i, count in enumerate(model_counts):
        ax.text(
            count + 1.0, y_pos[i], str(count), ha="left", va="center",
            fontsize=fs.FS_TICK, color=fs.NEAR_BLACK, fontweight="normal"
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(MODEL_BUCKET_LABELS, fontsize=fs.FS_TICK)
    ax.set_xlabel("Number of works (90 evaluated)", fontsize=fs.FS_AXIS)
    ax.set_xlim(0, 42)
    ax.set_ylim(-0.65, len(MODEL_BUCKET_LABELS) - 0.35)


def plot_survey_landscape(data: dict, out_pdf: Path, out_png: Path) -> None:
    """Plot the full survey landscape figure at publication dimensions and save."""
    fs.apply()
    fig = plt.figure(figsize=(fs.TEXT_WIDTH_IN, 2.85))

    gs = fig.add_gridspec(
        1, 2,
        width_ratios=[2.05, 1.0],
        wspace=0.36,
        left=0.25,
        right=0.965,
        top=0.72,
        bottom=0.12
    )

    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])

    plot_panel_a(ax_a, data["matrix"], data["row_totals"], data["col_totals"])
    plot_panel_b(ax_b, data["model_counts"])

    # Align panel titles at figure y = 0.95
    fig.text(0.015, 0.95, "(a) Works by kind and task category",
             fontsize=fs.FS_TITLE, fontweight="bold", color=fs.NEAR_BLACK, ha="left", va="top")

    pos_b = ax_b.get_position()
    fig.text(pos_b.x0, 0.95, "(b) Models evaluated",
             fontsize=fs.FS_TITLE, fontweight="bold", color=fs.NEAR_BLACK, ha="left", va="top")

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fs.savefig(fig, out_pdf, out_png, dpi=300)
    plt.close(fig)
    print(f"Generated {out_pdf} and {out_png}")


def main():
    parser = argparse.ArgumentParser(description="Generate survey landscape figure.")
    parser.add_argument("--repo", type=str, default=str(Path(__file__).resolve().parent.parent),
                        help="Path to AI4Fire repository root.")
    parser.add_argument("--out-pdf", type=str, default=None,
                        help="Path for output PDF file.")
    parser.add_argument("--out-png", type=str, default=None,
                        help="Path for output PNG file.")
    args = parser.parse_args()

    repo_root = Path(args.repo).resolve()
    out_pdf = Path(args.out_pdf).resolve() if args.out_pdf else repo_root / "figures" / "survey_landscape.pdf"
    out_png = Path(args.out_png).resolve() if args.out_png else repo_root / "figures" / "survey_landscape.png"

    data = load_and_compute_counts(repo_root)
    print_verification_table(data)
    plot_survey_landscape(data, out_pdf, out_png)


if __name__ == "__main__":
    main()
