#!/usr/bin/env python
"""Generate Figure F3: allocation analogues (figures/allocation_analogues.pdf and .png).

This figure illustrates the two key allocation findings from Section 7.1:
  - Panel (a): Copying behavior under retrieval rule v1 across six models (300 items each).
               x is the displayed median next-day ratio of the six v1 analogues (at two decimals),
               y is the model's grounded-v1 prediction divided by persistence (its own next-day ratio).
               Points matching the displayed-median rule (prediction == round(persistence * shown_median))
               are plotted in coral; other predictions are in gray. Points fall on log2 axes with a gray
               diagonal reference line (y = x). Axes run from 0.18 to 5.5 so every point is shown; extreme ratios (up to 4.82 shown,
               4.86 prediction) align directly along the diagonal for copying models.
               Corner labels state copy counts: 123, 82, 154, 199, 284, 259 of 300.
  - Panel (b): Next-day ratio distributions over the 300 evaluation items for:
               1) Filed next-day ratio (ground truth: target / persistence; gray, IQR [0.93, 1.02])
               2) Rule v1 displayed analogue median (focal problem; coral, IQR [0.93, 1.07])
               3) Rule v2 displayed analogue median (narrowed retrieval; mint, IQR [0.96, 1.02])
               Horizontal boxplots with median at 1.00, interquartile boxes, 5th-95th percentile
               whiskers, and annotated quartile pairs.

Data sources:
  - Evaluation items: run_allocation.sample_items() (300 items with baseline_persistence, target_personnel)
  - Analogue pool: run_allocation.build_pool() from ICS-209-PLUS sitreps data/ics209/...
  - Rule v1 draws: task-allocation/responses-<stem>-grounded.jsonl (analogue_ids)
  - Rule v2 draws: task-allocation/responses-<stem>-grounded-v2.jsonl and rule-v2-draws.jsonl
  - Model grounded predictions: task-allocation/responses-<stem>-grounded.jsonl (prediction)

Outputs:
  - figures/allocation_analogues.pdf
  - figures/allocation_analogues.png
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Add figures/ to sys.path for figstyle
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import figstyle as fs


def load_data(repo_dir: Path, cache_path: Path | None = None) -> list[dict]:
    """Load or build the 300 evaluation item data records."""
    if cache_path and cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)

    import run_allocation as ra

    task_dir = repo_dir / "task-allocation"
    items = ra.sample_items()
    eval_incidents = {it["incident_id"] for it in items}
    pool = ra.build_pool(eval_incidents)
    flat = {r["analogue_id"]: (r["today"], r["next"]) for rows in pool.values() for r in rows}

    # Load v1 draws from any grounded file (identical across models)
    v1_file = task_dir / "responses-claude-opus-5-grounded.jsonl"
    v1_rows = [json.loads(l) for l in v1_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    v1_draws = {r["item_id"]: r.get("analogue_ids", []) for r in v1_rows}

    # Load v2 draws
    v2_file = task_dir / "responses-claude-opus-5-grounded-v2.jsonl"
    v2_rows = [json.loads(l) for l in v2_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    v2_draws = {r["item_id"]: r.get("analogue_ids", []) for r in v2_rows}

    # Load model predictions
    model_rows = {}
    for m in fs.MODELS:
        stem = m["stem"]
        p = task_dir / f"responses-{stem}-grounded.jsonl"
        model_rows[stem] = {r["item_id"]: r for r in (json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip())}

    records = []
    for it in items:
        iid = it["item_id"]
        persist = float(it["baseline_persistence"])
        target = float(it["target_personnel"])

        ids1 = v1_draws[iid]
        r1 = [flat[a][1] / max(flat[a][0], 1) for a in ids1]
        v1_exact = float(np.median(r1))
        v1_shown = float("%.2f" % v1_exact)
        v1_rule = int(round(persist * v1_shown))

        ids2 = v2_draws[iid]
        r2 = [flat[a][1] / max(flat[a][0], 1) for a in ids2]
        v2_exact = float(np.median(r2))
        v2_shown = float("%.2f" % v2_exact)

        item_dict = {
            "item_id": iid,
            "incident_id": it["incident_id"],
            "persistence": persist,
            "target": target,
            "actual_ratio": target / max(persist, 1.0),
            "v1_shown": v1_shown,
            "v1_exact": v1_exact,
            "v1_rule": v1_rule,
            "v2_shown": v2_shown,
            "v2_exact": v2_exact,
            "models": {},
        }

        for m in fs.MODELS:
            stem = m["stem"]
            mr = model_rows[stem].get(iid, {})
            pred = mr.get("prediction")
            item_dict["models"][stem] = {
                "pred": pred,
                "ratio": float(pred / max(persist, 1.0)) if pred is not None else None,
                "is_copy": bool(pred == v1_rule) if pred is not None else False,
            }
        records.append(item_dict)

    return records


def verify_numbers(records: list[dict], repo_dir: Path) -> dict[str, bool]:
    """Verify all plotted values row-by-row against paper numbers."""
    paper_copies = {
        "claude-opus-4.8": 123,
        "claude-opus-5": 82,
        "gemini-3.1-pro": 154,
        "gpt-6-astra": 199,
        "Qwen3-VL": 284,
        "Llama 4 Maverick": 259,
    }

    print("=" * 78)
    print("VERIFICATION TABLE: Reproduction of Paper Quantities")
    print("=" * 78)

    checks = {}
    print("\n1. Copy counts under retrieval rule v1 (300 items):")
    print(f"   {'Model':<20} {'Computed':<12} {'Paper':<10} {'Status'}")
    print("   " + "-" * 50)
    for m in fs.MODELS:
        label = m["label"]
        stem = m["stem"]
        computed = sum(it["models"][stem]["is_copy"] for it in records)
        expected = paper_copies[label]
        match = (computed == expected)
        checks[f"copies_{label}"] = match
        print(f"   {label:<20} {computed:>3} of 300    {expected:>3} of 300   {'OK' if match else 'MISMATCH'}")

    # Quartiles
    actual = [it["actual_ratio"] for it in records]
    v1_shown = [it["v1_shown"] for it in records]
    v2_shown = [it["v2_shown"] for it in records]

    q_actual = [float("%.2f" % np.percentile(actual, 25)), float("%.2f" % np.percentile(actual, 75))]
    q_v1 = [float("%.2f" % np.percentile(v1_shown, 25)), float("%.2f" % np.percentile(v1_shown, 75))]
    q_v2 = [float("%.2f" % np.percentile(v2_shown, 25)), float("%.2f" % np.percentile(v2_shown, 75))]

    print("\n2. Quartile pairs of next-day ratio distributions:")
    print(f"   {'Distribution':<28} {'Computed':<16} {'Paper':<14} {'Status'}")
    print("   " + "-" * 68)
    for name, comp, exp, key in [
        ("Filed next-day ratio", q_actual, [0.93, 1.02], "quartiles_filed"),
        ("Rule v1 displayed median", q_v1, [0.93, 1.07], "quartiles_v1"),
        ("Rule v2 displayed median", q_v2, [0.96, 1.02], "quartiles_v2"),
    ]:
        match = (comp == exp)
        checks[key] = match
        print(f"   {name:<28} [{comp[0]:.2f}, {comp[1]:.2f}]     [{exp[0]:.2f}, {exp[1]:.2f}]       {'OK' if match else 'MISMATCH'}")

    # Analogue-only rule normalized errors
    retrieval_v2_json = repo_dir / "analysis" / "retrieval_v2.json"
    if retrieval_v2_json.exists():
        with open(retrieval_v2_json, encoding="utf-8") as f:
            v2_data = json.load(f)
        err_v1 = float("%.3f" % v2_data["rule-v1"]["nmae"])
        err_v2 = float("%.3f" % v2_data["rule-v2"]["nmae"])
        print("\n3. Analogue-only rule normalized error (Sec 7.1 prose & Table 4):")
        print(f"   Rule v1: computed {err_v1:.3f} vs paper 0.249 -> {'OK' if err_v1 == 0.249 else 'MISMATCH'}")
        print(f"   Rule v2: computed {err_v2:.3f} vs paper 0.190 -> {'OK' if err_v2 == 0.190 else 'MISMATCH'}")
        checks["err_v1"] = (err_v1 == 0.249)
        checks["err_v2"] = (err_v2 == 0.190)

    print("=" * 78)
    all_ok = all(checks.values())
    print(f"Overall Verification Result: {'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED'}\n")
    return checks


def plot_figure(records: list[dict], out_pdf: Path, out_png: Path) -> None:
    """Render the allocation analogues figure and write PDF and PNG."""
    fs.apply()

    fig = plt.figure(figsize=(fs.TEXT_WIDTH_IN, 3.10))
    # 2 rows, 4 columns: cols 0-2 for 6 models (a), col 3 for ratio distributions (b)
    gs = fig.add_gridspec(2, 4, width_ratios=[1.0, 1.0, 1.0, 1.65], wspace=0.28, hspace=0.38,
                          left=0.07, right=0.98, bottom=0.13, top=0.85)

    models = fs.MODELS
    axes_a = []

    # (a) Model scatter panels
    for idx, m in enumerate(models):
        r = idx // 3
        c = idx % 3
        ax = fig.add_subplot(gs[r, c])
        fs.bare(ax, grid=None)
        axes_a.append(ax)

        stem = m["stem"]
        label = m["label"]

        x_pts = []
        y_pts = []
        colors = []
        copies = 0

        for it in records:
            x = it["v1_shown"]
            mr = it["models"][stem]
            y = mr["ratio"]
            is_copy = mr["is_copy"]
            if is_copy:
                copies += 1
                colors.append(fs.CORAL)
            else:
                colors.append(fs.GRAY)
            x_pts.append(x)
            y_pts.append(y)

        x_pts = np.array(x_pts)
        y_pts = np.array(y_pts)
        colors = np.array(colors)

        # Log2 axes: preserves multiplicative symmetry and keeps all extreme ratios visible
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        lims = (0.18, 5.5)
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        # Diagonal reference line y = x
        ax.plot([0.18, 5.5], [0.18, 5.5], color=fs.LIGHT_GRAY, lw=0.9, ls="--", zorder=1)

        # Non-copies (gray) plotted beneath copies (coral)
        mask_copy = (colors == fs.CORAL)
        ax.scatter(x_pts[~mask_copy], y_pts[~mask_copy], c=fs.GRAY, s=6, alpha=0.45, zorder=2, edgecolors="none")
        ax.scatter(x_pts[mask_copy], y_pts[mask_copy], c=fs.CORAL, s=8, alpha=0.90, zorder=3, edgecolors="none")

        ax.set_xticks([0.25, 0.5, 1.0, 2.0, 4.0])
        ax.set_xticklabels(["0.25", "0.5", "1", "2", "4"] if r == 1 else [])
        ax.set_yticks([0.25, 0.5, 1.0, 2.0, 4.0])
        ax.set_yticklabels(["0.25", "0.5", "1", "2", "4"] if c == 0 else [])

        # Subpanel model label
        ax.set_title(label, fontsize=fs.FS_TITLE - 0.5, pad=3, fontweight="bold")

        # Copy count inset in top-left
        ax.text(0.06, 0.90, f"{copies} of 300", transform=ax.transAxes, ha="left", va="top",
                fontsize=fs.FS_TICK, fontweight="bold", color=fs.NEAR_BLACK)

    # Panel (a) titles and axis labels
    fig.text(0.07, 0.96, "(a) Model response vs. analogue median", fontsize=fs.FS_TITLE, fontweight="bold", color=fs.NEAR_BLACK)
    fig.text(0.07, 0.90, "Coral: prediction equals the displayed-median rule; gray: other predictions",
             fontsize=fs.FS_SMALL, color=fs.SUBTITLE)
    fig.text(0.24, 0.02, "Displayed analogue median ratio (v1)", ha="center", fontsize=fs.FS_AXIS)
    fig.text(0.015, 0.49, "Model next-day ratio (prediction / persistence)", va="center", rotation="vertical", fontsize=fs.FS_AXIS)

    # (b) Next-day ratio distributions
    ax_b = fig.add_subplot(gs[:, 3])
    fs.bare(ax_b, grid="x")
    fs.panel_title(ax_b, "(b)", "Next-day ratio spread", x=0.0, y=1.05)

    actual = [it["actual_ratio"] for it in records]
    v1_shown = [it["v1_shown"] for it in records]
    v2_shown = [it["v2_shown"] for it in records]

    # Series ordered: Filed (top), Rule v1 (middle), Rule v2 (bottom)
    series = [
        ("Rule v2 analogue median", v2_shown, "#D8EFE7", fs.MINT_EDGE, "0.96", "1.02"),
        ("Rule v1 analogue median", v1_shown, "#FDE8E0", fs.CORAL, "0.93", "1.07"),
        ("Filed next-day ratio", actual, "#ECECEC", fs.GRAY, "0.93", "1.02"),
    ]

    y_pos = [0, 1, 2]
    ax_b.set_ylim(-0.55, 2.70)
    ax_b.set_xlim(0.45, 1.55)

    # Reference line at x = 1.0 (persistence)
    ax_b.axvline(1.0, color=fs.LIGHT_GRAY, lw=0.9, ls="--", zorder=1)

    for y, (name, vals, fill_c, edge_c, q1_str, q3_str) in zip(y_pos, series):
        q25 = float(np.percentile(vals, 25))
        q50 = float(np.percentile(vals, 50))
        q75 = float(np.percentile(vals, 75))
        p5 = float(np.percentile(vals, 5))
        p95 = float(np.percentile(vals, 95))

        # Box
        h = 0.32
        rect = plt.Rectangle((q25, y - h / 2), q75 - q25, h, facecolor=fill_c, edgecolor=edge_c, lw=1.3, zorder=3)
        ax_b.add_patch(rect)
        # Median line
        ax_b.plot([q50, q50], [y - h / 2, y + h / 2], color=edge_c, lw=2.0, zorder=4)
        # Whiskers (5th to 95th percentile)
        ax_b.plot([p5, q25], [y, y], color=edge_c, lw=1.1, zorder=2)
        ax_b.plot([q75, p95], [y, y], color=edge_c, lw=1.1, zorder=2)
        ax_b.plot([p5, p5], [y - h / 4, y + h / 4], color=edge_c, lw=1.1, zorder=2)
        ax_b.plot([p95, p95], [y - h / 4, y + h / 4], color=edge_c, lw=1.1, zorder=2)

        # Centered text above and below the box
        ax_b.text(1.00, y + 0.27, name, fontsize=fs.FS_AXIS, fontweight="bold", ha="center", color=fs.NEAR_BLACK)
        ax_b.text(1.00, y - 0.29, f"Quartiles: [{q1_str}, {q3_str}]", fontsize=fs.FS_SMALL, ha="center",
                  color=edge_c if edge_c != fs.GRAY else fs.SUBTITLE, fontweight="bold")

    ax_b.set_yticks([])
    ax_b.set_xlabel("Ratio to persistence (1.0 = no change)", fontsize=fs.FS_AXIS)
    ax_b.set_xticks([0.6, 0.8, 1.0, 1.2, 1.4])

    fs.savefig(fig, out_pdf, out_png)
    plt.close(fig)
    print(f"Generated {out_pdf}")
    print(f"Generated {out_png}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT, help="Path to AI4Fire repository root.")
    parser.add_argument("--cache", type=Path, default=None, help="Optional JSON cache of the 300 records; the default rebuilds them from the SIT record.")
    parser.add_argument("--out-pdf", type=Path, default=None, help="Output PDF path.")
    parser.add_argument("--out-png", type=Path, default=None, help="Output PNG path.")
    args = parser.parse_args()

    repo_dir = args.repo.resolve()
    figs_dir = repo_dir / "figures"
    out_pdf = args.out_pdf or (figs_dir / "allocation_analogues.pdf")
    out_png = args.out_png or (figs_dir / "allocation_analogues.png")

    cache_path = args.cache

    print(f"Loading data from repo: {repo_dir} (cache: {cache_path})")
    records = load_data(repo_dir, cache_path=cache_path)
    print(f"Loaded {len(records)} evaluation items.")

    verify_numbers(records, repo_dir)
    plot_figure(records, out_pdf, out_png)


if __name__ == "__main__":
    main()
