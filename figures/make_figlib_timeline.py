#!/usr/bin/env python
"""Generate the FIgLib detection timeline figure (figures/figlib_timeline.pdf and .png).

This figure displays how smoke detection accuracy depends on time since the first visible plume,
and where grounding's gain lands:
  - Panel (a): Detection accuracy per offset bucket (before the plume, 0 to 10 min, 10 to 25 min,
               and 25 min or more) for each model, drawn as connected lines under bare
               (gray) and grounded (coral) conditions on the paired items per model.
  - Panel (b): Paired grounded-minus-bare gain per bucket, plotted as dots for each model and
               horizontal bars for the pooled gain across models, with the pooled gain and the
               number of correct decisions it adds printed above each bucket.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Import shared figure style
FIGURES_DIR = Path(__file__).resolve().parent
REPO_ROOT = FIGURES_DIR.parent
if str(FIGURES_DIR) not in sys.path:
    sys.path.insert(0, str(FIGURES_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import figstyle as fs
import models
from models import add_model_args, resolve_models

BUCKET_KEYS = ["before", "after 0 to 10 min", "after 10 to 25 min", "after 25 min or more"]
BUCKET_LABELS = ["Before plume", "0 to 10 min", "10 to 25 min", "≥25 min"]


def load_data(repo_root: Path) -> dict:
    """Load precomputed paired data from analysis/figlib-paired.json."""
    paired_path = repo_root / "analysis" / "figlib-paired.json"
    with open(paired_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def get_bucket_ns(data: dict) -> list[int]:
    """Extract bucket sizes from data rows."""
    if not data.get("rows"):
        return [84, 28, 47, 37]
    r0 = data["rows"][0]
    return [
        r0.get("fpr_negatives", 84),
        r0.get("bucket_detail", {}).get("after 0 to 10 min", {}).get("n", 28),
        r0.get("bucket_detail", {}).get("after 10 to 25 min", {}).get("n", 47),
        r0.get("bucket_detail", {}).get("after 25 min or more", {}).get("n", 37),
    ]


def compute_metrics(data: dict, selected_models) -> dict:
    """Extract and compute per-model and pooled metrics for each bucket."""
    rows_by_mc = {(r["model"], r["condition"]): r for r in data["rows"]}
    valid_models = [m for m in selected_models if (m.stem, "bare") in rows_by_mc and (m.stem, "grounded") in rows_by_mc]
    bucket_ns = get_bucket_ns(data)

    acc_bare = {m.label: [] for m in valid_models}
    acc_grd = {m.label: [] for m in valid_models}
    gain_acc = {m.label: [] for m in valid_models}
    gain_frames = {m.label: [] for m in valid_models}

    for m in valid_models:
        stem = m.stem
        r_b = rows_by_mc[(stem, "bare")]
        r_g = rows_by_mc[(stem, "grounded")]
        for b_key, n_f in zip(BUCKET_KEYS, bucket_ns):
            if b_key == "before":
                ab = 1.0 - r_b["fpr"]
                ag = 1.0 - r_g["fpr"]
            else:
                ab = r_b["bucket_detail"][b_key]["accuracy"]
                ag = r_g["bucket_detail"][b_key]["accuracy"]
            acc_bare[m.label].append(ab)
            acc_grd[m.label].append(ag)
            gain_acc[m.label].append(ag - ab)
            gain_frames[m.label].append(round(ag * n_f) - round(ab * n_f))

    pooled_bare = []
    pooled_grd = []
    pooled_gain = []
    pooled_frame_gain = []

    for j, n_f in enumerate(bucket_ns):
        cb = sum(round(acc_bare[m.label][j] * n_f) for m in valid_models)
        cg = sum(round(acc_grd[m.label][j] * n_f) for m in valid_models)
        tot = n_f * len(valid_models) if valid_models else 1
        pb = cb / tot
        pg = cg / tot
        pooled_bare.append(pb)
        pooled_grd.append(pg)
        pooled_gain.append(pg - pb)
        pooled_frame_gain.append(cg - cb)

    return {
        "models": valid_models,
        "rows_by_mc": rows_by_mc,
        "bucket_ns": bucket_ns,
        "acc_bare": acc_bare,
        "acc_grd": acc_grd,
        "gain_acc": gain_acc,
        "gain_frames": gain_frames,
        "pooled_bare": pooled_bare,
        "pooled_grd": pooled_grd,
        "pooled_gain": pooled_gain,
        "pooled_frame_gain": pooled_frame_gain,
    }


def print_verification(metrics: dict) -> None:
    """Print complete row-by-row table of computed metrics."""
    bucket_ns = metrics["bucket_ns"]
    print("=" * 100)
    print("FIGLIB DETECTION TIMELINE: COMPUTED METRICS")
    print("=" * 100)

    print("\n1. ACCURACY ACROSS POST-PLUME BUCKETS")
    print(f"{'Model':<20} {'Cond':<9} {'0-10m [n=' + str(bucket_ns[1]) + ']':<18} {'10-25m [n=' + str(bucket_ns[2]) + ']':<18} {'≥25m [n=' + str(bucket_ns[3]) + ']':<18}")
    print("-" * 100)
    for m in metrics["models"]:
        lbl = m.label
        stem = m.stem
        for c in ["bare", "grounded"]:
            r = metrics["rows_by_mc"][(stem, c)]
            b0_10 = r["bucket_detail"]["after 0 to 10 min"]["accuracy"]
            b10_25 = r["bucket_detail"]["after 10 to 25 min"]["accuracy"]
            b25_plus = r["bucket_detail"]["after 25 min or more"]["accuracy"]
            print(f"{lbl:<20} {c:<9} {b0_10:<18.2f} {b10_25:<18.2f} {b25_plus:<18.2f}")

    print("\n2. RECALL ON SMOKE (PAIRED ITEMS)")
    print(f"{'Model':<20} {'Cond':<9} {'Recall':<12}")
    print("-" * 100)
    for m in metrics["models"]:
        lbl = m.label
        stem = m.stem
        for c in ["bare", "grounded"]:
            rec = metrics["rows_by_mc"][(stem, c)]["recall_smoke"]
            print(f"{lbl:<20} {c:<9} {rec:<12.3f}")

    print("\n3. BEFORE-PLUME ACCURACY (1 - FPR)")
    print(f"{'Model':<20} {'Bare':<12} {'Grounded':<12}")
    print("-" * 100)
    for m in metrics["models"]:
        lbl = m.label
        ab = metrics["acc_bare"][lbl][0]
        ag = metrics["acc_grd"][lbl][0]
        print(f"{lbl:<20} {ab:.3f}        {ag:.3f}")

    print("\n4. 0 TO 10 MIN DIP & GAIN")
    print(f"{'Model':<20} {'Bare':<10} {'Grounded':<10} {'Gain (pp)':<14} {'Gain (frames)':<16}")
    print("-" * 100)
    for m in metrics["models"]:
        lbl = m.label
        ab = metrics["acc_bare"][lbl][1]
        ag = metrics["acc_grd"][lbl][1]
        g_pp = metrics["gain_acc"][lbl][1] * 100
        g_f = metrics["gain_frames"][lbl][1]
        print(f"{lbl:<20} {ab:.3f}      {ag:.3f}      {g_pp:+.1f} pp        {g_f:+d} frames")

    print("\n5. POOLED GAINS PER BUCKET")
    pooled_obs = metrics["pooled_frame_gain"][1:]
    pooled_denoms = [n * len(metrics["models"]) for n in bucket_ns[1:]]
    print(f"Pooled correct decisions added per post-plume window: {pooled_obs} of {pooled_denoms}")
    lifts = [g / max(d, 1) for g, d in zip(pooled_obs, pooled_denoms)]
    print(f"Pooled accuracy lifts: {', '.join(f'{v:.3f}' for v in lifts)}")

    print("\n6. FRAME COUNTS PER BUCKET")
    print(f"Before plume: {bucket_ns[0]} frames | Post-plume: {bucket_ns[1:]} frames | Total: {sum(bucket_ns)} frames")
    print("=" * 100)


def plot_timeline_figure(metrics: dict, out_pdf: Path | None, out_png: Path) -> None:
    """Render the 2-panel FIgLib detection timeline figure at 6.5 x 2.6 in."""
    fs.apply()

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2,
        figsize=(fs.TEXT_WIDTH_IN, 2.6),
        gridspec_kw={"wspace": 0.28, "left": 0.08, "right": 0.98, "top": 0.86, "bottom": 0.17},
    )

    fs.bare(ax_a, grid="y")
    fs.bare(ax_b, grid="y")

    x = np.arange(4)
    models = metrics["models"]
    bucket_ns = metrics["bucket_ns"]

    # (a) Detection accuracy per offset bucket
    for m in models:
        lbl = m.label
        ax_a.plot(
            x, metrics["acc_bare"][lbl],
            color=fs.GRAY, linewidth=0.9, alpha=0.75,
            marker="o", markersize=3.2, zorder=2,
        )
        ax_a.plot(
            x, metrics["acc_grd"][lbl],
            color=fs.CORAL, linewidth=1.1, alpha=0.85,
            marker="o", markersize=3.2, zorder=3,
        )

    ax_a.set_ylim(0.18, 1.05)
    ax_a.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(
        [f"{name}\n({n} frames)" for name, n in zip(BUCKET_LABELS, bucket_ns)],
        fontsize=fs.FS_TICK,
    )
    ax_a.set_ylabel("Detection accuracy", fontsize=fs.FS_AXIS)
    fs.panel_title(ax_a, "(a)", f"Accuracy by time since plume ({len(models)} models)")

    # Legend for Grounded vs Bare in lower left
    ax_a.plot([], [], color=fs.CORAL, linewidth=1.4, marker="o", markersize=3.5, label="Grounded")
    ax_a.plot([], [], color=fs.GRAY, linewidth=1.4, marker="o", markersize=3.5, label="Bare")
    ax_a.legend(
        loc="lower left", frameon=False, fontsize=fs.FS_SMALL + 0.3,
        handlelength=1.2, handletextpad=0.4, borderaxespad=0.6,
    )

    # (b) Grounded minus bare gain per bucket
    ax_b.axhline(0, color=fs.GRAY, linestyle="--", linewidth=0.75, zorder=1)

    offsets = np.linspace(-0.14, 0.14, len(models)) if len(models) > 1 else [0.0]

    for i, m in enumerate(models):
        lbl = m.label
        ax_b.plot(
            x + offsets[i], metrics["gain_acc"][lbl],
            color=fs.CORAL, marker="o",
            markersize=4.6, markeredgecolor="white", markeredgewidth=0.6,
            linestyle="None", zorder=3,
        )

    # Bold pooled gain bars across models
    for j, pool_g in enumerate(metrics["pooled_gain"]):
        ax_b.plot(
            [x[j] - 0.20, x[j] + 0.20], [pool_g, pool_g],
            color=fs.NEAR_BLACK, linewidth=2.0, zorder=4,
        )
        frames = metrics["pooled_frame_gain"][j]
        ax_b.text(
            x[j], 0.285, f"{pool_g:+.2f}\n{frames:+d} decisions".replace("-", "\u2212"),
            fontsize=fs.FS_SMALL, color=fs.NEAR_BLACK, ha="center", va="bottom", zorder=5,
        )

    ax_b.set_ylim(-0.06, 0.345)
    ax_b.set_yticks([-0.05, 0.0, 0.05, 0.10, 0.15, 0.20, 0.25])
    ax_b.set_yticklabels(["−0.05", "0.00", "+0.05", "+0.10", "+0.15", "+0.20", "+0.25"])
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(
        [f"{name}\n({n} frames)" for name, n in zip(BUCKET_LABELS, bucket_ns)],
        fontsize=fs.FS_TICK,
    )
    ax_b.set_ylabel("Grounded − bare gain (accuracy)", fontsize=fs.FS_AXIS)
    fs.panel_title(ax_b, "(b)", "Grounded minus bare, per bucket")

    # Direct legend in Panel (b) in the open space above bucket 0
    ax_b.plot([-0.32, -0.08], [0.17, 0.17], color=fs.NEAR_BLACK, linewidth=2.0, zorder=5)
    ax_b.text(-0.02, 0.17, "Pooled over models", fontsize=fs.FS_SMALL + 0.2, va="center", color=fs.NEAR_BLACK)
    ax_b.plot(-0.20, 0.12, color=fs.CORAL, marker="o", markersize=4.6, markeredgecolor="white", markeredgewidth=0.6, zorder=5)
    ax_b.text(-0.02, 0.12, "One model", fontsize=fs.FS_SMALL + 0.2, va="center", color=fs.NEAR_BLACK)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    if out_pdf is not None:
        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        fs.savefig(fig, out_pdf, out_png, dpi=300)
        print(f"Generated {out_pdf} and {out_png}")
    else:
        fig.savefig(out_png, dpi=300)
        print(f"Generated {out_png}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        type=str,
        default=str(REPO_ROOT),
        help="Path to repository root.",
    )
    parser.add_argument(
        "--out-pdf",
        type=Path,
        default=None,
        help="Path for output PDF file.",
    )
    parser.add_argument(
        "--out-png",
        type=Path,
        default=None,
        help="Path for output PNG file.",
    )
    add_model_args(parser, default_tier="core")
    args = parser.parse_args()

    repo_root = Path(args.repo).resolve()
    figs_dir = repo_root / "figures"
    if args.out_pdf:
        out_pdf = args.out_pdf
    elif args.out_png:
        out_pdf = None
    else:
        out_pdf = figs_dir / "figlib_timeline.pdf"
    out_png = args.out_png or (figs_dir / "figlib_timeline.png")

    selected_models = resolve_models(args, task="figlib", default_tier="core")

    data = load_data(repo_root)
    metrics = compute_metrics(data, selected_models)
    print_verification(metrics)
    plot_timeline_figure(metrics, out_pdf, out_png)


if __name__ == "__main__":
    main()
