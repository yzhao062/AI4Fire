#!/usr/bin/env python
"""Generate the FIgLib detection timeline figure (figures/figlib_timeline.pdf and .png).

This figure displays how smoke detection accuracy depends on time since the first visible plume,
and where grounding's gain lands:
  - Panel (a): Detection accuracy per offset bucket (before the plume, 0 to 10 min, 10 to 25 min,
               and 25 min or more) for each of the six models, drawn as connected lines under bare
               (gray) and grounded (coral) conditions on the 196 paired items per model.
  - Panel (b): Paired grounded-minus-bare gain per bucket, plotted as dots for each model and
               horizontal bars for the pooled gain across models, with the pooled gain and the
               number of correct decisions it adds printed above each bucket. The three post-plume
               windows gain 13, 26, and 17 correct decisions pooled over six models (out of 168,
               282, and 222 model-frame pairs), accuracy lifts of 0.077, 0.092, and 0.077, so the
               first ten minutes stay the weakest window under grounding too.

Why a single 2-panel layout rather than six small multiples in panel (a):
  A 2-panel layout across the paper's 6.5 in text width gives each panel ~3.1 in of width and
  2.55 in of height, allowing type sizes to remain comfortably at or above the 6.2 pt floor.
  Overlaying the six models in panel (a) immediately conveys their tight envelope: high accuracy
  (0.94 to 1.00) before the plume, a universal plunge to 0.25-0.57 in the first 10 minutes, and
  a universal rebound where grounded is systematically above bare. Six separate subplots would
  crowd axes, shrink labels below the type floor, and fragment this shared visual envelope.

Model order everywhere:
  1. claude-opus-4.8
  2. claude-opus-5
  3. gemini-3.1-pro
  4. gpt-6-astra
  5. Qwen3-VL
  6. Llama 4 Maverick

Data sources in the AI4Fire repository:
  - analysis/figlib_paired.py and analysis/figlib-paired.json:
    supplies the 196 paired items per model, bucket definitions, bucket-level accuracy,
    and recall / false-positive rates.
  - task-figlib/responses-<stem>-bare.jsonl and responses-<stem>-grounded.jsonl:
    stored model responses for the six models.
  - task-figlib/items.jsonl:
    task items with labels, offset_seconds, and sequences.

Styling conventions (CatchBench submission style):
  - Width: 6.5 in (TEXT_WIDTH_IN), height <= 2.8 in.
  - Typography: Sans-serif (DejaVu Sans / Arial), TrueType fonts (pdf.fonttype 42), >= 6.2 pt.
  - Spines: no top/right spines, gray (#999999) left/bottom spines, no tick marks.
  - Palette: Coral (#ED8D5A) for focal result (grounded), Gray (#999999) for bare/context,
             Near-black (#1A1A1A) for text and pooled summaries.
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
if str(FIGURES_DIR) not in sys.path:
    sys.path.insert(0, str(FIGURES_DIR))
import figstyle as fs

BUCKET_KEYS = ["before", "after 0 to 10 min", "after 10 to 25 min", "after 25 min or more"]
BUCKET_LABELS = ["Before plume", "0 to 10 min", "10 to 25 min", "≥25 min"]
BUCKET_NS = [84, 28, 47, 37]  # 84 + 28 + 47 + 37 = 196 frames


def load_data(repo_root: Path) -> dict:
    """Load precomputed paired data from analysis/figlib-paired.json."""
    paired_path = repo_root / "analysis" / "figlib-paired.json"
    with open(paired_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def compute_metrics(data: dict) -> dict:
    """Extract and compute per-model and pooled metrics for each bucket."""
    rows_by_mc = {(r["model"], r["condition"]): r for r in data["rows"]}
    models = fs.MODELS

    acc_bare = {m["label"]: [] for m in models}
    acc_grd = {m["label"]: [] for m in models}
    gain_acc = {m["label"]: [] for m in models}
    gain_frames = {m["label"]: [] for m in models}

    for m in models:
        stem = m["stem"]
        r_b = rows_by_mc[(stem, "bare")]
        r_g = rows_by_mc[(stem, "grounded")]
        for b_key, n_f in zip(BUCKET_KEYS, BUCKET_NS):
            if b_key == "before":
                ab = 1.0 - r_b["fpr"]
                ag = 1.0 - r_g["fpr"]
            else:
                ab = r_b["bucket_detail"][b_key]["accuracy"]
                ag = r_g["bucket_detail"][b_key]["accuracy"]
            acc_bare[m["label"]].append(ab)
            acc_grd[m["label"]].append(ag)
            gain_acc[m["label"]].append(ag - ab)
            gain_frames[m["label"]].append(round(ag * n_f) - round(ab * n_f))

    # Pooled metrics across all 6 models per bucket
    pooled_bare = []
    pooled_grd = []
    pooled_gain = []
    pooled_frame_gain = []

    for j, n_f in enumerate(BUCKET_NS):
        cb = sum(round(acc_bare[m["label"]][j] * n_f) for m in models)
        cg = sum(round(acc_grd[m["label"]][j] * n_f) for m in models)
        tot = n_f * len(models)
        pb = cb / tot
        pg = cg / tot
        pooled_bare.append(pb)
        pooled_grd.append(pg)
        pooled_gain.append(pg - pb)
        pooled_frame_gain.append(cg - cb)

    return {
        "models": models,
        "rows_by_mc": rows_by_mc,
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
    """Print complete row-by-row verification against the paper's tables and prose."""
    print("=" * 100)
    print("VERIFICATION: REPRODUCING NUMBERS FROM PAPER TABLES AND SECTION 7.2 PROSE")
    print("=" * 100)

    # 1. Table tab:figlib-buckets check
    paper_buckets = {
        ("claude-opus-4.8", "bare"): (0.43, 0.57, 0.62),
        ("claude-opus-4.8", "grounded"): (0.46, 0.66, 0.68),
        ("claude-opus-5", "bare"): (0.50, 0.66, 0.62),
        ("claude-opus-5", "grounded"): (0.50, 0.72, 0.65),
        ("gemini-3.1-pro", "bare"): (0.25, 0.64, 0.65),
        ("gemini-3.1-pro", "grounded"): (0.50, 0.81, 0.81),
        ("gpt-6-astra", "bare"): (0.46, 0.70, 0.70),
        ("gpt-6-astra", "grounded"): (0.57, 0.83, 0.76),
        ("Qwen3-VL", "bare"): (0.36, 0.57, 0.65),
        ("Qwen3-VL", "grounded"): (0.43, 0.64, 0.73),
        ("Llama 4 Maverick", "bare"): (0.36, 0.60, 0.51),
        ("Llama 4 Maverick", "grounded"): (0.36, 0.64, 0.59),
    }

    print("\n1. CHECK AGAINST TABLE tab:figlib-buckets (Accuracy across 3 post-plume buckets)")
    print(f"{'Model':<18} {'Cond':<9} {'0-10m [n=28]':<22} {'10-25m [n=47]':<22} {'≥25m [n=37]':<22}")
    print("-" * 100)
    all_bucket_match = True
    for m in metrics["models"]:
        lbl = m["label"]
        stem = m["stem"]
        for c in ["bare", "grounded"]:
            r = metrics["rows_by_mc"][(stem, c)]
            b0_10 = r["bucket_detail"]["after 0 to 10 min"]["accuracy"]
            b10_25 = r["bucket_detail"]["after 10 to 25 min"]["accuracy"]
            b25_plus = r["bucket_detail"]["after 25 min or more"]["accuracy"]
            p0, p1, p2 = paper_buckets[(lbl, c)]
            m0 = round(b0_10, 2) == p0
            m1 = round(b10_25, 2) == p1
            m2 = round(b25_plus, 2) == p2
            if not (m0 and m1 and m2):
                all_bucket_match = False
            s0 = f"{b0_10:.2f} ({p0:.2f}) {'OK' if m0 else 'FAIL'}"
            s1 = f"{b10_25:.2f} ({p1:.2f}) {'OK' if m1 else 'FAIL'}"
            s2 = f"{b25_plus:.2f} ({p2:.2f}) {'OK' if m2 else 'FAIL'}"
            print(f"{lbl:<18} {c:<9} {s0:<22} {s1:<22} {s2:<22}")
    print(f"All Table tab:figlib-buckets cells matched exactly: {all_bucket_match}")

    # 2. Table tab:results-figlib recall check
    paper_recall = {
        ("claude-opus-4.8", "bare"): 0.554,
        ("claude-opus-4.8", "grounded"): 0.616,
        ("claude-opus-5", "bare"): 0.607,
        ("claude-opus-5", "grounded"): 0.643,
        ("gemini-3.1-pro", "bare"): 0.545,
        ("gemini-3.1-pro", "grounded"): 0.732,
        ("gpt-6-astra", "bare"): 0.643,
        ("gpt-6-astra", "grounded"): 0.741,
        ("Qwen3-VL", "bare"): 0.545,
        ("Qwen3-VL", "grounded"): 0.616,
        ("Llama 4 Maverick", "bare"): 0.509,
        ("Llama 4 Maverick", "grounded"): 0.554,
    }

    print("\n2. CHECK AGAINST TABLE tab:results-figlib (Recall on smoke, 196 paired items)")
    print(f"{'Model':<18} {'Cond':<9} {'Recall Calc (Paper)':<25}")
    print("-" * 100)
    all_recall_match = True
    for m in metrics["models"]:
        lbl = m["label"]
        stem = m["stem"]
        for c in ["bare", "grounded"]:
            rec = metrics["rows_by_mc"][(stem, c)]["recall_smoke"]
            p_rec = paper_recall[(lbl, c)]
            m_ok = rec == p_rec
            if not m_ok:
                all_recall_match = False
            print(f"{lbl:<18} {c:<9} {rec:.3f} ({p_rec:.3f}) {'OK' if m_ok else 'FAIL'}")
    print(f"All Table tab:results-figlib recall values matched exactly: {all_recall_match}")

    # 3. Before-plume accuracy range check (paper: 0.94 to 1.00)
    print("\n3. CHECK BEFORE-PLUME ACCURACY (Section 7.2 prose: '0.94 to 1.00 throughout')")
    print(f"{'Model':<18} {'Bare':<12} {'Grounded':<12}")
    print("-" * 100)
    all_before = []
    for m in metrics["models"]:
        lbl = m["label"]
        ab = metrics["acc_bare"][lbl][0]
        ag = metrics["acc_grd"][lbl][0]
        all_before.extend([ab, ag])
        print(f"{lbl:<18} {ab:.3f}        {ag:.3f}")
    min_bef, max_bef = min(all_before), max(all_before)
    range_ok = round(min_bef, 2) == 0.94 and round(max_bef, 2) == 1.00
    print(f"Observed before-plume range: {min_bef:.3f} to {max_bef:.3f} (rounded: {min_bef:.2f} to {max_bef:.2f})")
    print(f"Matches paper statement '0.94 to 1.00': {range_ok}")

    # 4. First ten minutes dip and gain check
    # Paper: "every configuration sits between 0.25 and 0.57 on 28 frames, where grounding gains zero to seven frames per model"
    print("\n4. CHECK 0 TO 10 MIN DIP & GAIN (Section 7.2 prose: 0.25 to 0.57, gains 0 to 7 frames)")
    print(f"{'Model':<18} {'Bare':<10} {'Grounded':<10} {'Gain (pp)':<14} {'Gain (frames on 28)':<20}")
    print("-" * 100)
    all_dip = []
    all_f_gain_0_10 = []
    for m in metrics["models"]:
        lbl = m["label"]
        ab = metrics["acc_bare"][lbl][1]
        ag = metrics["acc_grd"][lbl][1]
        g_pp = metrics["gain_acc"][lbl][1] * 100
        g_f = metrics["gain_frames"][lbl][1]
        all_dip.extend([ab, ag])
        all_f_gain_0_10.append(g_f)
        print(f"{lbl:<18} {ab:.3f}      {ag:.3f}      {g_pp:+.1f} pp        {g_f:+d} frames")
    dip_ok = round(min(all_dip), 2) == 0.25 and round(max(all_dip), 2) == 0.57
    gain_0_10_ok = min(all_f_gain_0_10) == 0 and max(all_f_gain_0_10) == 7
    print(f"Observed 0-10m range: {min(all_dip):.2f} to {max(all_dip):.2f} -> matches '0.25 and 0.57': {dip_ok}")
    print(f"Observed 0-10m frame gains: [{min(all_f_gain_0_10)}, {max(all_f_gain_0_10)}] -> matches 'zero to seven frames': {gain_0_10_ok}")

    # 5. Post-plume window gains: the per-model 10 to 25 min points (Table tab:figlib-buckets) and the pooled
    #    correct decisions of the corrected Section 7.2 sentence (13 of 168, 26 of 282, 17 of 222)
    print("\n5. CHECK POST-PLUME WINDOW GAINS (Table tab:figlib-buckets 10 to 25 min column; Section 7.2 pooled counts)")
    print(f"{'Model':<18} {'Bare':<10} {'Grounded':<10} {'Gain (points)':<16} {'Gain (frames on 47)':<20}")
    print("-" * 100)
    all_pts_10_25 = []
    for m in metrics["models"]:
        lbl = m["label"]
        ab = metrics["acc_bare"][lbl][2]
        ag = metrics["acc_grd"][lbl][2]
        # Points in paper table: grounded - bare rounded to 2 decimals
        pts_table = round(ag, 2) - round(ab, 2)
        pts_exact = (ag - ab) * 100
        gf = metrics["gain_frames"][lbl][2]
        all_pts_10_25.append(round(pts_table * 100))
        print(f"{lbl:<18} {ab:.2f}      {ag:.2f}      {pts_table*100:+.0f} ({pts_exact:+.1f}) pp      {gf:+d} frames")
    gain_10_25_ok = min(all_pts_10_25) == 4 and max(all_pts_10_25) == 17
    print(f"Observed 10-25m table point gains: [{min(all_pts_10_25)}, {max(all_pts_10_25)}] -> range 4 to 17 points: {gain_10_25_ok}")
    pooled_expected = [13, 26, 17]
    pooled_obs = metrics["pooled_frame_gain"][1:]
    pooled_denoms = [n * len(metrics["models"]) for n in BUCKET_NS[1:]]
    pooled_ok = pooled_obs == pooled_expected and pooled_denoms == [168, 282, 222]
    lifts = [g / d for g, d in zip(pooled_obs, pooled_denoms)]
    print(f"Pooled correct decisions added per post-plume window: {pooled_obs} of {pooled_denoms} "
          f"(Section 7.2: 13 of 168, 26 of 282, 17 of 222) -> {pooled_ok}")
    print(f"Pooled accuracy lifts: {', '.join(f'{v:.3f}' for v in lifts)} (Section 7.2: 0.08 to 0.09) -> "
          f"{all(0.075 <= v <= 0.095 for v in lifts)}")

    # 6. Frame counts check
    print("\n6. CHECK FRAME COUNTS PER BUCKET")
    print(f"Before plume: {BUCKET_NS[0]} frames")
    print(f"Post-plume buckets: {BUCKET_NS[1:]} frames (Paper: 28, 47, 37)")
    print(f"Total paired items: {sum(BUCKET_NS)} frames (Paper: 196)")
    counts_ok = BUCKET_NS == [84, 28, 47, 37] and sum(BUCKET_NS) == 196
    print(f"Frame counts match paper: {counts_ok}")
    print("=" * 100)


def plot_timeline_figure(metrics: dict, out_pdf: Path, out_png: Path) -> None:
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

    # -------------------------------------------------------------------------
    # Panel (a): Detection accuracy per offset bucket (6 models)
    # -------------------------------------------------------------------------
    for m in models:
        lbl = m["label"]
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
        [f"{name}\n({n} frames)" for name, n in zip(BUCKET_LABELS, BUCKET_NS)],
        fontsize=fs.FS_TICK,
    )
    ax_a.set_ylabel("Detection accuracy", fontsize=fs.FS_AXIS)
    fs.panel_title(ax_a, "(a)", "Accuracy by time since plume (6 models)")

    # Clean legend for Grounded vs Bare in lower left
    ax_a.plot([], [], color=fs.CORAL, linewidth=1.4, marker="o", markersize=3.5, label="Grounded")
    ax_a.plot([], [], color=fs.GRAY, linewidth=1.4, marker="o", markersize=3.5, label="Bare")
    ax_a.legend(
        loc="lower left", frameon=False, fontsize=fs.FS_SMALL + 0.3,
        handlelength=1.2, handletextpad=0.4, borderaxespad=0.6,
    )

    # -------------------------------------------------------------------------
    # Panel (b): Grounded minus bare gain per bucket
    # -------------------------------------------------------------------------
    ax_b.axhline(0, color=fs.GRAY, linestyle="--", linewidth=0.75, zorder=1)

    # Horizontal spread of 6 dots per bucket so individual model points are distinct
    offsets = np.linspace(-0.14, 0.14, len(models))

    for i, m in enumerate(models):
        lbl = m["label"]
        ax_b.plot(
            x + offsets[i], metrics["gain_acc"][lbl],
            color=fs.CORAL, marker="o",
            markersize=4.6, markeredgecolor="white", markeredgewidth=0.6,
            linestyle="None", zorder=3,
        )

    # Bold pooled gain bars across all 6 models
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
        [f"{name}\n({n} frames)" for name, n in zip(BUCKET_LABELS, BUCKET_NS)],
        fontsize=fs.FS_TICK,
    )
    ax_b.set_ylabel("Grounded − bare gain (accuracy)", fontsize=fs.FS_AXIS)
    fs.panel_title(ax_b, "(b)", "Grounded minus bare, per bucket")

    # Direct legend in Panel (b) in the open space above bucket 0
    ax_b.plot([-0.32, -0.08], [0.17, 0.17], color=fs.NEAR_BLACK, linewidth=2.0, zorder=5)
    ax_b.text(-0.02, 0.17, "Pooled over models", fontsize=fs.FS_SMALL + 0.2, va="center", color=fs.NEAR_BLACK)
    ax_b.plot(-0.20, 0.12, color=fs.CORAL, marker="o", markersize=4.6, markeredgecolor="white", markeredgewidth=0.6, zorder=5)
    ax_b.text(-0.02, 0.12, "One model", fontsize=fs.FS_SMALL + 0.2, va="center", color=fs.NEAR_BLACK)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fs.savefig(fig, out_pdf, out_png, dpi=300)
    plt.close(fig)
    print(f"Generated {out_pdf} and {out_png}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        type=str,
        default=str(FIGURES_DIR.parent),
        help="Path to AI4Fire repository root.",
    )
    parser.add_argument(
        "--out-pdf",
        type=str,
        default=None,
        help="Path for output PDF file.",
    )
    parser.add_argument(
        "--out-png",
        type=str,
        default=None,
        help="Path for output PNG file.",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo).resolve()
    out_pdf = Path(args.out_pdf).resolve() if args.out_pdf else repo_root / "figures" / "figlib_timeline.pdf"
    out_png = Path(args.out_png).resolve() if args.out_png else repo_root / "figures" / "figlib_timeline.png"

    data = load_data(repo_root)
    metrics = compute_metrics(data)
    print_verification(metrics)
    plot_timeline_figure(metrics, out_pdf, out_png)


if __name__ == "__main__":
    main()
