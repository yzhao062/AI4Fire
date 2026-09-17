#!/usr/bin/env python
"""Generate the fire danger calibration figure (figures/calibration.pdf and .png).

This figure displays reliability diagrams for the six benchmark models on the 386 fire-danger
items of the Mesogeos Track A holdout (2021-2022, fold 0, base rate 0.339), comparing bare
and grounded stated probabilities against observed fire frequency.

Panels (2 rows by 3 columns, standard model order):
  - Row 1:
      (a) claude-opus-4.8
      (b) claude-opus-5
      (c) gemini-3.1-pro
  - Row 2:
      (d) gpt-6-astra
      (e) Qwen3-VL
      (f) Llama 4 Maverick

Each panel displays:
  - Diagonal in light gray (dashed, y = x, ideal calibration).
  - Bare curve in gray (fs.GRAY) through non-empty bins of ten equal-width intervals [0, 1].
  - Grounded curve in coral (fs.CORAL) through non-empty bins.
  - Marker area proportional to bin count (empty bins omitted).
  - Calendar-month prior reliability curve in mint (fs.MINT / fs.MINT_EDGE) repeated on every
    panel behind model curves as the common reference (ECE = 0.037).
  - Expected Calibration Error (ECE) printed in the corner in each curve's color:
    bare (gray), grounded (coral), and the prior's ECE printed once in panel (a) (mint).

Data sources in the AI4Fire repository:
  - task-mesogeos/items.jsonl: 386 holdout items (split == 'test', fold == 0, base rate 0.339).
  - task-mesogeos/responses-<stem>-bare.jsonl: stated probability and binary calls.
  - task-mesogeos/responses-<stem>-grounded.jsonl: stated probability and binary calls.
  - analysis/calibration_mesogeos.py: reusable calibration and binning functions
    (load_items, compute_calibration_and_murphy, TRAIN_PRIOR_BY_MONTH, TRAIN_PRIOR_MEAN).
  - Precomputed reference summary: analysis/calibration_mesogeos.json.

Styling conventions (CatchBench submission style):
  - Width: 6.5 in (fs.TEXT_WIDTH_IN), height <= 3.0 in (2.85 in).
  - Typography: Sans-serif (DejaVu Sans / Arial), TrueType fonts (pdf.fonttype 42), >= 6 pt.
  - Spines: no top/right spines, gray left/bottom spines, no tick marks.
  - Palette: Coral (#ED8D5A) grounded focal result, Gray (#999999) bare context,
    Mint (#BFDFD2 / #8FB7A6) comparison prior, Near-black (#1A1A1A) text.
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

# Ensure figures/ and analysis/ are on sys.path
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
if str(_REPO_ROOT / "analysis") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "analysis"))

import figstyle as fs
import calibration_mesogeos as cm

# Color constants
COLOR_PRIOR_TEXT = "#3D735E"  # Dark mint for high-contrast legible text on white
COLOR_GRAY_TEXT = "#666666"   # Contrast-enhanced gray for text labels

# Paper reference values from Table tab:calibration and Section 7.3 for verification
PAPER_CALIBRATION = {
    "base_rate": 0.339,
    "prior": {"mean_p": 0.333, "ece": 0.037},
    "claude-opus-4.8": {"bare": {"mean_p": 0.187, "ece": 0.154}, "grounded": {"mean_p": 0.220, "ece": 0.119}},
    "claude-opus-5":   {"bare": {"mean_p": 0.117, "ece": 0.222}, "grounded": {"mean_p": 0.116, "ece": 0.224}},
    "gemini-3.1-pro":  {"bare": {"mean_p": 0.195, "ece": 0.169}, "grounded": {"mean_p": 0.192, "ece": 0.185}},
    "gpt-6-astra":     {"bare": {"mean_p": 0.060, "ece": 0.279}, "grounded": {"mean_p": 0.106, "ece": 0.233}},
    "Qwen3-VL":        {"bare": {"mean_p": 0.598, "ece": 0.326}, "grounded": {"mean_p": 0.609, "ece": 0.326}},
    "Llama 4 Maverick":{"bare": {"mean_p": 0.200, "ece": 0.218}, "grounded": {"mean_p": 0.236, "ece": 0.226}},
}


def load_calibration_data(repo_root: Path) -> dict:
    """Compute calibration data directly from the record items and model responses."""
    items = cm.load_items()
    assert len(items) == 386, f"Expected 386 test items, found {len(items)}"
    y_test = np.array([it["label"] for it in items], dtype=int)
    base_rate = float(np.mean(y_test))

    task_dir = repo_root / "task-mesogeos"

    # 1. Calendar-month prior
    m_ends = np.array([int(it["context"]["window_end"][5:7]) for it in items])
    prior_probs = np.array([cm.TRAIN_PRIOR_BY_MONTH.get(m, cm.TRAIN_PRIOR_MEAN) for m in m_ends])
    prior_cal = cm.compute_calibration_and_murphy(y_test, prior_probs, n_bins=10)

    # 2. Six models, bare and grounded
    models_data = []
    for m in fs.MODELS:
        stem = m["stem"]
        label = m["label"]
        model_entry = {"label": label, "stem": stem}
        for cond in ("bare", "grounded"):
            resp_path = task_dir / f"responses-{stem}-{cond}.jsonl"
            rows = [json.loads(line) for line in resp_path.read_text(encoding="utf-8").splitlines()]
            y_prob = np.array([float(r["probability"]) for r in rows])
            y_true = np.array([int(r["label"]) for r in rows])
            assert np.array_equal(y_true, y_test), f"Label mismatch in {resp_path}"
            cal = cm.compute_calibration_and_murphy(y_true, y_prob, n_bins=10)
            model_entry[cond] = cal
        models_data.append(model_entry)

    return {
        "base_rate": base_rate,
        "n_items": len(items),
        "prior": prior_cal,
        "models": models_data,
    }


def print_verification_table(cal_data: dict) -> None:
    """Print complete verification table comparing computed values to Table tab:calibration."""
    print("=" * 110)
    print("VERIFICATION OF CALIBRATION METRICS AGAINST PAPER TABLE tab:calibration & SEC 7.3")
    print("=" * 110)
    base_rec = cal_data["base_rate"]
    base_pap = PAPER_CALIBRATION["base_rate"]
    print(f"Dataset Base Rate: Record = {base_rec:.3f}, Paper = {base_pap:.3f}, Match = {round(base_rec, 3) == base_pap}")
    print("-" * 110)

    prior_rec = cal_data["prior"]
    p_mean_rec, p_ece_rec = prior_rec["mean_predicted"], prior_rec["ece_equal_width"]
    p_mean_pap, p_ece_pap = PAPER_CALIBRATION["prior"]["mean_p"], PAPER_CALIBRATION["prior"]["ece"]
    m_mean = round(p_mean_rec, 3) == p_mean_pap
    m_ece = round(p_ece_rec, 3) == p_ece_pap
    print(f"Reference: Calendar-Month Prior")
    print(f"  Mean p: Record = {p_mean_rec:.3f}, Paper = {p_mean_pap:.3f} (Match: {m_mean})")
    print(f"  ECE:    Record = {p_ece_rec:.3f}, Paper = {p_ece_pap:.3f} (Match: {m_ece})")
    print("-" * 110)

    print(f"{'Model':<18} {'Arm':<10} {'Mean p (rec)':>12} {'Mean p (pap)':>12} {'Match':>7} | {'ECE (rec)':>10} {'ECE (pap)':>10} {'Match':>7}")
    print("-" * 110)

    all_matched = m_mean and m_ece
    for m in cal_data["models"]:
        label = m["label"]
        pap_m = PAPER_CALIBRATION[label]
        for cond in ("bare", "grounded"):
            rec_cal = m[cond]
            rec_mean = rec_cal["mean_predicted"]
            rec_ece = rec_cal["ece_equal_width"]
            pap_mean = pap_m[cond]["mean_p"]
            pap_ece = pap_m[cond]["ece"]
            ok_mean = round(rec_mean, 3) == pap_mean
            ok_ece = round(rec_ece, 3) == pap_ece
            if not (ok_mean and ok_ece):
                all_matched = False
            print(f"{label:<18} {cond:<10} {rec_mean:>12.3f} {pap_mean:>12.3f} {str(ok_mean):>7} | {rec_ece:>10.3f} {pap_ece:>10.3f} {str(ok_ece):>7}")

    print("=" * 110)
    print(f"All values matched paper specifications exactly: {all_matched}")
    print("=" * 110)


def plot_calibration(cal_data: dict, out_pdf: Path, out_png: Path) -> None:
    """Draw the 6-panel CatchBench reliability diagram at 6.5 x 2.85 in."""
    fs.apply()

    fig, axes = plt.subplots(
        2, 3,
        figsize=(fs.TEXT_WIDTH_IN, 2.55),
        sharex=True,
        sharey=True,
        gridspec_kw={
            "left": 0.08,
            "right": 0.98,
            "bottom": 0.13,
            "top": 0.89,
            "wspace": 0.22,
            "hspace": 0.36,
        },
    )

    prior_bins = [b for b in cal_data["prior"]["reliability_table_equal_width"] if b["count"] > 0]
    prior_x = [b["mean_pred"] for b in prior_bins]
    prior_y = [b["obs_rate"] for b in prior_bins]
    prior_cnt = [b["count"] for b in prior_bins]
    prior_ece = cal_data["prior"]["ece_equal_width"]

    tags = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]

    for idx, m in enumerate(cal_data["models"]):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        label = m["label"]

        fs.bare(ax, grid=None)
        fs.panel_title(ax, tags[idx], label, x=0.0, y=1.04)

        # 1. Diagonal line of perfect calibration (y = x)
        ax.plot([0, 1], [0, 1], linestyle="--", color=fs.LIGHT_GRAY, linewidth=0.8, zorder=1)

        # 2. Calendar-month prior reliability curve (drawn behind model curves)
        ax.plot(prior_x, prior_y, color=fs.MINT_EDGE, linewidth=1.1, zorder=2, alpha=0.9)
        ax.scatter(
            prior_x, prior_y,
            s=[max(c * 0.35, 3.0) for c in prior_cnt],
            facecolor=fs.MINT, edgecolor=fs.MINT_EDGE, linewidth=0.6, zorder=2
        )

        # 3. Bare curve (gray)
        bare_data = m["bare"]
        bare_bins = [b for b in bare_data["reliability_table_equal_width"] if b["count"] > 0]
        bx = [b["mean_pred"] for b in bare_bins]
        by = [b["obs_rate"] for b in bare_bins]
        bc = [b["count"] for b in bare_bins]
        ax.plot(bx, by, color=fs.GRAY, linewidth=1.2, zorder=3)
        ax.scatter(
            bx, by,
            s=[max(c * 0.35, 3.0) for c in bc],
            facecolor=fs.GRAY, edgecolor="#666666", linewidth=0.5, zorder=3
        )

        # 4. Grounded curve (coral)
        grd_data = m["grounded"]
        grd_bins = [b for b in grd_data["reliability_table_equal_width"] if b["count"] > 0]
        gx = [b["mean_pred"] for b in grd_bins]
        gy = [b["obs_rate"] for b in grd_bins]
        gc = [b["count"] for b in grd_bins]
        ax.plot(gx, gy, color=fs.CORAL, linewidth=1.3, zorder=4)
        ax.scatter(
            gx, gy,
            s=[max(c * 0.35, 3.0) for c in gc],
            facecolor=fs.CORAL, edgecolor="#D46B36", linewidth=0.5, zorder=4
        )

        # 5. ECE text in curve's color
        b_ece = bare_data["ece_equal_width"]
        g_ece = grd_data["ece_equal_width"]

        # Corner placement: Qwen3-VL has points concentrated at high probabilities (x > 0.6, y < 0.53),
        # so top-left is open; other models are compressed toward zero, so bottom-right is open.
        if label == "Qwen3-VL":
            tx, ty = 0.05, 0.95
            lines = [
                ("ECE", fs.SUBTITLE),
                (f"bare {b_ece:.3f}", COLOR_GRAY_TEXT),
                (f"grd. {g_ece:.3f}", fs.CORAL),
            ]
            for line_i, (t_str, col_t) in enumerate(lines):
                ax.text(
                    tx, ty - line_i * 0.11, t_str, transform=ax.transAxes,
                    ha="left", va="top", fontsize=fs.FS_SMALL, fontweight="bold", color=col_t, zorder=5
                )
        else:
            tx, ty = 0.97, 0.05
            lines = [
                ("ECE", fs.SUBTITLE),
                (f"bare {b_ece:.3f}", COLOR_GRAY_TEXT),
                (f"grd. {g_ece:.3f}", fs.CORAL),
            ]
            if idx == 0:
                lines.append((f"prior {prior_ece:.3f}", COLOR_PRIOR_TEXT))

            # Bottom-up rendering
            for line_i, (t_str, col_t) in enumerate(reversed(lines)):
                ax.text(
                    tx, ty + line_i * 0.11, t_str, transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=fs.FS_SMALL, fontweight="bold", color=col_t, zorder=5
                )

        # Coordinate bounds and ticks
        ax.set_xlim(-0.07, 1.03)
        ax.set_ylim(-0.07, 1.03)
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.set_yticks([0.0, 0.5, 1.0])

    # Shared axis labels
    for ax in axes[1, :]:
        ax.set_xlabel("Mean stated probability", fontsize=fs.FS_AXIS, labelpad=2)
    for ax in axes[:, 0]:
        ax.set_ylabel("Observed frequency", fontsize=fs.FS_AXIS, labelpad=2)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fs.savefig(fig, out_pdf, out_png, dpi=300)
    plt.close(fig)
    print(f"Generated {out_pdf} and {out_png}")


def main():
    parser = argparse.ArgumentParser(description="Generate fire danger calibration figure.")
    parser.add_argument("--repo", type=str, default=str(_REPO_ROOT),
                        help="Path to AI4Fire repository root.")
    parser.add_argument("--out-pdf", type=str, default=None,
                        help="Path for output PDF file.")
    parser.add_argument("--out-png", type=str, default=None,
                        help="Path for output PNG file.")
    args = parser.parse_args()

    repo_root = Path(args.repo).resolve()
    out_pdf = Path(args.out_pdf).resolve() if args.out_pdf else repo_root / "figures" / "calibration.pdf"
    out_png = Path(args.out_png).resolve() if args.out_png else repo_root / "figures" / "calibration.png"

    cal_data = load_calibration_data(repo_root)
    print_verification_table(cal_data)
    plot_calibration(cal_data, out_pdf, out_png)


if __name__ == "__main__":
    main()
