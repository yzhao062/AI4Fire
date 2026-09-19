#!/usr/bin/env python
"""Generate the fire danger calibration figure (figures/calibration.pdf and .png).

This figure displays reliability diagrams for benchmark models on the 386 fire-danger
items of the Mesogeos Track A holdout (2021-2022, fold 0, base rate 0.339), comparing bare
and grounded stated probabilities against observed fire frequency.

Panels (standard model order):
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
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Ensure figures/, repo root, and analysis/ are on sys.path
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "analysis") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "analysis"))

import figstyle as fs
import calibration_mesogeos as cm
import models
from models import add_model_args, resolve_models

# Color constants
COLOR_PRIOR_TEXT = "#3D735E"  # Dark mint for high-contrast legible text on white
COLOR_GRAY_TEXT = "#666666"   # Contrast-enhanced gray for text labels


def load_calibration_data(repo_root: Path, selected_models) -> dict:
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

    # 2. Selected models, bare and grounded
    models_data = []
    for m in selected_models:
        stem = m.stem
        label = m.label
        bare_p = task_dir / f"responses-{stem}-bare.jsonl"
        grd_p = task_dir / f"responses-{stem}-grounded.jsonl"
        if not (bare_p.exists() and grd_p.exists()):
            continue
        model_entry = {"label": label, "stem": stem}
        for cond, resp_path in (("bare", bare_p), ("grounded", grd_p)):
            rows = [json.loads(line) for line in resp_path.read_text(encoding="utf-8").splitlines() if line.strip()]
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


def print_verification_table(cal_data: dict, repo_root: Path) -> None:
    """Print verification table from computed calibration data and recorded results."""
    ref_json_path = repo_root / "analysis" / "calibration_mesogeos.json"
    ref_data = {}
    if ref_json_path.exists():
        try:
            ref_data = json.loads(ref_json_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    print("=" * 110)
    print("CALIBRATION METRICS ON MESOGEOS TRACK A (386 ITEMS)")
    print("=" * 110)
    base_rec = cal_data["base_rate"]
    print(f"Dataset Base Rate: {base_rec:.3f}")
    print("-" * 110)

    prior_rec = cal_data["prior"]
    p_mean_rec, p_ece_rec = prior_rec["mean_predicted"], prior_rec["ece_equal_width"]
    print(f"Reference: Calendar-Month Prior")
    print(f"  Mean p: {p_mean_rec:.3f} | ECE: {p_ece_rec:.3f}")
    print("-" * 110)

    print(f"{'Model':<20} {'Arm':<10} {'Mean p':>10} {'ECE(w)':>10} {'Brier':>10}")
    print("-" * 110)

    for m in cal_data["models"]:
        label = m["label"]
        for cond in ("bare", "grounded"):
            rec_cal = m[cond]
            rec_mean = rec_cal["mean_predicted"]
            rec_ece = rec_cal["ece_equal_width"]
            rec_brier = rec_cal["brier_score"]
            print(f"{label:<20} {cond:<10} {rec_mean:>10.3f} {rec_ece:>10.3f} {rec_brier:>10.3f}")

    print("=" * 110)


def plot_calibration(cal_data: dict, out_pdf: Path | None, out_png: Path) -> None:
    """Draw the reliability diagrams."""
    fs.apply()

    n_models = len(cal_data["models"])
    n_cols = 3
    n_rows = 2 if n_models <= 6 else math.ceil(n_models / n_cols)
    fig_height = 2.55 if n_models <= 6 else max(2.55, 1.25 * n_rows)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(fs.TEXT_WIDTH_IN, fig_height),
        sharex=True,
        sharey=True,
        gridspec_kw={
            "left": 0.08,
            "right": 0.98,
            "bottom": 0.13 if n_models <= 6 else 0.08,
            "top": 0.89 if n_models <= 6 else 0.94,
            "wspace": 0.22,
            "hspace": 0.36,
        },
    )
    axes = np.atleast_2d(axes)

    prior_bins = [b for b in cal_data["prior"]["reliability_table_equal_width"] if b["count"] > 0]
    prior_x = [b["mean_pred"] for b in prior_bins]
    prior_y = [b["obs_rate"] for b in prior_bins]
    prior_cnt = [b["count"] for b in prior_bins]
    prior_ece = cal_data["prior"]["ece_equal_width"]

    for idx, m in enumerate(cal_data["models"]):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        label = m["label"]
        tag = f"({chr(ord('a') + idx)})" if idx < 26 else f"({idx + 1})"

        fs.bare(ax, grid=None)
        fs.panel_title(ax, tag, label, x=0.0, y=1.04)

        # 1. Diagonal line of perfect calibration (y = x)
        ax.plot([0, 1], [0, 1], linestyle="--", color=fs.LIGHT_GRAY, linewidth=0.8, zorder=1)

        # 2. Calendar-month prior reliability curve
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

            for line_i, (t_str, col_t) in enumerate(reversed(lines)):
                ax.text(
                    tx, ty + line_i * 0.11, t_str, transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=fs.FS_SMALL, fontweight="bold", color=col_t, zorder=5
                )

        ax.set_xlim(-0.07, 1.03)
        ax.set_ylim(-0.07, 1.03)
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.set_yticks([0.0, 0.5, 1.0])

    # Hide unused subplots
    for empty_idx in range(n_models, n_rows * n_cols):
        r = empty_idx // n_cols
        c = empty_idx % n_cols
        axes[r, c].set_visible(False)

    # Axis labels on perimeter
    for idx in range(n_models):
        r = idx // n_cols
        c = idx % n_cols
        if r == n_rows - 1 or idx + n_cols >= n_models:
            axes[r, c].set_xlabel("Mean stated probability", fontsize=fs.FS_AXIS, labelpad=2)
        if c == 0:
            axes[r, c].set_ylabel("Observed frequency", fontsize=fs.FS_AXIS, labelpad=2)

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
    parser = argparse.ArgumentParser(description="Generate fire danger calibration figure.")
    parser.add_argument("--repo", type=str, default=str(_REPO_ROOT),
                        help="Path to repository root.")
    parser.add_argument("--out-pdf", type=Path, default=None,
                        help="Path for output PDF file.")
    parser.add_argument("--out-png", type=Path, default=None,
                        help="Path for output PNG file.")
    add_model_args(parser, default_tier="core")
    args = parser.parse_args()

    repo_root = Path(args.repo).resolve()
    figs_dir = repo_root / "figures"
    if args.out_pdf:
        out_pdf = args.out_pdf
    elif args.out_png:
        out_pdf = None
    else:
        out_pdf = figs_dir / "calibration.pdf"
    out_png = args.out_png or (figs_dir / "calibration.png")

    selected_models = resolve_models(args, task="mesogeos", default_tier="core")

    cal_data = load_calibration_data(repo_root, selected_models)
    print_verification_table(cal_data, repo_root)
    plot_calibration(cal_data, out_pdf, out_png)


if __name__ == "__main__":
    main()
