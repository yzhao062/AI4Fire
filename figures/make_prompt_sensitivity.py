#!/usr/bin/env python
"""Generate the prompt sensitivity figure (figures/prompt_sensitivity.pdf and .png).

This figure displays item-by-item prompt sensitivity on fire danger forecasting
(Mesogeos Track A, 386 items) for benchmark models across prompt paraphrases:
  - claude-opus-5 under p1 and p2
  - gemini-3.1-pro under p1 and p2
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Add repo and analysis/figures to sys.path
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "figures") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "figures"))
if str(REPO_ROOT / "analysis") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "analysis"))

import figstyle as fs
import cluster_uncertainty as cu
import run_mesogeos as rm
import models
from models import add_model_args, resolve_models

DEFAULT_MODELS = ["claude-opus-5", "gemini-3.1-pro"]
VARIANTS = ["p0", "p1", "p2"]


def count_to_size(count: int | float) -> float:
    """Marker area proportional to count: s = 3.0 * count (points squared)."""
    return 3.0 * count


def load_jsonl_map(path: Path) -> dict[str, dict]:
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return {r["item_id"]: r for r in rows}


def load_data(repo_root: Path, models_list: list[str]):
    task_dir = repo_root / "task-mesogeos"
    items_raw = [json.loads(l) for l in (task_dir / "items.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    items = [i for i in items_raw if i["split"] == "test" and i.get("fold", 0) == 0]
    item_ids = [i["item_id"] for i in items]
    labels = np.array([i["label"] for i in items], dtype=int)

    by_model_variant = {}
    valid_models = []
    for model in models_list:
        p0_path = task_dir / f"responses-{model}-bare.jsonl"
        p1_path = task_dir / f"responses-{model}-bare-p1.jsonl"
        p2_path = task_dir / f"responses-{model}-bare-p2.jsonl"
        if not (p0_path.exists() and p1_path.exists() and p2_path.exists()):
            continue
        valid_models.append(model)
        by_model_variant[model] = {
            "p0": load_jsonl_map(p0_path),
            "p1": load_jsonl_map(p1_path),
            "p2": load_jsonl_map(p2_path),
        }

    scores = {}
    probs = {}
    auprcs = {}
    calls = {}
    omitted = {}
    for model in valid_models:
        scores[model] = {}
        probs[model] = {}
        auprcs[model] = {}
        calls[model] = {}
        omitted[model] = {}
        for v in VARIANTS:
            rows = [by_model_variant[model][v][i] for i in item_ids]
            s = rm.score(rows, v)
            scores[model][v] = s
            omitted[model][v] = sum(1 for r in rows if r.get("call") is None and r.get("probability") is not None)
            prob_arr = np.array([cu.mesogeos_score(r) for r in rows], dtype=float)
            probs[model][v] = prob_arr
            auprcs[model][v] = cu.average_precision(labels, prob_arr)
            call_arr = np.array([r["call"] if r.get("call") is not None else (r.get("probability") or 0) >= 0.5
                                 for r in rows], dtype=bool)
            calls[model][v] = call_arr

    pairwise = {}
    triples_map = {}
    for model in valid_models:
        pairwise[model] = {}
        triples_map[model] = {}
        p0_prob = probs[model]["p0"]
        for v in ["p1", "p2"]:
            pv_prob = probs[model][v]
            rho = float(stats.spearmanr(p0_prob, pv_prob).correlation)
            d_auprc = auprcs[model][v] - auprcs[model]["p0"]
            pairwise[model][v] = {
                "spearman": rho,
                "d_auprc": d_auprc,
                "ci": [0.0, 0.0],
            }
            triples = []
            for i in range(len(item_ids)):
                triples.append((p0_prob[i], pv_prob[i], int(labels[i])))
            triples_map[model][v] = Counter(triples)

    ps_json_path = repo_root / "analysis" / "prompt_sensitivity.json"
    if ps_json_path.exists():
        try:
            ps_json = json.loads(ps_json_path.read_text(encoding="utf-8"))
            for model in valid_models:
                for v in ["p1", "p2"]:
                    if model in ps_json and v in ps_json[model]:
                        pairwise[model][v]["ci"] = ps_json[model][v].get("d_auprc_ci", [0.0, 0.0])
        except Exception:
            pass

    return {
        "models": valid_models,
        "items": items,
        "item_ids": item_ids,
        "labels": labels,
        "scores": scores,
        "probs": probs,
        "auprcs": auprcs,
        "omitted": omitted,
        "pairwise": pairwise,
        "triples_map": triples_map,
    }


def print_verification_table(data: dict):
    print("=" * 125)
    print("PROMPT SENSITIVITY: COMPUTED METRICS")
    print("=" * 125)
    header = f"{'Model':<16} {'Prompt':<8} {'AUPRC':<10} {'F1':<10} {'Call':<10} {'Omit':<6} {'Diff [95% CI]':<26} {'Spearman':<10}"
    print(header)
    print("-" * 125)
    for model in data["models"]:
        for v in VARIANTS:
            s = data["scores"][model][v]
            au = data["auprcs"][model][v]
            f1 = s["f1_fire"]
            call_rate = s["positive_rate_called"]
            omit = data["omitted"][model][v]
            if v == "p0":
                diff_str = "baseline (p0)"
                sp_str = "-"
            else:
                pw = data["pairwise"][model][v]
                diff_str = f"{pw['d_auprc']:+.3f} [{pw['ci'][0]:+.3f}, {pw['ci'][1]:+.3f}]"
                sp_str = f"{pw['spearman']:.2f}"
            print(f"{model:<16} {v:<8} {au:<10.3f} {f1:<10.3f} {call_rate:<10.3f} {omit:<6} {diff_str:<26} {sp_str:<10}")
    print("=" * 125)


def plot_prompt_sensitivity(data: dict, out_pdf: Path | None, out_png: Path):
    fs.apply()
    valid_models = data["models"]
    n_models = len(valid_models)

    if n_models == 2:
        # The paper's layout, kept byte-for-byte so the committed figure regenerates: one row of four panels with
        # the margins the figure shipped with. The refactor to a registry-driven model list changed these numbers
        # once and moved the canvas; do not retune them without regenerating figures/prompt_sensitivity.pdf.
        fig, axes = plt.subplots(1, 4, figsize=(fs.TEXT_WIDTH_IN, 2.32), sharex=True, sharey=True)
        fig.subplots_adjust(left=0.075, right=0.985, bottom=0.18, top=0.81, wspace=0.14)
        panel_configs = [
            (axes[0], valid_models[0], "p1", "(a)", f"{valid_models[0]}, p1"),
            (axes[1], valid_models[0], "p2", "(b)", f"{valid_models[0]}, p2"),
            (axes[2], valid_models[1], "p1", "(c)", f"{valid_models[1]}, p1"),
            (axes[3], valid_models[1], "p2", "(d)", f"{valid_models[1]}, p2"),
        ]
    else:
        fig, axes_2d = plt.subplots(
            n_models, 2,
            figsize=(fs.TEXT_WIDTH_IN, max(2.32, 2.0 * n_models)),
            sharex=True,
            sharey=True,
            gridspec_kw={"left": 0.10, "right": 0.98, "bottom": 0.10, "top": 0.92, "wspace": 0.20, "hspace": 0.35},
        )
        axes_2d = np.atleast_2d(axes_2d)
        panel_configs = []
        idx = 0
        for r, m in enumerate(valid_models):
            for c, v in enumerate(["p1", "p2"]):
                tag = f"({chr(ord('a') + idx)})" if idx < 26 else f"({idx + 1})"
                panel_configs.append((axes_2d[r, c], m, v, tag, f"{m}, {v}"))
                idx += 1

    for ax, model, v, tag, title in panel_configs:
        fs.bare(ax, grid=None)
        ax.plot([0, 1], [0, 1], color=fs.LIGHT_GRAY, linestyle="--", linewidth=0.85, zorder=1)

        counts = data["triples_map"][model][v]
        items0 = [(k, cnt) for k, cnt in counts.items() if k[2] == 0]
        items1 = [(k, cnt) for k, cnt in counts.items() if k[2] == 1]

        # Larger bubbles first within each class, so a small bubble stays visible on top of a large one.
        items0.sort(key=lambda x: x[1], reverse=True)
        items1.sort(key=lambda x: x[1], reverse=True)

        for (x_val, y_val, _), cnt in items0:
            ax.scatter(x_val, y_val, s=count_to_size(cnt), color=fs.GRAY, alpha=0.8,
                       edgecolors="white", linewidths=0.4, zorder=2)

        for (x_val, y_val, _), cnt in items1:
            ax.scatter(x_val, y_val, s=count_to_size(cnt), color=fs.CORAL, alpha=0.8,
                       edgecolors="white", linewidths=0.4, zorder=3)

        ax.set_xlim(-0.07, 1.03)
        ax.set_ylim(-0.07, 1.03)
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.set_yticks([0.0, 0.5, 1.0])

        rho = data["pairwise"][model][v]["spearman"]
        a0 = data["auprcs"][model]["p0"]
        av = data["auprcs"][model][v]

        ax.text(0.0, 1.14, f"{tag} {title}", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=fs.FS_TITLE, fontweight="bold", color=fs.NEAR_BLACK)
        ax.text(0.0, 1.03, f"$\\rho = {rho:.2f}$, AUPRC {a0:.3f} to {av:.3f}", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=fs.FS_SMALL, color=fs.SUBTITLE)

    if n_models == 2:
        axes[0].set_ylabel("stated probability, paraphrase", fontsize=fs.FS_AXIS)
        fig.text(0.53, 0.04, "stated probability, original prompt (p0)", ha="center", va="center", fontsize=fs.FS_AXIS)
        ax_first = axes[0]
    else:
        for r in range(n_models):
            axes_2d[r, 0].set_ylabel("paraphrase p", fontsize=fs.FS_AXIS)
        for c in range(2):
            axes_2d[-1, c].set_xlabel("original p0", fontsize=fs.FS_AXIS)
        ax_first = axes_2d[0, 0]

    h_fire = ax_first.scatter([], [], s=count_to_size(15), color=fs.CORAL, alpha=0.8, edgecolors="white", linewidths=0.4)
    h_nofire = ax_first.scatter([], [], s=count_to_size(15), color=fs.GRAY, alpha=0.8, edgecolors="white", linewidths=0.4)
    h1 = ax_first.scatter([], [], s=count_to_size(1), color=fs.GRAY, alpha=0.8, edgecolors="white", linewidths=0.4)
    h10 = ax_first.scatter([], [], s=count_to_size(10), color=fs.GRAY, alpha=0.8, edgecolors="white", linewidths=0.4)
    h100 = ax_first.scatter([], [], s=count_to_size(100), color=fs.GRAY, alpha=0.8, edgecolors="white", linewidths=0.4)

    leg_class = ax_first.legend([h_fire, h_nofire], ["fire", "no fire"], loc="upper left", bbox_to_anchor=(0.03, 0.98),
                                frameon=False, fontsize=fs.FS_SMALL, ncol=1, handletextpad=0.3, labelspacing=0.25)
    ax_first.add_artist(leg_class)
    ax_first.legend([h1, h10, h100], ["1", "10", "100 items"], loc="upper left", bbox_to_anchor=(0.03, 0.74),
                    frameon=False, fontsize=fs.FS_SMALL, ncol=1, handletextpad=1.2, labelspacing=1.0, handlelength=1.6)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    if out_pdf is not None:
        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
        print(f"Generated {out_pdf}")
    fig.savefig(out_png, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated {out_png}")


def main():
    parser = argparse.ArgumentParser(description="Generate prompt sensitivity figure.")
    parser.add_argument("--repo", type=str, default=str(REPO_ROOT),
                        help="Path to AI4Fire repository root.")
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
        out_pdf = figs_dir / "prompt_sensitivity.pdf"
    out_png = args.out_png or (figs_dir / "prompt_sensitivity.png")

    if args.models or args.tier != "core":
        selected = resolve_models(args, default_tier="core")
        models_list = [m.stem for m in selected]
    else:
        models_list = DEFAULT_MODELS

    data = load_data(repo_root, models_list)
    print_verification_table(data)
    plot_prompt_sensitivity(data, out_pdf, out_png)


if __name__ == "__main__":
    main()
