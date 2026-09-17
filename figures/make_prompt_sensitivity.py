#!/usr/bin/env python
"""Generate the prompt sensitivity figure (figures/prompt_sensitivity.pdf and .png).

This figure displays item-by-item prompt sensitivity on fire danger forecasting
(Mesogeos Track A, 386 items) for the two leading models across two prompt paraphrases:
  - Panel (a): claude-opus-5 under p1
  - Panel (b): claude-opus-5 under p2
  - Panel (c): gemini-3.1-pro under p1
  - Panel (d): gemini-3.1-pro under p2

Encoding:
  - Items are aggregated per distinct (p0 value, paraphrase value, label) triple.
  - Marker area is proportional to the item count sharing each distinct triple:
      s = 2.2 * count + 3.0 (in points squared).
  - No-fire bubbles are drawn first in gray (#999999, alpha=0.80, zorder=2).
  - Fire bubbles are drawn on top in coral (#ED8D5A, alpha=0.80, zorder=3).
  - Both carry a thin white edge (0.4 pt) so overlapping bubbles separate cleanly.
  - The dashed light-gray line marks identity (y = x, unchanged probabilities).
  - Subtitles under each panel's bold title report Spearman rank correlation (rho)
    and AUPRC under p0 and under the paraphrase (fs.SUBTITLE color, fs.FS_SMALL).
  - One size legend in panel (a) shows reference bubbles labeled "1", "10", "100 items".
  - One class legend in panel (a) shows "fire" (coral) and "no fire" (gray).
  - Axis labels: x "stated probability, original prompt (p0)", y "stated probability, paraphrase".

Data sources in the AI4Fire repository:
  - task-mesogeos/items.jsonl: test split items (fold 0, 386 items).
  - task-mesogeos/responses-claude-opus-5-bare.jsonl (-bare-p1.jsonl, -bare-p2.jsonl)
  - task-mesogeos/responses-gemini-3.1-pro-bare.jsonl (-bare-p1.jsonl, -bare-p2.jsonl)
  - analysis/prompt_sensitivity.py / prompt_sensitivity.json:
    paired differences, Spearman correlations, and 95% cluster bootstrap intervals
    (20,000 resamples, seed 20260915, 352 spatial 1-degree x month blocks).

Styling conventions (CatchBench submission style / figstyle.py):
  - Width: 6.5 in (fs.TEXT_WIDTH_IN), height <= 2.4 in (2.32 in).
  - Typography: Sans-serif (DejaVu Sans / Arial), TrueType fonts (pdf.fonttype 42), >= 6.2 pt.
  - Spines: no top/right spines, gray (#999999) left/bottom spines, no tick marks.
  - Palette: Coral (#ED8D5A) for fire, Gray (#999999) for no fire,
    Light gray (#C9C9C9) for reference diagonal, Near-black (#1A1A1A) for text.
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
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "figures"))
sys.path.insert(0, str(REPO_ROOT / "analysis"))

import figstyle as fs
import cluster_uncertainty as cu
import run_mesogeos as rm

MODELS = ["claude-opus-5", "gemini-3.1-pro"]
VARIANTS = ["p0", "p1", "p2"]

PAPER_TABLE_VALUES = {
    ("claude-opus-5", "p0"): {
        "auprc": 0.688, "f1": 0.269, "call": 0.065, "omitted": 46,
        "d_auprc": None, "ci": None, "spearman": None,
    },
    ("claude-opus-5", "p1"): {
        "auprc": 0.731, "f1": 0.308, "call": 0.065, "omitted": 0,
        "d_auprc": 0.043, "ci": [0.009, 0.080], "spearman": 0.92,
    },
    ("claude-opus-5", "p2"): {
        "auprc": 0.703, "f1": 0.240, "call": 0.049, "omitted": 4,
        "d_auprc": 0.015, "ci": [-0.018, 0.049], "spearman": 0.93,
    },
    ("gemini-3.1-pro", "p0"): {
        "auprc": 0.697, "f1": 0.607, "call": 0.207, "omitted": 0,
        "d_auprc": None, "ci": None, "spearman": None,
    },
    ("gemini-3.1-pro", "p1"): {
        "auprc": 0.648, "f1": 0.549, "call": 0.161, "omitted": 0,
        "d_auprc": -0.049, "ci": [-0.093, -0.009], "spearman": 0.77,
    },
    ("gemini-3.1-pro", "p2"): {
        "auprc": 0.672, "f1": 0.452, "call": 0.119, "omitted": 0,
        "d_auprc": -0.026, "ci": [-0.075, 0.020], "spearman": 0.70,
    },
}


def count_to_size(count: int | float) -> float:
    """Marker area proportional to count: s = 3.0 * count (points squared), so a 100-item bubble has 100 times the area of a 1-item bubble."""
    return 3.0 * count


def load_jsonl_map(path: Path) -> dict[str, dict]:
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
    return {r["item_id"]: r for r in rows}


def load_data(repo_root: Path):
    task_dir = repo_root / "task-mesogeos"
    items_raw = [json.loads(l) for l in (task_dir / "items.jsonl").read_text(encoding="utf-8").splitlines()]
    items = [i for i in items_raw if i["split"] == "test" and i.get("fold", 0) == 0]
    item_ids = [i["item_id"] for i in items]

    by_model_variant = {}
    for model in MODELS:
        by_model_variant[model] = {}
        for v in VARIANTS:
            suffix = "" if v == "p0" else f"-{v}"
            fpath = task_dir / f"responses-{model}-bare{suffix}.jsonl"
            by_model_variant[model][v] = load_jsonl_map(fpath)

    # Reference labels
    labels = np.array([by_model_variant["claude-opus-5"]["p0"][i]["label"] for i in item_ids], dtype=int)

    # Probabilities and runner scores
    scores = {}
    probs = {}
    auprcs = {}
    calls = {}
    omitted = {}
    for model in MODELS:
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

    # Pairwise metrics vs p0 and bubble counts
    pairwise = {}
    triples_map = {}
    for model in MODELS:
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
            }
            # Distinct (p0 value, paraphrase value, label) triples
            triples = []
            for i, it_id in enumerate(item_ids):
                x_val = p0_prob[i]
                y_val = pv_prob[i]
                lbl = int(labels[i])
                triples.append((x_val, y_val, lbl))
            c = Counter(triples)
            triples_map[model][v] = c

    # Load bootstrap intervals from prompt_sensitivity.json
    ps_json_path = repo_root / "analysis" / "prompt_sensitivity.json"
    ps_json = json.loads(ps_json_path.read_text(encoding="utf-8"))
    for model in MODELS:
        for v in ["p1", "p2"]:
            pairwise[model][v]["ci"] = ps_json[model][v]["d_auprc_ci"]

    return {
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
    print("VERIFICATION OF DATA POINTS, DIFFERENCES, AND INTERVALS AGAINST TABLE tab:prompt-sensitivity")
    print("=" * 125)
    header = f"{'Model':<16} {'Prompt':<8} {'AUPRC (calc/paper)':<20} {'F1 (calc/paper)':<18} {'Call (calc/paper)':<18} {'Omit':<6} {'Diff [95% CI] (calc vs paper)':<32} {'Spearman':<12}"
    print(header)
    print("-" * 125)

    all_matched = True
    for model in MODELS:
        for v in VARIANTS:
            s = data["scores"][model][v]
            omit = data["omitted"][model][v]
            calc_auprc = s["auprc"]
            calc_f1 = s["f1_fire"]
            calc_call = s["positive_rate_called"]

            p_vals = PAPER_TABLE_VALUES[(model, v)]
            p_auprc = p_vals["auprc"]
            p_f1 = p_vals["f1"]
            p_call = p_vals["call"]
            p_omit = p_vals["omitted"]

            auprc_str = f"{calc_auprc:.3f} / {p_auprc:.3f}"
            f1_str = f"{calc_f1:.3f} / {p_f1:.3f}"
            call_str = f"{calc_call:.3f} / {p_call:.3f}"
            omit_str = f"{omit} / {p_omit}"

            if v == "p0":
                diff_str = "--"
                spear_str = "--"
            else:
                pw = data["pairwise"][model][v]
                calc_diff = pw["d_auprc"]
                calc_ci = pw["ci"]
                calc_spear = pw["spearman"]

                p_diff = p_vals["d_auprc"]
                p_ci = p_vals["ci"]
                p_spear = p_vals["spearman"]

                diff_calc_str = f"{calc_diff:+.3f} [{calc_ci[0]:+.3f}, {calc_ci[1]:+.3f}]"
                diff_paper_str = f"{p_diff:+.3f} [{p_ci[0]:+.3f}, {p_ci[1]:+.3f}]"
                diff_str = f"{diff_calc_str} vs {diff_paper_str}"
                spear_str = f"{calc_spear:.2f} / {p_spear:.2f}"

            match = (
                abs(calc_auprc - p_auprc) < 1e-3 and
                abs(calc_f1 - p_f1) < 1e-3 and
                abs(calc_call - p_call) < 1e-3 and
                omit == p_omit
            )
            if v != "p0":
                match = (
                    match and
                    abs(pw["d_auprc"] - p_vals["d_auprc"]) < 1e-3 and
                    abs(pw["ci"][0] - p_vals["ci"][0]) < 1e-3 and
                    abs(pw["ci"][1] - p_vals["ci"][1]) < 1e-3 and
                    abs(pw["spearman"] - p_vals["spearman"]) < 0.015
                )

            status = "OK" if match else "MISMATCH"
            if not match:
                all_matched = False

            print(f"{model:<16} {v:<8} {auprc_str:<20} {f1_str:<18} {call_str:<18} {omit_str:<6} {diff_str:<32} {spear_str:<12} [{status}]")

    print("-" * 125)
    if all_matched:
        print("ALL 6 ROWS MATCH TABLE tab:prompt-sensitivity EXACTLY.\n")
    else:
        print("WARNING: ONE OR MORE VALUES DID NOT MATCH TABLE tab:prompt-sensitivity.\n")

    # Verification of bubble counts, distinct triples, and max counts per panel
    print("=" * 125)
    print("BUBBLE ENCODING VERIFICATION: DISTINCT TRIPLES, MAX COUNT, AND TOTAL SUM (ASSERT 386)")
    print("=" * 125)
    for model in MODELS:
        for v in ["p1", "p2"]:
            c = data["triples_map"][model][v]
            n_triples = len(c)
            top_triple, max_cnt = c.most_common(1)[0]
            total_items = sum(c.values())
            print(f"Panel ({model}, {v}): {n_triples} distinct triples, max count = {max_cnt} at (p0={top_triple[0]}, paraphrase={top_triple[1]}, label={top_triple[2]}), total items = {total_items}")
            assert total_items == 386, f"Expected 386 items for {model} {v}, got {total_items}"
    print("Assertion passed: all panels sum to exactly 386 items.\n")


def plot_prompt_sensitivity(data: dict, out_pdf: Path, out_png: Path):
    fs.apply()

    fig, axes = plt.subplots(1, 4, figsize=(fs.TEXT_WIDTH_IN, 2.32), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.18, top=0.81, wspace=0.14)

    panels_spec = [
        ("claude-opus-5", "p1", "(a)", "claude-opus-5, p1"),
        ("claude-opus-5", "p2", "(b)", "claude-opus-5, p2"),
        ("gemini-3.1-pro", "p1", "(c)", "gemini-3.1-pro, p1"),
        ("gemini-3.1-pro", "p2", "(d)", "gemini-3.1-pro, p2"),
    ]

    for ax, (model, v, tag, title) in zip(axes, panels_spec):
        fs.bare(ax, grid=None)
        # Reference diagonal (identity, y = x)
        ax.plot([0, 1], [0, 1], color=fs.LIGHT_GRAY, linestyle="--", linewidth=0.85, zorder=1)

        c = data["triples_map"][model][v]
        items0 = [(k, cnt) for k, cnt in c.items() if k[2] == 0]
        items1 = [(k, cnt) for k, cnt in c.items() if k[2] == 1]

        # Draw larger bubbles first so smaller bubbles are drawn on top
        items0.sort(key=lambda x: x[1], reverse=True)
        items1.sort(key=lambda x: x[1], reverse=True)

        # No-fire bubbles in gray (alpha=0.8, thin white edge 0.4 pt)
        for (x, y, _), cnt in items0:
            ax.scatter(x, y, s=count_to_size(cnt), color=fs.GRAY, alpha=0.8,
                       edgecolors="white", linewidths=0.4, zorder=2)

        # Fire bubbles in coral on top (alpha=0.8, thin white edge 0.4 pt)
        for (x, y, _), cnt in items1:
            ax.scatter(x, y, s=count_to_size(cnt), color=fs.CORAL, alpha=0.8,
                       edgecolors="white", linewidths=0.4, zorder=3)

        ax.set_xlim(-0.07, 1.03)
        ax.set_ylim(-0.07, 1.03)
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.set_yticks([0.0, 0.5, 1.0])

        # Bold title and gray subtitle outside plotting area
        rho = data["pairwise"][model][v]["spearman"]
        a0 = data["auprcs"][model]["p0"]
        av = data["auprcs"][model][v]

        ax.text(0.0, 1.14, f"{tag} {title}", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=fs.FS_TITLE, fontweight="bold", color=fs.NEAR_BLACK)
        ax.text(0.0, 1.03, f"$\\rho = {rho:.2f}$, AUPRC {a0:.3f} to {av:.3f}", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=fs.FS_SMALL, color=fs.SUBTITLE)

    axes[0].set_ylabel("stated probability, paraphrase", fontsize=fs.FS_AXIS)
    fig.text(0.53, 0.04, "stated probability, original prompt (p0)", ha="center", va="center", fontsize=fs.FS_AXIS)

    # Class and size legends drawn once in panel (a) upper left
    ax_a = axes[0]
    h_fire = ax_a.scatter([], [], s=count_to_size(15), color=fs.CORAL, alpha=0.8, edgecolors="white", linewidths=0.4)
    h_nofire = ax_a.scatter([], [], s=count_to_size(15), color=fs.GRAY, alpha=0.8, edgecolors="white", linewidths=0.4)

    h1 = ax_a.scatter([], [], s=count_to_size(1), color=fs.GRAY, alpha=0.8, edgecolors="white", linewidths=0.4)
    h10 = ax_a.scatter([], [], s=count_to_size(10), color=fs.GRAY, alpha=0.8, edgecolors="white", linewidths=0.4)
    h100 = ax_a.scatter([], [], s=count_to_size(100), color=fs.GRAY, alpha=0.8, edgecolors="white", linewidths=0.4)

    leg_class = ax_a.legend([h_fire, h_nofire], ["fire", "no fire"], loc="upper left", bbox_to_anchor=(0.03, 0.98),
                            frameon=False, fontsize=fs.FS_SMALL, ncol=1, handletextpad=0.3, labelspacing=0.25)
    ax_a.add_artist(leg_class)

    leg_size = ax_a.legend([h1, h10, h100], ["1", "10", "100 items"], loc="upper left", bbox_to_anchor=(0.03, 0.74),
                           frameon=False, fontsize=fs.FS_SMALL, ncol=1, handletextpad=1.2, labelspacing=1.0, handlelength=1.6)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    fig.savefig(out_png, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated {out_pdf} and {out_png}")


def main():
    parser = argparse.ArgumentParser(description="Generate prompt sensitivity figure.")
    parser.add_argument("--repo", type=str, default=str(REPO_ROOT),
                        help="Path to AI4Fire repository root.")
    parser.add_argument("--out-pdf", type=str, default=None,
                        help="Path for output PDF file.")
    parser.add_argument("--out-png", type=str, default=None,
                        help="Path for output PNG file.")
    args = parser.parse_args()

    repo_root = Path(args.repo).resolve()
    out_pdf = Path(args.out_pdf).resolve() if args.out_pdf else repo_root / "figures" / "prompt_sensitivity.pdf"
    out_png = Path(args.out_png).resolve() if args.out_png else repo_root / "figures" / "prompt_sensitivity.png"

    data = load_data(repo_root)
    print_verification_table(data)
    plot_prompt_sensitivity(data, out_pdf, out_png)


if __name__ == "__main__":
    main()
