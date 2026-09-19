"""Frame-clustered and question-clustered intervals for WildFireVQA.

This script computes bootstrap confidence intervals for the WildFireVQA aerial question-answering
benchmark (408 items across 390 FLAME 3 aerial frames and 34 question types) across evaluated models.

Clustering Methodology:
- Primary unit (frame-clustered): 390 FLAME 3 frames (identified by image_uid in items.jsonl).
  Frames are resampled with replacement using np.random.default_rng. The paired grounded-minus-bare
  difference is scored on the identical resampled cluster draw, preserving the within-frame pairing.
- Secondary unit (question-clustered): 34 question types (identified by question_id in items.jsonl).
  Used for sensitivity analysis to test whether grouping by question type alters inferential conclusions.

For each model and condition, the script computes:
1. Bare accuracy and 95% percentile cluster bootstrap confidence interval.
2. Grounded accuracy and 95% percentile cluster bootstrap confidence interval.
3. Paired grounded-minus-bare accuracy contrast and 95% percentile confidence interval.
4. Verdict relative to the 0.627 per-question majority baseline (256/408 = 0.62745...):
   whether the confidence interval lies strictly 'above', strictly 'below', or 'spans' 0.627.

Settings match the AI4Fire benchmark standards: 20,000 resamples and seed 20260915.
Seeds for each model arm are derived deterministically using stable_seed to guarantee reproducibility.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any, Dict, List, Tuple

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import cluster_uncertainty as cu  # noqa: E402
import models  # noqa: E402
from models import add_model_args, resolve_models  # noqa: E402

TASK = ROOT / "task-wildfirevqa"
CONDITIONS = ["bare", "grounded"]
MAJORITY_THRESHOLD = 0.627


def load_responses(path: pathlib.Path) -> Dict[str, Dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {r["item_id"]: r for r in rows if "item_id" in r}


def cluster_bootstrap_arm(
    groups: List[str],
    correct: np.ndarray,
    resamples: int,
    seed: int,
    ci: float = 95.0,
) -> Tuple[float, float, int]:
    """Compute percentile bootstrap interval for mean(correct) clustered by groups."""
    flat, starts, sizes, keys = cu.build_cluster_index(groups)
    rng = np.random.default_rng(seed)
    n_clusters = len(keys)
    means = np.empty(resamples, dtype=float)
    for r in range(resamples):
        draw = rng.integers(0, n_clusters, size=n_clusters)
        idx = cu.ragged_gather(flat, starts, sizes, draw)
        means[r] = correct[idx].mean()
    lo, hi, _ = cu.percentile_interval(means, ci)
    return float(lo), float(hi), n_clusters


def paired_cluster_bootstrap(
    groups: List[str],
    correct_bare: np.ndarray,
    correct_grounded: np.ndarray,
    resamples: int,
    seed: int,
    ci: float = 95.0,
) -> Tuple[float, float, int]:
    """Compute paired percentile bootstrap interval for grounded minus bare on identical draws."""
    flat, starts, sizes, keys = cu.build_cluster_index(groups)
    rng = np.random.default_rng(seed)
    n_clusters = len(keys)
    diffs = np.empty(resamples, dtype=float)
    for r in range(resamples):
        draw = rng.integers(0, n_clusters, size=n_clusters)
        idx = cu.ragged_gather(flat, starts, sizes, draw)
        diffs[r] = correct_grounded[idx].mean() - correct_bare[idx].mean()
    lo, hi, _ = cu.percentile_interval(diffs, ci)
    return float(lo), float(hi), n_clusters


def get_verdict(lo: float, hi: float, threshold: float = MAJORITY_THRESHOLD) -> str:
    if lo > threshold:
        return "above"
    elif hi < threshold:
        return "below"
    else:
        return "spans"


def analyze_model(
    m: models.Model,
    ids: List[str],
    by_id: Dict[str, Any],
    resamples: int,
    seed: int,
) -> Dict[str, Any]:
    stem = m.stem
    bare_path = TASK / f"responses-{stem}-bare.jsonl"
    grd_path = TASK / f"responses-{stem}-grounded.jsonl"
    if not bare_path.exists() or not grd_path.exists():
        return None

    bare_rows = load_responses(bare_path)
    grd_rows = load_responses(grd_path)

    # Use only items present in both arms
    common_ids = [iid for iid in ids if iid in bare_rows and iid in grd_rows]
    cb = np.array([1.0 if bare_rows[iid].get("correct") is True else 0.0 for iid in common_ids])
    cg = np.array([1.0 if grd_rows[iid].get("correct") is True else 0.0 for iid in common_ids])

    bare_acc = float(cb.mean())
    grd_acc = float(cg.mean())
    diff = float(cg.mean() - cb.mean())

    frame_groups = [by_id[iid]["image"]["image_uid"] for iid in common_ids]
    qid_groups = [by_id[iid]["question_id"] for iid in common_ids]

    # Frame-clustered
    f_b_lo, f_b_hi, n_frames = cluster_bootstrap_arm(
        frame_groups, cb, resamples, cu.stable_seed(seed, stem, "bare", "acc")
    )
    f_g_lo, f_g_hi, _ = cluster_bootstrap_arm(
        frame_groups, cg, resamples, cu.stable_seed(seed, stem, "grounded", "acc")
    )
    f_d_lo, f_d_hi, _ = paired_cluster_bootstrap(
        frame_groups, cb, cg, resamples, cu.stable_seed(seed, stem, "grounded-bare")
    )

    # Question-clustered
    q_b_lo, q_b_hi, n_qids = cluster_bootstrap_arm(
        qid_groups, cb, resamples, cu.stable_seed(seed, stem, "bare", "acc")
    )
    q_g_lo, q_g_hi, _ = cluster_bootstrap_arm(
        qid_groups, cg, resamples, cu.stable_seed(seed, stem, "grounded", "acc")
    )
    q_d_lo, q_d_hi, _ = paired_cluster_bootstrap(
        qid_groups, cb, cg, resamples, cu.stable_seed(seed, stem, "grounded-bare")
    )

    def diff_status(lo: float, hi: float) -> Tuple[bool, str]:
        if lo > 0.0:
            return True, "positive"
        elif hi < 0.0:
            return True, "negative"
        else:
            return False, "spans_zero"

    f_ex, f_dir = diff_status(f_d_lo, f_d_hi)
    q_ex, q_dir = diff_status(q_d_lo, q_d_hi)

    return {
        "label": m.label,
        "stem": m.stem,
        "order": m.order,
        "vendor": m.vendor,
        "weights": m.weights,
        "tier": m.tier,
        "items": len(common_ids),
        "frame_clustered": {
            "clusters": n_frames,
            "bare": {
                "accuracy": bare_acc,
                "ci": [f_b_lo, f_b_hi],
                "verdict_0_627": get_verdict(f_b_lo, f_b_hi),
            },
            "grounded": {
                "accuracy": grd_acc,
                "ci": [f_g_lo, f_g_hi],
                "verdict_0_627": get_verdict(f_g_lo, f_g_hi),
            },
            "grounded_minus_bare": {
                "d": diff,
                "ci": [f_d_lo, f_d_hi],
                "excludes_zero": f_ex,
                "direction": f_dir,
            },
        },
        "question_clustered": {
            "clusters": n_qids,
            "bare": {
                "accuracy": bare_acc,
                "ci": [q_b_lo, q_b_hi],
                "verdict_0_627": get_verdict(q_b_lo, q_b_hi),
            },
            "grounded": {
                "accuracy": grd_acc,
                "ci": [q_g_lo, q_g_hi],
                "verdict_0_627": get_verdict(q_g_lo, q_g_hi),
            },
            "grounded_minus_bare": {
                "d": diff,
                "ci": [q_d_lo, q_d_hi],
                "excludes_zero": q_ex,
                "direction": q_dir,
            },
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Compute frame-clustered intervals for WildFireVQA")
    parser.add_argument("--resamples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--out", type=pathlib.Path, default=HERE / "wildfirevqa_intervals.json")
    add_model_args(parser, default_tier="all")
    args = parser.parse_args()

    selected_models = resolve_models(args, task="wildfirevqa", default_tier="all")

    items_raw = [json.loads(l) for l in (TASK / "items.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    by_id = {i["item_id"]: i for i in items_raw}
    ids_all = [i["item_id"] for i in items_raw]
    maj_correct = np.array([1.0 if by_id[i]["baseline_majority_correct"] else 0.0 for i in ids_all])
    maj_mean = float(maj_correct.mean())

    model_records = {}
    for m in selected_models:
        rec = analyze_model(m, ids_all, by_id, args.resamples, args.seed)
        if rec:
            model_records[m.label] = rec

    # Summarize counts
    def compute_counts(cluster_key: str):
        b_above = sum(1 for r in model_records.values() if r[cluster_key]["bare"]["verdict_0_627"] == "above")
        b_below = sum(1 for r in model_records.values() if r[cluster_key]["bare"]["verdict_0_627"] == "below")
        b_spans = sum(1 for r in model_records.values() if r[cluster_key]["bare"]["verdict_0_627"] == "spans")

        g_above = sum(1 for r in model_records.values() if r[cluster_key]["grounded"]["verdict_0_627"] == "above")
        g_below = sum(1 for r in model_records.values() if r[cluster_key]["grounded"]["verdict_0_627"] == "below")
        g_spans = sum(1 for r in model_records.values() if r[cluster_key]["grounded"]["verdict_0_627"] == "spans")

        diff_pos = sum(
            1
            for r in model_records.values()
            if r[cluster_key]["grounded_minus_bare"]["excludes_zero"]
            and r[cluster_key]["grounded_minus_bare"]["direction"] == "positive"
        )
        diff_neg = sum(
            1
            for r in model_records.values()
            if r[cluster_key]["grounded_minus_bare"]["excludes_zero"]
            and r[cluster_key]["grounded_minus_bare"]["direction"] == "negative"
        )
        diff_spans = sum(
            1
            for r in model_records.values()
            if not r[cluster_key]["grounded_minus_bare"]["excludes_zero"]
        )
        return {
            "bare": {"above": b_above, "below": b_below, "spans": b_spans},
            "grounded": {"above": g_above, "below": g_below, "spans": g_spans},
            "grounded_minus_bare": {
                "excludes_zero_positive": diff_pos,
                "excludes_zero_negative": diff_neg,
                "spans_zero": diff_spans,
            },
        }

    summary = {
        "metadata": {
            "task": "wildfirevqa",
            "items": len(ids_all),
            "frames": len(set(by_id[i]["image"]["image_uid"] for i in ids_all)),
            "questions": len(set(by_id[i]["question_id"] for i in ids_all)),
            "majority_baseline": maj_mean,
            "resamples": args.resamples,
            "seed": args.seed,
            "ci": 95.0,
        },
        "models": model_records,
        "summary_counts": {
            "frame_clustered": compute_counts("frame_clustered"),
            "question_clustered": compute_counts("question_clustered"),
        },
    }

    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")

    # Print summary table (Frame clustered)
    print("\n" + "=" * 110)
    print(f"{'Model':<20} | {'Bare Acc [95% CI]':<26} | {'Grounded Acc [95% CI]':<26} | {'Difference [95% CI]':<23} | {'Verdict (Bare / Grd)':<20}")
    print("-" * 110)
    for m in selected_models:
        r = model_records[m.label]
        fc = r["frame_clustered"]
        b_str = f"{fc['bare']['accuracy']:.3f} [{fc['bare']['ci'][0]:.3f}, {fc['bare']['ci'][1]:.3f}]"
        g_str = f"{fc['grounded']['accuracy']:.3f} [{fc['grounded']['ci'][0]:.3f}, {fc['grounded']['ci'][1]:.3f}]"
        d_str = f"{fc['grounded_minus_bare']['d']:+.3f} [{fc['grounded_minus_bare']['ci'][0]:+.3f}, {fc['grounded_minus_bare']['ci'][1]:+.3f}]"
        v_str = f"{fc['bare']['verdict_0_627']} / {fc['grounded']['verdict_0_627']}"
        print(f"{m.label:<20} | {b_str:<26} | {g_str:<26} | {d_str:<23} | {v_str:<20}")
    print("=" * 110)

    # Print LaTeX rows
    print("\nLaTeX rows (registry order, 3 decimals):")
    for m in selected_models:
        r = model_records[m.label]
        fc = r["frame_clustered"]
        b_val = fc['bare']['accuracy']
        b_lo, b_hi = fc['bare']['ci']
        g_val = fc['grounded']['accuracy']
        g_lo, g_hi = fc['grounded']['ci']
        d_val = fc['grounded_minus_bare']['d']
        d_lo, d_hi = fc['grounded_minus_bare']['ci']
        v_bare = fc['bare']['verdict_0_627']
        v_grd = fc['grounded']['verdict_0_627']
        verdict = f"{v_bare} / {v_grd}" if v_bare != v_grd else v_bare

        # Format row: Label & Bare & Grounded & Diff & Verdict \\
        b_cell = f"{b_val:.3f} [{b_lo:.3f}, {b_hi:.3f}]"
        g_cell = f"{g_val:.3f} [{g_lo:.3f}, {g_hi:.3f}]"
        d_cell = f"{d_val:+.3f} [{d_lo:+.3f}, {d_hi:+.3f}]"
        print(f"{m.label} & {b_cell} & {g_cell} & {d_cell} & {verdict} \\\\")


if __name__ == "__main__":
    main()
