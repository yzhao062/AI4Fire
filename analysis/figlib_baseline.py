"""Non-LLM comparators for the FIgLib wildfire smoke detection task.

Four of the five tasks in AI4Fire are scored beside non-LLM comparators. This script provides the
missing comparators for FIgLib smoke detection, evaluated on exactly the same 196 paired items and
with the same metrics as the model evaluation:

  1. Constant baselines:
     - always_clear:  always predict 'no smoke' (all False)
     - always_smoke:  always predict 'smoke' (all True; majority class)
  2. Reference-frame difference detectors:
     - diff_mean_lofo:  mean absolute grayscale pixel difference between query frame and reference frame;
                        threshold selected by leave-one-fire-out cross-validation over the 17 fire clusters.
     - diff_mean_eval:  mean absolute pixel difference; best threshold fitted on the 196 evaluation items
                        (answers in hand, upper bound).
     - diff_frac15_lofo:  fraction of pixels with absolute change > 15; leave-one-fire-out threshold.
     - diff_frac15_eval:  fraction of pixels with absolute change > 15; best threshold on evaluation items.

Report per comparator: accuracy, recall on smoke, false-positive rate on clear, sequences detected,
bucket-level accuracies, and a paired comparison against each core model's bare and grounded arms
with a fire-clustered bootstrap (17 clusters, 20,000 resamples, house seeds).

Usage:
    python analysis/figlib_baseline.py
    python analysis/figlib_baseline.py --resamples 20000 --seed 20260915
"""
import argparse
import collections
import json
import os
import pathlib
import sys

import numpy as np
from PIL import Image

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import cluster_uncertainty as cu  # noqa: E402
import models  # noqa: E402

TASK_DIR = ROOT / "task-figlib"
ITEMS_PATH = TASK_DIR / "items.jsonl"

# Image directory locations, preferring the downsampled copy when it is present.
CANDIDATE_IMG_DIRS = [
    TASK_DIR / "images-1568",
    TASK_DIR / "images",
]

CORE_MODELS = [
    ("claude-opus-4.8", "claude-opus-4.8"),
    ("claude-opus-5", "claude-opus-5"),
    ("gemini-3.1-pro", "gemini-3.1-pro"),
    ("gpt-6-astra", "gpt-6-astra"),
    ("bedrock_qwen.qwen3-vl-235b-a22b", "Qwen3-VL"),
    ("bedrock_us.meta.llama4-maverick-17b-instruct-v1_0", "Llama 4 Maverick"),
]

BUCKET_ORDER = [
    "before 25 min or more",
    "before 10 to 25 min",
    "before 0 to 10 min",
    "after 0 to 10 min",
    "after 10 to 25 min",
    "after 25 min or more",
]


def bucket(offset):
    a = abs(offset)
    return ("%s %s" % ("after" if offset > 0 else "before",
                       "0 to 10 min" if a <= 600 else "10 to 25 min" if a <= 1500 else "25 min or more"))


def resolve_image_path(img_rel, img_dirs):
    p = pathlib.Path(img_rel)
    name = p.name
    for base in img_dirs:
        cand = base / name
        if cand.exists():
            return cand
    for base in img_dirs:
        parent_cand = base.parent / img_rel
        if parent_cand.exists():
            return parent_cand
    raise FileNotFoundError("Image %s not found in candidate paths: %r" % (img_rel, [str(d) for d in img_dirs]))


def find_best_threshold(scores, y):
    """Maximize accuracy; choose the lower middle maximizing cut.

    The two infinite cuts keep the all-smoke and all-clear classifiers in the search. Taking the
    median of the tied cuts instead of one of them can select an inferior cut lying between two
    maximizers, which happened in 11 of the 17 training folds. Review round 1, Codex N2.
    """
    values = np.unique(scores)
    cuts = np.r_[-np.inf, (values[:-1] + values[1:]) / 2.0, np.inf]
    accs = np.array([float(np.mean((scores >= cut) == y)) for cut in cuts])
    best = np.flatnonzero(accs == accs.max())
    index = best[(len(best) - 1) // 2]
    return float(cuts[index]), float(accs[index])


def fit_lofo_thresholds(scores, y, groups):
    """Leave-one-fire-out threshold fitting."""
    preds = np.zeros(len(scores), dtype=bool)
    cuts_by_group = {}
    unique_groups = sorted(set(groups))
    for g in unique_groups:
        mask = np.array([grp == g for grp in groups])
        train_s = scores[~mask]
        train_y = y[~mask]
        c_opt, _ = find_best_threshold(train_s, train_y)
        cuts_by_group[g] = c_opt
        preds[mask] = scores[mask] >= c_opt
    return preds, cuts_by_group


def compute_metrics(preds, y, eval_items):
    """Compute standard FIgLib evaluation metrics."""
    n = len(preds)
    correct = (preds == y)
    acc = float(np.mean(correct))
    smoke_mask = y
    rec = float(np.mean(preds[smoke_mask])) if smoke_mask.any() else None
    fpr = float(np.mean(preds[~smoke_mask])) if (~smoke_mask).any() else None

    # Bucket accuracy
    by_bucket = {}
    for b in BUCKET_ORDER:
        idx = [i for i, it in enumerate(eval_items) if it["bucket"] == b]
        if idx:
            by_bucket[b] = {
                "n": len(idx),
                "accuracy": round(float(np.mean(preds[idx] == y[idx])), 3),
                "smoke_count": int(np.sum(y[idx])),
            }

    # Sequences detected: count of smoke sequences with at least one smoke frame called
    seq_hits = collections.defaultdict(list)
    for it, pred in zip(eval_items, preds):
        if it["offset_seconds"] > 0:
            seq_hits[it["sequence"]].append((it["offset_seconds"], bool(pred)))

    detected_count = 0
    first_hit_offsets = []
    total_seqs = len(seq_hits)
    for seq, pairs in seq_hits.items():
        hits = [off for off, p in sorted(pairs) if p]
        if hits:
            detected_count += 1
            first_hit_offsets.append(hits[0])

    return {
        "n_items": n,
        "accuracy": round(acc, 3),
        "accuracy_raw": acc,
        "recall_on_smoke": round(rec, 3) if rec is not None else None,
        "recall_raw": rec,
        "false_positive_rate": round(fpr, 3) if fpr is not None else None,
        "fpr_raw": fpr,
        "sequences_detected": f"{detected_count} of {total_seqs}",
        "sequences_detected_ratio": detected_count / total_seqs if total_seqs else None,
        "median_detection_offset_seconds": float(np.median(first_hit_offsets)) if first_hit_offsets else None,
        "accuracy_by_bucket": by_bucket,
    }


def paired_bootstrap(groups, a_correct, b_correct, resamples, seed):
    """Cluster bootstrap interval on mean(b) - mean(a)."""
    flat, starts, sizes, keys = cu.build_cluster_index(groups)
    rng = np.random.default_rng(seed)
    diffs = np.empty(resamples)
    n_clusters = len(keys)
    for r in range(resamples):
        draw = rng.integers(0, n_clusters, size=n_clusters)
        idx = cu.ragged_gather(flat, starts, sizes, draw)
        diffs[r] = b_correct[idx].mean() - a_correct[idx].mean()
    lo, hi, _ = cu.percentile_interval(diffs, 95.0)
    return lo, hi, n_clusters


def paired_recall_bootstrap(groups, y, a_pred, b_pred, resamples, seed):
    """Cluster bootstrap interval on recall(b) - recall(a) on positive items."""
    flat, starts, sizes, keys = cu.build_cluster_index(groups)
    rng = np.random.default_rng(seed)
    diffs = np.empty(resamples)
    n_clusters = len(keys)
    for r in range(resamples):
        draw = rng.integers(0, n_clusters, size=n_clusters)
        idx = cu.ragged_gather(flat, starts, sizes, draw)
        mask = y[idx]
        if not mask.any():
            diffs[r] = np.nan
        else:
            diffs[r] = b_pred[idx][mask].mean() - a_pred[idx][mask].mean()
    lo, hi, _ = cu.percentile_interval(diffs, 95.0)
    return lo, hi, n_clusters


def paired_fpr_bootstrap(groups, y, a_pred, b_pred, resamples, seed):
    """Cluster bootstrap interval on fpr(b) - fpr(a) on clear items."""
    flat, starts, sizes, keys = cu.build_cluster_index(groups)
    rng = np.random.default_rng(seed)
    diffs = np.empty(resamples)
    n_clusters = len(keys)
    for r in range(resamples):
        draw = rng.integers(0, n_clusters, size=n_clusters)
        idx = cu.ragged_gather(flat, starts, sizes, draw)
        mask = ~y[idx]
        if not mask.any():
            diffs[r] = np.nan
        else:
            diffs[r] = b_pred[idx][mask].mean() - a_pred[idx][mask].mean()
    lo, hi, _ = cu.percentile_interval(diffs, 95.0)
    return lo, hi, n_clusters


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--resamples", type=int, default=20000, help="Number of bootstrap resamples (default 20000)")
    ap.add_argument("--seed", type=int, default=20260915, help="Base seed for reproducibility (default 20260915)")
    ap.add_argument("--out", type=pathlib.Path, default=HERE / "figlib_baseline.json", help="Output JSON path")
    ap.add_argument("--images-dir", type=pathlib.Path, default=None, help="Path to images directory")
    args = ap.parse_args()

    img_dirs = [args.images_dir] if args.images_dir else []
    img_dirs += [d for d in CANDIDATE_IMG_DIRS if d.exists()]
    if not img_dirs:
        raise SystemExit("Cannot find images directory. Pass --images-dir explicitly.")

    items_list = [json.loads(l) for l in ITEMS_PATH.read_text(encoding="utf-8").splitlines()]
    for it in items_list:
        it["bucket"] = bucket(it["offset_seconds"])

    reference = {}
    for it in items_list:
        cur = reference.get(it["sequence"])
        if cur is None or it["offset_seconds"] < cur["offset_seconds"]:
            reference[it["sequence"]] = it

    ref_ids = {it["item_id"] for it in reference.values()}
    eval_items = [it for it in items_list if it["item_id"] not in ref_ids]

    n_eval = len(eval_items)
    eval_ids = [it["item_id"] for it in eval_items]
    y = np.array([it["label"] == "smoke" for it in eval_items], dtype=bool)
    groups = [it["fire_name"] for it in eval_items]
    n_smoke = int(np.sum(y))
    n_clear = n_eval - n_smoke
    n_fires = len(set(groups))
    n_seqs = len({it["sequence"] for it in eval_items})

    print(f"FIgLib Smoke Detection Baselines: {n_eval} items ({n_smoke} smoke, {n_clear} clear)")
    print(f"Clusters: {n_fires} fires, {n_seqs} sequences | Resamples: {args.resamples} | Seed: {args.seed}")
    shown = [str(d.relative_to(ROOT)) if d.is_relative_to(ROOT) else d.name for d in img_dirs]
    print(f"Image search path (relative to repository root): {shown}")

    # Load images as grayscale float arrays
    cache = {}
    needed_items = eval_items + list(reference.values())
    print(f"Loading {len(needed_items)} image frames into memory ...", flush=True)
    for it in needed_items:
        iid = it["item_id"]
        if iid not in cache:
            p = resolve_image_path(it["image"], img_dirs)
            with Image.open(p) as im:
                cache[iid] = np.asarray(im.convert("L"), dtype=np.float32)

    # Compute frame differences
    diff_mean_arr = np.empty(n_eval, dtype=np.float32)
    diff_frac15_arr = np.empty(n_eval, dtype=np.float32)
    for idx, it in enumerate(eval_items):
        q_arr = cache[it["item_id"]]
        r_arr = cache[reference[it["sequence"]]["item_id"]]
        d = np.abs(q_arr - r_arr)
        diff_mean_arr[idx] = d.mean()
        diff_frac15_arr[idx] = (d > 15.0).mean()

    # Threshold fitting
    best_c_mean, best_acc_mean = find_best_threshold(diff_mean_arr, y)
    lofo_pred_mean, lofo_cuts_mean = fit_lofo_thresholds(diff_mean_arr, y, groups)
    eval_pred_mean = diff_mean_arr >= best_c_mean

    best_c_frac, best_acc_frac = find_best_threshold(diff_frac15_arr, y)
    lofo_pred_frac, lofo_cuts_frac = fit_lofo_thresholds(diff_frac15_arr, y, groups)
    eval_pred_frac = diff_frac15_arr >= best_c_frac

    pred_always_clear = np.zeros(n_eval, dtype=bool)
    pred_always_smoke = np.ones(n_eval, dtype=bool)

    comparators = {
        "always_clear": {
            "label": "Constant: always clear",
            "assumption": "Do-nothing constant; predicts no smoke on all frames.",
            "preds": pred_always_clear,
            "threshold_method": "constant False",
        },
        "always_smoke": {
            "label": "Constant: always smoke (majority)",
            "assumption": "Do-nothing constant / best-constant majority class (112 smoke vs 84 clear).",
            "preds": pred_always_smoke,
            "threshold_method": "constant True",
        },
        "diff_mean_lofo": {
            "label": "Frame difference: mean |diff| (held-out LOFO)",
            "assumption": "Mean absolute grayscale pixel change; threshold fit leave-one-fire-out across 17 fire clusters.",
            "preds": lofo_pred_mean,
            "threshold_method": "leave-one-fire-out cross-validation",
            "threshold_median": float(np.median(list(lofo_cuts_mean.values()))),
            "threshold_range": [float(min(lofo_cuts_mean.values())), float(max(lofo_cuts_mean.values()))],
            "scores": diff_mean_arr.tolist(),
        },
        "diff_mean_eval": {
            "label": "Frame difference: mean |diff| (eval answers in hand)",
            "assumption": "Mean absolute pixel change; optimal accuracy threshold fitted on all 196 evaluation items.",
            "preds": eval_pred_mean,
            "threshold_method": "oracle best cut on evaluation items",
            "threshold": best_c_mean,
            "scores": diff_mean_arr.tolist(),
        },
        "diff_frac15_lofo": {
            "label": "Frame difference: fraction > 15 (held-out LOFO)",
            "assumption": "Fraction of pixels moving > 15 grayscale levels; leave-one-fire-out threshold.",
            "preds": lofo_pred_frac,
            "threshold_method": "leave-one-fire-out cross-validation",
            "threshold_median": float(np.median(list(lofo_cuts_frac.values()))),
            "threshold_range": [float(min(lofo_cuts_frac.values())), float(max(lofo_cuts_frac.values()))],
            "scores": diff_frac15_arr.tolist(),
        },
        "diff_frac15_eval": {
            "label": "Frame difference: fraction > 15 (eval answers in hand)",
            "assumption": "Fraction of pixels moving > 15 levels; optimal threshold on all 196 evaluation items.",
            "preds": eval_pred_frac,
            "threshold_method": "oracle best cut on evaluation items",
            "threshold": best_c_frac,
            "scores": diff_frac15_arr.tolist(),
        },
    }

    comp_metrics = {}
    for k, v in comparators.items():
        comp_metrics[k] = compute_metrics(v["preds"], y, eval_items)

    # Load core models
    model_data = {}
    for stem, label in CORE_MODELS:
        for arm in ("bare", "grounded"):
            path = TASK_DIR / f"responses-{stem}-{arm}.jsonl"
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            by_id = {r["item_id"]: r for r in rows}
            model_preds = np.array([bool(by_id[iid]["prediction"]) for iid in eval_ids], dtype=bool)
            model_data[(stem, arm)] = {
                "stem": stem,
                "label": label,
                "arm": arm,
                "preds": model_preds,
                "metrics": compute_metrics(model_preds, y, eval_items),
            }

    # Run paired bootstrap contrasts: model minus comparator
    contrasts = []
    print(f"\nRunning {args.resamples} cluster bootstrap resamples per model-comparator pair ...", flush=True)

    for stem, label in CORE_MODELS:
        for arm in ("bare", "grounded"):
            m_entry = model_data[(stem, arm)]
            m_preds = m_entry["preds"]
            m_correct = (m_preds == y).astype(float)
            m_rec = m_entry["metrics"]["recall_raw"]
            m_fpr = m_entry["metrics"]["fpr_raw"]

            row = {
                "stem": stem,
                "label": label,
                "arm": arm,
                "accuracy": m_entry["metrics"]["accuracy_raw"],
                "recall": m_rec,
                "fpr": m_fpr,
                "comparisons": {},
            }

            for comp_name, comp_entry in comparators.items():
                c_preds = comp_entry["preds"]
                c_correct = (c_preds == y).astype(float)
                c_metrics = comp_metrics[comp_name]

                # Accuracy contrast
                seed_acc = cu.stable_seed(args.seed, stem, arm, "model-" + comp_name, "acc")
                lo_acc, hi_acc, _ = paired_bootstrap(groups, c_correct, m_correct, args.resamples, seed_acc)
                diff_acc = float(m_correct.mean() - c_correct.mean())

                # Recall contrast on smoke
                seed_rec = cu.stable_seed(args.seed, stem, arm, "model-" + comp_name, "rec")
                lo_rec, hi_rec, _ = paired_recall_bootstrap(groups, y, c_preds, m_preds, args.resamples, seed_rec)
                diff_rec = float(m_rec - c_metrics["recall_raw"])

                # FPR contrast on clear
                seed_fpr = cu.stable_seed(args.seed, stem, arm, "model-" + comp_name, "fpr")
                lo_fpr, hi_fpr, _ = paired_fpr_bootstrap(groups, y, c_preds, m_preds, args.resamples, seed_fpr)
                diff_fpr = float(m_fpr - c_metrics["fpr_raw"])

                row["comparisons"][comp_name] = {
                    "accuracy_difference": diff_acc,
                    "accuracy_ci": [float(lo_acc), float(hi_acc)],
                    "accuracy_excludes_zero": bool(lo_acc > 0 or hi_acc < 0),
                    "recall_difference": diff_rec,
                    "recall_ci": [float(lo_rec), float(hi_rec)],
                    "recall_excludes_zero": bool(lo_rec > 0 or hi_rec < 0),
                    "fpr_difference": diff_fpr,
                    "fpr_ci": [float(lo_fpr), float(hi_fpr)],
                    "fpr_excludes_zero": bool(lo_fpr > 0 or hi_fpr < 0),
                    "seed": int(seed_acc),
                }

            contrasts.append(row)

    out = {
        "benchmark": "AI4Fire",
        "task": "FIgLib smoke detection",
        "n_items": n_eval,
        "n_smoke": n_smoke,
        "n_clear": n_clear,
        "n_clusters": n_fires,
        "n_sequences": n_seqs,
        "resamples": args.resamples,
        "seed": args.seed,
        "seed_recipe": "cluster_uncertainty.stable_seed(seed, stem, arm, 'model-' + comp_name, metric)",
        "comparators": {
            k: {
                "label": v["label"],
                "assumption": v["assumption"],
                "threshold_method": v["threshold_method"],
                "threshold": v.get("threshold", v.get("threshold_median")),
                "metrics": comp_metrics[k],
            }
            for k, v in comparators.items()
        },
        "models": {
            f"{stem}_{arm}": {
                "stem": stem,
                "label": label,
                "arm": arm,
                "metrics": model_data[(stem, arm)]["metrics"],
            }
            for stem, label in CORE_MODELS
            for arm in ("bare", "grounded")
        },
        "contrasts": contrasts,
    }

    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote results to {args.out}")

    # Print summary tables
    print("\n" + "=" * 115)
    print("TABLE 1: Non-LLM Comparators for FIgLib Smoke Detection (196 items, 17 fire clusters)")
    print("=" * 115)
    print(f"{'Comparator':<45} {'Accuracy':>9} {'Recall':>8} {'FPR':>8} {'Seqs':>10}  {'Threshold Info'}")
    print("-" * 115)
    for k, v in comparators.items():
        m = comp_metrics[k]
        th_info = v["threshold_method"]
        if "threshold" in v:
            th_info += f" (cut={v['threshold']:.3f})"
        elif "threshold_median" in v:
            th_info += f" (median cut={v['threshold_median']:.3f})"
        print(f"{v['label']:<45} {m['accuracy']:>9.3f} {m['recall_on_smoke']:>8.3f} {m['false_positive_rate']:>8.3f} {m['sequences_detected']:>10}  {th_info}")

    print("\n" + "=" * 115)
    print("TABLE 2: Core Models vs. Comparators (Accuracy difference: Model minus Comparator, 95% Cluster CI)")
    print("=" * 115)
    print(f"{'Model':<18} {'Arm':<9} {'Acc':>6}  {'vs Always Smoke (Majority)':<30}  {'vs Frame Diff LOFO':<28}  {'vs Frame Diff Eval'}")
    print("-" * 115)
    for row in contrasts:
        c_maj = row["comparisons"]["always_smoke"]
        c_lofo = row["comparisons"]["diff_mean_lofo"]
        c_eval = row["comparisons"]["diff_mean_eval"]

        s_maj = f"{c_maj['accuracy_difference']:+.3f} [{c_maj['accuracy_ci'][0]:+.3f},{c_maj['accuracy_ci'][1]:+.3f}]{'*' if c_maj['accuracy_excludes_zero'] else ' '}"
        s_lofo = f"{c_lofo['accuracy_difference']:+.3f} [{c_lofo['accuracy_ci'][0]:+.3f},{c_lofo['accuracy_ci'][1]:+.3f}]{'*' if c_lofo['accuracy_excludes_zero'] else ' '}"
        s_eval = f"{c_eval['accuracy_difference']:+.3f} [{c_eval['accuracy_ci'][0]:+.3f},{c_eval['accuracy_ci'][1]:+.3f}]{'*' if c_eval['accuracy_excludes_zero'] else ' '}"

        print(f"{row['label']:<18} {row['arm']:<9} {row['accuracy']:>6.3f}  {s_maj:<30}  {s_lofo:<28}  {s_eval}")
    print("* marks an interval excluding zero.")

    print("\n" + "=" * 115)
    print("TABLE 3: Core Models vs. Frame Diff LOFO (Decomposed by Metric)")
    print("=" * 115)
    print(f"{'Model':<18} {'Arm':<9}  {'Accuracy Diff [95% CI]':<28}  {'Recall Diff [95% CI]':<28}  {'FPR Diff [95% CI]'}")
    print("-" * 115)
    for row in contrasts:
        c_lofo = row["comparisons"]["diff_mean_lofo"]
        s_acc = f"{c_lofo['accuracy_difference']:+.3f} [{c_lofo['accuracy_ci'][0]:+.3f},{c_lofo['accuracy_ci'][1]:+.3f}]{'*' if c_lofo['accuracy_excludes_zero'] else ' '}"
        s_rec = f"{c_lofo['recall_difference']:+.3f} [{c_lofo['recall_ci'][0]:+.3f},{c_lofo['recall_ci'][1]:+.3f}]{'*' if c_lofo['recall_excludes_zero'] else ' '}"
        s_fpr = f"{c_lofo['fpr_difference']:+.3f} [{c_lofo['fpr_ci'][0]:+.3f},{c_lofo['fpr_ci'][1]:+.3f}]{'*' if c_lofo['fpr_excludes_zero'] else ' '}"
        print(f"{row['label']:<18} {row['arm']:<9}  {s_acc:<28}  {s_rec:<28}  {s_fpr}")
    print("* marks an interval excluding zero. (Note: for FPR, negative difference means model has lower false alarm rate).")


if __name__ == "__main__":
    main()
