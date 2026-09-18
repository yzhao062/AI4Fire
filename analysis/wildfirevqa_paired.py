"""Paired grounded-against-bare comparison on the WildFireVQA aerial question-answering task (408 items, 390 frames).

Both arms of run_wildfirevqa.py answer the same 408 questions over the same two images per frame; the grounded arm
adds the temp_summary block of seven radiometric statistics. This script reports accuracy per arm, overall and by
category, the paired difference in accuracy with a cluster bootstrap interval over frame (390 clusters, the same
percentile machinery and seed as cluster_uncertainty.py), the majority-answer baseline the items carry, the closed-form
rule on the 48 items that the temp_summary block answers on its own (Appendix C of the paper), the applicability
strata, and the parse failures behind the residual errors.

    python analysis/wildfirevqa_paired.py
    python analysis/wildfirevqa_paired.py --resamples 20000 --seed 20260915
"""
import argparse
import collections
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import cluster_uncertainty as cu  # noqa: E402

TASK = ROOT / "task-wildfirevqa"
MODELS = {
    "bedrock_qwen.qwen3-vl-235b-a22b": "Qwen3-VL",
    "bedrock_us.meta.llama4-maverick-17b-instruct-v1_0": "Llama 4 Maverick",
    "claude-opus-4.8": "claude-opus-4.8",
    "claude-opus-5": "claude-opus-5",
    "gemini-3.1-pro": "gemini-3.1-pro",
    "gpt-6-astra": "gpt-6-astra",
}
CONDITIONS = ["bare", "grounded"]


def load(path):
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
    return {r["item_id"]: r for r in rows}


def paired_bootstrap(groups, a, b, resamples, seed):
    """Interval on mean(b) - mean(a) over resampled clusters, both arms on the same draw."""
    flat, starts, sizes, keys = cu.build_cluster_index(groups)
    rng = np.random.default_rng(seed)
    diffs = np.empty(resamples)
    for r in range(resamples):
        draw = rng.integers(0, len(keys), size=len(keys))
        idx = cu.ragged_gather(flat, starts, sizes, draw)
        diffs[r] = b[idx].mean() - a[idx].mean()
    lo, hi, _ = cu.percentile_interval(diffs, 95.0)
    return lo, hi, len(keys)


def cluster_interval(groups, x, resamples, seed):
    """Interval on mean(x) alone over resampled clusters."""
    flat, starts, sizes, keys = cu.build_cluster_index(groups)
    rng = np.random.default_rng(seed)
    means = np.empty(resamples)
    for r in range(resamples):
        draw = rng.integers(0, len(keys), size=len(keys))
        idx = cu.ragged_gather(flat, starts, sizes, draw)
        means[r] = x[idx].mean()
    lo, hi, _ = cu.percentile_interval(means, 95.0)
    return lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resamples", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20260915)
    args = ap.parse_args()
    items = [json.loads(l) for l in (TASK / "items.jsonl").read_text(encoding="utf-8").splitlines()]
    by_id = {i["item_id"]: i for i in items}
    ids_all = [i["item_id"] for i in items]
    groups_all = [by_id[i]["image"]["image_uid"] for i in ids_all]
    majority = np.array([1.0 if by_id[i]["baseline_majority_correct"] else 0.0 for i in ids_all])
    closed = [i for i in ids_all if by_id[i]["no_image_answerable"] == "closed_form"]
    rule = np.array([1.0 if by_id[i]["temp_summary_rule_correct"] else 0.0 for i in closed])
    cats = sorted({i["category"] for i in items})
    out = {"items": len(items), "frames": len(set(groups_all)), "categories": cats,
           "majority_baseline": {"accuracy": float(majority.mean()),
                                 "by_category": {c: float(np.mean([majority[k] for k, i in enumerate(ids_all) if by_id[i]["category"] == c])) for c in cats}},
           "closed_form_items": len(closed), "temp_summary_rule_accuracy_on_closed_form": float(rule.mean()) if len(rule) else None,
           "models": {}}
    print("items %d | frames %d | majority baseline %.3f | closed-form items %d, rule accuracy %.3f"
          % (len(items), out["frames"], majority.mean(), len(closed), rule.mean() if len(rule) else float("nan")))
    for stem, name in MODELS.items():
        files = {c: TASK / ("responses-%s-%s.jsonl" % (stem, c)) for c in CONDITIONS}
        if not all(p.exists() for p in files.values()):
            print("%s: arm files missing, skipped" % name)
            continue
        arms = {c: load(p) for c, p in files.items()}
        ids = [i for i in ids_all if all(i in a for a in arms.values())]
        groups = [by_id[i]["image"]["image_uid"] for i in ids]
        correct = {c: np.array([1.0 if arms[c][i].get("correct") is True else 0.0 for i in ids]) for c in CONDITIONS}
        rec = {"items": len(ids)}
        print("\n== %s, %d items, %d frames" % (name, len(ids), len(set(groups))))
        for c in CONDITIONS:
            rows = [arms[c][i] for i in ids]
            by_cat = {cat: float(np.mean([correct[c][k] for k, i in enumerate(ids) if by_id[i]["category"] == cat])) for cat in cats}
            lo, hi = cluster_interval(groups, correct[c], args.resamples, cu.stable_seed(args.seed, stem, c, "acc"))
            app = [r["applicability"] for r in rows if r.get("applicability") is not None]
            hi_app = [correct[c][k] for k, r in enumerate(rows) if (r.get("applicability") or 0.0) >= 0.5]
            lo_app = [correct[c][k] for k, r in enumerate(rows) if r.get("applicability") is not None and r["applicability"] < 0.5]
            cf = [correct[c][k] for k, i in enumerate(ids) if by_id[i]["no_image_answerable"] == "closed_form"]
            other = [correct[c][k] for k, i in enumerate(ids) if by_id[i]["no_image_answerable"] != "closed_form"]
            rec[c] = {"accuracy": float(correct[c].mean()), "accuracy_ci": [lo, hi], "by_category": by_cat,
                      "closed_form_accuracy": float(np.mean(cf)) if cf else None, "other_accuracy": float(np.mean(other)) if other else None,
                      "parse_failures": sum(1 for r in rows if r.get("prediction") is None), "errors": sum(1 for r in rows if r.get("error")),
                      "applicability_mean": float(np.mean(app)) if app else None, "items_applicability_lt_0.5": len(lo_app),
                      "accuracy_applicability_ge_0.5": float(np.mean(hi_app)) if hi_app else None,
                      "accuracy_applicability_lt_0.5": float(np.mean(lo_app)) if lo_app else None,
                      "tokens_in": sum((r.get("usage") or {}).get("prompt_tokens", 0) for r in rows),
                      "tokens_out": sum((r.get("usage") or {}).get("completion_tokens", 0) for r in rows)}
            print("  %-8s acc %.3f [%.3f, %.3f]  closed-form %.3f  other %.3f  unparsed %d  errors %d  applicability mean %.2f (%d below 0.5, acc there %s)"
                  % (c, rec[c]["accuracy"], lo, hi, rec[c]["closed_form_accuracy"] or 0, rec[c]["other_accuracy"] or 0, rec[c]["parse_failures"],
                     rec[c]["errors"], rec[c]["applicability_mean"] or 0, len(lo_app), "%.3f" % np.mean(lo_app) if lo_app else "n/a"))
            print("           by category: %s" % {k[:14]: round(v, 3) for k, v in by_cat.items()})
        d = float(correct["grounded"].mean() - correct["bare"].mean())
        lo, hi, k = paired_bootstrap(groups, correct["bare"], correct["grounded"], args.resamples, cu.stable_seed(args.seed, stem, "grounded-bare"))
        rec["grounded_minus_bare"] = {"d": d, "ci": [lo, hi], "clusters": k, "resamples": args.resamples}
        flips = collections.Counter()
        for k2, i in enumerate(ids):
            flips[("bare" if correct["bare"][k2] else "wrong") + "->" + ("grounded" if correct["grounded"][k2] else "wrong")] += 1
        rec["flips"] = {"bare_right_grounded_wrong": sum(1 for k2 in range(len(ids)) if correct["bare"][k2] and not correct["grounded"][k2]),
                        "bare_wrong_grounded_right": sum(1 for k2 in range(len(ids)) if not correct["bare"][k2] and correct["grounded"][k2])}
        by_cat_d = {cat: float(np.mean([correct["grounded"][k2] - correct["bare"][k2] for k2, i in enumerate(ids) if by_id[i]["category"] == cat])) for cat in cats}
        rec["grounded_minus_bare"]["by_category"] = by_cat_d
        print("  grounded - bare: %+.3f [%+.3f, %+.3f] over %d frame clusters | gained %d, lost %d | by category %s"
              % (d, lo, hi, k, rec["flips"]["bare_wrong_grounded_right"], rec["flips"]["bare_right_grounded_wrong"], {c[:14]: round(v, 3) for c, v in by_cat_d.items()}))
        out["models"][name] = rec
    (HERE / "wildfirevqa_paired.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("\nwrote", HERE / "wildfirevqa_paired.json")


if __name__ == "__main__":
    main()
