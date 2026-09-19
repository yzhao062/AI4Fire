"""Paired contrasts of each aerial model against the two non-model comparators.

`wildfirevqa_paired.py` exports grounded-minus-bare contrasts. The manuscript also reads each
model against two references that do not vary by arm:

  majority  the per-question-id majority class, computed on the released pool
  hybrid    that majority, with the closed-form temperature rules substituted on the 48 items
            they resolve

Both references are scored on the same 408 items as the models, so the contrast is paired and
clustered over the 390 frames, exactly as the arm contrasts are. Seeds follow the house
convention, `cluster_uncertainty.stable_seed(base, *parts)`, so a rerun reproduces every endpoint.

Usage:
    python analysis/wildfirevqa_comparators.py
    python analysis/wildfirevqa_comparators.py --resamples 20000 --seed 20260915
"""
import argparse
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import cluster_uncertainty as cu  # noqa: E402
import wildfirevqa_paired as wp  # noqa: E402

TASK = HERE.parent / "task-wildfirevqa"

MODELS = [
    ("claude-opus-4.8", "claude-opus-4.8"),
    ("claude-opus-5", "claude-opus-5"),
    ("gemini-3.1-pro", "gemini-3.1-pro"),
    ("gpt-6-astra", "gpt-6-astra"),
    ("bedrock_qwen.qwen3-vl-235b-a22b", "Qwen3-VL"),
    ("bedrock_us.meta.llama4-maverick-17b-instruct-v1_0", "Llama 4 Maverick"),
]


def load_items():
    rows = [json.loads(l) for l in (TASK / "items.jsonl").read_text(encoding="utf-8").splitlines()]
    return rows, {r["item_id"]: r for r in rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resamples", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20260915)
    ap.add_argument("--out", type=pathlib.Path, default=HERE / "wildfirevqa_comparators.json")
    args = ap.parse_args()

    items, by_id = load_items()
    ids = [r["item_id"] for r in items]
    groups = [by_id[i]["image"]["image_uid"] for i in ids]

    majority = np.array([1.0 if by_id[i]["baseline_majority_correct"] else 0.0 for i in ids])
    closed = np.array([by_id[i].get("no_image_answerable") == "closed_form" for i in ids])
    rule_ok = np.array([1.0 if by_id[i].get("temp_summary_rule_correct") else 0.0 for i in ids])
    hybrid = np.where(closed, rule_ok, majority)

    refs = {"majority": majority, "hybrid": hybrid}
    out = {
        "n_items": len(ids),
        "n_clusters": len(set(groups)),
        "resamples": args.resamples,
        "seed": args.seed,
        "seed_recipe": "cluster_uncertainty.stable_seed(seed, model_stem, arm, 'model-' + reference)",
        "references": {
            "majority": {"accuracy": float(majority.mean())},
            "hybrid": {
                "accuracy": float(hybrid.mean()),
                "closed_form_items": int(closed.sum()),
                "accuracy_on_closed_form": float(hybrid[closed].mean()),
                "accuracy_on_other": float(hybrid[~closed].mean()),
            },
        },
        "contrasts": [],
    }

    for stem, label in MODELS:
        for arm in ("bare", "grounded"):
            path = TASK / ("responses-%s-%s.jsonl" % (stem, arm))
            rows = {r["item_id"]: r for r in
                    (json.loads(l) for l in path.read_text(encoding="utf-8").splitlines())}
            correct = np.array([1.0 if rows[i].get("correct") is True else 0.0 for i in ids])
            row = {"model": label, "stem": stem, "arm": arm, "accuracy": float(correct.mean())}
            for ref_name, ref in refs.items():
                seed = cu.stable_seed(args.seed, stem, arm, "model-" + ref_name)
                lo, hi, n_clusters = wp.paired_bootstrap(groups, ref, correct, args.resamples, seed)
                row[ref_name] = {
                    "difference": float(correct.mean() - ref.mean()),
                    "lo": float(lo),
                    "hi": float(hi),
                    "excludes_zero": bool(lo > 0 or hi < 0),
                    "seed": int(seed),
                }
                out["n_clusters"] = n_clusters
            out["contrasts"].append(row)

    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("items %d, clusters %d, resamples %d, base seed %d"
          % (out["n_items"], out["n_clusters"], args.resamples, args.seed))
    print("majority %.4f   hybrid %.4f (%d closed-form items at %.4f, other at %.4f)"
          % (out["references"]["majority"]["accuracy"],
             out["references"]["hybrid"]["accuracy"],
             out["references"]["hybrid"]["closed_form_items"],
             out["references"]["hybrid"]["accuracy_on_closed_form"],
             out["references"]["hybrid"]["accuracy_on_other"]))
    print()
    print("%-18s %-9s %8s  %-26s %-26s" % ("model", "arm", "acc", "vs majority", "vs hybrid"))
    for row in out["contrasts"]:
        cells = []
        for ref_name in ("majority", "hybrid"):
            d = row[ref_name]
            cells.append("%+.4f [%+.4f,%+.4f]%s"
                         % (d["difference"], d["lo"], d["hi"], "*" if d["excludes_zero"] else " "))
        print("%-18s %-9s %8.4f  %-26s %-26s" % (row["model"], row["arm"], row["accuracy"], *cells))
    print("\n* marks an interval excluding zero. Written to %s" % args.out)


if __name__ == "__main__":
    main()
