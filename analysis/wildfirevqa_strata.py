"""Aerial contrasts stratified by label source and applicability, and the clustering they depend on.

Three reviewers asked the same question: 276 of the 408 reference answers come from multimodal
model verification, so does the ranking survive on the items a sensor labelled? This script
answers it, and reports why the answer cannot be attributed to label source.

Two facts drive the reading:

  Label source is constant within a question id. All 34 question ids draw their answers from one
  source, so the 132 sensor-labelled items and the 276 model-labelled items are disjoint sets of
  question ids. Label source and question type are collinear by construction and no
  within-type comparison exists.

  The design is crossed, 390 frames by 34 question ids, and the majority reference is defined per
  question id. Clustering on frames alone leaves the correlation that matters unmodelled: the
  sensor-labelled stratum spans 7 question ids, not 84 independent draws. This script therefore
  reports each contrast under three resamplings, frame, question id, and the two-way pigeonhole
  bootstrap that resamples both, and prints where they disagree.

The overall arm contrast, grounded minus bare, holds under all three. The model-against-reference
contrasts do not, so the script prints them as sensitive rather than picking the flattering one.

Usage:
    python analysis/wildfirevqa_strata.py
    python analysis/wildfirevqa_strata.py --resamples 20000 --seed 20260915
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
ARMS = ("bare", "grounded")


def load(resamples, seed):
    items = [json.loads(l) for l in (TASK / "items.jsonl").read_text(encoding="utf-8").splitlines()]
    by_id = {i["item_id"]: i for i in items}
    ids = [i["item_id"] for i in items]
    d = {
        "ids": ids,
        "majority": np.array([1.0 if by_id[i]["baseline_majority_correct"] else 0.0 for i in ids]),
        "closed": np.array([by_id[i].get("no_image_answerable") == "closed_form" for i in ids]),
        "rule": np.array([1.0 if by_id[i].get("temp_summary_rule_correct") else 0.0 for i in ids]),
        "frame": np.array([by_id[i]["image"]["image_uid"] for i in ids]),
        "qtype": np.array([by_id[i]["question_id"] for i in ids]),
        "prov": np.array([by_id[i]["answer_provenance"] for i in ids]),
        "app": np.array([float(by_id[i]["applicability_score"]) for i in ids]),
    }
    d["hybrid"] = np.where(d["closed"], d["rule"], d["majority"])
    d["correct"] = {}
    for stem, label in MODELS:
        for arm in ARMS:
            path = TASK / ("responses-%s-%s.jsonl" % (stem, arm))
            rows = {r["item_id"]: r for r in (json.loads(l) for l in path.read_text(encoding="utf-8").splitlines())}
            d["correct"][(label, arm)] = np.array([1.0 if rows[i].get("correct") is True else 0.0 for i in ids])
    return d


def twoway(frame, qtype, a, b, resamples, seed):
    """Pigeonhole bootstrap over both crossed dimensions; each item is weighted by the product of draws."""
    fk, fi = np.unique(frame, return_inverse=True)
    qk, qi = np.unique(qtype, return_inverse=True)
    rng = np.random.default_rng(seed)
    diffs = np.empty(resamples)
    delta = b - a
    for r in range(resamples):
        wf = np.bincount(rng.integers(0, len(fk), len(fk)), minlength=len(fk))
        wq = np.bincount(rng.integers(0, len(qk), len(qk)), minlength=len(qk))
        w = wf[fi] * wq[qi]
        s = w.sum()
        diffs[r] = (w * delta).sum() / s if s else 0.0
    lo, hi, _ = cu.percentile_interval(diffs, 95.0)
    return lo, hi


def intervals(d, mask, ref, stem, arm, label, tag, resamples, seed):
    """One contrast under all three resamplings."""
    c = d["correct"][(label, arm)][mask]
    r = ref[mask]
    out = {"delta": float(c.mean() - r.mean())}
    for name, groups in (("frame", d["frame"][mask]), ("qtype", d["qtype"][mask])):
        lo, hi, _ = wp.paired_bootstrap(list(groups), r, c, resamples,
                                        cu.stable_seed(seed, stem, arm, "%s-%s-%s" % (tag, name, label)))
        out[name] = (float(lo), float(hi))
    lo, hi = twoway(d["frame"][mask], d["qtype"][mask], r, c, resamples,
                    cu.stable_seed(seed, stem, arm, "%s-twoway-%s" % (tag, label)))
    out["twoway"] = (float(lo), float(hi))
    return out


def excludes(pair):
    return pair[0] > 0 or pair[1] < 0


def show(title, d, mask, ref, refname, tag, resamples, seed, out):
    n = int(mask.sum())
    rec = {"items": n, "frames": len(set(d["frame"][mask])), "question_ids": len(set(d["qtype"][mask])),
           "reference": float(ref[mask].mean()), "reference_name": refname, "contrasts": {}}
    print("\n=== %s ===" % title)
    print("    %d items, %d frames, %d question ids, %s reference %.4f"
          % (n, rec["frames"], rec["question_ids"], refname, rec["reference"]))
    print("    %-20s %8s  %-20s %-20s %-20s" % ("", "delta", "by frame", "by question id", "two-way"))
    for stem, label in MODELS:
        for arm in ARMS:
            r = intervals(d, mask, ref, stem, arm, label, tag, resamples, seed)
            rec["contrasts"]["%s/%s" % (label, arm)] = r
            cells = ["[%+.3f,%+.3f]%s" % (r[k][0], r[k][1], "*" if excludes(r[k]) else " ")
                     for k in ("frame", "qtype", "twoway")]
            print("    %-20s %+8.3f  %-20s %-20s %-20s" % (label + "/" + arm[0], r["delta"], *cells))
    sens = [k for k, r in rec["contrasts"].items()
            if excludes(r["frame"]) != excludes(r["twoway"])]
    rec["sensitive_to_clustering"] = sens
    if sens:
        print("    sensitive to the clustering choice: %s" % ", ".join(sens))
    out[title] = rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resamples", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20260915)
    ap.add_argument("--out", type=pathlib.Path, default=HERE / "wildfirevqa_strata.json")
    args = ap.parse_args()
    d = load(args.resamples, args.seed)
    out = {"resamples": args.resamples, "seed": args.seed}

    # Identifiability first: if no question id mixes label sources, the stratification cannot separate
    # label quality from question type, and every number below is descriptive only.
    mixed = [q for q in set(d["qtype"])
             if len({p for p, t in zip(d["prov"], d["qtype"]) if t == q}) > 1]
    out["question_ids_with_mixed_label_source"] = len(mixed)
    print("question ids whose items draw on more than one label source: %d of %d"
          % (len(mixed), len(set(d["qtype"]))))
    print("label source is %s with question type"
          % ("collinear" if not mixed else "partially separable"))

    nonmodel = d["prov"] != "mllm_verified"
    strata = [
        ("all items", np.ones(len(d["ids"]), dtype=bool)),
        ("sensor-labelled", nonmodel),
        ("sensor-labelled, excluding closed form", nonmodel & ~d["closed"]),
        ("model-labelled", d["prov"] == "mllm_verified"),
        ("applicability 1.0", d["app"] >= 1.0),
        ("applicability below 1.0", d["app"] < 1.0),
        ("sensor-labelled and applicability 1.0", nonmodel & (d["app"] >= 1.0)),
    ]
    for title, mask in strata:
        show(title, d, mask, d["majority"], "majority", "strata", args.resamples, args.seed, out)
    show("all items, against the hybrid", d, np.ones(len(d["ids"]), dtype=bool), d["hybrid"],
         "hybrid", "hybrid", args.resamples, args.seed, out)

    print("\n=== grounded minus bare, the direction claim ===")
    arm = {}
    for stem, label in MODELS:
        b, g = d["correct"][(label, "bare")], d["correct"][(label, "grounded")]
        cells, rec = [], {"delta": float(g.mean() - b.mean())}
        for name, groups in (("frame", d["frame"]), ("qtype", d["qtype"])):
            lo, hi, _ = wp.paired_bootstrap(list(groups), b, g, args.resamples,
                                            cu.stable_seed(args.seed, stem, "armdiff", name))
            rec[name] = (float(lo), float(hi))
        rec["twoway"] = tuple(map(float, twoway(d["frame"], d["qtype"], b, g, args.resamples,
                                                cu.stable_seed(args.seed, stem, "armdiff", "twoway"))))
        cells = ["[%+.3f,%+.3f]%s" % (rec[k][0], rec[k][1], "*" if excludes(rec[k]) else " ")
                 for k in ("frame", "qtype", "twoway")]
        print("    %-20s %+8.3f  %-20s %-20s %-20s" % (label, rec["delta"], *cells))
        arm[label] = rec
    out["arm_contrast"] = arm
    print("\n    * marks an interval excluding zero.")

    args.out.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("\nwrote %s" % args.out)


if __name__ == "__main__":
    main()
