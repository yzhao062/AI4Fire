"""Does the tool-arm ceiling survive rewording the 156 questions?

The tool arm looks saturated: four proprietary models make one error across 624 answers. A
reviewer's objection is that the 156 questions come from twelve deterministic templates over one
frozen snapshot, so a perfect score may show template familiarity rather than language
understanding. Two rewrites test it, each changing one thing:

  p1  surface phrasing rewritten, every COLUMN = 'value' predicate kept verbatim.
      Isolates whether the wording of the carrier sentence matters.

  p2  original phrasing kept, schema literals replaced by natural language
      ("STATE = 'AK'" becomes "Alaska"). Isolates whether the ceiling depends on being handed
      the column names and literal values. The schema still reaches the model in the system
      prompt, so nothing the model needs is withheld; it must do the mapping itself.

Answers, tolerances, and reference queries are byte-identical across all three sets, so the
existing scorer applies unchanged and the contrasts are paired per item. Intervals cluster over
the twelve question families, matching analysis/tooluse_paired.py.

Usage:
    python analysis/tooluse_paraphrase.py
    python analysis/tooluse_paraphrase.py --resamples 20000 --seed 20260915
"""
import argparse
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import cluster_uncertainty as cu  # noqa: E402

TASK = HERE.parent / "task-tooluse"

MODELS = [
    ("claude-opus-4.8", "claude-opus-4.8"),
    ("claude-opus-5", "claude-opus-5"),
    ("gemini-3.1-pro", "gemini-3.1-pro"),
    ("gpt-6-astra", "gpt-6-astra"),
    ("bedrock_qwen.qwen3-vl-235b-a22b", "Qwen3-VL"),
    ("bedrock_us.meta.llama4-maverick-17b-instruct-v1_0", "Llama 4 Maverick"),
]
VARIANTS = [("p0", ""), ("p1", "-p1"), ("p2", "-p2")]


def paired_bootstrap(groups, a, b, resamples, seed):
    flat, starts, sizes, keys = cu.build_cluster_index(groups)
    rng = np.random.default_rng(seed)
    diffs = np.empty(resamples)
    for r in range(resamples):
        draw = rng.integers(0, len(keys), size=len(keys))
        idx = cu.ragged_gather(flat, starts, sizes, draw)
        diffs[r] = b[idx].mean() - a[idx].mean()
    lo, hi, _ = cu.percentile_interval(diffs, 95.0)
    return lo, hi, len(keys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resamples", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20260915)
    ap.add_argument("--out", type=pathlib.Path, default=HERE / "tooluse_paraphrase.json")
    args = ap.parse_args()

    items = [json.loads(l) for l in (TASK / "items.jsonl").read_text(encoding="utf-8").splitlines()]
    ids = [i["item_id"] for i in items]
    fam = [i["family"] for i in items]
    by_id = {i["item_id"]: i for i in items}

    correct, missing = {}, []
    for stem, label in MODELS:
        for tag, suffix in VARIANTS:
            path = TASK / ("responses-%s-tool%s.jsonl" % (stem, suffix))
            if not path.exists():
                missing.append(path.name)
                continue
            rows = {r["item_id"]: r for r in (json.loads(l) for l in path.read_text(encoding="utf-8").splitlines())}
            if len(rows) != len(ids):
                missing.append("%s (%d of %d rows)" % (path.name, len(rows), len(ids)))
                continue
            correct[(label, tag)] = np.array([1.0 if rows[i].get("correct") is True else 0.0 for i in ids])
    if missing:
        print("not scored, absent or short: %s\n" % ", ".join(missing))

    out = {"resamples": args.resamples, "seed": args.seed, "items": len(ids),
           "families": len(set(fam)), "models": {}}
    print("%-20s %8s %8s %8s   %-24s %-24s" % ("model", "p0", "p1", "p2", "p1 - p0", "p2 - p0"))
    for stem, label in MODELS:
        acc, cells, rec = {}, [], {}
        for tag, _ in VARIANTS:
            acc[tag] = correct[(label, tag)].mean() if (label, tag) in correct else None
        for tag in ("p1", "p2"):
            if acc[tag] is None or acc["p0"] is None:
                cells.append("%-24s" % "")
                continue
            d = acc[tag] - acc["p0"]
            lo, hi, k = paired_bootstrap(fam, correct[(label, "p0")], correct[(label, tag)],
                                         args.resamples, cu.stable_seed(args.seed, stem, "tool-" + tag))
            star = "*" if (lo > 0 or hi < 0) else " "
            cells.append("%+.4f [%+.3f,%+.3f]%s" % (d, lo, hi, star))
            rec[tag + "_minus_p0"] = {"d": float(d), "ci": [float(lo), float(hi)], "clusters": k}
        rec.update({t: (float(acc[t]) if acc[t] is not None else None) for t, _ in VARIANTS})
        out["models"][label] = rec
        fmt = lambda v: ("%.4f" % v) if v is not None else "  --  "
        print("%-20s %8s %8s %8s   %-24s %-24s"
              % (label, fmt(acc["p0"]), fmt(acc["p1"]), fmt(acc["p2"]), cells[0], cells[1]))

    print("\n* interval excludes zero; intervals cluster over the %d question families." % len(set(fam)))

    # Where the errors land, by family, so a drop can be read rather than only counted.
    for tag in ("p1", "p2"):
        # Compare against p0 on the same models only, so a partial run cannot read as a drop.
        paired = [label for _, label in MODELS if (label, tag) in correct and (label, "p0") in correct]
        if not paired:
            continue
        print("\nerrors by family on %s, over the %d models with both arms scored (13 items each)"
              % (tag, len(paired)))
        for f in sorted(set(fam)):
            m = np.array([x == f for x in fam])
            base = sum(int(13 - correct[(l, "p0")][m].sum()) for l in paired)
            now = sum(int(13 - correct[(l, tag)][m].sum()) for l in paired)
            if base or now:
                print("   %-26s p0 %2d -> %s %2d" % (f, base, tag, now))

    args.out.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("\nwrote %s" % args.out)


if __name__ == "__main__":
    main()
