"""Paired tool-against-bare comparison on the FPA-FOD tool-use task (156 items, 12 question families).

Both arms of run_tooluse.py answer the same 156 questions; the bare arm answers from memory and the tool arm may
query the FPA-FOD SQLite through one read-only SQL tool.  This script scores the two arms with the runner's own
score_item(), reports accuracy per arm and per tier, the paired difference in accuracy with a cluster bootstrap
interval over question family (12 clusters, the same percentile machinery and seed as cluster_uncertainty.py),
the abstention and answer-shape counts behind the residual errors, and the per-family best-constant baseline the
task ships in naive-baseline.csv.

    python analysis/tooluse_paired.py
    python analysis/tooluse_paired.py --resamples 20000 --seed 20260915
"""
import argparse
import collections
import json
import pathlib
import statistics
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import cluster_uncertainty as cu  # noqa: E402
import run_tooluse as rt  # noqa: E402

TASK = ROOT / "task-tooluse"
# response-file stems, the file-safe spellings run_tooluse.py writes
MODELS = {
    "bedrock_qwen.qwen3-vl-235b-a22b": "Qwen3-VL",
    "bedrock_us.meta.llama4-maverick-17b-instruct-v1_0": "Llama 4 Maverick",
    "claude-opus-4.8": "claude-opus-4.8",
    "claude-opus-5": "claude-opus-5",
    "gemini-3.1-pro": "gemini-3.1-pro",
    "gpt-6-astra": "gpt-6-astra",
}
CONDITIONS = ["bare", "tool"]


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


def used_column(row, column):
    return any(column in (c.get("sql") or "") for c in (row.get("tool_calls") or []))


def code_like(raw):
    first = (raw or "").strip().split("\n")[0]
    return "print(" in (raw or "") or ("=" in first and "ANSWER" not in first)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resamples", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20260915)
    args = ap.parse_args()
    items = [json.loads(l) for l in (TASK / "items.jsonl").read_text(encoding="utf-8").splitlines()]
    by_id = {i["item_id"]: i for i in items}
    naive_by_fam, naive_overall = rt.load_naive_baseline()
    out = {"naive_baseline_overall": naive_overall, "naive_baseline_by_family": naive_by_fam, "models": {}}
    for stem, name in MODELS.items():
        files = {c: TASK / ("responses-%s-%s.jsonl" % (stem, c)) for c in CONDITIONS}
        if not all(p.exists() for p in files.values()):
            print("%s: arm files missing, skipped" % name)
            continue
        arms = {c: load(p) for c, p in files.items()}
        ids = [i["item_id"] for i in items if all(i["item_id"] in a for a in arms.values())]
        groups = [by_id[i]["family"] for i in ids]
        correct = {c: np.array([1.0 if arms[c][i].get("correct") is True else 0.0 for i in ids]) for c in CONDITIONS}
        rec = {"items": len(ids)}
        print("\n== %s, %d items, %d families" % (name, len(ids), len(set(groups))))
        for c in CONDITIONS:
            rows = [arms[c][i] for i in ids]
            tiers = sorted(set(by_id[i]["tier"] for i in ids))
            by_tier = {t: float(np.mean([correct[c][k] for k, i in enumerate(ids) if by_id[i]["tier"] == t])) for t in tiers}
            abst = sum(1 for r in rows if r.get("abstained") is True)
            parse = sum(1 for r in rows if r.get("failure") == "parse")
            calls = [r.get("n_tool_calls", 0) for r in rows]
            rel = []
            for r in rows:
                it = by_id[r["item_id"]]
                if it["answer_type"] in ("integer", "acres", "percent") and isinstance(r.get("prediction"), (int, float)) and it["answer"]:
                    rel.append(abs(r["prediction"] - it["answer"]) / abs(it["answer"]))
            rec[c] = {"accuracy": float(correct[c].mean()), "by_tier": by_tier, "abstentions": abst, "parse_failures": parse,
                      "mean_tool_calls": statistics.fmean(calls) if calls else 0.0, "max_tool_calls": max(calls, default=0),
                      "numeric_answered": len(rel), "numeric_median_rel_error": statistics.median(rel) if rel else None,
                      "numeric_within_10pct": sum(1 for x in rel if x <= 0.1)}
            print("  %s  acc %.3f  tiers %s  abstained %d  parse failures %d  tool calls mean %.2f max %d  numeric answered %d, median rel. error %s, within 10%% %d"
                  % (c, rec[c]["accuracy"], {t: round(v, 3) for t, v in by_tier.items()}, abst, parse, rec[c]["mean_tool_calls"], rec[c]["max_tool_calls"],
                     len(rel), "%.2f" % statistics.median(rel) if rel else "n/a", rec[c]["numeric_within_10pct"]))
        d = float(correct["tool"].mean() - correct["bare"].mean())
        lo, hi, k = paired_bootstrap(groups, correct["bare"], correct["tool"], args.resamples, cu.stable_seed(args.seed, stem, "tool-bare"))
        rec["tool_minus_bare"] = {"d": d, "ci": [lo, hi], "clusters": k, "resamples": args.resamples}
        print("  tool - bare: %+.3f [%+.3f, %+.3f] over %d family clusters" % (d, lo, hi, k))
        # residual errors in the tool arm, by family and shape
        wrong = [arms["tool"][i] for i in ids if arms["tool"][i].get("correct") is not True]
        fam_counts = collections.Counter(r["family"] for r in wrong)
        rec["tool_errors"] = {}
        for fam, n in fam_counts.most_common():
            sub = [r for r in wrong if r["family"] == fam]
            allfam = [arms["tool"][i] for i in ids if by_id[i]["family"] == fam]
            rec["tool_errors"][fam] = {"wrong": n, "of": len(allfam),
                                      "predicted_zero": sum(1 for r in sub if r.get("prediction") in (0, 0.0)),
                                      "abstained": sum(1 for r in sub if r.get("abstained")),
                                      "code_like_answer": sum(1 for r in sub if code_like(r.get("raw"))),
                                      "filtered_on_discovery_date": sum(1 for r in sub if used_column(r, "DISCOVERY_DATE")),
                                      "family_used_doy": sum(1 for r in allfam if used_column(r, "DISCOVERY_DOY"))}
            print("  tool errors in %s: %d of %d (zero %d, abstained %d, code-like %d, filtered on DISCOVERY_DATE %d; family rows using DISCOVERY_DOY %d)"
                  % (fam, n, len(allfam), rec["tool_errors"][fam]["predicted_zero"], rec["tool_errors"][fam]["abstained"], rec["tool_errors"][fam]["code_like_answer"],
                     rec["tool_errors"][fam]["filtered_on_discovery_date"], rec["tool_errors"][fam]["family_used_doy"]))
        out["models"][name] = rec
    (HERE / "tooluse_paired.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("\nwrote", HERE / "tooluse_paired.json")


if __name__ == "__main__":
    main()
