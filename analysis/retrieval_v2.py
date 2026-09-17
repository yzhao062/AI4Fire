"""Score the grounded allocation runs under the movement-conditioned analogue rule (rule v2) beside the paper's
bare and grounded (rule v1) runs, on the same 300 items.

For each model the script prints the Table 4 columns for bare, grounded v1, and grounded v2, the paired
incident-cluster bootstrap interval on the normalized-error difference (v2 minus bare, v2 minus v1), and the
copy-agreement count under v2: predictions equal to persistence times the displayed median ratio of the drawn
analogues, rounded to an integer.  It also scores the two analogue-only rules themselves on the 300 items, so
the information handed to the model is stated beside what the model did with it.

    python analysis/retrieval_v2.py
    python analysis/retrieval_v2.py --resamples 20000 --seed 20260915
"""
import argparse
import json
import math
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import cluster_uncertainty as cu  # noqa: E402
import run_allocation as ra  # noqa: E402

TASK = ROOT / "task-allocation"
MODELS = ["claude-opus-4.8", "claude-opus-5", "gemini-3.1-pro", "gpt-6-astra",
          "bedrock_qwen.qwen3-vl-235b-a22b", "bedrock_us.meta.llama4-maverick-17b-instruct-v1_0"]
HEADER = "%-46s %6s %7s %6s %7s %7s %6s %6s %6s" % ("run", "mae", "nmae", "beatp", "stable", "moving", "false", "missed", "dirok")


def load(path):
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return {r["item_id"]: r for r in rows if r.get("prediction") is not None}


def metrics(rows, ids):
    y = np.array([rows[i]["target"] for i in ids], float)
    p = np.array([rows[i]["prediction"] for i in ids], float)
    b = np.array([rows[i]["persistence"] for i in ids], float)
    s = np.array([rows[i]["fire_mean"] for i in ids], float)
    moved = np.abs(y - b) / np.maximum(b, 1) > 0.1
    pm = np.abs(p - b) / np.maximum(b, 1) > 0.1
    err = np.abs(y - p) / s
    berr = np.abs(y - b) / s
    direction = np.sign(y - b) == np.sign(p - b)
    return {"mae": float(np.mean(np.abs(y - p))), "nmae": float(np.mean(err)), "beatp": float(np.mean(err < berr)),
            "stable": float(np.mean(err[~moved])), "moving": float(np.mean(err[moved])),
            "false": float(np.mean(pm[~moved])), "missed": float(np.mean(~pm[moved])),
            "dirok": float(np.mean(direction[moved])), "err": err}


def fmt(label, m):
    return "%-46s %6.2f %7.4f %6.3f %7.4f %7.4f %6.3f %6.3f %6.3f" % (
        label, m["mae"], m["nmae"], m["beatp"], m["stable"], m["moving"], m["false"], m["missed"], m["dirok"])


def paired(err_a, err_b, groups, resamples, seed):
    """Interval on mean(err_b) - mean(err_a) over resampled incidents."""
    flat, starts, sizes, keys = cu.build_cluster_index(groups)
    rng = np.random.default_rng(seed)
    diffs = np.empty(resamples)
    for r in range(resamples):
        idx = cu.ragged_gather(flat, starts, sizes, rng.integers(0, len(keys), size=len(keys)))
        diffs[r] = float(np.mean(err_b[idx]) - np.mean(err_a[idx]))
    lo, hi, _ = cu.percentile_interval(diffs, 95.0)
    return lo, hi


def rule_rows(items, draws, flat, fire_mean):
    """The analogue-only prediction of a draw, as a response-row dict per item."""
    out = {}
    for it in items:
        ids = draws.get(it["item_id"], [])
        if not ids:
            continue
        ratios = [flat[a][1] / max(flat[a][0], 1) for a in ids]
        median = float(np.median(ratios))
        # Table 4's analogue-only row uses the exact median, rounded half up to a head count, as
        # analysis/information_only_baselines.py does; the copy-agreement count below uses the two-decimal
        # value the prompt displays, as copy_agreement.py does.
        out[it["item_id"]] = {"prediction": math.floor(it["baseline_persistence"] * median + 0.5), "target": it["target_personnel"],
                              "persistence": it["baseline_persistence"], "fire_mean": fire_mean[it["incident_id"]],
                              "median_ratio": median}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resamples", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20260915)
    args = ap.parse_args()
    items = ra.sample_items()
    fire_mean = {}
    allrows = [json.loads(l) for l in (TASK / "items.jsonl").read_text(encoding="utf-8").splitlines()]
    acc = {}
    for r in allrows:
        acc.setdefault(r["incident_id"], []).append(r["target_personnel"])
    fire_mean = {k: float(np.mean(v)) for k, v in acc.items()}
    pool = ra.build_pool({i["incident_id"] for i in items})
    flat = {r["analogue_id"]: (r["today"], r["next"]) for rows in pool.values() for r in rows}
    incident = {i["item_id"]: i["incident_id"] for i in items}
    out = {}

    print(HEADER)
    # The two analogue-only rules on the 300 items, from the stored draws of any model's grounded file.
    for rule, pattern in (("v1", "responses-claude-opus-5-grounded.jsonl"), ("v2", "responses-claude-opus-5-grounded-v2.jsonl")):
        path = TASK / pattern
        if not path.exists():
            continue
        draws = {r["item_id"]: r.get("analogue_ids", []) for r in (json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip())}
        rr = rule_rows(items, draws, flat, fire_mean)
        ids = sorted(rr)
        m = metrics(rr, ids)
        med = np.array([rr[i]["median_ratio"] for i in ids])
        actual = np.array([rr[i]["target"] / max(rr[i]["persistence"], 1) for i in ids])
        print(fmt("analogue-only rule %s (%d items)" % (rule, len(ids)), m)
              + "   median-ratio IQR [%.2f, %.2f] actual [%.2f, %.2f]" % (np.percentile(med, 25), np.percentile(med, 75), np.percentile(actual, 25), np.percentile(actual, 75)))
        out["rule-" + rule] = {k: v for k, v in m.items() if k != "err"}
        out["rule-" + rule]["items"] = len(ids)
        out["rule-" + rule]["median_ratio_iqr"] = [float(np.percentile(med, 25)), float(np.percentile(med, 75))]

    for model in MODELS:
        files = {"bare": TASK / ("responses-%s-bare.jsonl" % model), "v1": TASK / ("responses-%s-grounded.jsonl" % model),
                 "v2": TASK / ("responses-%s-grounded-v2.jsonl" % model)}
        if not files["v2"].exists():
            print("%s: no v2 file yet" % model)
            continue
        rows = {k: load(p) for k, p in files.items()}
        ids = sorted(set(rows["bare"]) & set(rows["v1"]) & set(rows["v2"]))
        groups = [incident[i] for i in ids]
        ms = {k: metrics(rows[k], ids) for k in rows}
        print()
        for k in ("bare", "v1", "v2"):
            print(fmt("%s %s" % (model, {"bare": "bare", "v1": "grounded v1", "v2": "grounded v2"}[k]), ms[k]))
        d_bare = ms["v2"]["nmae"] - ms["bare"]["nmae"]
        d_v1 = ms["v2"]["nmae"] - ms["v1"]["nmae"]
        lo1, hi1 = paired(ms["bare"]["err"], ms["v2"]["err"], groups, args.resamples, cu.stable_seed(args.seed, model, "v2-bare"))
        lo2, hi2 = paired(ms["v1"]["err"], ms["v2"]["err"], groups, args.resamples, cu.stable_seed(args.seed, model, "v2-v1"))
        # copy agreement under v2
        same = 0
        n = 0
        for i in ids:
            r = rows["v2"][i]
            if not r.get("analogue_ids"):
                continue
            ratios = [flat[a][1] / max(flat[a][0], 1) for a in r["analogue_ids"]]
            shown = float("%.2f" % float(np.median(ratios)))
            n += 1
            same += int(r["prediction"] == int(round(r["persistence"] * shown)))
        print("  %d items, %d incidents: v2 - bare %+.4f [%+.4f, %+.4f]; v2 - v1 %+.4f [%+.4f, %+.4f]; equals displayed-median rule on %d of %d (%.3f)"
              % (len(ids), len(set(groups)), d_bare, lo1, hi1, d_v1, lo2, hi2, same, n, same / max(n, 1)))
        out[model] = {k: {kk: vv for kk, vv in ms[k].items() if kk != "err"} for k in ms}
        out[model].update({"items": len(ids), "d_v2_bare": d_bare, "d_v2_bare_ci": [lo1, hi1], "d_v2_v1": d_v1, "d_v2_v1_ci": [lo2, hi2],
                           "copy_same": same, "copy_n": n})
    (HERE / "retrieval_v2.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
