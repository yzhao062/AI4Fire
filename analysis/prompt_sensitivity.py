"""Prompt-sensitivity check on fire danger forecasting (Mesogeos Track A, 386 items).

Two paraphrases of the paper's prompt, p1 and p2 in run_mesogeos.py, carry the same numbers and the same
answer schema and differ in framing, field order, and layout.  Each was run once, bare, on the two leading
proprietary models (2026-09-16), on the two open-weight models through Bedrock (2026-09-17), and on the other two
proprietary models (2026-09-17 evening), so every model of the paper has all three prompts.
This script scores every variant file with the runner's own score() and reports, per model, the change
in AUPRC, fire-class F1, and call rate from the paper's prompt (p0), with a paired block-by-month cluster
bootstrap interval on the AUPRC and F1 differences, the same clustering unit and seed as
cluster_uncertainty.py.  It also reports how far the two probability rankings agree item by item.

    python analysis/prompt_sensitivity.py
    python analysis/prompt_sensitivity.py --resamples 20000 --seed 20260915
"""
import argparse
import json
import os
import pathlib
import sys

import numpy as np
from scipy import stats

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import cluster_uncertainty as cu  # noqa: E402
import run_mesogeos as rm  # noqa: E402
import models
from models import add_model_args, resolve_models

TASK = ROOT / "task-mesogeos"
# response-file stems; the Bedrock stems are the file-safe spellings run_mesogeos.py writes
MODELS = [m.stem for m in models.models(tier="core")]
VARIANTS = ["p0", "p1", "p2"]


def load(path):
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
    return {r["item_id"]: r for r in rows}


def aligned(items, by_variant):
    """Arrays over the items every variant file answered, in items.jsonl order."""
    ids = [i["item_id"] for i in items if all(i["item_id"] in v for v in by_variant.values())]
    y = np.array([by_variant["p0"][i]["label"] for i in ids], dtype=float)
    groups = []
    for i in ids:
        c = next(it for it in items if it["item_id"] == i)["context"]
        groups.append(cu.mesogeos_block_key(c.get("longitude"), c.get("latitude"), c.get("target_date"), 1.0, "month"))
    prob = {v: np.array([cu.mesogeos_score(by_variant[v][i]) for i in ids], dtype=float) for v in by_variant}
    call = {v: np.array([by_variant[v][i]["call"] if by_variant[v][i]["call"] is not None
                         else (by_variant[v][i]["probability"] or 0) >= 0.5 for i in ids], dtype=bool) for v in by_variant}
    return ids, y, groups, prob, call


def f1_fire(y, call):
    tp = float(np.sum(call & (y == 1)))
    fp = float(np.sum(call & (y == 0)))
    fn = float(np.sum(~call & (y == 1)))
    return 2 * tp / max(2 * tp + fp + fn, 1e-12)


def paired_bootstrap(y, groups, a, b, kind, resamples, seed):
    """Interval on metric(b) - metric(a), both arms scored on the same resampled clusters."""
    flat, starts, sizes, keys = cu.build_cluster_index(groups)
    rng = np.random.default_rng(seed)
    diffs = np.empty(resamples)
    for r in range(resamples):
        draw = rng.integers(0, len(keys), size=len(keys))
        idx = cu.ragged_gather(flat, starts, sizes, draw)
        if kind == "auprc":
            diffs[r] = cu.average_precision(y[idx], b[idx]) - cu.average_precision(y[idx], a[idx])
        else:
            diffs[r] = f1_fire(y[idx], b[idx]) - f1_fire(y[idx], a[idx])
    lo, hi, _ = cu.percentile_interval(diffs, 95.0)
    return lo, hi, len(keys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resamples", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20260915)
    ap.add_argument("--out", type=pathlib.Path, default=HERE / "prompt_sensitivity.json")
    add_model_args(ap)
    args = ap.parse_args()
    items = [json.loads(l) for l in (TASK / "items.jsonl").read_text(encoding="utf-8").splitlines()]
    items = [i for i in items if i["split"] == "test" and i.get("fold", 0) == 0]
    out = {}
    selected_models = [m.stem for m in resolve_models(args)]
    for model in selected_models:
        by_variant = {}
        for v in VARIANTS:
            path = TASK / ("responses-%s-bare%s.jsonl" % (model, "" if v == "p0" else "-" + v))
            if path.exists():
                by_variant[v] = load(path)
        if "p0" not in by_variant or len(by_variant) < 2:
            print("%s: variant files missing, skipped" % model)
            continue
        print("\n== %s, bare, %s" % (model, ", ".join(sorted(by_variant))))
        for v in sorted(by_variant):
            rows = list(by_variant[v].values())
            s = rm.score(rows, v)
            omitted = sum(1 for r in rows if r.get("call") is None and r.get("probability") is not None)
            print("  %s  n=%d  auprc %.3f  f1 %.3f  call %.3f  omitted %d" % (v, s["items"], s["auprc"], s["f1_fire"], s["positive_rate_called"], omitted))
        ids, y, groups, prob, call = aligned(items, by_variant)
        out[model] = {}
        for v in sorted(by_variant):
            if v == "p0":
                continue
            rho = stats.spearmanr(prob["p0"], prob[v]).correlation
            changed = int(np.sum(call["p0"] != call[v]))
            d_auprc = cu.average_precision(y, prob[v]) - cu.average_precision(y, prob["p0"])
            d_f1 = f1_fire(y, call[v]) - f1_fire(y, call["p0"])
            lo_a, hi_a, k = paired_bootstrap(y, groups, prob["p0"], prob[v], "auprc", args.resamples, cu.stable_seed(args.seed, model, v, "auprc"))
            lo_f, hi_f, _ = paired_bootstrap(y, groups, call["p0"], call[v], "f1", args.resamples, cu.stable_seed(args.seed, model, v, "f1"))
            print("  %s - p0 on %d items, %d blocks: AUPRC %+.3f [%+.3f, %+.3f]; fire F1 %+.3f [%+.3f, %+.3f]; Spearman(prob) %.3f; calls changed %d"
                  % (v, len(ids), k, d_auprc, lo_a, hi_a, d_f1, lo_f, hi_f, rho, changed))
            out[model][v] = {"items": len(ids), "blocks": k, "d_auprc": d_auprc, "d_auprc_ci": [lo_a, hi_a],
                             "d_f1": d_f1, "d_f1_ci": [lo_f, hi_f], "spearman": float(rho), "calls_changed": changed}
    args.out.write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
