"""Compare each repeated bare run against its first run, item by item, on the items both answered.

FIgLib: how many frame verdicts flipped, and accuracy/recall on each run.
Mesogeos: how many boolean calls flipped, mean absolute change in probability, AUPRC of each run.
Allocation: mean absolute change in the predicted count, normalized error of each run.
"""
import json
import pathlib

import numpy as np
from sklearn.metrics import average_precision_score

R = pathlib.Path(__file__).parent / "repeat"


def load(p):
    return {r["item_id"]: r for r in (json.loads(l) for l in p.read_text(encoding="utf-8").splitlines())}


a, b = load(R / "figlib-qwen3-vl-bare-run1.jsonl"), load(R / "figlib-qwen3-vl-bare-run2.jsonl")
ids = [i for i in a if i in b and a[i].get("prediction") is not None and b[i].get("prediction") is not None]
flips = sum(a[i]["prediction"] != b[i]["prediction"] for i in ids)
for name, d in (("run1", a), ("run2", b)):
    lab = np.array([d[i]["label"] == "smoke" for i in ids]); pred = np.array([bool(d[i]["prediction"]) for i in ids])
    print("figlib qwen bare %s: n=%d acc %.3f recall %.3f fpr %.3f" % (name, len(ids), (lab == pred).mean(), pred[lab].mean(), pred[~lab].mean()))
print("figlib qwen bare: verdict flips %d of %d" % (flips, len(ids)))

a, b = load(R / "mesogeos-qwen3-vl-bare-run1.jsonl"), load(R / "mesogeos-qwen3-vl-bare-run2.jsonl")
ids = [i for i in a if i in b and a[i].get("probability") is not None and b[i].get("probability") is not None]
y = np.array([a[i]["label"] for i in ids])
pa = np.array([a[i]["probability"] for i in ids]); pb = np.array([b[i]["probability"] for i in ids])
ca = np.array([a[i]["call"] if a[i]["call"] is not None else a[i]["probability"] >= 0.5 for i in ids])
cb = np.array([b[i]["call"] if b[i]["call"] is not None else b[i]["probability"] >= 0.5 for i in ids])
print("mesogeos qwen bare: n=%d auprc run1 %.3f run2 %.3f | call flips %d | mean |dp| %.3f | items with |dp|>0.1: %d"
      % (len(ids), average_precision_score(y, pa), average_precision_score(y, pb), (ca != cb).sum(), np.abs(pa - pb).mean(), (np.abs(pa - pb) > 0.1).sum()))

a = load(R / "allocation-llama4-maverick-bare-run1.jsonl")
b = load(pathlib.Path(__file__).parent / "task-allocation" / "responses-bedrock_us.meta.llama4-maverick-17b-instruct-v1_0-bare.jsonl")
ids = [i for i in a if i in b and a[i].get("prediction") is not None and b[i].get("prediction") is not None]
y = np.array([a[i]["target"] for i in ids]); s = np.array([a[i]["fire_mean"] for i in ids])
pa = np.array([a[i]["prediction"] for i in ids]); pb = np.array([b[i]["prediction"] for i in ids])
print("allocation llama bare: n=%d nMAE run1 %.3f run2 %.3f | mean |dpred| %.2f persons | items changed %d"
      % (len(ids), (np.abs(y - pa) / s).mean(), (np.abs(y - pb) / s).mean(), np.abs(pa - pb).mean(), (pa != pb).sum()))

# gpt-6-astra runs at its endpoint's default temperature of 1 (the only value it accepts), so its repeat is the
# one measurement of run-to-run variation at that setting. The main file is run 1; the repeat is run 2.
a = load(R / "allocation-gpt-6-astra-bare-run1.jsonl")
b = load(R / "allocation-gpt-6-astra-bare-run2.jsonl")
ids = [i for i in a if i in b and a[i].get("prediction") is not None and b[i].get("prediction") is not None]
y = np.array([a[i]["target"] for i in ids]); s = np.array([a[i]["fire_mean"] for i in ids])
pa = np.array([a[i]["prediction"] for i in ids]); pb = np.array([b[i]["prediction"] for i in ids])
print("allocation gpt-6-astra bare (temperature 1): n=%d nMAE run1 %.4f run2 %.4f | mean |dpred| %.2f persons | items changed %d"
      % (len(ids), (np.abs(y - pa) / s).mean(), (np.abs(y - pb) / s).mean(), np.abs(pa - pb).mean(), (pa != pb).sum()))

# Second bare Mesogeos runs at temperature zero for claude-opus-5 and gemini-3.1-pro (2026-09-17 22:34, the same
# prompt p0 and the same 386 items), so the paraphrase differences of analysis/prompt_sensitivity.py can be read
# against run-to-run variation on the same two models. The paired interval on AUPRC(run2) - AUPRC(run1) uses the
# same 1-degree-block-by-month cluster bootstrap, seed, and resample count as the paraphrase intervals.
import sys as _sys
_sys.path.insert(0, str(pathlib.Path(__file__).parent / "analysis"))
import cluster_uncertainty as cu  # noqa: E402

items = [json.loads(l) for l in (pathlib.Path(__file__).parent / "task-mesogeos" / "items.jsonl").read_text(encoding="utf-8").splitlines()]
ctx = {i["item_id"]: i["context"] for i in items}
for model in ("claude-opus-5", "gemini-3.1-pro"):
    a, b = load(R / ("mesogeos-%s-bare-run1.jsonl" % model)), load(R / ("mesogeos-%s-bare-run2.jsonl" % model))
    ids = [i for i in a if i in b and a[i].get("probability") is not None and b[i].get("probability") is not None]
    y = np.array([a[i]["label"] for i in ids], dtype=float)
    pa = np.array([cu.mesogeos_score(a[i]) for i in ids]); pb = np.array([cu.mesogeos_score(b[i]) for i in ids])
    ca = np.array([a[i]["call"] if a[i]["call"] is not None else a[i]["probability"] >= 0.5 for i in ids])
    cb = np.array([b[i]["call"] if b[i]["call"] is not None else b[i]["probability"] >= 0.5 for i in ids])
    groups = [cu.mesogeos_block_key(ctx[i].get("longitude"), ctx[i].get("latitude"), ctx[i].get("target_date"), 1.0, "month") for i in ids]
    flat, starts, sizes, keys = cu.build_cluster_index(groups)
    rng = np.random.default_rng(cu.stable_seed(20260915, model, "repeat", "auprc"))
    diffs = np.empty(20000)
    for r in range(20000):
        draw = rng.integers(0, len(keys), size=len(keys))
        idx = cu.ragged_gather(flat, starts, sizes, draw)
        diffs[r] = cu.average_precision(y[idx], pb[idx]) - cu.average_precision(y[idx], pa[idx])
    lo, hi, _ = cu.percentile_interval(diffs, 95.0)
    omitted = [sum(1 for i in ids if d[i].get("call") is None) for d in (a, b)]
    print("mesogeos %s bare (temperature 0): n=%d auprc run1 %.3f run2 %.3f | run2 - run1 %+.3f [%+.3f, %+.3f] over %d blocks | call flips %d | mean |dp| %.3f | items with |dp|>0.1: %d | omitted %d / %d | call rate %.3f / %.3f"
          % (model, len(ids), cu.average_precision(y, pa), cu.average_precision(y, pb), cu.average_precision(y, pb) - cu.average_precision(y, pa), lo, hi, len(keys),
             (ca != cb).sum(), np.abs(pa - pb).mean(), (np.abs(pa - pb) > 0.1).sum(), omitted[0], omitted[1], ca.mean(), cb.mean()))
