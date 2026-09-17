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
