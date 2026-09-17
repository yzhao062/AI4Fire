"""Final FIgLib numbers, paired per model, with the answer-failure and cross-run checks the review asked for.

The grounded arm cannot run on the 28 reference frames, so each model is scored on the items both of its
conditions answered. Timestamps are printed because a partial rerun can leave one condition newer than the
other, which pairs cleanly on item_id and silently compares two different runs.
"""
import datetime
import itertools
import json
import pathlib
from math import comb

import numpy as np

T = pathlib.Path(r"C:\Users\yuezh\PycharmProjects\fire-bench\task-figlib")
items = {it["item_id"]: it for it in
         (json.loads(l) for l in (T / "items.jsonl").read_text(encoding="utf-8").splitlines())}
MODELS = ["claude-opus-4.8", "claude-opus-5", "gemini-3.1-pro", "gpt-6-astra", "bedrock_qwen.qwen3-vl-235b-a22b", "bedrock_us.meta.llama4-maverick-17b-instruct-v1_0"]


def load(model, cond):
    f = T / ("responses-%s-%s.jsonl" % (model, cond))
    if not f.exists():
        return None, None
    rows = {}
    for line in f.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        rows[r["item_id"]] = r
    return rows, f.stat().st_mtime


print("%-18s %-9s %5s %8s %s" % ("model", "cond", "rows", "answered", "written"))
data = {}
for m, c in itertools.product(MODELS, ("bare", "grounded")):
    rows, ts = load(m, c)
    if rows is None:
        print("%-18s %-9s  MISSING" % (m, c))
        continue
    ok = {k: v for k, v in rows.items() if v.get("prediction") is not None}
    data[(m, c)] = ok
    print("%-18s %-9s %5d %8d %s" % (m, c, len(rows), len(ok),
                                     datetime.datetime.fromtimestamp(ts).strftime("%m-%d %H:%M")))

print("\n%-18s %5s %9s %8s %8s %6s %s" % ("model / condition", "n", "accuracy", "recall", "FPR", "seqs", "note"))
for m in MODELS:
    if (m, "bare") not in data or (m, "grounded") not in data:
        print("%-18s  incomplete, skipped" % m)
        continue
    shared = sorted(set(data[(m, "bare")]) & set(data[(m, "grounded")]))
    smoke = [i for i in shared if items[i]["label"] == "smoke"]
    nseq = len({items[i]["sequence"] for i in smoke})
    for c in ("bare", "grounded"):
        d = data[(m, c)]
        y = np.array([items[i]["label"] == "smoke" for i in shared])
        p = np.array([bool(d[i]["prediction"]) for i in shared])
        seqs = {items[i]["sequence"] for i in shared
                if items[i]["label"] == "smoke" and d[i]["prediction"]}
        print("%-18s %5d %9.3f %8.3f %8.3f %6s" % (m + " " + c, len(shared), np.mean(p == y),
                                                   np.mean(p[y]), np.mean(p[~y]), "%d/%d" % (len(seqs), nseq)))
    b, g = data[(m, "bare")], data[(m, "grounded")]
    gain = sum(1 for i in smoke if not b[i]["prediction"] and g[i]["prediction"])
    loss = sum(1 for i in smoke if b[i]["prediction"] and not g[i]["prediction"])
    n = gain + loss
    pv = (sum(comb(n, k) for k in range(min(gain, loss) + 1)) / 2 ** n * 2) if n else 1.0
    print("      grounding: %d of %d smoke frames gained, %d lost, sign test p = %.3f"
          % (gain, len(smoke), loss, min(pv, 1.0)))
