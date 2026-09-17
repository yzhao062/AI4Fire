"""Re-query the allocation items whose rule-v2 grounded call failed with a transport error, and replace those rows.

Only rows whose stored record carries an error and no prediction are replaced. Each replaced row is appended to
task-allocation/requery-log-v2.jsonl with the model, so the failure stays on record. The prompt is rebuilt from the
same v2 draw the run used (RuleV2 is deterministic), and the fresh row stores the same analogue_ids.

Usage: python requery_allocation_v2.py <model>            # e.g. claude-opus-4.8, bedrock:qwen.qwen3-vl-235b-a22b
"""
import collections
import json
import re
import sys

import numpy as np

import gw
import run_allocation as ra

model = sys.argv[1]
safe = re.sub(r"[^A-Za-z0-9._-]", "_", model)
path = ra.TASK / ("responses-%s-grounded-v2.jsonl" % safe)
rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
todo = [i for i, r in enumerate(rows) if r.get("error") and r.get("prediction") is None]
print("%s: %d rows to re-query" % (path.name, len(todo)))
if not todo:
    raise SystemExit(0)

items = {it["item_id"]: it for it in ra.sample_items()}
fire_mean = collections.defaultdict(list)
for line in (ra.TASK / "items.jsonl").read_text(encoding="utf-8").splitlines():
    row = json.loads(line)
    fire_mean[row["incident_id"]].append(row["target_personnel"])
fire_mean = {k: float(np.mean(v)) for k, v in fire_mean.items()}
pool = ra.build_pool({i["incident_id"] for i in items.values()})
v2 = ra.RuleV2(pool)
key = gw.load_key()
for i in todo:
    old = rows[i]
    it = items[old["item_id"]]
    ana = v2.draw(it)
    msgs = ra.render(it, ra.grounded_block(ana, "v2"))
    text, usage, served = gw.call(key, model, msgs, max_tokens=ra.MAX_OUT)
    new = {"item_id": it["item_id"], "raw": text, "usage": usage, "served_model": served, "prediction": ra.parse(text),
           "target": it["target_personnel"], "persistence": it["baseline_persistence"], "fire_mean": fire_mean[it["incident_id"]],
           "analogue_ids": [r["analogue_id"] for r in ana], "rule": "v2", "requeried": True}
    with (ra.TASK / "requery-log-v2.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"model": model, "replaced": old}) + "\n")
    rows[i] = new
    print("  %s -> prediction %s" % (it["item_id"], new["prediction"]))
path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
print("rewrote", path.name)
