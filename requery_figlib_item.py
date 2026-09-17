"""Re-query one FIgLib item whose call failed with a transport error, and replace its row in the response file.

The replaced row is appended to task-figlib/requery-log.jsonl with the model and condition, so the failure stays
on record. Only rows whose stored record carries an error and no answer are replaced.

Usage: python requery_figlib_item.py <model> <bare|grounded> <item_id>
"""
import json
import re
import sys

import gw
import run_figlib as rf

model, cond, item_id = sys.argv[1:4]
safe = re.sub(r"[^A-Za-z0-9._-]", "_", model)
path = rf.TASK / ("responses-%s-%s.jsonl" % (safe, cond))
rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
idx = [i for i, r in enumerate(rows) if r["item_id"] == item_id]
assert len(idx) == 1, idx
old = rows[idx[0]]
assert old.get("error") and old.get("prediction") is None, "row has an answer already"
items = {json.loads(l)["item_id"]: json.loads(l) for l in (rf.TASK / "items.jsonl").read_text(encoding="utf-8").splitlines()}
it = items[item_id]
reference = None
if cond == "grounded":
    raise SystemExit("grounded re-query needs the sequence's reference frame; extend this script before using it")
key = gw.load_key()
text, usage, served = gw.call(key, model, rf.messages(it, reference), max_tokens=rf.MAX_OUT)
new = dict(old)
new.pop("error", None)
new.update({"raw": text, "usage": usage, "served_model": served, "prediction": rf.parse(text)})
with (rf.TASK / "requery-log.jsonl").open("a", encoding="utf-8") as fh:
    fh.write(json.dumps({"model": model, "condition": cond, "replaced": old}) + "\n")
rows[idx[0]] = new
path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
print(item_id, "->", new["prediction"], "| label", it["label"])
