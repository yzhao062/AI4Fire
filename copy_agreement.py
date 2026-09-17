"""How often does a grounded allocation prediction equal the count implied by the displayed analogue median ratio?

The grounded prompt lists each analogue's ratio and the band's median ratio to two decimals. The rule prediction is
persistence times that displayed median, rounded to an integer. A model that copies the analogues reproduces it.
"""
import json
import pathlib

import numpy as np

import run_allocation as ra

TASK = pathlib.Path(__file__).parent / "task-allocation"
items = ra.sample_items()
pool = ra.build_pool({i["incident_id"] for i in items})
flat = {r["analogue_id"]: (r["today"], r["next"]) for rows in pool.values() for r in rows}
for path in sorted(TASK.glob("responses-*-grounded.jsonl")):
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if len(rows) < 100:
        continue
    same, dev, n, bare_same = 0, [], 0, 0
    for r in rows:
        if r.get("prediction") is None or not r.get("analogue_ids"):
            continue
        ratios = [flat[a][1] / max(flat[a][0], 1) for a in r["analogue_ids"]]
        shown = float("%.2f" % float(np.median(ratios)))
        rule = int(round(r["persistence"] * shown))
        n += 1
        same += int(r["prediction"] == rule)
        dev.append(abs(r["prediction"] - rule))
    print("%-72s n=%d equals displayed-median rule on %d (%.3f); mean |pred - rule| %.3f; median %.1f" % (
        path.name, n, same, same / n, float(np.mean(dev)), float(np.median(dev))))
# persistence copying in the bare arms
for path in sorted(TASK.glob("responses-*-bare.jsonl")):
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if len(rows) < 100:
        continue
    ok = [r for r in rows if r.get("prediction") is not None]
    print("%-72s equals persistence on %d of %d (%.3f)" % (path.name, sum(r["prediction"] == r["persistence"] for r in ok), len(ok),
          np.mean([r["prediction"] == r["persistence"] for r in ok])))
