"""Score every Mesogeos response file with the runner's own score() and count omitted boolean answers.

Prints one row per file in the paper's column order: AUPRC, F1 on fire, call rate, omitted. Also reports
the modification time of each file so bare and grounded can be checked for same-run pairing.
"""
import datetime
import json
import pathlib

import run_mesogeos as rm

TASK = rm.TASK
for path in sorted(TASK.glob("responses-*.jsonl")):
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
    if len(rows) < 100:
        continue
    s = rm.score(rows, path.stem.replace("responses-", ""))
    omitted = sum(1 for r in rows if r.get("call") is None and r.get("probability") is not None)
    errors = sum(1 for r in rows if r.get("error"))
    mtime = datetime.datetime.fromtimestamp(path.stat().st_mtime).strftime("%m-%d %H:%M")
    print("%-58s n=%d parsed=%d auprc %.3f f1 %.3f call %.3f omitted %d errors %d  [%s]"
          % (s["run"], s["items"], s["parsed"], s["auprc"], s["f1_fire"], s["positive_rate_called"], omitted, errors, mtime))
