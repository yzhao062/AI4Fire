"""Score every Mesogeos response file with the runner's own score() and count omitted boolean answers.

Prints one row per file in the paper's column order: AUPRC, F1 on fire, call rate, omitted. Also reports
the modification time of each file so bare and grounded can be checked for same-run pairing.
"""
import argparse
import datetime
import json
import os
import pathlib

import run_mesogeos as rm

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--manifest", type=str, default=None, help="Path to manifest JSON restricting response files.")
args = parser.parse_args()

TASK = rm.TASK
if args.manifest:
    with open(args.manifest, encoding="utf-8") as f:
        manifest = json.load(f)
    reported = manifest.get("reported", [])
    root_dir = pathlib.Path(__file__).resolve().parent
    allowed_rel = {
        os.path.normpath(r["path"])
        for r in reported
        if r.get("task") in ("mesogeos", "task-mesogeos")
    }
    paths = [p for p in sorted(TASK.glob("responses-*.jsonl")) if os.path.normpath(p.resolve().relative_to(root_dir)) in allowed_rel]
else:
    paths = sorted(TASK.glob("responses-*.jsonl"))

for path in paths:
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
    if not args.manifest and len(rows) < 100:
        continue
    s = rm.score(rows, path.stem.replace("responses-", ""))
    omitted = sum(1 for r in rows if r.get("call") is None and r.get("probability") is not None)
    errors = sum(1 for r in rows if r.get("error"))
    mtime = datetime.datetime.fromtimestamp(path.stat().st_mtime).strftime("%m-%d %H:%M")
    print("%-58s n=%d parsed=%d auprc %.3f f1 %.3f call %.3f omitted %d errors %d  [%s]"
          % (s["run"], s["items"], s["parsed"], s["auprc"], s["f1_fire"], s["positive_rate_called"], omitted, errors, mtime))
