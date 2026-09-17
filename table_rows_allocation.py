"""Print the full allocation table rows the paper quotes: MAE, normalized error, share beating persistence,
stable and moving normalized error, false move, missed move, direction right, and the within-25% share.
Skips probe files with fewer than 100 rows."""
import argparse
import json
import os
import pathlib

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--manifest", type=str, default=None, help="Path to manifest JSON restricting response files.")
args = parser.parse_args()

TASK = pathlib.Path(__file__).parent / "task-allocation"
if args.manifest:
    with open(args.manifest, encoding="utf-8") as f:
        manifest = json.load(f)
    reported = manifest.get("reported", [])
    root_dir = pathlib.Path(__file__).resolve().parent
    allowed_rel = {
        os.path.normpath(r["path"])
        for r in reported
        if r.get("task") in ("allocation", "task-allocation")
    }
    paths = [p for p in sorted(TASK.glob("responses-*.jsonl")) if os.path.normpath(p.resolve().relative_to(root_dir)) in allowed_rel]
else:
    paths = sorted(TASK.glob("responses-*.jsonl"))

print("%-64s %6s %7s %7s %7s %7s %7s %7s %7s %7s" % ("run", "mae", "nmae", "beatp", "stable", "moving", "falsemv", "missmv", "dirok", "w25"))
for path in paths:
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not args.manifest and len(rows) < 100:
        continue
    ok = [r for r in rows if r.get("prediction") is not None]
    y = np.array([r["target"] for r in ok]); p = np.array([r["prediction"] for r in ok])
    b = np.array([r["persistence"] for r in ok]); s = np.array([r["fire_mean"] for r in ok])
    moved = np.abs(y - b) / np.maximum(b, 1) > 0.1
    pm = np.abs(p - b) / np.maximum(b, 1) > 0.1
    err = np.abs(y - p) / s; berr = np.abs(y - b) / s
    direction = np.sign(y - b) == np.sign(p - b)
    print("%-64s %6.2f %7.4f %7.3f %7.4f %7.4f %7.3f %7.3f %7.3f %7.3f" % (
        path.stem.replace("responses-", ""), float(np.mean(np.abs(y - p))), float(np.mean(err)),
        float(np.mean(err < berr)), float(np.mean(err[~moved])), float(np.mean(err[moved])),
        float(np.mean(pm[~moved])), float(np.mean(~pm[moved])), float(np.mean(direction[moved])),
        float(np.mean(np.abs(p - y) <= 0.25 * y))))
    if "opus-5-bare" in path.name:
        print("%-64s %6.2f %7.4f %7s %7.4f %7.4f %7s %7s %7s %7.3f" % ("persistence", float(np.mean(np.abs(y - b))), float(np.mean(berr)), "-",
              float(np.mean(berr[~moved])), float(np.mean(berr[moved])), "0", "1", "-", float(np.mean(np.abs(b - y) <= 0.25 * y))))
