"""Break the allocation results into stable and moving days, and measure how often a model moves a number it should not."""
import json
import pathlib

import numpy as np

TASK = pathlib.Path(__file__).parent / "task-allocation"
runs = sorted(TASK.glob("responses-*.jsonl"))

print("%-30s %7s %7s %7s %7s %7s %7s" % ("run", "n", "stable", "moving", "falsemv", "missmv", "dirok"))
for path in runs:
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
    ok = [r for r in rows if r.get("prediction") is not None]
    y = np.array([r["target"] for r in ok])
    p = np.array([r["prediction"] for r in ok])
    b = np.array([r["persistence"] for r in ok])
    s = np.array([r["fire_mean"] for r in ok])
    truth_moved = np.abs(y - b) / np.maximum(b, 1) > 0.1
    pred_moved = np.abs(p - b) / np.maximum(b, 1) > 0.1
    err = np.abs(y - p) / s
    base_err = np.abs(y - b) / s
    direction = np.sign(y - b) == np.sign(p - b)
    print("%-30s %7d %7.3f %7.3f %7.2f %7.2f %7.2f" % (
        path.stem.replace("responses-", ""), len(ok),
        float(np.mean(err[~truth_moved])), float(np.mean(err[truth_moved])),
        float(np.mean(pred_moved[~truth_moved])), float(np.mean(~pred_moved[truth_moved])),
        float(np.mean(direction[truth_moved]))))

rows = [json.loads(l) for l in (runs[0]).read_text(encoding="utf-8").splitlines()]
ok = [r for r in rows if r.get("prediction") is not None]
y = np.array([r["target"] for r in ok]); b = np.array([r["persistence"] for r in ok]); s = np.array([r["fire_mean"] for r in ok])
moved = np.abs(y - b) / np.maximum(b, 1) > 0.1
print("\npersistence                    %7d %7.3f %7.3f %7s %7s %7s" % (
    len(ok), float(np.mean((np.abs(y - b) / s)[~moved])), float(np.mean((np.abs(y - b) / s)[moved])), "0.00", "1.00", "-"))
print("\nstable days: %d of %d (%.0f%%)" % ((~moved).sum(), len(ok), 100 * (~moved).mean()))
print("columns: stable and moving are normalised mean error on those days; falsemv is the share of stable days the")
print("model moved by more than 10 percent; missmv is the share of moving days it held flat; dirok is the share of")
print("moving days it moved the right way.")
