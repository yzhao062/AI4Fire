"""Print LaTeX rows for the two Bedrock models in the paper's three results tables, from the same sources the
existing rows use: scores.json for MAE, normalized MAE, beats-persistence, AUPRC, F1, call rate and sequences
detected; analyze_allocation's definitions for stable, moving, false move, missed move, direction; the paired
FIgLib scorer for the smoke rows; the response files for omitted booleans.
"""
import json
import pathlib

import numpy as np

S = pathlib.Path(__file__).parent
NEW = {"bedrock_qwen.qwen3-vl-235b-a22b": "Qwen3-VL-235B-A22B",
       "bedrock_us.meta.llama4-maverick-17b-instruct-v1_0": "Llama 4 Maverick"}
RUNKEY = {"bedrock_qwen.qwen3-vl-235b-a22b": "bedrock:qwen.qwen3-vl-235b-a22b",
          "bedrock_us.meta.llama4-maverick-17b-instruct-v1_0": "bedrock:us.meta.llama4-maverick-17b-instruct-v1:0"}


def rows(p):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()]


print("% allocation rows")
scores = {s["run"]: s for s in json.loads((S / "task-allocation" / "scores.json").read_text(encoding="utf-8"))}
for safe, name in NEW.items():
    for cond in ("bare", "grounded"):
        s = scores[RUNKEY[safe] + "/" + cond]
        ok = [r for r in rows(S / "task-allocation" / ("responses-%s-%s.jsonl" % (safe, cond))) if r.get("prediction") is not None]
        y = np.array([r["target"] for r in ok]); p = np.array([r["prediction"] for r in ok])
        b = np.array([r["persistence"] for r in ok]); fm = np.array([r["fire_mean"] for r in ok])
        truth_moved = np.abs(y - b) / np.maximum(b, 1) > 0.1
        pred_moved = np.abs(p - b) / np.maximum(b, 1) > 0.1
        err = np.abs(y - p) / fm
        direction = np.sign(y - b) == np.sign(p - b)
        print("%s %s & %.2f & %.3f & %.2f & %.3f & %.3f & %.2f & %.2f & %.2f \\\\ %% src: benchmark run 2026-09-16, Bedrock"
              % (name, cond, s["mae"], s["mae_norm"], s["beats_persistence_share"], err[~truth_moved].mean(), err[truth_moved].mean(),
                 pred_moved[~truth_moved].mean(), (~pred_moved[truth_moved]).mean(), direction[truth_moved].mean()))

print("% mesogeos rows")
import run_mesogeos as rm
for safe, name in NEW.items():
    for cond in ("bare", "grounded"):
        rr = rows(S / "task-mesogeos" / ("responses-%s-%s.jsonl" % (safe, cond)))
        s = rm.score(rr, safe)
        omitted = sum(1 for r in rr if r.get("call") is None and r.get("probability") is not None)
        print("%s %s & %.3f & %.3f & %.3f & %d \\\\ %% src: benchmark run 2026-09-16, Bedrock" % (name, cond, s["auprc"], s["f1_fire"], s["positive_rate_called"], omitted))

print("% figlib rows (paired on the 196 items both conditions answered)")
fs = {s["run"]: s for s in json.loads((S / "task-figlib" / "scores.json").read_text(encoding="utf-8"))}
paired = json.loads(pathlib.Path(r"C:\Users\yuezh\AppData\Local\Temp\claude\C--Users-yuezh-PycharmProjects-internal-writing\5298f142-9cba-41ca-a975-67b31eb2587b\scratchpad\agent-io\fire-r3\wf-p1\five\figlib-paired.json").read_text(encoding="utf-8"))
for safe, name in NEW.items():
    for cond in ("bare", "grounded"):
        r = next(x for x in paired["rows"] if x["model"] == safe and x["condition"] == cond)
        seq = fs[RUNKEY[safe] + "/" + cond]["sequences_detected"]
        print("%s %s & %d & %.3f & %.3f & %.3f & %s \\\\ %% src: benchmark run 2026-09-16, Bedrock" % (name, cond, r["n_paired"], r["accuracy"], r["recall_smoke"], r["fpr"], seq))
for s in paired["sign_tests"]:
    if s["model"] in NEW:
        print("%% sign test %s: gained %d lost %d p=%.4f fires %s" % (NEW[s["model"]], s["gained"], s["lost"], s["p_two_sided_full"], s.get("fires_touched")))
