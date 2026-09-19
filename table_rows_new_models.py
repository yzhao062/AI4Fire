"""Print LaTeX rows for models in the paper's three results tables, from the same sources the
existing rows use: scores.json for MAE, normalized MAE, beats-persistence, AUPRC, F1, call rate and sequences
detected; analyze_allocation's definitions for stable, moving, false move, missed move, direction; the paired
FIgLib scorer for the smoke rows; the response files for omitted booleans.
"""
import argparse
import json
import pathlib
import sys

import numpy as np

S = pathlib.Path(__file__).resolve().parent
if str(S) not in sys.path:
    sys.path.insert(0, str(S))
import models
from models import add_model_args, resolve_models
import run_mesogeos as rm


def rows(p):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]



def run_date(m) -> str:
    """The date the model's allocation, fire-danger, and smoke arms ran: the core six on 2026-09-16, the sweep on 2026-09-18."""
    return "2026-09-16" if m.tier == "core" else "2026-09-18"

def display_name(m):
    if "qwen3-vl" in m.stem.lower():
        return "Qwen3-VL-235B-A22B"
    return m.label


def main():
    ap = argparse.ArgumentParser(description="Print LaTeX table rows for benchmark models.")
    add_model_args(ap, default_tier="core")
    args = ap.parse_args()
    selected_models = resolve_models(args, default_tier="core")

    print("% allocation rows")

    for m in selected_models:
        name = display_name(m)
        vendor_comment = "Bedrock" if m.path == "bedrock" else (m.vendor.capitalize() if m.vendor else "Unknown")
        for cond in ("bare", "grounded"):
            resp_path = S / "task-allocation" / f"responses-{m.stem}-{cond}.jsonl"
            if not resp_path.exists():
                continue
            ok = [r for r in rows(resp_path) if r.get("prediction") is not None]
            if not ok:
                continue
            y = np.array([r["target"] for r in ok])
            p = np.array([r["prediction"] for r in ok])
            b = np.array([r["persistence"] for r in ok])
            fm = np.array([r["fire_mean"] for r in ok])
            truth_moved = np.abs(y - b) / np.maximum(b, 1) > 0.1
            pred_moved = np.abs(p - b) / np.maximum(b, 1) > 0.1
            err = np.abs(y - p) / fm
            direction = np.sign(y - b) == np.sign(p - b)

            # every column comes from the recorded responses; a cached scores.json entry can predate the
            # current scorer and would print a row the paper does not carry
            mae = float(np.mean(np.abs(y - p)))
            mae_norm = float(np.mean(err))
            beatp = float(np.mean(err < (np.abs(y - b) / fm)))

            print("%s %s & %.2f & %.3f & %.2f & %.3f & %.3f & %.2f & %.2f & %.2f \\\\ %% src: benchmark run %s, %s"
                  % (name, cond, mae, mae_norm, beatp, err[~truth_moved].mean(), err[truth_moved].mean(),
                     pred_moved[~truth_moved].mean(), (~pred_moved[truth_moved]).mean(), direction[truth_moved].mean(),
                     run_date(m), vendor_comment))

    print("% mesogeos rows")
    for m in selected_models:
        name = display_name(m)
        vendor_comment = "Bedrock" if m.path == "bedrock" else (m.vendor.capitalize() if m.vendor else "Unknown")
        for cond in ("bare", "grounded"):
            resp_path = S / "task-mesogeos" / f"responses-{m.stem}-{cond}.jsonl"
            if not resp_path.exists():
                continue
            rr = rows(resp_path)
            s = rm.score(rr, m.stem)
            omitted = sum(1 for r in rr if r.get("call") is None and r.get("probability") is not None)
            print("%s %s & %.3f & %.3f & %.3f & %d \\\\ %% src: benchmark run %s, %s"
                  % (name, cond, s["auprc"], s["f1_fire"], s["positive_rate_called"], omitted, run_date(m), vendor_comment))

    print("% figlib rows (paired on the items both conditions answered)")
    scores_figlib_path = S / "task-figlib" / "scores.json"
    fs = {}
    if scores_figlib_path.exists():
        fs = {s["run"]: s for s in json.loads(scores_figlib_path.read_text(encoding="utf-8"))}

    figlib_paired_path = S / "analysis" / "figlib-paired.json"
    paired = {}
    if figlib_paired_path.exists():
        paired = json.loads(figlib_paired_path.read_text(encoding="utf-8"))

    paired_rows = paired.get("rows", [])
    for m in selected_models:
        name = display_name(m)
        vendor_comment = "Bedrock" if m.path == "bedrock" else (m.vendor.capitalize() if m.vendor else "Unknown")
        for cond in ("bare", "grounded"):
            r = next((x for x in paired_rows if x.get("model") == m.stem and x.get("condition") == cond), None)
            if r is None:
                continue
            s_entry = fs.get(f"{m.model}/{cond}") or fs.get(f"{m.stem}/{cond}") or {}
            seq = s_entry.get("sequences_detected", "n/a")
            print("%s %s & %d & %.3f & %.3f & %.3f & %s \\\\ %% src: benchmark run %s, %s"
                  % (name, cond, r["n_paired"], r["accuracy"], r["recall_smoke"], r["fpr"], seq, run_date(m), vendor_comment))

    selected_stems = {m.stem for m in selected_models}
    name_map = {m.stem: display_name(m) for m in selected_models}
    for s in paired.get("sign_tests", []):
        if s.get("model") in selected_stems:
            m_label = name_map[s["model"]]
            print("%% sign test %s: gained %d lost %d p=%.4f fires %s"
                  % (m_label, s["gained"], s["lost"], s["p_two_sided_full"], s.get("fires_touched")))


if __name__ == "__main__":
    main()
