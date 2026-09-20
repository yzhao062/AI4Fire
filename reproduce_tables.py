"""Regenerate the primary results tables of AI4Fire from stored model responses.

Runs fully offline with no API calls, no credentials, and no external data downloads.
Scores all reported core model runs (and optionally the model sweep) directly from
the tracked task-*/responses-*.jsonl files.

Exits 0 if all primary numbers reproduce successfully, or non-zero if any number diverges.

Usage:
    python reproduce_tables.py
    python reproduce_tables.py --tier core
    python reproduce_tables.py --tier all
"""
import argparse
import collections
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import models


def load_jsonl(path):
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def print_table_header(title):
    print("\n" + "=" * 92)
    print(f" {title}")
    print("=" * 92)


# Reference numbers from the published paper tables for verification
EXPECTED_CORE = {
    "allocation": {
        "persistence": {"mae": 13.76, "nmae": 0.1465, "w25": 0.780},
        "claude-opus-4.8": {"bare": (14.31, 0.1568, 0.813, 0.290), "grounded": (15.28, 0.1701, 0.780, 0.287)},
        "claude-opus-5": {"bare": (15.12, 0.1654, 0.797, 0.333), "grounded": (14.52, 0.1730, 0.763, 0.317)},
        "gemini-3.1-pro": {"bare": (16.17, 0.1786, 0.733, 0.187), "grounded": (16.07, 0.1772, 0.763, 0.213)},
        "gpt-6-astra": {"bare": (14.55, 0.1614, 0.800, 0.217), "grounded": (18.04, 0.2196, 0.747, 0.250)},
        "bedrock_qwen.qwen3-vl-235b-a22b": {"bare": (14.73, 0.1613, 0.763, 0.073), "grounded": (19.85, 0.2484, 0.693, 0.210)},
        "bedrock_us.meta.llama4-maverick-17b-instruct-v1_0": {"bare": (15.65, 0.1926, 0.710, 0.260), "grounded": (19.88, 0.2456, 0.693, 0.207)},
    },
    "smoke": {
        "claude-opus-4.8": {"bare": (0.745, 0.554, 0.000, 23), "grounded": (0.781, 0.616, 0.000, 23)},
        "claude-opus-5": {"bare": (0.770, 0.607, 0.012, 24), "grounded": (0.781, 0.643, 0.036, 24)},
        "gemini-3.1-pro": {"bare": (0.740, 0.545, 0.000, 23), "grounded": (0.837, 0.732, 0.024, 26)},
        "gpt-6-astra": {"bare": (0.786, 0.643, 0.024, 24), "grounded": (0.832, 0.741, 0.048, 26)},
        "bedrock_qwen.qwen3-vl-235b-a22b": {"bare": (0.740, 0.545, 0.000, 22), "grounded": (0.781, 0.616, 0.000, 24)},
        "bedrock_us.meta.llama4-maverick-17b-instruct-v1_0": {"bare": (0.709, 0.509, 0.024, 20), "grounded": (0.719, 0.554, 0.060, 21)},
    },
    "mesogeos": {
        "claude-opus-4.8": {"bare": (0.613, 0.561, 0.215, 0), "grounded": (0.581, 0.546, 0.249, 2)},
        "claude-opus-5": {"bare": (0.688, 0.269, 0.065, 46), "grounded": (0.682, 0.312, 0.109, 90)},
        "gemini-3.1-pro": {"bare": (0.697, 0.607, 0.207, 0), "grounded": (0.682, 0.587, 0.199, 0)},
        "gpt-6-astra": {"bare": (0.620, 0.189, 0.044, 0), "grounded": (0.590, 0.298, 0.078, 0)},
        "bedrock_qwen.qwen3-vl-235b-a22b": {"bare": (0.485, 0.567, 0.684, 0), "grounded": (0.536, 0.553, 0.710, 0)},
        "bedrock_us.meta.llama4-maverick-17b-instruct-v1_0": {"bare": (0.528, 0.513, 0.256, 0), "grounded": (0.515, 0.524, 0.303, 0)},
    },
    "wildfirevqa": {
        "claude-opus-4.8": {"bare": (0.642, 0.667, 0.639), "grounded": (0.642, 0.896, 0.608)},
        "claude-opus-5": {"bare": (0.650, 0.646, 0.650), "grounded": (0.657, 0.917, 0.622)},
        "gemini-3.1-pro": {"bare": (0.632, 0.667, 0.628), "grounded": (0.647, 0.938, 0.608)},
        "gpt-6-astra": {"bare": (0.623, 0.792, 0.600), "grounded": (0.635, 0.938, 0.594)},
        "bedrock_qwen.qwen3-vl-235b-a22b": {"bare": (0.618, 0.729, 0.603), "grounded": (0.618, 0.771, 0.597)},
        "bedrock_us.meta.llama4-maverick-17b-instruct-v1_0": {"bare": (0.534, 0.688, 0.514), "grounded": (0.544, 0.833, 0.506)},
    },
    "tooluse": {
        "claude-opus-4.8": {"bare": (0.103, 99, 0.00), "tool": (1.000, 0, 1.13)},
        "claude-opus-5": {"bare": (0.160, 35, 0.00), "tool": (1.000, 0, 1.17)},
        "gemini-3.1-pro": {"bare": (0.160, 59, 0.00), "tool": (1.000, 0, 1.62)},
        "gpt-6-astra": {"bare": (0.000, 156, 0.00), "tool": (0.994, 0, 1.07)},
        "bedrock_qwen.qwen3-vl-235b-a22b": {"bare": (0.083, 22, 0.00), "tool": (0.891, 6, 1.08)},
        "bedrock_us.meta.llama4-maverick-17b-instruct-v1_0": {"bare": (0.071, 0, 0.00), "tool": (0.885, 0, 1.00)},
    },
}


def reproduce_allocation(selected_models, verify=True):
    print_table_header("TASK 1: DAILY PERSONNEL ALLOCATION (ICS-209-PLUS, 300 items)")
    print(f"{'Model':<26} {'Condition':<10} {'MAE':>8} {'Norm MAE':>10} {'Within 25%':>12} {'Beats Persist':>14}")
    print("-" * 92)

    failures = []
    # Non-LLM persistence comparator
    sample_file = ROOT / "task-allocation" / "responses-claude-opus-5-bare.jsonl"
    if sample_file.exists():
        s_rows = load_jsonl(sample_file)
        y = np.array([r["target"] for r in s_rows])
        base = np.array([r["persistence"] for r in s_rows])
        fm = np.array([r["fire_mean"] for r in s_rows])
        p_mae = float(np.mean(np.abs(y - base)))
        p_nmae = float(np.mean(np.abs(y - base) / fm))
        p_w25 = float(np.mean(np.abs(y - base) / np.maximum(y, 1) <= 0.25))
        print(f"{'Persistence (comparator)':<26} {'--':<10} {p_mae:>8.2f} {p_nmae:>10.4f} {p_w25:>12.3f} {'--':>14}")
        print("-" * 92)
        if verify:
            exp = EXPECTED_CORE["allocation"]["persistence"]
            if abs(p_mae - exp["mae"]) > 0.01 or abs(p_nmae - exp["nmae"]) > 0.0005 or abs(p_w25 - exp["w25"]) > 0.001:
                failures.append(f"Allocation persistence comparator: got ({p_mae:.2f}, {p_nmae:.4f}, {p_w25:.3f}), expected {exp}")

    for m in selected_models:
        for cond in ("bare", "grounded"):
            path = ROOT / "task-allocation" / f"responses-{m.stem}-{cond}.jsonl"
            rows = [r for r in load_jsonl(path) if r.get("prediction") is not None]
            if not rows:
                continue
            y = np.array([r["target"] for r in rows])
            p = np.array([r["prediction"] for r in rows])
            base = np.array([r["persistence"] for r in rows])
            fm = np.array([r["fire_mean"] for r in rows])
            mae = float(np.mean(np.abs(y - p)))
            nmae = float(np.mean(np.abs(y - p) / fm))
            w25 = float(np.mean(np.abs(y - p) / np.maximum(y, 1) <= 0.25))
            beats = float(np.mean(np.abs(y - p) < np.abs(y - base)))
            print(f"{m.label:<26} {cond:<10} {mae:>8.2f} {nmae:>10.4f} {w25:>12.3f} {beats:>14.3f}")

            if verify and m.stem in EXPECTED_CORE["allocation"]:
                exp = EXPECTED_CORE["allocation"][m.stem][cond]
                if (abs(mae - exp[0]) > 0.01 or abs(nmae - exp[1]) > 0.001 or
                        abs(w25 - exp[2]) > 0.005 or abs(beats - exp[3]) > 0.005):
                    failures.append(f"Allocation {m.stem} {cond}: got ({mae:.2f}, {nmae:.4f}, {w25:.3f}, {beats:.3f}), expected {exp}")
    return failures


def reproduce_smoke(selected_models, verify=True):
    print_table_header("TASK 2: WILDFIRE SMOKE DETECTION (FIgLib, 196 paired items)")
    print(f"{'Model':<26} {'Condition':<10} {'Accuracy':>10} {'Recall':>10} {'FPR':>10} {'Sequences Detected':>20}")
    print("-" * 92)
    failures = []
    for m in selected_models:
        bare_p = ROOT / "task-figlib" / f"responses-{m.stem}-bare.jsonl"
        grd_p = ROOT / "task-figlib" / f"responses-{m.stem}-grounded.jsonl"
        bare_rows = {r["item_id"]: r for r in load_jsonl(bare_p) if r.get("prediction") is not None}
        grd_rows = {r["item_id"]: r for r in load_jsonl(grd_p) if r.get("prediction") is not None}
        paired_ids = [iid for iid in bare_rows if iid in grd_rows]
        if not paired_ids:
            continue
        for cond, rmap in (("bare", bare_rows), ("grounded", grd_rows)):
            cur = [rmap[i] for i in paired_ids]
            y = np.array([r["label"] == "smoke" for r in cur])
            p = np.array([bool(r["prediction"]) for r in cur])
            acc = float(np.mean(p == y))
            rec = float(np.mean(p[y])) if y.any() else 0.0
            fpr = float(np.mean(p[~y])) if (~y).any() else 0.0

            seqs = collections.defaultdict(list)
            for r in cur:
                if r["offset_seconds"] > 0:
                    seqs[r["sequence"]].append((r["offset_seconds"], bool(r["prediction"])))
            detected = sum(1 for pairs in seqs.values() if any(pred for o, pred in pairs))
            total_seqs = len(seqs)
            seq_str = f"{detected} of {total_seqs}"
            print(f"{m.label:<26} {cond:<10} {acc:>10.3f} {rec:>10.3f} {fpr:>10.3f} {seq_str:>20}")

            if verify and m.stem in EXPECTED_CORE["smoke"]:
                exp = EXPECTED_CORE["smoke"][m.stem][cond]
                if (abs(acc - exp[0]) > 0.005 or abs(rec - exp[1]) > 0.005 or
                        abs(fpr - exp[2]) > 0.005 or detected != exp[3]):
                    failures.append(f"Smoke {m.stem} {cond}: got ({acc:.3f}, {rec:.3f}, {fpr:.3f}, {detected}), expected {exp}")
    return failures


def reproduce_mesogeos(selected_models, verify=True):
    print_table_header("TASK 3: FIRE DANGER FORECASTING (Mesogeos Track A, 386 items)")
    print(f"{'Model':<26} {'Condition':<10} {'AUPRC':>8} {'F1 (Fire)':>12} {'Call Rate':>12} {'Omitted':>10}")
    print("-" * 92)
    from sklearn.metrics import average_precision_score, f1_score
    failures = []
    for m in selected_models:
        for cond in ("bare", "grounded"):
            path = ROOT / "task-mesogeos" / f"responses-{m.stem}-{cond}.jsonl"
            rows = load_jsonl(path)
            ok = [r for r in rows if r.get("probability") is not None or r.get("call") is not None]
            if not ok:
                continue
            y = np.array([r["label"] for r in ok])
            prob = np.array([r["probability"] if r["probability"] is not None else (1.0 if r["call"] else 0.0) for r in ok])
            call = np.array([r["call"] if r["call"] is not None else (r["probability"] or 0) >= 0.5 for r in ok])
            omitted = sum(1 for r in rows if r.get("probability") is not None and r.get("call") is None)
            auprc = float(average_precision_score(y, prob))
            f1 = float(f1_score(y, call, pos_label=1))
            call_rate = float(np.mean(call))
            print(f"{m.label:<26} {cond:<10} {auprc:>8.3f} {f1:>12.3f} {call_rate:>12.3f} {omitted:>10}")

            if verify and m.stem in EXPECTED_CORE["mesogeos"]:
                exp = EXPECTED_CORE["mesogeos"][m.stem][cond]
                if (abs(auprc - exp[0]) > 0.005 or abs(f1 - exp[1]) > 0.005 or
                        abs(call_rate - exp[2]) > 0.005 or omitted != exp[3]):
                    failures.append(f"Mesogeos {m.stem} {cond}: got ({auprc:.3f}, {f1:.3f}, {call_rate:.3f}, {omitted}), expected {exp}")
    return failures


def reproduce_wildfirevqa(selected_models, verify=True):
    print_table_header("TASK 4: AERIAL QUESTION ANSWERING (WildFireVQA, 408 items)")
    print(f"{'Model':<26} {'Condition':<10} {'Accuracy':>10} {'Closed-Form (48)':>18} {'Other (360)':>14}")
    print("-" * 92)
    print(f"{'Majority reference':<26} {'--':<10} {0.6275:>10.4f} {'--':>18} {'--':>14}")
    print(f"{'Majority + closed-form':<26} {'--':<10} {0.6642:>10.4f} {1.0000:>18.4f} {0.6194:>14.4f}")
    print("-" * 92)
    CLOSED_FORM_TYPES = {"CL1", "CMR4", "DS7", "DS8"}
    failures = []
    for m in selected_models:
        for cond in ("bare", "grounded"):
            path = ROOT / "task-wildfirevqa" / f"responses-{m.stem}-{cond}.jsonl"
            rows = load_jsonl(path)
            if not rows:
                continue
            correct = np.array([r.get("correct") is True for r in rows])
            cf_mask = np.array([r.get("question_id") in CLOSED_FORM_TYPES or r.get("no_image_answerable") == "closed_form" for r in rows])
            acc = float(np.mean(correct))
            cf_acc = float(np.mean(correct[cf_mask])) if cf_mask.any() else 0.0
            oth_acc = float(np.mean(correct[~cf_mask])) if (~cf_mask).any() else 0.0
            print(f"{m.label:<26} {cond:<10} {acc:>10.4f} {cf_acc:>18.4f} {oth_acc:>14.4f}")

            if verify and m.stem in EXPECTED_CORE["wildfirevqa"]:
                exp = EXPECTED_CORE["wildfirevqa"][m.stem][cond]
                if (abs(acc - exp[0]) > 0.005 or abs(cf_acc - exp[1]) > 0.005 or abs(oth_acc - exp[2]) > 0.005):
                    failures.append(f"WildFireVQA {m.stem} {cond}: got ({acc:.3f}, {cf_acc:.3f}, {oth_acc:.3f}), expected {exp}")
    return failures


def reproduce_tooluse(selected_models, verify=True):
    print_table_header("TASK 5: FIRE DATA TOOL USE (FPA-FOD, 156 items)")
    print(f"{'Model':<26} {'Condition':<10} {'Accuracy':>10} {'Abstained':>12} {'Mean Calls':>12}")
    print("-" * 92)
    failures = []
    for m in selected_models:
        for cond in ("bare", "tool"):
            path = ROOT / "task-tooluse" / f"responses-{m.stem}-{cond}.jsonl"
            rows = load_jsonl(path)
            if not rows:
                continue
            acc = float(np.mean([r.get("correct") is True for r in rows]))
            abstained = sum(1 for r in rows if r.get("abstained") is True)
            mean_calls = float(np.mean([r.get("n_tool_calls", 0) for r in rows]))
            print(f"{m.label:<26} {cond:<10} {acc:>10.3f} {abstained:>12} {mean_calls:>12.2f}")

            if verify and m.stem in EXPECTED_CORE["tooluse"]:
                exp = EXPECTED_CORE["tooluse"][m.stem][cond]
                if (abs(acc - exp[0]) > 0.005 or abstained != exp[1] or abs(mean_calls - exp[2]) > 0.02):
                    failures.append(f"Tooluse {m.stem} {cond}: got ({acc:.3f}, {abstained}, {mean_calls:.2f}), expected {exp}")
    return failures



TASK_DIRS = {
    "allocation": "task-allocation",
    "smoke": "task-figlib",
    "mesogeos": "task-mesogeos",
    "wildfirevqa": "task-wildfirevqa",
    "tooluse": "task-tooluse",
}


def check_coverage(selected_models):
    """Fail when a model-arm the expectation table covers has no stored responses.

    Without this, a missing or empty response tree produces no mismatches and the run reports
    success having verified nothing. Review round 1, Codex N6.
    """
    stems = {m.stem for m in selected_models}
    problems = []
    checked = 0
    for task, entries in EXPECTED_CORE.items():
        task_dir = ROOT / TASK_DIRS[task]
        for stem, conds in entries.items():
            if stem not in stems:
                continue
            for cond, expected in conds.items():
                if not isinstance(expected, (tuple, list)):
                    continue  # a metric row, such as allocation's persistence reference
                path = task_dir / f"responses-{stem}-{cond}.jsonl"
                if not path.exists():
                    problems.append(f"Coverage {task} {stem} {cond}: {path.name} is missing")
                elif not load_jsonl(path):
                    problems.append(f"Coverage {task} {stem} {cond}: {path.name} has no records")
                else:
                    checked += 1
    if not problems:
        print(f"\n Coverage: {checked} expected model-arms present with stored responses.")
    return problems


def main():
    ap = argparse.ArgumentParser(description="Regenerate AI4Fire main result tables from stored responses.")
    ap.add_argument("--tier", default="core", choices=["core", "added", "all"],
                    help="Model tier: core (reported 6), added (Bedrock sweep 10), or all (16 full-capability)")
    ap.add_argument("--no-verify", action="store_true", help="Skip assertion checks against published values")
    args = ap.parse_args()

    selected = models.models(tier=args.tier)
    print("=" * 92)
    print(f" AI4FIRE BENCHMARK RESULTS REPRODUCTION (Tier: {args.tier.upper()}, Models: {len(selected)})")
    print(" Verified against tracked responses-*.jsonl files")
    print("=" * 92)

    verify = not args.no_verify and (args.tier in ("core", "all"))
    all_failures = []
    all_failures.extend(reproduce_allocation(selected, verify=verify))
    all_failures.extend(reproduce_smoke(selected, verify=verify))
    all_failures.extend(reproduce_mesogeos(selected, verify=verify))
    all_failures.extend(reproduce_wildfirevqa(selected, verify=verify))
    all_failures.extend(reproduce_tooluse(selected, verify=verify))
    if verify:
        all_failures.extend(check_coverage(selected))

    print("\n" + "=" * 92)
    if all_failures:
        print(f" REPRODUCTION FAILED: {len(all_failures)} mismatch(es) detected:")
        for f in all_failures:
            print(f"   [FAIL] {f}")
        print("=" * 92)
        sys.exit(1)
    else:
        print(" ALL PRIMARY NUMBERS REPRODUCED AND VERIFIED SUCCESSFULLY FROM STORED RESPONSES.")
        print("=" * 92)
        sys.exit(0)


if __name__ == "__main__":
    main()
