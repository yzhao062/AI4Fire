"""Calibration and decision-consistency analysis for Mesogeos Track A holdout (386 items).

Computes:
1. Decision consistency: agreement between `call` and (probability >= 0.5), omitted calls,
   and direction/distribution of disagreements.
2. Calibration: mean predicted probability vs base rate (0.339), expected calibration error (ECE)
   with 10 equal-width bins, ECE with 10 equal-count bins, Brier score, and reliability tables.
3. Murphy Brier score decomposition: Calibration (Reliability) and Refinement (Uncertainty - Resolution).
4. Probability granularity: number of distinct probability values and 10 most frequent.
5. Reference baselines: calendar-month prior (month_prior.py) and last-day max t2m rule.

Outputs formatted tables to stdout and saves structured results to analysis/calibration_mesogeos.json.
"""

import argparse
import collections
import json
import math
import pathlib
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score

ROOT = pathlib.Path(__file__).resolve().parent.parent
TASK = ROOT / "task-mesogeos"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import models
from models import add_model_args, resolve_models

DISPLAY_NAMES = {
    "claude-opus-4.8": "Claude Opus 4.8",
    "claude-opus-5": "Claude Opus 5",
    "gemini-3.1-pro": "Gemini 3.1 Pro",
    "gpt-6-astra": "GPT-6-Astra",
    "bedrock_qwen.qwen3-vl-235b-a22b": "Qwen3-VL 235B",
    "bedrock_us.meta.llama4-maverick-17b-instruct-v1_0": "Llama 4 Maverick 17B",
}


def get_model_order(selected_models=None):
    if selected_models is None:
        selected_models = models.models(tier="core", task="mesogeos")
    order = []
    for m in selected_models:
        name = DISPLAY_NAMES.get(m.stem, m.label)
        for cond in ("bare", "grounded"):
            order.append((m.stem, cond, f"{name} ({cond})"))
    return order


MODEL_ORDER = get_model_order()

TRAIN_PRIOR_BY_MONTH = {
    1: 0.06451612903225806,
    2: 0.1928374655647383,
    3: 0.11091127098321343,
    4: 0.04505716207128446,
    5: 0.046027742749054225,
    6: 0.21708378672470077,
    7: 0.5626283367556468,
    8: 0.6420522656437305,
    9: 0.3276887871853547,
    10: 0.15371621621621623,
    11: 0.0804953560371517,
    12: 0.06722689075630252,
}
TRAIN_PRIOR_MEAN = 0.3308039376538146


def load_items() -> List[Dict[str, Any]]:
    path = TASK / "items.jsonl"
    all_items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    test_items = [it for it in all_items if it.get("split") == "test" and it.get("fold", 0) == 0]
    return test_items


def compute_consistency(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    consistent = 0
    inconsistent = 0
    omitted = 0
    call_yes_p_low = []
    call_no_p_high = []

    for r in rows:
        c = r.get("call")
        p = r.get("probability")
        if c is None:
            omitted += 1
        elif c == (p >= 0.5):
            consistent += 1
        else:
            inconsistent += 1
            if c and p < 0.5:
                call_yes_p_low.append(float(p))
            elif (not c) and p >= 0.5:
                call_no_p_high.append(float(p))

    p_low_summary = None
    if call_yes_p_low:
        p_low_arr = np.array(call_yes_p_low)
        p_low_summary = {
            "count": len(call_yes_p_low),
            "min": float(np.min(p_low_arr)),
            "p25": float(np.percentile(p_low_arr, 25)),
            "median": float(np.median(p_low_arr)),
            "p75": float(np.percentile(p_low_arr, 75)),
            "max": float(np.max(p_low_arr)),
            "top_values": collections.Counter(round(x, 4) for x in call_yes_p_low).most_common(5),
        }

    p_high_summary = None
    if call_no_p_high:
        p_high_arr = np.array(call_no_p_high)
        p_high_summary = {
            "count": len(call_no_p_high),
            "min": float(np.min(p_high_arr)),
            "median": float(np.median(p_high_arr)),
            "max": float(np.max(p_high_arr)),
            "top_values": collections.Counter(round(x, 4) for x in call_no_p_high).most_common(5),
        }

    return {
        "n_items": n,
        "consistent": consistent,
        "consistent_rate": consistent / n if n else 0.0,
        "inconsistent": inconsistent,
        "inconsistent_rate": inconsistent / n if n else 0.0,
        "omitted": omitted,
        "omitted_rate": omitted / n if n else 0.0,
        "call_yes_p_low": len(call_yes_p_low),
        "call_no_p_high": len(call_no_p_high),
        "p_low_dist": p_low_summary,
        "p_high_dist": p_high_summary,
    }


def compute_calibration_and_murphy(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> Dict[str, Any]:
    N = len(y_prob)
    base_rate = float(np.mean(y_true))
    mean_p = float(np.mean(y_prob))
    brier = float(np.mean((y_prob - y_true) ** 2))

    # Equal-width bins (10 bins: [0, 0.1), [0.1, 0.2), ..., [0.9, 1.0])
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    binids_w = np.searchsorted(bin_edges[1:-1], y_prob)

    bins_table = []
    ece_w = 0.0
    cal_term = 0.0
    ref_term = 0.0
    res_term = 0.0

    for k in range(n_bins):
        mask = binids_w == k
        cnt = int(np.sum(mask))
        low = bin_edges[k]
        high = bin_edges[k + 1]
        if cnt > 0:
            p_mean = float(np.mean(y_prob[mask]))
            y_mean = float(np.mean(y_true[mask]))
            wk = cnt / N
            ece_w += wk * abs(p_mean - y_mean)
            cal_term += wk * ((p_mean - y_mean) ** 2)
            ref_term += wk * y_mean * (1.0 - y_mean)
            res_term += wk * ((y_mean - base_rate) ** 2)
            bins_table.append({
                "bin_idx": k,
                "range": f"[{low:.1f}, {high:.1f}]" if k == 0 else f"({low:.1f}, {high:.1f}]",
                "count": cnt,
                "weight": float(wk),
                "mean_pred": p_mean,
                "obs_rate": y_mean,
            })
        else:
            bins_table.append({
                "bin_idx": k,
                "range": f"[{low:.1f}, {high:.1f}]" if k == 0 else f"({low:.1f}, {high:.1f}]",
                "count": 0,
                "weight": 0.0,
                "mean_pred": 0.0,
                "obs_rate": 0.0,
            })

    unc_term = base_rate * (1.0 - base_rate)

    # Equal-count bins (10 bins by sorting, chunks of N//10)
    order = np.argsort(y_prob, kind="stable")
    chunks = np.array_split(order, n_bins)
    ece_c = 0.0
    equal_count_table = []
    for k, ch in enumerate(chunks):
        cnt = len(ch)
        p_mean = float(np.mean(y_prob[ch]))
        y_mean = float(np.mean(y_true[ch]))
        wk = cnt / N
        ece_c += wk * abs(p_mean - y_mean)
        equal_count_table.append({
            "bin_idx": k,
            "count": cnt,
            "min_pred": float(np.min(y_prob[ch])),
            "max_pred": float(np.max(y_prob[ch])),
            "mean_pred": p_mean,
            "obs_rate": y_mean,
        })

    return {
        "base_rate": base_rate,
        "mean_predicted": mean_p,
        "ratio_mean_to_base": mean_p / base_rate if base_rate > 0 else 0.0,
        "brier_score": brier,
        "ece_equal_width": float(ece_w),
        "ece_equal_count": float(ece_c),
        "murphy": {
            "calibration": float(cal_term),
            "refinement": float(ref_term),
            "resolution": float(res_term),
            "uncertainty": float(unc_term),
            "brier_binned": float(cal_term + ref_term),
        },
        "reliability_table_equal_width": bins_table,
        "reliability_table_equal_count": equal_count_table,
    }


def compute_granularity(y_prob: np.ndarray) -> Dict[str, Any]:
    counts = collections.Counter(float(p) for p in y_prob)
    n = len(y_prob)
    distinct_count = len(counts)
    top10 = [
        {"value": float(val), "count": cnt, "share": float(cnt / n)}
        for val, cnt in counts.most_common(10)
    ]
    return {
        "distinct_values": distinct_count,
        "top10_most_frequent": top10,
    }


def analyze_all(selected_models=None) -> Dict[str, Any]:
    items = load_items()
    y_test = np.array([it["label"] for it in items])
    assert len(items) == 386, f"Expected 386 test items, found {len(items)}"
    base_rate = float(np.mean(y_test))

    results = {
        "dataset": "Mesogeos Track A (2021-2022 holdout, fold 0)",
        "n_items": len(items),
        "base_rate": base_rate,
        "runs": {},
        "references": {},
    }

    model_order = get_model_order(selected_models)
    for model_id, cond, display_name in model_order:
        file_path = TASK / f"responses-{model_id}-{cond}.jsonl"
        if not file_path.exists():
            print(f"Warning: {file_path} not found", file=sys.stderr)
            continue
        rows = [json.loads(line) for line in file_path.read_text(encoding="utf-8").splitlines()]
        y_prob = np.array([r["probability"] for r in rows], dtype=float)
        y_true = np.array([r["label"] for r in rows], dtype=int)

        # Basic verification against labels
        assert np.array_equal(y_true, y_test), f"Label mismatch in {file_path}"

        consistency = compute_consistency(rows)
        calibration = compute_calibration_and_murphy(y_true, y_prob)
        granularity = compute_granularity(y_prob)

        # AUPRC and F1 for reference
        imputed_calls = np.array([r["call"] if r["call"] is not None else (r["probability"] or 0) >= 0.5 for r in rows])
        auprc = float(average_precision_score(y_true, y_prob))
        f1_fire = float(f1_score(y_true, imputed_calls, pos_label=1))
        call_rate = float(np.mean(imputed_calls))

        run_key = f"{model_id}/{cond}"
        results["runs"][run_key] = {
            "model_id": model_id,
            "condition": cond,
            "display_name": display_name,
            "auprc": auprc,
            "f1_fire": f1_fire,
            "call_rate_imputed": call_rate,
            "consistency": consistency,
            "calibration": calibration,
            "granularity": granularity,
        }

    # Trivial References
    # Reference 1: Calendar-month prior from training years
    m_ends = np.array([int(it["context"]["window_end"][5:7]) for it in items])
    prior_probs = np.array([TRAIN_PRIOR_BY_MONTH.get(m, TRAIN_PRIOR_MEAN) for m in m_ends])
    ref_cal = compute_calibration_and_murphy(y_test, prior_probs)
    ref_gran = compute_granularity(prior_probs)
    ref_auprc = float(average_precision_score(y_test, prior_probs))

    results["references"]["calendar_month_prior"] = {
        "name": "Calendar-month prior (training years 2006-2019)",
        "is_probability": True,
        "auprc": ref_auprc,
        "calibration": ref_cal,
        "granularity": ref_gran,
    }

    # Reference 2: Last-day max 2 m temperature (t2m)
    t2m_scores = np.array([float(it["context"]["daily"]["t2m"][-1]) for it in items])
    t2m_auprc = float(average_precision_score(y_test, t2m_scores))
    results["references"]["last_day_t2m"] = {
        "name": "Last-day maximum 2 m temperature (t2m)",
        "is_probability": False,
        "auprc": t2m_auprc,
        "note": "Raw physical temperature in Kelvin (range 274.1-319.4 K). It is an uncalibrated ranking score, not a probability; calibration metrics (mean p, ECE, Brier, Murphy decomposition) do not apply.",
    }

    return results


def print_tables(results: Dict[str, Any]):
    print("=" * 110)
    print("AI4FIRE MESOGEOS: DECISION CONSISTENCY AND CALIBRATION ANALYSIS (386 items, base rate = 0.339)")
    print("=" * 110)

    # -------------------------------------------------------------
    # 1. Decision Consistency Table
    # -------------------------------------------------------------
    print("\n" + "-" * 110)
    print("TABLE 1: DECISION CONSISTENCY ON 386 TEST ITEMS")
    print("Compares explicit boolean `call` with the canonical threshold (stated probability p >= 0.5)")
    print("-" * 110)
    header1 = (
        f"{'Model / Arm':<38} | {'N':>4} | {'Consistent':>10} | {'Inconsistent':>12} | {'Omitted':>7} | "
        f"{'Call=Y (p<0.5)':>14} | {'Call=N (p>=0.5)':>15} | {'p distribution when Call=Y (p<0.5)':<25}"
    )
    print(header1)
    print("-" * 110)

    for run_key, r in results["runs"].items():
        name = r["display_name"]
        c = r["consistency"]
        p_low = c["p_low_dist"]
        if p_low:
            dist_str = f"min={p_low['min']:.2f}, med={p_low['median']:.2f}, max={p_low['max']:.2f}"
        else:
            dist_str = "n/a"

        print(
            f"{name:<38} | {c['n_items']:>4} | {c['consistent']:>10} | {c['inconsistent']:>12} | {c['omitted']:>7} | "
            f"{c['call_yes_p_low']:>14} | {c['call_no_p_high']:>15} | {dist_str:<25}"
        )
    print("-" * 110)

    # -------------------------------------------------------------
    # 2. Calibration & Murphy Decomposition Table
    # -------------------------------------------------------------
    print("\n" + "-" * 110)
    print("TABLE 2: CALIBRATION AND MURPHY BRIER SCORE DECOMPOSITION")
    print("Murphy decomposition: Brier = Calibration (Reliability) + Refinement = Calibration - Resolution + Uncertainty")
    print(f"Sample Uncertainty = y_bar * (1 - y_bar) = {results['base_rate']*(1-results['base_rate']):.3f} (base rate = {results['base_rate']:.3f})")
    print("-" * 110)
    header2 = (
        f"{'Model / Reference':<36} | {'Mean p':>6} | {'Base':>5} | {'ECE(w)':>6} | {'ECE(c)':>6} | "
        f"{'Brier':>6} | {'Cal (Rel)':>9} | {'Refine':>6} | {'Resol':>6} | {'AUPRC':>6}"
    )
    print(header2)
    print("-" * 110)

    for run_key, r in results["runs"].items():
        name = r["display_name"]
        cal = r["calibration"]
        m = cal["murphy"]
        print(
            f"{name:<36} | {cal['mean_predicted']:>6.3f} | {cal['base_rate']:>5.3f} | {cal['ece_equal_width']:>6.3f} | "
            f"{cal['ece_equal_count']:>6.3f} | {cal['brier_score']:>6.3f} | {m['calibration']:>9.3f} | "
            f"{m['refinement']:>6.3f} | {m['resolution']:>6.3f} | {r['auprc']:>6.3f}"
        )

    # Add Reference 1
    ref1 = results["references"]["calendar_month_prior"]
    c1 = ref1["calibration"]
    m1 = c1["murphy"]
    print("-" * 110)
    print(
        f"{'Calendar-Month Prior (Ref 1)':<36} | {c1['mean_predicted']:>6.3f} | {c1['base_rate']:>5.3f} | {c1['ece_equal_width']:>6.3f} | "
        f"{c1['ece_equal_count']:>6.3f} | {c1['brier_score']:>6.3f} | {m1['calibration']:>9.3f} | "
        f"{m1['refinement']:>6.3f} | {m1['resolution']:>6.3f} | {ref1['auprc']:>6.3f}"
    )
    ref2 = results["references"]["last_day_t2m"]
    print(
        f"{'Last-Day Max t2m (Ref 2, score)':<36} | {'n/a':>6} | {results['base_rate']:>5.3f} | {'n/a':>6} | "
        f"{'n/a':>6} | {'n/a':>6} | {'n/a':>9} | {'n/a':>6} | {'n/a':>6} | {ref2['auprc']:>6.3f}"
    )
    print("-" * 110)

    # -------------------------------------------------------------
    # 3. Probability Granularity Table
    # -------------------------------------------------------------
    print("\n" + "-" * 110)
    print("TABLE 3: PROBABILITY GRANULARITY (NUMBER OF DISTINCT VALUES & TOP 10 FREQUENT VALUES)")
    print("-" * 110)

    for run_key, r in results["runs"].items():
        name = r["display_name"]
        g = r["granularity"]
        print(f"\n>> {name} (Distinct probabilities: {g['distinct_values']})")
        top10_items = g["top10_most_frequent"]
        top_str = ", ".join(f"{it['value']:.3f} (n={it['count']}, {it['share']*100:.1f}%)" for it in top10_items)
        print(f"   Top 10: {top_str}")

    # Month prior granularity
    ref1_g = results["references"]["calendar_month_prior"]["granularity"]
    print(f"\n>> Calendar-Month Prior (Ref 1) (Distinct probabilities: {ref1_g['distinct_values']})")
    ref1_top = ref1_g["top10_most_frequent"]
    ref1_str = ", ".join(f"{it['value']:.3f} (n={it['count']}, {it['share']*100:.1f}%)" for it in ref1_top)
    print(f"   Top 10: {ref1_str}")
    print("-" * 110)

    # -------------------------------------------------------------
    # 4. Reliability Tables (Equal-Width 10 Bins)
    # -------------------------------------------------------------
    print("\n" + "-" * 110)
    print("TABLE 4: RELIABILITY TABLES (10 EQUAL-WIDTH BINS)")
    print("-" * 110)
    all_runs_to_print = list(results["runs"].keys()) + ["calendar_month_prior"]
    for run_key in all_runs_to_print:
        if run_key == "calendar_month_prior":
            entry = results["references"]["calendar_month_prior"]
            display_name = entry["name"]
            cal = entry["calibration"]
        else:
            entry = results["runs"][run_key]
            display_name = entry["display_name"]
            cal = entry["calibration"]

        print(f"\n>> Reliability Table: {display_name}")
        print(f"{'Bin':<12} | {'Count':>6} | {'Weight':>7} | {'Mean Pred p':>11} | {'Obs Fire Rate':>13} | {'Gap |p - y|':>11}")
        print("-" * 75)
        for b in cal["reliability_table_equal_width"]:
            if b["count"] > 0:
                gap = abs(b["mean_pred"] - b["obs_rate"])
                print(f"{b['range']:<12} | {b['count']:>6} | {b['weight']:>7.3f} | {b['mean_pred']:>11.3f} | {b['obs_rate']:>13.3f} | {gap:>11.3f}")
            else:
                print(f"{b['range']:<12} | {b['count']:>6} | {b['weight']:>7.3f} | {'-':>11} | {'-':>13} | {'-':>11}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, default=ROOT / "analysis" / "calibration_mesogeos.json")
    add_model_args(ap, default_tier="core")
    args = ap.parse_args()
    selected_models = resolve_models(args, task="mesogeos", default_tier="core")
    results = analyze_all(selected_models)
    print_tables(results)

    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote full structured calibration data to {out_path}")


if __name__ == "__main__":
    main()
