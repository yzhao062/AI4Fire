"""Meta-analysis of the AI4Fire 35-model benchmark sweep.

Answers five core questions (plus benchmark-wide discoveries) on what thirty-five models
reveal that six models could not:

1. Does grounding help or hurt as a function of bare skill?
   - Quantifies grounded-minus-bare delta vs bare score per task over 16 full-capability models
     and all available models (up to 35).
   - Computes Spearman rank correlation rho, p-value, and 95% bootstrap confidence interval.
   - Explains the profound divergence across tasks (allocation vs smoke detection vs fire danger).

2. Analogue copying against skill and extreme analogue copying.
   - Measures displayed analogue median ratio copy rate across all 35 grounded allocation models.
   - Relates copy rate to bare normalized error (nMAE) and grounded-minus-bare delta nMAE.
   - Identifies and counts every extreme copy across all models where predictions duplicate
     displayed analogue values onto incidents of wildly different scale (e.g. copying 866 personnel
     onto an incident staffed by two).

3. Persistence reproduction on bare allocation.
   - Computes per-model exact persistence reproduction rates (prediction == persistence).
   - Correlates persistence copy rate with bare normalized error (nMAE).
   - Resolves whether low error among weaker models reflects predictive skill or trivial rule reproduction.

4. Within-family size contrasts across five model families.
   - Tabulates task scores across Gemma 3 (27B, 12B, 4B), Amazon Nova (Pro, Lite, 2 Lite, Micro),
     Llama 4 (Maverick, Scout), GPT-OSS (120B, 20B), and Z.AI GLM (5, 4.7, 4.7 Flash).
   - Evaluates monotonicity per task and identifies where architectural or generational differences
     confound naive scaling comparisons.

5. Open-weight against proprietary model comparison.
   - Compares the 16 full-capability models (7 proprietary, 9 open-weight) on all five benchmark tasks.
   - Evaluates Mann-Whitney U rank tests, rank-biserial correlations, and 95% bootstrap intervals
     on median differences.
   - Formulates the definitive verdict on whether the sweep supports an open-proprietary performance gap.

Additional findings:
   - Tool use capability cliff (<20B failure threshold vs larger open models matching proprietary).
   - Reasoning model token blowouts (MiniMax M2.5 truncations under 8192 cap).
   - False-positive rate inflation under grounding in smoke detection.
   - Benchmark-wide failure to surpass simple physical or persistence baselines.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "figures") not in sys.path:
    sys.path.insert(0, str(ROOT / "figures"))

import figstyle as fs
import models


def bootstrap_ci(
    func,
    args: Tuple[np.ndarray, ...],
    n_boot: int = 10000,
    seed: int = 42,
    ci: float = 95.0,
) -> Tuple[float, float]:
    """Calculate bootstrap confidence interval using percentile method."""
    rng = np.random.default_rng(seed)
    n = len(args[0])
    estimates = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        resampled_args = tuple(a[idx] for a in args)
        val = func(*resampled_args)
        if val is not None and not np.isnan(val):
            estimates.append(val)
    if not estimates:
        return float("nan"), float("nan")
    lower = (100.0 - ci) / 2.0
    upper = 100.0 - lower
    return float(np.percentile(estimates, lower)), float(np.percentile(estimates, upper))


def spearman_func(x: np.ndarray, y: np.ndarray) -> float:
    if len(np.unique(x)) <= 1 or len(np.unique(y)) <= 1:
        return float("nan")
    r, _ = stats.spearmanr(x, y)
    return float(r)


def median_diff_func(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.median(x) - np.median(y))


def load_sweep_data() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    sweep_path = ROOT / "analysis" / "model_sweep.json"
    with open(sweep_path, encoding="utf-8") as f:
        data = json.load(f)
    return data["models"], data.get("tallies", {})


def load_analogue_cache() -> Tuple[Dict[str, Tuple[float, float]], Dict[str, List[str]]]:
    """Analogue staffing pairs, from the local cache when it exists and from the task sources otherwise.

    The cache is gitignored, so a fresh clone rebuilds it here from the recorded task inputs the way
    copy_agreement.py does. No model call is involved either way.
    """
    cache_path = ROOT / "analysis" / ".analogue-pool-cache.json"
    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            cache = json.load(f)
        flat = {k: (float(v[0]), float(v[1])) for k, v in cache.get("flat", {}).items()}
        return flat, cache.get("drawn", {})

    import run_allocation as ra

    items = ra.sample_items()
    pool = ra.build_pool({i["incident_id"] for i in items})
    flat = {r["analogue_id"]: (float(r["today"]), float(r["next"])) for rows in pool.values() for r in rows}
    drawn = {i["item_id"]: list(i.get("analogue_ids", [])) for i in items if i.get("analogue_ids")}
    try:
        cache_path.write_text(json.dumps({"key": "rebuilt from task sources", "flat":
                                          {k: list(v) for k, v in flat.items()}, "drawn": drawn}), encoding="utf-8")
    except OSError:
        pass
    return flat, drawn



def shared_baseline_null(bare: np.ndarray, grounded: np.ndarray, draws: int = 5000,
                         seed: int = 20260915) -> Dict[str, Any]:
    """Correlation of bare with (grounded - bare) when grounded carries no information about bare.

    Both variables contain the bare score, so the statistic is negative by construction. Permuting the
    grounded scores across models gives the distribution to read the observed value against.
    """
    rng = np.random.default_rng(seed)
    vals = np.empty(draws, dtype=float)
    for i in range(draws):
        permuted = rng.permutation(grounded)
        vals[i] = stats.spearmanr(bare, permuted - bare).statistic
    lo, hi = np.percentile(vals, [2.5, 97.5])
    observed = stats.spearmanr(bare, grounded - bare).statistic
    return {
        "draws": draws,
        "seed": seed,
        "median": float(np.median(vals)),
        "central_range_95": [float(lo), float(hi)],
        "observed": float(observed),
        "observed_inside_null": bool(lo <= observed <= hi),
    }


def _spearman_pair(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Spearman correlation of the two arms themselves, which shares no term between the variables."""
    res = stats.spearmanr(x, y)
    return {"rho": float(res.statistic), "p_value": float(res.pvalue)}

def question_1_grounding_vs_skill(models_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Question 1: Grounded-minus-bare delta vs bare score per task."""
    full_models = [m for m in models_list if m["tier"] in ("core", "added")]
    tasks_config = {
        "allocation": {
            "metric": "nmae",
            "higher_is_better": False,
            "arms": ("bare", "grounded"),
            "groups": [("16_full", full_models), ("35_all", models_list)],
        },
        "mesogeos": {
            "metric": "auprc",
            "higher_is_better": True,
            "arms": ("bare", "grounded"),
            "groups": [("16_full", full_models), ("35_all", models_list)],
        },
        "figlib": {
            "metric": "recall",
            "higher_is_better": True,
            "arms": ("bare", "grounded"),
            "groups": [("16_full", full_models)],
        },
        "tooluse": {
            "metric": "accuracy",
            "higher_is_better": True,
            "arms": ("bare", "tool"),
            "groups": [("16_full", full_models), ("33_all", [m for m in models_list if "tooluse" in m["tasks"]])],
        },
        "wildfirevqa": {
            "metric": "accuracy",
            "higher_is_better": True,
            "arms": ("bare", "grounded"),
            "groups": [("16_full", full_models)],
        },
    }

    out = {}
    for task_name, cfg in tasks_config.items():
        out[task_name] = {}
        metric = cfg["metric"]
        arm_b, arm_g = cfg["arms"]
        for grp_name, group in cfg["groups"]:
            bare_scores = []
            grounded_scores = []
            model_labels = []
            for m in group:
                if task_name in m["tasks"]:
                    b_val = m["tasks"][task_name][arm_b][metric]
                    g_val = m["tasks"][task_name][arm_g][metric]
                    bare_scores.append(float(b_val))
                    grounded_scores.append(float(g_val))
                    model_labels.append(m["label"])
            bare_arr = np.array(bare_scores, dtype=float)
            grnd_arr = np.array(grounded_scores, dtype=float)
            deltas = grnd_arr - bare_arr

            rho, pval = stats.spearmanr(bare_arr, deltas)
            ci_low, ci_high = bootstrap_ci(spearman_func, (bare_arr, deltas), seed=42)
            null = shared_baseline_null(bare_arr, grnd_arr)

            out[task_name][grp_name] = {
                "n_models": len(model_labels),
                "metric": metric,
                "higher_is_better": cfg["higher_is_better"],
                "bare_min": float(np.min(bare_arr)),
                "bare_max": float(np.max(bare_arr)),
                "bare_mean": float(np.mean(bare_arr)),
                "bare_median": float(np.median(bare_arr)),
                "delta_min": float(np.min(deltas)),
                "delta_max": float(np.max(deltas)),
                "delta_mean": float(np.mean(deltas)),
                "delta_median": float(np.median(deltas)),
                "spearman_rho": float(rho),
                "p_value": float(pval),
                "ci_95": [ci_low, ci_high],
                "shared_baseline_null": null,
                "rho_bare_vs_grounded": _spearman_pair(bare_arr, grnd_arr),
                "grounded_improved_count": int(
                    np.sum(deltas < 0) if not cfg["higher_is_better"] else np.sum(deltas > 0)
                ),
                "grounded_hurt_count": int(
                    np.sum(deltas > 0) if not cfg["higher_is_better"] else np.sum(deltas < 0)
                ),
            }
    return out


def question_2_analogue_copying(
    models_list: List[Dict[str, Any]], flat_analogues: Dict[str, Tuple[float, float]]
) -> Dict[str, Any]:
    """Question 2: Copy agreement with displayed analogue median and extreme copying."""
    model_copy_data = {}
    extreme_copies = {}

    for m in models_list:
        if "allocation" not in m["tasks"]:
            continue
        lbl = m["label"]
        stem = m["stem"]
        path = ROOT / "task-allocation" / f"responses-{stem}-grounded.jsonl"
        if not path.exists():
            continue
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]

        same, total = 0, 0
        model_extreme = []
        for r in rows:
            pred = r.get("prediction")
            a_ids = r.get("analogue_ids", [])
            if pred is None or not a_ids:
                continue
            ratios = [flat_analogues[a][1] / max(flat_analogues[a][0], 1.0) for a in a_ids if a in flat_analogues]
            if not ratios:
                continue
            shown_med = float("%.2f" % float(np.median(ratios)))
            rule_pred = int(round(r["persistence"] * shown_med))
            total += 1
            if pred == rule_pred:
                same += 1

            # Extreme copy check: prediction matches a displayed analogue outcome
            # where the incident is low-staffed (pers <= 5) and the analogue is high (>= 100)
            displayed_pairs = [flat_analogues[a] for a in a_ids if a in flat_analogues]
            for p_today, p_next in displayed_pairs:
                if pred == p_next and p_next >= 100 and r["persistence"] <= 5:
                    model_extreme.append({
                        "item_id": r["item_id"],
                        "prediction": float(pred),
                        "persistence": float(r["persistence"]),
                        "target": float(r["target"]),
                        "analogue_today": float(p_today),
                        "analogue_next": float(p_next),
                    })
                    break

        copy_rate = same / total if total > 0 else 0.0
        bare_nmae = m["tasks"]["allocation"]["bare"]["nmae"]
        grnd_nmae = m["tasks"]["allocation"]["grounded"]["nmae"]
        delta_nmae = grnd_nmae - bare_nmae

        model_copy_data[lbl] = {
            "stem": stem,
            "copy_count": same,
            "total_evaluated": total,
            "copy_rate": float(copy_rate),
            "bare_nmae": float(bare_nmae),
            "grounded_nmae": float(grnd_nmae),
            "delta_nmae": float(delta_nmae),
        }
        if model_extreme:
            extreme_copies[lbl] = model_extreme

    labels = sorted(model_copy_data.keys())
    cr_arr = np.array([model_copy_data[l]["copy_rate"] for l in labels], dtype=float)
    bn_arr = np.array([model_copy_data[l]["bare_nmae"] for l in labels], dtype=float)
    dn_arr = np.array([model_copy_data[l]["delta_nmae"] for l in labels], dtype=float)

    rho_bn, pval_bn = stats.spearmanr(cr_arr, bn_arr)
    ci_bn = bootstrap_ci(spearman_func, (cr_arr, bn_arr), seed=42)

    rho_dn, pval_dn = stats.spearmanr(cr_arr, dn_arr)
    ci_dn = bootstrap_ci(spearman_func, (cr_arr, dn_arr), seed=42)

    return {
        "per_model": model_copy_data,
        "extreme_copies": extreme_copies,
        "extreme_copies_summary": {lbl: len(ec) for lbl, ec in extreme_copies.items()},
        "copy_rate_range": [float(np.min(cr_arr)), float(np.max(cr_arr))],
        "copy_rate_mean": float(np.mean(cr_arr)),
        "copy_rate_median": float(np.median(cr_arr)),
        "correlation_with_bare_nmae": {
            "spearman_rho": float(rho_bn),
            "p_value": float(pval_bn),
            "ci_95": [ci_bn[0], ci_bn[1]],
        },
        "correlation_with_delta_nmae": {
            "spearman_rho": float(rho_dn),
            "p_value": float(pval_dn),
            "ci_95": [ci_dn[0], ci_dn[1]],
        },
    }


def question_3_persistence_reproduction(models_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Question 3: Persistence exact reproduction on bare allocation."""
    pers_data = {}
    for m in models_list:
        if "allocation" not in m["tasks"]:
            continue
        lbl = m["label"]
        stem = m["stem"]
        path = ROOT / "task-allocation" / f"responses-{stem}-bare.jsonl"
        if not path.exists():
            continue
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        ok = [r for r in rows if r.get("prediction") is not None]
        n_exact = sum(r["prediction"] == r["persistence"] for r in ok)
        rep_rate = n_exact / len(ok) if ok else 0.0
        bare_nmae = m["tasks"]["allocation"]["bare"]["nmae"]
        pers_data[lbl] = {
            "stem": stem,
            "persistence_copies": int(n_exact),
            "parsed_rows": len(ok),
            "copy_rate": float(rep_rate),
            "bare_nmae": float(bare_nmae),
        }

    labels = sorted(pers_data.keys())
    pr_arr = np.array([pers_data[l]["copy_rate"] for l in labels], dtype=float)
    bn_arr = np.array([pers_data[l]["bare_nmae"] for l in labels], dtype=float)

    rho, pval = stats.spearmanr(pr_arr, bn_arr)
    ci = bootstrap_ci(spearman_func, (pr_arr, bn_arr), seed=42)

    sorted_by_rate = sorted(pers_data.items(), key=lambda x: x[1]["copy_rate"], reverse=True)

    return {
        "per_model": pers_data,
        "copy_rate_range": [float(np.min(pr_arr)), float(np.max(pr_arr))],
        "copy_rate_mean": float(np.mean(pr_arr)),
        "copy_rate_median": float(np.median(pr_arr)),
        "correlation_with_bare_nmae": {
            "spearman_rho": float(rho),
            "p_value": float(pval),
            "ci_95": [ci[0], ci[1]],
        },
        "top_copiers": [
            {"label": lbl, "copy_rate": d["copy_rate"], "count": d["persistence_copies"], "bare_nmae": d["bare_nmae"]}
            for lbl, d in sorted_by_rate[:10]
        ],
        "lowest_copiers": [
            {"label": lbl, "copy_rate": d["copy_rate"], "count": d["persistence_copies"], "bare_nmae": d["bare_nmae"]}
            for lbl, d in sorted_by_rate[-10:]
        ],
    }


def question_4_family_size_contrasts(models_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Question 4: Within-family size contrasts across five families."""
    by_lbl = {m["label"]: m for m in models_list}
    families_def = {
        "Gemma 3": {
            "order": ["Gemma 3 27B", "Gemma 3 12B", "Gemma 3 4B"],
            "notes": "Three sizes of same generation and architecture. Gemma 12B and 4B fail tool use entirely.",
        },
        "Amazon Nova": {
            "order": ["Nova Pro", "Nova Lite", "Nova 2 Lite", "Nova Micro"],
            "notes": "Nova Pro and Lite are v1 siblings (Pro > Lite). Nova 2 Lite is a v2 release with catastrophic 866 copy. Nova Micro is text-only.",
        },
        "Llama 4": {
            "order": ["Llama 4 Maverick", "Llama 4 Scout"],
            "notes": "Both 17B active MoE models; Maverick has larger total parameters. Two-point contrast, not a scaling law.",
        },
        "GPT-OSS": {
            "order": ["GPT-OSS 120B", "GPT-OSS 20B"],
            "notes": "Two sizes of open weights; text-only tier.",
        },
        "Z.AI GLM": {
            "order": ["GLM 5", "GLM 4.7", "GLM 4.7 Flash"],
            "notes": "GLM 5 (flagship) > GLM 4.7 (standard) > GLM 4.7 Flash (distilled). Flash inverts allocation via 98% persistence copy.",
        },
    }

    results = {}
    for fam_name, info in families_def.items():
        fam_scores = {}
        for m_name in info["order"]:
            m = by_lbl.get(m_name)
            if not m:
                continue
            fam_scores[m_name] = {}
            for tname, tdata in m["tasks"].items():
                if tname == "allocation":
                    fam_scores[m_name]["allocation_bare_nmae"] = tdata["bare"]["nmae"]
                    fam_scores[m_name]["allocation_grounded_nmae"] = tdata["grounded"]["nmae"]
                elif tname == "mesogeos":
                    fam_scores[m_name]["mesogeos_bare_auprc"] = tdata["bare"]["auprc"]
                    fam_scores[m_name]["mesogeos_grounded_auprc"] = tdata["grounded"]["auprc"]
                elif tname == "figlib":
                    fam_scores[m_name]["figlib_bare_recall"] = tdata["bare"]["recall"]
                    fam_scores[m_name]["figlib_grounded_recall"] = tdata["grounded"]["recall"]
                elif tname == "tooluse":
                    fam_scores[m_name]["tooluse_bare_accuracy"] = tdata["bare"]["accuracy"]
                    fam_scores[m_name]["tooluse_tool_accuracy"] = tdata["tool"]["accuracy"]
                elif tname == "wildfirevqa":
                    fam_scores[m_name]["wildfirevqa_bare_accuracy"] = tdata["bare"]["accuracy"]
                    fam_scores[m_name]["wildfirevqa_grounded_accuracy"] = tdata["grounded"]["accuracy"]

        # Monotonicity check across metrics
        metrics_monotonicity = {}
        all_metrics = set()
        for s in fam_scores.values():
            all_metrics.update(s.keys())

        for met in sorted(all_metrics):
            vals = [fam_scores[m_name][met] for m_name in info["order"] if met in fam_scores[m_name]]
            if len(vals) < 2:
                continue
            is_nmae = "nmae" in met
            # For nmae: smaller is better, so decreasing order in model size means vals should be increasing
            # For auprc/recall/accuracy: larger is better, so decreasing order in model size means vals should be decreasing
            is_mono = True
            for i in range(len(vals) - 1):
                if is_nmae:
                    if vals[i] > vals[i + 1]:  # larger model has worse error
                        is_mono = False
                        break
                else:
                    if vals[i] < vals[i + 1]:  # larger model has worse score
                        is_mono = False
                        break
            metrics_monotonicity[met] = {
                "values": vals,
                "monotone_in_size": is_mono,
            }

        results[fam_name] = {
            "notes": info["notes"],
            "models_ordered": info["order"],
            "scores": fam_scores,
            "monotonicity": metrics_monotonicity,
        }
    return results


def question_5_open_vs_proprietary(models_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Question 5: Open-weight against proprietary on 16 full-capability models."""
    full_models = [m for m in models_list if m["tier"] in ("core", "added")]
    prop_models = [m for m in full_models if m["weights"] == "proprietary"]
    open_models = [m for m in full_models if m["weights"] == "open"]

    tasks_arms = [
        ("allocation", "bare", "nmae", False),
        ("allocation", "grounded", "nmae", False),
        ("mesogeos", "bare", "auprc", True),
        ("mesogeos", "grounded", "auprc", True),
        ("figlib", "bare", "recall", True),
        ("figlib", "grounded", "recall", True),
        ("tooluse", "bare", "accuracy", True),
        ("tooluse", "tool", "accuracy", True),
        ("wildfirevqa", "bare", "accuracy", True),
        ("wildfirevqa", "grounded", "accuracy", True),
    ]

    tests = {}
    for task_name, arm, metric, higher_better in tasks_arms:
        prop_vals = np.array([m["tasks"][task_name][arm][metric] for m in prop_models], dtype=float)
        open_vals = np.array([m["tasks"][task_name][arm][metric] for m in open_models], dtype=float)

        u_stat, pval = stats.mannwhitneyu(prop_vals, open_vals, alternative="two-sided")
        n_p, n_o = len(prop_vals), len(open_vals)
        rank_biserial_r = 1.0 - (2.0 * u_stat) / (n_p * n_o)

        med_p = float(np.median(prop_vals))
        med_o = float(np.median(open_vals))
        diff = med_p - med_o

        # Bootstrap interval for median difference (prop - open)
        rng = np.random.default_rng(42)
        boot_diffs = []
        for _ in range(10000):
            bp = rng.choice(prop_vals, size=n_p, replace=True)
            bo = rng.choice(open_vals, size=n_o, replace=True)
            boot_diffs.append(float(np.median(bp) - np.median(bo)))
        ci_diff = [float(np.percentile(boot_diffs, 2.5)), float(np.percentile(boot_diffs, 97.5))]

        key = f"{task_name}_{arm}_{metric}"
        tests[key] = {
            "task": task_name,
            "arm": arm,
            "metric": metric,
            "higher_is_better": higher_better,
            "n_proprietary": n_p,
            "n_open": n_o,
            "prop_median": med_p,
            "open_median": med_o,
            "median_difference": diff,
            "ci_95_median_diff": ci_diff,
            "mann_whitney_u": float(u_stat),
            "p_value": float(pval),
            "rank_biserial_r": float(rank_biserial_r),
            "statistically_significant_05": bool(pval < 0.05),
        }

    return {
        "proprietary_models": [m["label"] for m in prop_models],
        "open_models": [m["label"] for m in open_models],
        "tests": tests,
    }


def additional_discoveries(models_list: List[Dict[str, Any]], tallies_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Additional findings not covered in the five primary questions."""
    # 1. Tool use capability threshold
    tool_models = [m for m in models_list if "tooluse" in m["tasks"]]
    zero_tool = [m["label"] for m in tool_models if m["tasks"]["tooluse"]["tool"]["accuracy"] == 0.0]
    perfect_tool = [m["label"] for m in tool_models if m["tasks"]["tooluse"]["tool"]["accuracy"] >= 0.99]

    # 2. Reasoning model output cap truncations
    trunc_models = tallies_dict.get("truncated_rows", {})

    # 3. Smoke FPR explosion
    full_models = [m for m in models_list if m["tier"] in ("core", "added")]
    fig_bare_fpr = [m["tasks"]["figlib"]["bare"]["fpr"] for m in full_models if "figlib" in m["tasks"]]
    fig_grnd_fpr = [m["tasks"]["figlib"]["grounded"]["fpr"] for m in full_models if "figlib" in m["tasks"]]
    fpr_delta = [g - b for g, b in zip(fig_grnd_fpr, fig_bare_fpr)]

    return {
        "tool_use_cliff": {
            "zero_accuracy_models": zero_tool,
            "perfect_or_near_perfect_models": perfect_tool,
            "glm5_tool_accuracy": next(
                m["tasks"]["tooluse"]["tool"]["accuracy"] for m in models_list if m["label"] == "GLM 5"
            ),
        },
        "reasoning_model_truncations": trunc_models,
        "smoke_fpr_inflation": {
            "bare_fpr_median": float(np.median(fig_bare_fpr)),
            "grounded_fpr_median": float(np.median(fig_grnd_fpr)),
            "fpr_delta_median": float(np.median(fpr_delta)),
            "models_with_increased_fpr": int(np.sum(np.array(fpr_delta) > 0)),
            "total_models": len(fpr_delta),
        },
        "benchmark_wide_baselines": {
            "models_beating_persistence_bare_alloc": tallies_dict.get("allocation", {}).get("bare_beats_persistence", 0),
            "models_beating_persistence_grounded_alloc": tallies_dict.get("allocation", {}).get("grounded_beats_persistence", 0),
            "models_beating_temp_rule_mesogeos": tallies_dict.get("mesogeos", {}).get("bare_above_temperature_rule_0.654", 0),
            "models_beating_majority_vqa": tallies_dict.get("wildfirevqa", {}).get("grounded_above_majority", 0),
        },
    }


def generate_figure(pers_data: Dict[str, Any], out_dir: Path) -> None:
    """Write figures/sweep_meta.{pdf,png}: bare allocation error against persistence reproduction.

    The point of the panel is that the models closest to the persistence baseline are the ones that
    reproduce it, so their low error is not skill. A point beyond the axis is clipped at the edge and
    labelled, the idiom figures/make_grounding_effects.py uses.
    """
    fs.apply()
    fig, ax = plt.subplots(figsize=(fs.TEXT_WIDTH_IN * 0.72, 2.8))
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.19, top=0.86)
    fs.bare(ax, grid="both")

    pers_baseline = 0.14647
    y_top = 0.30
    ax.axhline(pers_baseline, color=fs.GRAY, linestyle="--", linewidth=0.8, zorder=1)
    ax.text(0.055, pers_baseline, "persistence 0.146", fontsize=fs.FS_SMALL, color=fs.SUBTITLE,
            va="bottom", ha="left")

    models_info = models.by_label()
    label_these = {
        "Gemma 3 4B": (-7, -3, "right"),
        "GLM 4.7 Flash": (-7, 5, "right"),
        "Llama 4 Maverick": (7, 2, "left"),
        "Nova Lite": (-7, 0, "right"),
    }
    off_axis = 0  # stagger the labels of the points that sit beyond the axis
    for lbl, d in pers_data["per_model"].items():
        m = models_info.get(lbl)
        color = fs.CORAL if (m and m.weights == "proprietary") else fs.PALETTE_OPEN[0]
        x, y = d["copy_rate"], d["bare_nmae"]
        if y > y_top:
            ax.plot(x, y_top, marker="^", markersize=4.5, color=color, zorder=4, clip_on=False)
            ax.annotate(f"{lbl} {y:.2f}", (x, y_top), xytext=(6, -1 - 9 * off_axis),
                        textcoords="offset points", fontsize=fs.FS_SMALL, color=fs.NEAR_BLACK,
                        ha="left", va="center")
            off_axis += 1
            continue
        ax.scatter(x, y, color=color, s=26, alpha=0.85, edgecolors="none", zorder=3)
        if lbl in label_these:
            dx, dy, ha = label_these[lbl]
            rate = d["copy_rate"] * 100
            shown = f"{rate:.1f}" if rate > 99 else f"{rate:.0f}"  # 99.7 and 100 are different claims
            ax.annotate(f"{lbl} ({shown}%)", (x, y), xytext=(dx, dy),
                        textcoords="offset points", fontsize=fs.FS_SMALL, color=fs.NEAR_BLACK, ha=ha)

    fs.panel_title(ax, "", "Bare allocation error against persistence reproduction")
    ax.set_xlabel("share of the 300 bare items predicted equal to persistence", fontsize=fs.FS_AXIS)
    ax.set_ylabel("bare normalized error", fontsize=fs.FS_AXIS)
    ax.set_xlim(0.05, 1.05)
    ax.set_ylim(0.13, y_top)

    rho = pers_data["correlation_with_bare_nmae"]["spearman_rho"]
    pval = pers_data["correlation_with_bare_nmae"]["p_value"]
    ax.text(0.98, 0.95, f"Spearman $\\rho = {rho:.3f}$, $p = {pval:.1g}$", transform=ax.transAxes,
            fontsize=fs.FS_SMALL, color=fs.SUBTITLE, ha="right", va="top")

    out_dir.mkdir(parents=True, exist_ok=True)
    fs.savefig(fig, out_dir / "sweep_meta.pdf", out_dir / "sweep_meta.png")
    plt.close(fig)
    print(f"Generated {out_dir / 'sweep_meta.png'} and {out_dir / 'sweep_meta.pdf'}")

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-json", type=Path, default=ROOT / "analysis" / "sweep_meta.json")
    parser.add_argument("--out-fig-dir", type=Path, default=ROOT / "figures")
    args = parser.parse_args()

    models_list, tallies_dict = load_sweep_data()
    flat_analogues, _ = load_analogue_cache()

    q1 = question_1_grounding_vs_skill(models_list)
    q2 = question_2_analogue_copying(models_list, flat_analogues)
    q3 = question_3_persistence_reproduction(models_list)
    q4 = question_4_family_size_contrasts(models_list)
    q5 = question_5_open_vs_proprietary(models_list)
    addl = additional_discoveries(models_list, tallies_dict)

    combined = {
        "question_1_grounding_vs_bare_skill": q1,
        "question_2_analogue_copying": q2,
        "question_3_persistence_reproduction": q3,
        "question_4_within_family_size": q4,
        "question_5_open_vs_proprietary": q5,
        "additional_discoveries": addl,
    }

    args.out_json.write_text(json.dumps(combined, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote meta-analysis results to {args.out_json}")

    generate_figure(q3, args.out_fig_dir)


if __name__ == "__main__":
    main()
