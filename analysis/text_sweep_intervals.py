"""Compute clustered uncertainty intervals for the nineteen text-only models.

Tasks and clustering units:
1. Personnel allocation (300 items):
   - Metric: Normalized Mean Absolute Error (NMAE), defined as absolute error divided
     by the incident's mean staffing over parsed rows.
   - Resampling unit: Incident (245 clusters, keyed by incident_id in task-allocation/items.jsonl).
   - Paired condition difference: grounded minus bare (and bare minus grounded).
   - Baseline comparison: persistence rule (normalized error 0.146).

2. Fire danger forecasting (386 items):
   - Metric: Average Precision (AUPRC), step-wise area under the precision-recall curve.
   - Resampling unit: Spatial-temporal block (1.0 degree lat/lon cell by calendar month,
     352 clusters).
   - Paired condition difference: grounded minus bare (and bare minus grounded).
   - Baseline comparison: temperature rule (AUPRC 0.654).

3. Fire data tool use (156 items, seventeen models with .tools == True):
   - Metric: Accuracy (share of correct answers, unparsed or truncated rows scored as incorrect).
   - Resampling unit: Question family (12 clusters, from task-tooluse/items.jsonl).
   - Paired condition difference: tool minus bare.

Truncation:
   Rows truncated at the output cap (usage.stop_reason in ('max_tokens', 'length')) with null
   answers are treated as unparsed (dropped in allocation and fire danger, scored wrong in tool use).
"""

from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import cluster_uncertainty as cu
import models


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--repo", default=str(ROOT), help="AI4Fire checkout root")
    parser.add_argument(
        "--resamples",
        type=int,
        default=20000,
        help="bootstrap replicates per interval (default: 20000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260915,
        help="base seed for deterministic draws (default: 20260915)",
    )
    parser.add_argument(
        "--ci", type=float, default=95.0, help="confidence interval percentile (default: 95.0)"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=HERE / "text_sweep_intervals.json",
        help="output JSON path",
    )
    return parser.parse_args()


def count_truncations(rows: List[dict], task: str) -> Tuple[int, int]:
    """Count (truncated_no_answer, truncated_with_answer)."""
    trunc_no_ans, trunc_with_ans = 0, 0
    for r in rows:
        stop = (r.get("usage") or {}).get("stop_reason")
        if stop in ("max_tokens", "length"):
            if task == "allocation":
                has_ans = r.get("prediction") is not None
            elif task == "mesogeos":
                has_ans = r.get("probability") is not None or r.get("call") is not None
            elif task == "tooluse":
                has_ans = r.get("prediction") is not None
            else:
                has_ans = False
            if has_ans:
                trunc_with_ans += 1
            else:
                trunc_no_ans += 1
    return trunc_no_ans, trunc_with_ans


def evaluate_allocation(m: models.Model, repo: Path, cfg: Any, items: dict) -> dict:
    task_dir = repo / "task-allocation"
    bare_path = task_dir / f"responses-{m.stem}-bare.jsonl"
    grd_path = task_dir / f"responses-{m.stem}-grounded.jsonl"

    bare_raw, _ = cu.read_jsonl(str(bare_path))
    grd_raw, _ = cu.read_jsonl(str(grd_path))
    bare_rows, _ = cu.dedupe_by_item(bare_raw)
    grd_rows, _ = cu.dedupe_by_item(grd_raw)

    t_b_no, t_b_with = count_truncations(bare_raw, "allocation")
    t_g_no, t_g_with = count_truncations(grd_raw, "allocation")

    prep = cu.prepare_allocation(items, bare_rows, grd_rows, cfg)
    n_pair = prep["n"]

    # Also compute persistence on the paired items
    common = sorted(set(bare_rows) & set(grd_rows))
    err_p = []
    for iid in common:
        rb, rg = bare_rows[iid], grd_rows[iid]
        if rb.get("prediction") is None or rg.get("prediction") is None:
            continue
        scale = rb.get("fire_mean", rg.get("fire_mean"))
        target = rb.get("target", rg.get("target"))
        pers = rb.get("persistence", rg.get("persistence"))
        err_p.append(abs(float(target) - float(pers)) / float(scale))
    err_p = np.asarray(err_p, dtype=float)

    full = np.arange(n_pair)
    pt_bare = float(prep["metric_bare"](full))
    pt_grd = float(prep["metric_grounded"](full))
    pt_pers = float(err_p.mean())
    pt_diff_gb = pt_grd - pt_bare
    pt_diff_bg = pt_bare - pt_grd
    pt_diff_bp = pt_bare - pt_pers

    seed = cu.stable_seed(cfg.seed, "task-allocation", m.stem, "cluster")
    flat, starts, sizes, keys = cu.build_cluster_index(prep["groups"])
    n_clusters = len(keys)
    rng = np.random.default_rng(seed)

    cl_b = np.empty(cfg.resamples, dtype=float)
    cl_g = np.empty(cfg.resamples, dtype=float)
    cl_p = np.empty(cfg.resamples, dtype=float)

    for b in range(cfg.resamples):
        draw = rng.integers(0, n_clusters, size=n_clusters)
        idx = cu.ragged_gather(flat, starts, sizes, draw)
        cl_b[b] = prep["metric_bare"](idx)
        cl_g[b] = prep["metric_grounded"](idx)
        cl_p[b] = err_p[idx].mean()

    lo_b, hi_b, _ = cu.percentile_interval(cl_b, cfg.ci)
    lo_g, hi_g, _ = cu.percentile_interval(cl_g, cfg.ci)
    lo_p, hi_p, _ = cu.percentile_interval(cl_p, cfg.ci)
    lo_gb, hi_gb, _ = cu.percentile_interval(cl_g - cl_b, cfg.ci)
    lo_bg, hi_bg, _ = cu.percentile_interval(cl_b - cl_g, cfg.ci)
    lo_bp, hi_bp, _ = cu.percentile_interval(cl_b - cl_p, cfg.ci)

    return {
        "n_bare": len(bare_rows),
        "n_grounded": len(grd_rows),
        "n_paired": n_pair,
        "n_clusters": n_clusters,
        "truncation": {
            "bare_no_ans": t_b_no,
            "bare_with_ans": t_b_with,
            "grounded_no_ans": t_g_no,
            "grounded_with_ans": t_g_with,
            "total_no_ans": t_b_no + t_g_no,
        },
        "bare": {"point": pt_bare, "ci": [lo_b, hi_b]},
        "grounded": {"point": pt_grd, "ci": [lo_g, hi_g]},
        "persistence": {"point": pt_pers, "ci": [lo_p, hi_p]},
        "grounded_minus_bare": {"point": pt_diff_gb, "ci": [lo_gb, hi_gb]},
        "bare_minus_grounded": {"point": pt_diff_bg, "ci": [lo_bg, hi_bg]},
        "bare_minus_persistence": {
            "point": pt_diff_bp,
            "ci": [lo_bp, hi_bp],
            "contains_zero": bool(lo_bp <= 0 <= hi_bp),
        },
    }


def evaluate_persistence_overall(repo: Path, cfg: Any, items: dict) -> dict:
    """Evaluate persistence rule on all 300 allocation items."""
    task_dir = repo / "task-allocation"
    # Read any valid response file to get target, persistence, fire_mean
    p = task_dir / "responses-bedrock_amazon.nova-micro-v1_0-bare.jsonl"
    rows, _ = cu.read_jsonl(str(p))
    err_p, groups = [], []
    for r in rows:
        iid = r["item_id"]
        target = r["target"]
        pers = r["persistence"]
        scale = r["fire_mean"]
        inc = (
            items[iid]["incident_id"]
            if (iid in items and "incident_id" in items[iid])
            else cu.allocation_incident_from_item_id(iid)
        )
        err_p.append(abs(float(target) - float(pers)) / float(scale))
        groups.append(inc)

    err_p = np.asarray(err_p, dtype=float)
    flat, starts, sizes, keys = cu.build_cluster_index(groups)
    n_clusters = len(keys)
    rng = np.random.default_rng(cu.stable_seed(cfg.seed, "task-allocation", "persistence", "cluster"))
    cl_p = np.empty(cfg.resamples, dtype=float)
    for b in range(cfg.resamples):
        draw = rng.integers(0, n_clusters, size=n_clusters)
        idx = cu.ragged_gather(flat, starts, sizes, draw)
        cl_p[b] = err_p[idx].mean()

    lo, hi, _ = cu.percentile_interval(cl_p, cfg.ci)
    return {
        "items": len(err_p),
        "clusters": n_clusters,
        "point": float(err_p.mean()),
        "ci": [lo, hi],
    }



def _last_valid(series):
    """The last observed value of a daily driver, which is what the temperature rule ranks."""
    for x in reversed(series or []):
        if x is not None:
            return float(x)
    return None


def _average_precision(labels, scores) -> float:
    order = np.argsort(-np.asarray(scores, dtype=float), kind="mergesort")
    y = np.asarray(labels, dtype=int)[order]
    tp = np.cumsum(y)
    precision = tp / np.arange(1, len(y) + 1)
    total = int(y.sum())
    return float((precision * y).sum() / total) if total else float("nan")


def temperature_rule_ap(items: dict, item_ids) -> float:
    """AUPRC of the raw last-day t2m ranking on the given items.

    This is the comparator reported as 0.654 on all 386 items (analysis/information_only_baselines.py,
    "raw last-day t2m, no climatology"). On a model's parsed subset the matched value differs, so a
    conditional score has to be read against the rule recomputed on that same subset.
    """
    ids = [i for i in item_ids if i in items]
    labels, scores = [], []
    for i in ids:
        x = _last_valid((items[i].get("context") or {}).get("daily", {}).get("t2m"))
        if x is None:
            continue
        labels.append(int(items[i]["label"]))
        scores.append(x)
    return _average_precision(labels, scores) if labels else float("nan")

def evaluate_mesogeos(m: models.Model, repo: Path, cfg: Any, items: dict) -> dict:
    task_dir = repo / "task-mesogeos"
    bare_path = task_dir / f"responses-{m.stem}-bare.jsonl"
    grd_path = task_dir / f"responses-{m.stem}-grounded.jsonl"

    bare_raw, _ = cu.read_jsonl(str(bare_path))
    grd_raw, _ = cu.read_jsonl(str(grd_path))
    bare_rows, _ = cu.dedupe_by_item(bare_raw)
    grd_rows, _ = cu.dedupe_by_item(grd_raw)

    t_b_no, t_b_with = count_truncations(bare_raw, "mesogeos")
    t_g_no, t_g_with = count_truncations(grd_raw, "mesogeos")

    prep = cu.prepare_mesogeos(items, bare_rows, grd_rows, cfg)
    n_pair = prep["n"]

    full = np.arange(n_pair)
    pt_bare = float(prep["metric_bare"](full))
    pt_grd = float(prep["metric_grounded"](full))
    pt_diff_gb = pt_grd - pt_bare
    pt_diff_bg = pt_bare - pt_grd

    seed = cu.stable_seed(cfg.seed, "task-mesogeos", m.stem, "cluster")
    cl_b, cl_g, n_clusters = cu.bootstrap_arms(
        prep["metric_bare"], prep["metric_grounded"], prep["groups"], prep["n"], cfg.resamples, seed
    )

    lo_b, hi_b, _ = cu.percentile_interval(cl_b, cfg.ci)
    lo_g, hi_g, _ = cu.percentile_interval(cl_g, cfg.ci)
    lo_gb, hi_gb, _ = cu.percentile_interval(cl_g - cl_b, cfg.ci)
    lo_bg, hi_bg, _ = cu.percentile_interval(cl_b - cl_g, cfg.ci)

    temp_rule = 0.654  # the rule on all 386 items, the value the main tables quote
    # the rule has to be scored on the items this model actually contributes, not on every shared row
    matched_rule = temperature_rule_ap(items, prep.get("kept") or sorted(set(bare_rows) & set(grd_rows)))
    return {
        "n_bare": len(bare_rows),
        "n_grounded": len(grd_rows),
        "n_paired": n_pair,
        "n_clusters": n_clusters,
        "truncation": {
            "bare_no_ans": t_b_no,
            "bare_with_ans": t_b_with,
            "grounded_no_ans": t_g_no,
            "grounded_with_ans": t_g_with,
            "total_no_ans": t_b_no + t_g_no,
        },
        "matched_temperature_rule": matched_rule,
        "bare": {
            "point": pt_bare,
            "ci": [lo_b, hi_b],
            "above_temperature_rule": bool(lo_b > temp_rule),
            "overlaps_temperature_rule": bool(lo_b <= temp_rule <= hi_b),
            "below_temperature_rule": bool(hi_b < temp_rule),
            "above_matched_rule": bool(lo_b > matched_rule),
            "overlaps_matched_rule": bool(lo_b <= matched_rule <= hi_b),
            "below_matched_rule": bool(hi_b < matched_rule),
        },
        "grounded": {
            "point": pt_grd,
            "ci": [lo_g, hi_g],
            "above_temperature_rule": bool(lo_g > temp_rule),
            "overlaps_temperature_rule": bool(lo_g <= temp_rule <= hi_g),
            "below_temperature_rule": bool(hi_g < temp_rule),
        },
        "grounded_minus_bare": {"point": pt_diff_gb, "ci": [lo_gb, hi_gb]},
        "bare_minus_grounded": {"point": pt_diff_bg, "ci": [lo_bg, hi_bg]},
    }


def evaluate_tooluse(m: models.Model, repo: Path, cfg: Any, items: List[dict]) -> dict:
    if not m.tools:
        return None

    task_dir = repo / "task-tooluse"
    bare_path = task_dir / f"responses-{m.stem}-bare.jsonl"
    tool_path = task_dir / f"responses-{m.stem}-tool.jsonl"

    bare_raw, _ = cu.read_jsonl(str(bare_path))
    tool_raw, _ = cu.read_jsonl(str(tool_path))
    bare_rows = {r["item_id"]: r for r in bare_raw if "item_id" in r}
    tool_rows = {r["item_id"]: r for r in tool_raw if "item_id" in r}

    t_b_no, t_b_with = count_truncations(bare_raw, "tooluse")
    t_t_no, t_t_with = count_truncations(tool_raw, "tooluse")

    by_id = {i["item_id"]: i for i in items}
    ids = [i["item_id"] for i in items if i["item_id"] in bare_rows and i["item_id"] in tool_rows]
    groups = [by_id[i]["family"] for i in ids]

    cb = np.array([1.0 if bare_rows[i].get("correct") is True else 0.0 for i in ids])
    ct = np.array([1.0 if tool_rows[i].get("correct") is True else 0.0 for i in ids])

    pt_bare = float(cb.mean())
    pt_tool = float(ct.mean())
    pt_diff = pt_tool - pt_bare

    flat, starts, sizes, keys = cu.build_cluster_index(groups)
    n_clusters = len(keys)
    seed = cu.stable_seed(cfg.seed, m.stem, "tool-bare")
    rng = np.random.default_rng(seed)

    cl_b = np.empty(cfg.resamples, dtype=float)
    cl_t = np.empty(cfg.resamples, dtype=float)
    cl_d = np.empty(cfg.resamples, dtype=float)

    for r in range(cfg.resamples):
        draw = rng.integers(0, n_clusters, size=n_clusters)
        idx = cu.ragged_gather(flat, starts, sizes, draw)
        cl_b[r] = cb[idx].mean()
        cl_t[r] = ct[idx].mean()
        cl_d[r] = cl_t[r] - cl_b[r]

    lo_b, hi_b, _ = cu.percentile_interval(cl_b, cfg.ci)
    lo_t, hi_t, _ = cu.percentile_interval(cl_t, cfg.ci)
    lo_d, hi_d, _ = cu.percentile_interval(cl_d, cfg.ci)

    excludes_zero = bool(lo_d > 0 or hi_d < 0)
    excludes_zero_down = bool(hi_d < 0)
    excludes_zero_up = bool(lo_d > 0)
    overlaps_zero = bool(lo_d <= 0 <= hi_d)

    return {
        "items": len(ids),
        "clusters": n_clusters,
        "truncation": {
            "bare_no_ans": t_b_no,
            "bare_with_ans": t_b_with,
            "tool_no_ans": t_t_no,
            "tool_with_ans": t_t_with,
            "total_no_ans": t_b_no + t_t_no,
        },
        "bare": {"point": pt_bare, "ci": [lo_b, hi_b]},
        "tool": {"point": pt_tool, "ci": [lo_t, hi_t]},
        "tool_minus_bare": {
            "point": pt_diff,
            "ci": [lo_d, hi_d],
            "excludes_zero": excludes_zero,
            "excludes_zero_down": excludes_zero_down,
            "excludes_zero_up": excludes_zero_up,
            "overlaps_zero": overlaps_zero,
        },
    }


def format_table_rows(records: List[dict]) -> List[str]:
    lines = []
    for r in records:
        lbl = r["label"]
        cap = f"{r['max_out']:,}"
        a_b = f"{r['allocation']['bare']['point']:.3f} [{r['allocation']['bare']['ci'][0]:.3f}, {r['allocation']['bare']['ci'][1]:.3f}]"
        a_g = f"{r['allocation']['grounded']['point']:.3f} [{r['allocation']['grounded']['ci'][0]:.3f}, {r['allocation']['grounded']['ci'][1]:.3f}]"
        m_b = f"{r['mesogeos']['bare']['point']:.3f} [{r['mesogeos']['bare']['ci'][0]:.3f}, {r['mesogeos']['bare']['ci'][1]:.3f}]"
        m_g = f"{r['mesogeos']['grounded']['point']:.3f} [{r['mesogeos']['grounded']['ci'][0]:.3f}, {r['mesogeos']['grounded']['ci'][1]:.3f}]"

        if r["tooluse"] is not None:
            t_b = f"{r['tooluse']['bare']['point']:.3f}"
            t_t = f"{r['tooluse']['tool']['point']:.3f}"
            t_diff = f"{r['tooluse']['tool_minus_bare']['point']:+.3f} [{r['tooluse']['tool_minus_bare']['ci'][0]:+.3f}, {r['tooluse']['tool_minus_bare']['ci'][1]:+.3f}]"
        else:
            t_b = "--"
            t_t = "--"
            t_diff = "--"

        row_str = f"{lbl} & {cap} & {a_b} & {a_g} & {m_b} & {m_g} & {t_b} & {t_t} & {t_diff} \\\\"
        lines.append(row_str)
    return lines


def main():
    args = parse_args()
    repo = Path(args.repo)

    # Configure cu config mock
    cfg = type(
        "Cfg",
        (),
        {
            "repo": str(repo),
            "resamples": args.resamples,
            "seed": args.seed,
            "ci": args.ci,
            "min_clusters": 5,
            "min_pair": 30,
            "meso_deg": 1.0,
            "meso_time": "month",
            "meso_coarse_deg": 2.0,
            "meso_coarse_time": "quarter",
        },
    )()

    alloc_items = cu.load_items(str(repo / "task-allocation"), "task-allocation", [])
    meso_items = cu.load_items(str(repo / "task-mesogeos"), "task-mesogeos", [])
    tool_items_path = repo / "task-tooluse" / "items.jsonl"
    tool_items = [json.loads(l) for l in tool_items_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    persistence_overall = evaluate_persistence_overall(repo, cfg, alloc_items)

    text_models = models.models(tier="text", root=repo)

    model_records = []
    trunc_summary = {}

    for m in text_models:
        print(f"Processing {m.label}...", file=sys.stderr)
        alloc_res = evaluate_allocation(m, repo, cfg, alloc_items)
        meso_res = evaluate_mesogeos(m, repo, cfg, meso_items)
        tool_res = evaluate_tooluse(m, repo, cfg, tool_items)

        total_trunc_no_ans = alloc_res["truncation"]["total_no_ans"] + meso_res["truncation"]["total_no_ans"]
        if tool_res:
            total_trunc_no_ans += tool_res["truncation"]["total_no_ans"]

        trunc_summary[m.label] = {
            "allocation_bare_no_ans": alloc_res["truncation"]["bare_no_ans"],
            "allocation_grounded_no_ans": alloc_res["truncation"]["grounded_no_ans"],
            "mesogeos_bare_no_ans": meso_res["truncation"]["bare_no_ans"],
            "mesogeos_grounded_no_ans": meso_res["truncation"]["grounded_no_ans"],
            "tooluse_bare_no_ans": tool_res["truncation"]["bare_no_ans"] if tool_res else 0,
            "tooluse_tool_no_ans": tool_res["truncation"]["tool_no_ans"] if tool_res else 0,
            "total_truncated_no_answer": total_trunc_no_ans,
            "with_answer_truncated": (
                alloc_res["truncation"]["bare_with_ans"]
                + alloc_res["truncation"]["grounded_with_ans"]
                + meso_res["truncation"]["bare_with_ans"]
                + meso_res["truncation"]["grounded_with_ans"]
                + (tool_res["truncation"]["bare_with_ans"] + tool_res["truncation"]["tool_with_ans"] if tool_res else 0)
            ),
        }

        rec = {
            "label": m.label,
            "stem": m.stem,
            "vendor": m.vendor,
            "weights": m.weights,
            "order": m.order,
            "max_out": m.max_out,
            "tools": m.tools,
            "allocation": alloc_res,
            "mesogeos": meso_res,
            "tooluse": tool_res,
        }
        model_records.append(rec)

    # Persistence copies check (the 5 models)
    pers_5_stems = [
        "bedrock_qwen.qwen3-coder-30b-a3b-v1_0",
        "bedrock_google.gemma-3-4b-it",
        "bedrock_qwen.qwen3-32b-v1_0",
        "bedrock_nvidia.nemotron-super-3-120b",
        "bedrock_zai.glm-4.7-flash",
    ]
    pers_5_results = {}
    for stem in pers_5_stems:
        m_entry = models.by_stem().get(stem)
        lbl = m_entry.label if m_entry else stem
        # If model was in text sweep, fetch from model_records
        match = next((r for r in model_records if r["stem"] == stem), None)
        if match:
            bp = match["allocation"]["bare_minus_persistence"]
            pers_5_results[lbl] = {
                "stem": stem,
                "copies": 294 if "glm" in stem or "nemotron" in stem else (297 if "32b" in stem else 299),
                "bare_nmae": match["allocation"]["bare"]["point"],
                "persistence_nmae": match["allocation"]["persistence"]["point"],
                "diff_bare_minus_persistence": bp["point"],
                "ci": bp["ci"],
                "contains_zero": bp["contains_zero"],
            }
        else:
            # Gemma 3 4B is in tier added; run allocation bootstrap directly
            b_path = repo / "task-allocation" / f"responses-{stem}-bare.jsonl"
            g_path = repo / "task-allocation" / f"responses-{stem}-grounded.jsonl"
            b_raw, _ = cu.read_jsonl(str(b_path))
            g_raw, _ = cu.read_jsonl(str(g_path))
            b_rows, _ = cu.dedupe_by_item(b_raw)
            g_rows, _ = cu.dedupe_by_item(g_raw)
            prep = cu.prepare_allocation(alloc_items, b_rows, g_rows, cfg)
            common = sorted(set(b_rows) & set(g_rows))
            err_p = []
            for iid in common:
                rb = b_rows[iid]
                if rb.get("prediction") is None:
                    continue
                scale = rb.get("fire_mean")
                target = rb.get("target")
                pers = rb.get("persistence")
                err_p.append(abs(float(target) - float(pers)) / float(scale))
            err_p = np.asarray(err_p, dtype=float)
            flat, starts, sizes, keys = cu.build_cluster_index(prep["groups"])
            rng = np.random.default_rng(cu.stable_seed(cfg.seed, "task-allocation", stem, "cluster"))
            cl_b = np.empty(cfg.resamples, dtype=float)
            cl_p = np.empty(cfg.resamples, dtype=float)
            for b in range(cfg.resamples):
                draw = rng.integers(0, len(keys), size=len(keys))
                idx = cu.ragged_gather(flat, starts, sizes, draw)
                cl_b[b] = prep["metric_bare"](idx)
                cl_p[b] = err_p[idx].mean()
            lo_bp, hi_bp, _ = cu.percentile_interval(cl_b - cl_p, cfg.ci)
            pt_b = float(prep["metric_bare"](np.arange(prep["n"])))
            pt_p = float(err_p.mean())
            pers_5_results[lbl] = {
                "stem": stem,
                "copies": 299,
                "bare_nmae": pt_b,
                "persistence_nmae": pt_p,
                "diff_bare_minus_persistence": pt_b - pt_p,
                "ci": [lo_bp, hi_bp],
                "contains_zero": bool(lo_bp <= 0 <= hi_bp),
            }

    # Tallies and questions
    alloc_below_pers_pt = sum(1 for r in model_records if r["allocation"]["bare"]["ci"][1] < 0.146467)
    alloc_overlap_pers_pt = sum(
        1 for r in model_records if r["allocation"]["bare"]["ci"][0] <= 0.146467 <= r["allocation"]["bare"]["ci"][1]
    )
    alloc_above_pers_pt = sum(1 for r in model_records if r["allocation"]["bare"]["ci"][0] > 0.146467)

    meso_above_rule = sum(1 for r in model_records if r["mesogeos"]["bare"]["above_temperature_rule"])
    meso_overlap_rule = sum(1 for r in model_records if r["mesogeos"]["bare"]["overlaps_temperature_rule"])
    meso_below_rule = sum(1 for r in model_records if r["mesogeos"]["bare"]["below_temperature_rule"])

    tool_models = [r for r in model_records if r["tooluse"] is not None]
    tool_excl_zero = sum(1 for r in tool_models if r["tooluse"]["tool_minus_bare"]["excludes_zero"])
    tool_excl_zero_down = sum(1 for r in tool_models if r["tooluse"]["tool_minus_bare"]["excludes_zero_down"])
    tool_excl_zero_up = sum(1 for r in tool_models if r["tooluse"]["tool_minus_bare"]["excludes_zero_up"])
    tool_overlap_zero = sum(1 for r in tool_models if r["tooluse"]["tool_minus_bare"]["overlaps_zero"])

    alloc_paired_above = sum(1 for r in model_records
                             if r["allocation"]["bare_minus_persistence"]["ci"][0] > 0)
    alloc_paired_below = sum(1 for r in model_records
                             if r["allocation"]["bare_minus_persistence"]["ci"][1] < 0)
    alloc_paired_zero = len(model_records) - alloc_paired_above - alloc_paired_below
    meso_above_matched = sum(1 for r in model_records if r["mesogeos"]["bare"]["above_matched_rule"])
    meso_overlap_matched = sum(1 for r in model_records if r["mesogeos"]["bare"]["overlaps_matched_rule"])
    meso_below_matched = sum(1 for r in model_records if r["mesogeos"]["bare"]["below_matched_rule"])

    tallies = {
        # Matched comparisons: model and comparator scored on the same items. Prefer these.
        "allocation_paired_bare_minus_persistence": {
            "above_zero_worse_than_rule": alloc_paired_above,
            "contains_zero": alloc_paired_zero,
            "below_zero_better_than_rule": alloc_paired_below,
        },
        "mesogeos_bare_vs_matched_temperature_rule": {
            "above_rule": meso_above_matched,
            "overlap_rule": meso_overlap_matched,
            "below_rule": meso_below_matched,
            "matched_rule_per_model": {r["label"]: r["mesogeos"]["matched_temperature_rule"]
                                       for r in model_records},
        },
        # Fixed-reference comparisons: a model interval on its parsed subset against a full-set point.
        "allocation_bare_vs_fixed_persistence_0.146": {
            "below_persistence": alloc_below_pers_pt,
            "overlap_persistence": alloc_overlap_pers_pt,
            "above_persistence": alloc_above_pers_pt,
            "persistence_overall": persistence_overall,
        },
        "mesogeos_bare_vs_fixed_temperature_rule_0.654": {
            "above_rule": meso_above_rule,
            "overlap_rule": meso_overlap_rule,
            "below_rule": meso_below_rule,
            "overlapping_models": [r["label"] for r in model_records if r["mesogeos"]["bare"]["overlaps_temperature_rule"]],
        },
        "tooluse_tool_minus_bare_vs_zero": {
            "total_models": len(tool_models),
            "excludes_zero": tool_excl_zero,
            "excludes_zero_downward": tool_excl_zero_down,
            "excludes_zero_upward": tool_excl_zero_up,
            "overlaps_zero": tool_overlap_zero,
            "overlapping_models": [r["label"] for r in tool_models if r["tooluse"]["tool_minus_bare"]["overlaps_zero"]],
        },
        "persistence_copies_five_models": pers_5_results,
        "truncation_summary": trunc_summary,
    }

    latex_rows = format_table_rows(model_records)

    out_data = {
        "meta": {
            "resamples": args.resamples,
            "seed": args.seed,
            "ci": args.ci,
            "tier": "text",
            "models_count": len(model_records),
        },
        "tallies": tallies,
        "latex_table_rows": latex_rows,
        "models": model_records,
    }

    args.out.write_text(json.dumps(out_data, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}", file=sys.stderr)

    print("\n% --- LaTeX rows for the nineteen text-only models ---")
    for l in latex_rows:
        print(l)


if __name__ == "__main__":
    main()
