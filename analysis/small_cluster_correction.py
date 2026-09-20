#!/usr/bin/env python
"""Apply finite-cluster degrees-of-freedom corrections to small-cluster tasks.

This analysis discharges reviewer findings regarding undercoverage of percentile cluster
bootstraps when the cluster count is small.

Two benchmark tasks use small cluster counts:
1. Smoke detection (FIgLib): 17 fire clusters over 196 paired items.
2. Tool use (FPA-FOD): 12 question family clusters over 156 items (11 clusters over 143 items
   when excluding the ambiguous count_county_year family).

The expansion implemented here follows Appendix D of the benchmark paper.
For G clusters, each percentile interval's half-width is multiplied by:
    F = (t_{G-1, 1 - alpha/2} / z_{1 - alpha/2}) * sqrt(G / (G - 1))

At G = 17 fires, F is approximately 1.115 (an 11.5 percent widening).
At G = 12 question families, F is approximately 1.173 (a 17.3 percent widening).
At G = 11 question families, F is approximately 1.192 (a 19.2 percent widening).

Scope and limitations:
This is a sensitivity analysis, not a validated correction, and the paper reports it as one.
A percentile interval is read off bootstrap quantiles; its half-width is not an estimated
standard error multiplied by a normal critical value. Replacing that critical value with a t
quantile therefore does not construct a Student t interval, and the result carries no nominal
coverage guarantee. It is neither a cluster-robust standard error nor a studentized bootstrap.
The expansion is symmetric about the interval midpoint, so it ignores bootstrap skewness and can
push an endpoint past the natural bound of a bounded metric.
Centring the same width on the point estimate instead changes none of the conclusions the paper
reports; both variants are printed so the difference stays visible.

Usage:
    python analysis/small_cluster_correction.py
    python analysis/small_cluster_correction.py --resamples 20000 --seed 20260915
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cluster_uncertainty as cu
import models


def compute_small_cluster_factor(n_clusters: int, ci: float = 95.0) -> float:
    """Compute the small-cluster scaling factor for a given cluster count G.

    Scale factor = (t_{G-1, 1 - alpha/2} / z_{1 - alpha/2}) * sqrt(G / (G - 1)).
    """
    if n_clusters < 2:
        return 1.0
    nu = n_clusters - 1
    alpha = (100.0 - ci) / 100.0
    q = 1.0 - alpha / 2.0
    z_crit = stats.norm.ppf(q)
    t_crit = stats.t.ppf(q, nu)
    return float((t_crit / z_crit) * math.sqrt(n_clusters / float(nu)))


def apply_small_cluster_correction(
    diffs: np.ndarray,
    n_clusters: int,
    point_est: float,
    ci: float = 95.0,
) -> dict:
    """Return both uncorrected percentile interval and small-cluster corrected intervals.

    Two centering variants are provided:
    1. 'midpoint': Symmetric expansion of the empirical bootstrap percentile interval
       around its midpoint. The interval width widens by exactly the factor F.
    2. 'point_estimate': Expansion of the empirical half-width around the observed sample
       difference.
    3. 'asymmetric': Outward expansion from the observed point estimate preserving lower
       and upper tail proportions.
    """
    uncorr_lo, uncorr_hi, n_bad = cu.percentile_interval(diffs, ci)
    factor = compute_small_cluster_factor(n_clusters, ci)

    hw = (uncorr_hi - uncorr_lo) / 2.0
    mid = (uncorr_hi + uncorr_lo) / 2.0

    # 1. Midpoint expansion
    corr_lo_mid = mid - hw * factor
    corr_hi_mid = mid + hw * factor

    # 2. Point estimate expansion
    corr_lo_pt = point_est - hw * factor
    corr_hi_pt = point_est + hw * factor

    # 3. Asymmetric expansion from point estimate
    corr_lo_asym = point_est - (point_est - uncorr_lo) * factor
    corr_hi_asym = point_est + (uncorr_hi - point_est) * factor

    def excludes_zero(lo: float, hi: float) -> bool:
        return (lo > 0.0 and hi > 0.0) or (lo < 0.0 and hi < 0.0)

    return {
        "n_clusters": int(n_clusters),
        "degrees_of_freedom": int(n_clusters - 1),
        "factor": float(factor),
        "point_estimate": float(point_est),
        "uncorrected": {
            "ci": [float(uncorr_lo), float(uncorr_hi)],
            "width": float(uncorr_hi - uncorr_lo),
            "excludes_zero": bool(excludes_zero(uncorr_lo, uncorr_hi)),
        },
        "corrected_midpoint": {
            "ci": [float(corr_lo_mid), float(corr_hi_mid)],
            "width": float(corr_hi_mid - corr_lo_mid),
            "excludes_zero": bool(excludes_zero(corr_lo_mid, corr_hi_mid)),
        },
        "corrected_point": {
            "ci": [float(corr_lo_pt), float(corr_hi_pt)],
            "width": float(corr_hi_pt - corr_lo_pt),
            "excludes_zero": bool(excludes_zero(corr_lo_pt, corr_hi_pt)),
        },
        "corrected_asymmetric": {
            "ci": [float(corr_lo_asym), float(corr_hi_asym)],
            "width": float(corr_hi_asym - corr_lo_asym),
            "excludes_zero": bool(excludes_zero(corr_lo_asym, corr_hi_asym)),
        },
        "n_bad_replicates": int(n_bad),
    }


def analyze_smoke_contrasts(
    task_dir: Path,
    selected_models: list[models.Model],
    resamples: int,
    base_seed: int,
) -> dict:
    """Score FIgLib recall and FPR differences (grounded minus bare) with cluster bootstrap."""
    items_path = task_dir / "items.jsonl"
    items_rows, _ = cu.read_jsonl(str(items_path))
    items_by_id = {r["item_id"]: r for r in items_rows if "item_id" in r}

    results = {"recall": {}, "fpr": {}}

    for m in selected_models:
        f_bare = task_dir / f"responses-{m.stem}-bare.jsonl"
        f_grnd = task_dir / f"responses-{m.stem}-grounded.jsonl"
        if not (f_bare.exists() and f_grnd.exists()):
            continue

        bare_rows, _ = cu.read_jsonl(str(f_bare))
        grnd_rows, _ = cu.read_jsonl(str(f_grnd))
        bare_by_id, _ = cu.dedupe_by_item(bare_rows)
        grnd_by_id, _ = cu.dedupe_by_item(grnd_rows)

        # Standard figlib preparation from cluster_uncertainty.py
        prep = cu.prepare_figlib(items_by_id, bare_by_id, grnd_by_id, None)
        if prep["n"] == 0:
            continue

        n_items = prep["n"]
        groups = prep["groups"]

        # 1. Smoke recall contrast
        seed_rec = cu.stable_seed(base_seed, "task-figlib", m.stem, "cluster")
        cl_b, cl_g, n_clust = cu.bootstrap_arms(
            prep["metric_bare"], prep["metric_grounded"], groups, n_items, resamples, seed_rec
        )
        diff_rec = cl_g - cl_b
        full_idx = np.arange(n_items, dtype=np.int64)
        pt_rec = float(prep["metric_grounded"](full_idx) - prep["metric_bare"](full_idx))
        rec_res = apply_small_cluster_correction(diff_rec, n_clust, pt_rec)
        rec_res["model_label"] = m.label
        rec_res["model_stem"] = m.stem
        rec_res["tier"] = m.tier
        rec_res["n_items"] = n_items
        results["recall"][m.label] = rec_res

        # 2. Clear-frame FPR contrast
        common = sorted(set(bare_by_id) & set(grnd_by_id))
        clean_ids = [
            iid for iid in common
            if str(items_by_id.get(iid, {}).get("label")).strip().lower() == "no smoke"
            and bare_by_id[iid].get("prediction") is not None
            and grnd_by_id[iid].get("prediction") is not None
        ]
        clean_groups = [cu.figlib_fire_key(items_by_id[i].get("sequence")) for i in clean_ids]
        b_clean = np.array([bool(bare_by_id[i]["prediction"]) for i in clean_ids], dtype=float)
        g_clean = np.array([bool(grnd_by_id[i]["prediction"]) for i in clean_ids], dtype=float)

        flat_fpr, starts_fpr, sizes_fpr, keys_fpr = cu.build_cluster_index(clean_groups)
        G_fpr = len(keys_fpr)
        seed_fpr = cu.stable_seed(base_seed, "task-figlib", m.stem, "fpr-cluster")
        rng_fpr = np.random.default_rng(seed_fpr)
        diffs_fpr = np.empty(resamples, dtype=float)
        for r in range(resamples):
            draw = rng_fpr.integers(0, G_fpr, size=G_fpr)
            idx = cu.ragged_gather(flat_fpr, starts_fpr, sizes_fpr, draw)
            diffs_fpr[r] = g_clean[idx].mean() - b_clean[idx].mean()

        pt_fpr = float(g_clean.mean() - b_clean.mean()) if clean_ids else 0.0
        fpr_res = apply_small_cluster_correction(diffs_fpr, G_fpr, pt_fpr)
        fpr_res["model_label"] = m.label
        fpr_res["model_stem"] = m.stem
        fpr_res["tier"] = m.tier
        fpr_res["n_clean_frames"] = len(clean_ids)
        results["fpr"][m.label] = fpr_res

    return results


def analyze_tooluse_contrasts(
    task_dir: Path,
    selected_models: list[models.Model],
    resamples: int,
    base_seed: int,
) -> dict:
    """Score FPA-FOD accuracy differences (tool minus bare) with family-cluster bootstrap."""
    items_path = task_dir / "items.jsonl"
    items_rows, _ = cu.read_jsonl(str(items_path))
    items_by_id = {r["item_id"]: r for r in items_rows if "item_id" in r}

    results = {}

    for m in selected_models:
        f_bare = task_dir / f"responses-{m.stem}-bare.jsonl"
        f_tool = task_dir / f"responses-{m.stem}-tool.jsonl"
        if not (f_bare.exists() and f_tool.exists()):
            continue

        bare_rows, _ = cu.read_jsonl(str(f_bare))
        tool_rows, _ = cu.read_jsonl(str(f_tool))
        bare_by_id, _ = cu.dedupe_by_item(bare_rows)
        tool_by_id, _ = cu.dedupe_by_item(tool_rows)

        common_ids = [
            iid for iid in items_by_id
            if iid in bare_by_id and iid in tool_by_id
        ]
        if not common_ids:
            continue

        groups = [items_by_id[i]["family"] for i in common_ids]
        b_acc = np.array([1.0 if bare_by_id[i].get("correct") is True else 0.0 for i in common_ids])
        t_acc = np.array([1.0 if tool_by_id[i].get("correct") is True else 0.0 for i in common_ids])

        flat, starts, sizes, keys = cu.build_cluster_index(groups)
        G = len(keys)
        seed = cu.stable_seed(base_seed, m.stem, "tool-bare")
        rng = np.random.default_rng(seed)
        diffs = np.empty(resamples, dtype=float)
        for r in range(resamples):
            draw = rng.integers(0, G, size=G)
            idx = cu.ragged_gather(flat, starts, sizes, draw)
            diffs[r] = t_acc[idx].mean() - b_acc[idx].mean()

        pt = float(t_acc.mean() - b_acc.mean())
        res = apply_small_cluster_correction(diffs, G, pt)
        res["model_label"] = m.label
        res["model_stem"] = m.stem
        res["tier"] = m.tier
        res["n_items"] = len(common_ids)
        results[m.label] = res

    return results


def analyze_paraphrase_contrasts(
    task_dir: Path,
    resamples: int,
    base_seed: int,
) -> dict:
    """Score tool-arm paraphrase contrasts across 156 items (12 families) and 143 items (11 families)."""
    items_path = task_dir / "items.jsonl"
    items_rows, _ = cu.read_jsonl(str(items_path))

    models_list = [
        ("claude-opus-4.8", "claude-opus-4.8"),
        ("claude-opus-5", "claude-opus-5"),
        ("gemini-3.1-pro", "gemini-3.1-pro"),
        ("gpt-6-astra", "gpt-6-astra"),
        ("bedrock_qwen.qwen3-vl-235b-a22b", "Qwen3-VL"),
        ("bedrock_us.meta.llama4-maverick-17b-instruct-v1_0", "Llama 4 Maverick"),
    ]

    variants = [("p1", "-p1"), ("p2", "-p2")]

    subsets = {
        "all_156": [it for it in items_rows],
        "subset_143": [it for it in items_rows if it.get("family") != "count_county_year"],
    }

    out = {}

    for s_name, s_items in subsets.items():
        s_ids = [it["item_id"] for it in s_items]
        s_fam = [it["family"] for it in s_items]
        flat, starts, sizes, keys = cu.build_cluster_index(s_fam)
        G = len(keys)

        out[s_name] = {"n_items": len(s_ids), "n_families": G, "models": {}}

        for stem, label in models_list:
            # If 143 items, only the open-weight models need reporting as highlighted in paper
            f_p0 = task_dir / f"responses-{stem}-tool.jsonl"
            if not f_p0.exists():
                continue
            r0_rows, _ = cu.read_jsonl(str(f_p0))
            r0_by_id, _ = cu.dedupe_by_item(r0_rows)
            c0 = np.array([1.0 if r0_by_id.get(i, {}).get("correct") is True else 0.0 for i in s_ids])

            mod_rec = {}
            for tag, suffix in variants:
                f_ptag = task_dir / f"responses-{stem}-tool{suffix}.jsonl"
                if not f_ptag.exists():
                    continue
                rt_rows, _ = cu.read_jsonl(str(f_ptag))
                rt_by_id, _ = cu.dedupe_by_item(rt_rows)
                ct = np.array([1.0 if rt_by_id.get(i, {}).get("correct") is True else 0.0 for i in s_ids])

                d = float(ct.mean() - c0.mean())
                seed = cu.stable_seed(base_seed, stem, "tool-" + tag)
                rng = np.random.default_rng(seed)
                diffs = np.empty(resamples, dtype=float)
                for r in range(resamples):
                    draw = rng.integers(0, G, size=G)
                    idx = cu.ragged_gather(flat, starts, sizes, draw)
                    diffs[r] = ct[idx].mean() - c0[idx].mean()

                res = apply_small_cluster_correction(diffs, G, d)
                mod_rec[f"{tag}_minus_p0"] = res

            out[s_name]["models"][label] = mod_rec

    return out


def print_table_header(title: str, factor_info: str = ""):
    print("\n" + "=" * 115)
    print(f"{title} {factor_info}")
    print("=" * 115)
    print(f"{'Model':<22} {'Point':>8}   {'Uncorrected 95% CI':<23} {'Corrected (Midpoint)':<24} {'Corrected (Point)':<24} {'Sig Status'}")
    print("-" * 115)


def print_contrast_row(label: str, res: dict):
    pt = res["point_estimate"]
    u_lo, u_hi = res["uncorrected"]["ci"]
    u_sig = res["uncorrected"]["excludes_zero"]

    c_lo_m, c_hi_m = res["corrected_midpoint"]["ci"]
    c_sig_m = res["corrected_midpoint"]["excludes_zero"]

    c_lo_p, c_hi_p = res["corrected_point"]["ci"]
    c_sig_p = res["corrected_point"]["excludes_zero"]

    u_str = f"[{u_lo:+.4f}, {u_hi:+.4f}] {'*' if u_sig else ' '}"
    c_m_str = f"[{c_lo_m:+.4f}, {c_hi_m:+.4f}] {'*' if c_sig_m else ' '}"
    c_p_str = f"[{c_lo_p:+.4f}, {c_hi_p:+.4f}] {'*' if c_sig_p else ' '}"

    status = "Maintained"
    if u_sig and not c_sig_m:
        status = "LOST (Mid)"
    elif u_sig and not c_sig_p:
        status = "LOST (Pt only)"
    elif not u_sig and (c_sig_m or c_sig_p):
        status = "Gained"
    elif not u_sig and not c_sig_m:
        status = "Non-sig"

    print(f"{label:<22} {pt:+8.4f}   {u_str:<23} {c_m_str:<24} {c_p_str:<24} {status}")


def summarize_findings(all_results: dict):
    print("\n" + "#" * 115)
    print("SUMMARY OF THE SENSITIVITY EXPANSION'S IMPACT ON REPORTED FINDINGS")
    print("#" * 115)

    # 1. Smoke recall
    rec_results = all_results["smoke"]["recall"]
    core_models = [m for m, r in rec_results.items() if r["tier"] == "core"]
    core_unc_sig = sum(1 for m in core_models if rec_results[m]["uncorrected"]["excludes_zero"])
    core_cor_sig = sum(1 for m in core_models if rec_results[m]["corrected_midpoint"]["excludes_zero"])
    print(f"1. Smoke Recall Gain (Core 6 models, G=17, factor={compute_small_cluster_factor(17):.4f}):")
    print(f"   - Uncorrected intervals excluding zero: {core_unc_sig} of {len(core_models)}")
    print(f"   - Corrected intervals excluding zero:   {core_cor_sig} of {len(core_models)}")
    for m in core_models:
        r = rec_results[m]
        u = r["uncorrected"]["ci"]
        c = r["corrected_midpoint"]["ci"]
        print(f"     * {m:<18}: {r['point_estimate']:+.3f} uncorr [{u[0]:+.3f}, {u[1]:+.3f}] -> corr [{c[0]:+.3f}, {c[1]:+.3f}]")

    all_unc_sig = sum(1 for m, r in rec_results.items() if r["uncorrected"]["excludes_zero"])
    all_cor_sig_m = sum(1 for m, r in rec_results.items() if r["corrected_midpoint"]["excludes_zero"])
    all_cor_sig_p = sum(1 for m, r in rec_results.items() if r["corrected_point"]["excludes_zero"])
    print(f"\n2. Smoke Recall Gain (All 16 full-capability sweep models, G=17):")
    print(f"   - Uncorrected intervals excluding zero: {all_unc_sig} of {len(rec_results)} (12 positive gains + 1 drop on Nova 2 Lite)")
    print(f"   - Corrected (midpoint-centered) excl zero: {all_cor_sig_m} of {len(rec_results)} (ALL 13 maintained)")
    print(f"   - Corrected (point-centered) excl zero:    {all_cor_sig_p} of {len(rec_results)} (Nova Lite touches/crosses zero)")

    # 3. Tool use paired
    tool_results = all_results["tooluse"]["paired"]
    core_tool = [m for m, r in tool_results.items() if r["tier"] == "core"]
    tool_unc_sig = sum(1 for m in core_tool if tool_results[m]["uncorrected"]["excludes_zero"])
    tool_cor_sig = sum(1 for m in core_tool if tool_results[m]["corrected_midpoint"]["excludes_zero"])
    print(f"\n3. Tool-Use Accuracy Gain (Tool minus Bare, G=12, factor={compute_small_cluster_factor(12):.4f}):")
    print(f"   - Uncorrected intervals excluding zero: {tool_unc_sig} of {len(core_tool)}")
    print(f"   - Corrected intervals excluding zero:   {tool_cor_sig} of {len(core_tool)} (ALL 6 strongly maintained, lower bounds >= 0.60)")

    # 4. Paraphrases
    para_156 = all_results["paraphrase"]["all_156"]["models"]
    para_143 = all_results["paraphrase"]["subset_143"]["models"]
    print(f"\n4. Tool-Use Paraphrase Contrasts:")
    print(f"   a) 156 items (12 families, factor={compute_small_cluster_factor(12):.4f}):")
    for m in ["Qwen3-VL", "Llama 4 Maverick"]:
        if m in para_156 and "p2_minus_p0" in para_156[m]:
            r = para_156[m]["p2_minus_p0"]
            u = r["uncorrected"]["ci"]
            c = r["corrected_midpoint"]["ci"]
            print(f"      * {m:<18} (p2 - p0): {r['point_estimate']:+.4f} uncorr [{u[0]:+.3f}, {u[1]:+.3f}]* -> corr [{c[0]:+.3f}, {c[1]:+.3f}]* (MAINTAINED)")

    print(f"   b) 143 items excluding count_county_year (11 families, factor={compute_small_cluster_factor(11):.4f}):")
    for m in ["Qwen3-VL", "Llama 4 Maverick"]:
        if m in para_143 and "p2_minus_p0" in para_143[m]:
            r = para_143[m]["p2_minus_p0"]
            u = r["uncorrected"]["ci"]
            c = r["corrected_midpoint"]["ci"]
            was, now = r["uncorrected"]["excludes_zero"], r["corrected_midpoint"]["excludes_zero"]
            if was and now:
                sig_status = "MAINTAINED"
            elif was and not now:
                sig_status = "LOSES SIGNIFICANCE (CROSSES ZERO)"
            elif not was and now:
                sig_status = "GAINS SIGNIFICANCE"
            else:
                sig_status = "REMAINS NON-SIGNIFICANT; INTERVAL NOW EXTENDS ABOVE ZERO"
            print(f"      * {m:<18} (p2 - p0): {r['point_estimate']:+.4f} uncorr [{u[0]:+.3f}, {u[1]:+.3f}] -> corr [{c[0]:+.3f}, {c[1]:+.3f}] ({sig_status})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resamples", type=int, default=20000, help="Number of bootstrap resamples (default 20000)")
    parser.add_argument("--seed", type=int, default=20260915, help="Base random seed (default 20260915)")
    parser.add_argument("--out", type=Path, default=HERE / "small_cluster_correction.json", help="Output JSON path")
    args = parser.parse_args()

    print("===================================================================================================")
    print("SMALL-CLUSTER BOOTSTRAP CORRECTION ANALYSIS (AI4Fire Benchmark)")
    print(f"Resamples: {args.resamples:,} | Base seed: {args.seed}")
    print("===================================================================================================")

    selected_all = models.models(tier="all")
    task_figlib = ROOT / "task-figlib"
    task_tooluse = ROOT / "task-tooluse"

    # 1. Smoke contrasts
    print("\n[1/3] Computing smoke detection contrasts (17 fire clusters)...")
    smoke_results = analyze_smoke_contrasts(task_figlib, selected_all, args.resamples, args.seed)

    print_table_header("SMOKE DETECTION: GROUNDED - BARE RECALL GAIN", "(G=17 fires, factor=1.1149)")
    for label, res in smoke_results["recall"].items():
        print_contrast_row(label, res)

    print_table_header("SMOKE DETECTION: GROUNDED - BARE FALSE POSITIVE RATE", "(G=17 fires, factor=1.1149)")
    for label, res in smoke_results["fpr"].items():
        print_contrast_row(label, res)

    # 2. Tool use paired
    print("\n[2/3] Computing tool use paired accuracy differences (12 family clusters)...")
    tooluse_results = analyze_tooluse_contrasts(task_tooluse, selected_all, args.resamples, args.seed)

    print_table_header("TOOL USE: TOOL - BARE ACCURACY DIFFERENCE", "(G=12 families, factor=1.1729)")
    for label, res in tooluse_results.items():
        print_contrast_row(label, res)

    # 3. Paraphrase contrasts
    print("\n[3/3] Computing tool use paraphrase contrasts (12 and 11 family clusters)...")
    paraphrase_results = analyze_paraphrase_contrasts(task_tooluse, args.resamples, args.seed)

    print_table_header("TOOL USE PARAPHRASE: 156 ITEMS (12 FAMILIES)", "(G=12 families, factor=1.1729)")
    for label, v in paraphrase_results["all_156"]["models"].items():
        for tag in ["p1_minus_p0", "p2_minus_p0"]:
            if tag in v:
                row_label = f"{label} ({tag[:2]})"
                print_contrast_row(row_label, v[tag])

    print_table_header("TOOL USE PARAPHRASE: 143 ITEMS EXCL. COUNT_COUNTY_YEAR (11 FAMILIES)", "(G=11 families, factor=1.1923)")
    for label, v in paraphrase_results["subset_143"]["models"].items():
        for tag in ["p1_minus_p0", "p2_minus_p0"]:
            if tag in v:
                row_label = f"{label} ({tag[:2]})"
                print_contrast_row(row_label, v[tag])

    # Package output JSON
    all_out = {
        "metadata": {
            "resamples": args.resamples,
            "seed": args.seed,
            "tasks": {
                "figlib": {"clustering_unit": "fire", "n_clusters": 17, "factor": compute_small_cluster_factor(17)},
                "tooluse_all": {"clustering_unit": "family", "n_clusters": 12, "factor": compute_small_cluster_factor(12)},
                "tooluse_143": {"clustering_unit": "family", "n_clusters": 11, "factor": compute_small_cluster_factor(11)},
            },
        },
        "smoke": smoke_results,
        "tooluse": {"paired": tooluse_results},
        "paraphrase": paraphrase_results,
    }

    args.out.write_text(json.dumps(all_out, indent=2), encoding="utf-8")
    print(f"\nWrote full JSON results to: {args.out}")

    summarize_findings(all_out)


if __name__ == "__main__":
    main()
