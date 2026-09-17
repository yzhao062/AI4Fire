"""Paired scoring of the FIgLib smoke task for five models under bare and grounded conditions.

Paired convention (as used by the paper): for each model, keep only the items that appear
in both the bare and the grounded response file with a non-null prediction in both, and
score both conditions on exactly that item set.

Reads only. Writes a single JSON file next to this script.
"""
import argparse
import collections
import datetime
import json
import math
import os
import sys

TASK_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "task-figlib")
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figlib-paired.json")

MODELS = ["claude-opus-4.8", "claude-opus-5", "gemini-3.1-pro", "gpt-6-astra", "bedrock_qwen.qwen3-vl-235b-a22b", "bedrock_us.meta.llama4-maverick-17b-instruct-v1_0"]
CONDITIONS = ["bare", "grounded"]
PAIR_WINDOW_MIN = 60.0
TOKEN_CAP = 1536
TOKEN_NEAR_CAP = 1500

BUCKET_ORDER = [
    "before 25 min or more",
    "before 10 to 25 min",
    "before 0 to 10 min",
    "after 0 to 10 min",
    "after 10 to 25 min",
    "after 25 min or more",
]


def read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def mtime_iso(path):
    return datetime.datetime.fromtimestamp(os.stat(path).st_mtime).isoformat(sep=" ", timespec="seconds")


def label_is_smoke(label):
    if label == "smoke":
        return True
    if label == "no smoke":
        return False
    raise ValueError(f"unexpected label value {label!r}")


def binom_two_sided_p(k, n):
    """Exact two-sided binomial test p-value for k successes in n trials at p=0.5.

    Uses the sum of all outcome probabilities no larger than the observed one, which for
    the symmetric p=0.5 case equals twice the smaller tail (capped at 1).
    """
    if n == 0:
        return float("nan")
    probs = [math.comb(n, i) / (2 ** n) for i in range(n + 1)]
    obs = probs[k]
    return min(1.0, sum(p for p in probs if p <= obs * (1 + 1e-12)))


def score(rows_by_id, ids):
    """rows_by_id: item_id -> response row. ids: the paired item set."""
    n = len(ids)
    correct = 0
    smoke_n = smoke_tp = 0
    nosmoke_n = nosmoke_fp = 0
    per_bucket = collections.defaultdict(lambda: [0, 0])  # bucket -> [correct, n]
    for iid in ids:
        r = rows_by_id[iid]
        truth = label_is_smoke(r["label"])
        pred = r["prediction"]
        assert isinstance(pred, bool), (iid, pred)
        ok = pred == truth
        correct += ok
        if truth:
            smoke_n += 1
            smoke_tp += pred
        else:
            nosmoke_n += 1
            nosmoke_fp += pred
        per_bucket[r["bucket"]][0] += ok
        per_bucket[r["bucket"]][1] += 1
    buckets_present = [b for b in BUCKET_ORDER if b in per_bucket] + sorted(
        b for b in per_bucket if b not in BUCKET_ORDER
    )
    bucket_str = "; ".join(
        f"{b} {per_bucket[b][0] / per_bucket[b][1]:.2f} (n={per_bucket[b][1]})" for b in buckets_present
    )
    return {
        "n_paired": n,
        "accuracy": round(correct / n, 3),
        "recall_smoke": round(smoke_tp / smoke_n, 3),
        "fpr": round(nosmoke_fp / nosmoke_n, 3),
        "bucket_accuracy": bucket_str,
        "counts": {
            "correct": correct,
            "smoke_frames": smoke_n,
            "smoke_true_positive": smoke_tp,
            "no_smoke_frames": nosmoke_n,
            "no_smoke_false_positive": nosmoke_fp,
        },
        "bucket_detail": {b: {"accuracy": round(per_bucket[b][0] / per_bucket[b][1], 3), "n": per_bucket[b][1]} for b in buckets_present},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=str, default=None, help="Path to manifest JSON restricting response files.")
    args = parser.parse_args()

    models = MODELS
    if args.manifest:
        with open(args.manifest, encoding="utf-8") as fh:
            manifest = json.load(fh)
        reported = manifest.get("reported", [])
        manifest_models = []
        for r in reported:
            if r.get("task") in ("figlib", "task-figlib") and r.get("condition") == "bare":
                manifest_models.append(r["model_token"])
        if manifest_models:
            models = manifest_models

    items = read_jsonl(os.path.join(TASK_DIR, "items.jsonl"))
    items_by_id = {r["item_id"]: r for r in items}
    assert len(items_by_id) == len(items), "duplicate item_id in items.jsonl"

    out = {
        "task_dir": TASK_DIR,
        "generated_at": datetime.datetime.now().isoformat(sep=" ", timespec="seconds"),
        "items": {
            "rows": len(items),
            "label_values": dict(collections.Counter(r["label"] for r in items)),
            "sequences": len({r["sequence"] for r in items}),
            "mtime": mtime_iso(os.path.join(TASK_DIR, "items.jsonl")),
        },
        "files": {},
        "pairing": {},
        "rows": [],
        "sign_tests": [],
        "truncation": {},
        "checks": [],
    }

    loaded = {}
    for m in models:
        for c in CONDITIONS:
            path = os.path.join(TASK_DIR, f"responses-{m}-{c}.jsonl")
            rows = read_jsonl(path)
            by_id = {r["item_id"]: r for r in rows}
            assert len(by_id) == len(rows), f"duplicate item_id in {path}"
            ct = [r.get("usage", {}).get("completion_tokens") for r in rows]
            ct = [x for x in ct if x is not None]
            loaded[(m, c)] = by_id
            out["files"][f"{m}-{c}"] = {
                "path": path,
                "mtime": mtime_iso(path),
                "rows": len(rows),
                "label_values": dict(collections.Counter(r["label"] for r in rows)),
                "prediction_values": dict(collections.Counter(repr(r.get("prediction")) for r in rows)),
                "rows_with_error_key": sum(1 for r in rows if r.get("error") is not None),
                "bucket_values": dict(collections.Counter(r["bucket"] for r in rows)),
                "served_model": sorted({r.get("served_model") for r in rows}),
                "completion_tokens_max": max(ct) if ct else None,
                "completion_tokens_at_or_above_1500": sum(1 for x in ct if x >= TOKEN_NEAR_CAP),
                "completion_tokens_rows": len(ct),
            }
            # Cross-check label / sequence / bucket against items.jsonl.
            mismatched_label = [iid for iid, r in by_id.items() if items_by_id[iid]["label"] != r["label"]]
            mismatched_seq = [iid for iid, r in by_id.items() if items_by_id[iid]["sequence"] != r["sequence"]]
            missing_from_items = [iid for iid in by_id if iid not in items_by_id]
            out["checks"].append(
                {
                    "file": f"{m}-{c}",
                    "item_ids_not_in_items_jsonl": len(missing_from_items),
                    "label_mismatch_vs_items": len(mismatched_label),
                    "sequence_mismatch_vs_items": len(mismatched_seq),
                }
            )

    # Pairing by modification time.
    pairing_ok = True
    for m in models:
        pb = os.path.join(TASK_DIR, f"responses-{m}-bare.jsonl")
        pg = os.path.join(TASK_DIR, f"responses-{m}-grounded.jsonl")
        delta_min = abs(os.stat(pb).st_mtime - os.stat(pg).st_mtime) / 60.0
        within = delta_min <= PAIR_WINDOW_MIN
        pairing_ok = pairing_ok and within
        out["pairing"][m] = {
            "bare_mtime": mtime_iso(pb),
            "grounded_mtime": mtime_iso(pg),
            "delta_minutes": round(delta_min, 1),
            "within_60_min": within,
            "status": "SAME-RUN (within 60 min)" if within else "CROSS-RUN (more than 60 min apart)",
        }
    out["pairing_ok"] = pairing_ok

    # Paired scoring.
    for m in models:
        bare = loaded[(m, "bare")]
        grounded = loaded[(m, "grounded")]
        paired_ids = sorted(
            iid
            for iid in bare
            if iid in grounded and bare[iid].get("prediction") is not None and grounded[iid].get("prediction") is not None
        )
        # Bucket must agree between the two files for the same item.
        bucket_disagree = [iid for iid in paired_ids if bare[iid]["bucket"] != grounded[iid]["bucket"]]
        out["checks"].append(
            {
                "model": m,
                "bare_rows": len(bare),
                "grounded_rows": len(grounded),
                "in_both": sum(1 for iid in bare if iid in grounded),
                "grounded_only": sum(1 for iid in grounded if iid not in bare),
                "bare_null_prediction": sum(1 for r in bare.values() if r.get("prediction") is None),
                "grounded_null_prediction": sum(1 for r in grounded.values() if r.get("prediction") is None),
                "paired": len(paired_ids),
                "bucket_disagreement_between_files": len(bucket_disagree),
                "dropped_bare_only_buckets": dict(
                    collections.Counter(bare[iid]["bucket"] for iid in bare if iid not in grounded)
                ),
            }
        )
        for c, rows_by_id in (("bare", bare), ("grounded", grounded)):
            s = score(rows_by_id, paired_ids)
            out["rows"].append({"model": m, "condition": c, **s})

        # Sign test on smoke frames in the paired set.
        gained_ids, lost_ids = [], []
        smoke_frames = 0
        both_right = both_wrong = 0
        for iid in paired_ids:
            if not label_is_smoke(bare[iid]["label"]):
                continue
            smoke_frames += 1
            b_ok = bare[iid]["prediction"] is True
            g_ok = grounded[iid]["prediction"] is True
            if g_ok and not b_ok:
                gained_ids.append(iid)
            elif b_ok and not g_ok:
                lost_ids.append(iid)
            elif b_ok and g_ok:
                both_right += 1
            else:
                both_wrong += 1
        gained, lost = len(gained_ids), len(lost_ids)
        n_disc = gained + lost
        p = binom_two_sided_p(gained, n_disc) if n_disc else None
        gained_seqs = sorted({bare[iid]["sequence"] for iid in gained_ids})
        lost_seqs = sorted({bare[iid]["sequence"] for iid in lost_ids})
        out["sign_tests"].append(
            {
                "model": m,
                "smoke_frames": smoke_frames,
                "gained": gained,
                "lost": lost,
                "both_right": both_right,
                "both_wrong": both_wrong,
                "discordant": n_disc,
                "p_two_sided": (round(p, 4) if p is not None else None),
                "p_two_sided_full": p,
                "fires_touched": len(gained_seqs),
                "fires_touched_str": f"{len(gained_seqs)} distinct sequences among {gained} gained frames",
                "gained_sequences": gained_seqs,
                "gained_by_bucket": dict(collections.Counter(bare[iid]["bucket"] for iid in gained_ids)),
                "lost_sequences": lost_seqs,
                "lost_by_bucket": dict(collections.Counter(bare[iid]["bucket"] for iid in lost_ids)),
            }
        )

    # Truncation check for the gemini files.
    for c in CONDITIONS:
        key = f"gemini-3.1-pro-{c}"
        rows = loaded[("gemini-3.1-pro", c)]
        near = [
            {
                "item_id": iid,
                "completion_tokens": r["usage"]["completion_tokens"],
                "reasoning_tokens": r["usage"].get("completion_tokens_details", {}).get("reasoning_tokens"),
                "text_tokens": r["usage"].get("completion_tokens_details", {}).get("text_tokens"),
                "prediction": r.get("prediction"),
                "raw_head": (r.get("raw") or "")[:200],
            }
            for iid, r in rows.items()
            if r.get("usage", {}).get("completion_tokens", 0) >= TOKEN_NEAR_CAP
        ]
        out["truncation"][key] = {
            "cap": TOKEN_CAP,
            "max_completion_tokens": out["files"][key]["completion_tokens_max"],
            "count_at_or_above_1500": out["files"][key]["completion_tokens_at_or_above_1500"],
            "rows_at_or_above_1500": near,
        }

    # Provenance: do the files on disk match the run that the repo's own score files were computed from?
    # run_figlib.py rewrites every responses file with write_text and merges scores.json per run, so an
    # entry in scores.json can describe an earlier run than the file that now sits on disk.
    prov = {"repo_score_files": {}, "per_run": {}}
    for fn in ("scores.json", "figlib-paired.json", "figlib-shared-subset.json", "run-1568.log", "run-opus5.log"):
        p = os.path.join(TASK_DIR, fn)
        if os.path.exists(p):
            prov["repo_score_files"][fn] = {"mtime": mtime_iso(p)}
    scores_path = os.path.join(TASK_DIR, "scores.json")
    if os.path.exists(scores_path):
        with open(scores_path, encoding="utf-8") as fh:
            scores = {s["run"]: s for s in json.load(fh)}
        for m in models:
            for c in CONDITIONS:
                rows = loaded[(m, c)].values()
                tout = sum((r.get("usage") or {}).get("completion_tokens", 0) for r in rows)
                parsed = sum(1 for r in rows if r.get("prediction") is not None)
                s = scores.get(f"{m}/{c}")
                prov["per_run"][f"{m}-{c}"] = {
                    "file_parsed": parsed,
                    "file_tokens_out": tout,
                    "scores_json_parsed": s.get("parsed") if s else None,
                    "scores_json_tokens_out": s.get("tokens_out") if s else None,
                    "scores_json_recall_on_smoke": s.get("recall_on_smoke") if s else None,
                    "file_matches_scores_json_run": bool(s) and s.get("tokens_out") == tout and s.get("parsed") == parsed,
                }
    paired_path = os.path.join(TASK_DIR, "figlib-paired.json")
    if os.path.exists(paired_path):
        with open(paired_path, encoding="utf-8") as fh:
            prov["repo_figlib_paired_json"] = json.load(fh)
    out["provenance"] = prov

    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    json.dump(out, sys.stdout, indent=2)
    print()
    print("wrote", OUT_PATH)


if __name__ == "__main__":
    main()
