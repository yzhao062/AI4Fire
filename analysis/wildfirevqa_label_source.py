"""Aerial accuracy by label source and by applicability, the hotspot references, and exact tests by item group.

WildFireVQA fixes the source of each reference answer per question type. On the 408 aerial items, 132 items over 11
types carry a label from a radiometric formula, flight telemetry, or an object detector, and 276 items over the other
23 types carry an answer a multimodal model generated and a person verified. The release also rates each item's
applicability, 1.0 on 284 items and below 1.0 on 124. This script scores the six core models, bare and grounded,
within those strata beside the held-out per-question majority, and records why the label-source split cannot bound
label error: label source is constant within a question type, so the two groups are disjoint sets of types.

It backs, in the paper's extended-results appendix on aerial question answering, the paragraphs "Accuracy by Label
Source and by Applicability" and "Exact Paired Tests by Item Group" with Tables tab:aerial-strata (which also carries
the label tab:aerial-applicability) and tab:aerial-exact; the label-source and applicability sentences of the aerial
results paragraph; insight I1's sentence on the 347 items the thermal block does not answer; the hotspot sentences of
the Limitations; and the first sentence of "Label Source and Question Type Are Collinear" in the build appendix.

JSON blocks, in order:
  structure            label source per item and type, shared frames, closed-form placement, applicability strata
  source_pool          the same collinearity on the 25,027-record Sycan pool, the held-out majority refit, and the
                       hotspot formula's threshold and size floor (declared-optional input, see Inputs)
  columns              Table tab:aerial-strata. Six columns: all items; formula, telemetry, or detector labels; the
                       same without the 48 closed-form items; model-verified labels; applicability 1.0; below 1.0.
                       Per core model and arm: accuracy, model minus majority, and (per model) grounded minus bare,
                       each with a paired 95 percent percentile interval under three resamplings (the 390 frames,
                       the paper's unit; the 34 question types; the two-way pigeonhole bootstrap over both) and an
                       exact two-sided McNemar test, Holm-adjusted over the twelve arms of each column
  table_summary        separations per resampling unit, McNemar agreement with the frame markers, and unmarked
                       frame intervals that end at exactly zero
  per_type_open        per question type on the 84 formula items outside the closed-form set
  hotspot_references   the 14 items of DS1, DS3, PD1, and PD7 whose reference reads "nothing burning" although the
                       frame maximum is at least 200 C, with every aerial arm on disk and the majority on them
  post_hoc_without_hotspot_references
                       the columns without those 14 items; chosen after seeing the failures, so a pointer for a
                       label audit rather than a test
  exact_tests_by_item_group
                       Table tab:aerial-exact: exact McNemar tests of grounded against bare on the 48 closed-form
                       items, the 13 no-fire-branch items, and the 347 items the thermal block does not answer, for
                       the sixteen full-capability models, with Holm across the six core models on the 347

Resampling reuses the house machinery unchanged: wildfirevqa_paired.paired_bootstrap (one-way cluster bootstrap,
both series scored on the same draw) and wildfirevqa_strata.twoway (pigeonhole bootstrap), 20,000 resamples, base
seed 20260915, per-row seeds from cluster_uncertainty.stable_seed:
    model minus majority   stable_seed(base, stem, arm, "model-majority"[, tag][, unit])
    grounded minus bare    stable_seed(base, stem, "grounded-bare"[, tag][, unit])
where tag is the column's seed tag and unit is "qtype" or "twoway". The all-items frame rows omit both parts, so they
equal the published intervals of wildfirevqa_comparators.py and wildfirevqa_paired.py. The post hoc exclusion
appends "-nohot" to the tag. The seed tags keep the strings the paper's intervals were drawn with ("sensor" for the
formula column); renaming one moves its Monte Carlo stream. The crossed-design table of the pooled contrasts
(tab:wildfirevqa-clustering) and the hybrid-reference intervals stay with wildfirevqa_strata.py and its own tags.

The comparator is the carried baseline_majority_correct field; source_pool shows that refitting the per-type
majority on the pool without the 408 evaluation records leaves every item's score unchanged, so it is held out.

Inputs:
  task-wildfirevqa/items.jsonl, and responses-<stem>-{bare,grounded}.jsonl for the sixteen full-capability models
    (models.py tier "full"; the six core models are tier "core")
  task-wildfirevqa/source-prompt-template.txt, for the prompt's definition of a hotspot
  data/wildfirevqa/vqa_response_Sycan_*.json, the three Sycan Marsh files of the raw WildFireVQA question release
    (huggingface.co/datasets/mobiiin/WildFire_VQA; build_items_wildfirevqa.py downloads them to that folder).
    Declared-optional: without them the source_pool block is skipped with a printed notice and the rest runs.
Output: analysis/wildfirevqa_label_source.json

    python analysis/wildfirevqa_label_source.py        # about a minute on this machine
    python analysis/wildfirevqa_label_source.py --resamples 20000 --seed 20260915
"""
import argparse
import collections
import json
import pathlib
import sys

import numpy as np
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT))

import cluster_uncertainty as cu  # noqa: E402
import models  # noqa: E402
import wildfirevqa_paired as wp  # noqa: E402
import wildfirevqa_strata as ws  # noqa: E402
from answer_failures import holm  # noqa: E402

TASK = ROOT / "task-wildfirevqa"
POOL_DIR = ROOT / "data" / "wildfirevqa"
POOL_FILES = ["vqa_response_%s_altitude_GT_hotspot_PD7_DS1_DS3_DS7_DS8_LD1_CMR4_PD8_CL1_CL1.json" % unit
              for unit in ("Sycan_2A_FIRE", "Sycan_2D_FIRE", "Sycan_2D_NoFIRE")]
# build_items_wildfirevqa.py cannot be imported (it downloads and rewrites the item file at import time), so its
# validity filter is restated here: the answer matches an option after case folding, temp_summary carries all seven
# keys, and applicability_score is a number above 0.
TS_KEYS = ("min", "max", "mean", "std", "top3_mean", "pct_over_200", "pct_over_400")

ARMS = ("bare", "grounded")
MODEL_VERIFIED = "mllm_verified"
UNITS = ("frame", "question_type", "two_way")
# The table columns: JSON key, seed tag, heading.
COLUMNS = (
    ("all", "pooled", "All items"),
    ("formula", "sensor", "Formula, telemetry, or detector labels"),
    ("formula_open", "sensor_open", "Same, without the 48 closed-form items"),
    ("model_verified", "model", "Model-verified labels"),
    ("app_full", "app_full", "Applicability 1.0"),
    ("app_below", "app_below", "Applicability below 1.0"),
)
# The thermal block's no-fire threshold: the builder's no-fire branch answers "nothing burning" below it.
NO_FIRE_C = 200.0
# The four question types whose reference counts radiometric hotspots (source_pool confirms these are the types
# whose ground truth applies a minimum hotspot radius), with the option that reads "nothing burning".
HOTSPOT_NOTHING = {"DS1": "No active hotspots", "DS3": "No active hotspots", "PD1": "No", "PD7": "No fire"}
# The prespecified item partition of the thermal block (the items' no_image_answerable field, null for the rest).
ITEM_GROUPS = (
    ("closed_form", "answered by a closed-form rule on the thermal block"),
    ("no_fire_branch", "answered by the no-fire branch, frame maximum below 200 C"),
    ("not_answered", "not answered by the thermal block"),
)


# ---------------------------------------------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------------------------------------------


def load():
    """Items in file order, and per model and arm a 0/1 correctness vector and an error flag, aligned to the items."""
    items = [json.loads(l) for l in (TASK / "items.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    ids = [i["item_id"] for i in items]
    d = {
        "items": items,
        "ids": ids,
        "frame": np.array([i["image"]["image_uid"] for i in items]),
        "qtype": np.array([i["question_id"] for i in items]),
        "prov": np.array([i["answer_provenance"] for i in items]),
        "app": np.array([float(i["applicability_score"]) for i in items]),
        "group": np.array([i.get("no_image_answerable") or "not_answered" for i in items]),
        "majority": np.array([1.0 if i["baseline_majority_correct"] else 0.0 for i in items]),
        "tmax": np.array([float(i["temp_summary"]["max"]) for i in items]),
        "core": models.models(tier="core", task="wildfirevqa"),
        "full": models.models(tier="full", task="wildfirevqa"),
        "correct": {},
        "error": {},
    }
    for m in d["full"]:
        for arm in ARMS:
            path = TASK / ("responses-%s-%s.jsonl" % (m.stem, arm))
            rows, bad = cu.read_jsonl(path)
            by_id, _ = cu.dedupe_by_item(rows)
            missing = [i for i in ids if i not in by_id]
            if bad or missing:
                raise SystemExit("%s: %d unreadable lines, %d of %d items missing" % (path, bad, len(missing), len(ids)))
            d["correct"][(m.stem, arm)] = np.array([1.0 if by_id[i].get("correct") is True else 0.0 for i in ids])
            d["error"][(m.stem, arm)] = np.array([bool(by_id[i].get("error")) for i in ids])
    return d


def canon(answer, options):
    """The released answer matched to its option after case folding, as build_items_wildfirevqa.canon does."""
    if not isinstance(answer, str):
        return None
    if answer in options:
        return answer
    return {o.lower(): o for o in options}.get(answer.strip().lower())


def load_pool():
    """(valid Sycan records of the raw release, drop counts, missing files); records is None when a file is missing."""
    paths = [POOL_DIR / name for name in POOL_FILES]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        return None, None, missing
    records, dropped = [], collections.Counter()
    for path in paths:
        for rec in json.loads(path.read_text(encoding="utf-8")):
            qid = rec.get("question_id")
            if qid is None:
                continue                          # the one prompt_template header per file
            answer = canon(rec.get("answer"), rec["options"])
            ts, ap = rec.get("temp_summary"), rec.get("applicability_score")
            if answer is None:
                dropped["answer not an option"] += 1
                continue
            if not isinstance(ts, dict) or any(k not in ts for k in TS_KEYS):
                dropped["temp_summary incomplete"] += 1
                continue
            if not isinstance(ap, (int, float)) or float(ap) <= 0:
                dropped["applicability missing or zero"] += 1
                continue
            site, unit, _, burn = rec["rgb_path"].split("Flame 3 Computer Vision Sets/", 1)[1].split("/")[:4]
            if not site.startswith("Sycan"):
                dropped["not the Sycan burn"] += 1
                continue
            records.append({"qid": qid, "uid": "%s/%s/%s/%s" % (site, unit, burn, rec["image_id"]), "answer": answer,
                            "max": float(ts["max"]),
                            "gt": [v for k, v in rec.items() if k.endswith("_gt") and isinstance(v, dict)]})
    return records, dropped, []


# ---------------------------------------------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------------------------------------------


def verdict(lo, hi):
    return "above" if lo > 0 else "below" if hi < 0 else "spans"


def exact_mcnemar(b_only, a_only):
    """Two-sided exact McNemar test, a binomial test on the discordant pairs; 1.0 when there are none."""
    n = b_only + a_only
    return float(stats.binomtest(min(b_only, a_only), n, 0.5).pvalue) if n else 1.0


def paired_contrast(d, mask, a, b, seed_parts, tag, names, resamples, base, units=UNITS):
    """mean(b) - mean(a) on the masked items, a paired interval per resampling unit, and an exact McNemar test.

    names labels the two discordant counts, (b right and a wrong, a right and b wrong).
    """
    a, b = a[mask], b[mask]
    frames, qtypes = d["frame"][mask], d["qtype"][mask]
    out = {"difference": float(b.mean() - a.mean())}
    for unit in units:
        if unit == "frame":
            seed = cu.stable_seed(base, *(seed_parts if tag == "pooled" else seed_parts + (tag,)))
            lo, hi, k = wp.paired_bootstrap(list(frames), a, b, resamples, seed)
        elif unit == "question_type":
            seed = cu.stable_seed(base, *(seed_parts + (tag, "qtype")))
            lo, hi, k = wp.paired_bootstrap(list(qtypes), a, b, resamples, seed)
        else:
            seed = cu.stable_seed(base, *(seed_parts + (tag, "twoway")))
            lo, hi = ws.twoway(frames, qtypes, a, b, resamples, seed)
            k = [len(set(frames.tolist())), len(set(qtypes.tolist()))]
        out[unit] = {"lo": float(lo), "hi": float(hi), "verdict": verdict(lo, hi), "clusters": k, "seed": int(seed)}
    b_only = int(((b == 1) & (a == 0)).sum())
    a_only = int(((b == 0) & (a == 1)).sum())
    out["mcnemar"] = {names[0]: b_only, names[1]: a_only, "p": exact_mcnemar(b_only, a_only)}
    return out


def value_range(values):
    return [float(min(values)), float(max(values))]


# ---------------------------------------------------------------------------------------------------------------
# blocks
# ---------------------------------------------------------------------------------------------------------------


def column_masks(d):
    formula = d["prov"] != MODEL_VERIFIED
    full = d["app"] >= 1.0
    return {"all": np.ones(len(d["ids"]), dtype=bool), "formula": formula,
            "formula_open": formula & (d["group"] != "closed_form"), "model_verified": ~formula,
            "app_full": full, "app_below": ~full}


def structure(d, masks):
    items, qtype, frame = d["items"], d["qtype"], d["frame"]
    types = sorted(set(qtype.tolist()))
    source_of = {q: sorted(set(d["prov"][qtype == q].tolist())) for q in types}
    app_values = {q: sorted(set(d["app"][qtype == q].tolist())) for q in types}
    majority_class = {q: sorted({i["baseline_majority"] for i in items if i["question_id"] == q}) for q in types}
    formula, model = masks["formula"], masks["model_verified"]

    def group(mask):
        return {"items": int(mask.sum()), "frames": len(set(frame[mask].tolist())),
                "question_types": sorted(set(qtype[mask].tolist())),
                "items_by_item_group": dict(collections.Counter(d["group"][mask].tolist()))}

    hotspot_line = [l.strip() for l in (TASK / "source-prompt-template.txt").read_text(encoding="utf-8").splitlines()
                    if "hotspot is defined" in l]
    return {
        "items": len(items), "frames": len(set(frame.tolist())), "question_types": len(types),
        "items_by_label_source": dict(collections.Counter(d["prov"].tolist())),
        "label_source_by_question_type": {q: s[0] if len(s) == 1 else s for q, s in source_of.items()},
        "question_types_mixing_label_sources": [q for q, s in source_of.items() if len(s) > 1],
        "question_types_with_more_than_one_majority_class": [q for q, s in majority_class.items() if len(s) > 1],
        "formula_telemetry_detector": group(formula),
        "model_verified": group(model),
        "frames_shared_by_the_two_groups": len(set(frame[formula].tolist()) & set(frame[model].tolist())),
        "items_by_item_group": dict(collections.Counter(d["group"].tolist())),
        "applicability": {
            "items_at_1.0": int(masks["app_full"].sum()), "items_below_1.0": int(masks["app_below"].sum()),
            "minimum": float(d["app"].min()),
            "question_types_at_1.0": len(set(qtype[masks["app_full"]].tolist())),
            "question_types_below_1.0": len(set(qtype[masks["app_below"]].tolist())),
            "question_types_with_more_than_one_value": sum(len(v) > 1 for v in app_values.values()),
            "question_types_on_both_sides_of_1.0": [q for q, v in app_values.items() if min(v) < 1.0 <= max(v)],
            "question_types_wholly_below_1.0": [q for q, v in app_values.items() if max(v) < 1.0],
            "question_types_wholly_at_1.0": [q for q, v in app_values.items() if min(v) >= 1.0],
            "closed_form_items_at_1.0": int((masks["app_full"] & (d["group"] == "closed_form")).sum()),
            "closed_form_items_below_1.0": int((masks["app_below"] & (d["group"] == "closed_form")).sum()),
            "items_by_label_group": {
                name: {"at_1.0": int((mask & masks["app_full"]).sum()), "below_1.0": int((mask & masks["app_below"]).sum())}
                for name, mask in (("formula_telemetry_detector", formula), ("model_verified", model))},
        },
        "prompt_hotspot_definition": hotspot_line[0] if hotspot_line else None,
    }


def source_pool(d, records, dropped, hot):
    """Label-source collinearity, the held-out majority, and the hotspot formula, on the raw Sycan pool."""
    by_q = collections.defaultdict(list)
    for r in records:
        by_q[r["qid"]].append(r)
    out = {"files": POOL_FILES, "records": len(records), "images": len({r["uid"] for r in records}),
           "question_types": len(by_q), "dropped": dict(dropped),
           "duplicate_frame_and_type_keys": sum(n > 1 for n in collections.Counter(
               (r["uid"], r["qid"]) for r in records).values())}

    # Label source: a formula, telemetry, or detector type carries one ground-truth dict on every record.
    gt_counts = {q: dict(collections.Counter(len(r["gt"]) for r in v)) for q, v in sorted(by_q.items())}
    with_gt = sorted(q for q, c in gt_counts.items() if set(c) == {1})
    without_gt = sorted(q for q, c in gt_counts.items() if set(c) == {0})
    formula_types = sorted(set(d["qtype"][d["prov"] != MODEL_VERIFIED].tolist()))
    out["ground_truth_dicts_per_record_by_type"] = gt_counts
    out["question_types_with_a_ground_truth_dict_on_every_record"] = with_gt
    out["question_types_mixing_ground_truth_presence"] = sorted(set(gt_counts) - set(with_gt) - set(without_gt))
    out["ground_truth_types_equal_the_item_file_formula_types"] = with_gt == formula_types

    # Held-out majority: refit the per-type majority class without the 408 evaluation records.
    items = d["items"]
    eval_keys = {(i["image"]["image_uid"], i["question_id"]) for i in items}
    held = [r for r in records if (r["uid"], r["qid"]) not in eval_keys]
    in_place = {q: collections.Counter(r["answer"] for r in v).most_common(1)[0][0] for q, v in by_q.items()}
    held_counts = collections.defaultdict(collections.Counter)
    for r in held:
        held_counts[r["qid"]][r["answer"]] += 1
    refit, margin = {}, {}
    for q, c in held_counts.items():
        top = c.most_common(2)
        refit[q] = top[0][0]
        margin[q] = top[0][1] - (top[1][1] if len(top) > 1 else 0)
    scored = [refit[i["question_id"]] == i["answer"] for i in items]
    out["held_out_majority"] = {
        "evaluation_records_found": sum((r["uid"], r["qid"]) in eval_keys for r in records),
        "records": len(held),
        "in_place_class_equals_carried_class": all(in_place[i["question_id"]] == i["baseline_majority"] for i in items),
        "question_types_whose_class_changes": sorted(q for q in refit if refit[q] != in_place[q]),
        "smallest_winning_margin_records": min(margin.values()),
        "smallest_margin_types": sorted(q for q, m in margin.items() if m == min(margin.values())),
        "correct": sum(scored), "accuracy": sum(scored) / len(items),
        "equals_carried_field_on_every_item": all(s == bool(i["baseline_majority_correct"])
                                                  for s, i in zip(scored, items)),
    }

    # The hotspot formula: which types apply a minimum hotspot radius, and what their references do above 200 C.
    floor_types = sorted(q for q, v in by_q.items()
                         if any("r_min_m" in g or "min_radius_m" in g for r in v for g in r["gt"]))
    formula = {}
    for q in floor_types:
        gts = [r["gt"][0] for r in by_q[q] if r["gt"]]
        radii = [h["equiv_radius_m"] for g in gts for h in (g.get("hotspots") or []) if "equiv_radius_m" in h]
        hot_frames = [r for r in by_q[q] if r["max"] >= NO_FIRE_C]
        nothing = [r for r in hot_frames if r["answer"] == HOTSPOT_NOTHING.get(q)]
        formula[q] = {
            "records": len(gts),
            "threshold_c": dict(collections.Counter(str(g.get("threshold_c")) for g in gts)),
            "minimum_radius_m": dict(collections.Counter(str(g.get("r_min_m", g.get("min_radius_m"))) for g in gts)),
            "smallest_listed_hotspot_radius_m": min(radii) if radii else None,
            "frames_with_maximum_at_least_200c": len(hot_frames),
            "of_which_reference_reads_nothing": len(nothing),
            "of_those_with_zero_listed_hotspots": sum(bool(r["gt"]) and not r["gt"][0].get("hotspot_count")
                                                      for r in nothing),
        }
    out["hotspot_formula"] = {"types_with_a_minimum_radius": floor_types,
                              "equal_the_hotspot_types": floor_types == sorted(HOTSPOT_NOTHING), "by_type": formula}

    # The ground truth behind the hotspot references themselves.
    want = {(str(d["frame"][k]), str(d["qtype"][k])) for k in np.flatnonzero(hot)}
    gts = [r["gt"][0] for r in records if (r["uid"], r["qid"]) in want]
    out["hotspot_references_ground_truth"] = {
        "records_found": len(gts),
        "threshold_c": dict(collections.Counter(str(g.get("threshold_c")) for g in gts)),
        "minimum_radius_m": dict(collections.Counter(str(g.get("r_min_m", g.get("min_radius_m"))) for g in gts)),
        "hotspot_count": dict(collections.Counter(str(g.get("hotspot_count")) for g in gts)),
    }
    return out


def score_columns(d, masks, args):
    """Table tab:aerial-strata, one record per column."""
    names_mm = ("arm_only_right", "majority_only_right")
    names_gb = ("gained", "lost")
    out = {}
    for key, tag, title in COLUMNS:
        mask = masks[key]
        rec = {"title": title, "seed_tag": tag, "items": int(mask.sum()), "frames": len(set(d["frame"][mask].tolist())),
               "question_types": len(set(d["qtype"][mask].tolist())),
               "majority": {"correct": int(d["majority"][mask].sum()), "accuracy": float(d["majority"][mask].mean())},
               "models": {}}
        for m in d["core"]:
            cells = {"stem": m.stem}
            for arm in ARMS:
                c = d["correct"][(m.stem, arm)]
                cells[arm] = {"correct": int(c[mask].sum()), "accuracy": float(c[mask].mean()),
                              "minus_majority": paired_contrast(d, mask, d["majority"], c, (m.stem, arm, "model-majority"),
                                                                tag, names_mm, args.resamples, args.seed)}
            cells["grounded_minus_bare"] = paired_contrast(
                d, mask, d["correct"][(m.stem, "bare")], d["correct"][(m.stem, "grounded")], (m.stem, "grounded-bare"),
                tag, names_gb, args.resamples, args.seed)
            rec["models"][m.label] = cells
        # Holm over the twelve comparator tests of the column.
        tests = [rec["models"][m.label][arm]["minus_majority"]["mcnemar"] for m in d["core"] for arm in ARMS]
        for t, p in zip(tests, holm([t["p"] for t in tests])):
            t["holm_p"] = p
        rec["summary"] = column_summary(rec)
        out[key] = rec
        print("  scored column %-15s %3d items" % (key, rec["items"]), flush=True)
    return out


def column_summary(rec):
    arms = [(label, arm, cells[arm]) for label, cells in rec["models"].items() for arm in ARMS]
    gb = [(label, cells["grounded_minus_bare"]) for label, cells in rec["models"].items()]
    s = {
        "bare_accuracy_range": value_range([c["accuracy"] for _, a, c in arms if a == "bare"]),
        "grounded_accuracy_range": value_range([c["accuracy"] for _, a, c in arms if a == "grounded"]),
        "minus_majority_range": value_range([c["minus_majority"]["difference"] for _, _, c in arms]),
        "arms_above_majority": sum(c["minus_majority"]["difference"] > 0 for _, _, c in arms),
        "arms_below_majority": sum(c["minus_majority"]["difference"] < 0 for _, _, c in arms),
        "grounded_minus_bare_range": value_range([c["difference"] for _, c in gb]),
        "grounded_minus_bare_positive": sum(c["difference"] > 0 for _, c in gb),
        "grounded_minus_bare_negative": sum(c["difference"] < 0 for _, c in gb),
        "separations": {},
    }
    for unit in UNITS:
        s["separations"][unit] = {
            "above_majority": ["%s/%s" % (l, a) for l, a, c in arms if c["minus_majority"][unit]["verdict"] == "above"],
            "below_majority": ["%s/%s" % (l, a) for l, a, c in arms if c["minus_majority"][unit]["verdict"] == "below"],
            "grounded_above_bare": [l for l, c in gb if c[unit]["verdict"] == "above"],
            "grounded_below_bare": [l for l, c in gb if c[unit]["verdict"] == "below"],
        }
    holm_p = [(c["minus_majority"]["mcnemar"]["holm_p"], "%s/%s" % (l, a)) for l, a, c in arms]
    s["smallest_holm_p"] = min(holm_p)[0]
    s["smallest_holm_p_at"] = min(holm_p)[1]
    return s


def table_summary(columns):
    """Counts over the six columns: 72 comparator cells and 36 arm-contrast cells."""
    cells = []
    for key, rec in columns.items():
        for label, m in rec["models"].items():
            for arm in ARMS:
                cells.append(("comparator", "%s/%s/%s" % (key, label, arm), m[arm]["minus_majority"]))
            cells.append(("arm", "%s/%s" % (key, label), m["grounded_minus_bare"]))
    out = {"separations_per_unit": {}}
    for unit in UNITS:
        out["separations_per_unit"][unit] = {
            kind: ["%s %s" % (name, c[unit]["verdict"]) for k, name, c in cells
                   if k == kind and c[unit]["verdict"] != "spans"] for kind in ("comparator", "arm")}
    marked = [(k, name, c) for k, name, c in cells if c["frame"]["verdict"] != "spans"]
    out["mcnemar_on_frame_markers"] = {
        "comparator_markers": sum(k == "comparator" for k, _, _ in marked),
        "comparator_markers_with_p_below_0.05": sum(k == "comparator" and c["mcnemar"]["p"] < 0.05
                                                    for k, _, c in marked),
        "arm_markers": sum(k == "arm" for k, _, _ in marked),
        "arm_markers_with_p_below_0.05": sum(k == "arm" and c["mcnemar"]["p"] < 0.05 for k, _, c in marked),
        "arm_marker_p_values": {name: c["mcnemar"]["p"] for k, name, c in marked if k == "arm"},
        "unmarked_cells_with_p_below_0.05": [name for k, name, c in cells
                                             if c["frame"]["verdict"] == "spans" and c["mcnemar"]["p"] < 0.05],
    }
    # Differences move in whole items, so an endpoint can sit exactly on zero and flip with the Monte Carlo stream.
    out["unmarked_frame_intervals_ending_at_zero"] = {
        name: [c["frame"]["lo"], c["frame"]["hi"]] for k, name, c in cells
        if c["frame"]["verdict"] == "spans" and 0.0 in (c["frame"]["lo"], c["frame"]["hi"])}
    holm_p = [(c["mcnemar"]["holm_p"], name) for k, name, c in cells if k == "comparator"]
    out["holm_within_column"] = {"smallest_adjusted_p": min(holm_p)[0], "at": min(holm_p)[1],
                                 "adjusted_below_0.05": [n for p, n in holm_p if p < 0.05]}
    return out


def per_type_open(d, masks):
    """Each question type among the 84 formula items outside the closed-form set, against the majority."""
    mask = masks["formula_open"]
    out = {}
    for q in sorted(set(d["qtype"][mask].tolist())):
        idx = mask & (d["qtype"] == q)
        maj = float(d["majority"][idx].mean())
        acc = {"%s/%s" % (m.label, arm): float(d["correct"][(m.stem, arm)][idx].mean())
               for m in d["core"] for arm in ARMS}
        out[q] = {"items": int(idx.sum()), "label_source": str(d["prov"][idx][0]), "majority_accuracy": maj,
                  "arm_accuracy": acc, "every_arm_trails": all(v < maj for v in acc.values()),
                  "every_arm_beats": all(v > maj for v in acc.values())}
    return {"types": out, "types_where_every_arm_trails": [q for q, r in out.items() if r["every_arm_trails"]],
            "types_where_every_arm_beats": [q for q, r in out.items() if r["every_arm_beats"]]}


def hotspot_mask(d):
    """Hotspot-type items whose reference reads 'nothing burning' although the frame maximum is at least 200 C."""
    answers = np.array([i["answer"] for i in d["items"]])
    return np.array([q in HOTSPOT_NOTHING and a == HOTSPOT_NOTHING[q] for q, a in zip(d["qtype"], answers)]) \
        & (d["tmax"] >= NO_FIRE_C)


def hotspot_references(d, masks, hot):
    items = d["items"]
    per_arm = {"%s/%s" % (m.label, arm): int(d["correct"][(m.stem, arm)][hot].sum()) for m in d["full"] for arm in ARMS}
    return {
        "rule": "reference is the type's nothing-burning option and temp_summary.max >= %.0f C" % NO_FIRE_C,
        "nothing_option": HOTSPOT_NOTHING,
        "nothing_option_is_the_majority_class": {
            q: all(i["baseline_majority"] == opt for i in items if i["question_id"] == q)
            for q, opt in HOTSPOT_NOTHING.items()},
        "items": int(hot.sum()),
        "items_by_type": dict(collections.Counter(d["qtype"][hot].tolist())),
        "item_ids": [d["ids"][k] for k in np.flatnonzero(hot)],
        "frame_maximum_c_range": value_range(d["tmax"][hot]),
        "items_per_column": {key: int((masks[key] & hot).sum()) for key, _, _ in COLUMNS},
        "majority_correct": int(d["majority"][hot].sum()),
        "arms_on_disk": len(per_arm),
        "correct_per_arm": per_arm,
        "arms_with_any_correct": [k for k, v in per_arm.items() if v > 0],
    }


def post_hoc(d, masks, hot, args):
    """Model minus majority without the hotspot references; frame intervals for the all-items column."""
    out = {"note": "exclusion chosen after seeing the failures; a pointer for a label audit, not a test",
           "columns": {}}
    for key, tag, _ in COLUMNS:
        mask, kept = masks[key], masks[key] & ~hot
        rec = {"items": int(kept.sum()), "removed": int((mask & hot).sum()),
               "majority_accuracy": float(d["majority"][kept].mean()), "arms": {}}
        for m in d["core"]:
            for arm in ARMS:
                c = d["correct"][(m.stem, arm)]
                before = float(c[mask].mean() - d["majority"][mask].mean())
                if key == "all":
                    cell = paired_contrast(d, kept, d["majority"], c, (m.stem, arm, "model-majority"), tag + "-nohot",
                                           ("arm_only_right", "majority_only_right"), args.resamples, args.seed,
                                           units=("frame",))
                else:
                    cell = {"difference": float(c[kept].mean() - d["majority"][kept].mean())}
                cell["difference_with_them"] = before
                rec["arms"]["%s/%s" % (m.label, arm)] = cell
        rec["arms_raised"] = sum(c["difference"] > c["difference_with_them"] for c in rec["arms"].values())
        rec["arms_unchanged"] = sum(c["difference"] == c["difference_with_them"] for c in rec["arms"].values())
        if key == "all":
            rec["frame_separations"] = {v: [k for k, c in rec["arms"].items() if c["frame"]["verdict"] == v]
                                        for v in ("above", "below")}
        out["columns"][key] = rec
    return out


def exact_tests(d):
    """Table tab:aerial-exact: grounded against bare on each item group of the thermal block."""
    out = {"groups": {g: {"items": int((d["group"] == g).sum()), "description": text} for g, text in ITEM_GROUPS},
           "models": {}}
    for m in d["full"]:
        rec = {"stem": m.stem, "tier": m.tier}
        for g, _ in ITEM_GROUPS:
            idx = d["group"] == g
            b, gr = d["correct"][(m.stem, "bare")][idx], d["correct"][(m.stem, "grounded")][idx]
            gained, lost = int(((gr == 1) & (b == 0)).sum()), int(((gr == 0) & (b == 1)).sum())
            rec[g] = {"bare_correct": int(b.sum()), "grounded_correct": int(gr.sum()),
                      "bare_accuracy": float(b.mean()), "grounded_accuracy": float(gr.mean()),
                      "gained": gained, "lost": lost, "p": exact_mcnemar(gained, lost),
                      "errors_bare": int(d["error"][(m.stem, "bare")][idx].sum()),
                      "errors_grounded": int(d["error"][(m.stem, "grounded")][idx].sum())}
        out["models"][m.label] = rec
    full = out["models"]
    core = [m.label for m in d["core"]]
    cf = {k: v["closed_form"] for k, v in full.items()}
    out["closed_form_summary"] = {
        "models": len(cf),
        "raised": [k for k, v in cf.items() if v["gained"] > v["lost"]],
        "raised_at_p_below_0.05": [k for k, v in cf.items() if v["gained"] > v["lost"] and v["p"] < 0.05],
        "declined_at_p_below_0.05": [k for k, v in cf.items() if v["gained"] < v["lost"] and v["p"] < 0.05],
    }
    rest = {k: full[k]["not_answered"] for k in core}
    adj = holm([rest[k]["p"] for k in core])
    out["not_answered_summary"] = {
        "core_models_whose_accuracy_falls": [k for k in core if rest[k]["grounded_accuracy"] < rest[k]["bare_accuracy"]],
        "p_below_0.05": [k for k in core if rest[k]["p"] < 0.05],
        "holm_p_across_the_core_six": dict(zip(core, adj)),
        "holm_p_below_0.05": [k for k, p in zip(core, adj) if p < 0.05],
    }
    out["no_fire_branch_summary"] = {"core_p_below_0.05": [k for k in core if full[k]["no_fire_branch"]["p"] < 0.05]}
    return out


# ---------------------------------------------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------------------------------------------


def fmt_p(p):
    """Three decimals, as the paper prints them; a value that rounds to zero shows as < 0.001."""
    return "< 0.001" if round(p, 3) == 0 else "%.3f" % p


def mark(cell):
    return {"above": "^", "below": "v", "spans": " "}[cell["frame"]["verdict"]]


def report(out):
    st, cols = out["structure"], out["columns"]
    print("\nlabel source: %s; question types mixing sources: %d; frames shared by the two groups: %d of %d"
          % (st["items_by_label_source"], len(st["question_types_mixing_label_sources"]),
             st["frames_shared_by_the_two_groups"], st["frames"]))
    ap = st["applicability"]
    print("applicability: %d items at 1.0 (%d types), %d below (%d types); %d types vary, %d straddle 1.0"
          % (ap["items_at_1.0"], ap["question_types_at_1.0"], ap["items_below_1.0"], ap["question_types_below_1.0"],
             ap["question_types_with_more_than_one_value"], len(ap["question_types_on_both_sides_of_1.0"])))
    pool = out["source_pool"]
    if "skipped" in pool:
        print("source pool: SKIPPED, %s" % pool["skipped"])
    else:
        ho = pool["held_out_majority"]
        print("source pool: %d records, types mixing ground-truth presence %s; held-out majority %d/%d = %.4f, "
              "equal to the carried field on every item: %s"
              % (pool["records"], pool["question_types_mixing_ground_truth_presence"] or "none", ho["correct"],
                 st["items"], ho["accuracy"], ho["equals_carried_field_on_every_item"]))

    print("\nTable tab:aerial-strata (^ / v: frame-clustered interval wholly above / below zero)")
    print("%-26s" % "" + "".join("%12s" % ("%d (%d)" % (c["items"], c["question_types"])) for c in cols.values()))
    print("%-26s" % "held-out majority" + "".join("%12.3f" % c["majority"]["accuracy"] for c in cols.values()))
    labels = list(next(iter(cols.values()))["models"])
    for label in labels:
        for arm in ARMS:
            print("%-26s" % ("%s %s" % (label, arm)) + "".join(
                "%11.3f%s" % (c["models"][label][arm]["accuracy"], mark(c["models"][label][arm]["minus_majority"]))
                for c in cols.values()))
    for label in labels:
        print("%-26s" % ("%s g-b" % label) + "".join(
            "%+11.3f%s" % (c["models"][label]["grounded_minus_bare"]["difference"],
                           mark(c["models"][label]["grounded_minus_bare"])) for c in cols.values()))

    ts = out["table_summary"]
    print("\nseparations per unit (comparator, arm): %s" % {u: (len(v["comparator"]), len(v["arm"]))
                                                          for u, v in ts["separations_per_unit"].items()})
    mc = ts["mcnemar_on_frame_markers"]
    print("McNemar p < 0.05 on %d of %d comparator markers and %d of %d arm markers; unmarked cells at p < 0.05: %d"
          % (mc["comparator_markers_with_p_below_0.05"], mc["comparator_markers"], mc["arm_markers_with_p_below_0.05"],
             mc["arm_markers"], len(mc["unmarked_cells_with_p_below_0.05"])))
    print("unmarked frame intervals ending at exactly zero: %d" % len(ts["unmarked_frame_intervals_ending_at_zero"]))
    print("Holm within column: smallest adjusted comparator p %.4f (%s)"
          % (ts["holm_within_column"]["smallest_adjusted_p"], ts["holm_within_column"]["at"]))
    print("types on the 84 where every arm trails the majority: %s" % out["per_type_open"]["types_where_every_arm_trails"])

    hr = out["hotspot_references"]
    print("\nhotspot references: %d items %s, frame maximum %.1f to %.1f C, majority right on %d, "
          "arms with any right: %d of %d" % (hr["items"], hr["items_by_type"], *hr["frame_maximum_c_range"],
                                             hr["majority_correct"], len(hr["arms_with_any_correct"]), hr["arms_on_disk"]))
    ph = out["post_hoc_without_hotspot_references"]["columns"]["all"]
    c5 = ph["arms"]["claude-opus-5/grounded"]
    print("without them (%d items): claude-opus-5 grounded %+.3f [%+.3f, %+.3f], McNemar p %.3f; frame separations %s"
          % (ph["items"], c5["difference"], c5["frame"]["lo"], c5["frame"]["hi"], c5["mcnemar"]["p"],
             ph["frame_separations"]))

    ex = out["exact_tests_by_item_group"]
    print("\nTable tab:aerial-exact")
    print("%-18s %14s %8s %16s %12s %8s" % ("model", "48 gained/lost", "p", "347 bare->grnd", "gained/lost", "p"))
    for label, rec in ex["models"].items():
        cf, no = rec["closed_form"], rec["not_answered"]
        tail = ("%7.3f -> %.3f %6d / %-4d %8s" % (no["bare_accuracy"], no["grounded_accuracy"], no["gained"],
                                                   no["lost"], fmt_p(no["p"]))) if rec["tier"] == "core" else ""
        print("%-18s %6d / %-5d %8s %s" % (label, cf["gained"], cf["lost"], fmt_p(cf["p"]), tail))
    print("closed form: raised %d of %d, %d at p < 0.05; 347: falls for %d core models, p < 0.05 for %s, Holm below 0.05: %s"
          % (len(ex["closed_form_summary"]["raised"]), ex["closed_form_summary"]["models"],
             len(ex["closed_form_summary"]["raised_at_p_below_0.05"]),
             len(ex["not_answered_summary"]["core_models_whose_accuracy_falls"]),
             ex["not_answered_summary"]["p_below_0.05"], ex["not_answered_summary"]["holm_p_below_0.05"] or "none"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--resamples", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20260915)
    ap.add_argument("--out", type=pathlib.Path, default=ROOT / "analysis" / "wildfirevqa_label_source.json")
    args = ap.parse_args()

    d = load()
    masks = column_masks(d)
    hot = hotspot_mask(d)
    print("items %d, core models %d, full-capability models %d, resamples %d, base seed %d"
          % (len(d["ids"]), len(d["core"]), len(d["full"]), args.resamples, args.seed))
    out = {"resamples": args.resamples, "seed": args.seed,
           "seed_recipe": {"model_minus_majority": "stable_seed(base, stem, arm, 'model-majority'[, tag][, unit])",
                           "grounded_minus_bare": "stable_seed(base, stem, 'grounded-bare'[, tag][, unit])",
                           "note": "the all-items frame rows omit tag and unit; the post hoc exclusion uses tag + '-nohot'"},
           "core_models": [m.label for m in d["core"]], "full_capability_models": [m.label for m in d["full"]],
           "structure": structure(d, masks)}

    records, dropped, missing = load_pool()
    if records is None:
        out["source_pool"] = {"skipped": "%d of the three Sycan question files of the raw WildFireVQA release "
                                         "(vqa_response_Sycan_*.json, huggingface.co/datasets/mobiiin/WildFire_VQA) "
                                         "are missing under %s" % (len(missing), POOL_DIR.relative_to(ROOT)),
                              "missing": missing}
        print("NOTE: source_pool skipped: %s" % out["source_pool"]["skipped"])
    else:
        out["source_pool"] = source_pool(d, records, dropped, hot)

    out["columns"] = score_columns(d, masks, args)
    out["table_summary"] = table_summary(out["columns"])
    out["per_type_open"] = per_type_open(d, masks)
    out["hotspot_references"] = hotspot_references(d, masks, hot)
    out["post_hoc_without_hotspot_references"] = post_hoc(d, masks, hot, args)
    out["exact_tests_by_item_group"] = exact_tests(d)

    args.out.write_text(json.dumps(out, indent=1), encoding="utf-8")
    report(out)
    print("\nwrote %s" % args.out)


if __name__ == "__main__":
    main()
