"""Clipped thermal maxima in WildFireVQA, and grounded misses on the closed-form items.

Four WildFireVQA question types (CL1, CMR4, DS7, DS8) have labels that a fixed rule reproduces from the seven-number
thermal summary block (build_items_wildfirevqa.TS_RULES): class cuts at 50, 80, and 200 C for CL1, "No fire
detected" below 200 C for CMR4, and "None" below 400 C or 200 C for DS7 and DS8. The prompt states none of these
cuts. The rules read each frame's maximum. On 67 FLAME 3 frames filed under Fire, the maximum is exactly
187.424 C. Many pixels share that value, and no Sycan maximum lies between it and 200 C. These patterns support
clipping in the released values; the files do not show whether the camera's range setting or later processing
imposed it. The release labels all 67 as smoldering or fire-free on every temperature-derived question type.

This script backs, in the paper, the clipped-maximum statements of the abstract, Section 3.5, Key Finding (v),
insights I1 and I5, and the Conclusion; the aerial rows of Table 4 (tab:failure-modes); the source-audit row and
the clipped-maximum subsection of the defects appendix; the unit-gloss paragraph of the harness appendix; the
breakdown of the closed-form misses in the aerial results appendix; and the clipping and unit-gloss sentences of
the Limitations.

JSON blocks, in order:
  tiff_ceiling        the 738 Sycan Marsh Celsius TIFFs: frames whose maximum is exactly 187.424 C, their burn
                      folder, how many pixels sit at the maximum there and elsewhere, the pile-up against the 1 C
                      band below the maximum, the gap between 187.424 C and 200 C, and the shared minimum
                      (declared-optional input, see Inputs)
  release_ceiling     the raw WildFireVQA release: maxima that recur exactly across frames, per unit; the labels
                      the release gives the 187.424 C frames on the nine temperature-derived types; the
                      Shoetank frames that share one maximum; and PD1 on every frame below 200 C
                      (declared-optional input)
  items               evaluation items on the clipped frames, by question type and item group, the three PD1
                      items below 200 C that the no-fire branch of the builder leaves out, and the items with a
                      percent value above 1
  closed_form_misses  every grounded miss of the sixteen full-capability models on the 48 closed-form items,
                      classed by the first matching rule: on a clipped frame; a CL1 answer matching an area
                      reading ("Smoldering" where the maximum is at least 200 C); a DS answer matching
                      fraction-based binning; a DS answer matching the other percent field; "None" matching a
                      displayed share of 0.0; other. A class records which reading an answer matches, never the
                      model's reasoning, which the stored answers do not give. For the core six,
                      also what the block did to each miss (the bare answer was the same, a different wrong
                      answer, or correct) and gained and lost items per model
  no_fire_branch      grounded misses of the core six on the 13 no-fire-branch items, and on those on clipped frames

Inputs:
  task-wildfirevqa/items.jsonl, and responses-<stem>-{bare,grounded}.jsonl for the sixteen full-capability models
    (models.py tier "full"; the six core models are tier "core")
  <data>/wildfirevqa/vqa_response_*.json, the eight files of the raw WildFireVQA question release
    (huggingface.co/datasets/mobiiin/WildFire_VQA; build_items_wildfirevqa.py downloads the three Sycan files, and
    the other five come from the same page). Declared-optional: without them release_ceiling is skipped.
  <data>/flame3/FLAME 3 CV Dataset (Sycan Marsh)/{Fire,No Fire}/Thermal/Celsius TIFF/*.TIFF, the FLAME 3
    computer-vision subset that fetch_flame3.py unpacks. Declared-optional: without it tiff_ceiling is skipped.
  <data> defaults to data/ under the repository root.
Output: analysis/wildfirevqa_ceiling.json

    python analysis/wildfirevqa_ceiling.py                  # about 20 seconds with the TIFFs, 2 without
    python analysis/wildfirevqa_ceiling.py --data /path/to/data
"""
import argparse
import collections
import glob
import json
import os
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT))

import cluster_uncertainty as cu  # noqa: E402
import models  # noqa: E402

TASK = ROOT / "task-wildfirevqa"
OUT = ROOT / "analysis" / "wildfirevqa_ceiling.json"
ARMS = ("bare", "grounded")
CEILING = 187.424            # the recurring Sycan maximum, as the release and items.jsonl round it
TEMP_TYPES = ("CL1", "CMR4", "DS7", "DS8", "PD1", "PD7", "DS1", "DS3", "LD1")
# The DS7 and DS8 bins of build_items_wildfirevqa.TS_RULES, applied to a share given in percent.
BINS = {"DS7": ((2, "<2%"), (4, "2-4%"), (6, "4-6%")), "DS8": ((5, "<5%"), (10, "5-10%"), (15, "10-15%"))}
TOP = {"DS7": ">6%", "DS8": ">15%"}
FIELD = {"DS7": "pct_over_400", "DS8": "pct_over_200"}
CUT = {"DS7": 400, "DS8": 200}


def at_ceiling(value):
    return abs(float(value) - CEILING) < 5e-4


def share_bin(qid, pct):
    """The DS7 or DS8 option a share in percent falls in."""
    for edge, label in BINS[qid]:
        if pct < edge:
            return label
    return TOP[qid]


def load(d_stems):
    items = [json.loads(l) for l in (TASK / "items.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    resp = {}
    for m in d_stems:
        for arm in ARMS:
            rows, bad = cu.read_jsonl(TASK / ("responses-%s-%s.jsonl" % (m.stem, arm)))
            by_id, _ = cu.dedupe_by_item(rows)
            if bad or any(i["item_id"] not in by_id for i in items):
                raise SystemExit("responses-%s-%s.jsonl: unreadable lines or missing items" % (m.stem, arm))
            resp[(m.stem, arm)] = by_id
    return items, resp


# ---------------------------------------------------------------------------------------------------------------
# sources
# ---------------------------------------------------------------------------------------------------------------


def tiff_ceiling(data):
    root = data / "flame3" / "FLAME 3 CV Dataset (Sycan Marsh)"
    if not root.is_dir():
        print("tiff_ceiling: %s not found, block skipped" % root)
        return None
    from PIL import Image
    rows = []
    for burn in ("Fire", "No Fire"):
        for p in sorted((root / burn / "Thermal" / "Celsius TIFF").glob("*.TIFF")):
            a = np.asarray(Image.open(p), dtype=np.float32).ravel()
            mx = float(a.max())
            rows.append({"burn": burn, "stem": p.stem, "max": mx, "min": float(a.min()),
                         "at_max": int((a == a.max()).sum()),
                         "band_below": int(((a >= mx - 1.5) & (a < mx - 0.5)).sum())})
    ceil = [r for r in rows if at_ceiling(r["max"])]
    rest = [r for r in rows if not at_ceiling(r["max"])]
    mins = collections.Counter(round(r["min"], 3) for r in ceil).most_common(1)[0]
    return {
        "frames": len(rows),
        "frames_by_burn": dict(collections.Counter(r["burn"] for r in rows)),
        "at_ceiling": len(ceil),
        "at_ceiling_by_burn": dict(collections.Counter(r["burn"] for r in ceil)),
        "pixels_at_max_median": {"ceiling": float(np.median([r["at_max"] for r in ceil])),
                                 "other": float(np.median([r["at_max"] for r in rest]))},
        "ceiling_frames_with_10_or_more_at_max": sum(r["at_max"] >= 10 for r in ceil),
        "ceiling_frames_more_at_max_than_in_1C_band_below": sum(r["at_max"] > r["band_below"] for r in ceil),
        "frames_with_max_between_ceiling_and_200": sum(CEILING + 5e-4 < r["max"] < 200 for r in rows),
        "most_common_min_on_ceiling_frames": {"value": mins[0], "frames": mins[1]},
    }


def release_ceiling(data):
    files = sorted(glob.glob(str(data / "wildfirevqa" / "vqa_response_*.json")))
    if not files:
        print("release_ceiling: no release files under %s, block skipped" % (data / "wildfirevqa"))
        return None
    labels, tmax = collections.defaultdict(dict), {}
    pct_checked = pct_match = 0
    for f in files:
        unit = os.path.basename(f).split("_altitude")[0].replace("vqa_response_", "")
        for r in json.loads(pathlib.Path(f).read_text(encoding="utf-8")):
            q = r.get("question_id")
            if q is None:
                continue
            key = (unit, r["image_id"])
            labels[key][q] = r.get("answer")
            if q == "CL1" and isinstance(r.get("cl1_gt"), dict) and r["cl1_gt"].get("tmax_c") is not None:
                tmax[key] = float(r["cl1_gt"]["tmax_c"])
            gt = r.get(q.lower() + "_gt") if q in ("DS7", "DS8") else None
            if isinstance(gt, dict) and gt.get("total_pixels"):
                pct_checked += 1
                pct_match += abs(100 * gt["count_pixels_ge_threshold"] / gt["total_pixels"]
                                 - gt["pct_scene_ge_threshold"]) < 1e-9
    repeated = collections.Counter(round(t, 3) for t in tmax.values())
    by_unit = collections.defaultdict(collections.Counter)
    for (unit, _), t in tmax.items():
        by_unit[unit][round(t, 3)] += 1
    ceil = [k for k, t in tmax.items() if at_ceiling(t)]
    shoe = [k for k in tmax if k[0] == "Shoetank_FIRE"]
    shoe_top, shoe_n = by_unit["Shoetank_FIRE"].most_common(1)[0]
    shoe_at = [k for k in shoe if round(tmax[k], 3) == shoe_top]
    below = [k for k, t in tmax.items() if t < 200]
    return {
        "frames": len(tmax),
        "most_repeated_maxima": repeated.most_common(4),
        "most_repeated_by_unit": {u: c.most_common(2) for u, c in sorted(by_unit.items())},
        "ceiling_frames": len(ceil),
        "ceiling_frames_by_unit": dict(collections.Counter(k[0] for k in ceil)),
        "ceiling_labels": {q: dict(collections.Counter(labels[k].get(q) for k in ceil)) for q in TEMP_TYPES},
        "ceiling_label_records": sum(1 for k in ceil for q in TEMP_TYPES if q in labels[k]),
        "shoetank_fire_frames": len(shoe),
        "shoetank_shared_max": {"value": shoe_top, "frames": shoe_n,
                                "labels": {q: dict(collections.Counter(labels[k].get(q) for k in shoe_at))
                                           for q in ("CL1", "CMR4", "DS7", "DS8")}},
        "frames_below_200": len(below),
        "pd1_labels_below_200": dict(collections.Counter(labels[k].get("PD1") for k in below)),
        "labels_below_200": {q: dict(collections.Counter(labels[k].get(q) for k in below if q in labels[k]))
                             for q in TEMP_TYPES},
        "pct_records_checked": pct_checked,
        "pct_equals_100_count_over_total": pct_match,
    }


# ---------------------------------------------------------------------------------------------------------------
# items and responses
# ---------------------------------------------------------------------------------------------------------------


def item_block(items):
    on = [i for i in items if at_ceiling(i["temp_summary"]["max"])]
    temp = [i for i in on if i["question_id"] in TEMP_TYPES]
    pd1 = [i["item_id"] for i in items if i["question_id"] == "PD1" and i["temp_summary"]["max"] < 200]
    above_one = [i for i in items if i["temp_summary"]["pct_over_200"] > 1 or i["temp_summary"]["pct_over_400"] > 1]
    return {
        "items_with_a_pct_value_above_1": len(above_one),
        "items_on_ceiling_frames": len(on),
        "ceiling_frames_evaluated": len({i["image"]["image_uid"] for i in on}),
        "temperature_derived_items_on_ceiling_frames": len(temp),
        "by_group": dict(collections.Counter(i.get("no_image_answerable") or "other" for i in temp)),
        "ids": sorted(i["item_id"] for i in temp),
        "labels": {i["item_id"]: i["answer"] for i in temp},
        "pd1_items_below_200": pd1,
        "pd1_labels_below_200": dict(collections.Counter(i["answer"] for i in items if i["item_id"] in pd1)),
    }


def classify(item, pred):
    """Class of one grounded miss on a closed-form item, first rule that fits."""
    q, ts = item["question_id"], item["temp_summary"]
    if at_ceiling(ts["max"]):
        return "clipped_frame"
    if q == "CL1" and ts["max"] >= 200 and pred == "Smoldering":
        return "area_reading_match"
    if q in BINS and ts["max"] >= CUT[q]:
        if pred == share_bin(q, 100 * ts[FIELD[q]]):
            return "fraction_match"
        other = FIELD["DS8" if q == "DS7" else "DS7"]
        if pred == share_bin(q, ts[other]):
            return "other_field_match"
        if pred == "None" and ts[FIELD[q]] == 0:
            return "none_matches_displayed_zero"
    return "other"


def closed_form(items, resp, full):
    cf = [i for i in items if i.get("no_image_answerable") == "closed_form"]
    out = {"items": len(cf), "per_model": {}}
    core_misses = []
    for m in full:
        g, b = resp[(m.stem, "grounded")], resp[(m.stem, "bare")]
        misses = [i for i in cf if g[i["item_id"]].get("correct") is not True]
        cls = collections.Counter(classify(i, g[i["item_id"]].get("prediction")) for i in misses)
        rec = {"tier": m.tier, "misses": len(misses), "by_class": dict(cls),
               "gained": sum(g[i["item_id"]].get("correct") is True and b[i["item_id"]].get("correct") is not True
                             for i in cf),
               "lost": sum(g[i["item_id"]].get("correct") is not True and b[i["item_id"]].get("correct") is True
                           for i in cf)}
        if m.tier == "core":
            for i in misses:
                gp, bp = g[i["item_id"]].get("prediction"), b[i["item_id"]].get("prediction")
                effect = ("bare_correct" if b[i["item_id"]].get("correct") is True else
                          "same_as_bare" if gp == bp else "other_wrong_bare")
                core_misses.append({"model": m.label, "item": i["item_id"], "reference": i["answer"],
                                    "grounded": gp, "bare": bp, "class": classify(i, gp), "effect": effect,
                                    "weights": m.weights})
        out["per_model"][m.label] = rec
    out["core_misses"] = core_misses
    dom = sorted({r["item"] for r in core_misses if r["class"] == "area_reading_match"})
    by_id = {i["item_id"]: i for i in cf}
    out["area_reading_items"] = {iid: by_id[iid]["temp_summary"]["pct_over_200"] for iid in dom}
    out["core_total"] = len(core_misses)
    out["core_by_class"] = dict(collections.Counter(r["class"] for r in core_misses))
    out["core_by_effect"] = dict(collections.Counter(r["effect"] for r in core_misses))
    out["core_introduced_by_class_and_weights"] = {"%s|%s" % k: v for k, v in collections.Counter(
        (r["class"], r["weights"]) for r in core_misses if r["effect"] == "bare_correct").most_common()}
    out["core_by_class_and_weights"] = {"%s|%s" % k: v for k, v in collections.Counter(
        (r["class"], r["weights"]) for r in core_misses).items()}
    added = [r for lab, r in out["per_model"].items() if r["tier"] == "added"]
    out["added_ten"] = {"misses": sum(r["misses"] for r in added),
                        "by_class": dict(sum((collections.Counter(r["by_class"]) for r in added),
                                             collections.Counter()))}
    return out


def no_fire_branch(items, resp, core):
    nf = [i for i in items if i.get("no_image_answerable") == "no_fire_branch"]
    miss = [(m.label, i) for m in core for i in nf if resp[(m.stem, "grounded")][i["item_id"]].get("correct") is not True]
    on = [i for i in nf if at_ceiling(i["temp_summary"]["max"])]
    return {"items": len(nf), "answers": len(nf) * len(core), "misses": len(miss),
            "items_on_ceiling": len(on), "answers_on_ceiling": len(on) * len(core),
            "misses_on_ceiling": sum(at_ceiling(i["temp_summary"]["max"]) for _, i in miss)}


def report(out):
    t, r, it, cf, nf = (out.get(k) for k in ("tiff_ceiling", "release_ceiling", "items", "closed_form_misses",
                                            "no_fire_branch"))
    if t:
        print("TIFFs: %d frames; %d at %.3f C, by burn %s; pixels at the max, median %.0f there against %.0f elsewhere;"
              " %d with more pixels at the max than in the 1 C band below; %d frames with a maximum between the"
              " 187.424 C and 200 C; minimum %.3f C on %d of them"
              % (t["frames"], t["at_ceiling"], CEILING, t["at_ceiling_by_burn"], t["pixels_at_max_median"]["ceiling"],
                 t["pixels_at_max_median"]["other"], t["ceiling_frames_more_at_max_than_in_1C_band_below"],
                 t["frames_with_max_between_ceiling_and_200"], t["most_common_min_on_ceiling_frames"]["value"],
                 t["most_common_min_on_ceiling_frames"]["frames"]))
    if r:
        print("release: %d frames; %d at the clipped maximum %s; %d label records; labels %s"
              % (r["frames"], r["ceiling_frames"], r["ceiling_frames_by_unit"], r["ceiling_label_records"],
                 r["ceiling_labels"]))
        s = r["shoetank_shared_max"]
        print("release: %d of %d Shoetank fire frames share the maximum %.2f C, labels %s; most repeated maxima %s"
              % (s["frames"], r["shoetank_fire_frames"], s["value"], s["labels"], r["most_repeated_maxima"]))
        print("release: PD1 on the %d frames below 200 C: %s" % (r["frames_below_200"], r["pd1_labels_below_200"]))
        print("release: labels on frames below 200 C: %s" % r["labels_below_200"])
        print("release: DS7/DS8 pct equals 100 x count / total on %d of %d records"
              % (r["pct_equals_100_count_over_total"], r["pct_records_checked"]))
    print("items: %d on %d clipped frames, %d temperature-derived %s; PD1 below 200 C: %s"
          % (it["items_on_ceiling_frames"], it["ceiling_frames_evaluated"],
             it["temperature_derived_items_on_ceiling_frames"], it["by_group"], it["pd1_items_below_200"]))
    print("items with a pct value above 1: %d; area-reading items and their pct_over_200: %s"
          % (it["items_with_a_pct_value_above_1"], cf["area_reading_items"]))
    print("closed form, core six: %d grounded misses; by class %s; by effect %s; introduced %s"
          % (cf["core_total"], cf["core_by_class"], cf["core_by_effect"], cf["core_introduced_by_class_and_weights"]))
    for lab, rec in cf["per_model"].items():
        print("  %-24s %-5s misses %2d  gained/lost %2d/%2d  %s"
              % (lab, rec["tier"], rec["misses"], rec["gained"], rec["lost"], rec["by_class"]))
    print("closed form, added ten: %d misses, by class %s" % (cf["added_ten"]["misses"], cf["added_ten"]["by_class"]))
    print("no-fire branch, core six: %d of %d grounded answers miss; %d of %d on the %d clipped-frame items"
          % (nf["misses"], nf["answers"], nf["misses_on_ceiling"], nf["answers_on_ceiling"], nf["items_on_ceiling"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", type=pathlib.Path, default=ROOT / "data")
    args = ap.parse_args()
    full = models.models(tier="full", task="wildfirevqa")
    core = [m for m in full if m.tier == "core"]
    items, resp = load(full)
    out = {"tiff_ceiling": tiff_ceiling(args.data), "release_ceiling": release_ceiling(args.data),
           "items": item_block(items), "closed_form_misses": closed_form(items, resp, full),
           "no_fire_branch": no_fire_branch(items, resp, core)}
    OUT.write_text(json.dumps(out, indent=1, default=str) + "\n", encoding="utf-8")
    report(out)
    print("wrote %s" % OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
