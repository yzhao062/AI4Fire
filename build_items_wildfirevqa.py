"""Build the version-1 temperature-grounded aerial VQA task from WildFireVQA over FLAME 3 imagery.

Selection rule, published with the item set:
  - the Sycan Marsh burn only (units Sycan 2A and Sycan 2D, 738 of the 6,097 released image samples),
    because that is the one FLAME 3 burn whose imagery is a public download; the Shoetank and Willamette
    units in the same question files belong to the full six-burn FLAME 3 set, which the FLAME 3 paper
    releases only "upon request"
  - a record is valid when its answer matches one of its options after case folding, its temp_summary
    carries all seven keys, and its applicability_score is a number above 0
  - 12 items per question id, over all 34 question ids, so the item set spans the six question
    categories in the proportion the source uses
  - within a question id, candidates are ordered by (unit, burn state, image id) and 12 evenly spaced
    positions are taken, with the starting offset rotated by the question id's rank so that different
    questions land on different images
The reference answer is the released answer string, and every item records where that answer came from:
a radiometric threshold on the Celsius TIFF, the flight log, an object detector, or the paper's
MLLM-plus-manual-verification pipeline. The item set ships the question, the options and the temp_summary
block, which is all a text-only run needs; the images are a separate download and are not shipped here.

Getting the images (a person has to do the account step; nothing below needs a request or an approval):
  1. Create a free IEEE account, then download CVSubset.zip (6.23 GB) from the FLAME 3 record at
     https://ieee-dataport.org/open-access/flame-3-radiometric-thermal-uav-imagery-wildfire-management
     (DOI 10.21227/w0mz-aq48). The page states "Open Access dataset files are accessible to all logged in
     users. Don't have a login? Create a free IEEE account." IEEE membership is not required. Skip
     NADIRPlots.zip (32.59 GB): it is the Hanna Hammock plot set and holds none of these images.
     A mirror of the same burn is on Kaggle behind a free Kaggle account, 7.54 GB, 5,740 downloads:
     https://www.kaggle.com/datasets/brycehopkins/flame-3-computer-vision-subset-sycan-marsh
  2. The zip holds one Fire folder of 622 quartets and one No Fire folder of 116, numbered flat, while
     the paths in this item set are WildFireVQA's per-unit layout (Sycan 2A 00001-00364 fire, Sycan 2D
     00001-00258 fire, Sycan 2D 00001-00116 no fire). Establish the correspondence by fingerprint rather
     than by guessing the concatenation order: each item's temp_summary is min/max/mean/std of its own
     Celsius TIFF to three decimals, so reading the 738 TIFFs and matching those four numbers recovers the
     mapping exactly and checks the item set at the same time.
"""
import json
import pathlib
from collections import Counter, defaultdict

import httpx

S = pathlib.Path(__file__).parent
DATA = S / "data" / "wildfirevqa"
OUT = S / "task-wildfirevqa"
HF = "https://huggingface.co/datasets/mobiiin/WildFire_VQA/resolve/main/"
STEM = "_altitude_GT_hotspot_PD7_DS1_DS3_DS7_DS8_LD1_CMR4_PD8_CL1_CL1.json"
# Every released question file. The three Sycan files are the ones this task draws on; the rest name the
# units whose imagery needs an author request, and are downloaded only so the pool counts can be printed.
UNITS = ["Sycan_2A_FIRE", "Sycan_2D_FIRE", "Sycan_2D_NoFIRE", "Shoetank_FIRE", "Shoetank_NoFIRE",
         "Wilamette_Hay_FIRE", "Wilamette_Office_NoFIRE", "Wilamette_Rathbone_FIRE"]
PUBLIC_BURN = "Sycan"      # set to "" to pool every unit, once the full FLAME 3 set is in hand
PER_QUESTION = 12
TS_KEYS = ("min", "max", "mean", "std", "top3_mean", "pct_over_200", "pct_over_400")

# Where each reference answer comes from. The nine radiometric ids are thresholded on the per-pixel Celsius
# TIFF, FP2 comes from the drone's EXIF altitude minus an SRTM ground elevation, and PD8 comes from a
# YOLO-World open-vocabulary detector, which is a model and not a sensor. Everything else is the paper's
# "MLLM-based answer generation ... and manual verification".
PROVENANCE = {"PD1": "radiometric", "PD7": "radiometric", "CL1": "radiometric", "DS1": "radiometric",
              "DS3": "radiometric", "DS7": "radiometric", "DS8": "radiometric", "LD1": "radiometric",
              "CMR4": "radiometric", "FP2": "telemetry", "PD8": "object_detector"}
GT_KEY = {"PD1": "hotspot_gt", "PD7": "pd7_gt", "PD8": "pd8_gt", "CL1": "cl1_gt", "DS1": "ds1_gt",
          "DS3": "ds3_gt", "DS7": "ds7_gt", "DS8": "ds8_gt", "LD1": "ld1_gt", "CMR4": "cmr4_gt",
          "FP2": "altitude_gt"}

# Closed-form rules over temp_summary alone. Each was checked against all 6,097 released samples before
# being written down here; build_items_wildfirevqa re-checks them on the pool it builds and prints the rate.
TS_RULES = {
    "CMR4": ("bin temp_summary.max at 200/300/400/500, below 200 is 'No fire detected'",
             lambda t: ("No fire detected" if t["max"] < 200 else "200-300" if t["max"] < 300 else
                        "300-400" if t["max"] < 400 else "400-500" if t["max"] < 500 else ">500")),
    "CL1": ("bin temp_summary.max at 50/80/200",
            lambda t: ("Active fire" if t["max"] >= 200 else "Smoldering" if t["max"] >= 80 else
                       "Extinguished" if t["max"] >= 50 else "No fire")),
    "DS7": ("'None' when temp_summary.max < 400, else bin temp_summary.pct_over_400 at 2/4/6",
            lambda t: ("None" if t["max"] < 400 else "<2%" if t["pct_over_400"] < 2 else
                       "2-4%" if t["pct_over_400"] < 4 else "4-6%" if t["pct_over_400"] < 6 else ">6%")),
    "DS8": ("'None' when temp_summary.max < 200, else bin temp_summary.pct_over_200 at 5/10/15",
            lambda t: ("None" if t["max"] < 200 else "<5%" if t["pct_over_200"] < 5 else
                       "5-10%" if t["pct_over_200"] < 10 else "10-15%" if t["pct_over_200"] < 15 else ">15%")),
}
# Four more question ids whose 'nothing is burning' answer, and only that answer, follows from
# temp_summary.max < 200; above that threshold they need the thermal image and the rule abstains.
NO_FIRE = {"PD7": "No fire", "DS1": "No active hotspots", "DS3": "No active hotspots", "LD1": "No hotspots"}

OUT.mkdir(exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)
client = httpx.Client(timeout=300, follow_redirects=True)
for unit in UNITS:
    dest = DATA / ("vqa_response_%s%s" % (unit, STEM))
    if dest.exists():
        continue
    with client.stream("GET", HF + dest.name) as r:
        r.raise_for_status()
        with dest.open("wb") as fh:
            for chunk in r.iter_bytes(1 << 20):
                fh.write(chunk)
    print("downloaded %s (%d bytes)" % (dest.name, dest.stat().st_size))


def canon(answer, options):
    """The released answer is sometimes cased differently from its option ('No Fire' against 'No fire')."""
    if not isinstance(answer, str):
        return None
    if answer in options:
        return answer
    return {o.lower(): o for o in options}.get(answer.strip().lower())


def parse_path(p):
    """.../Flame 3 Computer Vision Sets/<site>/<unit>/Images/<burn state>/... -> the archive-relative tail."""
    tail = p.split("Flame 3 Computer Vision Sets/", 1)[1]
    parts = tail.split("/")
    return tail, parts[0], parts[1], parts[3]


rows, tiffs, flight, prompts, dropped = [], {}, {}, [], Counter()
for unit in UNITS:
    for rec in json.loads((DATA / ("vqa_response_%s%s" % (unit, STEM))).read_text(encoding="utf-8")):
        qid = rec.get("question_id")
        if qid is None:
            prompts.append(rec.get("prompt_template", ""))
            continue                                  # the one prompt_template header per file
        ts, ap = rec.get("temp_summary"), rec.get("applicability_score")
        ans = canon(rec.get("answer"), rec["options"])
        rel, site, unit_name, burn = parse_path(rec["rgb_path"])
        uid = "%s/%s/%s/%s" % (site, unit_name, burn, rec["image_id"])
        gt = rec.get(GT_KEY.get(qid, ""))
        # The TIFF path and the flight log are per-image facts rather than answers, so they are indexed
        # before the validity filter runs; otherwise an image whose FP2 record is dropped loses its position.
        if isinstance(gt, dict) and gt.get("tiff_path"):
            tiffs[uid] = parse_path(gt["tiff_path"])[0]
        if qid == "FP2" and isinstance(gt, dict):
            flight[uid] = {k: gt.get(k) for k in ("agl_m", "lat_deg", "lon_deg")}
        if ans is None:
            dropped["answer not an option"] += 1
            continue
        if not isinstance(ts, dict) or any(k not in ts for k in TS_KEYS):
            dropped["temp_summary incomplete"] += 1
            continue
        if not isinstance(ap, (int, float)) or float(ap) <= 0:
            dropped["applicability missing or zero"] += 1
            continue
        rows.append({"qid": qid, "cat": rec["category"], "q": rec["question"], "opts": rec["options"],
                     "ans": ans, "ap": float(ap), "ts": {k: ts[k] for k in TS_KEYS}, "uid": uid,
                     "site": site, "unit": unit_name, "burn": burn, "image_id": rec["image_id"],
                     "rgb": rel, "thermal": parse_path(rec["thermal_path"])[0], "gt": gt,
                     "gt_method": gt.get("method") if isinstance(gt, dict) else None})

print("valid records: %d | dropped: %s" % (len(rows), dict(dropped)))
pool = [r for r in rows if r["site"].startswith(PUBLIC_BURN)] if PUBLIC_BURN else rows
print("pool after keeping burn %r: %d records over %d images and %d question ids"
      % (PUBLIC_BURN or "(all)", len(pool), len({r["uid"] for r in pool}), len({r["qid"] for r in pool})))

by_qid = defaultdict(list)
for r in pool:
    by_qid[r["qid"]].append(r)
qids = sorted(by_qid)

# The majority class per question id, computed on the released pool, is the strongest trivial strategy and
# travels with the item set so a reader can reproduce it.
majority = {q: Counter(r["ans"] for r in by_qid[q]).most_common(1)[0] for q in qids}
ts_rate = {}
for q, (_, fn) in TS_RULES.items():
    hits = sum(fn(r["ts"]) == r["ans"] for r in by_qid[q])
    ts_rate[q] = hits / len(by_qid[q])

items = []
for rank, qid in enumerate(qids):
    cand = sorted(by_qid[qid], key=lambda r: (r["unit"], r["burn"], r["image_id"]))
    n = len(cand)
    step, phase = n / PER_QUESTION, rank / len(qids)
    for i in sorted({int((j + phase) * step) % n for j in range(PER_QUESTION)}):
        r = cand[i]
        rule = TS_RULES.get(qid)
        items.append({
            "item_id": "wfvqa-%s-%s-%s-%s" % (r["unit"].replace(" ", ""), r["burn"].replace(" ", ""),
                                              r["image_id"], qid),
            "question_id": qid,
            "category": r["cat"],
            # what a model sees
            "question": r["q"],
            "options": r["opts"],
            "temp_summary": r["ts"],
            "image": {
                "image_uid": r["uid"],
                "site": r["site"], "unit": r["unit"], "burn_state": r["burn"], "image_id": r["image_id"],
                "flame3_rgb_corrected_fov": r["rgb"],
                "flame3_thermal_jpg": r["thermal"],
                "flame3_thermal_celsius_tiff": tiffs.get(r["uid"]),
                "agl_m": None, "lat_deg": None, "lon_deg": None,
            },
            # reference answer and where it comes from
            "answer": r["ans"],
            "answer_index": r["opts"].index(r["ans"]),
            "answer_provenance": PROVENANCE.get(qid, "mllm_verified"),
            "gt_method": r["gt_method"],
            "applicability_score": r["ap"],
            # naive baselines, carried per item so a scorer needs no second pass
            "baseline_majority": majority[qid][0],
            "baseline_majority_correct": majority[qid][0] == r["ans"],
            "temp_summary_rule": rule[0] if rule else None,
            "temp_summary_rule_answer": rule[1](r["ts"]) if rule else None,
            "temp_summary_rule_correct": (rule[1](r["ts"]) == r["ans"]) if rule else None,
            # whether this item can be answered with certainty before the imagery arrives
            "no_image_answerable": ("closed_form" if rule else
                                    "no_fire_branch" if qid in NO_FIRE and r["ts"]["max"] < 200 else None),
            # dating
            "image_collected_start": "2022-10-25",
            "image_collected_end": "2022-10-27",
            # The FLAME 3 paper gives October 25-27 2022 in its site table and again in its text; the
            # IEEE DataPort and Kaggle blurbs both write 10/25/23-10/27/23. The paper is taken as the source.
            "image_collected_note": "FLAME 3 paper says 2022; the DataPort and Kaggle pages say 2023",
            "images_public_since": "2024-12-22",
            "questions_public_since": "2026-04-15",
            "questions_last_modified": "2026-06-11",
            "source_images": "FLAME 3, IEEE DataPort, DOI 10.21227/w0mz-aq48",
            "source_questions": "WildFireVQA, huggingface.co/datasets/mobiiin/WildFire_VQA, arXiv 2604.20190",
        })

for it in items:
    it["image"].update(flight.get(it["image"]["image_uid"], {}))

(OUT / "items.jsonl").write_text("\n".join(json.dumps(r) for r in items) + "\n", encoding="utf-8")
(OUT / "source-prompt-template.txt").write_text(prompts[0], encoding="utf-8")

# ---- report ----------------------------------------------------------------------------------------
imgs = {i["image"]["image_uid"] for i in items}
print("\nitems: %d | question ids: %d | categories: %d | distinct images: %d of %d in the pool"
      % (len(items), len({i["question_id"] for i in items}), len({i["category"] for i in items}),
         len(imgs), len({r["uid"] for r in pool})))
print("items by category:", dict(Counter(i["category"] for i in items)))
print("items by unit and burn state:",
      dict(Counter("%s/%s" % (i["image"]["unit"], i["image"]["burn_state"]) for i in items)))
print("items by answer provenance:", dict(Counter(i["answer_provenance"] for i in items)))
print("applicability: 1.0 on %d items, below 1.0 on %d, minimum %.1f"
      % (sum(i["applicability_score"] == 1.0 for i in items),
         sum(i["applicability_score"] < 1.0 for i in items), min(i["applicability_score"] for i in items)))

rand = sum(1 / len(i["options"]) for i in items) / len(items)
maj = sum(i["baseline_majority_correct"] for i in items) / len(items)
solved = [q for q in TS_RULES if ts_rate[q] == 1.0]
mixed = sum((i["temp_summary_rule_correct"] if i["question_id"] in solved else i["baseline_majority_correct"])
            for i in items) / len(items)
non_deg = [i for i in items if majority[i["question_id"]][1] / len(by_qid[i["question_id"]]) < 0.9]
print("\nnaive baselines on the %d items" % len(items))
print("  A uniform random over the options            %.3f" % rand)
print("  B per-question-id majority class             %.3f" % maj)
print("  C B, with the temp_summary rule on %s  %.3f" % (",".join(sorted(solved)), mixed))
print("  B on the %d items outside the near-degenerate question ids  %.3f"
      % (len(non_deg), sum(i["baseline_majority_correct"] for i in non_deg) / len(non_deg)))

print("\nclosed-form temp_summary rules, checked on all %d pool records per question id:" % len(by_qid[qids[0]]))
for q in sorted(TS_RULES):
    print("  %-5s %.4f   %s" % (q, ts_rate[q], TS_RULES[q][0]))

cf = [i for i in items if i["no_image_answerable"] == "closed_form"]
nf = [i for i in items if i["no_image_answerable"] == "no_fire_branch"]
print("\nanswerable from temp_summary alone, with no image at all")
print("  closed form on %s: %d items, rule agrees with the reference answer on %d"
      % (",".join(sorted(TS_RULES)), len(cf), sum(i["temp_summary_rule_correct"] for i in cf)))
print("  the no-fire branch of %s (temp_summary.max < 200): %d items, right on %d"
      % (",".join(sorted(NO_FIRE)), len(nf), sum(i["answer"] == NO_FIRE[i["question_id"]] for i in nf)))
print("  total %d of %d items = %.3f, runnable before the FLAME 3 imagery arrives"
      % (len(cf) + len(nf), len(items), (len(cf) + len(nf)) / len(items)))
print("  in the full release this is %d of %d questions on the 4 closed-form ids, plus the no-fire branch"
      % (4 * 6097, 34 * 6097))

print("\nper question id: majority share on the pool, and share of the 12 sampled items it gets right")
for q in qids:
    share = majority[q][1] / len(by_qid[q])
    got = [i for i in items if i["question_id"] == q]
    flag = "  <-- near-degenerate" if share >= 0.9 else ""
    print("  %-5s %-28s |opts|=%d  pool majority %.3f (%s)  sampled %d/%d%s"
          % (q, next(i["category"] for i in got), len(got[0]["options"]), share, majority[q][0][:26],
             sum(i["baseline_majority_correct"] for i in got), len(got), flag))

print("\nwrote %s (%d bytes)" % (OUT / "items.jsonl", (OUT / "items.jsonl").stat().st_size))
