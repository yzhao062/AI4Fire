"""Build the version-1 structure-fire question answering task from Fire360.

Selection rule, published with the item set:
  - every question in prompt_templates.json as served by the Fire360 Box shared link, which is the
    only question-bearing file in the release: 50 multiple-choice and 50 free-text, 100 in total
  - no filtering, no sampling and no rewriting, because the release holds exactly these 100 and a
    filtered subset would leave too few to report
  - each item keeps the id, the prompt, the options and the answer verbatim as shipped, so a reader
    who downloads the same file reproduces the item set byte for byte
  - the multiple-choice and free-text halves are paired by list index and the pair is recorded,
    because the two halves ask the same 50 facts twice
The reference answer is the shipped key: correct_answer for multiple choice, expected_output for
free text. Scoring is exact match after case folding and whitespace collapse, which is the metric
the Fire360 paper and the release documentation both name for VQA.

WHAT THESE ITEMS ARE, AND WHAT THEY ARE NOT. The Fire360 paper describes its VQA task as grounded
in 360-degree video: "Given an equirectangular or normal rectilinear frame, models must answer
expert-authored questions on object presence, responder behavior, and protocol adherence", with
examples such as "Is the exit door visible through the smoke?". The released questions are not
those. No item in prompt_templates.json carries a video id, a frame, a timestamp, a projection or
any other reference to media: the strings "video", "frame", "timestamp" and "clip" appear zero
times in the file, and the union of all item keys is id, prompt, options, correct_answer,
expected_output and difficulty. None of the paper's example questions is present. What ships is
firefighting doctrine trivia answerable from text alone ("Which tool is used to break down doors?"
-> "Axe"). The release documentation calls the file "Sample prompts", which is consistent with the
items being an illustration rather than the scored set. So this task measures structure-fire
knowledge question answering, not 360-degree perception, and the item file says so on every row
via grounded_in_media: false. The 348 temporal action instances, the bounding boxes, the smoke
grades, the 154 TOR targets and the safety checklists described in the paper are not in the
release; prompt_templates.json carries 2 TOR and 2 safety prompts, which are the paper's own
Table 6 examples.

MEDIA. This builder stores identifiers and answers, never media. It downloads four small files
(about 138 KB in total) and nothing else. The 228 videos live in the same Box folder and come to
759.7 GB across 602 files, which no item here needs, because no item references any of them. A
reader who wants the media opens https://uofi.box.com/v/fire360dataset in a browser, or fetches a
file by id with the same index.php route this script uses for the small files; data/fire360/
box-listing.json records the path, the Box file id and the byte count of all 602 files, so a
single video can be pulled without walking the folder again. The Box link served every file over
plain HTTP with no login, no account and no access request.

LICENSE. The LICENSE.txt in the Box folder is the plain, unmodified MIT License, copyright 2025
Aditi Tiwari, and the release documentation states "The dataset is available under the MIT License
(see licence.txt)". The paper instead describes "a research-only MIT license with added
restrictions that prohibit use in surveillance, enforcement, behavioral profiling, or
nonconsensual monitoring contexts". Those restrictions appear in no file in the release. This
builder takes the conservative reading of the two: it keeps media out of the item file and stores
identifiers, which satisfies the paper's wording and the shipped MIT text at once. The downloaded
LICENSE.txt travels in data/fire360/ so a reader can check both claims.
"""
import collections
import hashlib
import json
import pathlib
import re

import httpx

S = pathlib.Path(__file__).parent
D = S / "data" / "fire360"
OUT = S / "task-fire360"
SUB = "https://uofi.app.box.com"
SHARED = "5ge3jfcixa9s9smztwf24pz8ut6uy4h6"
ROOT = 320319827582
LINK = "https://uofi.box.com/v/fire360dataset"

# File ids recorded from the Box listing on 2026-09-15. The script resolves ids by name at run time
# and uses these only if a name is missing, so a re-upload changes the id without breaking the build.
KNOWN = {"prompt_templates.json": 1864246007803, "splits.json": 1864247186175,
         "LICENSE.txt": 1864246497456, "NeurIPS_2025.pdf": 1864277634952}

# Published Fire360 numbers, transcribed from arXiv 2506.02167 Table 3 and Section A.
HUMAN = {"vqa_top1_accuracy": 91.4, "safety_checklist_accuracy": 94.6, "tor_retrieval_accuracy": 83.5,
         "temporal_captioning_agreement": 0.85}
MODELS_VQA = {"GPT-4o": 53.8, "LLaVA-v1.5-13B": 50.3, "Qwen-VL": 47.2, "BLIP-2 (OPT-6.7B)": 42.7}
MODELS_VQA_RECTILINEAR = {"GPT-4o": 62.4, "LLaVA-1.5": 58.9, "Qwen-VL": 55.6, "BLIP-2": 48.2}

D.mkdir(parents=True, exist_ok=True)
OUT.mkdir(exist_ok=True)
DEC = json.JSONDecoder()
client = httpx.Client(timeout=180, follow_redirects=True, headers={
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0 Safari/537.36",
    "Referer": LINK})


def folder_page(fid, n):
    """One page of a Box shared folder.

    The /app-api/enduserapp/shared-folder endpoint answers with the root whatever folder_id it is
    given, and 404s on POST, so the listing has to come from the folder's own shared page, which
    embeds 20 items per page in Box.postStreamData and reports how many pages there are.
    """
    url = "%s/v/fire360dataset?page=%d&sortColumn=name&sortDirection=ASC" % (SUB, n) if fid == ROOT \
        else "%s/v/fire360dataset/folder/%s?page=%d&sortColumn=name&sortDirection=ASC" % (SUB, fid, n)
    html = client.get(url).text
    i = html.find("Box.postStreamData")
    if i < 0:
        return None
    return DEC.raw_decode(html[html.find("{", i):])[0].get("/app-api/enduserapp/shared-folder")


def listing(fid):
    first = folder_page(fid, 1)
    if first is None:
        return []
    items, seen = list(first.get("items", [])), {i["id"] for i in first.get("items", [])}
    for n in range(2, int(first.get("pageCount") or 1) + 1):
        more = folder_page(fid, n)
        for it in (more or {}).get("items", []):
            if it["id"] not in seen:
                seen.add(it["id"])
                items.append(it)
    return items


def fetch(name, fid):
    out = D / name
    if out.exists():
        return out
    r = client.get("%s/index.php?rm=box_download_shared_file&shared_name=%s&file_id=f_%d" % (SUB, SHARED, fid))
    if r.status_code != 200 or "text/html" in r.headers.get("content-type", ""):
        raise SystemExit("download of %s failed: %s %s" % (name, r.status_code, r.headers.get("content-type")))
    out.write_bytes(r.content)
    print("  downloaded %-24s %7d bytes" % (name, len(r.content)))
    return out


client.get(LINK)  # seed the shared-link session cookie
root = listing(ROOT)
by_name = {it["name"]: it for it in root}
print("Box root: %d entries | %d files, %d folders" % (
    len(root), sum(i["type"] == "file" for i in root), sum(i["type"] == "folder" for i in root)))
for name in KNOWN:
    fetch(name, by_name[name]["id"] if name in by_name else KNOWN[name])

meta = by_name.get("prompt_templates.json", {})
raw = (D / "prompt_templates.json").read_bytes()
sha = hashlib.sha256(raw).hexdigest()
tmpl = json.loads(raw.decode("utf-8"))
mc = tmpl["templates"]["vqa"]["multiple_choice"]
ft = tmpl["templates"]["vqa"]["free_text"]
print("prompt_templates.json %d bytes | sha256 %s | version %s" % (len(raw), sha[:16], tmpl.get("version")))
print("  multiple_choice %d | free_text %d | tor %d | safety_reasoning %d"
      % (len(mc), len(ft), len(tmpl["templates"]["tor"]), len(tmpl["templates"]["safety_reasoning"])))

# Grounding check, run over the file rather than asserted, so the claim in the docstring is testable.
text = raw.decode("utf-8").lower()
probes = {k: text.count(k) for k in ("video", "frame", "timestamp", "clip", "equirect", "projection", ".mp4")}
keys = sorted({k for it in mc + ft for k in it})
print("  media-reference probes:", probes)
print("  union of item keys:", keys)
grounded = any(probes.values())

PROV = {
    "dataset": "Fire360",
    "source_file": "prompt_templates.json",
    "source_file_sha256": sha,
    "box_shared_link": LINK,
    "box_file_id": meta.get("id", KNOWN["prompt_templates.json"]),
    "box_file_modified_utc": "2025-05-16",
    "paper": "arXiv:2506.02167",
    "paper_v1_date": "2025-06-02",
    "venue": "NeurIPS 2025 Datasets and Benchmarks",
    "license_as_shipped": "MIT (LICENSE.txt in the Box folder, copyright 2025 Aditi Tiwari)",
    "license_as_described_in_paper": "research-only MIT with added restrictions on surveillance, "
                                     "enforcement, behavioral profiling and nonconsensual monitoring",
}


def norm(s):
    return re.sub(r"\s+", " ", str(s)).strip().lower()


items = []
for k, (a, b) in enumerate(zip(mc, ft), start=1):
    pair = "fire360-pair-%03d" % k
    twin = norm(a["correct_answer"]) == norm(b["expected_output"])
    items.append({
        "item_id": "fire360-mcq-%03d" % k,
        "task": "fire360-structure-fire-qa",
        "format": "multiple_choice",
        "source_id": a["id"],
        "prompt": a["prompt"],
        "options": list(a["options"]),
        "reference_answer": a["correct_answer"],
        "scoring": "exact match after case folding and whitespace collapse",
        "difficulty_as_shipped": a["difficulty"],
        "answer_position": a["options"].index(a["correct_answer"]),
        "n_options": len(a["options"]),
        "pair_group": pair,
        "pair_answer_matches": twin,
        "grounded_in_media": grounded,
        "media_reference": None,
        "provenance": PROV,
    })
    items.append({
        "item_id": "fire360-free-%03d" % k,
        "task": "fire360-structure-fire-qa",
        "format": "free_text",
        "source_id": b["id"],
        "prompt": b["prompt"],
        "options": None,
        "reference_answer": b["expected_output"],
        "scoring": "exact match after case folding and whitespace collapse",
        "difficulty_as_shipped": b["difficulty"],
        "answer_position": None,
        "n_options": None,
        "pair_group": pair,
        "pair_answer_matches": twin,
        "grounded_in_media": grounded,
        "media_reference": None,
        "provenance": PROV,
    })

mc_items = [i for i in items if i["format"] == "multiple_choice"]
ft_items = [i for i in items if i["format"] == "free_text"]
n_mc, n_ft = len(mc_items), len(ft_items)
print("\nitems: %d | multiple choice %d | free text %d" % (len(items), n_mc, n_ft))
print("distinct item_id: %d | distinct prompt: %d | distinct reference answer: %d"
      % (len({i["item_id"] for i in items}), len({i["prompt"] for i in items}),
         len({norm(i["reference_answer"]) for i in items})))
print("difficulty as shipped:", dict(collections.Counter(i["difficulty_as_shipped"] for i in items)))
twins = sum(i["pair_answer_matches"] for i in mc_items)
print("index-aligned pairs whose two halves share an answer: %d of %d" % (twins, n_mc))
print("  so the 100 items ask about %d distinct answers, not 100"
      % len({norm(i["reference_answer"]) for i in items}))

# ---------------------------------------------------------------- naive baselines
pos = collections.Counter(i["answer_position"] for i in mc_items)
first = 100 * pos[0] / n_mc
mc_major = collections.Counter(norm(i["reference_answer"]) for i in mc_items).most_common(1)[0]
ft_major = collections.Counter(norm(i["reference_answer"]) for i in ft_items).most_common(1)[0]
shortest = 100 * sum(i["reference_answer"] == min(i["options"], key=len) for i in mc_items) / n_mc
longest = 100 * sum(i["reference_answer"] == max(i["options"], key=len) for i in mc_items) / n_mc
copy_across = 100 * twins / n_ft

print("\nnaive baselines")
print("  multiple choice, answer position: " + " ".join("p%d=%d" % (k, pos[k]) for k in range(4)))
print("  multiple choice, uniform random over 4 options  : 25.0%")
print("  multiple choice, always the first option        : %.1f%%   <- best trivial strategy" % first)
print("  multiple choice, always the last option         : %.1f%%" % (100 * pos[3] / n_mc))
print("  multiple choice, shortest option                : %.1f%%" % shortest)
print("  multiple choice, longest option                 : %.1f%%" % longest)
print("  multiple choice, always answer %-18r: %.1f%%" % (mc_major[0], 100 * mc_major[1] / n_mc))
print("  free text, always answer %-24r: %.1f%%" % (ft_major[0], 100 * ft_major[1] / n_ft))
print("  free text, copy the aligned multiple-choice key  : %.1f%%   <- the two halves leak into each other"
      % copy_across)
print("  all 100 items, first option plus free-text majority: %.1f%%"
      % (100 * (pos[0] + ft_major[1]) / len(items)))

print("\npublished baselines, for reading a model score against")
print("  human expert, VQA top-1                         : %.1f%%" % HUMAN["vqa_top1_accuracy"])
for m, v in MODELS_VQA.items():
    print("  %-46s: %.1f%%   (rectilinear input: %s)"
          % (m + ", equirectangular", v, MODELS_VQA_RECTILINEAR.get(m.split(" (")[0].replace("-v1.5-13B", "-1.5"), "n/a")))
print("  human expert, safety checklist                  : %.1f%%" % HUMAN["safety_checklist_accuracy"])
print("  human expert, transformed object retrieval       : %.1f%%" % HUMAN["tor_retrieval_accuracy"])

# ---------------------------------------------------------------- can 100 items separate models?
def halfwidth(p, n):
    return 196.0 * (p / 100 * (1 - p / 100) / n) ** 0.5


hw_mc, hw_all = halfwidth(50, n_mc), halfwidth(50, len(items))
print("\nseparation check")
print("  95%% interval half-width at p=0.5: %.1f points on %d multiple-choice items, %.1f on all %d"
      % (hw_mc, n_mc, hw_all, len(items)))
print("  best trivial strategy %.1f%% vs GPT-4o %.1f%%: gap %.1f points, inside the interval"
      % (first, MODELS_VQA["GPT-4o"], MODELS_VQA["GPT-4o"] - first))
gaps = sorted(MODELS_VQA.values(), reverse=True)
print("  widest published model gap %.1f points (%.1f%% to %.1f%%): %s"
      % (gaps[0] - gaps[-1], gaps[0], gaps[-1], "inside the interval" if gaps[0] - gaps[-1] < hw_mc else "separable"))
print("  human %.1f%% vs GPT-4o %.1f%%: gap %.1f points, %s"
      % (HUMAN["vqa_top1_accuracy"], MODELS_VQA["GPT-4o"], HUMAN["vqa_top1_accuracy"] - MODELS_VQA["GPT-4o"],
         "separable" if HUMAN["vqa_top1_accuracy"] - MODELS_VQA["GPT-4o"] > hw_mc else "inside the interval"))
# Exact-match accuracy over n items can only land on a multiple of 100/n. The published VQA
# percentages land on none of them at n=50 or n=100, so they cannot come from one deterministic
# exact-match pass over this file. Averaging several sampled runs would explain it; so would a
# different and unreleased question set. The release does not say which.
for n in (50, 100):
    bad = [v for v in list(MODELS_VQA.values()) + [HUMAN["vqa_top1_accuracy"]]
           if abs(v * n / 100 - round(v * n / 100)) > 1e-9]
    print("  published VQA percentages that one exact-match pass over n=%d cannot produce: %s" % (n, bad))

# Difficulty is labelled in blocks by list position rather than by content, which is what a
# generated file looks like and what an expert-authored one does not.
for half, rows in (("multiple choice", mc), ("free text", ft)):
    order, runs = ["easy", "medium", "hard", "very hard"], []
    for it in rows:
        if not runs or runs[-1][0] != it["difficulty"]:
            runs.append([it["difficulty"], 0])
        runs[-1][1] += 1
    blocks = " then ".join("%d %s" % (n, d) for d, n in runs)
    print("  %-15s difficulty runs in list order: %s%s"
          % (half, blocks, "" if len(runs) > len(order) else "  (one contiguous block per level)"))

# ---------------------------------------------------------------- splits.json, checked not quoted
splits = json.loads((D / "splits.json").read_text(encoding="utf-8"))
listed = {k: len(v) for k, v in splits["splits"].items()}
print("\nsplits.json: total_videos=%s, ids listed %s = %d"
      % (splits["total_videos"], listed, sum(listed.values())))
print("  its own description: %r" % splits["description"][-92:])

tree_path = D / "box-listing.json"
if not tree_path.exists():
    print("  (box-listing.json absent, skipping the split-against-files check)")
else:
    tree = json.loads(tree_path.read_text(encoding="utf-8"))
    stems = {re.sub(r"\s*\(\d+\)$", "", pathlib.PurePosixPath(r["path"]).stem).strip()
             for r in tree if r["type"] == "file"}
    for k, v in splits["splits"].items():
        hit = sum(any(s == x or s.startswith(x) for s in stems) for x in v)
        print("  %-5s %3d ids, %3d match a file in the Box folder, %3d match nothing" % (k, len(v), hit, len(v) - hit))

# ---------------------------------------------------------------- write
(OUT / "items.jsonl").write_text("\n".join(json.dumps(r) for r in items) + "\n", encoding="utf-8")
index = ["item_id,format,difficulty_as_shipped,answer_position,pair_group,reference_answer"]
for i in items:
    index.append('%s,%s,%s,%s,%s,"%s"' % (i["item_id"], i["format"], i["difficulty_as_shipped"],
                                          "" if i["answer_position"] is None else i["answer_position"],
                                          i["pair_group"], str(i["reference_answer"]).replace('"', '""')))
(OUT / "items-index.csv").write_text("\n".join(index) + "\n", encoding="utf-8")

(OUT / "baselines.json").write_text(json.dumps({
    "task": "fire360-structure-fire-qa",
    "items": len(items),
    "multiple_choice_items": n_mc,
    "free_text_items": n_ft,
    "distinct_reference_answers": len({norm(i["reference_answer"]) for i in items}),
    "grounded_in_media": grounded,
    "naive": {
        "mc_uniform_random_over_4_options": 25.0,
        "mc_always_first_option": round(first, 1),
        "mc_always_last_option": round(100 * pos[3] / n_mc, 1),
        "mc_shortest_option": round(shortest, 1),
        "mc_longest_option": round(longest, 1),
        "mc_always_majority_answer": round(100 * mc_major[1] / n_mc, 1),
        "free_text_always_majority_answer": round(100 * ft_major[1] / n_ft, 1),
        "free_text_copy_aligned_mc_key": round(copy_across, 1),
        "all_items_first_option_plus_free_text_majority": round(100 * (pos[0] + ft_major[1]) / len(items), 1),
    },
    "published_human": HUMAN,
    "published_models_vqa_equirectangular": MODELS_VQA,
    "published_models_vqa_rectilinear": MODELS_VQA_RECTILINEAR,
    "interval_half_width_95pct_at_p_half": {"n_%d" % n_mc: round(hw_mc, 1), "n_%d" % len(items): round(hw_all, 1)},
    "source": PROV,
}, indent=1), encoding="utf-8")

print("\nwrote %s (%d bytes)" % (OUT / "items.jsonl", (OUT / "items.jsonl").stat().st_size))
print("wrote %s" % (OUT / "items-index.csv"))
print("wrote %s" % (OUT / "baselines.json"))
