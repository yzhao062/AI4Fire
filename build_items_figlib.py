"""Build the version-1 smoke detection task from FIgLib, the HPWREN fire ignition image library.

Selection rule, published with the item set:
  - sequences dated 2026-01-01 or later, which is after the training cutoff of every model version 1 evaluates
  - per sequence, the frames nearest a fixed offset ladder, measured in seconds from the first visible plume
  - frames within 180 seconds of the plume are dropped, because the source itself calls that window ambiguous
  - a negative offset is the no-smoke label, a positive offset the smoke label
FIgLib carries no formal license and asks for a credit reference, so this task ships identifiers and this
downloader rather than the images.
"""
import io
import json
import pathlib
import re
from concurrent.futures import ThreadPoolExecutor

import httpx
from PIL import Image

BASE = "https://cdn.hpwren.ucsd.edu/HPWREN-FIgLib-Data"
S = pathlib.Path(__file__).parent
TASK = S / "task-figlib"
IMG = TASK / "images-1568"
LADDER = [-2400, -1800, -1200, -600, 300, 900, 1500, 2400]
MIN_ABS_OFFSET = 180
FROM_DATE = "20260101"
MAX_WIDTH = 1568  # the widest the vision endpoint keeps; 1024 cost 11 points of recall

TASK.mkdir(exist_ok=True)
IMG.mkdir(exist_ok=True)
client = httpx.Client(timeout=60, follow_redirects=True)

index = client.get(BASE + "/index.html").text
seqs = [m for m in re.findall(r"href=([0-9]{8}_[^/>]+)/index\.html", index)]
recent = [s for s in seqs if s[:8] >= FROM_DATE]
print("sequences listed: %d | dated %s or later: %d" % (len(seqs), FROM_DATE, len(recent)))


def frames_for(seq):
    try:
        page = client.get("%s/%s/index.html" % (BASE, seq)).text
    except Exception as exc:
        print("  listing failed for %s: %s" % (seq, exc))
        return []
    # A positive offset is written with a leading plus, as in 1788995756_+00500.jpg, so both signs are matched here.
    found = [(int(off), name) for name, off in
             ((m.group(0), m.group(1)) for m in re.finditer(r"\d{10}_([+-]?\d+)\.jpg", page))]
    found = [(o, n) for o, n in found if abs(o) >= MIN_ABS_OFFSET]
    if not found:
        return []
    picked, used = [], set()
    for target in LADDER:
        best = min(found, key=lambda x: abs(x[0] - target))
        if best[1] in used or abs(best[0] - target) > 600:
            continue
        used.add(best[1])
        picked.append((target, best[0], best[1]))
    return picked


def fetch(args):
    seq, target, offset, name = args
    out = IMG / ("%s__%s.jpg" % (seq, name.replace(".jpg", "")))
    if out.exists():
        return seq, target, offset, name, out.stat().st_size
    try:
        # A positive offset carries a literal plus in the file name, which the CDN answers 403 for unless it is
        # percent-encoded, so every positive frame fails silently without this.
        raw = client.get("%s/%s/%s" % (BASE, seq, name.replace("+", "%2B"))).content
        im = Image.open(io.BytesIO(raw))
        if im.width > MAX_WIDTH:
            im = im.resize((MAX_WIDTH, round(im.height * MAX_WIDTH / im.width)), Image.LANCZOS)
        im.convert("RGB").save(out, "JPEG", quality=85)
        return seq, target, offset, name, out.stat().st_size
    except Exception as exc:
        print("  download failed for %s/%s: %s" % (seq, name, exc))
        return None


jobs = []
for seq in recent:
    for target, offset, name in frames_for(seq):
        jobs.append((seq, target, offset, name))
print("frames to fetch:", len(jobs))

with ThreadPoolExecutor(max_workers=8) as ex:
    done = [r for r in ex.map(fetch, jobs) if r]

items = []
for seq, target, offset, name, size in done:
    date, fire, camera = seq.split("_", 2)
    items.append({
        "item_id": "figlib-%s-%s" % (seq, name.replace(".jpg", "")),
        "sequence": seq,
        "fire_name": fire,
        "camera": camera,
        "date": "%s-%s-%s" % (date[:4], date[4:6], date[6:]),
        "offset_seconds": offset,
        "ladder_target": target,
        "label": "smoke" if offset > 0 else "no smoke",
        "image": str((IMG / ("%s__%s.jpg" % (seq, name.replace(".jpg", "")))).relative_to(S)),
        "source_url": "%s/%s/%s" % (BASE, seq, name),
        "stored_bytes": size,
    })

(TASK / "items.jsonl").write_text("\n".join(json.dumps(r) for r in items) + "\n", encoding="utf-8")
pos = sum(1 for i in items if i["label"] == "smoke")
print("items: %d | smoke %d | no smoke %d | sequences %d | fires %d"
      % (len(items), pos, len(items) - pos, len({i["sequence"] for i in items}), len({i["fire_name"] for i in items})))
print("stored: %.1f MB" % (sum(i["stored_bytes"] for i in items) / 1e6))
print("dates: %s to %s" % (min(i["date"] for i in items), max(i["date"] for i in items)))
print("wrote", TASK / "items.jsonl")
