"""Resolution sensitivity check for FIgLib: does the 1024 pixel width hide the plume?

Re-fetches the positive frames at 1568 pixels, the widest the vision endpoint keeps, and asks the same question.
Only frames at least 25 minutes after the plume are used, because a model should find those if it can find any.
"""
import base64
import io
import json
import pathlib
import re
from concurrent.futures import ThreadPoolExecutor

import httpx
from PIL import Image

import gw

S = pathlib.Path(__file__).parent
TASK = S / "task-figlib"
BIG = TASK / "images-1568"
BIG.mkdir(exist_ok=True)
client = httpx.Client(timeout=90, follow_redirects=True)
MODEL = "claude-opus-5"

items = [json.loads(l) for l in (TASK / "items.jsonl").read_text(encoding="utf-8").splitlines()]
late = [i for i in items if i["offset_seconds"] >= 1500]
print("late positive frames:", len(late))


def big_copy(item):
    out = BIG / pathlib.Path(item["image"]).name
    if out.exists():
        return out
    url = item["source_url"].replace("+", "%2B")
    im = Image.open(io.BytesIO(client.get(url).content))
    if im.width > 1568:
        im = im.resize((1568, round(im.height * 1568 / im.width)), Image.LANCZOS)
    im.convert("RGB").save(out, "JPEG", quality=88)
    return out


with ThreadPoolExecutor(max_workers=6) as ex:
    paths = list(ex.map(big_copy, late))
print("fetched at 1568 px:", len(paths), "| mean KB:", round(sum(p.stat().st_size for p in paths) / len(paths) / 1024))

SYSTEM = ('You judge fixed wildland camera images. Answer with one JSON object and nothing else: '
          '{"smoke": true or false, "reasoning": "<one short sentence>"}.')
QUESTION = ("Does this frame show smoke from a wildland fire? Haze, fog, cloud, and dust are not smoke. "
            "Answer for this frame alone.")


def ask(path, key):
    data = base64.b64encode(path.read_bytes()).decode()
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + data}},
                                         {"type": "text", "text": QUESTION}]}]
    try:
        text, usage, _ = gw.chat(key, MODEL, msgs, max_tokens=200)
    except Exception as exc:
        return None, str(exc)[:120], {}
    m = re.search(r'"smoke"\s*:\s*(true|false)', text or "", re.I)
    return (m.group(1).lower() == "true") if m else None, text, usage


key = gw.load_key()
with ThreadPoolExecutor(max_workers=6) as ex:
    out = list(ex.map(lambda p: ask(p, key), paths))

hits = [o[0] for o in out if o[0] is not None]
prior = {json.loads(l)["item_id"]: json.loads(l).get("prediction")
         for l in (TASK / "responses-claude-opus-5-bare.jsonl").read_text(encoding="utf-8").splitlines()}
prior_hits = [prior[i["item_id"]] for i in late if prior.get(i["item_id"]) is not None]
print("\nrecall at 1024 px: %.3f (n=%d)" % (sum(1 for h in prior_hits if h) / len(prior_hits), len(prior_hits)))
print("recall at 1568 px: %.3f (n=%d)" % (sum(1 for h in hits if h) / len(hits), len(hits)))
(TASK / "resolution-check.json").write_text(json.dumps(
    {"model": MODEL, "frames": [i["item_id"] for i in late],
     "recall_1024": sum(1 for h in prior_hits if h) / len(prior_hits),
     "recall_1568": sum(1 for h in hits if h) / len(hits),
     "answers_1568": [{"item": i["item_id"], "smoke": o[0], "raw": (o[1] or "")[:300]} for i, o in zip(late, out)]},
    indent=1), encoding="utf-8")
print("wrote", TASK / "resolution-check.json")
