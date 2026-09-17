"""Run the FIgLib smoke detection task: one frame per call, bare or with an earlier frame from the same camera.

    python run_figlib.py --dry-run
    python run_figlib.py --models claude-opus-5 --conditions bare grounded

Bare asks for a judgment on one frame. Grounded prepends the earliest frame of the same sequence, which is the same
camera view before the plume, so the model can compare rather than judge an unfamiliar scene cold. The reference
frame itself is dropped from the grounded arm, since it cannot serve as its own reference.
"""
import argparse
import base64
import collections
import json
import pathlib
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import gw

S = pathlib.Path(__file__).parent
MAX_OUT = 1536  # see the note in the module docstring on why this is not a few hundred
TASK = S / "task-figlib"
SYSTEM = ('You judge fixed wildland camera images. Answer with one JSON object and nothing else: '
          '{"smoke": true or false, "reasoning": "<one short sentence>"}.')
QUESTION = ("Does this frame show smoke from a wildland fire? Haze, fog, cloud, and dust are not smoke. "
            "Answer for this frame alone.")


def b64(path):
    return base64.b64encode((S / path).read_bytes()).decode()


def messages(item, reference=None):
    content = []
    if reference:
        content += [{"type": "text", "text": "Reference frame: the same camera earlier in the day, before any plume."},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64(reference)}},
                    {"type": "text", "text": "Frame to judge:"}]
    content += [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64(item["image"])}},
                {"type": "text", "text": QUESTION}]
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]


def parse(text):
    if not text:
        return None
    m = re.search(r'"smoke"\s*:\s*(true|false)', text, re.I)
    if m:
        return m.group(1).lower() == "true"
    low = text.lower()
    if "no smoke" in low or low.strip().startswith("no"):
        return False
    if "smoke" in low:
        return True
    return None


def bucket(offset):
    a = abs(offset)
    return ("%s %s" % ("after" if offset > 0 else "before",
                       "0 to 10 min" if a <= 600 else "10 to 25 min" if a <= 1500 else "25 min or more"))


def score(rows, label):
    ok = [r for r in rows if r["prediction"] is not None]
    if not ok:
        return {"run": label, "items": len(rows), "parsed": 0}
    y = np.array([r["label"] == "smoke" for r in ok])
    p = np.array([bool(r["prediction"]) for r in ok])
    by_bucket = {}
    for b in sorted({r["bucket"] for r in ok}):
        sel = [i for i, r in enumerate(ok) if r["bucket"] == b]
        by_bucket[b] = round(float(np.mean(p[sel] == y[sel])), 3)
    # Time to detection: per sequence, the smallest positive offset the model called smoke.
    seqs, detected = collections.defaultdict(list), []
    for r in ok:
        if r["offset_seconds"] > 0:
            seqs[r["sequence"]].append((r["offset_seconds"], bool(r["prediction"])))
    for seq, pairs in seqs.items():
        hits = [o for o, pred in sorted(pairs) if pred]
        detected.append(hits[0] if hits else None)
    found = [d for d in detected if d is not None]
    return {"run": label, "items": len(rows), "parsed": len(ok),
            "accuracy": float(np.mean(p == y)),
            "recall_on_smoke": float(np.mean(p[y])) if y.any() else None,
            "false_positive_rate": float(np.mean(p[~y])) if (~y).any() else None,
            "accuracy_by_bucket": by_bucket,
            "sequences_detected": "%d of %d" % (len(found), len(detected)),
            "median_detection_offset_seconds": float(np.median(found)) if found else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["claude-opus-5"])
    ap.add_argument("--conditions", nargs="*", default=["bare", "grounded"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    items = [json.loads(l) for l in (TASK / "items.jsonl").read_text(encoding="utf-8").splitlines()]
    for it in items:
        it["bucket"] = bucket(it["offset_seconds"])
    reference = {}
    for it in items:
        cur = reference.get(it["sequence"])
        if cur is None or it["offset_seconds"] < cur["offset_seconds"]:
            reference[it["sequence"]] = it
    if args.limit:
        items = items[:args.limit]
    print("items: %d | smoke %d | sequences %d | reference frames %d"
          % (len(items), sum(1 for i in items if i["label"] == "smoke"),
             len({i["sequence"] for i in items}), len(reference)))
    print("buckets:", json.dumps(collections.Counter(i["bucket"] for i in items), indent=None))

    if args.dry_run:
        it = items[0]
        m = messages(it, reference[it["sequence"]]["image"] if reference[it["sequence"]]["item_id"] != it["item_id"] else None)
        print("\nsample user content parts:", [p.get("type") for p in m[1]["content"]])
        print("prompt text:", [p["text"] for p in m[1]["content"] if p["type"] == "text"])
        print("image bytes (base64 chars):", sum(len(p["image_url"]["url"]) for p in m[1]["content"] if p["type"] == "image_url"))
        print("first item:", it["item_id"], "| label", it["label"], "| offset", it["offset_seconds"])
        return

    key = gw.load_key()
    summaries = []
    for model in args.models:
        for cond in args.conditions:
            pool = [it for it in items
                    if cond == "bare" or reference[it["sequence"]]["item_id"] != it["item_id"]]

            def one(it):
                ref = reference[it["sequence"]]["image"] if cond == "grounded" else None
                base = {"item_id": it["item_id"], "sequence": it["sequence"], "label": it["label"],
                        "offset_seconds": it["offset_seconds"], "bucket": it["bucket"]}
                try:
                    text, usage, served = gw.call(key, model, messages(it, ref), max_tokens=MAX_OUT)
                except Exception as exc:
                    return dict(base, error=str(exc)[:200], prediction=None)
                return dict(base, raw=text, usage=usage, served_model=served, prediction=parse(text))

            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                rows = list(ex.map(one, pool))
            safe = re.sub(r"[^A-Za-z0-9._-]", "_", model)
            (TASK / ("responses-%s-%s.jsonl" % (safe, cond))).write_text(
                "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
            s = score(rows, "%s/%s" % (model, cond))
            s["errors"] = sum(1 for r in rows if r.get("error"))
            s["tokens_in"] = sum((r.get("usage") or {}).get("prompt_tokens", 0) for r in rows)
            s["tokens_out"] = sum((r.get("usage") or {}).get("completion_tokens", 0) for r in rows)
            summaries.append(s)
            print(json.dumps(s))

    path = TASK / "scores.json"
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    fresh = {s["run"] for s in summaries}
    path.write_text(json.dumps([s for s in old if s["run"] not in fresh] + summaries, indent=1), encoding="utf-8")
    print("\nwrote", path)


if __name__ == "__main__":
    main()
