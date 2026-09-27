"""Smoke detection rerun with the time cues removed: masked capture times, gap-matched references, neutral wording.

    python run_figlib_masked.py build
    python run_figlib_masked.py run --dry-run
    python run_figlib_masked.py run --models claude-opus-5 gemini-3.1-pro

Version 1 (run_figlib.py) paired every target with its sequence's earliest frame, about 40 minutes before the plume.
That reference precedes clear targets by 9 to 30 minutes and smoke targets by 44 to 80, so the gap alone separates
the labels; every frame also prints its capture time in a top strip, and the reference was introduced as "before any
plume". This rerun removes all three:
  - targets are the six ladder positions that have a partner 10, 20, or 30 minutes earlier on the pre-plume side:
    clear frames at -1800, -1200, and -600 s and smoke frames at +300, +900, and +1500 s, 84 of each; the -2400 s
    frames (version 1's references) and the +2400 s frames are not targets
  - each target's reference is the frame nearest 10, 20, or 30 minutes before it, in the order listed above, taken
    from the full FIgLib listing and always at least 180 s before the first visible plume, so every reference is
    clear and each label meets each gap once per sequence
  - both arms see every frame with its top 32 rows blacked out; the printed caption (site, date, clock time, and
    Unix epoch), descenders included, ends by row 24 in all 280 frames, and build checks every frame again
  - the reference is introduced as "an earlier frame from the same camera", which says nothing about smoke
Outputs go to task-figlib-masked/ and never touch the version-1 files.
"""
import argparse
import base64
import collections
import io
import json
import os
import pathlib
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import httpx
import numpy as np
from PIL import Image

import gw
from run_figlib import MAX_OUT, QUESTION, SYSTEM, parse

S = pathlib.Path(__file__).parent
V1 = S / "task-figlib"
OUT = S / "task-figlib-masked"
BASE = "https://cdn.hpwren.ucsd.edu/HPWREN-FIgLib-Data"
GAP = {-1800: 600, -1200: 1200, -600: 1800, 300: 600, 900: 1200, 1500: 1800}  # ladder target -> seconds back
MIN_PRE = -180  # FIgLib calls the three minutes around the plume ambiguous
MAX_WIDTH = 1568  # as build_items_figlib.py
MASK_ROWS = 32
REFERENCE_TEXT = "Reference frame: an earlier frame from the same camera."
SEED = 20260927
RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}


def yellow_rows(im, lo=0, hi=None):
    """Rows holding the printed caption: saturated yellow text, the colour HPWREN overlays on every frame."""
    a = np.asarray(im.convert("RGB")).astype(int)[lo:hi]
    y = (a[..., 0] - a[..., 2] > 60) & (a[..., 1] - a[..., 2] > 60) & (a[..., 0] > 120) & (a[..., 1] > 120)
    return [lo + int(r) for r in np.where(y.sum(axis=1) >= 2)[0]]


def mask(src, dst):
    im = Image.open(src).convert("RGB")
    a = np.asarray(im).copy()
    a[:MASK_ROWS] = 0
    Image.fromarray(a).save(dst, "JPEG", quality=95)
    return im


def listing(client, seq):
    page = client.get("%s/%s/index.html" % (BASE, seq)).text
    return sorted({(int(m.group(1)), m.group(0)) for m in re.finditer(r"\d{10}_([+-]?\d+)\.jpg", page)})


def fetch(client, seq, name, out):
    if out.exists():
        return
    raw = client.get("%s/%s/%s" % (BASE, seq, name.replace("+", "%2B")))
    raw.raise_for_status()
    im = Image.open(io.BytesIO(raw.content))
    if im.width > MAX_WIDTH:
        im = im.resize((MAX_WIDTH, round(im.height * MAX_WIDTH / im.width)), Image.LANCZOS)
    im.convert("RGB").save(out, "JPEG", quality=85)


def auroc(score, label):
    """Probability that a random smoke item has a larger score than a random clear one, ties counted half."""
    pos = [s for s, l in zip(score, label) if l]
    neg = [s for s, l in zip(score, label) if not l]
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def build():
    (OUT / "reference-1568").mkdir(parents=True, exist_ok=True)
    (OUT / "masked-1568").mkdir(exist_ok=True)
    v1 = [json.loads(l) for l in (V1 / "items.jsonl").read_text(encoding="utf-8").splitlines()]
    targets = [it for it in v1 if it["ladder_target"] in GAP]
    client = httpx.Client(timeout=60, follow_redirects=True)
    items, problems = [], []
    for seq in sorted({it["sequence"] for it in targets}):
        frames = [(o, n) for o, n in listing(client, seq) if o <= MIN_PRE]
        for it in sorted((t for t in targets if t["sequence"] == seq), key=lambda t: t["offset_seconds"]):
            want = it["offset_seconds"] - GAP[it["ladder_target"]]
            ref_off, ref_name = min(frames, key=lambda f: (abs(f[0] - want), f[0]))
            raw_ref = OUT / "reference-1568" / ("%s__%s" % (seq, ref_name))
            fetch(client, seq, ref_name, raw_ref)
            src_target = S / it["image"].replace("\\", "/")
            m_target = OUT / "masked-1568" / src_target.name
            m_ref = OUT / "masked-1568" / raw_ref.name
            im_t, im_r = mask(src_target, m_target), mask(raw_ref, m_ref)
            for label, im, path in (("target", im_t, src_target), ("reference", im_r, raw_ref)):
                rows = yellow_rows(im, 0, 80)
                if im.size != (1568, 1045) or (rows and max(rows) >= MASK_ROWS):
                    problems.append("%s %s: size %s, caption rows %s" % (label, path.name, im.size, rows[-3:]))
            for path in (m_target, m_ref):
                left = yellow_rows(Image.open(path), MASK_ROWS, 80)
                if left:
                    problems.append("masked %s keeps caption-coloured rows %s" % (path.name, left[:5]))
            gap = it["offset_seconds"] - ref_off
            items.append({
                "item_id": it["item_id"], "sequence": seq, "fire_name": it["fire_name"], "camera": it["camera"],
                "date": it["date"], "offset_seconds": it["offset_seconds"], "ladder_target": it["ladder_target"],
                "label": it["label"], "image": str(m_target.relative_to(S)), "reference_image": str(m_ref.relative_to(S)),
                "reference_offset_seconds": ref_off, "gap_seconds": gap, "nominal_gap_seconds": GAP[it["ladder_target"]],
                "source_image": it["image"].replace("\\", "/"), "reference_source_url": "%s/%s/%s" % (BASE, seq, ref_name)})
            if abs(gap - GAP[it["ladder_target"]]) > 90:
                problems.append("%s: gap %d s against nominal %d" % (it["item_id"], gap, GAP[it["ladder_target"]]))
    (OUT / "items.jsonl").write_text("\n".join(json.dumps(r) for r in items) + "\n", encoding="utf-8")
    smoke = [r["label"] == "smoke" for r in items]
    print("items: %d | smoke %d | clear %d | sequences %d | fires %d | references %d"
          % (len(items), sum(smoke), len(items) - sum(smoke), len({r["sequence"] for r in items}),
             len({r["fire_name"] for r in items}), len({r["reference_image"] for r in items})))
    for g in (600, 1200, 1800):
        for lab in ("no smoke", "smoke"):
            v = [r["gap_seconds"] for r in items if r["nominal_gap_seconds"] == g and r["label"] == lab]
            print("nominal gap %4d s | %-8s | n %d | min %d | median %.0f | max %d" % (g, lab, len(v), min(v), np.median(v), max(v)))
    print("reference offsets: clear targets %d to %d s, smoke targets %d to %d s" % (
        min(r["reference_offset_seconds"] for r in items if r["label"] != "smoke"),
        max(r["reference_offset_seconds"] for r in items if r["label"] != "smoke"),
        min(r["reference_offset_seconds"] for r in items if r["label"] == "smoke"),
        max(r["reference_offset_seconds"] for r in items if r["label"] == "smoke")))
    print("AUROC of the gap for the smoke label: %.3f (0.5 is no separation)" % auroc([r["gap_seconds"] for r in items], smoke))
    print("problems: %d" % len(problems))
    for p in problems:
        print("  " + p)
    print("wrote", OUT / "items.jsonl")


def b64(rel):
    return base64.b64encode((S / rel).read_bytes()).decode()


def messages(item, cond):
    content = []
    if cond == "grounded":
        content += [{"type": "text", "text": REFERENCE_TEXT},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64(item["reference_image"])}},
                    {"type": "text", "text": "Frame to judge:"}]
    content += [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64(item["image"])}},
                {"type": "text", "text": QUESTION}]
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]


def scrub(text):
    """Error strings can quote the request URL; keep the gateway address out of every stored file."""
    for v in {os.environ.get("NAIRR_GATEWAY_URL") or "", (os.environ.get("NAIRR_GATEWAY_URL") or "").rstrip("/")}:
        if v:
            text = text.replace(v, "<gateway>")
    return text


def retriable(exc):
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in RETRY_STATUS
    if isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
        return True
    resp = getattr(exc, "response", None)
    if isinstance(resp, dict):  # botocore ClientError; throttling is already retried inside botocore
        return resp.get("Error", {}).get("Code") not in {"ValidationException", "AccessDeniedException"}
    return type(exc).__name__ in {"ReadTimeoutError", "ConnectTimeoutError", "EndpointConnectionError"}


def call(key, model, msgs, attempts=6, max_tokens=MAX_OUT):
    for k in range(attempts):
        try:
            return gw.call(key, model, msgs, max_tokens=max_tokens)
        except Exception as exc:
            if k == attempts - 1 or not retriable(exc):
                raise
            time.sleep(min(90, 5 * 2 ** k) * random.uniform(0.5, 1.5))


def run(args):
    out = pathlib.Path(args.out) if args.out else OUT
    out.mkdir(parents=True, exist_ok=True)
    items = [json.loads(l) for l in (OUT / "items.jsonl").read_text(encoding="utf-8").splitlines()]
    if args.limit:
        keep = {s for s in sorted({i["sequence"] for i in items})[:1]}
        items = [i for i in items if i["sequence"] in keep][:args.limit]
    print("items: %d | smoke %d | conditions %s" % (len(items), sum(i["label"] == "smoke" for i in items), args.conditions))
    if args.dry_run:
        for cond in args.conditions:
            m = messages(items[0], cond)
            print("\n%s: parts %s" % (cond, [p["type"] for p in m[1]["content"]]))
            print("  text:", [p["text"] for p in m[1]["content"] if p["type"] == "text"])
            print("  base64 chars:", sum(len(p["image_url"]["url"]) for p in m[1]["content"] if p["type"] == "image_url"))
        it = items[0]
        print("\nfirst item:", it["item_id"], "| label", it["label"], "| offset", it["offset_seconds"],
              "| reference offset", it["reference_offset_seconds"], "| gap", it["gap_seconds"])
        return

    key = gw.load_key()
    for model in args.models:
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", model)
        paths = {c: out / ("responses-%s-%s.jsonl" % (safe, c)) for c in args.conditions}
        done = {c: {} for c in args.conditions}
        for c, p in paths.items():
            if p.exists():
                for line in p.read_text(encoding="utf-8").splitlines():
                    r = json.loads(line)
                    if not r.get("error"):
                        done[c][r["item_id"]] = r
        jobs = [(c, it) for c in args.conditions for it in items if it["item_id"] not in done[c]]
        random.Random("%d-%s" % (SEED, model)).shuffle(jobs)  # interleave the arms so drift in the endpoint hits both
        print("%s: %d calls to make, %d already stored" % (model, len(jobs), sum(len(v) for v in done.values())), flush=True)
        lock, fresh, t0 = threading.Lock(), collections.Counter(), time.time()

        def one(job):
            cond, it = job
            base = {"item_id": it["item_id"], "sequence": it["sequence"], "label": it["label"],
                    "offset_seconds": it["offset_seconds"], "ladder_target": it["ladder_target"],
                    "gap_seconds": it["gap_seconds"], "condition": cond,
                    "started": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            try:
                text, usage, served = call(key, model, messages(it, cond))
                row = dict(base, raw=text, usage=usage, served_model=served, prediction=parse(text))
            except Exception as exc:
                row = dict(base, error=scrub("%s: %s" % (type(exc).__name__, exc))[:300], prediction=None)
            with lock:
                with paths[cond].open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
                fresh["done"] += 1
                fresh["errors"] += bool(row.get("error"))
                if fresh["done"] % 50 == 0 or fresh["done"] == len(jobs):
                    print("%s: %d/%d | errors %d | %.0f s" % (model, fresh["done"], len(jobs), fresh["errors"],
                                                              time.time() - t0), flush=True)
            return row

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            list(ex.map(one, jobs))
        order = {it["item_id"]: i for i, it in enumerate(items)}
        for c, p in paths.items():  # keep one row per item, the successful one when there is one, in item order
            best = {}
            for line in p.read_text(encoding="utf-8").splitlines():
                r = json.loads(line)
                if r["item_id"] not in best or best[r["item_id"]].get("error"):
                    best[r["item_id"]] = r
            rows = sorted(best.values(), key=lambda r: order.get(r["item_id"], 10 ** 6))
            p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
            ok = [r for r in rows if r["prediction"] is not None]
            y = np.array([r["label"] == "smoke" for r in ok])
            pr = np.array([bool(r["prediction"]) for r in ok])
            print(json.dumps({"run": "%s/%s" % (model, c), "rows": len(rows), "parsed": len(ok),
                              "errors": sum(1 for r in rows if r.get("error")),
                              "recall": round(float(pr[y].mean()), 3) if y.any() else None,
                              "false_positive_rate": round(float(pr[~y].mean()), 3) if (~y).any() else None}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["build", "run"])
    ap.add_argument("--models", nargs="*", default=["claude-opus-5"])
    ap.add_argument("--conditions", nargs="*", default=["bare", "grounded"])
    ap.add_argument("--limit", type=int, default=0, help="first N items of the first sequence, for a smoke test")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="", help="output directory; default task-figlib-masked/")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    build() if args.step == "build" else run(args)


if __name__ == "__main__":
    sys.exit(main())
