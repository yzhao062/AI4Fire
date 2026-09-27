"""Fire danger rerun with the sample's base rate stated in the prompt, in both arms.

    python run_mesogeos_baserate.py --dry-run
    python run_mesogeos_baserate.py --models claude-opus-5 gemini-3.1-pro

Mesogeos Track A subsamples negatives, so 131 of the 386 fold-0 test items (0.339) are fires, far above regional
prevalence, and the version-1 prompts never said so. The paper's prior-shift rescoring moved stored probabilities to
that rate after the fact; this rerun states the rate in the prompt and asks again. The prompt is run_mesogeos.render
unchanged except for one line placed just before the question in both arms, so bare still carries the driver window
alone and grounded still adds the monthly climatology. Outputs go to task-mesogeos-baserate/ and never touch the
version-1 files.
"""
import argparse
import collections
import json
import pathlib
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np

import gw
import run_mesogeos as rm
from run_figlib_masked import call, scrub

S = pathlib.Path(__file__).parent
OUT = S / "task-mesogeos-baserate"
SEED = 20260927


def rate_line(items):
    sizes = {i["context"]["size_class_hectares"] for i in items}
    if len(sizes) != 1:
        raise SystemExit("items mix size classes %s" % sorted(sizes))
    rate = np.mean([i["label"] for i in items])
    return ("Base rate: in this evaluation sample, %.1f percent of items had a wildfire of at least %d hectares start "
            "on the target date." % (100 * rate, sizes.pop()))


def render(item, clim, line):
    msgs = rm.render(item, clim)
    lines = msgs[1]["content"].split("\n")
    if not (lines[-1].startswith("Question:") and lines[-2] == ""):
        raise SystemExit("run_mesogeos.render no longer ends with a blank line and the question")
    lines.insert(len(lines) - 1, line)
    msgs[1]["content"] = "\n".join(lines)
    return msgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["claude-opus-5"])
    ap.add_argument("--conditions", nargs="*", default=["bare", "grounded"])
    ap.add_argument("--limit", type=int, default=0, help="first N items, for a smoke test")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="", help="output directory; default task-mesogeos-baserate/")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    out = pathlib.Path(args.out) if args.out else OUT
    out.mkdir(parents=True, exist_ok=True)

    allitems = [json.loads(l) for l in (rm.TASK / "items.jsonl").read_text(encoding="utf-8").splitlines()]
    items = [i for i in allitems if i["split"] == "test" and i.get("fold", 0) == 0]
    line = rate_line(items)  # the rate of all 386 items, also when --limit cuts the run short
    clim = rm.climatology()
    if args.limit:
        items = items[:args.limit]
    print("items: %d | positive rate %.3f | %s" % (len(items), np.mean([i["label"] for i in items]), line))

    def prompt(it, cond):
        month = it["context"]["window_end"][5:7]
        return render(it, clim.get(month) if cond == "grounded" else None, line)

    if args.dry_run:
        print("\n--- bare tail\n" + prompt(items[0], "bare")[1]["content"][-500:])
        print("\n--- grounded tail\n" + prompt(items[0], "grounded")[1]["content"][-900:])
        return

    key = gw.load_key()
    for model in args.models:
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", model)
        paths = {c: out / ("responses-%s-%s.jsonl" % (safe, c)) for c in args.conditions}
        done = {c: {} for c in args.conditions}
        for c, p in paths.items():
            if p.exists():
                for l in p.read_text(encoding="utf-8").splitlines():
                    r = json.loads(l)
                    if not r.get("error"):
                        done[c][r["item_id"]] = r
        jobs = [(c, it) for c in args.conditions for it in items if it["item_id"] not in done[c]]
        random.Random("%d-%s" % (SEED, model)).shuffle(jobs)  # interleave the arms so drift in the endpoint hits both
        print("%s: %d calls to make, %d already stored" % (model, len(jobs), sum(len(v) for v in done.values())), flush=True)
        lock, fresh, t0 = threading.Lock(), collections.Counter(), time.time()

        def one(job):
            cond, it = job
            base = {"item_id": it["item_id"], "label": it["label"], "target_date": it["target_date"],
                    "variant": "baserate", "condition": cond,
                    "started": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            try:
                text, usage, served = call(key, model, prompt(it, cond), max_tokens=rm.MAX_OUT)
                usage["max_out"] = rm.MAX_OUT
                p, f = rm.parse(text)
                row = dict(base, raw=text, usage=usage, served_model=served, probability=p, call=f)
            except Exception as exc:
                row = dict(base, error=scrub("%s: %s" % (type(exc).__name__, exc))[:300], probability=None, call=None)
            with lock:
                with paths[cond].open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
                fresh["done"] += 1
                fresh["errors"] += bool(row.get("error"))
                if fresh["done"] % 100 == 0 or fresh["done"] == len(jobs):
                    print("%s: %d/%d | errors %d | %.0f s" % (model, fresh["done"], len(jobs), fresh["errors"],
                                                              time.time() - t0), flush=True)
            return row

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            list(ex.map(one, jobs))
        order = {it["item_id"]: i for i, it in enumerate(items)}
        for c, p in paths.items():  # keep one row per item, the successful one when there is one, in item order
            best = {}
            for l in p.read_text(encoding="utf-8").splitlines():
                r = json.loads(l)
                if r["item_id"] not in best or best[r["item_id"]].get("error"):
                    best[r["item_id"]] = r
            rows = sorted(best.values(), key=lambda r: order.get(r["item_id"], 10 ** 6))
            p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
            s = rm.score(rows, "%s/%s-baserate" % (model, c))
            s.pop("published_auprc", None)
            s["errors"] = sum(1 for r in rows if r.get("error"))
            print(json.dumps(s), flush=True)


if __name__ == "__main__":
    main()
