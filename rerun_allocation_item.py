"""Re-query the grounded condition for one allocation item whose analogue draw changed (round-2 review, B4).

The round-2 review found one drawn analogue whose outcome day (the day after its input day) fell on the item's
target day, so the grounded prompt showed a next-day staffing figure that was not on file when the forecast was
made. The eligibility rule now requires both analogue days to precede the item's report day. On items-v1 that
rule changes the draw for exactly one item (check_b4_draws.py), so this script re-queries only that item, for
every model, and writes the fresh rows to task-allocation/b4-rerun/. splice_b4.py moves them into the response
files and keeps the replaced rows.

Usage: python rerun_allocation_item.py alloc-2018_9144638_MILES-2018-09-11 [--models ...]
"""
import argparse
import collections
import hashlib
import json
import random
import re

import numpy as np
import pandas as pd

import gw
import run_allocation as ra

DEFAULT_MODELS = ["claude-opus-5", "claude-opus-4.8", "gemini-3.1-pro",
                  "bedrock:qwen.qwen3-vl-235b-a22b", "bedrock:us.meta.llama4-maverick-17b-instruct-v1:0"]


def draw_strict(pool, item, k=6):
    """The corrected rule: an analogue's input day and outcome day both precede the item's report day."""
    c = item["context"]
    report = pd.Timestamp(item["report_date"])
    hits = [r for r in pool.get(ra.key_of(c["acres"], c["percent_contained"], c["personnel_today"]), [])
            if r["date"] + pd.Timedelta(days=1) < report]
    seed = int.from_bytes(hashlib.blake2b(item["item_id"].encode("utf-8"), digest_size=8).digest(), "big")
    return random.Random(seed).sample(hits, min(k, len(hits)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("item_id")
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    args = ap.parse_args()
    items = ra.sample_items()
    it = next(i for i in items if i["item_id"] == args.item_id)
    fire_mean = collections.defaultdict(list)
    for line in (ra.TASK / "items.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        fire_mean[row["incident_id"]].append(row["target_personnel"])
    fm = float(np.mean(fire_mean[it["incident_id"]]))
    pool = ra.build_pool({i["incident_id"] for i in items})
    rows = draw_strict(pool, it)
    print("item", it["item_id"], "report", it["report_date"], "target", it["target_date"])
    print("old analogues:", [r["analogue_id"] for r in ra.analogues(pool, it)])
    print("new analogues:", [r["analogue_id"] for r in rows])
    out_dir = ra.TASK / "b4-rerun"
    out_dir.mkdir(exist_ok=True)
    key = gw.load_key()
    msgs = ra.render(it, ra.grounded_block(rows))
    (out_dir / ("prompt-grounded-%s.txt" % it["item_id"])).write_text(msgs[1]["content"], encoding="utf-8")
    for model in args.models:
        text, usage, served = gw.call(key, model, msgs, max_tokens=ra.MAX_OUT)
        rec = {"item_id": it["item_id"], "raw": text, "usage": usage, "served_model": served,
               "prediction": ra.parse(text), "target": it["target_personnel"],
               "persistence": it["baseline_persistence"], "fire_mean": fm,
               "analogue_ids": [r["analogue_id"] for r in rows]}
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", model)
        (out_dir / ("responses-%s-grounded.jsonl" % safe)).write_text(json.dumps(rec) + "\n", encoding="utf-8")
        print("%-50s prediction %s  target %s  persistence %s" % (model, rec["prediction"], rec["target"], rec["persistence"]))


if __name__ == "__main__":
    main()
