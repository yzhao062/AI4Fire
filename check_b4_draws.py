"""B4 check: how many items change their analogue draw if the outcome day must precede the target day (round-2 review)."""
import json, pathlib, random, hashlib
import pandas as pd
import run_allocation as ra

items = ra.sample_items()
pool = ra.build_pool({i["incident_id"] for i in items})
stored = {}
for line in (ra.TASK / "responses-claude-opus-5-grounded.jsonl").read_text(encoding="utf-8").splitlines():
    if line.strip():
        r = json.loads(line); stored[r["item_id"]] = r["analogue_ids"]

def draw(item, rule):
    c = item["context"]; cutoff = pd.Timestamp(item["target_date"]); report = pd.Timestamp(item["report_date"])
    cands = pool.get(ra.key_of(c["acres"], c["percent_contained"], c["personnel_today"]), [])
    if rule == "old":
        hits = [r for r in cands if r["date"] < cutoff]
    elif rule == "outcome<target":   # outcome = date + 1 day must precede the target day
        hits = [r for r in cands if r["date"] + pd.Timedelta(days=1) < cutoff]
    elif rule == "outcome<report":
        hits = [r for r in cands if r["date"] + pd.Timedelta(days=1) < report]
    seed = int.from_bytes(hashlib.blake2b(item["item_id"].encode("utf-8"), digest_size=8).digest(), "big")
    return [r["analogue_id"] for r in random.Random(seed).sample(hits, min(6, len(hits)))]

old_ok = sum(draw(it, "old") == stored[it["item_id"]] for it in items)
print("old rule reproduces stored draws on", old_ok, "of", len(items))
for rule in ("outcome<target", "outcome<report"):
    changed = [it["item_id"] for it in items if draw(it, rule) != stored[it["item_id"]]]
    print(rule, "changes", len(changed), "items", changed[:8])
