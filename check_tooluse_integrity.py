"""Final integrity pass on the shipped item set: unique ids, unique questions, required fields, and disk footprint."""
import json
import pathlib
from collections import Counter

S = pathlib.Path(__file__).parent
P = S / "task-tooluse" / "items.jsonl"
ITEMS = [json.loads(l) for l in P.read_text(encoding="utf-8").splitlines() if l]

print("items:", len(ITEMS))
ids, qs = Counter(i["item_id"] for i in ITEMS), Counter(i["prompt"]["question"] for i in ITEMS)
print("duplicate item_ids:", [k for k, v in ids.items() if v > 1])
print("duplicate questions:", [k[:70] for k, v in qs.items() if v > 1])

REQ = ["item_id", "family", "tier", "prompt", "answer", "answer_type", "tolerance", "reference_query",
       "reference_params", "memorization_risk", "dating", "source"]
missing = [(i["item_id"], f) for i in ITEMS for f in REQ if f not in i or i[f] is None]
print("items missing a required field:", missing[:5], "count:", len(missing))
noans = [i["item_id"] for i in ITEMS if i["answer"] is None or i["answer"] == ""]
print("items with an empty answer:", noans)
nodate = [i["item_id"] for i in ITEMS if not i["dating"]["fire_years_covered"]]
print("items with no year range:", nodate)
opts = [i["item_id"] for i in ITEMS if i["answer_type"] == "categorical"
        and (not i["prompt"]["options"] or i["answer"] not in i["prompt"]["options"])]
print("categorical items whose answer is not in the options list:", opts)

print("\nper family:", dict(Counter(i["family"] for i in ITEMS)))
print("\none full item:\n", json.dumps(ITEMS[80], indent=2)[:2400])

print("\ndisk footprint under data/tooluse:")
tot = 0
for f in sorted((S / "data" / "tooluse").iterdir()):
    tot += f.stat().st_size
    print("  %-46s %12.1f MB" % (f.name, f.stat().st_size / 1e6))
print("  total on disk %.1f MB" % (tot / 1e6))
print("task-tooluse:", *["%s %.0f KB" % (f.name, f.stat().st_size / 1e3)
                         for f in sorted((S / "task-tooluse").iterdir())])
