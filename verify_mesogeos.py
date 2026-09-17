"""Check the Mesogeos items against the source csvs: traceability, leak-safety, and schema."""
import json
import math
import pathlib
import random

import pandas as pd

S = pathlib.Path(__file__).parent
D = S / "data" / "mesogeos"
items = [json.loads(l) for l in (S / "task-mesogeos" / "items.jsonl").open(encoding="utf-8")]
print("items %d | unique ids %d" % (len(items), len({i["item_id"] for i in items})))

required = {"item_id", "split", "label", "answer", "burned_area_ha", "source_file", "source_sample",
            "target_date", "context"}
ctx_required = {"longitude", "latitude", "window_start", "window_end", "target_date", "history_days",
                "size_class_hectares", "cell_km", "daily", "static", "units"}
assert all(required <= set(i) for i in items), "an item is missing a top-level key"
assert all(ctx_required <= set(i["context"]) for i in items), "an item is missing a context key"

# Nothing in context may reveal the answer. The three withheld columns must be absent, and no context value
# may correlate perfectly with the label by construction.
banned = {"burned_areas", "ignition_points", "burned_area_has", "burned_area_ha", "label", "answer"}
for i in items:
    blob = json.dumps(i["context"])
    for b in banned:
        assert '"%s"' % b not in blob, "%s leaked into context of %s" % (b, i["item_id"])
print("leak check: none of %s appears inside any context block" % sorted(banned))

for i in items:
    ws, we, td = (pd.Timestamp(i["context"][k]) for k in ("window_start", "window_end", "target_date"))
    assert (we - ws).days == 29, i["item_id"]
    assert (td - we).days == 1, i["item_id"]
    assert i["context"]["target_date"] == i["target_date"]
    assert len(i["context"]["daily"]) == 12 and all(len(v) == 30 for v in i["context"]["daily"].values())
    assert len(i["context"]["static"]) == 12
    assert (i["burned_area_ha"] is None) == (i["label"] == 0)
    if i["label"]:
        assert i["burned_area_ha"] >= 30, i["item_id"]
print("shape check: every item carries a 30-day window, 12 daily series, 12 static values, target = end + 1")
print("positives all carry a burned area of at least 30 ha; negatives carry none")

lc = [c for c in items[0]["context"]["static"] if c.startswith("lc_")]
bad = [i["item_id"] for i in items if abs(sum(i["context"]["static"][c] for c in lc) - 1) > 2e-3]
print("land cover fractions sum to 1 in %d of %d items" % (len(items) - len(bad), len(items)))

# Trace a random sample of items back to the exact source rows.
random.seed(0)
checked = 0
for name, want in (("positives.csv", 1), ("negatives.csv", 0)):
    df = pd.read_csv(D / name, low_memory=False)
    df = df.set_index(["sample", "time_idx"])
    picks = random.sample([i for i in items if i["label"] == want], 12)
    for it in picks:
        sid = it["source_sample"]
        rows = df.loc[sid].sort_index()
        assert len(rows) == 30
        assert str(pd.Timestamp(rows.iloc[-1]["time"]).date()) == it["context"]["window_end"], it["item_id"]
        for col, series in it["context"]["daily"].items():
            src = rows[col].to_numpy()
            for k, v in enumerate(series):
                s = src[k]
                if v is None:
                    assert math.isnan(s), "%s %s day %d: item says missing, source says %s" % (it["item_id"], col, k, s)
                else:
                    assert abs(v - s) <= abs(s) * 1e-5 + 1e-12, "%s %s day %d: %s vs %s" % (it["item_id"], col, k, v, s)
        for col, v in it["context"]["static"].items():
            s = rows.iloc[0][col]
            assert abs(v - s) <= abs(s) * 1e-5 + 1e-12, "%s %s: %s vs %s" % (it["item_id"], col, v, s)
        checked += 1
print("traceability: %d random items match their source rows to six significant figures" % checked)

idx = pd.read_csv(S / "task-mesogeos" / "items-index.csv")
assert list(idx["item_id"]) == [i["item_id"] for i in items], "index and jsonl disagree on order"
print("index csv matches items.jsonl row for row")
print("median item %d bytes | file %.1f MB"
      % (sorted(len(json.dumps(i)) for i in items)[len(items) // 2],
         (S / "task-mesogeos" / "items.jsonl").stat().st_size / 1e6))
