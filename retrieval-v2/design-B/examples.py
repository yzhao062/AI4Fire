import json, pathlib, random, sys
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import dev_eval, rule as rulemod
pool = dev_eval.load_pool(); rulemod.prepare(pool)
items = dev_eval.load_items()
fn = rulemod.RULES[rulemod.FROZEN]
# pick: one item that jumped yesterday, one flat for days, one late-run demob
def chg(it):
    h = it["context"]["personnel_last_days"]; return h[-1] / max(h[-2], 1)
picks = [max(items, key=chg), min(items, key=chg), next(it for it in items if chg(it) == 1.0 and it["day_of_run"] > 20 and it["context"]["percent_contained"] == 100.0)]
for it in picks:
    c = it["context"]
    print("ITEM %s | day %d | pers %s -> target %s | acres %s new %s pct %s aerial %s" % (
        it["item_id"], it["day_of_run"], c["personnel_last_days"], it["target_personnel"], c["acres"], c["new_acres"], c["percent_contained"], c["aerial_resources"]))
    rows = fn(dev_eval.candidates(pool, it), it, random.Random(0))
    for r in rows:
        print("   %-45s day %3d | prev %6s today %6.0f next %6.0f ratio %.2f | acres %8.0f new %6.0f pct %5s aerial %3.0f" % (
            r["analogue_id"], r["day_of_run"], "-" if r["prev"] is None else "%.0f" % r["prev"], r["today"], r["next"], r["next"] / max(r["today"], 1),
            r["acres"], r["new_acres"], r["pct"], r["aerial"]))
    print("   median ratio %.2f | v1 draw median %.2f" % (np.median([r["next"] / max(r["today"], 1) for r in rows]),
          np.median([r["next"] / max(r["today"], 1) for r in dev_eval.rule_v1(dev_eval.candidates(pool, it), it, random.Random(dev_eval.item_seed(it["item_id"])))] or [1.0])))
