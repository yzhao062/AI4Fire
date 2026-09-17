"""Fresh-process check of the frozen rule: no prepare() call, draws must equal the ones run.py saved, rng unused."""
import json, pathlib, random, sys, time
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import dev_eval, rule as rulemod
pool = dev_eval.load_pool()
items = dev_eval.load_items()
saved = json.load(open(HERE / "results.json"))["frozen_drawn_ids"]
fn = rulemod.RULES[rulemod.FROZEN]
t0 = time.time(); n = 0
for it in items[::7]:
    cands = dev_eval.candidates(pool, it)
    got = [r["analogue_id"] for r in fn(cands, it, random.Random(0))]
    assert got == saved[it["item_id"]], (it["item_id"], got, saved[it["item_id"]])
    assert len(got) <= 6 and len(set(got)) == len(got)
    n += 1
print("%s: %d items redrawn without prepare() in a fresh process, all identical to the saved draws (%.1fs)" % (rulemod.FROZEN, n, time.time() - t0))
# constants actually frozen: the module reads scales.json and never refits
print("scales:", json.dumps({f: [round(rulemod.SCALES["mean"][f], 4), round(rulemod.SCALES["std"][f], 4)] for f in rulemod.FEATURES}))
print("weights:", rulemod.WEIGHTS[fn.weights], "scope:", fn.scope, "cap:", fn.cap)
