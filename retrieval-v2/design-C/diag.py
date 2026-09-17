"""Diagnostics for the report: seed robustness and draw composition. Not a deliverable."""
import pathlib, pickle, sys, random, collections
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
from dev_eval import evaluate, load_items, load_pool, rule_v1, candidates, item_seed
import rule as R
pool = pickle.load(open(HERE / "pool-cache.pkl", "rb")) if (HERE / "pool-cache.pkl").exists() else load_pool()
items = load_items()

def reseeded(fn, k):
    return lambda cands, item, rng: fn(cands, item, random.Random(item_seed(item["item_id"]) + k))

print("seed robustness (nMAE / falsmv / mAUC) over 5 alternative seeds:")
for name in ("v1", "C-model", "C-model-dir", "C-hgb-dir"):
    fn = rule_v1 if name == "v1" else R.RULES[name]
    vals = []
    for k in range(1, 6):
        o = evaluate(reseeded(fn, k), pool, items, name=name, quiet=True)
        vals.append((o["nmae"], o["false_move"], o["move_auc"]))
    v = np.array(vals)
    print("  %-12s nMAE %.3f..%.3f (mean %.3f)  falsmv %.2f..%.2f  mAUC %.3f..%.3f" % (
        name, v[:, 0].min(), v[:, 0].max(), v[:, 0].mean(), v[:, 1].min(), v[:, 1].max(), v[:, 2].min(), v[:, 2].max()))

# draw composition of the frozen variant under the harness seeds
nm, nu, thin, pred = collections.Counter(), collections.Counter(), 0, collections.Counter()
for it in items:
    cands = candidates(pool, it)
    rng = random.Random(item_seed(it["item_id"]))
    p = R.p_move_model(it); q = R.p_up_model(it)
    rows = R.rule_model_dir(cands, it, rng)
    c = it["context"]; key = R.key_of(c["acres"], c["percent_contained"], c["personnel_today"])
    n_move = R._round_half_up(6 * p); n_up = R._round_half_up(n_move * q)
    nm[n_move] += 1; nu[(n_move, n_up)] += 1
    if any(R.row_key(r) != key for r in rows):
        thin += 1
    m = float(np.median([R.ratio(r) for r in rows])) if rows else 1.0
    pred["move" if abs(m - 1) > 0.1 else "hold"] += 1
print("C-model-dir: n_move distribution", dict(sorted(nm.items())))
print("C-model-dir: (n_move, n_up) distribution", dict(sorted(nu.items())))
print("C-model-dir: items with at least one personnel-band top-up row: %d of %d" % (thin, len(items)))
print("C-model-dir: analogue-only prediction moves on %d items, holds on %d" % (pred["move"], pred["hold"]))
