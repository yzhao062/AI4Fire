"""Diagnostics for one Family A variant on development: which ladder level served each item, and the error
against v1 by the item's one-day personnel-change bucket. Usage: python diag.py [variant]  (default: rule.FROZEN)"""
import collections
import pathlib
import random
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
from dev_eval import candidates, evaluate, item_seed, key_of, load_items, load_pool, ratio, rule_v1  # noqa: E402
import rule as R  # noqa: E402

name = sys.argv[1] if len(sys.argv) > 1 else R.FROZEN
fn = R.RULES[name]
pool = load_pool(); items = load_items()
ref = evaluate(rule_v1, pool, items, name="v1", quiet=True)
out = evaluate(fn, pool, items, name=name, reference=ref, quiet=True)
print("%s: nMAE %.4f (v1 %.4f) diff %+.4f [%+.4f, %+.4f] mAUC %.4f false-move %.4f cov3 %.4f" % (
    name, out["nmae"], ref["nmae"], out["diff_vs_reference"], out["diff_ci"][0], out["diff_ci"][1],
    out["move_auc"], out["false_move"], out["coverage_3"]))

# ladder level served: recompute the level by matching the drawn rows' keys against the item key
levels = collections.Counter()
by_bucket = collections.defaultdict(list)
y = np.array([it["target_personnel"] for it in items]); b = np.array([it["baseline_persistence"] for it in items])
moved = np.abs(y - b) / np.maximum(b, 1) > 0.1
e_new = np.array(out["errors"]); e_v1 = np.array(ref["errors"])
comps = fn.components
for i, it in enumerate(items):
    c = it["context"]
    d = R._item_dyn(it)
    ik = (key_of(c["acres"], c["percent_contained"], c["personnel_today"]),) + tuple(R.COMPONENTS[k](d) for k in comps[1:])
    cands = candidates(pool, it)
    rows = fn(cands, it, random.Random(item_seed(it["item_id"])))
    ck = "_kA:" + "+".join(comps)
    if not rows:
        levels["none"] += 1
    else:
        # deepest key prefix shared by every drawn row and the item (the ladder level that served it)
        depth = 0
        for n in range(len(comps), 0, -1):
            if all(r[ck][:n] == ik[:n] for r in rows):
                depth = n
                break
        levels["+".join(comps[:depth]) if depth else "none"] += 1
    by_bucket[R._p5(d)].append(i)
print("ladder level that served the items:")
for k, v in sorted(levels.items(), key=lambda kv: -kv[1]):
    print("  %-28s %4d  (%.2f)" % (k, v, v / len(items)))
print("error by the item's five-way one-day personnel change (n, P(moved), nMAE new, nMAE v1, stable-day nMAE new/v1):")
for k in ("F", "f", "h", "r", "R", "u"):
    idx = np.array(by_bucket.get(k, []), dtype=int)
    if len(idx) == 0:
        continue
    st = idx[~moved[idx]]
    print("  %-2s n=%3d  P(moved)=%.2f  nMAE %.3f vs %.3f  | stable-day nMAE %.3f vs %.3f (n=%d)" % (
        k, len(idx), moved[idx].mean(), e_new[idx].mean(), e_v1[idx].mean(),
        e_new[st].mean() if len(st) else float("nan"), e_v1[st].mean() if len(st) else float("nan"), len(st)))
print("median-ratio IQR new %s | v1 %s | actual %s" % (
    [round(x, 3) for x in out["median_ratio_iqr"]], [round(x, 3) for x in ref["median_ratio_iqr"]], [round(x, 3) for x in out["actual_ratio_iqr"]]))
print("mean drawn %.2f | coverage_1 %.3f | items per second ok" % (out["mean_drawn"], out["coverage_1"]))
