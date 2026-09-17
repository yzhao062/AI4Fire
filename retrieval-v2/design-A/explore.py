"""Exploration: how much do one-day dynamics predict next-day movement? Pool rows before 2015 and dev items."""
import sys, pathlib, collections
import numpy as np, pandas as pd
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from dev_eval import load_pool, load_items, key_of

pool = load_pool()
items = load_items()
print("pool rows", len(pool))
pre = [r for r in pool if r["date"] < pd.Timestamp("2015-01-01")]
print("pre-2015 rows", len(pre))

def pchg(r):
    if r["prev"] is None:
        return None
    return r["today"] / max(r["prev"], 1)

def moved(r):
    return abs(r["next"] / max(r["today"], 1) - 1) > 0.1

rows = pre
pc = np.array([pchg(r) if pchg(r) is not None else np.nan for r in rows])
mv = np.array([moved(r) for r in rows])
print("share prev None: %.3f" % np.isnan(pc).mean())
ok = ~np.isnan(pc)
print("share exactly equal (today==prev): %.3f" % np.mean(pc[ok] == 1.0))
print("share within 10%%: %.3f" % np.mean(np.abs(pc[ok] - 1) <= 0.1))
print("P(move next) overall: %.3f" % mv.mean())
for name, m in [("exact hold", pc == 1.0), ("hold (<=10%, not exact)", (np.abs(pc - 1) <= 0.1) & (pc != 1.0)),
                ("fell 10-25%", (pc < 0.9) & (pc >= 0.75)), ("fell >25%", pc < 0.75),
                ("rose 10-25%", (pc > 1.1) & (pc <= 1.25)), ("rose >25%", pc > 1.25), ("prev None", np.isnan(pc))]:
    m = m & (ok | np.isnan(pc))
    if m.sum():
        nr = np.array([r["next"] / max(r["today"], 1) for r in rows])[m]
        print("  %-26s n=%6d  P(move)=%.3f  median ratio=%.3f  IQR=[%.2f, %.2f]  P(up)=%.3f P(down)=%.3f" % (
            name, m.sum(), mv[m].mean(), np.median(nr), np.quantile(nr, .25), np.quantile(nr, .75),
            np.mean(nr > 1.1), np.mean(nr < 0.9)))

# acres growth
def agrow(r):
    if r["acres"] is None or r["acres_prev"] is None:
        return None
    return r["acres"] > r["acres_prev"]
ag = [agrow(r) for r in rows]
for name, f in [("acres grew", lambda a: a is True), ("acres held", lambda a: a is False), ("unknown", lambda a: a is None)]:
    m = np.array([f(a) for a in ag])
    print("  %-12s n=%6d P(move)=%.3f" % (name, m.sum(), mv[m].mean()))
na = np.array([r["new_acres"] if r["new_acres"] is not None else np.nan for r in rows])
for name, m in [("new_acres>0", na > 0), ("new_acres==0", na == 0), ("new_acres nan", np.isnan(na))]:
    print("  %-14s n=%6d P(move)=%.3f" % (name, m.sum(), mv[m].mean()))
# containment change
def pchg2(r):
    if r["pct"] is None or r["pct_prev"] is None:
        return None
    return r["pct"] - r["pct_prev"]
pd_ = [pchg2(r) for r in rows]
for name, f in [("pct rose", lambda a: a is not None and a > 0), ("pct held", lambda a: a is not None and a == 0),
                ("pct fell", lambda a: a is not None and a < 0), ("unknown", lambda a: a is None)]:
    m = np.array([f(a) for a in pd_])
    print("  %-10s n=%6d P(move)=%.3f" % (name, m.sum(), mv[m].mean()))
# joint: personnel hold x acres growth
for pn, pm in [("exact hold", pc == 1.0), ("moved >10%", np.abs(pc - 1) > 0.1)]:
    for an, af in [("grew", lambda a: a is True), ("held", lambda a: a is False)]:
        m = pm & np.array([af(a) for a in ag])
        print("  %-12s x acres %-5s n=%6d P(move)=%.3f" % (pn, an, m.sum(), mv[m].mean()))
# day_of_run
dr = np.array([r["day_of_run"] for r in rows])
for lo, hi in [(1, 2), (2, 4), (4, 8), (8, 15), (15, 30), (30, 1000)]:
    m = (dr >= lo) & (dr < hi)
    print("  day_of_run [%d,%d) n=%6d P(move)=%.3f" % (lo, hi, m.sum(), mv[m].mean()))
# two-day history via same-incident lookup
byinc = collections.defaultdict(dict)
for r in pool:
    byinc[r["incident_id"]][r["date"]] = r
def prev2(r):
    p = byinc[r["incident_id"]].get(r["date"] - pd.Timedelta(days=1))
    return None if p is None else p["prev"]
p2 = np.array([ (prev2(r) if prev2(r) is not None else np.nan) for r in rows])
prv = np.array([ (r["prev"] if r["prev"] is not None else np.nan) for r in rows])
tod = np.array([r["today"] for r in rows])
hold2 = (prv == tod) & (p2 == prv)
hold1only = (prv == tod) & ~(p2 == prv) & ~np.isnan(p2)
print("  held 2 days exactly n=%d P(move)=%.3f | held 1 day only n=%d P(move)=%.3f" % (hold2.sum(), mv[hold2].mean(), hold1only.sum(), mv[hold1only].mean()))

print("\nDEV ITEMS")
y = np.array([it["target_personnel"] for it in items]); b = np.array([it["baseline_persistence"] for it in items])
imv = np.abs(y - b) / np.maximum(b, 1) > 0.1
ipc = np.array([it["context"]["personnel_last_days"][-1] / max(it["context"]["personnel_last_days"][-2], 1) for it in items])
print("len(personnel_last_days) counts:", collections.Counter(len(it["context"]["personnel_last_days"]) for it in items))
print("personnel_today == last_days[-1]: %.3f" % np.mean([it["context"]["personnel_today"] == it["context"]["personnel_last_days"][-1] for it in items]))
for name, m in [("exact hold", ipc == 1.0), ("hold (<=10%, not exact)", (np.abs(ipc - 1) <= 0.1) & (ipc != 1.0)),
                ("fell 10-25%", (ipc < 0.9) & (ipc >= 0.75)), ("fell >25%", ipc < 0.75),
                ("rose 10-25%", (ipc > 1.1) & (ipc <= 1.25)), ("rose >25%", ipc > 1.25)]:
    print("  %-26s n=%4d P(move)=%.3f  actual ratio median=%.3f" % (name, m.sum(), imv[m].mean() if m.sum() else float('nan'),
          np.median((y / np.maximum(b, 1))[m]) if m.sum() else float('nan')))
iag = np.array([ (it["context"]["acres_last_days"][-1] or 0) > (it["context"]["acres_last_days"][-2] or 0) if len(it["context"]["acres_last_days"]) >= 2 and it["context"]["acres_last_days"][-1] is not None and it["context"]["acres_last_days"][-2] is not None else False for it in items])
print("  acres grew n=%d P(move)=%.3f | not n=%d P(move)=%.3f" % (iag.sum(), imv[iag].mean(), (~iag).sum(), imv[~iag].mean()))
ina = np.array([ (it["context"]["new_acres"] or 0) > 0 for it in items])
print("  new_acres>0 n=%d P(move)=%.3f | not n=%d P(move)=%.3f" % (ina.sum(), imv[ina].mean(), (~ina).sum(), imv[~ina].mean()))
# hold streaks on items
pl = [it["context"]["personnel_last_days"] for it in items]
h2 = np.array([len(p) >= 3 and p[-1] == p[-2] == p[-3] for p in pl])
h1 = np.array([len(p) >= 3 and p[-1] == p[-2] and p[-2] != p[-3] for p in pl])
print("  held 2 days n=%d P(move)=%.3f | held 1 day only n=%d P(move)=%.3f" % (h2.sum(), imv[h2].mean(), h1.sum(), imv[h1].mean()))
# v1 band cell sizes for dev items
from dev_eval import candidates
sizes = []
for it in items[:100]:
    c = it["context"]; k = key_of(c["acres"], c["percent_contained"], c["personnel_today"])
    sizes.append(sum(1 for r in candidates(pool, it) if r["key"] == k))
print("  v1 band cell sizes (first 100 items): median %d, q10 %d, min %d" % (np.median(sizes), np.quantile(sizes, .1), min(sizes)))
