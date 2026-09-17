"""Structural checks only: missingness and ranges of the features the family-B rule uses. No evaluation on test items."""
import json, pathlib, sys, time
import numpy as np, pandas as pd
WF = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WF))
import dev_eval

t0 = time.time()
pool = dev_eval.load_pool()
print("pool rows %d built in %.1fs" % (len(pool), time.time() - t0))
df = pd.DataFrame(pool)
pre = df[df["date"] < pd.Timestamp("2015-01-01")]
print("pre-2015 rows: %d of %d" % (len(pre), len(df)))
for c in ["prev", "acres", "pct", "new_acres", "aerial", "acres_prev", "pct_prev"]:
    print("  missing %-10s all %.3f  pre2015 %.3f" % (c, df[c].isna().mean(), pre[c].isna().mean()))
print("day_of_run quantiles", np.quantile(df["day_of_run"], [0, .25, .5, .75, .9, .99, 1]))
print("today quantiles", np.quantile(df["today"], [0, .25, .5, .75, .9, .99, 1]))
print("new_acres quantiles (non-null)", np.nanquantile(df["new_acres"].astype(float), [0, .25, .5, .75, .9, .99, 1]))
print("aerial quantiles (non-null)", np.nanquantile(df["aerial"].astype(float), [0, .25, .5, .75, .9, .99, 1]))
print("negative new_acres share", float((df["new_acres"] < 0).mean()))
chg = np.log(df["today"] / df["prev"].astype(float))
print("log change quantiles (non-null)", np.nanquantile(chg, [0, .05, .25, .5, .75, .95, 1]))
rat = df["next"] / np.maximum(df["today"], 1)
print("pool moved share (|ratio-1|>0.1): %.3f" % float((np.abs(rat - 1) > 0.1).mean()))
# does yesterday's change predict today's move in the pool?
has = ~chg.isna()
mv_prev = np.abs(chg[has]) > np.log(1.1)
mv_next = (np.abs(rat - 1) > 0.1)[has]
print("P(move next | moved prev) = %.3f ; P(move next | flat prev) = %.3f" % (mv_next[mv_prev].mean(), mv_next[~mv_prev].mean()))

items = dev_eval.load_items()
test = dev_eval.load_items(dev_eval.FIRE_BENCH / "task-allocation" / "items-v1.jsonl")
for name, its in [("dev", items), ("test", test)]:
    ok_last = all(it["context"]["personnel_last_days"][-1] == it["context"]["personnel_today"] for it in its)
    n_hist = sorted({len(it["context"]["personnel_last_days"]) for it in its})
    dor = sorted({it["day_of_run"] for it in its})
    miss = {k: np.mean([it["context"][k] is None for it in its]) for k in ["acres", "percent_contained", "new_acres", "aerial_resources"]}
    print("%s: n=%d last_days[-1]==today: %s | hist lengths %s | day_of_run range %d..%d | missing %s" % (
        name, len(its), ok_last, n_hist, dor[0], dor[-1], {k: round(float(v), 3) for k, v in miss.items()}))
    print("   report years", sorted({it["report_date"][:4] for it in its}))
