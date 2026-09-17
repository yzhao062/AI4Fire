import pathlib, sys
import numpy as np, pandas as pd
WF = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WF))
import dev_eval
pool = dev_eval.load_pool()
df = pd.DataFrame(pool)
p = df["pct"].astype(float)
print("pct > 100:", int((p > 100).sum()), " pct < 0:", int((p < 0).sum()), " of non-null", int(p.notna().sum()))
print("largest pct values:", sorted(p.dropna().unique())[-12:])
print("pct quantiles (clipped to 0..100):", np.nanquantile(p.clip(0, 100), [0, .1, .25, .5, .75, .9, 1]))
print("mean/std clipped pre-2015:", float(p[df["date"] < "2015-01-01"].clip(0, 100).mean()), float(p[df["date"] < "2015-01-01"].clip(0, 100).std(ddof=0)))
items = dev_eval.load_items()
ip = np.array([it["context"]["percent_contained"] for it in items if it["context"]["percent_contained"] is not None], dtype=float)
print("dev items pct > 100:", int((ip > 100).sum()))
