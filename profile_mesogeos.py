"""Profile the Mesogeos Track A csvs: sample counts, split sizes, window alignment, and leakage columns."""
import pathlib

import numpy as np
import pandas as pd

D = pathlib.Path(__file__).parent / "data" / "mesogeos"

pos = pd.read_csv(D / "positives.csv", low_memory=False)
neg = pd.read_csv(D / "negatives.csv", low_memory=False)
for name, df in (("positives", pos), ("negatives", neg)):
    print("=" * 30, name, "=" * 30)
    print("rows %d | samples %d | rows/sample %s" % (len(df), df["sample"].nunique(),
                                                     sorted(df.groupby("sample").size().unique())))
    print("time_idx range", df["time_idx"].min(), df["time_idx"].max())
    last = df[df["time_idx"] == 29].copy()
    last["time"] = pd.to_datetime(last["time"])
    last["YEAR"] = last["time"].dt.year
    print("last-day years:\n", last.groupby("YEAR").size().to_string())
    print("date span %s .. %s" % (last["time"].min().date(), last["time"].max().date()))
    print("burned_area_has: min %s median %s max %s" % (last["burned_area_has"].min(),
                                                        last["burned_area_has"].median(),
                                                        last["burned_area_has"].max()))
    for col in ["burned_areas", "ignition_points"]:
        v = df[col]
        print("%s nonzero rows: %d / %d | last-day nonzero: %d / %d"
              % (col, int((v.fillna(0) != 0).sum()), len(df),
                 int((last[col].fillna(0) != 0).sum()), len(last)))
    print("duplicate (x,y,last-day) keys:", int(last.duplicated(subset=["x", "y", "time"]).sum()))
    print("nan share by column (top 8):")
    print((df.isna().mean().sort_values(ascending=False).head(8) * 100).round(2).to_string())

# Sample-index contiguity: dataset.py slices by row block of 30, which assumes sorted, contiguous blocks.
for name, df in (("positives", pos), ("negatives", neg)):
    idx = df["sample"].to_numpy()
    blocks = idx.reshape(-1, 30)
    ok = bool((blocks == blocks[:, :1]).all())
    print("%s: rows arrive as contiguous 30-row blocks, one sample each: %s" % (name, ok))
    t = pd.to_datetime(df["time"]).to_numpy().reshape(-1, 30)
    gaps = (t[:, 1:] - t[:, :-1]).astype("timedelta64[D]").astype(int)
    print("  within-window day gaps: unique %s" % np.unique(gaps))

pl = pos[pos["time_idx"] == 29].copy()
pl["time"] = pd.to_datetime(pl["time"])
nl = neg[neg["time_idx"] == 29].copy()
nl["time"] = pd.to_datetime(nl["time"])
test_p = pl[pl["time"].dt.year.isin([2021, 2022])]
test_n = nl[nl["time"].dt.year.isin([2021, 2022])]
print("\ntest-year positives %d | test-year negative pool %d | ratio %.2f"
      % (len(test_p), len(test_n), len(test_n) / max(len(test_p), 1)))
print("paper says test = 4107 = 1369 positives + 2738 negatives")
print("\ntest positives burned_area_has quantiles:")
print(test_p["burned_area_has"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.99]).round(1).to_string())
print("\ntest-year month histogram (positives / negatives):")
print(pd.concat([test_p["time"].dt.month.value_counts().sort_index().rename("pos"),
                 test_n["time"].dt.month.value_counts().sort_index().rename("neg")], axis=1).to_string())
