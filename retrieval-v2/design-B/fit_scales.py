"""Fit the standardization constants for the family-B features on pool rows dated before 2015-01-01 and freeze them.

Writes scales.json beside this file: per feature, the mean and standard deviation over the pre-2015 rows that have
the feature (NaN ignored). Run once; run.py refits and checks that the frozen file still matches the pool.

    python fit_scales.py            fits from a freshly built pool
"""
import json
import math
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import dev_eval  # noqa: E402

FEATURES = ["pers", "acres", "pct", "day", "change", "new", "aerial"]
FIT_BEFORE = "2015-01-01"


def _isnum(v):
    return v is not None and not (isinstance(v, float) and math.isnan(v))


def raw_row(r):
    """Identical to rule.raw_row; duplicated so this file imports without scales.json existing."""
    return [
        math.log(max(r["today"], 1.0)),
        math.log1p(max(r["acres"], 0.0)) if _isnum(r["acres"]) else math.nan,
        min(max(float(r["pct"]), 0.0), 100.0) if _isnum(r["pct"]) else math.nan,
        math.log(max(r["day_of_run"], 1)),
        math.log(max(r["today"], 1.0) / max(r["prev"], 1.0)) if _isnum(r["prev"]) else math.nan,
        math.log1p(max(r["new_acres"], 0.0)) if _isnum(r["new_acres"]) else math.nan,
        math.log1p(max(r["aerial"], 0.0)) if _isnum(r["aerial"]) else math.nan,
    ]


def fit(pool):
    cutoff = pd.Timestamp(FIT_BEFORE)
    rows = [r for r in pool if r["date"] < cutoff]
    raw = np.array([raw_row(r) for r in rows], dtype=float)
    mean = np.nanmean(raw, axis=0)
    std = np.nanstd(raw, axis=0)
    return {
        "fitted_on": "pool rows dated before %s (%d of %d rows)" % (FIT_BEFORE, len(rows), len(pool)),
        "n_rows": len(rows),
        "features": FEATURES,
        "mean": {f: float(m) for f, m in zip(FEATURES, mean)},
        "std": {f: float(s) for f, s in zip(FEATURES, std)},
        "n_present": {f: int(np.sum(~np.isnan(raw[:, i]))) for i, f in enumerate(FEATURES)},
    }


def main():
    pool = dev_eval.load_pool()
    scales = fit(pool)
    (HERE / "scales.json").write_text(json.dumps(scales, indent=1), encoding="utf-8")
    print(json.dumps(scales, indent=1))


if __name__ == "__main__":
    main()
