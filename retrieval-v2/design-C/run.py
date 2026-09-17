"""Family C development run: build the pool once, fit the frozen components, evaluate every variant and v1.

Usage: python run.py            (about 25 s for the pool, then a few seconds per variant)
Prints the harness table with the paired interval against v1, the fitted models' own AUC for P(move) on the
development items (frozen fit, and a refit that leaves the development incidents out of the training rows), and
writes results.json beside this file.
"""
import json
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from sklearn.metrics import roc_auc_score  # noqa: E402

from dev_eval import HEADER, evaluate, load_items, load_pool, rule_v1  # noqa: E402
import fit as F  # noqa: E402
import rule as R  # noqa: E402

MIN_RUN = 10  # training rows come from runs of at least ten reports, the population the items are built from


def main():
    t0 = time.time()
    pool = load_pool()
    items = load_items()
    print("pool rows: %d | dev items: %d | pool built in %.0fs" % (len(pool), len(items), time.time() - t0))

    # frozen fit: pre-2015 rows only, written to model.json and hgb.pkl
    spec, h = F.fit(pool, write=True, min_run=MIN_RUN)
    y = np.array([i["target_personnel"] for i in items])
    b = np.array([i["baseline_persistence"] for i in items])
    r = y / np.maximum(b, 1)
    moved = np.abs(r - 1) > R.MOVE
    up = r > 1
    pm, pu, pp, ph = F.probs_from_spec(spec, h, items)
    dev_inc = {i["incident_id"] for i in items}
    spec2, h2 = F.fit(pool, exclude_incidents=dev_inc, write=False, min_run=MIN_RUN, verbose=False)
    pm2, pu2, pp2, ph2 = F.probs_from_spec(spec2, h2, items)
    model_auc = {
        "frozen": {"logistic": roc_auc_score(moved, pm), "phase": roc_auc_score(moved, pp), "hgb": roc_auc_score(moved, ph),
                   "direction_given_move": roc_auc_score(up[moved], pu[moved]),
                   "direction_accuracy_given_move": float(np.mean((pu[moved] > 0.5) == up[moved]))},
        "dev_incidents_left_out": {"logistic": roc_auc_score(moved, pm2), "phase": roc_auc_score(moved, pp2), "hgb": roc_auc_score(moved, ph2),
                                   "direction_given_move": roc_auc_score(up[moved], pu2[moved]),
                                   "direction_accuracy_given_move": float(np.mean((pu2[moved] > 0.5) == up[moved]))},
        "dev_move_rate": float(moved.mean()), "mean_p_move_logistic": float(pm.mean()), "mean_p_move_phase": float(pp.mean()),
        "mean_p_move_hgb": float(ph.mean()),
    }
    print("model AUC for P(move) on development items (frozen fit):        " + ", ".join("%s %.3f" % kv for kv in model_auc["frozen"].items()))
    print("model AUC for P(move) on development items (dev incidents out): " + ", ".join("%s %.3f" % kv for kv in model_auc["dev_incidents_left_out"].items()))
    rows = F.training_rows(pool, min_run=MIN_RUN)
    X, Xh, mv, upr, groups = F.design(rows)
    cv = F.cv_auc(rows, X, Xh, mv, upr, groups)
    model_auc["grouped_cv_on_training_rows"] = cv
    print("grouped 5-fold CV AUC on the training rows:                    " + ", ".join("%s %.3f" % kv for kv in cv.items()))

    print(HEADER)
    ref = evaluate(rule_v1, pool, items, name="v1 (paper)")
    results = {"v1": {k: v for k, v in ref.items() if k not in ("errors", "drawn_ids")}}
    timing = {}
    for name, fn in R.RULES.items():
        t1 = time.time()
        out = evaluate(fn, pool, items, name=name, reference=ref)
        timing[name] = time.time() - t1
        results[name] = {k: v for k, v in out.items() if k not in ("errors", "drawn_ids")}
    print("seconds per variant (600 items, pool built):", {k: round(v, 1) for k, v in timing.items()})
    print("frozen variant:", R.FROZEN)
    json.dump({"results": results, "model_auc": model_auc, "timing": timing, "frozen": R.FROZEN, "min_run": MIN_RUN},
              open(HERE / "results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
