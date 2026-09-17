"""Fit the frozen components of family C on pool rows dated before 2015-01-01.

Writes model.json (two logistic models in raw-feature units, the phase table) and hgb.pkl beside rule.py.
Training rows: pool rows with date < 2015-01-01 and a previous day (day_of_run >= 2), since every item carries
its previous day's count. The move target is |next/today - 1| > 0.1, the harness's definition; the direction
target is next > today among moved rows.

fit(pool, exclude_incidents=...) returns the constants without writing when write=False, which run.py uses for
the leave-development-incidents-out check of the model's own AUC.
"""
import json
import pathlib
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rule as R  # noqa: E402

CUTOFF = pd.Timestamp("2015-01-01")
PHASE_SHRINK = 20.0  # pseudo-count toward the global rate in each phase cell
C_LOGIT = 1.0


def training_rows(pool, exclude_incidents=None, min_run=None):
    excl = set(exclude_incidents or [])
    rows = [r for r in pool if r["date"] < CUTOFF and r["prev"] is not None and r["incident_id"] not in excl]
    if min_run:
        # run length = max day_of_run within the run + 1; a run's rows are consecutive rows of one incident
        run_of, lens, by_inc = {}, {}, {}
        for r in pool:
            by_inc.setdefault(r["incident_id"], []).append(r)
        for iid, rs in by_inc.items():
            start = None
            for r in sorted(rs, key=lambda r: r["date"]):
                if r["day_of_run"] == 1 or start is None:
                    start = r["date"]
                run_of[r["analogue_id"]] = (iid, start)
                lens[(iid, start)] = max(lens.get((iid, start), 0), r["day_of_run"] + 1)
        rows = [r for r in rows if lens[run_of[r["analogue_id"]]] >= min_run]
    return rows


def design(rows):
    X = np.array([R.lin_features(*R.row_quantities(r)) for r in rows], dtype=float)
    Xh = np.array([R.hgb_features(*R.row_quantities(r)) for r in rows], dtype=float)
    ratios = np.array([R.ratio(r) for r in rows])
    moved = np.abs(ratios - 1) > R.MOVE
    up = ratios > 1
    groups = np.array([r["incident_id"] for r in rows])
    return X, Xh, moved, up, groups


def fit_logistic(X, y):
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    clf = LogisticRegression(C=C_LOGIT, max_iter=2000)
    clf.fit((X - mu) / sd, y)
    w = clf.coef_[0] / sd
    b0 = float(clf.intercept_[0] - np.sum(clf.coef_[0] * mu / sd))
    return {"intercept": b0, "weights": w.tolist(), "features": R.LIN_FEATURES}


def fit_phase(rows, moved):
    cells = {}
    g = float(moved.mean())
    for r, m in zip(rows, moved):
        cell = "%s|%s" % (R.band(r["day_of_run"], R.DOR_EDGES), R.band(r["pct"], R.PCT_EDGES))
        n, k = cells.get(cell, (0, 0))
        cells[cell] = (n + 1, k + int(m))
    table = {c: (k + PHASE_SHRINK * g) / (n + PHASE_SHRINK) for c, (n, k) in cells.items()}
    return {"table": table, "global": g, "shrink": PHASE_SHRINK, "counts": {c: n for c, (n, k) in cells.items()}}


def fit_hgb(Xh, y, seed=0):
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=100,
                                         l2_regularization=1.0, random_state=seed)
    clf.fit(Xh, y)
    return clf


def cv_auc(rows, X, Xh, moved, up, groups, folds=5):
    """Out-of-fold AUCs grouped by incident on the training rows themselves."""
    gkf = GroupKFold(n_splits=folds)
    p_lin, p_hgb, p_phase, p_up = np.zeros(len(rows)), np.zeros(len(rows)), np.zeros(len(rows)), np.full(len(rows), np.nan)
    for tr, te in gkf.split(X, moved, groups):
        spec = fit_logistic(X[tr], moved[tr])
        p_lin[te] = 1 / (1 + np.exp(-(spec["intercept"] + X[te] @ np.array(spec["weights"]))))
        ph = fit_phase([rows[i] for i in tr], moved[tr])
        p_phase[te] = [ph["table"].get("%s|%s" % (R.band(rows[i]["day_of_run"], R.DOR_EDGES), R.band(rows[i]["pct"], R.PCT_EDGES)), ph["global"]) for i in te]
        h = fit_hgb(Xh[tr], moved[tr])
        p_hgb[te] = h.predict_proba(Xh[te])[:, 1]
        mtr = tr[moved[tr]]
        su = fit_logistic(X[mtr], up[mtr])
        p_up[te] = 1 / (1 + np.exp(-(su["intercept"] + X[te] @ np.array(su["weights"]))))
    out = {"logistic": roc_auc_score(moved, p_lin), "phase": roc_auc_score(moved, p_phase), "hgb": roc_auc_score(moved, p_hgb),
           "direction_given_move": roc_auc_score(up[moved], p_up[moved]),
           "direction_accuracy_given_move": float(np.mean((p_up[moved] > 0.5) == up[moved]))}
    return out


def fit(pool, exclude_incidents=None, write=True, min_run=None, verbose=True):
    rows = training_rows(pool, exclude_incidents, min_run=min_run)
    X, Xh, moved, up, groups = design(rows)
    move_spec = fit_logistic(X, moved)
    up_spec = fit_logistic(X[moved], up[moved])
    phase = fit_phase(rows, moved)
    h = fit_hgb(Xh, moved)
    spec = {
        "fitted_on": "pool rows dated before %s with a previous day%s%s" % (
            CUTOFF.date(), "" if not exclude_incidents else ", excluding %d incidents" % len(set(exclude_incidents)),
            "" if not min_run else ", runs of at least %d days" % min_run),
        "n_rows": int(len(rows)), "n_incidents": int(len(set(groups))), "move_rate": float(moved.mean()),
        "up_rate_given_move": float(up[moved].mean()),
        "move": move_spec, "up": up_spec, "phase": phase,
        "hgb": {"features": R.HGB_FEATURES, "file": "hgb.pkl", "params": h.get_params()},
    }
    if verbose:
        print("fit: %d rows from %d incidents, move rate %.3f, up|move %.3f" % (len(rows), len(set(groups)), moved.mean(), up[moved].mean()))
        p = 1 / (1 + np.exp(-(move_spec["intercept"] + X @ np.array(move_spec["weights"]))))
        print("  in-sample AUC: logistic %.3f, hgb %.3f" % (roc_auc_score(moved, p), roc_auc_score(moved, h.predict_proba(Xh)[:, 1])))
        for name, w in zip(R.LIN_FEATURES, move_spec["weights"]):
            print("  move  %-20s %+.4f" % (name, w))
        for name, w in zip(R.LIN_FEATURES, up_spec["weights"]):
            print("  up    %-20s %+.4f" % (name, w))
    if write:
        spec["hgb"]["params"] = {k: v for k, v in spec["hgb"]["params"].items() if isinstance(v, (int, float, str, bool, type(None)))}
        (HERE / "model.json").write_text(json.dumps(spec, indent=1), encoding="utf-8")
        with open(HERE / "hgb.pkl", "wb") as f:
            pickle.dump({"move": h, "features": R.HGB_FEATURES}, f)
        R._MODEL = None
        R._HGB = None
    return spec, h


def probs_from_spec(spec, hmodel, items):
    """Model probabilities for items from a spec (without touching the module cache)."""
    X = np.array([R.lin_features(*R.item_quantities(it)) for it in items], dtype=float)
    Xh = np.array([R.hgb_features(*R.item_quantities(it)) for it in items], dtype=float)
    p_move = 1 / (1 + np.exp(-(spec["move"]["intercept"] + X @ np.array(spec["move"]["weights"]))))
    p_up = 1 / (1 + np.exp(-(spec["up"]["intercept"] + X @ np.array(spec["up"]["weights"]))))
    p_phase = np.array([spec["phase"]["table"].get("%s|%s" % (R.band(it["day_of_run"], R.DOR_EDGES), R.band(it["context"]["percent_contained"], R.PCT_EDGES)), spec["phase"]["global"]) for it in items])
    p_hgb = hmodel.predict_proba(Xh)[:, 1]
    return p_move, p_up, p_phase, p_hgb


if __name__ == "__main__":
    sys.path.insert(0, str(HERE.parent))
    from dev_eval import load_pool
    fit(load_pool())
