"""Evaluation harness for candidate analogue-retrieval rules on the allocation task.

A rule is a function  rule(cands, item, rng) -> list of pool rows (at most K), where
  cands  is the list of pool rows the item may draw from: rows from other incidents whose outcome day
         (the row's date plus one day) precedes the item's report day; nothing else is eligible,
  item   is the item dict (dev-items.jsonl or task-allocation/items-v1.jsonl, with "context"),
  rng    is random.Random seeded from the item id exactly as run_allocation.py seeds it.
The rule may read anything in cands and item; it may not read the item's target or anything dated on or
after the report day (cands already respects that).

The analogue-only prediction is persistence times the median next-day ratio of the drawn rows, the same
quantity the paper's "analogue-only rule" row reports, so a rule's dev score is the score of the
information it would hand a model. Items with no analogue fall back to persistence and count as such.

Usage:
    from dev_eval import load_pool, load_items, evaluate, rule_v1
    pool = load_pool(); items = load_items()
    report = evaluate(rule_v1, pool, items, name="v1")       # prints the table row, returns a dict
    evaluate(my_rule, pool, items, name="mine", reference=report)   # adds the paired interval against v1
Run this file directly to print the v1 reference on the development set.
"""
import hashlib
import json
import math
import pathlib
import random
import sys

import numpy as np
import pandas as pd

FIRE_BENCH = pathlib.Path(r"C:\Users\yuezh\PycharmProjects\fire-bench")
HERE = pathlib.Path(__file__).resolve().parent
SIT = FIRE_BENCH / "data" / "ics209" / "ics209plus-wildfire" / "ics209-plus-wf_sitreps_1999to2020.csv"
K = 6
ACRE_EDGES = [100, 1000, 5000, 20000, 100000]
PCT_EDGES = [10, 30, 60, 90]
PERS_EDGES = [25, 75, 200, 500]


def band(value, edges):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "unknown"
    for i, e in enumerate(edges):
        if value < e:
            return i
    return len(edges)


def key_of(acres, pct, pers):
    return (band(acres, ACRE_EDGES), band(pct, PCT_EDGES), band(pers, PERS_EDGES))


def item_seed(item_id):
    return int.from_bytes(hashlib.blake2b(item_id.encode("utf-8"), digest_size=8).digest(), "big")


def load_items(path=None):
    path = pathlib.Path(path) if path else HERE / "dev-items.jsonl"
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]


def load_pool(exclude_incidents=None):
    """Consecutive-day pairs from fires that started before 2015 or in California, never an evaluation incident.

    Richer than run_allocation.build_pool: each row also carries the personnel of the day before (prev, or None on
    a run's first day), the acres of the day before, the day index within its run, the state, and the start year,
    so that a candidate rule can match on recent dynamics. today, next, and the band key are identical to v1's.
    """
    eval_items = load_items(FIRE_BENCH / "task-allocation" / "items-v1.jsonl")
    excl = {i["incident_id"] for i in eval_items} | set(exclude_incidents or [])
    cols = ["INCIDENT_ID", "POO_STATE", "START_YEAR", "REPORT_TO_DATE", "TOTAL_PERSONNEL", "ACRES", "NEW_ACRES",
            "PCT_CONTAINED_COMPLETED", "TOTAL_AERIAL", "GROWTH_POTENTIAL", "TERRAIN", "CAUSE", "STR_DESTROYED"]
    sit = pd.read_csv(SIT, usecols=cols, low_memory=False)
    sit["date"] = pd.to_datetime(sit["REPORT_TO_DATE"], errors="coerce", format="mixed").dt.normalize()
    sit["TOTAL_PERSONNEL"] = pd.to_numeric(sit["TOTAL_PERSONNEL"], errors="coerce")
    sit = sit.dropna(subset=["date", "TOTAL_PERSONNEL"])
    sit = sit[(sit["TOTAL_PERSONNEL"] > 0) & (~sit["INCIDENT_ID"].isin(excl))]
    sit = sit[(sit["START_YEAR"] < 2015) | (sit["POO_STATE"] == "CA")]
    sit = sit.sort_values(["INCIDENT_ID", "date"]).groupby(["INCIDENT_ID", "date"], as_index=False).last()
    f = lambda v: None if pd.isna(v) else float(v)
    s = lambda v: None if pd.isna(v) else str(v)
    pool = []
    for iid, g in sit.groupby("INCIDENT_ID"):
        g = g.sort_values("date").reset_index(drop=True)
        gaps = g["date"].diff().dt.days.fillna(1)
        run_start = 0
        for t in range(len(g) - 1):
            if gaps.iloc[t] != 1:
                run_start = t
            if gaps.iloc[t + 1] != 1:
                continue
            row, nxt = g.loc[t], g.loc[t + 1]
            prev = g.loc[t - 1] if t > 0 and gaps.iloc[t] == 1 else None
            pool.append({
                "analogue_id": "%s@%s" % (iid, row["date"].date()),
                "incident_id": iid, "state": row["POO_STATE"], "start_year": int(row["START_YEAR"]),
                "date": row["date"],
                "today": float(row["TOTAL_PERSONNEL"]), "next": float(nxt["TOTAL_PERSONNEL"]),
                "prev": None if prev is None else float(prev["TOTAL_PERSONNEL"]),
                "acres": f(row["ACRES"]), "acres_prev": None if prev is None else f(prev["ACRES"]),
                "new_acres": f(row["NEW_ACRES"]), "pct": f(row["PCT_CONTAINED_COMPLETED"]),
                "pct_prev": None if prev is None else f(prev["PCT_CONTAINED_COMPLETED"]),
                "aerial": f(row["TOTAL_AERIAL"]), "growth_potential": s(row["GROWTH_POTENTIAL"]),
                "terrain": s(row["TERRAIN"]), "cause": s(row["CAUSE"]), "structures": f(row["STR_DESTROYED"]),
                "day_of_run": int(t - run_start + 1),
            })
            pool[-1]["key"] = key_of(pool[-1]["acres"], pool[-1]["pct"], pool[-1]["today"])
    return pool


def candidates(pool, item):
    """Rows the item may draw from: other incidents, outcome day before the report day."""
    report = pd.Timestamp(item["report_date"])
    cutoff = report - pd.Timedelta(days=1)
    return [r for r in pool if r["incident_id"] != item["incident_id"] and r["date"] < cutoff]


def rule_v1(cands, item, rng):
    """The paper's rule: same acres, containment, and personnel band; up to six drawn at random."""
    c = item["context"]
    key = key_of(c["acres"], c["percent_contained"], c["personnel_today"])
    hits = [r for r in cands if r["key"] == key]
    return rng.sample(hits, min(K, len(hits)))


def ratio(r):
    return r["next"] / max(r["today"], 1)


def evaluate(rule, pool, items, name="rule", reference=None, resamples=2000, seed=20260915, quiet=False):
    """Score the analogue-only prediction a rule produces. Returns a dict; prints one table row unless quiet."""
    preds, pmove, n_drawn, med_ratio = [], [], [], []
    drawn_ids = {}
    for it in items:
        cands = candidates(pool, it)
        rows = rule(cands, it, random.Random(item_seed(it["item_id"])))
        assert len(rows) <= 12, "a rule may not draw more than twelve analogues"
        for r in rows:
            assert r["incident_id"] != it["incident_id"] and r["date"] < pd.Timestamp(it["report_date"]) - pd.Timedelta(days=1), "ineligible analogue"
        drawn_ids[it["item_id"]] = [r["analogue_id"] for r in rows]
        base = it["baseline_persistence"]
        if rows:
            ratios = [ratio(r) for r in rows]
            m = float(np.median(ratios))
            preds.append(base * m)
            med_ratio.append(m)
            pmove.append(float(np.mean([abs(x - 1) > 0.1 for x in ratios])))
        else:
            preds.append(base)
            med_ratio.append(1.0)
            pmove.append(float("nan"))
        n_drawn.append(len(rows))
    y = np.array([it["target_personnel"] for it in items])
    b = np.array([it["baseline_persistence"] for it in items])
    s = np.array([it["fire_mean"] for it in items])
    p = np.array(preds)
    n_drawn = np.array(n_drawn)
    pmove = np.array(pmove)
    moved = np.abs(y - b) / np.maximum(b, 1) > 0.1
    pred_moved = np.abs(p - b) / np.maximum(b, 1) > 0.1
    err = np.abs(y - p) / s
    actual_ratio = y / np.maximum(b, 1)
    has = ~np.isnan(pmove)
    # movement prediction quality: Brier of the share of analogues that moved, and AUC against the actual move
    brier = float(np.mean((pmove[has] - moved[has]) ** 2)) if has.any() else float("nan")
    auc = float("nan")
    if has.any() and moved[has].any() and (~moved[has]).any():
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(moved[has], pmove[has]))
    out = {
        "name": name, "items": len(items),
        "coverage_1": float(np.mean(n_drawn >= 1)), "coverage_3": float(np.mean(n_drawn >= 3)),
        "mean_drawn": float(n_drawn.mean()),
        "nmae": float(err.mean()),
        "nmae_persistence": float((np.abs(y - b) / s).mean()),
        "beats_persistence": float(np.mean(np.abs(y - p) < np.abs(y - b))),
        "stable": float(err[~moved].mean()), "moving": float(err[moved].mean()),
        "false_move": float(pred_moved[~moved].mean()), "missed_move": float((~pred_moved[moved]).mean()),
        "direction_right": float((np.sign(y - b) == np.sign(p - b))[moved].mean()),
        "within_25": float(np.mean(np.abs(y - p) / np.maximum(y, 1) <= 0.25)),
        "move_brier": brier, "move_auc": auc,
        "median_ratio_iqr": [float(np.quantile(med_ratio, 0.25)), float(np.quantile(med_ratio, 0.75))],
        "actual_ratio_iqr": [float(np.quantile(actual_ratio, 0.25)), float(np.quantile(actual_ratio, 0.75))],
        "errors": err.tolist(), "drawn_ids": drawn_ids,
    }
    if reference is not None:
        # paired incident-cluster bootstrap on the difference in normalized error, rule minus reference
        ref = np.array(reference["errors"])
        groups = [it["incident_id"] for it in items]
        keys = sorted(set(groups))
        idx = {k: [] for k in keys}
        for i, g in enumerate(groups):
            idx[g].append(i)
        members = [np.array(idx[k]) for k in keys]
        rng = np.random.default_rng(seed)
        diffs = []
        for _ in range(resamples):
            draw = rng.integers(0, len(keys), size=len(keys))
            sel = np.concatenate([members[d] for d in draw])
            diffs.append(float(np.mean(err[sel] - ref[sel])))
        out["diff_vs_reference"] = float(err.mean() - ref.mean())
        out["diff_ci"] = [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]
    if not quiet:
        print(fmt_row(out))
    return out


HEADER = "%-22s %6s %6s %6s %6s %6s %6s %6s %6s %6s %6s %6s %s" % (
    "rule", "cov3", "nMAE", "beats", "stable", "moving", "falsmv", "missmv", "dirok", "w25", "mBrier", "mAUC", "med-ratio IQR | actual IQR | diff vs ref [95%]")


def fmt_row(o):
    d = ""
    if "diff_ci" in o:
        d = " | %+.3f [%+.3f, %+.3f]" % (o["diff_vs_reference"], o["diff_ci"][0], o["diff_ci"][1])
    return "%-22s %6.2f %6.3f %6.2f %6.3f %6.3f %6.2f %6.2f %6.2f %6.2f %6.3f %6.3f [%.2f, %.2f] | [%.2f, %.2f]%s" % (
        o["name"], o["coverage_3"], o["nmae"], o["beats_persistence"], o["stable"], o["moving"], o["false_move"],
        o["missed_move"], o["direction_right"], o["within_25"], o["move_brier"], o["move_auc"],
        o["median_ratio_iqr"][0], o["median_ratio_iqr"][1], o["actual_ratio_iqr"][0], o["actual_ratio_iqr"][1], d)


if __name__ == "__main__":
    pool = load_pool()
    items = load_items()
    print("pool rows: %d | dev items: %d | persistence nMAE on dev: %.3f" % (
        len(pool), len(items), float(np.mean([abs(i["target_personnel"] - i["baseline_persistence"]) / i["fire_mean"] for i in items]))))
    print(HEADER)
    ref = evaluate(rule_v1, pool, items, name="v1 (paper)")
    json.dump({k: v for k, v in ref.items() if k not in ("errors", "drawn_ids")}, open(HERE / "v1-dev-reference.json", "w"), indent=1)
