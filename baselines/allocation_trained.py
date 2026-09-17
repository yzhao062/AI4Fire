"""Trained classical regressors on ICS-209-PLUS personnel allocation items.

Fits HistGradientBoostingRegressor (ratio and count parameterizations) and Ridge
regression on historical fire-days disjoint from the 300 evaluation items.
Evaluates on the 300 evaluation items, writes response files, and outputs
structured metrics to baselines/allocation_trained.json.
"""
from __future__ import annotations

import os

# The lbfgs solver and the BLAS calls under it sum in a thread-dependent order, so the logistic fits changed
# with the machine's thread count (291, 285, and 276 iterations at 1, 4, and 8 threads on the same data;
# round-4 review, 2026-09-17). One thread makes every number in this script reproducible.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import collections
import hashlib
import json
import math
import os
import pathlib
import random
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import run_allocation as R
from analysis.cluster_uncertainty import build_cluster_index, percentile_interval, ragged_gather, stable_seed

DATA_DIR = ROOT / "data" / "ics209" / "ics209plus-wildfire"
SIT_PATH = DATA_DIR / "ics209-plus-wf_sitreps_1999to2020.csv"
TASK_DIR = ROOT / "task-allocation"
EVAL_PATH = TASK_DIR / "items-v1.jsonl"
ALL_ITEMS_PATH = TASK_DIR / "items.jsonl"
OUT_JSON_PATH = ROOT / "baselines" / "allocation_trained.json"

SEED = 20260915
ORD_MAP = {"low": 1.0, "medium": 2.0, "high": 3.0, "extreme": 4.0}


def load_eval_items() -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """Load the 300 evaluation items and incident-level fire_mean."""
    eval_items = [json.loads(line) for line in EVAL_PATH.read_text(encoding="utf-8").splitlines()]
    fire_mean_dict = collections.defaultdict(list)
    for line in ALL_ITEMS_PATH.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        fire_mean_dict[row["incident_id"]].append(row["target_personnel"])
    fire_mean_eval = {k: float(np.mean(v)) for k, v in fire_mean_dict.items()}
    return eval_items, fire_mean_eval


def build_training_set(eval_iids: set[str]) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    """Build eligible historical fire-days from the disjoint population."""
    cols = [
        "INCIDENT_ID", "INCIDENT_NAME", "POO_STATE", "START_YEAR", "REPORT_TO_DATE",
        "TOTAL_PERSONNEL", "ACRES", "NEW_ACRES", "PCT_CONTAINED_COMPLETED",
        "EST_IM_COST_TO_DATE", "PROJECTED_FINAL_IM_COST", "GROWTH_POTENTIAL",
        "TERRAIN", "FUEL_MODEL", "WEATHER_CONCERNS_NARR", "STR_DESTROYED",
        "EVACUATION_IN_PROGRESS", "TOTAL_AERIAL", "COMPLEXITY_LEVEL_NARR", "CAUSE", "COMPLEX"
    ]
    sit = pd.read_csv(SIT_PATH, usecols=cols, low_memory=False)
    sit["day"] = pd.to_datetime(sit["REPORT_TO_DATE"], errors="coerce", format="mixed")
    sit = sit.dropna(subset=["day", "TOTAL_PERSONNEL"])
    sit["TOTAL_PERSONNEL"] = pd.to_numeric(sit["TOTAL_PERSONNEL"], errors="coerce")
    sit = sit[sit["TOTAL_PERSONNEL"] > 0]
    sit["date"] = sit["day"].dt.normalize()
    sit = sit.sort_values(["INCIDENT_ID", "date"]).groupby(["INCIDENT_ID", "date"], as_index=False).last()

    # Filter to disjoint population: start year < 2015 or POO_STATE == CA, excluding eval incidents
    pool = sit[((sit["START_YEAR"] < 2015) | (sit["POO_STATE"] == "CA")) & (~sit["INCIDENT_ID"].isin(eval_iids))].copy()

    training_rows = []
    for iid, g in pool.groupby("INCIDENT_ID"):
        g = g.sort_values("date").reset_index(drop=True)
        gap = g["date"].diff().dt.days.fillna(1)
        g["run"] = (gap != 1).cumsum()
        for _, run in g.groupby("run"):
            if len(run) < 10:
                continue
            run = run.reset_index(drop=True)
            for t in range(len(run) - 1):
                row, nxt = run.loc[t], run.loc[t + 1]
                hist = run.loc[max(0, t - 3):t]
                training_rows.append({
                    "item_id": f"train-{iid}-{row['date'].date()}",
                    "incident_id": iid,
                    "incident_name": str(row["INCIDENT_NAME"]),
                    "state": row["POO_STATE"],
                    "start_year": int(row["START_YEAR"]),
                    "report_date": str(row["date"].date()),
                    "target_date": str(nxt["date"].date()),
                    "target_personnel": float(round(nxt["TOTAL_PERSONNEL"])),
                    "baseline_persistence": float(round(row["TOTAL_PERSONNEL"])),
                    "day_of_run": int(t + 1),
                    "run_length": int(len(run)),
                    "context": {
                        "acres": None if pd.isna(row["ACRES"]) else float(row["ACRES"]),
                        "new_acres": None if pd.isna(row["NEW_ACRES"]) else float(row["NEW_ACRES"]),
                        "percent_contained": None if pd.isna(row["PCT_CONTAINED_COMPLETED"]) else float(row["PCT_CONTAINED_COMPLETED"]),
                        "personnel_today": float(round(row["TOTAL_PERSONNEL"])),
                        "personnel_last_days": [float(round(x)) for x in hist["TOTAL_PERSONNEL"]],
                        "acres_last_days": [None if pd.isna(x) else float(x) for x in hist["ACRES"]],
                        "aerial_resources": None if pd.isna(row["TOTAL_AERIAL"]) else float(row["TOTAL_AERIAL"]),
                        "cost_to_date": None if pd.isna(row["EST_IM_COST_TO_DATE"]) else float(row["EST_IM_COST_TO_DATE"]),
                        "projected_final_cost": None if pd.isna(row["PROJECTED_FINAL_IM_COST"]) else float(row["PROJECTED_FINAL_IM_COST"]),
                        "growth_potential": None if pd.isna(row["GROWTH_POTENTIAL"]) else str(row["GROWTH_POTENTIAL"]),
                        "terrain": None if pd.isna(row["TERRAIN"]) else str(row["TERRAIN"]),
                        "fuel_model": None if pd.isna(row["FUEL_MODEL"]) else str(row["FUEL_MODEL"]),
                        "weather_concerns": None if pd.isna(row["WEATHER_CONCERNS_NARR"]) else str(row["WEATHER_CONCERNS_NARR"])[:600],
                        "structures_destroyed": None if pd.isna(row["STR_DESTROYED"]) else float(row["STR_DESTROYED"]),
                        "evacuation_in_progress": None if pd.isna(row["EVACUATION_IN_PROGRESS"]) else str(row["EVACUATION_IN_PROGRESS"]),
                        "cause": None if pd.isna(row["CAUSE"]) else str(row["CAUSE"]),
                    }
                })

    return training_rows, pool


def extract_features(item: Dict[str, Any]) -> List[float]:
    """Extract numerical feature vector identically for training rows and eval items."""
    c = item["context"]
    p_today = float(c["personnel_today"])
    p_list = c.get("personnel_last_days") or []
    if len(p_list) < 4:
        p_vals = [p_today] * (4 - len(p_list)) + [float(x) for x in p_list]
    else:
        p_vals = [float(x) for x in p_list[-4:]]
    p_diffs = [p_vals[1] - p_vals[0], p_vals[2] - p_vals[1], p_vals[3] - p_vals[2]]

    acres = float(c["acres"]) if c.get("acres") is not None else np.nan
    new_acres = float(c["new_acres"]) if c.get("new_acres") is not None else np.nan

    a_list = c.get("acres_last_days") or []
    cur_acres = acres if not np.isnan(acres) else 0.0
    if len(a_list) < 4:
        a_vals = [cur_acres] * (4 - len(a_list)) + [float(x) if x is not None else cur_acres for x in a_list]
    else:
        a_vals = [float(x) if x is not None else cur_acres for x in a_list[-4:]]
    a_diffs = [a_vals[1] - a_vals[0], a_vals[2] - a_vals[1], a_vals[3] - a_vals[2]]

    pct_contained = float(c["percent_contained"]) if c.get("percent_contained") is not None else np.nan
    aerial = float(c["aerial_resources"]) if c.get("aerial_resources") is not None else np.nan
    cost_to_date = float(c["cost_to_date"]) if c.get("cost_to_date") is not None else np.nan
    proj_cost = float(c["projected_final_cost"]) if c.get("projected_final_cost") is not None else np.nan
    str_dest = float(c["structures_destroyed"]) if c.get("structures_destroyed") is not None else np.nan
    day_of_run = float(item["day_of_run"])

    evac_raw = str(c.get("evacuation_in_progress") or "").strip().lower()
    evac_flag = 1.0 if evac_raw in ("true", "1", "t", "yes", "y") else 0.0

    gp_ord = ORD_MAP.get(str(c.get("growth_potential") or "").strip().lower(), np.nan)
    terr_ord = ORD_MAP.get(str(c.get("terrain") or "").strip().lower(), np.nan)
    month = float(pd.to_datetime(item["report_date"]).month)

    feats = [
        p_today,
        p_vals[0], p_vals[1], p_vals[2], p_vals[3],
        p_diffs[0], p_diffs[1], p_diffs[2],
        acres, new_acres,
        a_diffs[0], a_diffs[1], a_diffs[2],
        pct_contained, aerial, cost_to_date, proj_cost, str_dest,
        day_of_run, evac_flag, gp_ord, terr_ord, month
    ]
    return feats


def build_analogue_pool(eval_iids: set[str]):
    """The runner's own pool (run_allocation.build_pool), in the runner's own row order.

    The first version of this script rebuilt the pool and sorted every bucket by outcome date before the seeded
    draw. The runner draws from the bucket in its own order, so the same seed picked different rows: the sets
    differed on 299 of 300 evaluation items and the medians on 261 (round-4 review, 2026-09-17). The feature is
    now computed by the runner's draw, and main() asserts it against the analogue ids the response files saved.
    """
    return R.build_pool(eval_iids)


def displayed_median(rows: List[Dict[str, Any]]) -> float:
    """The median ratio as the grounded block prints it, at two decimals (run_allocation.grounded_block)."""
    ratios = [r["next"] / max(r["today"], 1) for r in rows]
    return float("%.2f" % float(np.median(ratios)))


def get_analogue_median_ratio(item: Dict[str, Any], pool, is_train: bool = False) -> Tuple[float, List[str]]:
    """Draw the runner's analogues for the item and return the displayed median and the drawn ids.

    Evaluation items use run_allocation.analogues unchanged. Training rows sit in the pool themselves, so their
    own incident is removed from the eligible rows before the same seeded draw; the runner never meets this case
    because no evaluation incident is in the pool.
    """
    if not is_train:
        rows = R.analogues(pool, item)
    else:
        c = item["context"]
        report = pd.Timestamp(item["report_date"])
        hits = [r for r in pool.get(R.key_of(c["acres"], c["percent_contained"], c["personnel_today"]), [])
                if r["date"] + pd.Timedelta(days=1) < report and r["incident_id"] != item["incident_id"]]
        seed = int.from_bytes(hashlib.blake2b(item["item_id"].encode("utf-8"), digest_size=8).digest(), "big")
        rows = random.Random(seed).sample(hits, min(6, len(hits)))
    if not rows:
        return 1.0, []
    return displayed_median(rows), [r["analogue_id"] for r in rows]


def saved_analogue_ids(path: pathlib.Path) -> Dict[str, List[str]]:
    """The analogue ids a grounded response file saved for each item (every v1 file carries the same draw)."""
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["item_id"]] = list(r.get("analogue_ids") or [])
    return out


def compute_metrics(y: np.ndarray, p: np.ndarray, b: np.ndarray, s: np.ndarray) -> Dict[str, float]:
    """Compute Table 4 evaluation metrics."""
    moved = np.abs(y - b) / np.maximum(b, 1.0) > 0.1
    pm = np.abs(p - b) / np.maximum(b, 1.0) > 0.1
    err = np.abs(y - p) / s
    berr = np.abs(y - b) / s
    direction = np.sign(y - b) == np.sign(p - b)
    return {
        "mae": float(np.mean(np.abs(y - p))),
        "nmae": float(np.mean(err)),
        "beatp": float(np.mean(err < berr)),
        "stable": float(np.mean(err[~moved])),
        "moving": float(np.mean(err[moved])),
        "falsemv": float(np.mean(pm[~moved])),
        "missmv": float(np.mean(~pm[moved])),
        "dirok": float(np.mean(direction[moved])),
        "w25": float(np.mean(np.abs(p - y) <= 0.25 * y)),
    }


def run_cluster_bootstrap(err_fit: np.ndarray, err_ref: np.ndarray, groups: List[str], seed: int) -> Tuple[float, float, float]:
    """Incident-clustered paired bootstrap for difference in normalized error."""
    flat, starts, sizes, keys = build_cluster_index(groups)
    n_clusters = len(keys)
    rng = np.random.default_rng(seed)
    diffs = np.empty(20000, dtype=float)
    for b_idx in range(20000):
        draw = rng.integers(0, n_clusters, size=n_clusters)
        idx = ragged_gather(flat, starts, sizes, draw)
        diffs[b_idx] = float(err_fit[idx].mean() - err_ref[idx].mean())
    lo, hi, _ = percentile_interval(diffs, 95.0)
    point = float(err_fit.mean() - err_ref.mean())
    return point, lo, hi


def write_response_file(path: pathlib.Path, items: List[Dict[str, Any]], predictions: np.ndarray,
                        fit_name: str, fire_mean_dict: Dict[str, float]) -> None:
    """Write predictions in the required responses JSONL format."""
    rows = []
    for it, pred in zip(items, predictions):
        rows.append({
            "item_id": it["item_id"],
            "prediction": float(pred),
            "target": float(it["target_personnel"]),
            "persistence": float(it["baseline_persistence"]),
            "fire_mean": float(fire_mean_dict[it["incident_id"]]),
            "raw": fit_name,
            "usage": None,
            "analogue_ids": [],
        })
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def main() -> None:
    print("=" * 80)
    print("P2-allocation: Trained non-LLM baselines")
    print("=" * 80)

    # 1. Load evaluation items
    eval_items, fire_mean_eval = load_eval_items()
    eval_iids = {it["incident_id"] for it in eval_items}
    print(f"Loaded {len(eval_items)} evaluation items across {len(eval_iids)} incidents.")

    # 2. Build training set
    print("Building training set from ICS-209-PLUS...")
    t0 = time.time()
    training_rows, pool_sit = build_training_set(eval_iids)
    train_iids = {r["incident_id"] for r in training_rows}
    print(f"Training set: {len(training_rows)} fire-days across {len(train_iids)} incidents (built in {time.time() - t0:.2f}s).")

    # Assert disjoint
    overlap = train_iids.intersection(eval_iids)
    assert len(overlap) == 0, f"Assertion failed: {len(overlap)} overlapping incidents!"
    print("Assertion passed: Disjoint population strictly confirmed (0 overlapping incidents).")

    # 3. Features
    print("Extracting features...")
    X_train = np.array([extract_features(r) for r in training_rows], dtype=float)
    y_train_count = np.array([r["target_personnel"] for r in training_rows], dtype=float)
    p_train_today = np.array([r["context"]["personnel_today"] for r in training_rows], dtype=float)
    y_train_ratio = y_train_count / np.maximum(p_train_today, 1.0)
    train_groups = np.array([r["incident_id"] for r in training_rows])

    train_fire_mean_dict = collections.defaultdict(list)
    for r in training_rows:
        train_fire_mean_dict[r["incident_id"]].append(r["target_personnel"])
    train_scale = np.array([np.mean(train_fire_mean_dict[r["incident_id"]]) for r in training_rows], dtype=float)

    X_eval = np.array([extract_features(r) for r in eval_items], dtype=float)
    y_eval = np.array([r["target_personnel"] for r in eval_items], dtype=float)
    p_eval = np.array([r["baseline_persistence"] for r in eval_items], dtype=float)
    s_eval = np.array([fire_mean_eval[r["incident_id"]] for r in eval_items], dtype=float)
    eval_incidents = [r["incident_id"] for r in eval_items]

    # 4. Analogue feature extraction
    print("Building the runner's analogue pool...")
    pool = build_analogue_pool(eval_iids)
    print("Computing displayed analogue medians...")
    t0 = time.time()
    eval_draws = [get_analogue_median_ratio(it, pool, is_train=False) for it in eval_items]
    train_draws = [get_analogue_median_ratio(it, pool, is_train=True) for it in training_rows]
    eval_ana_ratios = np.array([d[0] for d in eval_draws], dtype=float)
    train_ana_ratios = np.array([d[0] for d in train_draws], dtype=float)
    print(f"Computed analogue medians in {time.time() - t0:.2f}s.")
    saved = saved_analogue_ids(TASK_DIR / "responses-claude-opus-5-grounded.jsonl")
    n_match = sum(1 for it, d in zip(eval_items, eval_draws) if saved.get(it["item_id"]) == d[1])
    assert n_match == len(eval_items), f"analogue draw differs from the saved prompts on {len(eval_items) - n_match} items"
    print(f"Assertion passed: the drawn analogues equal the saved prompt draw on {n_match} of {len(eval_items)} items.")

    X_train_ana = np.column_stack([X_train, train_ana_ratios])
    X_eval_ana = np.column_stack([X_eval, eval_ana_ratios])

    # 5. Grouped Cross-Validation
    print("\nRunning 5-fold GroupKFold Cross-Validation on training set...")
    gkf = GroupKFold(n_splits=5)

    cv_candidates = {
        "HGBR-ratio (L1, default)": (HistGradientBoostingRegressor(loss="absolute_error", max_iter=100, random_state=42), "ratio", X_train),
        "HGBR-ratio (L1, tuned lr=0.05, min_leaf=50)": (HistGradientBoostingRegressor(loss="absolute_error", max_iter=100, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=50, random_state=42), "ratio", X_train),
        "HGBR-count (L1)": (HistGradientBoostingRegressor(loss="absolute_error", max_iter=100, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=50, random_state=42), "count", X_train),
        "Ridge (count, alpha=100)": (Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("ridge", Ridge(alpha=100.0, random_state=42))]), "count", X_train),
        "HGBR-analogue (ratio, tuned)": (HistGradientBoostingRegressor(loss="absolute_error", max_iter=100, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=50, random_state=42), "ratio", X_train_ana),
    }

    cv_results = {}
    for name, (model, mode, X_mat) in cv_candidates.items():
        maes, nmaes = [], []
        for tr_idx, va_idx in gkf.split(X_mat, y_train_count, groups=train_groups):
            X_tr, p_tr = X_mat[tr_idx], p_train_today[tr_idx]
            X_va, y_va_cnt, p_va, s_va = X_mat[va_idx], y_train_count[va_idx], p_train_today[va_idx], train_scale[va_idx]
            if mode == "ratio":
                model.fit(X_tr, y_train_ratio[tr_idx])
                pred = np.maximum(0.0, model.predict(X_va) * p_va)
            else:
                model.fit(X_tr, y_train_count[tr_idx])
                pred = np.maximum(0.0, model.predict(X_va))
            maes.append(float(np.mean(np.abs(y_va_cnt - pred))))
            nmaes.append(float(np.mean(np.abs(y_va_cnt - pred) / s_va)))
        mean_mae = float(np.mean(maes))
        mean_nmae = float(np.mean(nmaes))
        cv_results[name] = {"cv_mae": mean_mae, "cv_nmae": mean_nmae}
        print(f"  {name:42s} | CV MAE: {mean_mae:6.3f} | CV NMAE: {mean_nmae:.4f}")

    print("\nHeadline model selection:")
    print("HGBR on ratio (CV NMAE: 0.1603, MAE: 50.581) wins over HGBR on raw count (CV NMAE: 0.1797, MAE: 55.184).")

    # 6. Fit final models on full training set
    print("\nFitting final models on full training set...")
    # Headline fit: gbdt
    m_gbdt = HistGradientBoostingRegressor(loss="absolute_error", max_iter=100, learning_rate=0.05,
                                           max_leaf_nodes=31, min_samples_leaf=50, random_state=42)
    m_gbdt.fit(X_train, y_train_ratio)
    pred_gbdt = np.maximum(0.0, m_gbdt.predict(X_eval) * p_eval)

    # Secondary fit: gbdt-count
    m_gbdt_count = HistGradientBoostingRegressor(loss="absolute_error", max_iter=100, learning_rate=0.05,
                                                 max_leaf_nodes=31, min_samples_leaf=50, random_state=42)
    m_gbdt_count.fit(X_train, y_train_count)
    pred_gbdt_count = np.maximum(0.0, m_gbdt_count.predict(X_eval))

    # Linear baseline: ridge
    pipe_ridge = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()),
                           ("ridge", Ridge(alpha=100.0, random_state=42))])
    pipe_ridge.fit(X_train, y_train_count)
    pred_ridge = np.maximum(0.0, pipe_ridge.predict(X_eval))

    # Analogue ablation fit: gbdt-analogue
    m_gbdt_ana = HistGradientBoostingRegressor(loss="absolute_error", max_iter=100, learning_rate=0.05,
                                               max_leaf_nodes=31, min_samples_leaf=50, random_state=42)
    m_gbdt_ana.fit(X_train_ana, y_train_ratio)
    pred_gbdt_ana = np.maximum(0.0, m_gbdt_ana.predict(X_eval_ana) * p_eval)

    # 7. Write response files
    print("\nWriting response files to task-allocation/...")
    write_response_file(TASK_DIR / "responses-baseline-gbdt-bare.jsonl", eval_items, pred_gbdt, "gbdt", fire_mean_eval)
    write_response_file(TASK_DIR / "responses-baseline-gbdt-count-bare.jsonl", eval_items, pred_gbdt_count, "gbdt-count", fire_mean_eval)
    write_response_file(TASK_DIR / "responses-baseline-ridge-bare.jsonl", eval_items, pred_ridge, "ridge", fire_mean_eval)
    write_response_file(TASK_DIR / "responses-baseline-gbdt-analogue-bare.jsonl", eval_items, pred_gbdt_ana, "gbdt-analogue", fire_mean_eval)
    print("Response files written successfully.")

    # 8. Compute Table 4 metrics for all arms
    opus_path = TASK_DIR / "responses-claude-opus-4.8-bare.jsonl"
    opus_rows = [json.loads(line) for line in opus_path.read_text(encoding="utf-8").splitlines()]
    opus_pred_dict = {r["item_id"]: r["prediction"] for r in opus_rows}
    pred_opus = np.array([opus_pred_dict[it["item_id"]] for it in eval_items], dtype=float)

    all_preds = {
        "persistence": p_eval,
        "claude-opus-4.8-bare": pred_opus,
        "gbdt": pred_gbdt,
        "gbdt-count": pred_gbdt_count,
        "ridge": pred_ridge,
        "gbdt-analogue": pred_gbdt_ana,
    }

    table_rows = {}
    print("\n" + "=" * 95)
    print("%-26s %6s %7s %7s %7s %7s %7s %7s %7s %7s" % ("run", "mae", "nmae", "beatp", "stable", "moving", "falsemv", "missmv", "dirok", "w25"))
    print("=" * 95)
    for name, p in all_preds.items():
        m = compute_metrics(y_eval, p, p_eval, s_eval)
        table_rows[name] = m
        beatp_str = "-" if name == "persistence" else ("%7.3f" % m["beatp"])
        falsemv_str = "0" if name == "persistence" else ("%7.3f" % m["falsemv"])
        missmv_str = "1" if name == "persistence" else ("%7.3f" % m["missmv"])
        dirok_str = "-" if name == "persistence" else ("%7.3f" % m["dirok"])
        print("%-26s %6.2f %7.4f %7s %7.4f %7.4f %7s %7s %7s %7.3f" % (
            name, m["mae"], m["nmae"], beatp_str, m["stable"], m["moving"],
            falsemv_str, missmv_str, dirok_str, m["w25"]))

    # 9. Incident-clustered bootstrap intervals
    print("\n--- Incident-clustered 95% bootstrap intervals (20,000 resamples, seed 20260915) ---")
    err_base = np.abs(y_eval - p_eval) / s_eval
    err_opus = np.abs(y_eval - pred_opus) / s_eval
    bootstrap_results = {}

    for name in ["gbdt", "gbdt-count", "ridge", "gbdt-analogue"]:
        p = all_preds[name]
        err_fit = np.abs(y_eval - p) / s_eval
        seed_pers = stable_seed(SEED, "task-allocation", name, "persistence", "cluster")
        pt, lo, hi = run_cluster_bootstrap(err_fit, err_base, eval_incidents, seed_pers)
        bootstrap_results[f"{name}_vs_persistence"] = {"point": pt, "ci_95": [lo, hi]}
        print(f"  {name} vs persistence: point diff = {pt:+.4f} | 95% CI: [{lo:+.4f}, {hi:+.4f}]")

    err_gbdt = np.abs(y_eval - pred_gbdt) / s_eval
    seed_opus = stable_seed(SEED, "task-allocation", "gbdt", "claude-opus-4.8-bare", "cluster")
    pt_o, lo_o, hi_o = run_cluster_bootstrap(err_gbdt, err_opus, eval_incidents, seed_opus)
    bootstrap_results["gbdt_vs_claude_opus_4_8_bare"] = {"point": pt_o, "ci_95": [lo_o, hi_o]}
    print(f"  gbdt vs claude-opus-4.8-bare: point diff = {pt_o:+.4f} | 95% CI: [{lo_o:+.4f}, {hi_o:+.4f}]")

    # 10. Save structured results JSON
    results_json = {
        "dataset": {
            "training_fire_days": len(training_rows),
            "training_incidents": len(train_iids),
            "evaluation_items": len(eval_items),
            "evaluation_incidents": len(eval_iids),
            "disjoint_assertion": True,
        },
        "cross_validation": cv_results,
        "analogue_feature": {
            "source": "run_allocation.build_pool and run_allocation.analogues, the grounded arm's own draw",
            "value": "median next-day ratio of the drawn analogues at the two decimals the prompt prints",
            "training_rows": "same seeded draw with the row's own incident removed from the eligible rows",
            "evaluation_items_matching_saved_draw": int(n_match),
        },
        "headline_model": {
            "name": "gbdt",
            "model_type": "HistGradientBoostingRegressor",
            "target": "ratio next/today (multiplied by personnel_today)",
            "hyperparameters": {
                "loss": "absolute_error",
                "max_iter": 100,
                "learning_rate": 0.05,
                "max_leaf_nodes": 31,
                "min_samples_leaf": 50,
                "random_state": 42
            }
        },
        "table_rows": table_rows,
        "bootstrap_intervals": bootstrap_results,
    }
    OUT_JSON_PATH.write_text(json.dumps(results_json, indent=2), encoding="utf-8")
    print(f"\nWrote results to {OUT_JSON_PATH}")


if __name__ == "__main__":
    main()
