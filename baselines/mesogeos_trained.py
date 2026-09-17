"""Trained non-LLM baselines for Mesogeos Track A fire danger prediction.

Trains HistGradientBoostingClassifier and LogisticRegression on 2006-2019 training years,
selects hyperparameters on 2020 validation year, and scores on both the 386-item LLM evaluation
set and the full 2021-2022 holdout (4,120 samples).

Outputs:
  - task-mesogeos/responses-baseline-gbdt-prompt-bare.jsonl
  - task-mesogeos/responses-baseline-gbdt-window-bare.jsonl
  - task-mesogeos/responses-baseline-logit-prompt-bare.jsonl
  - task-mesogeos/responses-baseline-logit-window-bare.jsonl
  - baselines/mesogeos_trained.json
"""
import os

# The lbfgs solver and the BLAS calls under it sum in a thread-dependent order, so the logistic fits changed
# with the machine's thread count (291, 285, and 276 iterations at 1, 4, and 8 threads on the same data;
# round-4 review, 2026-09-17). One thread makes every number in this script reproducible.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import json
import math
import pathlib
import statistics
import sys
import warnings
import zlib

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Ensure repository analysis modules are importable
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
import cluster_uncertainty as cu

DATA_DIR = ROOT / "data" / "mesogeos"
TASK_DIR = ROOT / "task-mesogeos"
OUT_DIR = ROOT / "baselines"

LAG = 30
DYNAMIC = ["d2m", "lai", "lst_day", "lst_night", "ndvi", "rh", "smi", "sp", "ssrd", "t2m", "tp", "wind_speed"]
STATIC = [
    "dem", "roads_distance", "slope", "lc_agriculture", "lc_forest", "lc_grassland",
    "lc_settlement", "lc_shrubland", "lc_sparse_vegetation", "lc_water_bodies",
    "lc_wetland", "population"
]
SEED = 20260915
N_BOOTSTRAP = 20000


def r_sigfig(x):
    """Format to 6 significant figures matching build_items_mesogeos.py."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return float("%.6g" % x)


def r4(x):
    """The four significant figures the bare prompt prints (run_mesogeos.summarise uses %.4g)."""
    return x if x is None or (isinstance(x, float) and math.isnan(x)) else float("%.4g" % x)


def sig_arr(a, fmt):
    """Round every finite entry of an array to the given printf precision, leaving NaN in place."""
    out = np.array(a, dtype=float, copy=True)
    m = np.isfinite(out)
    out[m] = [float(fmt % v) for v in out[m]]
    return out


def row_fmean(d):
    """Mean of the observed values in each row, summed as run_mesogeos.summarise sums them (statistics.fmean)."""
    out = np.full(d.shape[0], np.nan)
    for i, row in enumerate(d):
        vals = [v for v in row if not math.isnan(v)]
        if vals:
            out[i] = statistics.fmean(vals)
    return out


def prompt_features_of_item(it):
    """The 121 prompt numbers of one item, recomputed from the item file the prompt is rendered from."""
    c = it["context"]
    feats = []
    for col in DYNAMIC:
        series = c["daily"][col]
        vals = [v for v in series if v is not None]
        feats += [np.nan if v is None else r4(v) for v in series[-6:]]
        feats += [r4(statistics.fmean(vals)), r4(min(vals)), r4(max(vals))] if vals else [np.nan] * 3
    feats += [np.nan if c["static"].get(s) is None else r4(c["static"][s]) for s in STATIC]
    feats.append(float(pd.Timestamp(c["window_end"]).month))
    return np.array(feats, dtype=float)


def verify_prompt_precision(items_386, X_prompt, order_386):
    """Assert that the 386 evaluation rows of the prompt table equal the numbers the prompt prints."""
    bad = 0
    for it, row_idx in zip(items_386, order_386):
        a, b = X_prompt[row_idx], prompt_features_of_item(it)
        if not np.array_equal(a, b, equal_nan=True):
            bad += 1
    assert bad == 0, f"{bad} of 386 evaluation rows differ from the printed prompt numbers"
    print("Prompt precision confirmed: the 386 evaluation rows equal the 121 numbers the bare prompt prints.")


def load_and_verify_items():
    """Load items.jsonl and get the 386 fold 0 items."""
    items_path = TASK_DIR / "items.jsonl"
    assert items_path.exists(), f"Missing {items_path}"
    all_items = [json.loads(line) for line in items_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    items_386 = [it for it in all_items if it.get("split") == "test" and it.get("fold") == 0]
    assert len(items_386) == 386, f"Expected 386 items in fold 0, got {len(items_386)}"
    return items_386


def verify_traceability(items_386, df_pos, df_neg):
    """Verify that every daily driver value in items_386 matches the source CSV rows exactly."""
    print("Verifying 100% data traceability between items.jsonl and source CSVs...")
    p_sids = df_pos["sample"].iloc[::LAG].to_numpy()
    n_sids = df_neg["sample"].iloc[::LAG].to_numpy()
    p_map = {sid: idx for idx, sid in enumerate(p_sids)}
    n_map = {sid: idx for idx, sid in enumerate(n_sids)}

    mismatches = 0
    for it in items_386:
        sfile = it["source_file"]
        sid = it["source_sample"]
        if sfile == "positives.csv":
            assert sid in p_map, f"Sample {sid} not in {sfile}"
            idx = p_map[sid]
            sub = df_pos.iloc[idx * LAG : (idx + 1) * LAG]
        else:
            assert sid in n_map, f"Sample {sid} not in {sfile}"
            idx = n_map[sid]
            sub = df_neg.iloc[idx * LAG : (idx + 1) * LAG]

        for c in DYNAMIC:
            csv_vals = [r_sigfig(v) for v in sub[c].to_numpy(dtype=float)]
            item_vals = it["context"]["daily"][c]
            if csv_vals != item_vals:
                mismatches += 1
                raise AssertionError(f"Mismatch in item {it['item_id']} feature {c}")
    print(f"Traceability confirmed: 386 items x 12 drivers x 30 days match CSVs with 0 errors.")


def build_tables(df_pos, df_neg):
    """Build prompt and full-window feature tables, labels, years, and sample identifiers."""
    print("Constructing prompt and full-window feature tables...")
    all_prompt = []
    all_window = []
    all_labels = []
    all_years = []
    all_keys = []

    for name, df, label in [("positives.csv", df_pos, 1), ("negatives.csv", df_neg, 0)]:
        n = len(df) // LAG
        sids = df["sample"].to_numpy().reshape(n, LAG)[:, 0]
        times = pd.to_datetime(df["time"].to_numpy().reshape(n, LAG)[:, -1])
        years = times.year.to_numpy()
        months = times.month.to_numpy()

        static_mat = np.hstack([df[c].to_numpy(dtype=float).reshape(n, LAG)[:, 0:1] for c in STATIC])
        daily_mat = {c: df[c].to_numpy(dtype=float).reshape(n, LAG) for c in DYNAMIC}

        # The prompt table carries the numbers as the prompt prints them: the item file keeps six significant
        # figures, the prompt prints four, and the window statistics are computed on the six-figure values.
        prompt_parts = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            for c in DYNAMIC:
                d6 = sig_arr(daily_mat[c], "%.6g")
                last6 = sig_arr(d6[:, -6:], "%.4g")
                mean_val = sig_arr(row_fmean(d6), "%.4g")[:, None]
                min_val = sig_arr(np.nanmin(d6, axis=1), "%.4g")[:, None]
                max_val = sig_arr(np.nanmax(d6, axis=1), "%.4g")[:, None]
                prompt_parts.extend([last6, mean_val, min_val, max_val])

        prompt_mat = np.hstack(prompt_parts + [sig_arr(sig_arr(static_mat, "%.6g"), "%.4g"), months[:, None]])
        window_mat = np.hstack([daily_mat[c] for c in DYNAMIC] + [static_mat, months[:, None]])

        all_prompt.append(prompt_mat)
        all_window.append(window_mat)
        all_labels.append(np.full(n, label, dtype=int))
        all_years.append(years)
        all_keys.extend([(name, int(sid)) for sid in sids])

    X_prompt = np.vstack(all_prompt)
    X_window = np.vstack(all_window)
    y = np.concatenate(all_labels)
    years = np.concatenate(all_years)

    return X_prompt, X_window, y, years, all_keys


def run_cluster_bootstrap(groups, y_true, scores_a, scores_b=None, seed=SEED, resamples=N_BOOTSTRAP):
    """Run cluster bootstrap using the same spatial-temporal block unit as cluster_uncertainty.py."""
    flat, starts, sizes, keys = cu.build_cluster_index(groups)
    n_clusters = len(keys)
    rng = np.random.default_rng(seed)

    boot_a = np.empty(resamples, dtype=float)
    if scores_b is not None:
        boot_diff = np.empty(resamples, dtype=float)

    for b in range(resamples):
        draw = rng.integers(0, n_clusters, size=n_clusters)
        idx = cu.ragged_gather(flat, starts, sizes, draw)
        ap_a = cu.average_precision(y_true[idx], scores_a[idx])
        boot_a[b] = ap_a
        if scores_b is not None:
            ap_b = cu.average_precision(y_true[idx], scores_b[idx])
            boot_diff[b] = ap_a - ap_b

    ci_a = cu.percentile_interval(boot_a, 95.0)
    if scores_b is not None:
        ci_diff = cu.percentile_interval(boot_diff, 95.0)
        return ci_a, ci_diff, n_clusters
    return ci_a, n_clusters


def main():
    OUT_DIR.mkdir(exist_ok=True)

    print("Loading Mesogeos positive and negative CSVs...")
    df_pos = pd.read_csv(DATA_DIR / "positives.csv", low_memory=False)
    df_neg = pd.read_csv(DATA_DIR / "negatives.csv", low_memory=False)

    items_386 = load_and_verify_items()
    verify_traceability(items_386, df_pos, df_neg)

    X_prompt, X_window, y, years, all_keys = build_tables(df_pos, df_neg)

    train_mask = (years >= 2006) & (years <= 2019)
    val_mask = (years == 2020)
    holdout_mask = (years >= 2021) & (years <= 2022)

    n_train = int(train_mask.sum())
    n_val = int(val_mask.sum())
    n_holdout = int(holdout_mask.sum())
    print(f"Dataset split sizes: Train (2006-2019) = {n_train} | Val (2020) = {n_val} | Holdout (2021-2022) = {n_holdout}")

    # Map the 386 items to their positions in the feature tables
    lookup_386 = {(it["source_file"], it["source_sample"]): idx for idx, it in enumerate(items_386)}
    holdout_indices = np.where(holdout_mask)[0]
    idx_in_data_to_386 = {lookup_386[all_keys[i]]: i for i in holdout_indices if all_keys[i] in lookup_386}
    assert len(idx_in_data_to_386) == 386, f"Mapped {len(idx_in_data_to_386)} of 386 items"
    order_386 = [idx_in_data_to_386[i] for i in range(386)]

    y_386 = np.array([it["label"] for it in items_386], dtype=int)
    assert (y[order_386] == y_386).all(), "Label alignment failure between CSV and items.jsonl"
    verify_prompt_precision(items_386, X_prompt, order_386)

    # Build cluster blocks for the 386 items
    groups_386 = []
    for it in items_386:
        c = it["context"]
        date = it["target_date"]
        block = cu.mesogeos_block_key(c["longitude"], c["latitude"], date, 1.0, "month")
        groups_386.append(block)

    # Load gemini-3.1-pro bare responses for comparison
    gem_path = TASK_DIR / "responses-gemini-3.1-pro-bare.jsonl"
    gem_rows = [json.loads(line) for line in gem_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    gem_map = {r["item_id"]: r["probability"] for r in gem_rows}
    scores_gemini = np.array([gem_map[it["item_id"]] for it in items_386], dtype=float)
    auprc_gemini = float(average_precision_score(y_386, scores_gemini))
    gem_seed = cu.stable_seed(SEED, "task-mesogeos", "gemini-3.1-pro", "cluster")
    ci_gemini, n_clusters = run_cluster_bootstrap(groups_386, y_386, scores_gemini, seed=gem_seed)
    print(f"Gemini-3.1-pro bare: AUPRC = {auprc_gemini:.4f}, 95% Cluster CI = [{ci_gemini[0]:.4f}, {ci_gemini[1]:.4f}] ({n_clusters} clusters)")

    # Grid search for HistGradientBoostingClassifier on 2020 validation set
    print("\n--- Tuning HistGradientBoosting on 2020 validation set ---")
    grid_lr = [0.03, 0.05, 0.1, 0.2]
    grid_iter = [50, 100, 150, 200]
    best_hgb = {}

    for feat_name, X_mat in [("prompt", X_prompt), ("window", X_window)]:
        X_tr, y_tr = X_mat[train_mask], y[train_mask]
        X_v, y_v = X_mat[val_mask], y[val_mask]
        best_score = -1.0
        best_cfg = None
        grid_results = {}
        for lr in grid_lr:
            for n_it in grid_iter:
                clf = HistGradientBoostingClassifier(learning_rate=lr, max_iter=n_it, random_state=SEED)
                clf.fit(X_tr, y_tr)
                p_val = clf.predict_proba(X_v)[:, 1]
                score = average_precision_score(y_v, p_val)
                grid_results[f"lr={lr}_iter={n_it}"] = float(score)
                if score > best_score:
                    best_score = score
                    best_cfg = {"learning_rate": lr, "max_iter": n_it}
        best_hgb[feat_name] = {"best_params": best_cfg, "best_val_auprc": float(best_score), "grid": grid_results}
        print(f"Best HGB {feat_name}: lr={best_cfg['learning_rate']}, max_iter={best_cfg['max_iter']} -> Val AUPRC = {best_score:.4f}")

    # Define the four fits
    fits = [
        {
            "name": "gbdt-prompt",
            "feature_set": "prompt",
            "model_class": "HistGradientBoostingClassifier",
            "estimator": HistGradientBoostingClassifier(
                learning_rate=best_hgb["prompt"]["best_params"]["learning_rate"],
                max_iter=best_hgb["prompt"]["best_params"]["max_iter"],
                random_state=SEED
            ),
            "X": X_prompt,
            "params": best_hgb["prompt"]["best_params"]
        },
        {
            "name": "gbdt-window",
            "feature_set": "window",
            "model_class": "HistGradientBoostingClassifier",
            "estimator": HistGradientBoostingClassifier(
                learning_rate=best_hgb["window"]["best_params"]["learning_rate"],
                max_iter=best_hgb["window"]["best_params"]["max_iter"],
                random_state=SEED
            ),
            "X": X_window,
            "params": best_hgb["window"]["best_params"]
        },
        {
            "name": "logit-prompt",
            "feature_set": "prompt",
            "model_class": "LogisticRegression",
            "estimator": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(random_state=SEED, max_iter=1000))
            ]),
            "X": X_prompt,
            "params": {"solver": "lbfgs", "max_iter": 1000, "imputation": "median", "scaling": "standard"}
        },
        {
            "name": "logit-window",
            "feature_set": "window",
            "model_class": "LogisticRegression",
            "estimator": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(random_state=SEED, max_iter=1000))
            ]),
            "X": X_window,
            "params": {"solver": "lbfgs", "max_iter": 1000, "imputation": "median", "scaling": "standard"}
        },
    ]

    print("\n--- Fitting models on 2006-2019 and evaluating ---")
    results = {}
    best_classical_name = None
    best_classical_auprc = -1.0
    best_classical_probs = None

    for fit in fits:
        fname = fit["name"]
        clf = fit["estimator"]
        X_mat = fit["X"]

        # Train on 2006-2019
        clf.fit(X_mat[train_mask], y[train_mask])

        # Evaluate on 386 items
        probs_386 = clf.predict_proba(X_mat[order_386])[:, 1]
        calls_386 = (probs_386 >= 0.5)
        auprc_386 = float(average_precision_score(y_386, probs_386))
        f1_386 = float(f1_score(y_386, calls_386.astype(int), pos_label=1))
        call_rate_386 = float(calls_386.mean())
        brier_386 = float(brier_score_loss(y_386, probs_386))

        # Evaluate on full holdout (4,120 samples)
        probs_holdout = clf.predict_proba(X_mat[holdout_mask])[:, 1]
        calls_holdout = (probs_holdout >= 0.5)
        y_holdout = y[holdout_mask]
        auprc_holdout = float(average_precision_score(y_holdout, probs_holdout))
        f1_holdout = float(f1_score(y_holdout, calls_holdout.astype(int), pos_label=1))
        call_rate_holdout = float(calls_holdout.mean())
        brier_holdout = float(brier_score_loss(y_holdout, probs_holdout))

        # Cluster bootstrap interval
        seed_fit = cu.stable_seed(SEED, "task-mesogeos", f"baseline-{fname}", "cluster")
        ci_fit, _ = run_cluster_bootstrap(groups_386, y_386, probs_386, seed=seed_fit)

        if auprc_386 > best_classical_auprc:
            best_classical_auprc = auprc_386
            best_classical_name = fname
            best_classical_probs = probs_386

        results[fname] = {
            "name": fname,
            "feature_set": fit["feature_set"],
            "model_class": fit["model_class"],
            "hyperparameters": fit["params"],
            "eval_386": {
                "n_items": 386,
                "auprc": round(auprc_386, 4),
                "f1_fire": round(f1_386, 4),
                "call_rate": round(call_rate_386, 4),
                "brier_score": round(brier_386, 4),
                "cluster_95_ci": [round(ci_fit[0], 4), round(ci_fit[1], 4)],
            },
            "eval_full_holdout": {
                "n_items": n_holdout,
                "auprc": round(auprc_holdout, 4),
                "f1_fire": round(f1_holdout, 4),
                "call_rate": round(call_rate_holdout, 4),
                "brier_score": round(brier_holdout, 4),
            }
        }

        print(f"[{fname:13s}] 386 Items: AUPRC={auprc_386:.3f} [{ci_fit[0]:.3f}, {ci_fit[1]:.3f}] | F1={f1_386:.3f} | Call={call_rate_386:.3f} | Brier={brier_386:.3f}")
        print(f"                Full Holdout: AUPRC={auprc_holdout:.3f} | F1={f1_holdout:.3f} | Call={call_rate_holdout:.3f} | Brier={brier_holdout:.3f}")

        # Write response file
        resp_file = TASK_DIR / f"responses-baseline-{fname}-bare.jsonl"
        lines = []
        for i, it in enumerate(items_386):
            row = {
                "item_id": it["item_id"],
                "label": int(it["label"]),
                "target_date": it["target_date"],
                "probability": float(probs_386[i]),
                "call": bool(calls_386[i]),
                "raw": f"{fit['model_class']} {fit['feature_set']}",
                "usage": None,
                "served_model": fit["model_class"]
            }
            lines.append(json.dumps(row))
        resp_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"  Wrote response file: {resp_file.name} ({len(lines)} items)")

    # Paired cluster bootstrap contrast against gemini-3.1-pro bare
    seed_paired = cu.stable_seed(SEED, "task-mesogeos", f"contrast-{best_classical_name}-vs-gemini", "cluster")
    _, ci_paired_diff, _ = run_cluster_bootstrap(
        groups_386, y_386, best_classical_probs, scores_b=scores_gemini, seed=seed_paired
    )
    point_diff = best_classical_auprc - auprc_gemini
    print(f"\nPaired contrast ({best_classical_name} minus gemini-3.1-pro bare):")
    print(f"  Difference in AUPRC = {point_diff:+.3f}, 95% Cluster CI = [{ci_paired_diff[0]:+.3f}, {ci_paired_diff[1]:+.3f}]")

    summary_out = {
        "dataset_sizes": {
            "train_2006_2019": n_train,
            "val_2020": n_val,
            "holdout_2021_2022": n_holdout,
            "eval_386": 386,
            "eval_386_positives": int(y_386.sum()),
            "eval_386_clusters": n_clusters
        },
        "hgb_validation_tuning": best_hgb,
        "fits": results,
        "best_classical_arm": best_classical_name,
        "gemini_3_1_pro_bare": {
            "auprc": round(auprc_gemini, 4),
            "cluster_95_ci": [round(ci_gemini[0], 4), round(ci_gemini[1], 4)]
        },
        "paired_contrast_best_classical_minus_gemini_bare": {
            "diff_auprc": round(point_diff, 4),
            "cluster_95_ci": [round(ci_paired_diff[0], 4), round(ci_paired_diff[1], 4)]
        }
    }

    out_json = OUT_DIR / "mesogeos_trained.json"
    out_json.write_text(json.dumps(summary_out, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote full metrics and configuration to {out_json}")


if __name__ == "__main__":
    main()
