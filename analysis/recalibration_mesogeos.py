#!/usr/bin/env python
"""Fire-danger calibration after a prior-shift correction and a cross-fitted recalibration.

Backs the appendix subsection "Calibration After a Prior-Shift Correction and Recalibration"
(app:recalibration) with every cell of Table tab:recalibration, the fire-danger paragraph and insight I3 of the
results, Key Finding (iii) of the introduction, and the prior-shift sentence of the limitations. No model is
called: every number comes from the stored responses.

Population: the 386 Mesogeos Track A items the paper scores (test split, fold 0), 131 fires, sample positive
rate 0.339, in 352 blocks (1-degree cell by calendar month of the target date, the paper's clustering unit,
cluster_uncertainty.mesogeos_block_key). Series: the twelve core-model runs (six models, bare and grounded)
and the calendar-month prior of calibration_mesogeos.py, fit on the 2006 to 2019 training years.

Versions of every series
  stated        the stored probability; reproduces Table tab:calibration.
  shifted       the label-shift (prior-shift) adjustment of Saerens, Latinne, and Decaestecker (2002): one
                constant delta added to every log-odds, chosen so that the adjusted probabilities average the
                sample rate (the fixed point of their EM with the target held at 0.339). A stated 0 or 1 stays
                put. The effective prior is the source prior that delta implies, expit(logit(0.339) - delta).
  recalibrated  isotonic regression of the label on the stated probability, fit on four folds and applied to
                the fifth; folds from GroupKFold(5, shuffle=True, random_state=7) over the blocks.
Two variants of the shift are sensitivities: the training-year positive share as the target (no test label;
calibration_mesogeos.TRAIN_PRIOR_MEAN, 0.331), and the run's mean stated probability as its source prior.

Computed
  1. Prompt check: renders the 772 paper prompts (386 items, two arms) and the 772 bare paraphrase prompts
     with run_mesogeos.render and render_variant, and searches them for base-rate wording.
  2. Response check: every core-model row re-parses to its stored fields and carries a probability; omitted
     yes or no answers per run.
  3. Metrics per series and version: ECE (ten equal-width bins), Brier score, binned reliability and
     resolution (calibration_mesogeos.compute_calibration_and_murphy), AUPRC, and the exact CORP decomposition
     of the Brier score (Dimitriadis, Gneiting, and Jordan 2021) into MCB, DSC, and UNC.
  4. Attribution of each run's stated Brier gap to the prior: the base-rate part the shift removes, residual
     miscalibration, and the discrimination deficit, which sum to the gap exactly.
  5. Paired block bootstrap, one draw set per run for every statistic: Brier minus prior (stated, shifted and
     its two variants, recalibrated with the cross-fit refit inside every resample and folds fixed per block,
     and recalibrated conditional on the cross-fitted predictions), and DSC run minus prior.
  6. ECE chance band of each shifted forecast (Broecker and Smith 2007): the 95th percentile of ECE when the
     labels are drawn from the forecast itself, plainly and with the shift refit to each simulated sample.
  7. Fold-draw dependence of the recalibration: ECE over 200 fold draws, and the refit-bootstrap verdict on
     Brier minus prior over 20 fold draws (10,000 resamples each).
  8. Call rates as scored (run_mesogeos.score) and after thresholding the shifted probabilities at one half.
  9. When data/mesogeos/{positives,negatives}.csv are present (fetch_mesogeos.py), month_prior.py re-derives
     the calendar-month prior from the training rows; otherwise that check is skipped with a note.

Seeds. Each Monte Carlo part keeps the seed recipe of the run whose numbers the paper prints, so every interval
and count reproduces draw for draw (cluster_uncertainty.stable_seed adds the crc32 of the joined parts to the
base seed):
  bootstrap         stable_seed(20260915, "task-mesogeos", "<stem>/<condition>", "calibration-r9")
  20 fold draws     stable_seed(20260915, "verify-foldseed", "<label>|<condition>", <fold seed>)
  ECE chance band   stable_seed(314, "verify-band2", "<label>|<condition>")
where <label> is the models.py label ("prior" for the calendar-month prior). The "verify-" tags and base 314
are those of the verification run whose fold-draw counts and band verdicts the paper prints; the band verdicts
are the same at base seeds 20260915 and 7. The recalibration folds keep their own seed, 7.

Output: analysis/recalibration_mesogeos.json (full precision). Stdout: the table in the paper's layout and the
numbers the text quotes.

Usage (about 4.5 minutes on one core of an Apple M5 Pro, about 4 of them in the 20 fold draws):
    python analysis/recalibration_mesogeos.py
    python analysis/recalibration_mesogeos.py --resamples 20000 --seed 20260915
"""
import argparse
import contextlib
import io
import json
import pathlib
import re
import runpy
import sys
import time

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import GroupKFold

ROOT = pathlib.Path(__file__).resolve().parent.parent
HERE = ROOT / "analysis"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import calibration_mesogeos as cm  # noqa: E402
import cluster_uncertainty as cu  # noqa: E402
import models  # noqa: E402
import run_mesogeos as rm  # noqa: E402

OUT = HERE / "recalibration_mesogeos.json"
PRIOR = "calendar_month_prior"
TABLE_VERSIONS = (("stated", "As stated"), ("shifted", "Prior-shifted to the sample positive rate"),
                  ("recalibrated", "Recalibrated by cross-fitted isotonic regression"))
N_FOLDS = 5
FOLD_SEED = 7                 # the recalibration's fold split, as printed in the paper
BASE_SEED = 20260915          # house bootstrap seed (Appendix J; cluster_uncertainty.py)
RESAMPLES = 20000
# Fold sizes of the seed-7 split that the paper's recalibrated numbers used (scikit-learn 1.9.1). GroupKFold gained
# shuffle in scikit-learn 1.6; a later change to its shuffling would move every recalibrated number, so the script
# stops rather than print different values under the same seed.
EXPECTED_FOLD_SIZES = (77, 79, 81, 72, 77)
CI = 95.0
ECE_FOLD_DRAWS = 200          # GroupKFold random_state 0 to 199
BRIER_FOLD_DRAWS = 20         # GroupKFold random_state 0 to 19
BRIER_FOLD_RESAMPLES = 10000
BAND_DRAWS = 20000
BAND_BASE_SEED = 314
BAND_CHUNK = 2000
TIE_TOL = 1e-9
BIN_EDGES = np.linspace(0.0, 1.0, 11)[1:-1]  # inner edges, binned as compute_calibration_and_murphy

# Words and signs that could carry a base rate into a prompt, searched case-insensitively.
BASE_RATE_PATTERNS = {
    "prevalence": r"prevalen", "base rate": r"base[ -]?rate", "percent sign": r"%", "percent word": r"\bpercent",
    "proportion": r"proportion", "frequency": r"frequen", "prior": r"\bprior\b", "subsample": r"subsampl|sub-sampl",
    "balance": r"balanc", "positive rate": r"positive", "odds": r"\bodds\b", "one in / out of": r"\bone in\b|\bout of\b",
    "rare / common": r"\brare\b|\bcommon\b|\bseldom\b|\bunlikely\b",
}


# ------------------------------------------------------------------------------------------------
# data
# ------------------------------------------------------------------------------------------------
def load():
    """Items in items.jsonl order, labels, blocks, the calendar-month prior, and the twelve core runs.

    Response rows are joined to the items by item_id; every file must hold each of the 386 items once, with the
    same label, and a probability on every row.
    """
    items = cm.load_items()
    assert len(items) == 386, len(items)
    y = np.array([int(it["label"]) for it in items], dtype=np.int64)
    blocks = [cu.mesogeos_block_key(it["context"]["longitude"], it["context"]["latitude"], it["target_date"], 1.0, "month")
              for it in items]
    assert None not in blocks
    months = [int(it["context"]["window_end"][5:7]) for it in items]
    series = {PRIOR: np.array([cm.TRAIN_PRIOR_BY_MONTH.get(m, cm.TRAIN_PRIOR_MEAN) for m in months], dtype=float)}
    runs = {}
    for m in models.models(tier="core", task="mesogeos"):
        for cond in ("bare", "grounded"):
            path = cm.TASK / f"responses-{m.stem}-{cond}.jsonl"
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            by_id = {r["item_id"]: r for r in rows}
            assert len(by_id) == len(rows) == len(items) and set(by_id) == {it["item_id"] for it in items}, path.name
            rows = [by_id[it["item_id"]] for it in items]
            assert all(int(r["label"]) == int(it["label"]) for r, it in zip(rows, items)), path.name
            assert all(r["probability"] is not None for r in rows), f"missing probability in {path.name}"
            key = f"{m.stem}/{cond}"
            runs[key] = {"model": m.stem, "label": m.label, "condition": cond, "weights": m.weights,
                         "seed_label": f"{m.label}|{cond}", "rows": rows}
            series[key] = np.array([r["probability"] for r in rows], dtype=float)
    return items, y, blocks, series, runs


def month_prior_check():
    """Re-derive the calendar-month prior with month_prior.py and compare it with calibration_mesogeos.py.

    month_prior.py reads the raw training rows in data/mesogeos (not redistributed; fetch_mesogeos.py fetches
    them), so the check is skipped with a note when they are absent. Its own printout is kept in the JSON.
    """
    csvs = [ROOT / "data" / "mesogeos" / name for name in ("positives.csv", "negatives.csv")]
    if not all(p.exists() for p in csvs):
        return {"status": "skipped: data/mesogeos/positives.csv and negatives.csv not found (fetch_mesogeos.py)"}
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        g = runpy.run_path(str(ROOT / "month_prior.py"), run_name="month_prior")
    prior, train = g["prior"], g["train"]
    diff = max(abs(float(prior[m]) - cm.TRAIN_PRIOR_BY_MONTH[int(m)]) for m in prior.index)
    assert set(int(m) for m in prior.index) == set(cm.TRAIN_PRIOR_BY_MONTH) and diff < 1e-12, diff
    return {
        "status": "ok",
        "training_samples": int(len(train)),
        "training_positives": int(train["label"].sum()),
        "training_positive_share": float(train["label"].mean()),
        "calibration_mesogeos_train_prior_mean": cm.TRAIN_PRIOR_MEAN,
        "max_abs_diff_by_month_vs_calibration_mesogeos": diff,
        "month_prior_stdout": printed.getvalue().splitlines(),
    }


def prompt_check(items):
    """Render every fire-danger prompt the runs saw and count base-rate wording in it.

    The stored rows hold no request text, so run_mesogeos.render and render_variant are the only source of the
    prompt. p0 is the paper's prompt in both arms; the grounded arm adds the monthly climatology, keyed by the
    window-end month as the runner keys it. p1 and p2 are the two bare paraphrases.
    """
    clim = rm.climatology()
    hits = {name: dict.fromkeys(BASE_RATE_PATTERNS, 0) for name in ("p0_bare", "p0_grounded", "p1_bare", "p2_bare")}
    drivers, clim_lines = [], []
    for it in items:
        prompts = {"p0_bare": rm.render(it), "p0_grounded": rm.render(it, clim.get(it["context"]["window_end"][5:7])),
                   "p1_bare": rm.render_variant(it, "p1"), "p2_bare": rm.render_variant(it, "p2")}
        for name, msgs in prompts.items():
            text = "\n".join(m["content"] for m in msgs)
            for pattern, rx in BASE_RATE_PATTERNS.items():
                hits[name][pattern] += len(re.findall(rx, text, flags=re.I))
        bare = sum(line.startswith("- ") for line in prompts["p0_bare"][1]["content"].splitlines())
        grounded = sum(line.startswith("- ") for line in prompts["p0_grounded"][1]["content"].splitlines())
        drivers.append(bare)
        clim_lines.append(grounded - bare)
    return {
        "items": len(items),
        "paper_prompts": 2 * len(items),
        "paraphrase_prompts": 2 * len(items),
        "system_message_p0": rm.SYSTEM,
        "driver_lines_per_prompt": [min(drivers), max(drivers)],
        "climatology_lines_per_grounded_prompt": [min(clim_lines), max(clim_lines)],
        "pattern_hits": hits,
        "total_hits": int(sum(v for h in hits.values() for v in h.values())),
    }


def response_check(runs, y):
    """Parseable probabilities and omitted yes or no answers, per run (run_mesogeos.parse on the raw text)."""
    out = {}
    for key, r in runs.items():
        rows = r["rows"]
        omitted = np.array([row["call"] is None for row in rows])
        out[key] = {
            "rows": len(rows),
            "rows_without_probability": sum(row["probability"] is None for row in rows),
            "raw_reparse_mismatches": sum(rm.parse(row.get("raw")) != (row["probability"], row["call"]) for row in rows),
            "omitted_yes_no": int(omitted.sum()),
            "omitted_share_fire": float(omitted[y == 1].mean()),
            "omitted_share_no_fire": float(omitted[y == 0].mean()),
        }
    return out


# ------------------------------------------------------------------------------------------------
# metrics
# ------------------------------------------------------------------------------------------------
def metrics(y, p):
    """Published calibration terms plus the bin-free CORP decomposition (PAV is the in-sample isotonic fit)."""
    c = cm.compute_calibration_and_murphy(y, p)
    pav = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0, increasing=True).fit(p, y).predict(p)
    brier = float(np.mean((p - y) ** 2))
    brier_pav = float(np.mean((pav - y) ** 2))
    unc = c["murphy"]["uncertainty"]
    return {
        "mean_p": c["mean_predicted"], "ece": c["ece_equal_width"], "brier": brier,
        "rel": c["murphy"]["calibration"], "res": c["murphy"]["resolution"], "unc": unc,
        "auprc": float(average_precision_score(y, p)),
        "corp_mcb": brier - brier_pav, "corp_dsc": unc - brier_pav,
    }


# ------------------------------------------------------------------------------------------------
# prior shift
# ------------------------------------------------------------------------------------------------
def adjust(p, delta):
    """Add delta to every log-odds; a stated 0 or 1 has odds 0 or infinity and stays put."""
    out = np.array(p, dtype=float, copy=True)
    inner = (p > 0.0) & (p < 1.0)
    out[inner] = expit(logit(p[inner]) + delta)
    return out


def delta_mean_match(p, target):
    """Log-odds shift at which the adjusted probabilities average the target (the Saerens et al. EM fixed point)."""
    return float(brentq(lambda d: float(np.mean(adjust(p, d))) - target, -40.0, 40.0, xtol=1e-14, rtol=1e-15, maxiter=500))


def saerens_em_forward(p, pi_s, iters=2000, tol=1e-13):
    """Run the Saerens et al. EM for the target prior from a given source prior (a check: it must return 0.339)."""
    pi_t = float(np.mean(p))
    for _ in range(iters):
        new = float(np.mean(adjust(p, logit(pi_t) - logit(pi_s))))
        done = abs(new - pi_t) < tol
        pi_t = new
        if done:
            break
    return pi_t


# ------------------------------------------------------------------------------------------------
# cross-fitted recalibration
# ------------------------------------------------------------------------------------------------
def make_folds(y, blocks, seed):
    """Fold of every item from GroupKFold over the blocks; asserts that no block spans two folds."""
    gkf = GroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    fold = np.full(len(y), -1, dtype=np.int64)
    for k, (_train, test) in enumerate(gkf.split(np.zeros(len(y)), y, groups=blocks)):
        fold[test] = k
    assert (fold >= 0).all()
    seen = {}
    for b, f in zip(blocks, fold):
        assert seen.setdefault(b, f) == f, b
    return fold


def crossfit_isotonic(p, y, fold):
    q = np.empty(len(p), dtype=float)
    for k in range(N_FOLDS):
        train, test = fold != k, fold == k
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0, increasing=True).fit(p[train], y[train])
        q[test] = iso.predict(p[test])
    return q


def pav_fit(ybar, w):
    """Weighted pool-adjacent-violators, increasing, over the sorted distinct values (all w > 0)."""
    vals, wts, cnts = [], [], []
    for yi, wi in zip(ybar.tolist(), w.tolist()):
        vals.append(yi)
        wts.append(wi)
        cnts.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            y2, w2, n2 = vals.pop(), wts.pop(), cnts.pop()
            tot = wts[-1] + w2
            vals[-1] = (vals[-1] * wts[-1] + y2 * w2) / tot
            wts[-1] = tot
            cnts[-1] += n2
    return np.repeat(np.array(vals), cnts)


class FastCrossfit:
    """Cross-fitted isotonic squared-error sum under item weights (bootstrap multiplicities).

    A resample repeats whole blocks, and a block sits in one fold, so the refit on a resample is a weighted
    isotonic fit on the other folds' distinct stated values; aggregating by distinct value makes it a PAV over a
    few dozen points. predict_unweighted reproduces sklearn's cross-fit at unit weights (checked in main).
    """

    def __init__(self, p, y, fold):
        self.values, self.vidx = np.unique(p, return_inverse=True)
        self.nv = len(self.values)
        self.y = y.astype(float)
        self.fold = fold
        self.code = fold * self.nv + self.vidx

    def _fold_maps(self, w):
        """Weight and fire weight per fold and distinct value, and each fold's map fit on the other folds."""
        W = np.bincount(self.code, weights=w, minlength=N_FOLDS * self.nv).reshape(N_FOLDS, self.nv)
        P = np.bincount(self.code, weights=w * self.y, minlength=N_FOLDS * self.nv).reshape(N_FOLDS, self.nv)
        Wt, Pt = W.sum(0), P.sum(0)
        maps = []
        for k in range(N_FOLDS):
            wtr, ptr = Wt - W[k], Pt - P[k]
            have = wtr > 0
            maps.append(np.interp(self.values, self.values[have], pav_fit(ptr[have] / wtr[have], wtr[have]))
                        if have.any() else None)
        return W, P, maps

    def loss_sum(self, w):
        W, P, maps = self._fold_maps(w)
        total = 0.0  # accumulated fold by fold; Python's sum() of floats rounds differently from 3.12 on
        for k, q in enumerate(maps):
            if q is None:
                return np.nan
            total += float(np.sum(P[k] * (1.0 - q) ** 2 + (W[k] - P[k]) * q ** 2))
        return total

    def predict_unweighted(self):
        _W, _P, maps = self._fold_maps(np.ones(len(self.y)))
        out = np.empty(len(self.y))
        for k, q in enumerate(maps):
            out[self.fold == k] = q[self.vidx[self.fold == k]]
        return out


class FastPAV:
    """In-sample weighted isotonic squared-error sum, for the CORP discrimination term under the bootstrap."""

    def __init__(self, p, y):
        self.values, self.vidx = np.unique(p, return_inverse=True)
        self.nv = len(self.values)
        self.y = y.astype(float)

    def loss_sum(self, w):
        W = np.bincount(self.vidx, weights=w, minlength=self.nv)
        P = np.bincount(self.vidx, weights=w * self.y, minlength=self.nv)
        have = W > 0
        fitted = pav_fit(P[have] / W[have], W[have])
        return float(np.sum(P[have] * (1.0 - fitted) ** 2 + (W[have] - P[have]) * fitted ** 2))


# ------------------------------------------------------------------------------------------------
# bootstrap
# ------------------------------------------------------------------------------------------------
def cluster_positions(blocks):
    """Number of blocks and each item's position among the sorted block keys (cu.build_cluster_index order)."""
    _flat, _starts, _sizes, keys = cu.build_cluster_index(blocks)
    pos = {k: j for j, k in enumerate(keys)}
    return len(keys), np.array([pos[b] for b in blocks], dtype=np.int64)


def run_bootstrap(blocks, fixed, refit, seed, resamples):
    """One draw set scored under every statistic, so all contrasts are paired.

    fixed: {name: per-item loss difference}; the statistic is its weighted mean over the drawn items.
    refit: {name: (A, B)} with loss_sum(w) methods; the statistic is [A(w) - B(w)] / sum(w).
    Draws follow cu.bootstrap_arms: rng.integers(0, G, size=G) per replicate over the sorted block keys.
    """
    G, cl = cluster_positions(blocks)
    rng = np.random.default_rng(seed)
    draws = {name: np.empty(resamples) for name in list(fixed) + list(refit)}
    for b in range(resamples):
        w = np.bincount(rng.integers(0, G, size=G), minlength=G)[cl].astype(float)
        sw = w.sum()
        for name, d in fixed.items():
            draws[name][b] = float(np.dot(w, d) / sw)
        for name, (fa, fb) in refit.items():
            draws[name][b] = (fa.loss_sum(w) - fb.loss_sum(w)) / sw
    out = {}
    for name, v in draws.items():
        lo, hi, dropped = cu.percentile_interval(v, CI)
        out[name] = {"lo": lo, "hi": hi, "dropped": dropped, "excludes_zero": bool(lo > 0 or hi < 0)}
    return out


def refit_check(series, run_keys, y, blocks, fold, base_seed, replicates=5):
    """Check the fast weighted refit against sklearn on real replicates (the first few of each run's draws).

    Each replicate is gathered with its duplicated blocks as cu.ragged_gather does, a duplicated block keeping
    its one fold, and the cross-fit (and the in-sample PAV) is refit with sklearn on the gathered sample.
    Returns the largest absolute difference of the replicate statistic, per method.
    """
    def sk_crossfit_loss(p, yy, ff):
        total = 0.0
        for k in range(N_FOLDS):
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0, increasing=True).fit(p[ff != k], yy[ff != k])
            total += float(np.sum((iso.predict(p[ff == k]) - yy[ff == k]) ** 2))
        return total

    def sk_pav_loss(p, yy):
        fit = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0, increasing=True).fit(p, yy).predict(p)
        return float(np.sum((fit - yy) ** 2))

    flat, starts, sizes, _keys = cu.build_cluster_index(blocks)
    G, cl = cluster_positions(blocks)
    prior = series[PRIOR]
    worst = {"crossfit": 0.0, "pav": 0.0}
    for k in run_keys:
        rng = np.random.default_rng(cu.stable_seed(base_seed, "task-mesogeos", k, "calibration-r9"))
        fa, fb = FastCrossfit(series[k], y, fold), FastCrossfit(prior, y, fold)
        pa, pb = FastPAV(series[k], y), FastPAV(prior, y)
        for _ in range(replicates):
            draw = rng.integers(0, G, size=G)
            idx = cu.ragged_gather(flat, starts, sizes, draw)
            w = np.bincount(draw, minlength=G)[cl].astype(float)
            fast = (fa.loss_sum(w) - fb.loss_sum(w)) / w.sum()
            slow = (sk_crossfit_loss(series[k][idx], y[idx], fold[idx])
                    - sk_crossfit_loss(prior[idx], y[idx], fold[idx])) / len(idx)
            worst["crossfit"] = max(worst["crossfit"], abs(fast - slow))
            fast = (pb.loss_sum(w) - pa.loss_sum(w)) / w.sum()
            slow = (sk_pav_loss(prior[idx], y[idx]) - sk_pav_loss(series[k][idx], y[idx])) / len(idx)
            worst["pav"] = max(worst["pav"], abs(fast - slow))
    return worst


# ------------------------------------------------------------------------------------------------
# ECE chance band
# ------------------------------------------------------------------------------------------------
def ece_rows(q, ys):
    """ECE of each row of labels ys (draws by items) against forecasts q (one row, or one per draw)."""
    ys = np.atleast_2d(ys)
    q = np.broadcast_to(q, ys.shape)
    R, n = ys.shape
    idx = (np.searchsorted(BIN_EDGES, q, side="left") + 10 * np.arange(R)[:, None]).ravel()
    sq = np.bincount(idx, weights=q.ravel(), minlength=10 * R)
    sy = np.bincount(idx, weights=ys.ravel(), minlength=10 * R)
    return np.abs(sq - sy).reshape(R, 10).sum(axis=1) / n


def refit_rows(q, targets):
    """adjust(q, d_r) for each target, with d_r the shift that makes the row average targets[r].

    Newton steps kept inside a bisection bracket; stated 0 and 1 stay put, as in adjust.
    """
    inner = (q > 0.0) & (q < 1.0)
    z, ones, n = logit(q[inner]), float(np.sum(q >= 1.0)), len(q)
    d = np.zeros(len(targets))
    lo, hi = np.full(len(targets), -40.0), np.full(len(targets), 40.0)
    for _ in range(200):
        s = expit(z[None, :] + d[:, None])
        f = (s.sum(axis=1) + ones) / n - targets
        if np.max(np.abs(f)) < 1e-14:
            break
        lo, hi = np.where(f < 0, d, lo), np.where(f > 0, d, hi)
        step = d - f / np.maximum((s * (1.0 - s)).sum(axis=1) / n, 1e-300)
        d = np.where((step > lo) & (step < hi), step, 0.5 * (lo + hi))
    out = np.broadcast_to(q, (len(targets), n)).copy()
    out[:, inner] = expit(z[None, :] + d[:, None])
    return out


def chance_band(q, observed, seed, draws):
    """95th percentile of ECE under labels drawn from the forecast q itself (Broecker and Smith 2007).

    plain: the forecast held fixed. refit: the shift refit so the forecast averages each simulated label mean,
    the null matching an observed forecast whose mean was fitted to the observed labels.
    """
    rng = np.random.default_rng(seed)
    plain, refit = np.empty(draws), np.full(draws, np.nan)
    for c0 in range(0, draws, BAND_CHUNK):
        c1 = min(draws, c0 + BAND_CHUNK)
        ys = (rng.random((c1 - c0, len(q))) < q).astype(float)
        plain[c0:c1] = ece_rows(q, ys)
        m = ys.mean(axis=1)
        ok = (m > 0) & (m < 1)
        refit[c0:c1][ok] = ece_rows(refit_rows(q, m[ok]), ys[ok])
    # A draw with the observed label count in every bin reproduces the observed ECE up to rounding; count it as
    # at or above, so the tail shares do not depend on the last bit.
    at_or_above = observed - TIE_TOL
    plain_p95, refit_p95 = float(np.percentile(plain, 95)), float(np.nanpercentile(refit, 95))
    return {
        "observed": observed,
        "plain_p95": plain_p95, "plain_share_at_or_above_observed": float(np.mean(plain >= at_or_above)),
        "refit_p95": refit_p95, "refit_share_at_or_above_observed": float(np.nanmean(refit >= at_or_above)),
        "inside_plain": bool(observed <= plain_p95),
        "inside_refit": bool(observed <= refit_p95),
    }


# ------------------------------------------------------------------------------------------------
# fold-draw dependence of the recalibration
# ------------------------------------------------------------------------------------------------
def ece_fold_sweep(series, run_keys, y, blocks, draws):
    """Recalibrated ECE of every series under fold draws GroupKFold random_state 0 to draws - 1.

    The cross-fit is sklearn's, as in the table. Isotonic fits pool labels into fractions such as 0.5 that sit
    on a bin edge; sklearn returns 0.5 exactly, while a hand-written weighted PAV can return 0.5000000000000001
    and move a whole pooled block across the edge. That changes single-run ECEs in a few draws (for
    claude-opus-5 bare, whose fits pool to 0.5) but not the prior's range or the range of the run medians.
    """
    ece = {k: np.empty(draws) for k in series}
    for s in range(draws):
        fold = make_folds(y, blocks, s)
        for k, p in series.items():
            ece[k][s] = cm.compute_calibration_and_murphy(y, crossfit_isotonic(p, y, fold))["ece_equal_width"]
    below = np.array([sum(ece[k][s] < ece[PRIOR][s] for k in run_keys) for s in range(draws)])
    return ece, {
        "draws": draws,
        "prior": {"min": float(ece[PRIOR].min()), "median": float(np.median(ece[PRIOR])), "max": float(ece[PRIOR].max())},
        "runs": {k: {"min": float(ece[k].min()), "median": float(np.median(ece[k])), "max": float(ece[k].max()),
                     "share_of_draws_below_prior": float(np.mean(ece[k] < ece[PRIOR]))} for k in run_keys},
        "runs_below_prior_per_draw": {str(v): int(c) for v, c in zip(*np.unique(below, return_counts=True))},
    }


def brier_fold_sweep(series, runs, y, blocks, draws, resamples, base_seed, t0):
    """Refit bootstrap of recalibrated Brier minus prior under fold draws 0 to draws - 1 (folds fixed per block)."""
    G, cl = cluster_positions(blocks)
    ones = np.ones(len(y))
    out = {k: [] for k in runs}
    for s in range(draws):
        fold = make_folds(y, blocks, s)
        prior_fc = FastCrossfit(series[PRIOR], y, fold)
        for k, r in runs.items():
            fc = FastCrossfit(series[k], y, fold)
            rng = np.random.default_rng(cu.stable_seed(base_seed, "verify-foldseed", r["seed_label"], s))
            d = np.empty(resamples)
            for b in range(resamples):
                w = np.bincount(rng.integers(0, G, size=G), minlength=G)[cl].astype(float)
                d[b] = (fc.loss_sum(w) - prior_fc.loss_sum(w)) / w.sum()
            lo, hi, _ = cu.percentile_interval(d, CI)
            out[k].append({"fold_seed": s, "point": (fc.loss_sum(ones) - prior_fc.loss_sum(ones)) / len(y),
                           "lo": lo, "hi": hi, "excludes_zero": bool(lo > 0 or hi < 0)})
        print("  fold draw %2d of %d done at %.0fs" % (s + 1, draws, time.time() - t0), flush=True)
    return {k: {"resolved_below": sum(e["hi"] < 0 for e in v), "resolved_above": sum(e["lo"] > 0 for e in v),
                "per_fold_seed": v} for k, v in out.items()}


# ------------------------------------------------------------------------------------------------
# calls
# ------------------------------------------------------------------------------------------------
def call_detail(runs, y, shift, versions):
    """Scored call rates (run_mesogeos.score fills a missing answer as p >= 0.5) and the shifted counterfactual."""
    out = {}
    for key, r in runs.items():
        rows = r["rows"]
        p = versions[key]["stated"]
        answered = np.array([row["call"] is not None for row in rows])
        scored = rm.score(rows, key)
        shifted_call = versions[key]["shifted"] >= 0.5
        out[key] = {
            "answered": int(answered.sum()), "omitted": int((~answered).sum()),
            "stated_yes": sum(bool(row["call"]) for row in rows if row["call"] is not None),
            "imputed_yes_among_omitted": int(np.sum((p >= 0.5) & ~answered)),
            "call_rate_scored": scored["positive_rate_called"], "f1_fire_scored": scored["f1_fire"],
            "shifted_call_rate_at_half": float(np.mean(shifted_call)),
            "shifted_f1_fire_at_half": float(f1_score(y, shifted_call, pos_label=1)),
            "stated_threshold_equivalent_to_half_after_shift": float(expit(-shift[key]["delta"])),
        }
    return out


# ------------------------------------------------------------------------------------------------
def summarize(y, runs, shift, table, attribution, boot, bands, ece_sweep, brier_sweep, calls, responses, versions):
    """The numbers and verdicts the paper's text quotes, each computed from the sections above."""
    keys = list(runs)
    prop = [k for k in keys if runs[k]["weights"] == "proprietary"]
    open_w = [k for k in keys if runs[k]["weights"] == "open"]
    qwen = [k for k in open_w if runs[k]["label"] == "Qwen3-VL"]
    llama = [k for k in open_w if runs[k]["label"] == "Llama 4 Maverick"]
    name = {k: "%s %s" % (runs[k]["label"], runs[k]["condition"]) for k in keys}

    def span(vals):
        vals = list(vals)
        return [min(vals), max(vals)]

    def resolved(stat, sign):
        return [name[k] for k in keys if (boot[k][stat]["hi"] < 0 if sign < 0 else boot[k][stat]["lo"] > 0)]

    share = {k: attribution[k]["share_of_gap_removed_by_shift"] for k in prop + qwen}
    dsc = {k: boot[k]["dsc_run_minus_prior"] for k in keys}
    zeros = {name.get(k, "prior"): {"items": int(np.sum(versions[k]["stated"] == 0)),
                                    "fires": int(np.sum(y[versions[k]["stated"] == 0]))}
             for k in versions if np.any(versions[k]["stated"] == 0)}
    lowest = sorted(keys, key=lambda k: calls[k]["call_rate_scored"])[:2]

    def training_share_move(field):
        return max(abs(table[k]["shifted_training_share"][field] - table[k]["shifted"][field]) for k in keys + [PRIOR])

    return {
        "sample_rate_target": float(y.mean()),
        "effective_prior": {"proprietary_range": span(shift[k]["effective_prior"] for k in prop),
                            "open_weight": {name[k]: shift[k]["effective_prior"] for k in open_w},
                            "calendar_month_prior": shift[PRIOR]["effective_prior"]},
        "stated_zeros": zeros,
        "proprietary_ece": {"stated": span(table[k]["stated"]["ece"] for k in prop),
                            "shifted": span(table[k]["shifted"]["ece"] for k in prop)},
        "proprietary_brier": {"stated": span(table[k]["stated"]["brier"] for k in prop),
                              "shifted": span(table[k]["shifted"]["brier"] for k in prop)},
        "prior_shifted": {"ece": table[PRIOR]["shifted"]["ece"], "brier": table[PRIOR]["shifted"]["brier"]},
        "every_run_shifted_ece_above_prior": all(table[k]["shifted"]["ece"] > table[PRIOR]["shifted"]["ece"] for k in keys),
        "every_run_stated_ece_above_prior": all(table[k]["stated"]["ece"] > table[PRIOR]["stated"]["ece"] for k in keys),
        "inside_ece_chance_band_refit": [name[k] for k in keys if bands[k]["inside_refit"]],
        "inside_ece_chance_band_plain": [name[k] for k in keys if bands[k]["inside_plain"]],
        "shifted_brier_resolved_below_prior": resolved("brier_minus_prior_shifted", -1),
        "shifted_brier_resolved_above_prior": resolved("brier_minus_prior_shifted", +1),
        "stated_brier_resolved_above_prior": resolved("brier_minus_prior_stated", +1),
        "base_rate_share_of_stated_gap_min": {"run": name[min(share, key=share.get)], "share": min(share.values()),
                                              "over": "proprietary and Qwen3-VL runs"},
        "llama": {name[k]: {"base_rate_part": attribution[k]["base_rate_part"],
                            "residual_miscalibration_part": attribution[k]["residual_miscalibration_part"],
                            "discrimination_deficit_part": attribution[k]["discrimination_deficit_part"],
                            "dsc_run_minus_prior_interval": [dsc[k]["lo"], dsc[k]["hi"]]} for k in llama},
        "dsc_resolved_above_prior": {name[k]: dsc[k]["point"] for k in keys if dsc[k]["lo"] > 0},
        "dsc_resolved_below_prior": {name[k]: dsc[k]["point"] for k in keys if dsc[k]["hi"] < 0},
        "recalibrated_fold_seed_7": {"ece_runs": span(table[k]["recalibrated"]["ece"] for k in keys),
                                     "ece_prior": table[PRIOR]["recalibrated"]["ece"],
                                     "refit_brier_resolved_below_prior": resolved("brier_minus_prior_recalibrated_refit", -1),
                                     "refit_brier_resolved_above_prior": resolved("brier_minus_prior_recalibrated_refit", +1)},
        "recalibrated_ece_over_fold_draws": {"draws": ece_sweep["draws"],
                                             "prior_range": [ece_sweep["prior"]["min"], ece_sweep["prior"]["max"]],
                                             "run_median_range": span(v["median"] for v in ece_sweep["runs"].values())},
        "recalibrated_brier_over_fold_draws": {name[k]: {"resolved_below": v["resolved_below"],
                                                         "resolved_above": v["resolved_above"]} for k, v in brier_sweep.items()},
        "training_share_target": {
            "target": cm.TRAIN_PRIOR_MEAN,
            "max_abs_ece_change": training_share_move("ece"),
            "max_abs_brier_change": training_share_move("brier"),
            "brier_resolved_below_prior": resolved("brier_minus_prior_shifted_training_share", -1),
            "brier_resolved_above_prior": resolved("brier_minus_prior_shifted_training_share", +1)},
        "mean_p_source_prior": {"brier_resolved_below_prior": resolved("brier_minus_prior_shifted_meanp", -1),
                                "brier_resolved_above_prior": resolved("brier_minus_prior_shifted_meanp", +1)},
        "lowest_scored_call_rates": {name[k]: {"call_rate_scored": calls[k]["call_rate_scored"],
                                               "mean_stated_p": table[k]["stated"]["mean_p"],
                                               "f1_fire_scored": calls[k]["f1_fire_scored"],
                                               "shifted_call_rate_at_half": calls[k]["shifted_call_rate_at_half"],
                                               "shifted_f1_fire_at_half": calls[k]["shifted_f1_fire_at_half"]} for k in lowest},
        "omissions": {"rows": sum(v["rows"] for v in responses.values()),
                      "rows_without_probability": sum(v["rows_without_probability"] for v in responses.values()),
                      "omitted_total": sum(v["omitted_yes_no"] for v in responses.values()),
                      "omitted_by_run": {name[k]: v["omitted_yes_no"] for k, v in responses.items() if v["omitted_yes_no"]}},
        "key_finding_iii": {
            "note": "share of each proprietary run's stated Brier gap to the prior that the prior shift removes",
            "run_shifted_prior_as_stated": {name[k]: attribution[k]["share_of_gap_removed_by_shift"] for k in prop},
            "run_and_prior_shifted_alike": {name[k]: attribution[k]["share_of_gap_removed_shifting_alike"] for k in prop},
        },
    }


def fmt(x):
    return ("%.3f" % x).replace("-0.000", "0.000")


def print_report(y, runs, shift, table, boot, claims, checks, prompts, month):
    keys = list(runs)
    print("\npopulation: %d items, %d fires, sample rate %.4f, %d blocks" % (len(y), int(y.sum()), y.mean(), checks["blocks"]))
    print("prompts: %d paper and %d paraphrase prompts, %d base-rate hits; %s to %s driver lines per prompt"
          % (prompts["paper_prompts"], prompts["paraphrase_prompts"], prompts["total_hits"], *prompts["driver_lines_per_prompt"]))
    if month["status"] == "ok":
        print("month_prior.py: %d training samples, positive share %.5f (calibration_mesogeos %.5f), by-month max diff %.1e"
              % (month["training_samples"], month["training_positive_share"], cm.TRAIN_PRIOR_MEAN,
                 month["max_abs_diff_by_month_vs_calibration_mesogeos"]))
    else:
        print("month_prior.py: " + month["status"])
    print("checks: " + ", ".join("%s %s" % (k, ("%.1e" % v) if isinstance(v, float) else v) for k, v in checks.items()))
    stat = {"stated": "brier_minus_prior_stated", "shifted": "brier_minus_prior_shifted",
            "recalibrated": "brier_minus_prior_recalibrated_refit"}
    head = "%-18s %-8s %9s %6s %6s %6s %6s %6s  %s"
    print("\n" + head % ("Model", "Cond.", "Eff.prior", "ECE", "Brier", "Rel.", "Res.", "AUPRC", "Brier minus prior [95% CI]"))
    for version, title in TABLE_VERSIONS:
        print("-- " + title)
        for k in [PRIOR] + keys:
            m = table[k][version]
            eff = fmt(shift[k]["effective_prior"]) if version == "stated" else "--"
            if k == PRIOR:
                label, cond, ci = "Calendar prior", "--", "--"
            else:
                b = boot[k][stat[version]]
                label, cond = runs[k]["label"], runs[k]["condition"]
                ci = "%s [%s, %s]%s" % (fmt(b["point"]), fmt(b["lo"]), fmt(b["hi"]), " *" if b["excludes_zero"] else "")
            print(head % (label, cond, eff, fmt(m["ece"]), fmt(m["brier"]), fmt(m["rel"]), fmt(m["res"]), fmt(m["auprc"]), ci))
    print("  (* interval excludes zero; recalibrated intervals refit the cross-fit in every resample)")
    print("\nnumbers the text quotes:")
    for k, v in claims.items():
        print("  %s: %s" % (k, json.dumps(v, default=float)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--resamples", type=int, default=RESAMPLES, help="block-bootstrap resamples (default 20000)")
    ap.add_argument("--seed", type=int, default=BASE_SEED, help="base seed of the bootstrap and fold-draw recipes")
    ap.add_argument("--out", type=pathlib.Path, default=OUT, help="output JSON (default analysis/recalibration_mesogeos.json)")
    args = ap.parse_args()
    t0 = time.time()

    items, y, blocks, series, runs = load()
    keys = list(runs)
    ybar = float(y.mean())
    n_blocks = len(set(blocks))
    month = month_prior_check()
    prompts = prompt_check(items)
    responses = response_check(runs, y)

    # versions of every series ----------------------------------------------------------------
    fold = make_folds(y, blocks, FOLD_SEED)
    sizes = tuple(int((fold == k).sum()) for k in range(N_FOLDS))
    if sizes != EXPECTED_FOLD_SIZES:
        sys.exit("GroupKFold(random_state=%d) gave fold sizes %s, not %s: this scikit-learn shuffles differently from "
                 "the 1.9.1 release the paper used" % (FOLD_SEED, sizes, EXPECTED_FOLD_SIZES))
    fold_summary = [{"fold": k, "items": int((fold == k).sum()), "fires": int(y[fold == k].sum()),
                     "blocks": len({b for b, f in zip(blocks, fold) if f == k})} for k in range(N_FOLDS)]
    shift, versions, fast_vs_sklearn = {}, {}, 0.0
    for k in [PRIOR] + keys:
        p = series[k]
        delta = delta_mean_match(p, ybar)
        pi_s = float(expit(logit(ybar) - delta))
        delta_train = delta_mean_match(p, cm.TRAIN_PRIOR_MEAN)
        delta_meanp = float(logit(ybar) - logit(np.mean(p)))
        shift[k] = {"delta": delta, "effective_prior": pi_s, "em_forward_returns": saerens_em_forward(p, pi_s),
                    "delta_training_share": delta_train, "delta_meanp": delta_meanp, "mean_stated_p": float(np.mean(p))}
        recal = crossfit_isotonic(p, y, fold)
        fast_vs_sklearn = max(fast_vs_sklearn, float(np.max(np.abs(FastCrossfit(p, y, fold).predict_unweighted() - recal))))
        versions[k] = {"stated": p, "shifted": adjust(p, delta), "recalibrated": recal,
                       "shifted_training_share": adjust(p, delta_train), "shifted_meanp": adjust(p, delta_meanp)}
    table = {k: {v: metrics(y, q) for v, q in versions[k].items()} for k in [PRIOR] + keys}

    # exact attribution of the stated Brier gap to the prior (CORP; DSC is invariant to the shift) ----------
    pr = table[PRIOR]
    attribution = {}
    for k in keys:
        st, sh = table[k]["stated"], table[k]["shifted"]
        gap = st["brier"] - pr["stated"]["brier"]
        a = {"brier_gap_stated": gap,
             "base_rate_part": st["corp_mcb"] - sh["corp_mcb"],
             "residual_miscalibration_part": sh["corp_mcb"] - pr["stated"]["corp_mcb"],
             "discrimination_deficit_part": pr["stated"]["corp_dsc"] - st["corp_dsc"],
             "dsc_invariance": abs(st["corp_dsc"] - sh["corp_dsc"])}
        a["share_of_gap_removed_by_shift"] = a["base_rate_part"] / gap
        a["share_of_gap_removed_shifting_alike"] = 1.0 - (sh["brier"] - pr["shifted"]["brier"]) / gap
        attribution[k] = a

    refit_vs_sklearn = refit_check(series, keys, y, blocks, fold, args.seed)
    checks = {
        "blocks": n_blocks,
        "fast_crossfit_vs_sklearn_max_abs_diff": fast_vs_sklearn,
        "fast_refit_vs_sklearn_on_replicates_crossfit": refit_vs_sklearn["crossfit"],
        "fast_refit_vs_sklearn_on_replicates_pav": refit_vs_sklearn["pav"],
        "em_forward_max_abs_deviation_from_target": max(abs(s["em_forward_returns"] - ybar) for s in shift.values()),
        "shift_auprc_max_abs_change": max(abs(table[k]["shifted"]["auprc"] - table[k]["stated"]["auprc"]) for k in table),
        "shift_moves_stated_zero": bool(any(np.any(versions[k]["shifted"][versions[k]["stated"] == 0] != 0) for k in versions)),
        "dsc_invariance_max": max(a["dsc_invariance"] for a in attribution.values()),
        "attribution_max_residual": max(abs(a["base_rate_part"] + a["residual_miscalibration_part"]
                                            + a["discrimination_deficit_part"] - a["brier_gap_stated"])
                                        for a in attribution.values()),
    }
    assert (len(y), int(y.sum()), n_blocks) == (386, 131, 352), (len(y), int(y.sum()), n_blocks)
    assert checks["fast_crossfit_vs_sklearn_max_abs_diff"] < 1e-12 and checks["em_forward_max_abs_deviation_from_target"] < 1e-9
    assert max(refit_vs_sklearn.values()) < 1e-12, refit_vs_sklearn
    assert checks["shift_auprc_max_abs_change"] < 1e-12 and not checks["shift_moves_stated_zero"]
    assert checks["dsc_invariance_max"] < 1e-12 and checks["attribution_max_residual"] < 1e-12
    print("versions and metrics done at %.0fs" % (time.time() - t0), flush=True)

    # paired block bootstrap ------------------------------------------------------------------
    boot = {}
    prior_fc, prior_pav = FastCrossfit(series[PRIOR], y, fold), FastPAV(series[PRIOR], y)
    for k in keys:
        loss = {v: (versions[k][v] - y) ** 2 - (versions[PRIOR][v] - y) ** 2 for v in versions[k]}
        fixed = {"brier_minus_prior_" + v: loss[v] for v in ("stated", "shifted", "shifted_training_share", "shifted_meanp")}
        fixed["brier_minus_prior_recalibrated_conditional"] = loss["recalibrated"]
        refit = {"brier_minus_prior_recalibrated_refit": (FastCrossfit(series[k], y, fold), prior_fc),
                 "dsc_run_minus_prior": (prior_pav, FastPAV(series[k], y))}  # PAV loss of the prior minus that of the run
        seed = cu.stable_seed(args.seed, "task-mesogeos", k, "calibration-r9")
        res = run_bootstrap(blocks, fixed, refit, seed, args.resamples)
        for name, d in fixed.items():
            res[name]["point"] = float(np.mean(d))
        res["brier_minus_prior_recalibrated_refit"]["point"] = res["brier_minus_prior_recalibrated_conditional"]["point"]
        res["dsc_run_minus_prior"]["point"] = table[k]["stated"]["corp_dsc"] - table[PRIOR]["stated"]["corp_dsc"]
        boot[k] = {"seed": seed, **res}
    print("bootstrap done at %.0fs" % (time.time() - t0), flush=True)

    # ECE chance bands of the shifted forecasts --------------------------------------------------
    bands = {}
    for k in [PRIOR] + keys:
        label = "prior" if k == PRIOR else runs[k]["seed_label"]
        q = versions[k]["shifted"]
        assert abs(ece_rows(q, y.astype(float))[0] - table[k]["shifted"]["ece"]) < 1e-12
        bands[k] = chance_band(q, table[k]["shifted"]["ece"], cu.stable_seed(BAND_BASE_SEED, "verify-band2", label), BAND_DRAWS)
    print("chance bands done at %.0fs" % (time.time() - t0), flush=True)

    # fold-draw dependence ---------------------------------------------------------------------
    ece_draws, ece_sweep = ece_fold_sweep(series, keys, y, blocks, ECE_FOLD_DRAWS)
    assert all(ece_draws[k][FOLD_SEED] == table[k]["recalibrated"]["ece"] for k in series)
    print("ECE over %d fold draws done at %.0fs" % (ECE_FOLD_DRAWS, time.time() - t0), flush=True)
    brier_sweep = brier_fold_sweep(series, runs, y, blocks, BRIER_FOLD_DRAWS, BRIER_FOLD_RESAMPLES, args.seed, t0)
    assert all(abs(v["per_fold_seed"][FOLD_SEED]["point"] - boot[k]["brier_minus_prior_recalibrated_refit"]["point"]) < 1e-12
               for k, v in brier_sweep.items())

    calls = call_detail(runs, y, shift, versions)
    claims = summarize(y, runs, shift, table, attribution, boot, bands, ece_sweep, brier_sweep, calls, responses, versions)

    out = {
        "population": {"items": len(y), "fires": int(y.sum()), "sample_rate": ybar, "blocks": n_blocks,
                       "uncertainty": ybar * (1.0 - ybar)},
        "runs": {k: {f: r[f] for f in ("model", "label", "condition", "weights")} for k, r in runs.items()},
        "config": {
            "folds": "GroupKFold(n_splits=5, shuffle=True, random_state=7) over the blocks", "fold_summary": fold_summary,
            "bootstrap": {"resamples": args.resamples, "ci": CI,
                          "seed": "stable_seed(%d, 'task-mesogeos', run, 'calibration-r9')" % args.seed},
            "ece_chance_band": {"draws": BAND_DRAWS, "seed": "stable_seed(%d, 'verify-band2', label)" % BAND_BASE_SEED},
            "ece_fold_draws": "GroupKFold random_state 0 to %d" % (ECE_FOLD_DRAWS - 1),
            "brier_fold_draws": {"fold_seeds": "0 to %d" % (BRIER_FOLD_DRAWS - 1), "resamples": BRIER_FOLD_RESAMPLES,
                                 "seed": "stable_seed(%d, 'verify-foldseed', label, fold_seed)" % args.seed},
            "shift_targets": {"sample_rate": ybar, "training_share": cm.TRAIN_PRIOR_MEAN},
        },
        "checks": checks,
        "month_prior": month,
        "prompts": prompts,
        "responses": responses,
        "shift": shift,
        "metrics": table,
        "attribution": attribution,
        "bootstrap": boot,
        "ece_chance_band": bands,
        "recalibrated_ece_over_fold_draws": ece_sweep,
        "recalibrated_brier_over_fold_draws": brier_sweep,
        "calls": calls,
        "claims": claims,
    }
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print_report(y, runs, shift, table, boot, claims, checks, prompts, month)
    print("\nwrote %s in %.0fs" % (args.out.relative_to(ROOT) if args.out.is_relative_to(ROOT) else args.out, time.time() - t0))


if __name__ == "__main__":
    main()
