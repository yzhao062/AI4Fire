#!/usr/bin/env python
"""Fire danger with the sample's base rate stated in the prompt, against version 1 and the calendar-month prior.

No model is called; every number comes from stored responses.

Population: the 386 Mesogeos Track A items the paper scores (test split, fold 0), 131 fires, sample positive rate
0.339, in 352 blocks (1-degree cell by calendar month of the target date, cluster_uncertainty.mesogeos_block_key).
Series per core model and arm:
  v1          the stored version-1 probability (task-mesogeos), no rate in the prompt
  v1 shifted  v1 moved to the sample rate by one log-odds constant (recalibration_mesogeos.adjust and
              delta_mean_match, the paper's prior-shift rescoring)
  v2          the rerun with "Base rate: in this evaluation sample, 33.9 percent of items had a wildfire of at least
              30 hectares start on the target date." before the question (run_mesogeos_baserate.py)
and the calendar-month prior of calibration_mesogeos.py, as stated and shifted to the sample rate alike.

Metrics: mean stated probability, ECE (ten equal-width bins, calibration_mesogeos.compute_calibration_and_murphy),
Brier score, average precision (AUPRC), and the call rate and F1 on the fire class as run_mesogeos.score scores
them. Paired block bootstrap (352 blocks, 20,000 resamples, one draw set per model and arm; seed
stable_seed(20260915, "task-mesogeos-baserate", "<stem>/<condition>", "baserate")) of Brier differences (v2 minus
v1, v2 minus v1 shifted, v2 minus the prior as stated and as shifted) and AUPRC differences (v2 minus v1); and per
model, of the grounded-minus-bare AUPRC change in each version (seed part "<stem>/arms").

    python analysis/mesogeos_baserate.py
"""
import argparse
import json
import pathlib
import sys

import numpy as np
from sklearn.metrics import average_precision_score

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
import calibration_mesogeos as cm  # noqa: E402
import cluster_uncertainty as cu  # noqa: E402
import models  # noqa: E402
import recalibration_mesogeos as rc  # noqa: E402
import run_mesogeos as rm  # noqa: E402

V2 = ROOT / "task-mesogeos-baserate"
OUT = ROOT / "analysis" / "mesogeos_baserate.json"
ARMS = ("bare", "grounded")


def read_rows(path, items):
    rows, bad = cu.read_jsonl(str(path))
    if bad:
        raise SystemExit("%s: %d unreadable lines" % (path.name, bad))
    best = {}
    for r in rows:
        if r["item_id"] not in best or best[r["item_id"]].get("error"):
            best[r["item_id"]] = r
    return [best.get(it["item_id"]) for it in items]


def prob(rows):
    """Stated probability, else the hard call as 1 or 0 (run_mesogeos.score), else NaN."""
    out = []
    for r in rows:
        if r is None or (r.get("probability") is None and r.get("call") is None):
            out.append(np.nan)
        elif r.get("probability") is not None:
            out.append(float(r["probability"]))
        else:
            out.append(1.0 if r["call"] else 0.0)
    return np.array(out, dtype=float)


class WeightedAP:
    """Average precision as sklearn defines it (steps at distinct scores), fast under bootstrap weights."""

    def __init__(self, y, s):
        self.order = np.argsort(-s, kind="mergesort")
        self.y = y[self.order].astype(float)
        ss = s[self.order]
        self.ends = np.r_[np.where(np.diff(ss) != 0)[0], len(ss) - 1]

    def __call__(self, w):
        w = w[self.order]
        tp = np.cumsum(w * self.y)[self.ends]
        fp = np.cumsum(w * (1.0 - self.y))[self.ends]
        if tp[-1] <= 0:
            return np.nan
        prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
        return float(np.sum(np.diff(np.r_[0.0, tp / tp[-1]]) * prec))


def summary(y, p, rows=None):
    c = cm.compute_calibration_and_murphy(y, p)
    out = {"mean_p": c["mean_predicted"], "ece": c["ece_equal_width"], "brier": float(np.mean((p - y) ** 2)),
           "auprc": float(average_precision_score(y, p))}
    if rows is not None:
        s = rm.score([dict(r, label=int(yy)) for r, yy in zip(rows, y)], "")
        out.update(call_rate=s["positive_rate_called"], f1_fire=s["f1_fire"],
                   omitted_calls=sum(r.get("call") is None for r in rows))
    return out


def block_bootstrap(blocks, fixed, ap_pairs, seed, resamples):
    """One draw set for every statistic: weighted means of per-item differences, and AP differences."""
    G, cl = rc.cluster_positions(blocks)
    rng = np.random.default_rng(seed)
    draws = {name: np.empty(resamples) for name in list(fixed) + list(ap_pairs)}
    for b in range(resamples):
        w = np.bincount(rng.integers(0, G, size=G), minlength=G)[cl].astype(float)
        sw = w.sum()
        for name, d in fixed.items():
            draws[name][b] = float(np.dot(w, d) / sw)
        for name, (fa, fb) in ap_pairs.items():
            draws[name][b] = fa(w) - fb(w)
    ones = np.ones(len(blocks))
    out = {}
    for name, v in draws.items():
        point = float(np.mean(fixed[name])) if name in fixed else ap_pairs[name][0](ones) - ap_pairs[name][1](ones)
        lo, hi, dropped = cu.percentile_interval(v, 95.0)
        out[name] = {"point": point, "lo": lo, "hi": hi, "dropped": dropped, "excludes_zero": bool(lo > 0 or hi < 0)}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--resamples", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20260915)
    ap.add_argument("--out", type=pathlib.Path, default=OUT)
    args = ap.parse_args()

    items = cm.load_items()
    assert len(items) == 386, len(items)
    y = np.array([int(it["label"]) for it in items], dtype=np.int64)
    rate = float(y.mean())
    blocks = [cu.mesogeos_block_key(it["context"]["longitude"], it["context"]["latitude"], it["target_date"], 1.0, "month")
              for it in items]
    months = [int(it["context"]["window_end"][5:7]) for it in items]
    prior = np.array([cm.TRAIN_PRIOR_BY_MONTH.get(m, cm.TRAIN_PRIOR_MEAN) for m in months], dtype=float)
    prior_s = rc.adjust(prior, rc.delta_mean_match(prior, rate))
    ones = np.ones(len(y))
    for p in (prior, prior_s):
        assert abs(WeightedAP(y, p)(ones) - average_precision_score(y, p)) < 1e-12

    core = models.models(tier="core")
    present = [m for m in core if all((V2 / ("responses-%s-%s.jsonl" % (m.stem, a))).exists() for a in ARMS)]
    out = {"items": len(y), "fires": int(y.sum()), "sample_rate": rate, "blocks": len(set(blocks)),
           "resamples": args.resamples, "base_seed": args.seed,
           "models_present": [m.label for m in present], "models_missing": [m.label for m in core if m not in present],
           "prior": {"stated": summary(y, prior), "shifted": summary(y, prior_s)}, "runs": {}, "arms": {}}
    for m in present:
        ps = {}
        for cond in ARMS:
            r1 = read_rows(cm.TASK / ("responses-%s-%s.jsonl" % (m.stem, cond)), items)
            r2 = read_rows(V2 / ("responses-%s-%s.jsonl" % (m.stem, cond)), items)
            p1, p2 = prob(r1), prob(r2)
            ok = np.isfinite(p2)
            if not ok.all():  # score an unanswered rerun item as the v1 run scored its own gaps: not at all
                print("%s %s: %d rerun items unanswered; they are dropped from every series of this run"
                      % (m.label, cond, int((~ok).sum())))
            p1s = rc.adjust(p1, rc.delta_mean_match(p1, rate))
            yy, bl = y[ok], [b for b, k in zip(blocks, ok) if k]
            p1, p1s, p2, pr, prs = p1[ok], p1s[ok], p2[ok], prior[ok], prior_s[ok]
            r1 = [r for r, k in zip(r1, ok) if k]
            r2 = [r for r, k in zip(r2, ok) if k]
            sq = lambda p: (p - yy) ** 2  # noqa: E731
            fixed = {"brier_v2_minus_v1": sq(p2) - sq(p1), "brier_v2_minus_v1_shifted": sq(p2) - sq(p1s),
                     "brier_v2_minus_prior": sq(p2) - sq(pr), "brier_v2_minus_prior_shifted": sq(p2) - sq(prs),
                     "brier_v1_minus_prior": sq(p1) - sq(pr), "brier_v1_shifted_minus_prior_shifted": sq(p1s) - sq(prs)}
            aps = {"auprc_v2_minus_v1": (WeightedAP(yy, p2), WeightedAP(yy, p1))}
            seed = cu.stable_seed(args.seed, "task-mesogeos-baserate", "%s/%s" % (m.stem, cond), "baserate")
            key = "%s/%s" % (m.label, cond)
            out["runs"][key] = {
                "answered": int(ok.sum()),
                "v1": summary(yy, p1, r1), "v1_shifted": summary(yy, p1s), "v2": summary(yy, p2, r2),
                "bootstrap": block_bootstrap(bl, fixed, aps, seed, args.resamples)}
            ps[cond] = (p1, p2, ok)
            print("done:", key, flush=True)
        both = ps["bare"][2] & ps["grounded"][2]
        if both.all():  # climatology effect in each version, on the same draws
            yy = y
            aps = {"auprc_grounded_minus_bare_v1": (WeightedAP(yy, ps["grounded"][0]), WeightedAP(yy, ps["bare"][0])),
                   "auprc_grounded_minus_bare_v2": (WeightedAP(yy, ps["grounded"][1]), WeightedAP(yy, ps["bare"][1]))}
            seed = cu.stable_seed(args.seed, "task-mesogeos-baserate", "%s/arms" % m.stem, "baserate")
            out["arms"][m.label] = block_bootstrap(blocks, {}, aps, seed, args.resamples)
    args.out.write_text(json.dumps(out, indent=1), encoding="utf-8")

    f = lambda x: ("%.3f" % x).replace("-0.000", "0.000")  # noqa: E731

    def ci(e):
        return "%s [%s, %s]%s" % (f(e["point"]), f(e["lo"]), f(e["hi"]), " *" if e["excludes_zero"] else "")

    print("\n%d items, %d fires, rate %.3f, %d blocks | models present: %s | missing: %s"
          % (out["items"], out["fires"], rate, out["blocks"], ", ".join(out["models_present"]),
             ", ".join(out["models_missing"]) or "none"))
    pr = out["prior"]
    print("calendar prior: ECE %s (shifted %s), Brier %s (shifted %s), AUPRC %s"
          % (f(pr["stated"]["ece"]), f(pr["shifted"]["ece"]), f(pr["stated"]["brier"]), f(pr["shifted"]["brier"]),
             f(pr["stated"]["auprc"])))
    head = "%-26s | %-23s | %-23s | %-17s | %-15s | %s"
    print("\n" + head % ("run", "mean p v1 -> v2", "ECE v1 / v1 shifted / v2", "Brier v1 -> v2", "call rate v1->v2",
                         "Brier v2-prior shifted [CI] | Brier v2-v1 [CI] | AUPRC v1 -> v2, change [CI]"))
    for key, r in out["runs"].items():
        b = r["bootstrap"]
        print(head % (key, "%s -> %s" % (f(r["v1"]["mean_p"]), f(r["v2"]["mean_p"])),
                      "%s / %s / %s" % (f(r["v1"]["ece"]), f(r["v1_shifted"]["ece"]), f(r["v2"]["ece"])),
                      "%s -> %s" % (f(r["v1"]["brier"]), f(r["v2"]["brier"])),
                      "%s -> %s" % (f(r["v1"]["call_rate"]), f(r["v2"]["call_rate"])),
                      "%s | %s | %s -> %s, %s" % (ci(b["brier_v2_minus_prior_shifted"]), ci(b["brier_v2_minus_v1"]),
                                                  f(r["v1"]["auprc"]), f(r["v2"]["auprc"]), ci(b["auprc_v2_minus_v1"]))))
    print("\nclimatology effect on AUPRC (grounded minus bare), v1 vs v2:")
    for label, a in out["arms"].items():
        print("  %-18s v1 %s | v2 %s" % (label, ci(a["auprc_grounded_minus_bare_v1"]), ci(a["auprc_grounded_minus_bare_v2"])))
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
