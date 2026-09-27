#!/usr/bin/env python
"""Smoke detection with the time cues removed: the masked, gap-matched rerun against version 1 on the same targets.

No model is called; every number comes from stored responses.

Rerun design (run_figlib_masked.py): 168 targets from 28 sequences of 17 fires, 84 clear (-1800, -1200, -600 s from
the first visible plume) and 84 smoke (+300, +900, +1500 s). Each target's reference is the frame nearest 10, 20, or
30 minutes before it, always before the plume, so each label meets each gap once per sequence. Both arms see every
frame with the printed capture time blacked out, and the reference is introduced as "an earlier frame from the same
camera".

Version 1 on the same 168 targets: the unmasked bare arm, and the grounded arm whose reference was the sequence's
earliest frame (about -2400 s), introduced as "the same camera earlier in the day, before any plume". Its clear
targets had the same references and gaps as the rerun; its smoke targets had gaps of 45 to 65 minutes.

Computed per core model, and pooled over the models present by stacking their items (figlib_timing.PairedRow):
  - recall on smoke frames, false-positive rate on clear frames, and balanced accuracy (their mean with
    specificity), bare and grounded, in the rerun (v2) and in version 1 on the same targets (v1)
  - the grounded-minus-bare change in each version, and v2 change minus v1 change
  - the masking effect on the bare arm, v2 bare minus v1 bare
  - v2 changes by reference gap (10, 20, 30 minutes)
  - seed-free counts of smoke frames gained and lost and clear-frame false positives added and removed
  - answers that quote a clock reading or a date (the same pattern as figlib_timing.py)
  - the frame-difference detector of figlib_baseline.py (mean absolute gray-level difference from the reference,
    threshold chosen by leave-one-fire-out cross-validation) on the rerun's masked pairs and, for comparison, on
    version 1's pairs for the same 168 targets
Pairs where either version or either arm lacks a parsed answer are dropped for that model and counted.
Intervals: fire-clustered percentile bootstrap (17 fires, 20,000 resamples), each widened about its midpoint by
F = 1.1149 as in the paper's Table tab:cross-task-summary; seeds stable_seed(20260915, "task-figlib-masked", <stem>,
"rerun"), with "pooled-<k>" as the stem of a pooled row.

    python analysis/figlib_masked.py
"""
import argparse
import json
import pathlib
import re
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
import cluster_uncertainty as cu  # noqa: E402
import models  # noqa: E402
import small_cluster_correction as scc  # noqa: E402

V1 = ROOT / "task-figlib"
V2 = ROOT / "task-figlib-masked"
OUT = ROOT / "analysis" / "figlib_masked.json"
ARMS = ("bare", "grounded")
GAPS = (600, 1200, 1800)
CLOCK_OR_DATE = re.compile(r"\b\d{1,2}:\d{2}\b|\b20\d\d-\d\d-\d\d\b|\b1[67]\d{8}\b")


def load_items():
    items = [json.loads(l) for l in (V2 / "items.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    fires = [cu.figlib_fire_key(it["sequence"]) for it in items]
    if any(f != it["fire_name"] for f, it in zip(fires, items)):
        raise SystemExit("figlib_fire_key disagrees with the fire_name column")
    return items, fires


def read_arm(task, stem, arm, ids):
    """Parsed smoke calls in item order (None where missing or unparsed) and the raw answers."""
    path = task / ("responses-%s-%s.jsonl" % (stem, arm))
    rows, bad = cu.read_jsonl(str(path))
    if bad:
        raise SystemExit("%s: %d unreadable lines" % (path.name, bad))
    best = {}
    for r in rows:  # a successful row wins over an error row for the same item
        if r["item_id"] not in best or best[r["item_id"]].get("error"):
            best[r["item_id"]] = r
    calls = [None if best.get(i, {}).get("prediction") is None else bool(best[i]["prediction"]) for i in ids]
    raws = [best.get(i, {}).get("raw") or "" for i in ids]
    return calls, raws


class Row:
    """Stacked paired calls for one model (k = 1) or several pooled; a draw of items expands to every model."""

    def __init__(self, items, calls, stems):
        n, k = len(items), len(stems)
        y = np.array([it["label"] == "smoke" for it in items])
        gap = np.array([it["nominal_gap_seconds"] for it in items])
        arrays = {key: [] for key in ("b2", "g2", "b1", "g1")}
        valid = []
        for s in stems:
            four = [calls[s][key] for key in ("b2", "g2", "b1", "g1")]
            valid.append(np.array([all(x is not None for x in t) for t in zip(*four)]))
            for key, a in zip(("b2", "g2", "b1", "g1"), four):
                arrays[key].append(np.array([bool(x) for x in a], dtype=float))
        self.a = {key: np.concatenate(v) for key, v in arrays.items()}
        self.v = np.concatenate(valid)
        self.y = np.tile(y, k)
        self.gap = np.tile(gap, k)
        self.offsets = np.arange(k) * n
        self.dropped = int((~self.v).sum())

    def stats(self, idx):
        j = (idx[None, :] + self.offsets[:, None]).ravel()
        j = j[self.v[j]]
        y, gap = self.y[j], self.gap[j]
        a = {key: v[j] for key, v in self.a.items()}
        out = {}
        with np.errstate(invalid="ignore", divide="ignore"):
            for ver in ("v2", "v1"):
                b, g = a["b" + ver[1]], a["g" + ver[1]]
                rb, rg, fb, fg = b[y].mean(), g[y].mean(), b[~y].mean(), g[~y].mean()
                out.update({ver + "_recall_bare": rb, ver + "_recall_grounded": rg, ver + "_recall_change": rg - rb,
                            ver + "_fpr_bare": fb, ver + "_fpr_grounded": fg, ver + "_fpr_change": fg - fb,
                            ver + "_ba_bare": 0.5 * (rb + 1 - fb), ver + "_ba_grounded": 0.5 * (rg + 1 - fg)})
                out[ver + "_ba_change"] = out[ver + "_ba_grounded"] - out[ver + "_ba_bare"]
                for gp in GAPS:
                    s, c = y & (gap == gp), ~y & (gap == gp)
                    out["%s_recall_change_%dmin" % (ver, gp // 60)] = g[s].mean() - b[s].mean()
                    out["%s_fpr_change_%dmin" % (ver, gp // 60)] = g[c].mean() - b[c].mean()
            for m in ("recall", "fpr", "ba"):
                out["v2_minus_v1_%s_change" % m] = out["v2_%s_change" % m] - out["v1_%s_change" % m]
            out["mask_effect_recall_bare"] = out["v2_recall_bare"] - out["v1_recall_bare"]
            out["mask_effect_fpr_bare"] = out["v2_fpr_bare"] - out["v1_fpr_bare"]
        return out

    def counts(self):
        v, y = self.v, self.y
        out = {"pairs": int(v.sum()), "dropped_pairs": self.dropped}
        for ver in ("v2", "v1"):
            b, g = self.a["b" + ver[1]] > 0.5, self.a["g" + ver[1]] > 0.5
            s, c = v & y, v & ~y
            out[ver] = {"smoke_gained": int((~b & g & s).sum()), "smoke_lost": int((b & ~g & s).sum()),
                        "bare_missed_smoke": int((~b & s).sum()),
                        "fp_added": int((~b & g & c).sum()), "fp_removed": int((b & ~g & c).sum()),
                        "bare_fp": int((b & c).sum()), "grounded_fp": int((g & c).sum())}
        return out


def bootstrap(statistic, groups, resamples, seed):
    """Fire-clustered percentile bootstrap of every entry of statistic(idx), plain and widened (figlib_timing.py)."""
    flat, starts, sizes, keys = cu.build_cluster_index(groups)
    rng = np.random.default_rng(seed)
    point = statistic(np.arange(len(groups)))
    names = list(point)
    reps = np.empty((resamples, len(names)))
    for r in range(resamples):
        st = statistic(cu.ragged_gather(flat, starts, sizes, rng.integers(0, len(keys), size=len(keys))))
        reps[r] = [st[k] for k in names]
    out = {}
    for j, k in enumerate(names):
        res = scc.apply_small_cluster_correction(reps[:, j], len(keys), float(point[k]))
        out[k] = {"point": float(point[k]), "ci": res["uncorrected"]["ci"],
                  "ci_widened": res["corrected_midpoint"]["ci"],
                  "excludes_zero_widened": res["corrected_midpoint"]["excludes_zero"]}
    return out


def auroc(score, label):
    pos = [s for s, l in zip(score, label) if l]
    neg = [s for s, l in zip(score, label) if not l]
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def design(items):
    out = {"targets": len(items), "smoke": sum(it["label"] == "smoke" for it in items),
           "sequences": len({it["sequence"] for it in items}), "fires": len({it["fire_name"] for it in items}),
           "references": len({it["reference_image"] for it in items}), "gap_seconds": {}}
    for gp in GAPS:
        for lab in ("no smoke", "smoke"):
            v = [it["gap_seconds"] for it in items if it["nominal_gap_seconds"] == gp and it["label"] == lab]
            out["gap_seconds"]["%d/%s" % (gp, lab)] = {"n": len(v), "min": min(v), "median": float(np.median(v)), "max": max(v)}
    out["gap_auroc_for_smoke"] = auroc([it["gap_seconds"] for it in items], [it["label"] == "smoke" for it in items])
    out["reference_offsets"] = {
        lab: [min(it["reference_offset_seconds"] for it in items if (it["label"] == "smoke") == s),
              max(it["reference_offset_seconds"] for it in items if (it["label"] == "smoke") == s)]
        for lab, s in (("clear_targets", False), ("smoke_targets", True))}
    return out


def detector(items, fires):
    """figlib_baseline.py's diff_mean_lofo on the rerun pairs and on version 1's pairs for the same targets."""
    import figlib_baseline as fb
    from PIL import Image

    v1 = [json.loads(l) for l in (V1 / "items.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    earliest = {}
    for it in v1:  # version 1's reference: the earliest frame of each sequence
        if it["sequence"] not in earliest or it["offset_seconds"] < earliest[it["sequence"]]["offset_seconds"]:
            earliest[it["sequence"]] = it

    def gray(rel):
        with Image.open(ROOT / rel.replace("\\", "/")) as im:
            return np.asarray(im.convert("L"), dtype=np.float32)

    y = np.array([it["label"] == "smoke" for it in items])
    out = {}
    for name, pairs in (("rerun_masked_matched", [(it["image"], it["reference_image"]) for it in items]),
                        ("v1_pairs_same_targets", [(it["source_image"], earliest[it["sequence"]]["image"]) for it in items])):
        score = np.array([float(np.abs(gray(t) - gray(r)).mean()) for t, r in pairs])
        pred, _ = fb.fit_lofo_thresholds(score, y, fires)
        rec, fpr = float(pred[y].mean()), float(pred[~y].mean())
        out[name] = {"predictions": [bool(v) for v in pred],
                     "accuracy": float((pred == y).mean()), "recall": rec, "false_positive_rate": fpr,
                     "balanced_accuracy": 0.5 * (rec + 1 - fpr),
                     "median_score_smoke": float(np.median(score[y])), "median_score_clear": float(np.median(score[~y])),
                     "recall_by_gap": {str(g // 60): float(pred[y & (np.array([it["nominal_gap_seconds"] for it in items]) == g)].mean())
                                       for g in GAPS},
                     "fpr_by_gap": {str(g // 60): float(pred[~y & (np.array([it["nominal_gap_seconds"] for it in items]) == g)].mean())
                                    for g in GAPS}}
    return out


def fmt(x, signed=False):
    return ("%+.3f" if signed else "%.3f") % x


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--resamples", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20260915)
    ap.add_argument("--out", type=pathlib.Path, default=OUT)
    args = ap.parse_args()

    items, fires = load_items()
    ids = [it["item_id"] for it in items]
    core = models.models(tier="core")
    present = [m for m in core if all((V2 / ("responses-%s-%s.jsonl" % (m.stem, a))).exists() for a in ARMS)]
    calls, clock = {}, {}
    for m in present:
        (b2, rb2), (g2, rg2) = (read_arm(V2, m.stem, a, ids) for a in ARMS)
        (b1, rb1), (g1, rg1) = (read_arm(V1, m.stem, a, ids) for a in ARMS)
        calls[m.stem] = {"b2": b2, "g2": g2, "b1": b1, "g1": g1}
        clock[m.label] = {"v2": sum(bool(CLOCK_OR_DATE.search(r)) for r in rb2 + rg2),
                          "v1": sum(bool(CLOCK_OR_DATE.search(r)) for r in rb1 + rg1)}

    out = {"design": design(items), "resamples": args.resamples, "base_seed": args.seed,
           "widening_factor": scc.compute_small_cluster_factor(len(set(fires))),
           "models_present": [m.label for m in present], "models_missing": [m.label for m in core if m not in present],
           "answers_quoting_clock_or_date": clock, "rows": {}}
    rows = [(m.label, m.stem, [m.stem]) for m in present]
    if len(present) > 1:
        rows.append(("pooled (%d)" % len(present), "pooled-%d" % len(present), [m.stem for m in present]))
    for label, stem, stems in rows:
        row = Row(items, calls, stems)
        seed = cu.stable_seed(args.seed, "task-figlib-masked", stem, "rerun")
        out["rows"][label] = {"counts": row.counts(), "stats": bootstrap(row.stats, fires, args.resamples, seed)}
        print("done:", label, flush=True)
    out["frame_difference_detector"] = det = detector(items, fires)
    # Each rerun arm against the detector on the same pairs: accuracy and balanced-accuracy margins, fire-clustered.
    y = np.array([it["label"] == "smoke" for it in items])
    d = np.array(det["rerun_masked_matched"]["predictions"])
    out["arms_vs_detector"] = {}
    for m in present:
        for arm, key in (("bare", "b2"), ("grounded", "g2")):
            c = calls[m.stem][key]
            ok = np.array([v is not None for v in c])
            pr = np.array([bool(v) for v in c])

            def margins(idx, pr=pr, ok=ok):
                idx = idx[ok[idx]]
                yy, a, b = y[idx], pr[idx], d[idx]
                with np.errstate(invalid="ignore"):
                    ba = lambda p: 0.5 * (p[yy].mean() + 1 - p[~yy].mean())  # noqa: E731
                    return {"accuracy_margin": float((a == yy).mean() - (b == yy).mean()), "ba_margin": float(ba(a) - ba(b))}
            seed = cu.stable_seed(args.seed, "task-figlib-masked", "%s/%s" % (m.stem, arm), "vs-detector")
            out["arms_vs_detector"]["%s/%s" % (m.label, arm)] = bootstrap(margins, fires, args.resamples, seed)
    args.out.write_text(json.dumps(out, indent=1), encoding="utf-8")

    d = out["design"]
    print("\ndesign: %d targets (%d smoke) from %d sequences of %d fires, %d reference frames; gap AUROC %.3f"
          % (d["targets"], d["smoke"], d["sequences"], d["fires"], d["references"], d["gap_auroc_for_smoke"]))
    print("models present: %s | missing: %s" % (", ".join(out["models_present"]), ", ".join(out["models_missing"]) or "none"))
    print("answers quoting a clock reading or a date, v2 vs v1 (both arms, 336 per model):",
          json.dumps({k: [v["v2"], v["v1"]] for k, v in clock.items()}))
    head = "%-18s | %s | %s | %s | %s"
    print("\n" + head % ("model", "v2 recall bare->grounded, change [widened 95% CI]",
                         "v2 FPR bare->grounded, change [CI]", "v2 BA change [CI]", "v1 recall change, FPR change (same items)"))
    for label, r in out["rows"].items():
        s = r["stats"]

        def ch(k):
            e = s[k]
            return "%s [%s, %s]%s" % (fmt(e["point"], True), fmt(e["ci_widened"][0], True), fmt(e["ci_widened"][1], True),
                                     " *" if e["excludes_zero_widened"] else "")
        print(head % (label, "%s->%s %s" % (fmt(s["v2_recall_bare"]["point"]), fmt(s["v2_recall_grounded"]["point"]), ch("v2_recall_change")),
                      "%s->%s %s" % (fmt(s["v2_fpr_bare"]["point"]), fmt(s["v2_fpr_grounded"]["point"]), ch("v2_fpr_change")),
                      ch("v2_ba_change"), "%s, %s" % (fmt(s["v1_recall_change"]["point"], True), fmt(s["v1_fpr_change"]["point"], True))))
        c = r["counts"]
        print("%-18s   counts v2: +%d/-%d smoke of %d missed, FP +%d/-%d | v1: +%d/-%d smoke, FP +%d/-%d | dropped pairs %d"
              % ("", c["v2"]["smoke_gained"], c["v2"]["smoke_lost"], c["v2"]["bare_missed_smoke"], c["v2"]["fp_added"],
                 c["v2"]["fp_removed"], c["v1"]["smoke_gained"], c["v1"]["smoke_lost"], c["v1"]["fp_added"],
                 c["v1"]["fp_removed"], c["dropped_pairs"]))
        print("%-18s   v2-v1 recall change %s | mask effect on bare recall %s, FPR %s | v2 recall change by gap 10/20/30: %s"
              % ("", ch("v2_minus_v1_recall_change"), ch("mask_effect_recall_bare"), ch("mask_effect_fpr_bare"),
                 " / ".join(fmt(s["v2_recall_change_%dmin" % (g // 60)]["point"], True) for g in GAPS)))
    for name, d in out["frame_difference_detector"].items():
        print("detector, %s: accuracy %.3f, recall %.3f, FPR %.3f, balanced accuracy %.3f; recall by gap %s, FPR by gap %s"
              % (name, d["accuracy"], d["recall"], d["false_positive_rate"], d["balanced_accuracy"],
                 d["recall_by_gap"], d["fpr_by_gap"]))
    print("arms against the detector on the rerun pairs (balanced-accuracy margin, widened 95% CI):")
    for key, st in out["arms_vs_detector"].items():
        e = st["ba_margin"]
        print("  %-26s %+.3f [%+.3f, %+.3f]%s | accuracy margin %+.3f" % (key, e["point"], e["ci_widened"][0], e["ci_widened"][1],
              " *" if e["excludes_zero_widened"] else "", st["accuracy_margin"]["point"]))
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
