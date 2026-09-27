"""Reference-to-target gap in FIgLib smoke detection, by ladder position and on the gap itself.

Backs the appendix paragraph "Reference-to-Target Gap" (app:figlib-timing) and Table tab:figlib-timing, the gap
and per-offset sentences of the smoke results paragraph (subsec:results-figlib), the smoke sentences of the
Limitations, and the "same reference-to-target gap" clause of the model sweep's smoke sentence.

The grounded arm shows the earliest frame of each camera sequence as a reference (the run_figlib.py rule). Every
reference sits at the -2400 s ladder position, so the reference precedes the 84 clear targets by 9 to 30 minutes
and the 112 smoke targets by 44 to 80 minutes: the gap alone separates the labels. The script measures that design
and asks whether the stored decisions of the six core models track the gap:

  1. Design: ladder, reference frames, gap ranges by label and position, the whole-minute gap thresholds that
     label all 196 targets, and the printed-time overlay strip (yellow text in the top rows) on every frame.
  2. Table panel (a): per model and ladder position, new/removed false positives on clear targets and gained/lost
     smoke frames; pooled over the six, the bare and grounded rates, their change, and the rescue rate (gained
     over bare-missed smoke frames).
  3. Table panel (b): least-squares slopes, per 10 minutes of gap, of the paired change over smoke items and over
     clear items and of the rescue rate, and balanced accuracy (mean of smoke recall and clear specificity).
  4. The frame-difference detector of figlib_baseline.py (mean absolute gray-level difference from the reference,
     leave-one-fire-out cut) on the same gaps; the same detector cut to each model's own grounded false-positive
     count ("Detector, matched"); and the scores of the clear frames the models newly flagged.
  5. Time words in the 2,520 stored core-model answers. A time-of-day word is checked against the local capture
     time given by the Unix epoch in the frame name, the instant the overlay prints.
  6. The sweep: grounded-minus-bare smoke recall of the sixteen full-capability models, paired as model_sweep.py
     pairs them, with the gap ranges of each model's paired items.

Uncertainty: paired percentile bootstrap over the 17 fires, 20,000 resamples, one draw scored under both arms and
every statistic of a row, each interval then widened about its midpoint by F = t(16) sqrt(17/16) / z = 1.1149
(small_cluster_correction.py). Seeds come from cluster_uncertainty.stable_seed with base 20260915; the seed parts,
recorded in the output, are the labels under which the paper's intervals were drawn, kept so the printed values
reproduce exactly. Every model contributes one stored run per arm, and no interval is adjusted for multiplicity.

Inputs (repository-relative):
  task-figlib/items.jsonl, task-figlib/responses-<stem>-{bare,grounded}.jsonl   tracked
  task-figlib/images-1568/   the frames the models saw. They are not redistributed (CC BY-NC-ND 4.0), so run
                             `python build_items_figlib.py` first to download them.
  task-figlib/images/        optional: an earlier local 1024-pixel download that also holds one clear frame per
                             sequence 3 to 4 minutes before the plume. build_items_figlib.py does not write it, so
                             that one check is skipped when the folder is absent.
No intermediate JSON from another analysis script is read.

Output: analysis/figlib_timing.json. Stdout: a summary and the LaTeX rows of Table tab:figlib-timing.

Usage:
    python analysis/figlib_timing.py                       # about 10 s on an Apple M5 Pro
    python analysis/figlib_timing.py --resamples 20000 --seed 20260915
"""
import argparse
import collections
import datetime
import json
import pathlib
import re
import sys
import time
import zoneinfo

import numpy as np
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
import cluster_uncertainty as cu  # noqa: E402
import figlib_baseline as fb  # noqa: E402
import figlib_paired as fp  # noqa: E402
import models  # noqa: E402
import small_cluster_correction as scc  # noqa: E402

TASK = ROOT / "task-figlib"
FRAMES = TASK / "images-1568"
EARLIER_FRAMES = TASK / "images"
OUT = ROOT / "analysis" / "figlib_timing.json"
GAP_UNIT = 600.0  # slopes are per 10 minutes of reference-to-target gap
ARMS = ("bare", "grounded")

# HPWREN prints site, date, clock time, and Unix epoch in yellow along the top of every frame.
OVERLAY_ROWS = 14
OVERLAY_MIN_PIXELS = 50

# Answer text: time-of-day words, and words or patterns that would name elapsed time, a clock reading, or a date.
TIME_OF_DAY = re.compile(r"\b(dawn|morning|noon|afternoon|evening|dusk)\b", re.I)
ELAPSED_TIME_WORDS = re.compile(r"\b(minutes?|mins?|hours?|hrs?|seconds?|timestamps?|clock|o'clock|elapsed|time|times|"
                                r"date|dated|epoch|a\.?m\.?|p\.?m\.?|utc|pst|pdt|overlay)\b", re.I)
CLOCK_OR_DATE = re.compile(r"\b\d{1,2}:\d{2}\b|\b20\d\d-\d\d-\d\d\b|\b1[67]\d{8}\b")
ITEM_EPOCH = re.compile(r"-(\d{10})_[+-]\d+$")
EARLIER_NAME = re.compile(r"^(?P<sequence>.+)__\d{10}_(?P<offset>[+-]?\d+)\.jpg$")


# Seed parts after the base seed. They are the labels under which the paper's intervals were first drawn (the
# matched-detector label included), kept verbatim so that the output reproduces the printed intervals exactly.
def row_seed(base, stem):
    return cu.stable_seed(base, "task-figlib", stem, "smoke-timing")


def detector_seed(base, statistic):
    return cu.stable_seed(base, "task-figlib", "frame-difference", statistic)


def matched_seed(base):
    return cu.stable_seed(base, "verify-w3", "matched-operating-point")


# ----------------------------------------------------------------------------------------------
# design
# ----------------------------------------------------------------------------------------------


def load_design():
    """Items, the grounded arm's reference frames, and the 196 scored targets in items.jsonl order."""
    items, bad = cu.read_jsonl(str(TASK / "items.jsonl"))
    if bad:
        raise SystemExit("task-figlib/items.jsonl has %d unreadable lines" % bad)
    reference = {}
    for it in items:  # run_figlib.py: the earliest frame of each sequence is the reference
        cur = reference.get(it["sequence"])
        if cur is None or it["offset_seconds"] < cur["offset_seconds"]:
            reference[it["sequence"]] = it
    ref_ids = {r["item_id"] for r in reference.values()}
    targets = [it for it in items if it["item_id"] not in ref_ids]
    fires = [cu.figlib_fire_key(it["sequence"]) for it in targets]
    if any(f != it["fire_name"] for f, it in zip(fires, targets)):
        raise SystemExit("figlib_fire_key disagrees with the fire_name column of items.jsonl")
    positions = sorted({it["ladder_target"] for it in targets})
    code = {p: j for j, p in enumerate(positions)}
    return {
        "items": items, "reference": reference, "targets": targets, "fires": fires, "positions": positions,
        "ids": [it["item_id"] for it in targets],
        "y": np.array([it["label"] == "smoke" for it in targets]),
        "gap": np.array([it["offset_seconds"] - reference[it["sequence"]]["offset_seconds"] for it in targets], float),
        "pos": np.array([it["ladder_target"] for it in targets]),
        "pos_code": np.array([code[it["ladder_target"]] for it in targets]),
    }


def design_summary(d):
    items, targets, y, gap = d["items"], d["targets"], d["y"], d["gap"]
    ladder = sorted({it["ladder_target"] for it in items})
    per_sequence = collections.defaultdict(set)
    for it in items:
        per_sequence[it["sequence"]].add(it["ladder_target"])
    refs = list(d["reference"].values())
    ref_offsets = [r["offset_seconds"] for r in refs]
    max_clear, min_smoke = float(gap[~y].max()), float(gap[y].min())
    thresholds = [t for t in range(int(min_smoke // 60) + 1) if max_clear < 60 * t < min_smoke]
    by_position = {}
    for p in d["positions"]:
        m = d["pos"] == p
        rows = [t for t, keep in zip(targets, m) if keep]
        by_position["%+d" % p] = {
            "label": rows[0]["label"], "frames": len(rows), "sequences": len({r["sequence"] for r in rows}),
            "fires": len({f for f, keep in zip(d["fires"], m) if keep}),
            "gap_seconds": [float(gap[m].min()), float(np.median(gap[m])), float(gap[m].max())],
            "gap_minutes_median": float(np.median(gap[m])) / 60.0,
        }
    if (len(items), len(per_sequence)) != (224, 28) or any(sorted(v) != ladder for v in per_sequence.values()):
        sys.exit("task-figlib/items.jsonl does not hold the 224-frame, 28-sequence ladder that the paper analyzes "
                 "(%d frames, %d sequences); a partial rebuild would change the design" % (len(items), len(per_sequence)))
    return {
        "items": len(items), "sequences": len(per_sequence),
        "fires": len({cu.figlib_fire_key(s) for s in per_sequence}), "ladder_seconds": ladder,
        "sequences_with_full_ladder": sum(1 for v in per_sequence.values() if sorted(v) == ladder),
        "reference": {
            "frames": len(refs), "ladder_targets": sorted({r["ladder_target"] for r in refs}),
            "labels": sorted({r["label"] for r in refs}), "offset_seconds": [min(ref_offsets), max(ref_offsets)],
            "minutes_before_plume": [-max(ref_offsets) / 60.0, -min(ref_offsets) / 60.0],
            "minutes_before_plume_median": -float(np.median(ref_offsets)) / 60.0,
        },
        "targets": {"frames": len(targets), "clear": int((~y).sum()), "smoke": int(y.sum()),
                    "per_sequence": sorted(set(collections.Counter(t["sequence"] for t in targets).values()))},
        "gap_seconds": {"clear": [float(gap[~y].min()), max_clear], "smoke": [min_smoke, float(gap[y].max())]},
        "gap_minutes": {"clear": [float(gap[~y].min()) / 60.0, max_clear / 60.0],
                        "smoke": [min_smoke / 60.0, float(gap[y].max()) / 60.0]},
        "separating_thresholds": {
            "margin_seconds": min_smoke - max_clear,
            "whole_minutes": [thresholds[0], thresholds[-1]] if thresholds else None,
            "all_label_every_target": bool(thresholds) and all(np.all((gap > 60 * t) == y) for t in thresholds),
        },
        "ladder_deviation_seconds": [min(t["offset_seconds"] - t["ladder_target"] for t in targets),
                                     max(t["offset_seconds"] - t["ladder_target"] for t in targets)],
        "by_position": by_position,
    }


# ----------------------------------------------------------------------------------------------
# paired model decisions
# ----------------------------------------------------------------------------------------------


def read_arm(stem, arm):
    rows, bad = cu.read_jsonl(str(TASK / ("responses-%s-%s.jsonl" % (stem, arm))))
    by_id, _ = cu.dedupe_by_item(rows)
    return by_id, bad


def paired_predictions(d, stems):
    """Bare and grounded smoke calls on the 196 targets; every core model answered all of them in both arms."""
    preds = {}
    for stem in stems:
        calls = []
        for arm in ARMS:
            by_id, bad = read_arm(stem, arm)
            missing = sum(1 for i in d["ids"] if by_id.get(i, {}).get("prediction") is None)
            if bad or missing:
                raise SystemExit("%s %s: %d unreadable lines, %d targets unanswered" % (stem, arm, bad, missing))
            calls.append(np.array([bool(by_id[i]["prediction"]) for i in d["ids"]]))
        preds[stem] = tuple(calls)
    return preds


def position_counts(d, preds, stems):
    """Seed-free counts by ladder position, summed over the stems of a row."""
    out = {}
    for p in d["positions"]:
        m = d["pos"] == p
        b = np.concatenate([preds[s][0][m] for s in stems])
        g = np.concatenate([preds[s][1][m] for s in stems])
        entry = {"pairs": int(b.size), "bare_positive": int(b.sum()), "grounded_positive": int(g.sum())}
        if p > 0:
            entry.update(gained=int((~b & g).sum()), lost=int((b & ~g).sum()), bare_missed=int((~b).sum()))
            entry["rescue_rate"] = entry["gained"] / entry["bare_missed"] if entry["bare_missed"] else None
        else:
            entry.update(new_false_positives=int((~b & g).sum()), removed_false_positives=int((b & ~g).sum()))
        out["%+d" % p] = entry
    return out


def ols_slope(x, v):
    if x.size < 2:
        return np.nan
    xc = x - x.mean()
    den = float((xc * xc).sum())
    if den <= 0:
        return np.nan
    return float((xc * (v - v.mean())).sum() / den)


class PairedRow:
    """Stacked paired calls for one row: a single model (k = 1) or the six core models pooled (k = 6).

    Pooling stacks the six models' item rows, so a pooled rate per position averages 6 x 28 model-item pairs.
    Because the models share items and gaps, the stacked slope equals the slope of the six-model mean change.
    A bootstrap draw of item indices is expanded to every stacked model, which keeps the fire resampling paired.
    """

    def __init__(self, d, preds, stems):
        k = len(stems)
        self.n = len(d["ids"])
        self.positions = d["positions"]
        self.b = np.concatenate([preds[s][0] for s in stems]).astype(float)
        self.g = np.concatenate([preds[s][1] for s in stems]).astype(float)
        self.y = np.tile(d["y"], k)
        self.x = np.tile(d["gap"] / GAP_UNIT, k)
        self.code = np.tile(d["pos_code"], k)
        self.offsets = np.arange(k) * self.n

    def stats(self, idx):
        j = (idx[None, :] + self.offsets[:, None]).ravel()
        b, g, y, x, code = self.b[j], self.g[j], self.y[j], self.x[j], self.code[j]
        npos = len(self.positions)
        count = np.bincount(code, minlength=npos).astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            rate_b = np.bincount(code, weights=b, minlength=npos) / count
            rate_g = np.bincount(code, weights=g, minlength=npos) / count
        out = {}
        for c, p in enumerate(self.positions):
            out["bare_%+d" % p] = rate_b[c]
            out["grounded_%+d" % p] = rate_g[c]
            out["change_%+d" % p] = rate_g[c] - rate_b[c]
        smoke, clear = y, ~y
        recall_b, recall_g = b[smoke].mean(), g[smoke].mean()
        fpr_b, fpr_g = b[clear].mean(), g[clear].mean()
        out["ba_bare"] = 0.5 * (recall_b + 1.0 - fpr_b)
        out["ba_grounded"] = 0.5 * (recall_g + 1.0 - fpr_g)
        out["ba_change"] = out["ba_grounded"] - out["ba_bare"]
        out["slope_smoke_gain"] = ols_slope(x[smoke], g[smoke] - b[smoke])
        missed = smoke & (b == 0)
        out["slope_rescue_rate"] = ols_slope(x[missed], g[missed])
        out["slope_clear_fpr_change"] = ols_slope(x[clear], g[clear] - b[clear])
        return out

    def nested(self, est):
        return {
            "by_position": {"%+d" % p: {k: est["%s_%+d" % (k, p)] for k in ("bare", "grounded", "change")}
                            for p in self.positions},
            "slope_smoke_gain": est["slope_smoke_gain"],
            "slope_rescue_rate": est["slope_rescue_rate"],
            "slope_clear_fpr_change": est["slope_clear_fpr_change"],
            "balanced_accuracy": {k: est["ba_" + k] for k in ("bare", "grounded", "change")},
        }


def bootstrap(statistic, groups, resamples, seed):
    """Fire-clustered percentile bootstrap of every entry of statistic(idx), plain and widened for 17 fires."""
    flat, starts, sizes, keys = cu.build_cluster_index(groups)
    rng = np.random.default_rng(seed)
    point = statistic(np.arange(len(groups)))
    names = list(point)
    reps = np.empty((resamples, len(names)))
    for r in range(resamples):
        draw = rng.integers(0, len(keys), size=len(keys))
        st = statistic(cu.ragged_gather(flat, starts, sizes, draw))
        reps[r] = [st[k] for k in names]
    out = {}
    for j, k in enumerate(names):
        res = scc.apply_small_cluster_correction(reps[:, j], len(keys), float(point[k]))
        out[k] = {"point": float(point[k]),
                  "ci": res["uncorrected"]["ci"], "excludes_zero": res["uncorrected"]["excludes_zero"],
                  "ci_widened": res["corrected_midpoint"]["ci"],
                  "excludes_zero_widened": res["corrected_midpoint"]["excludes_zero"]}
    return out


# ----------------------------------------------------------------------------------------------
# frames: scene-change score and printed-time overlay
# ----------------------------------------------------------------------------------------------


def open_frame(item):
    # items.jsonl stores Windows-style relative paths, so the frame is found by file name on every platform.
    return Image.open(fb.resolve_image_path(pathlib.PureWindowsPath(item["image"]).name, [FRAMES]))


def read_frames(d):
    """Mean absolute gray-level difference of each target from its reference (figlib_baseline.py's diff_mean
    score) and the overlay-yellow pixel counts of every frame."""
    if not FRAMES.is_dir():
        raise SystemExit("task-figlib/images-1568/ is missing; download the frames with `python build_items_figlib.py`")
    ref_gray = {}
    for ref in d["reference"].values():
        with open_frame(ref) as im:
            ref_gray[ref["item_id"]] = np.asarray(im.convert("L"), dtype=np.float32)
    target_index = {i: k for k, i in enumerate(d["ids"])}
    score = np.empty(len(d["ids"]), dtype=np.float32)
    overlay = {}
    for it in d["items"]:
        with open_frame(it) as im:
            rgb = np.asarray(im.convert("RGB")).astype(np.int16)
            k = target_index.get(it["item_id"])
            if k is not None:
                gray = np.asarray(im.convert("L"), dtype=np.float32)
                score[k] = np.abs(gray - ref_gray[d["reference"][it["sequence"]]["item_id"]]).mean()
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        yellow = (r > 170) & (g > 170) & (b < 110) & (r - b > 90)
        overlay[it["item_id"]] = (int(yellow[:OVERLAY_ROWS].sum()), float(yellow[OVERLAY_ROWS:].mean()))
    return score, overlay


def overlay_summary(overlay):
    top = [v[0] for v in overlay.values()]
    return {"frames": len(overlay), "top_rows": OVERLAY_ROWS, "min_pixels": OVERLAY_MIN_PIXELS,
            "frames_with_overlay": sum(1 for t in top if t >= OVERLAY_MIN_PIXELS),
            "top_yellow_pixels_min": min(top), "rest_of_frame_yellow_rate_max": max(v[1] for v in overlay.values())}


def detector(d, score, resamples, base):
    """The held-out frame-difference detector on the same reference, by ladder position and on the gap."""
    y, pos = d["y"], d["pos"]
    pred, cuts = fb.fit_lofo_thresholds(score, y, d["fires"])
    recall, fpr = float(pred[y].mean()), float(pred[~y].mean())
    by_position = {}
    for p in d["positions"]:
        m = pos == p
        by_position["%+d" % p] = {"rate": float(pred[m].mean()), "flagged": int(pred[m].sum()), "frames": int(m.sum()),
                                  "score_median": float(np.median(score[m]))}
    clear_positions = [p for p in d["positions"] if p < 0]
    per_sequence = collections.defaultdict(dict)
    for k, it in enumerate(d["targets"]):
        per_sequence[it["sequence"]][it["ladder_target"]] = score[k]
    rises = sum(1 for v in per_sequence.values() if v[clear_positions[-1]] > v[clear_positions[0]])
    x, flag = d["gap"] / GAP_UNIT, pred.astype(float)
    slopes = {}
    for name, key, mask in (("detector_fpr_slope_clear", "false_positive_rate", ~y),
                            ("detector_recall_slope_smoke", "recall", y)):
        def statistic(idx, name=name, mask=mask):
            m = mask[idx]
            return {name: ols_slope(x[idx][m], flag[idx][m])}
        slopes[key] = bootstrap(statistic, d["fires"], resamples, detector_seed(base, name))[name]
    return {
        "accuracy": float(np.mean(pred == y)), "recall": recall, "false_positive_rate": fpr,
        "balanced_accuracy": 0.5 * (recall + 1.0 - fpr), "clear_frames_flagged": int(pred[~y].sum()),
        "lofo_cut_range": [min(cuts.values()), max(cuts.values())], "by_position": by_position,
        "clear_score_rises_first_to_last_position": {"sequences": rises, "of": len(per_sequence)},
        "slope_per_10_min": slopes,
    }


def matched_detector(d, preds, stems, score, resamples, base):
    """The detector cut to each model's own grounded clear-frame false-positive count k: it flags the k clear frames
    with the most scene change. Pooled over the six models like the models' own rows."""
    y, pos, x = d["y"], d["pos"], d["gap"] / GAP_UNIT
    clear = np.flatnonzero(~y)
    ranked = clear[np.argsort(-score[clear], kind="mergesort")]
    k = {s: int(preds[s][1][~y].sum()) for s in stems}
    flags = np.zeros((len(stems), len(y)))
    for j, s in enumerate(stems):
        flags[j, ranked[:k[s]]] = 1.0
    pooled = flags.mean(axis=0)

    def statistic(idx):
        m = ~y[idx]
        return {"slope": ols_slope(x[idx][m], pooled[idx][m])}

    return {
        "k_per_model": {models.by_stem()[s].label: k[s] for s in stems},
        "most_clear_frames_flagged_by_a_model": max(int(preds[s][a][~y].sum()) for s in stems for a in (0, 1)),
        "false_positive_rate_by_position": {"%+d" % p: float(pooled[pos == p].mean())
                                            for p in d["positions"] if p < 0},
        "slope_per_10_min": bootstrap(statistic, d["fires"], resamples, matched_seed(base))["slope"],
    }


def newly_flagged_scores(d, preds, stems, score):
    """Scene-change score of clear frames each model newly flagged when grounded, against those it left negative."""
    new, stayed = [], []
    for s in stems:
        b, g = preds[s]
        negative = ~d["y"] & ~b
        new += score[negative & g].tolist()
        stayed += score[negative & ~g].tolist()
    return {"new_false_positive_pairs": len(new), "score_median_new": float(np.median(new)),
            "stayed_negative_pairs": len(stayed), "score_median_stayed": float(np.median(stayed))}


def earlier_frames(d):
    """Clear frames just before the plume in the optional earlier download, and the gaps they would give."""
    if not EARLIER_FRAMES.is_dir():
        return {"available": False}
    used = {pathlib.PureWindowsPath(it["image"]).name for it in d["items"]}
    extra = collections.defaultdict(list)
    for path in sorted(EARLIER_FRAMES.glob("*.jpg")):
        m = EARLIER_NAME.match(path.name)
        if m and path.name not in used and int(m.group("offset")) < 0:
            extra[m.group("sequence")].append(int(m.group("offset")))
    if not extra:
        return {"available": False}
    offsets = [o for v in extra.values() for o in v]
    gaps = collections.defaultdict(list)
    for it in d["targets"]:
        if it["label"] == "smoke":
            gaps["%+d" % it["ladder_target"]] += [it["offset_seconds"] - o for o in extra.get(it["sequence"], [])]
    max_clear = float(d["gap"][~d["y"]].max())
    return {
        "available": True, "sequences": len(extra), "of": len(d["reference"]), "frames": len(offsets),
        "offset_seconds": [min(offsets), max(offsets)],
        "minutes_before_plume": [-max(offsets) / 60.0, -min(offsets) / 60.0],
        "gap_minutes_to_smoke_targets": {k: [min(v) / 60.0, float(np.median(v)) / 60.0, max(v) / 60.0]
                                         for k, v in gaps.items()},
        "smoke_positions_within_clear_gaps": [k for k, v in gaps.items() if max(v) <= max_clear],
    }


# ----------------------------------------------------------------------------------------------
# stored answers and the sweep
# ----------------------------------------------------------------------------------------------


def capture_clock(epoch):
    """Local capture time (US Pacific, as the overlay prints it) of a frame's Unix epoch; None without tz data."""
    try:
        return datetime.datetime.fromtimestamp(epoch, zoneinfo.ZoneInfo("America/Los_Angeles"))
    except zoneinfo.ZoneInfoNotFoundError:  # Windows without the tzdata package
        return None


def matches_clock(word, clock):
    word = word.lower()
    if word in ("dawn", "morning"):
        return clock.hour < 12
    if word == "noon":
        return 11 <= clock.hour < 13
    return clock.hour >= 12  # afternoon, evening, dusk


def scan_answers(stems):
    """Time words and clock or date patterns in the raw text of every stored core-model answer."""
    scanned, elapsed, time_of_day = 0, [], []
    for s in stems:
        label = models.by_stem()[s].label
        for arm in ARMS:
            by_id, _ = read_arm(s, arm)
            scanned += len(by_id)
            for iid, row in by_id.items():
                text = row.get("raw") or ""
                if ELAPSED_TIME_WORDS.search(text) or CLOCK_OR_DATE.search(text):
                    elapsed.append({"model": label, "arm": arm, "item_id": iid})
                words = sorted({w.lower() for w in TIME_OF_DAY.findall(text)})
                if words:
                    clock = capture_clock(int(ITEM_EPOCH.search(iid).group(1)))
                    time_of_day.append({
                        "model": label, "arm": arm, "item_id": iid, "prediction": row.get("prediction"), "words": words,
                        "capture_clock": clock.strftime("%Y-%m-%d %H:%M %Z") if clock else None,
                        "matches_clock": all(matches_clock(w, clock) for w in words) if clock else None})
    return {
        "answers_scanned": scanned, "naming_elapsed_time_clock_or_date": len(elapsed), "elapsed_hits": elapsed,
        "naming_time_of_day": len(time_of_day),
        "time_of_day_no_smoke_answers": sum(1 for h in time_of_day if h["prediction"] is False),
        "time_of_day_words": sorted({w for h in time_of_day for w in h["words"]}),
        "time_of_day_matching_capture_clock": (sum(1 for h in time_of_day if h["matches_clock"])
                                               if all(h["matches_clock"] is not None for h in time_of_day) else None),
        "time_of_day_hits": time_of_day,
    }


def sweep(d):
    """Grounded-minus-bare smoke recall of the sixteen full-capability models on their paired items (the pairing and
    scorer of model_sweep.py), with the reference-to-target gaps those items carry."""
    targets = set(d["ids"])
    gap = dict(zip(d["ids"], d["gap"].tolist()))
    smoke = dict(zip(d["ids"], d["y"].tolist()))
    rows = []
    for m in models.models(tier="all", task="figlib"):
        bare, grounded = read_arm(m.stem, "bare")[0], read_arm(m.stem, "grounded")[0]
        ids = sorted(i for i in bare if i in grounded and bare[i].get("prediction") is not None
                     and grounded[i].get("prediction") is not None)
        sb, sg = fp.score(bare, ids)["counts"], fp.score(grounded, ids)["counts"]
        clear_gaps = [gap[i] for i in ids if i in targets and not smoke[i]]
        smoke_gaps = [gap[i] for i in ids if i in targets and smoke[i]]
        rows.append({
            "label": m.label, "stem": m.stem, "tier": m.tier, "paired": len(ids), "smoke_frames": sb["smoke_frames"],
            "found_delta": sg["smoke_true_positive"] - sb["smoke_true_positive"],
            "false_positive_delta": sg["no_smoke_false_positive"] - sb["no_smoke_false_positive"],
            "grounded_items_outside_targets": len(set(grounded) - targets),
            "gap_seconds": {"clear": [min(clear_gaps), max(clear_gaps)], "smoke": [min(smoke_gaps), max(smoke_gaps)]},
        })
    raised = [r["found_delta"] for r in rows if r["found_delta"] > 0]
    return {
        "models": len(rows), "raise_recall": len(raised),
        "found_delta_when_raised": [min(raised), max(raised)] if raised else None,
        "lower_recall": {r["label"]: r["found_delta"] for r in rows if r["found_delta"] < 0},
        "smoke_frames": {r["label"]: r["smoke_frames"] for r in rows},
        "raise_false_positive_rate": sum(1 for r in rows if r["false_positive_delta"] > 0),
        "every_grounded_item_is_a_target": all(r["grounded_items_outside_targets"] == 0 for r in rows),
        "gap_minutes": {k: [min(r["gap_seconds"][k][0] for r in rows) / 60.0,
                            max(r["gap_seconds"][k][1] for r in rows) / 60.0] for k in ("clear", "smoke")},
        "per_model": rows,
    }


# ----------------------------------------------------------------------------------------------
# Table tab:figlib-timing
# ----------------------------------------------------------------------------------------------


def fmt3(v, signed=False):
    """Three decimals as the table prints them: 0.000 for zero, a math-mode minus, a plus when signed."""
    if abs(v) < 5e-4:
        return "0.000"
    text = "%.3f" % abs(v)
    if v < 0:
        return "$-%s$" % text
    return "+" + text if signed else text


def fmt_ci(ci):
    return "[%s, %s]" % (fmt3(ci[0]), fmt3(ci[1]))


def table_rows(out, labels):
    """LaTeX rows of both panels of Table tab:figlib-timing."""
    positions = list(out["design"]["by_position"])
    pooled = out["rows"]["pooled"]
    det = out["detector"]
    lines = ["Gap (min) & " + " & ".join("%.0f" % out["design"]["by_position"][p]["gap_minutes_median"]
                                         for p in positions) + r" \\"]
    for label in labels:
        c = out["counts"][label]
        cells = ["%d/%d" % ((c[p]["gained"], c[p]["lost"]) if "gained" in c[p]
                            else (c[p]["new_false_positives"], c[p]["removed_false_positives"])) for p in positions]
        lines.append(label + " & " + " & ".join(cells) + r" \\")
    for name, key, signed in (("Pooled, bare", "bare", False), ("Pooled, grounded", "grounded", False),
                              ("Pooled change", "change", True)):
        lines.append(name + " & " + " & ".join(fmt3(pooled["by_position"][p][key]["point"], signed)
                                               for p in positions) + r" \\")
    lines.append(r"\quad 95\% interval & " + " & ".join(
        r"{\scriptsize " + fmt_ci(pooled["by_position"][p]["change"]["ci_widened"]) + "}" for p in positions) + r" \\")
    lines.append(r"\comprow Frame difference & " + " & ".join(fmt3(det["by_position"][p]["rate"])
                                                            for p in positions) + r" \\")

    def cell(est, per_model, show_ci):
        text = fmt3(est["point"], signed=True)
        if per_model and est["excludes_zero_widened"] and est["point"] > 0:
            text += r"$^{\blacktriangle}$"
        return text + (" " + fmt_ci(est["ci_widened"]) if show_ci else "")

    for label in labels + ["pooled"]:
        row, per_model = out["rows"][label], label != "pooled"
        ba = row["balanced_accuracy"]
        lines.append(("Pooled, core six" if label == "pooled" else label) + " & " + " & ".join([
            cell(row["slope_smoke_gain"], per_model, not per_model),
            cell(row["slope_clear_fpr_change"], per_model, not per_model),
            fmt3(ba["bare"]["point"]), fmt3(ba["grounded"]["point"]), cell(ba["change"], per_model, True)]) + r" \\")
    slopes = det["slope_per_10_min"]
    lines.append(r"\comprow Frame difference & " + cell(slopes["recall"], False, True) + " & "
                 + cell(slopes["false_positive_rate"], False, True) + " & -- & "
                 + fmt3(det["balanced_accuracy"]) + r" & -- \\")
    lines.append(r"\comprow Detector, matched & -- & " + cell(out["detector_matched"]["slope_per_10_min"], False, True)
                 + r" & -- & -- & -- \\")
    return lines


# ----------------------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--resamples", type=int, default=20000, help="bootstrap resamples (default 20000)")
    ap.add_argument("--seed", type=int, default=20260915, help="base seed (default 20260915)")
    ap.add_argument("--out", type=pathlib.Path, default=OUT, help="output JSON path")
    args = ap.parse_args()
    t0 = time.time()

    d = load_design()
    core = models.models(tier="core", task="figlib")
    stems, labels = [m.stem for m in core], [m.label for m in core]
    preds = paired_predictions(d, stems)
    n_fires = len(set(d["fires"]))

    out = {"settings": {
        "base_seed": args.seed, "resamples": args.resamples, "ci": 95.0, "fires": n_fires,
        "widening_factor": scc.compute_small_cluster_factor(n_fires), "gap_unit_seconds": GAP_UNIT,
        "core_models": labels, "frames": "task-figlib/images-1568",
        "seeds": {"rows": 'stable_seed(base, "task-figlib", <stem> or "pooled-core6", "smoke-timing")',
                  "detector": 'stable_seed(base, "task-figlib", "frame-difference", <statistic>)',
                  "detector_matched": 'stable_seed(base, "verify-w3", "matched-operating-point")'},
    }}
    out["design"] = design_summary(d)
    score, overlay = read_frames(d)
    out["design"]["overlay"] = overlay_summary(overlay)
    out["design"]["earlier_frames"] = earlier_frames(d)
    out["answers"] = scan_answers(stems)

    rows = [(label, [stem], stem) for stem, label in zip(stems, labels)] + [("pooled", stems, "pooled-core6")]
    out["counts"], out["rows"] = {}, {}
    for label, row_stems, seed_stem in rows:
        out["counts"][label] = position_counts(d, preds, row_stems)
        row = PairedRow(d, preds, row_stems)
        estimates = bootstrap(row.stats, d["fires"], args.resamples, row_seed(args.seed, seed_stem))
        out["rows"][label] = row.nested(estimates)

    out["detector"] = detector(d, score, args.resamples, args.seed)
    out["detector_matched"] = matched_detector(d, preds, stems, score, args.resamples, args.seed)
    out["newly_flagged_scores"] = newly_flagged_scores(d, preds, stems, score)
    out["sweep"] = sweep(d)

    args.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    ds, pooled, det, sw = out["design"], out["rows"]["pooled"], out["detector"], out["sweep"]
    print("FIgLib: %d frames, %d sequences, %d fires; reference at ladder %s, %.1f to %.1f min before the plume"
          % (ds["items"], ds["sequences"], ds["fires"], ds["reference"]["ladder_targets"],
             *ds["reference"]["minutes_before_plume"]))
    print("gap: clear %.2f to %.2f min, smoke %.2f to %.2f min; whole-minute thresholds %s label all %d targets"
          % (*ds["gap_minutes"]["clear"], *ds["gap_minutes"]["smoke"],
             ds["separating_thresholds"]["whole_minutes"], ds["targets"]["frames"]))
    ans = out["answers"]
    print("overlay strip on %d of %d frames; answers naming elapsed time, a clock, or a date: %d of %d; "
          "time of day: %d (%d no-smoke, %s matching the capture clock)"
          % (ds["overlay"]["frames_with_overlay"], ds["overlay"]["frames"], ans["naming_elapsed_time_clock_or_date"],
             ans["answers_scanned"], ans["naming_time_of_day"], ans["time_of_day_no_smoke_answers"],
             "n/a (no tz data)" if ans["time_of_day_matching_capture_clock"] is None
             else ans["time_of_day_matching_capture_clock"]))
    early = ds["earlier_frames"]
    print("earlier download: " + ("clear frame %.1f to %.1f min before the plume in %d of %d sequences"
                                  % (*early["minutes_before_plume"], early["sequences"], early["of"])
                                  if early["available"] else "task-figlib/images/ absent, check skipped"))
    print("pooled change by gap (widened 95% interval): " + "; ".join(
        "%.0f min %+.3f [%.3f, %.3f]" % (ds["by_position"][p]["gap_minutes_median"], v["change"]["point"],
                                        *v["change"]["ci_widened"]) for p, v in pooled["by_position"].items()))
    for key in ("slope_smoke_gain", "slope_clear_fpr_change", "slope_rescue_rate"):
        v = pooled[key]
        print("pooled %-22s %+.4f per 10 min, widened [%.4f, %.4f]" % (key, v["point"], *v["ci_widened"]))
    v = pooled["balanced_accuracy"]["change"]
    print("pooled balanced-accuracy change %+.4f, widened [%.4f, %.4f]" % (v["point"], *v["ci_widened"]))
    print("detector: accuracy %.3f, recall %.3f, false-positive rate %.3f, balanced accuracy %.3f; matched slope "
          "%+.4f [%.4f, %.4f]" % (det["accuracy"], det["recall"], det["false_positive_rate"], det["balanced_accuracy"],
                                  out["detector_matched"]["slope_per_10_min"]["point"],
                                  *out["detector_matched"]["slope_per_10_min"]["ci_widened"]))
    print("sweep: grounding raises smoke recall on %d of %d models, by %d to %d frames; lowers it on %s; "
          "false-positive rate up on %d; every grounded item a scored target: %s"
          % (sw["raise_recall"], sw["models"], *sw["found_delta_when_raised"],
             ", ".join("%s (%+d)" % kv for kv in sw["lower_recall"].items()) or "none",
             sw["raise_false_positive_rate"], sw["every_grounded_item_is_a_target"]))
    print("\nTable tab:figlib-timing rows:")
    for line in table_rows(out, labels):
        print(line)
    print("\nwrote %s in %.0f s" % (args.out.relative_to(ROOT) if args.out.is_relative_to(ROOT) else args.out,
                                     time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
