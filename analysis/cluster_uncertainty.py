#!/usr/bin/env python
"""Re-estimate the uncertainty of every bare-against-grounded contrast at the right clustering unit.

This answers round-1 TMLR OBJECTION A (wrong unit of analysis).  The reported frame-level and
day-level intervals treat items as independent.  They are not:

  FIgLib      224 frames come from 28 camera sequences covering only 17 distinct fires.  Frames
              inside one fire share terrain, plume, viewpoint and time of day, so they carry far
              less than 224 frames' worth of information.
  allocation  300 incident-days come from 245 ICS-209 incidents, some contributing two days.
  Mesogeos    386 cells share synoptic weather and geography with their spatial-temporal neighbours.

The fix is a cluster bootstrap: resample whole clusters with replacement rather than items.  The
pairing between the bare and the grounded arm is preserved, because one resampled index set is
scored under both conditions.  For contrast the script also runs the naive item-level bootstrap that
the round-1 analysis implied, and prints

    width(cluster interval) / width(naive item interval)

which is the factor by which the wrong unit was overstating precision.

TWO ratios are printed, and they answer different questions.

  CONTRAST ratio   applies to the paired bare-minus-grounded difference.  Pairing already removes
                   the shared cluster-level difficulty, because a hard fire is hard under both
                   conditions, so this ratio tends to sit near 1 and can fall below it when the two
                   arms disagree on only a handful of items spread across many clusters.
  LEVEL ratio      applies to each arm's own headline metric, which is what the round-1 tables
                   reported.  This is where the clustering objection lands hardest.

Quote the level ratio when defending a per-arm number and the contrast ratio when defending a
bare-against-grounded claim.  Reporting only one of them would misstate the correction.

NOTHING here is hardcoded from a previous run.  Models, conditions and items are rediscovered from
whatever responses-*.jsonl files are on disk, so the script can be re-executed unchanged after the
full model rerun overwrites them.

Usage
    python cluster_uncertainty.py
    python cluster_uncertainty.py --resamples 20000 --seed 7
    python cluster_uncertainty.py --meso-deg 2.0 --meso-time quarter   # blocking sensitivity
    python cluster_uncertainty.py --allow-unpaired                     # mid-rerun, do not fail

Because that rerun is live while this runs, the script also checks that the two files of an arm were
written by the same run.  An arm whose bare and grounded files are more than --max-skew-min apart is
printed, flagged CROSS-RUN, and kept out of every median: pairing a fresh bare arm against a stale
grounded arm lines up perfectly on item ids and produces a healthy looking number from data that does
not belong together, which is the only failure mode here that leaves no other trace.

Exit codes
    0  every task had its files and every model arm had both conditions
    1  a task directory, an items.jsonl, or one side of a model arm was missing or empty (all named
       on stdout).  A file that exists but holds no usable row counts as missing, because it carries
       the same information; --allow-unpaired downgrades both to a warning.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
import time
import zlib

import numpy as np

DEFAULT_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # the repository this script lives in
MIN_RESAMPLES = 2000

# ----------------------------------------------------------------------------------------------
# io helpers
# ----------------------------------------------------------------------------------------------


def read_jsonl(path):
    """Read a .jsonl file, skipping blank and unusable lines.  Returns (rows, n_bad).

    A line counts as unusable when it does not parse, and also when it parses to something that is
    not a JSON object.  The second case matters because a rerun writing over this file can be caught
    mid-flush: the tail of a partial write can land on a fragment that happens to be a valid scalar,
    and a scalar has no .get, so letting it through aborted the whole run with a traceback and lost
    the tasks that had not been reached yet.
    """
    rows, bad = [], 0
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            if isinstance(parsed, dict):
                rows.append(parsed)
            else:
                bad += 1
    return rows, bad


def dedupe_by_item(rows):
    """Keep the last row per item_id.  A rerun that appends rather than truncates leaves duplicates."""
    out = {}
    for row in rows:
        key = row.get("item_id")
        if key is not None:
            out[key] = row
    return out, len(rows) - len(out)


CONDITION_RE = re.compile(r"^(?P<model>.+?)-(?P<cond>bare|grounded)(?P<variant>-.*)?$")


def discover_arms(task_dir):
    """Map an arm name to {'bare': path, 'grounded': path, '_parts': (model, variant)}.

    Filenames are responses-<model>-<condition>.jsonl.  A trailing variant such as '-1024px' belongs
    to the arm rather than to the condition, so responses-claude-opus-5-bare-1024px.jsonl pairs with
    responses-claude-opus-5-grounded-1024px.jsonl and stays separate from the plain claude-opus-5 arm.
    """
    arms = collections.defaultdict(dict)
    if not os.path.isdir(task_dir):
        return arms
    for name in sorted(os.listdir(task_dir)):
        if not (name.startswith("responses-") and name.endswith(".jsonl")):
            continue
        stem = name[len("responses-"):-len(".jsonl")]
        match = CONDITION_RE.match(stem)
        if not match:
            continue
        model, variant = match.group("model"), (match.group("variant") or "")
        arm = model + variant
        arms[arm][match.group("cond")] = os.path.join(task_dir, name)
        arms[arm]["_parts"] = (model, variant)
    return arms


def expected_filename(parts, condition):
    model, variant = parts
    return "responses-%s-%s%s.jsonl" % (model, condition, variant)


# ----------------------------------------------------------------------------------------------
# cluster keys
# ----------------------------------------------------------------------------------------------


SEQ_RE = re.compile(r"^(?P<date>\d{8})_(?P<fire>.+)_(?P<camera>[^_]+)$")


def figlib_fire_key(sequence):
    """Derive the fire a FIgLib camera sequence watched, from the sequence name alone.

    HPWREN FIgLib names a sequence '<YYYYMMDD>_<FireName>_<cameraId>', for example
    '20260408_RainbowFir_rm-e-mobo-c'.  The first underscore field is always an eight digit date and
    camera ids are hyphenated and never carry an underscore, so the fire is everything between the
    first and the last underscore field.  On the current corpus that rule maps all 28 sequences onto
    17 fire names, exactly the count the round-1 review cites, and it reproduces the 'fire_name'
    column of items.jsonl on every row (the script checks this and warns on any mismatch).

    The date is deliberately dropped from the key.  'MissionFire' appears on 2026-05-02 and again on
    2026-06-18; keying on (date, fire) would split those and give 18 clusters.  Merging them is the
    conservative reading: fewer and larger clusters widen the interval, and two incidents sharing a
    name also share the camera network, terrain and vegetation that make frames dependent in the
    first place.  Splitting them would claim precision this design cannot support.
    """
    match = SEQ_RE.match(sequence or "")
    if match:
        return match.group("fire")
    # Unrecognised shape: fall back to the whole sequence, which is never coarser than one sequence.
    return sequence or "UNKNOWN"


ALLOC_ID_RE = re.compile(r"^alloc-(?P<incident>.+)-(?P<date>\d{4}-\d{2}-\d{2})$")


def allocation_incident_from_item_id(item_id):
    """Fallback incident key for an item absent from items.jsonl.

    Allocation item ids are 'alloc-<incident_id>-<YYYY-MM-DD>', for example
    'alloc-2015_2715639_WHITETAIL-2015-06-21'.  Incident ids contain underscores, hyphens and spaces,
    so the split anchors on the trailing ISO date rather than on a field count.
    """
    match = ALLOC_ID_RE.match(item_id or "")
    return match.group("incident") if match else (item_id or "UNKNOWN")


def mesogeos_block_key(lon, lat, target_date, deg, time_unit):
    """Spatial-temporal block for a Mesogeos cell.

    Group = (cell snapped to a `deg`-degree grid, calendar month of the target date).  The default
    `deg` is 1.0, about 110 km north to south, which is the scale over which the drivers this task
    serves (t2m, d2m, wind, relative humidity, soil moisture index) behave as one synoptic field, and
    one calendar month is the scale over which a fire-weather regime persists.  Two cells in the same
    block therefore see close to the same weather over close to the same land cover, which is the
    dependence the review objects to.  Month means year-and-month, not month-of-year, so 2021-07 and
    2022-07 stay separate; the sample spans two fire seasons and those are not interchangeable.

    Both dials are exposed (--meso-deg, --meso-time) because the honest answer on this task depends
    on them.  The evaluation cells are spatially dispersed, every one carries a distinct coordinate,
    and at 1 degree by month most blocks hold a single item, so the correction is small here by
    construction of the sample rather than by choice of estimator.  The script also reports the
    contrast at a deliberately coarse block (--meso-coarse-deg, --meso-coarse-time) so that the
    'your blocks were too small' follow-up is answered in the same table.
    """
    if lon is None or lat is None or not target_date:
        return None
    cell = (int(np.floor(float(lon) / deg)), int(np.floor(float(lat) / deg)))
    if time_unit == "quarter":
        stamp = "%s-Q%d" % (target_date[:4], (int(target_date[5:7]) - 1) // 3 + 1)
    elif time_unit == "year":
        stamp = target_date[:4]
    else:
        stamp = target_date[:7]
    return "%d/%d@%s" % (cell[0], cell[1], stamp)


# ----------------------------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------------------------


def average_precision(y_true, score):
    """Average precision, the step-wise sum used by sklearn.metrics.average_precision_score.

    Written out so the bootstrap can call it a few hundred thousand times without sklearn's per-call
    validation overhead.  _selfcheck_average_precision verifies it against sklearn at startup on the
    live data and aborts on any disagreement.
    """
    n_pos = float(y_true.sum())
    if n_pos == 0:
        return np.nan
    order = np.argsort(-score, kind="mergesort")
    y_sorted = y_true[order]
    s_sorted = score[order]
    # one breakpoint per distinct score, plus the final element
    distinct = np.flatnonzero(np.diff(s_sorted))
    idx = np.r_[distinct, y_sorted.size - 1]
    tps = np.cumsum(y_sorted)[idx]
    fps = (1.0 + idx) - tps
    precision = tps / np.maximum(tps + fps, 1e-12)
    recall = tps / n_pos
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def _selfcheck_average_precision(y_true, score):
    try:
        from sklearn.metrics import average_precision_score
    except ImportError:
        return "sklearn unavailable, average precision not cross-checked"
    mine = average_precision(y_true, score)
    theirs = float(average_precision_score(y_true, score))
    if not np.isclose(mine, theirs, atol=1e-12, rtol=1e-9):
        raise SystemExit("average_precision disagrees with sklearn: %r vs %r" % (mine, theirs))
    return "average precision matches sklearn to 1e-9 on the live data"


# ----------------------------------------------------------------------------------------------
# bootstrap machinery
# ----------------------------------------------------------------------------------------------


def build_cluster_index(group_ids):
    """Return (flat, starts, sizes, keys) for a ragged cluster -> item-index mapping."""
    order = collections.defaultdict(list)
    for position, gid in enumerate(group_ids):
        order[gid].append(position)
    keys = sorted(order)
    members = [np.asarray(order[k], dtype=np.int64) for k in keys]
    sizes = np.asarray([m.size for m in members], dtype=np.int64)
    starts = np.r_[0, np.cumsum(sizes)[:-1]].astype(np.int64) if sizes.size else np.empty(0, np.int64)
    flat = np.concatenate(members) if members else np.empty(0, dtype=np.int64)
    return flat, starts, sizes, keys


def ragged_gather(flat, starts, sizes, draw):
    """Concatenate the item indices of the drawn clusters, vectorised."""
    drawn_sizes = sizes[draw]
    total = int(drawn_sizes.sum())
    if total == 0:
        return np.empty(0, dtype=np.int64)
    base = np.repeat(starts[draw], drawn_sizes)
    cum = np.r_[0, np.cumsum(drawn_sizes)[:-1]]
    within = np.arange(total, dtype=np.int64) - np.repeat(cum, drawn_sizes)
    return flat[base + within]


def stable_seed(base_seed, *parts):
    """Deterministic per-row seed, so adding or removing a model never shifts another model's numbers.

    crc32 is used rather than hash() because hash() of a str is salted per process and would break
    reproducibility between runs.
    """
    text = "|".join(str(p) for p in parts).encode("utf-8")
    return int((int(base_seed) + zlib.crc32(text)) % (2 ** 32))


def bootstrap_arms(metric_bare, metric_grounded, group_ids, n_items, resamples, seed):
    """Resample once and score both conditions on every replicate.

    group_ids=None gives the naive item-level bootstrap (every item is its own cluster); otherwise
    whole clusters are drawn with replacement.  Scoring both conditions on the SAME resampled index
    set is what keeps the contrast paired, and it also lets the per-arm level intervals be read off
    the identical draws, so the level and contrast ratios are comparable.
    """
    if group_ids is None:
        group_ids = list(range(n_items))
    flat, starts, sizes, keys = build_cluster_index(group_ids)
    n_clusters = len(keys)
    rng = np.random.default_rng(seed)
    bare = np.empty(resamples, dtype=float)
    grounded = np.empty(resamples, dtype=float)
    for b in range(resamples):
        draw = rng.integers(0, n_clusters, size=n_clusters)
        idx = ragged_gather(flat, starts, sizes, draw)
        bare[b] = metric_bare(idx)
        grounded[b] = metric_grounded(idx)
    return bare, grounded, n_clusters


def percentile_interval(values, ci):
    """Percentile interval, ignoring replicates where the metric was undefined."""
    good = values[np.isfinite(values)]
    dropped = int(values.size - good.size)
    if good.size == 0:
        return (np.nan, np.nan, dropped)
    lo = (100.0 - ci) / 2.0
    return (float(np.percentile(good, lo)), float(np.percentile(good, 100.0 - lo)), dropped)


def width_ratio(cluster_values, naive_values, ci):
    """width(cluster percentile interval) / width(naive percentile interval)."""
    c_lo, c_hi, c_bad = percentile_interval(cluster_values, ci)
    n_lo, n_hi, n_bad = percentile_interval(naive_values, ci)
    c_w, n_w = c_hi - c_lo, n_hi - n_lo
    ratio = (c_w / n_w) if (np.isfinite(c_w) and np.isfinite(n_w) and n_w > 0) else np.nan
    return ratio, (c_lo, c_hi, c_bad), (n_lo, n_hi, n_bad)


# ----------------------------------------------------------------------------------------------
# per-task assembly
# ----------------------------------------------------------------------------------------------


def load_items(task_dir, task_name, problems):
    path = os.path.join(task_dir, "items.jsonl")
    if not os.path.isdir(task_dir):
        problems.append("%s: task directory missing (%s)" % (task_name, task_dir))
        return None
    if not os.path.isfile(path):
        problems.append("%s: items.jsonl missing (%s)" % (task_name, path))
        return None
    rows, bad = read_jsonl(path)
    if bad:
        print("  note: %s items.jsonl had %d unparseable lines" % (task_name, bad))
    return {r["item_id"]: r for r in rows if "item_id" in r}


def prepare_figlib(items, bare_rows, grounded_rows, cfg):
    """Aligned arrays for the smoke-recall contrast, clustered by fire."""
    common = sorted(set(bare_rows) & set(grounded_rows))
    pred_b, pred_g, is_smoke, groups, kept = [], [], [], [], []
    seq_mismatch = 0
    for iid in common:
        rb, rg = bare_rows[iid], grounded_rows[iid]
        if rb.get("prediction") is None or rg.get("prediction") is None:
            continue
        item = items.get(iid, {})
        sequence = item.get("sequence") or rb.get("sequence") or rg.get("sequence")
        label = item.get("label", rb.get("label"))
        if sequence is None or label is None:
            continue
        fire = figlib_fire_key(sequence)
        if item.get("fire_name") and item["fire_name"] != fire:
            seq_mismatch += 1
        pred_b.append(bool(rb["prediction"]))
        pred_g.append(bool(rg["prediction"]))
        is_smoke.append(str(label).strip().lower() == "smoke")
        groups.append(fire)
        kept.append(iid)
    pred_b = np.asarray(pred_b, dtype=bool)
    pred_g = np.asarray(pred_g, dtype=bool)
    is_smoke = np.asarray(is_smoke, dtype=bool)

    def make(pred):
        def metric(idx):
            mask = is_smoke[idx]
            if not mask.any():
                return np.nan
            return float(pred[idx][mask].mean())
        return metric

    return {
        "n": len(kept), "groups": groups, "metric_bare": make(pred_b), "metric_grounded": make(pred_g),
        "extra": {"smoke frames scored": int(is_smoke.sum())},
        "warn": ("%d rows where the fire derived from the sequence name disagreed with items.jsonl"
                 % seq_mismatch) if seq_mismatch else None,
    }


def prepare_allocation(items, bare_rows, grounded_rows, cfg):
    """Aligned arrays for normalised MAE, |target - prediction| / fire_mean, clustered by incident.

    The normalisation matches the repo's own analyze_allocation.py, which divides the absolute error
    by fire_mean, the incident's own mean personnel, so that a 20-person miss on a 30-person fire is
    not swamped by a 20-person miss on a 3000-person fire.
    """
    common = sorted(set(bare_rows) & set(grounded_rows))
    err_b, err_g, groups, kept = [], [], [], []
    target_conflicts, no_join = 0, 0
    for iid in common:
        rb, rg = bare_rows[iid], grounded_rows[iid]
        if rb.get("prediction") is None or rg.get("prediction") is None:
            continue
        # target and fire_mean are properties of the item; both arms record them, so cross-check.
        target = rb.get("target", rg.get("target"))
        scale = rb.get("fire_mean", rg.get("fire_mean"))
        if rb.get("target") is not None and rg.get("target") is not None and rb["target"] != rg["target"]:
            target_conflicts += 1
        if target is None or not scale:
            continue
        item = items.get(iid)
        if item is None:
            no_join += 1
            incident = allocation_incident_from_item_id(iid)
        else:
            incident = item.get("incident_id") or allocation_incident_from_item_id(iid)
        err_b.append(abs(float(target) - float(rb["prediction"])) / float(scale))
        err_g.append(abs(float(target) - float(rg["prediction"])) / float(scale))
        groups.append(incident)
        kept.append(iid)
    err_b = np.asarray(err_b, dtype=float)
    err_g = np.asarray(err_g, dtype=float)

    def make(err):
        def metric(idx):
            return float(err[idx].mean()) if idx.size else np.nan
        return metric

    warn = []
    if no_join:
        warn.append("%d items absent from items.jsonl, incident taken from the item id" % no_join)
    if target_conflicts:
        warn.append("%d items where the two arms disagree on the target" % target_conflicts)
    return {
        "n": len(kept), "groups": groups, "metric_bare": make(err_b), "metric_grounded": make(err_g),
        "extra": {}, "warn": "; ".join(warn) if warn else None,
    }


def mesogeos_score(row):
    """Score used for average precision, mirroring run_mesogeos.py: probability, else the hard call."""
    if row.get("probability") is not None:
        return float(row["probability"])
    if row.get("call") is not None:
        return 1.0 if row["call"] else 0.0
    return None


def prepare_mesogeos(items, bare_rows, grounded_rows, cfg):
    """Aligned arrays for average precision, clustered by spatial-temporal block."""
    common = sorted(set(bare_rows) & set(grounded_rows))
    score_b, score_g, labels, groups, coarse, kept = [], [], [], [], [], []
    no_join, no_block = 0, 0
    for iid in common:
        rb, rg = bare_rows[iid], grounded_rows[iid]
        sb, sg = mesogeos_score(rb), mesogeos_score(rg)
        if sb is None or sg is None:
            continue
        item = items.get(iid)
        if item is None:
            no_join += 1
            continue
        label = item.get("label", rb.get("label"))
        context = item.get("context") or {}
        date = item.get("target_date") or rb.get("target_date")
        block = mesogeos_block_key(context.get("longitude"), context.get("latitude"), date,
                                   cfg.meso_deg, cfg.meso_time)
        if label is None or block is None:
            no_block += 1
            continue
        score_b.append(sb)
        score_g.append(sg)
        labels.append(int(label))
        groups.append(block)
        coarse.append(mesogeos_block_key(context.get("longitude"), context.get("latitude"), date,
                                         cfg.meso_coarse_deg, cfg.meso_coarse_time))
        kept.append(iid)
    score_b = np.asarray(score_b, dtype=float)
    score_g = np.asarray(score_g, dtype=float)
    labels = np.asarray(labels, dtype=np.int64)

    def make(score):
        def metric(idx):
            if idx.size == 0:
                return np.nan
            return average_precision(labels[idx], score[idx])
        return metric

    warn = []
    if no_join:
        warn.append("%d items absent from items.jsonl, dropped (no coordinates to block on)" % no_join)
    if no_block:
        warn.append("%d items without usable coordinates or label, dropped" % no_block)
    return {
        "n": len(kept), "groups": groups, "groups_coarse": coarse,
        "metric_bare": make(score_b), "metric_grounded": make(score_g),
        "extra": {"positive cells scored": int(labels.sum())},
        "warn": "; ".join(warn) if warn else None,
        "selfcheck": (labels, score_b),
    }


TASKS = [
    {
        "name": "task-figlib",
        "label": "FIgLib",
        "metric": "recall on smoke",
        "unit_name": "fire",
        "higher_is_better": True,
        "unit": "fire, derived from the sequence name",
        "prepare": prepare_figlib,
    },
    {
        "name": "task-allocation",
        "label": "allocation",
        "metric": "normalised MAE",
        "unit_name": "incident",
        "higher_is_better": False,
        "unit": "incident (incident_id from items.jsonl)",
        "prepare": prepare_allocation,
    },
    {
        "name": "task-mesogeos",
        "label": "Mesogeos",
        "metric": "average precision",
        "unit_name": "block",
        "higher_is_better": True,
        "unit": None,  # filled from the cli, which names the block size
        "prepare": prepare_mesogeos,
    },
]


# ----------------------------------------------------------------------------------------------
# reporting
# ----------------------------------------------------------------------------------------------


def fmt(value, width=7, places=4):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "%*s" % (width, "n/a")
    return "%*.*f" % (width, places, value)


def stamp(epoch):
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(epoch))


def small_cluster_factor(n_clusters, ci):
    """How much wider a coverage-corrected interval would be at this cluster count.

    A percentile cluster bootstrap undercovers when the number of clusters is small, because the
    resampling sees only that many independent draws.  The usual correction is to scale by
    sqrt(G/(G-1)) and read the quantile off t with G-1 degrees of freedom rather than the normal.
    The factor is reported rather than applied: the brief asks for a percentile interval, and the
    width RATIO this script exists to produce is close to unaffected, since the same factor would
    widen the cluster side of every task.  It matters for the interval itself, which is why FIgLib
    at 17 fires gets the note printed.
    """
    if n_clusters is None or n_clusters < 2:
        return None
    try:
        from scipy import stats
    except ImportError:
        return None
    q = 1.0 - (1.0 - ci / 100.0) / 2.0
    return float(stats.t.ppf(q, n_clusters - 1) / stats.norm.ppf(q) * np.sqrt(n_clusters / (n_clusters - 1.0)))


def cluster_summary(groups):
    counts = collections.Counter(groups)
    sizes = np.asarray(sorted(counts.values())) if counts else np.zeros(0, dtype=int)
    in_multi = int(sizes[sizes > 1].sum()) if sizes.size else 0
    return {
        "n_clusters": len(counts),
        "max_size": int(sizes.max()) if sizes.size else 0,
        "mean_size": float(sizes.mean()) if sizes.size else 0.0,
        "share_in_multi": (in_multi / float(sizes.sum())) if sizes.sum() else 0.0,
    }


def median_or_nan(values):
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    arr = arr[np.isfinite(arr)]
    return (float(np.median(arr)), arr.size, float(arr.min()), float(arr.max())) if arr.size else (np.nan, 0, np.nan, np.nan)


def run_task(spec, cfg, problems):
    task_dir = os.path.join(cfg.repo, spec["name"])
    unit = spec["unit"] or ("spatial-temporal block, %.2f deg cell x calendar %s"
                            % (cfg.meso_deg, cfg.meso_time))
    print("-" * 124)
    print("%s  |  primary metric: %s  |  resampling unit: %s" % (spec["label"], spec["metric"], unit))
    print("-" * 124)

    items = load_items(task_dir, spec["name"], problems)
    if items is None:
        print("  SKIPPED, see the problem list at the end\n")
        return None

    arms = discover_arms(task_dir)
    if cfg.manifest_files is not None:
        filtered_arms = collections.defaultdict(dict)
        for arm, have in arms.items():
            if "bare" in have and os.path.normpath(os.path.relpath(have["bare"], cfg.repo)) not in cfg.manifest_files:
                continue
            if "grounded" in have and os.path.normpath(os.path.relpath(have["grounded"], cfg.repo)) not in cfg.manifest_files:
                continue
            filtered_arms[arm] = have
        arms = filtered_arms
    if not arms:
        problems.append("%s: no responses-*.jsonl files found in %s" % (spec["name"], task_dir))
        print("  SKIPPED, no responses-*.jsonl files\n")
        return None

    paired_arms = []
    for arm in sorted(arms):
        have = arms[arm]
        if "bare" in have and "grounded" in have:
            paired_arms.append(arm)
        else:
            present = "bare" if "bare" in have else "grounded"
            missing = "grounded" if present == "bare" else "bare"
            msg = ("%s: arm '%s' has only the %s condition, %s is missing"
                   % (spec["name"], arm, present,
                      os.path.join(task_dir, expected_filename(have["_parts"], missing))))
            if cfg.allow_unpaired or arm.startswith("baseline-"):
                print("  warning: " + msg)
            else:
                problems.append(msg)

    if not paired_arms:
        problems.append("%s: no model arm has both a bare and a grounded responses file" % spec["name"])
        print("  SKIPPED, no arm has both conditions\n")
        return None

    rows, selfcheck_note, n_stale = [], None, 0
    for arm in paired_arms:
        bare_raw, bad_b = read_jsonl(arms[arm]["bare"])
        grd_raw, bad_g = read_jsonl(arms[arm]["grounded"])
        bare_rows, dup_b = dedupe_by_item(bare_raw)
        grd_rows, dup_g = dedupe_by_item(grd_raw)

        # Freshness.  The two files of one arm are written a few minutes apart by a normal run, so a
        # large gap means the rerun has replaced one condition and not yet the other.  Pairing across
        # that boundary silently contrasts a new bare arm against an old grounded arm on item ids that
        # still line up perfectly, which is the one failure mode here that produces a healthy looking
        # number from data that does not belong together.  Such an arm is printed and excluded from
        # the task medians rather than fatal, so the script stays re-executable while a rerun is live.
        m_bare = os.path.getmtime(arms[arm]["bare"])
        m_grd = os.path.getmtime(arms[arm]["grounded"])
        skew_min = abs(m_bare - m_grd) / 60.0
        stale = skew_min > cfg.max_skew_min
        grd_rel = os.path.normpath(os.path.relpath(arms[arm]["grounded"], cfg.repo))
        if cfg.manifest_skew_overrides and grd_rel in cfg.manifest_skew_overrides:
            stale = False

        prep = spec["prepare"](items, bare_rows, grd_rows, cfg)
        notes = []
        if bad_b or bad_g:
            notes.append("%d unparseable lines" % (bad_b + bad_g))
        if dup_b or dup_g:
            notes.append("%d duplicate item_ids, last kept" % (dup_b + dup_g))
        if prep.get("warn"):
            notes.append(prep["warn"])
        for key, val in prep["extra"].items():
            notes.append("%s: %d" % (key, val))
        notes.append("written bare %s, grounded %s (%.0f min apart)"
                     % (stamp(m_bare), stamp(m_grd), skew_min))

        record = {"arm": arm, "n_bare": len(bare_rows), "n_grd": len(grd_rows),
                  "n_pair": prep["n"], "notes": notes, "prep": prep, "stale": stale}
        if stale:
            n_stale += 1

        # A condition file that exists but holds no usable row carries exactly as much information as
        # a missing one, so it is reported the same way.  Without this, deleting a grounded file exited
        # 1 while truncating it to zero bytes exited 0, and a rerun passes through the zero-byte state.
        starved = [c for c, n in (("bare", len(bare_rows)), ("grounded", len(grd_rows))) if n == 0]
        if starved:
            msg = ("%s: arm '%s' has a %s responses file with no usable rows (%s)"
                   % (spec["name"], arm, " and ".join(starved),
                      ", ".join(arms[arm][c] for c in starved)))
            if cfg.allow_unpaired:
                print("  warning: " + msg)
            else:
                problems.append(msg)

        if prep["n"] == 0:
            record["empty"] = True
            rows.append(record)
            continue

        if selfcheck_note is None and prep.get("selfcheck") is not None:
            labels, scores = prep["selfcheck"]
            if labels.sum() > 0:
                selfcheck_note = _selfcheck_average_precision(labels, scores)

        full = np.arange(prep["n"], dtype=np.int64)
        record["point_bare"] = prep["metric_bare"](full)
        record["point_grd"] = prep["metric_grounded"](full)
        record["point_diff"] = record["point_bare"] - record["point_grd"]
        record["struct"] = cluster_summary(prep["groups"])

        cl_b, cl_g, n_clusters = bootstrap_arms(
            prep["metric_bare"], prep["metric_grounded"], prep["groups"], prep["n"],
            cfg.resamples, stable_seed(cfg.seed, spec["name"], arm, "cluster"))
        nv_b, nv_g, _ = bootstrap_arms(
            prep["metric_bare"], prep["metric_grounded"], None, prep["n"],
            cfg.resamples, stable_seed(cfg.seed, spec["name"], arm, "naive"))
        record["n_clusters"] = n_clusters

        record["ratio_diff"], record["ci_cluster"], record["ci_naive"] = width_ratio(
            cl_b - cl_g, nv_b - nv_g, cfg.ci)
        record["ratio_level_bare"], _, _ = width_ratio(cl_b, nv_b, cfg.ci)
        record["ratio_level_grd"], _, _ = width_ratio(cl_g, nv_g, cfg.ci)

        if prep.get("groups_coarse"):
            co_b, co_g, n_coarse = bootstrap_arms(
                prep["metric_bare"], prep["metric_grounded"], prep["groups_coarse"], prep["n"],
                cfg.resamples, stable_seed(cfg.seed, spec["name"], arm, "coarse"))
            record["n_coarse"] = n_coarse
            record["ratio_coarse_diff"], _, _ = width_ratio(co_b - co_g, nv_b - nv_g, cfg.ci)
            record["ratio_coarse_level"], _, _ = width_ratio(co_b, nv_b, cfg.ci)

        # An arm too small to estimate a design effect, or one whose two conditions come from
        # different runs, does not enter the task medians.
        big_enough = (n_clusters >= cfg.min_clusters and prep["n"] >= cfg.min_pair)
        record["trusted"] = big_enough and not stale
        if not big_enough:
            record["flag"] = ("only %d %ss / %d paired items, below --min-clusters %d or --min-pair %d"
                              % (n_clusters, spec["unit_name"], prep["n"], cfg.min_clusters, cfg.min_pair))
        elif stale:
            record["flag"] = ("CROSS-RUN: the two conditions were written %.0f min apart, above "
                              "--max-skew-min %g, so this contrast mixes two different runs"
                              % (skew_min, cfg.max_skew_min))
        rows.append(record)

    biggest = max((r for r in rows if not r.get("empty")), key=lambda r: r["n_pair"], default=None)
    if biggest is not None:
        s = biggest["struct"]
        print("  cluster structure on the largest arm (%s): %d %ss over %d paired items, "
              "mean %.2f and max %d items per %s, %.0f%% of items share their %s with a neighbour"
              % (biggest["arm"], s["n_clusters"], spec["unit_name"], biggest["n_pair"],
                 s["mean_size"], s["max_size"], spec["unit_name"],
                 100.0 * s["share_in_multi"], spec["unit_name"]))
    if selfcheck_note:
        print("  metric check: %s" % selfcheck_note)
    if biggest is not None and biggest["struct"]["n_clusters"] < 30:
        factor = small_cluster_factor(biggest["struct"]["n_clusters"], cfg.ci)
        if factor and factor > 1.01:
            print("  small-cluster note: %d %ss is few enough that a percentile cluster bootstrap "
                  "undercovers.  Scaling by sqrt(G/(G-1)) and reading the quantile off t with %d "
                  "degrees of freedom would widen the cluster interval by %.0f%%.  The width ratios "
                  "below are close to unaffected; the cluster interval itself is optimistic by about "
                  "that much, and the paper should say so."
                  % (biggest["struct"]["n_clusters"], spec["unit_name"],
                     biggest["struct"]["n_clusters"] - 1, 100.0 * (factor - 1.0)))
    print()

    direction = ("grounded helps when diff < 0" if spec["higher_is_better"]
                 else "grounded helps when diff > 0")
    print("  PAIRED CONTRAST   diff = bare minus grounded in %s; %s" % (spec["metric"], direction))
    header = ("  %-30s %6s %6s %6s %6s %8s %8s %8s  %-21s %-21s %6s"
              % ("arm", "bare_n", "grd_n", "pair_n", "n_" + spec["unit_name"][:4],
                 "bare", "grounded", "diff",
                 "cluster %g%% CI" % cfg.ci, "naive item %g%% CI" % cfg.ci, "ratio"))
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in rows:
        if r.get("empty"):
            print("  %-30s %6d %6d %6d   no item was answered under both conditions"
                  % (r["arm"][:30], r["n_bare"], r["n_grd"], 0))
            continue
        cl, nv = r["ci_cluster"], r["ci_naive"]
        flag = ""
        if r.get("flag"):
            flag = "  <- " + r["flag"]
        elif cl[2] or nv[2]:
            flag = "  <- %d of %d replicates undefined and dropped" % (cl[2] + nv[2], 2 * cfg.resamples)
        print("  %-30s %6d %6d %6d %6d %s %s %s  [%s,%s] [%s,%s] %s%s"
              % (r["arm"][:30], r["n_bare"], r["n_grd"], r["n_pair"], r["n_clusters"],
                 fmt(r["point_bare"], 8), fmt(r["point_grd"], 8), fmt(r["point_diff"], 8),
                 fmt(cl[0], 7, 4), fmt(cl[1], 7, 4), fmt(nv[0], 7, 4), fmt(nv[1], 7, 4),
                 fmt(r["ratio_diff"], 6, 2), flag))
        for note in r["notes"]:
            print("  %-30s   note: %s" % ("", note))

    print()
    print("  PER-ARM LEVELS    width ratio for each condition's own %s, the quantity the round-1 tables reported"
          % spec["metric"])
    print("  %-30s %14s %14s" % ("arm", "bare level", "grounded level"))
    print("  " + "-" * 60)
    for r in rows:
        if r.get("empty"):
            continue
        suffix = "   <- not trusted, see above" if not r["trusted"] else ""
        print("  %-30s %s %s%s" % (r["arm"][:30], fmt(r["ratio_level_bare"], 13, 2),
                                   fmt(r["ratio_level_grd"], 13, 2), suffix))

    trusted = [r for r in rows if not r.get("empty") and r["trusted"]]
    summary = {"label": spec["label"], "unit": spec["unit_name"], "stale": n_stale}
    med_d, n_d, lo_d, hi_d = median_or_nan([r["ratio_diff"] for r in trusted])
    med_l, n_l, lo_l, hi_l = median_or_nan(
        [v for r in trusted for v in (r["ratio_level_bare"], r["ratio_level_grd"])])
    summary.update(contrast=med_d, contrast_n=n_d, level=med_l, level_n=n_l)

    print()
    if n_d:
        print("  WIDTH RATIO, %s, contrast : median %.2fx over %d arms (range %.2fx to %.2fx)"
              % (spec["label"], med_d, n_d, lo_d, hi_d))
        print("  WIDTH RATIO, %s, levels   : median %.2fx over %d arm-conditions (range %.2fx to %.2fx)"
              % (spec["label"], med_l, n_l, lo_l, hi_l))
    else:
        print("  WIDTH RATIO, %s: not computable, no arm cleared --min-clusters / --min-pair"
              % spec["label"])

    coarse = [r for r in trusted if r.get("ratio_coarse_diff") is not None]
    if coarse:
        med_cd, _, _, _ = median_or_nan([r["ratio_coarse_diff"] for r in coarse])
        med_cl, _, _, _ = median_or_nan([r["ratio_coarse_level"] for r in coarse])
        n_coarse = coarse[0].get("n_coarse")
        print("  BLOCKING SENSITIVITY: at a deliberately coarse %.2f deg x %s block (%d blocks), "
              "contrast %.2fx and levels %.2fx"
              % (cfg.meso_coarse_deg, cfg.meso_coarse_time, n_coarse, med_cd, med_cl))
        summary.update(coarse_contrast=med_cd, coarse_level=med_cl)
    print()
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default=DEFAULT_REPO, help="fire-bench checkout to read")
    parser.add_argument("--resamples", type=int, default=20000,
                        help="bootstrap replicates per interval (clamped up to %d).  The default is high "
                             "because FIgLib has only 17 clusters, which makes the percentile tails noisy: "
                             "at 2000 replicates the FIgLib ratio moved by 0.07 between counts, at 20000 it "
                             "is stable to about 0.02 across seeds, and the whole run still takes under a "
                             "minute" % MIN_RESAMPLES)
    parser.add_argument("--seed", type=int, default=20260915,
                        help="base seed; every row derives its own by crc32, so the set of files present "
                             "does not shift any other row's numbers")
    parser.add_argument("--ci", type=float, default=95.0, help="percentile interval level")
    parser.add_argument("--meso-deg", type=float, default=1.0,
                        help="Mesogeos spatial block size in degrees (default 1.0, about 110 km)")
    parser.add_argument("--meso-time", choices=["month", "quarter", "year"], default="month",
                        help="Mesogeos temporal block (default month)")
    parser.add_argument("--meso-coarse-deg", type=float, default=2.0,
                        help="deliberately coarse Mesogeos block for the sensitivity line")
    parser.add_argument("--meso-coarse-time", choices=["month", "quarter", "year"], default="quarter",
                        help="temporal half of the coarse sensitivity block")
    parser.add_argument("--min-clusters", type=int, default=5,
                        help="an arm with fewer clusters is printed but excluded from the task medians")
    parser.add_argument("--min-pair", type=int, default=30,
                        help="an arm with fewer paired items is printed but excluded from the task medians")
    parser.add_argument("--max-skew-min", type=float, default=60.0,
                        help="an arm whose bare and grounded files were written more than this many "
                             "minutes apart is treated as a cross-run pair: printed, flagged, and kept "
                             "out of the task medians.  A normal run writes the two a few minutes apart")
    parser.add_argument("--allow-unpaired", action="store_true",
                        help="warn instead of failing when an arm has only one of the two conditions, "
                             "or when a condition file exists but holds no usable row")
    parser.add_argument("--manifest", type=str, default=None,
                        help="path to manifest JSON restricting response files to reported runs and specifying skew overrides")
    cfg = parser.parse_args()

    cfg.manifest_files = None
    cfg.manifest_skew_overrides = set()
    if cfg.manifest:
        with open(cfg.manifest, encoding="utf-8") as f:
            m_data = json.load(f)
        cfg.manifest_files = set()
        for r in m_data.get("reported", []):
            norm_p = os.path.normpath(r["path"])
            cfg.manifest_files.add(norm_p)
            if r.get("skew_override"):
                cfg.manifest_skew_overrides.add(norm_p)

    if cfg.resamples < MIN_RESAMPLES:
        print("note: raising --resamples from %d to the %d minimum" % (cfg.resamples, MIN_RESAMPLES))
        cfg.resamples = MIN_RESAMPLES

    print("=" * 124)
    print("CLUSTER-ROBUST UNCERTAINTY FOR THE BARE-AGAINST-GROUNDED CONTRAST   (TMLR round-1 objection A)")
    print("=" * 124)
    print("repo      : %s" % cfg.repo)
    print("bootstrap : %d replicates per interval, %.0f%% percentile, base seed %d"
          % (cfg.resamples, cfg.ci, cfg.seed))
    print("pairing   : each contrast is scored only on the items BOTH conditions answered, and one")
    print("            resampled index set is scored under both conditions, which keeps it paired")
    print("ratios    : contrast ratio applies to the bare-minus-grounded difference; level ratio applies")
    print("            to each condition's own headline metric.  They differ because pairing already")
    print("            cancels the shared cluster-level difficulty.  Quote the level ratio when")
    print("            defending a per-arm number and the contrast ratio when defending a contrast.")
    print("freshness : an arm whose two condition files were written more than %g min apart is flagged"
          % cfg.max_skew_min)
    print("            CROSS-RUN and excluded from every median, since the rerun replaced one side only")
    print()

    problems, summaries = [], []
    for spec in TASKS:
        summary = run_task(spec, cfg, problems)
        if summary:
            summaries.append(summary)

    print("=" * 124)
    print("SUMMARY, interval-width inflation from resampling the right unit")
    print("=" * 124)
    if summaries:
        def as_x(value):
            if value is None or not np.isfinite(value):
                return "n/a"
            return "%.2fx" % value
        print("  %-12s %-10s %15s %14s  %s" % ("task", "unit", "contrast ratio", "level ratio",
                                               "cross-run arms excluded"))
        print("  " + "-" * 80)
        for s in summaries:
            print("  %-12s %-10s %15s %14s  %s" % (s["label"], s["unit"],
                                                   as_x(s.get("contrast")), as_x(s.get("level")),
                                                   s.get("stale") or ""))
    else:
        print("  nothing computable")
    print()
    total_stale = sum(s.get("stale", 0) for s in summaries)
    if total_stale:
        print("  %d arm(s) were excluded because their bare and grounded files were written more than" % total_stale)
        print("  %g minutes apart, which means the rerun has replaced one condition and not the other." % cfg.max_skew_min)
        print("  Those rows are printed above but do not enter any median.  Re-run once the rerun has")
        print("  written both conditions of every arm, or raise --max-skew-min if the gap is expected.")
        print()
    print("  A ratio above 1 means the cluster bootstrap is wider, that is, the frame-level and")
    print("  day-level intervals in the round-1 submission overstated precision by that factor.")
    print("  A contrast ratio at or below 1 is a real result, not a bug: when the two arms disagree")
    print("  on only a few items and those disagreements are spread across clusters, the paired")
    print("  difference carries almost no cluster-level variance for the resampling to recover.")
    print()
    print("  PROVISIONAL.  Every number above was computed from the responses-*.jsonl files present at")
    print("  run time.  A full model rerun is in progress and will overwrite those files.  Re-execute")
    print("  this script unchanged afterwards; the numbers it prints then supersede these.")
    print()

    if problems:
        print("=" * 124)
        print("MISSING OR INCOMPLETE FILES")
        print("=" * 124)
        for item in problems:
            print("  - %s" % item)
        print()
        print("  exiting non-zero; pass --allow-unpaired to downgrade a one-sided arm to a warning")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
