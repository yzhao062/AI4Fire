"""Information-only baselines for the three AI4Fire tasks (TMLR round-2, objection B).

On each task the grounded arm is handed something extra. This script scores a trivial predictor that sees
ONLY that extra thing, with no language model anywhere in the loop, and prints it beside every model run on
exactly the same items. A model that fails to beat its own grounding did not use the grounding; it was
carried by it.

    python information_only_baselines.py
    python information_only_baselines.py --repo <path> --tasks allocation mesogeos figlib

  allocation  the grounded arm gets up to 6 historical analogues, each a (personnel today, personnel next
              day) pair, plus their median ratio. Analogue-only predictor: today's personnel times the
              median analogue ratio. The analogue values are rebuilt through run_allocation.build_pool, keyed
              on the analogue_ids the run recorded, so the baseline reads exactly the rows the model read.
  mesogeos    the grounded arm gets each driver's climatology for that calendar month. Climatology-only
              predictor: the standardized anomaly of the last day's driver against that monthly mean and sd.
              Raw last-day t2m is reported as the reference point that uses no climatology at all.
  figlib      the grounded arm gets one earlier clear frame from the same camera. Pixel-difference predictor:
              the largest 16x16 block-mean absolute grayscale difference between the frame to judge and that
              reference frame. Its threshold is fitted leave-one-FIRE-out, since the 28 sequences come from
              only 17 fires and five of those fires appear under several cameras. Skipped, loudly, when the
              cached images are not on disk.

No number in this file is hardcoded from any particular run. Every score is recomputed from whatever
responses-*.jsonl files are present at the moment the script runs, so it stays valid after the models are
rerun and the response files are overwritten. It will therefore disagree with a stale task-*/scores.json,
which is the point.

Two things the script refuses to do quietly. It never puts two arms side by side under one column heading
unless they answered the same items; when they did not, it repeats the grounded arm restricted to the overlap
and labels both rows 'paired'. And it flags a run whose response file holds far fewer rows than the largest
run of the same task and condition, because a rerun in flight leaves exactly that.

Exits non-zero, naming what is absent, when a required input is missing. A response file whose last line is
half written counts as missing and is named. The FIgLib image cache is the one declared-optional input: its
absence prints a line and skips that one baseline.
"""
import argparse
import collections
import hashlib
import json
import math
import pathlib
import sys

import numpy as np

DEFAULT_REPO = pathlib.Path(__file__).resolve().parent.parent  # the repository this script lives in
HERE = pathlib.Path(__file__).resolve().parent
MISSING = []          # required inputs that were not found; a non-empty list means exit 1
SKIPPED = []          # declared-optional inputs that were not found
SUMMARY = {}          # machine-readable mirror of everything printed


def need(path, what):
    """Record a required input as missing and return False, or return True when it is there."""
    if pathlib.Path(path).exists():
        return True
    MISSING.append("%s  (%s)" % (path, what))
    return False


def head(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a".rjust(nd + 4)
    return ("%." + str(nd) + "f") % x


def rnd_int(x):
    """Round half up, the way a model asked for an integer head count would."""
    return float(math.floor(x + 0.5))


CLUSTER_NOTE = ("note: every margin above is a point estimate over items that are not independent (fires, "
                "sequences, cells).\n      Whether a margin survives a cluster-aware test is a separate "
                "question and a separate script.")


class InputError(Exception):
    """A required input is present but unreadable. Carries the text that goes in the MISSING list."""


def partial_runs(row_counts):
    """Runs whose response file holds far fewer rows than the largest run of the same task and condition.

    A rerun writing the directory right now, or a capped debug run left behind, both look like this. Scoring
    such a file is not wrong, but presenting it beside a full run without a word is.
    """
    if not row_counts:
        return {}
    top = max(row_counts.values())
    return {k: v for k, v in row_counts.items() if v < 0.9 * top}


def warn_partial(short, row_counts):
    flagged = partial_runs(row_counts)
    top = max(row_counts.values()) if row_counts else 0
    for k, v in sorted(flagged.items()):
        print("   ** PARTIAL RUN: %s holds %d rows against %d for the largest %s run in this directory. "
              "Its numbers\n      are an unrepresentative slice, not a peer result; read them as such or "
              "re-run that model." % (k, v, top, short))
    return flagged


def verdict(metric, model_value, base_value, lower_is_better, nd=3, unit=""):
    """State the gap between a model run and the rule that saw only its grounding. No mechanism is claimed:
    beating the rule is necessary for 'the model used the information', never sufficient."""
    gain = (base_value - model_value) if lower_is_better else (model_value - base_value)
    who = "model ahead" if gain > 0 else ("information-only rule ahead" if gain < 0 else "tied")
    print(("   verdict  %s: model %." + str(nd) + "f vs information-only %." + str(nd) + "f  ->  %s by %."
           + str(nd) + "f%s") % (metric, model_value, base_value, who, abs(gain), unit))


def read_jsonl(path):
    rows = []
    for n, line in enumerate(pathlib.Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except ValueError as exc:
            # A run writing this file at this moment leaves a half-written final line. Name the file: a decode
            # traceback from inside json is not a diagnosis, and this script is meant to run during a rerun.
            raise InputError("%s  (line %d is not valid JSON: %s -- a run may be writing this file now)"
                             % (path, n, exc)) from None
    return rows


def paired_split(ids_here, other_rows, ok):
    """Rows of the other arm, the ids both arms answered, and how the two sets relate.

    The two arms do not answer the same items by construction: on FIgLib the grounded arm drops each
    sequence's reference frame, and on any task the two arms fail to parse on different items. Restricting one
    arm to the other and labelling the result 'same items' is false whenever either arm lost an item.

    Three cases, because they need different rows printed. 'same' needs nothing. 'covered' means this arm's
    items are all present in the other, so restricting the other arm is a true pairing and this arm's own row
    stands. 'split' means each arm answered something the other did not, and then both rows have to be
    recomputed on the overlap.
    """
    other = {r["item_id"]: r for r in other_rows if ok(r)}
    mine, both = set(ids_here), set(ids_here) & set(other)
    rel = "same" if set(other) == mine else ("covered" if both == mine else "split")
    return other, both, rel


def response_files(task_dir, condition):
    """Every responses-<model>-<condition>.jsonl in the task directory, newest naming scheme included."""
    out = {}
    for p in sorted(pathlib.Path(task_dir).glob("responses-*-%s.jsonl" % condition)):
        model = p.name[len("responses-"):-len("-%s.jsonl" % condition)]
        out[model] = p
    return out


# ----------------------------------------------------------------------------------------------------------
# allocation
# ----------------------------------------------------------------------------------------------------------

def allocation(repo, run_allocation, cache_path):
    task = repo / "task-allocation"
    items_path = task / "items.jsonl"
    sitreps = pathlib.Path(run_allocation.SIT)
    ok = need(task, "task-allocation directory")
    ok = need(items_path, "allocation item pool, needed for the analogue draw") and ok
    ok = need(sitreps, "ICS-209 sitreps CSV, needed by run_allocation.build_pool") and ok
    if not ok:
        return

    grounded = response_files(task, "grounded")
    bare = response_files(task, "bare")
    if not grounded:
        MISSING.append("%s  (no responses-*-grounded.jsonl; the grounded arm is what this baseline mirrors)"
                       % (task / "responses-<model>-grounded.jsonl"))
        return

    items = {it["item_id"]: it for it in read_jsonl(items_path)}

    # Every item_id any grounded run scored. build_pool must exclude exactly these incidents, as the run did.
    scored_ids, wanted_analogues = set(), set()
    for path in grounded.values():
        for r in read_jsonl(path):
            scored_ids.add(r["item_id"])
            wanted_analogues.update(r.get("analogue_ids") or [])
    eval_incidents = {items[i]["incident_id"] for i in scored_ids if i in items}

    flat, drawn = analogue_values(run_allocation, items, scored_ids, eval_incidents, sitreps, items_path, cache_path)
    if flat is None:
        return

    # The recorded ids are authoritative: they are what the prompt actually carried. Redrawing them through
    # analogues() is a consistency check on the pool, not a substitute for the record.
    mismatch = 0
    for path in grounded.values():
        for r in read_jsonl(path):
            rec = r.get("analogue_ids")
            if rec is not None and r["item_id"] in drawn and list(rec) != list(drawn[r["item_id"]]):
                mismatch += 1
    unresolved = sorted(wanted_analogues - set(flat))

    head("ALLOCATION   analogue-only predictor:  next-day personnel = personnel today x median analogue ratio")
    print("pool rebuilt through run_allocation.build_pool; %d analogue day pairs; %d eval incidents excluded"
          % (len(flat), len(eval_incidents)))
    print("recorded analogue_ids that redrawing through run_allocation.analogues did not reproduce: %d" % mismatch)
    if unresolved:
        print("recorded analogue_ids absent from the rebuilt pool: %d (e.g. %s) -- those rows are dropped"
              % (len(unresolved), unresolved[0]))
    # The pool excludes the union of every grounded run's incidents. That is the smallest pool any one run
    # could have drawn from, so an id it cannot resolve is reported rather than quietly replaced.
    flagged = warn_partial("allocation grounded",
                           {m + "/grounded": len(read_jsonl(p)) for m, p in grounded.items()})
    print("\n%-42s %5s %9s %8s %9s" % ("run", "n", "MAE", "nMAE", "beats-pers"))
    print("-" * 100)

    out = {}
    for model, path in grounded.items():
        rows = read_jsonl(path)
        usable, unrecorded, empty_draw, unparsed = [], 0, 0, 0
        for r in rows:
            if r.get("prediction") is None:
                unparsed += 1
                continue
            if r.get("analogue_ids") is None:
                unrecorded += 1        # a run older than the analogue_ids record: what it showed is unknowable
                continue
            vals = [flat[a] for a in r["analogue_ids"] if a in flat]
            if not vals:
                empty_draw += 1
                continue
            usable.append((r, vals))
        if not usable:
            print("%-42s  nothing to score: %d unparsed, %d with no analogue_ids recorded, %d whose draw did "
                  "not resolve" % (model + "/grounded", unparsed, unrecorded, empty_draw))
            if unrecorded:
                print("%-42s  that run predates the analogue_ids record, so the analogues it displayed cannot be "
                      "rebuilt; redrawing them would assume a pool and a seed that run may not have used"
                      % "")
            continue

        ids_here = {r["item_id"] for r, _ in usable}
        ratios = {r["item_id"]: [nxt / max(today, 1.0) for today, nxt in vals] for r, vals in usable}

        def swap(base_rows, fn):
            return [dict(r, prediction=fn(r)) for r in base_rows]

        base = [r for r, _ in usable]
        exact = swap(base, lambda r: rnd_int(r["persistence"] * float(np.median(ratios[r["item_id"]]))))
        shown = swap(base, lambda r: rnd_int(r["persistence"] * float("%.2f" % float(np.median(ratios[r["item_id"]])))))
        pers = swap(base, lambda r: r["persistence"])

        scored = collections.OrderedDict()
        scored[model + "/grounded"] = run_allocation.score(base, model + "/grounded")
        pair_note = None
        if model in bare:
            brows, both, rel = paired_split(ids_here, read_jsonl(bare[model]),
                                            lambda r: r.get("prediction") is not None)
            bscore = lambda lab: run_allocation.score([brows[i] for i in sorted(both)], lab)
            if both and rel == "same":
                scored[model + "/bare (same items)"] = bscore(model + "/bare")
            elif both and rel == "covered":
                scored[model + "/bare (paired, %d items)" % len(both)] = bscore(model + "/bare-paired")
                pair_note = ("bare answered %d items to grounded's %d; the bare row is restricted to "
                             "grounded's items" % (len(brows), len(ids_here)))
            elif both:
                # Each arm answered something the other did not, so the grounded row is repeated on the
                # overlap: the two rows a reader subtracts have to be computed on one set.
                scored[model + "/grounded (paired)"] = run_allocation.score(
                    [r for r in base if r["item_id"] in both], model + "/grounded-paired")
                scored[model + "/bare (paired)"] = bscore(model + "/bare-paired")
                pair_note = ("the arms answered different items (grounded %d, bare %d, both %d), so the arm "
                             "contrast is the paired pair of rows, not the top row"
                             % (len(ids_here), len(brows), len(both)))
        scored["persistence"] = run_allocation.score(pers, "persistence")
        scored["ANALOGUE-ONLY median ratio"] = run_allocation.score(exact, "analogue-only")
        scored["ANALOGUE-ONLY ratio as shown (2 dp)"] = run_allocation.score(shown, "analogue-only-2dp")

        for label, s in scored.items():
            print("%-42s %5d %9s %8s %9s" % (label, s["parsed"], fmt(s.get("mae"), 2),
                                             fmt(s.get("mae_norm")), fmt(s.get("beats_persistence_share"))))
        g, a = scored[model + "/grounded"], scored["ANALOGUE-ONLY median ratio"]
        verdict("MAE", g["mae"], a["mae"], lower_is_better=True, nd=2, unit=" personnel")
        med = np.array([float(np.median(v)) for v in ratios.values()])
        true = np.array([r["target"] / max(r["persistence"], 1.0) for r in base])
        print("   median analogue ratio over items: p25 %.2f, median %.2f, p75 %.2f | the day's true ratio: "
              "p25 %.2f, median %.2f, p75 %.2f"
              % (np.percentile(med, 25), np.median(med), np.percentile(med, 75),
                 np.percentile(true, 25), np.median(true), np.percentile(true, 75)))
        if pair_note:
            print("   -> %s" % pair_note)
        if unparsed or unrecorded or empty_draw:
            print("   dropped %d unparsed, %d with no analogue_ids recorded, %d whose draw did not resolve"
                  % (unparsed, unrecorded, empty_draw))
        print()
        out[model] = {"scores": {k: v for k, v in scored.items()},
                      "arms_answered_the_same_items": pair_note is None,
                      "partial_run": model + "/grounded" in flagged}
    print(CLUSTER_NOTE)
    SUMMARY["allocation"] = {"analogue_pairs": len(flat), "redraw_mismatches": mismatch,
                             "unresolved_analogue_ids": len(unresolved),
                             "partial_runs": flagged, "runs": out}


def analogue_values(run_allocation, items, scored_ids, eval_incidents, sitreps, items_path, cache_path):
    """Map analogue_id -> (personnel today, personnel next day), plus the redrawn id list per item.

    build_pool reads a 300 MB CSV, so the result is cached beside this script. The cache never lives inside the
    benchmark repository. The key covers everything the result depends on: the CSV for the pool, the evaluation
    incident set for what the pool excludes, the scored ids for which draws are wanted, and items.jsonl, since
    analogues() keys the draw on each item's bands and target date. Leaving items.jsonl out was a way for a
    rebuilt item pool to be scored against a stale redraw without a word.
    """
    sst, ist = sitreps.stat(), pathlib.Path(items_path).stat()
    key = hashlib.blake2b(("%d|%d|%d|%d|" % (sst.st_size, int(sst.st_mtime),
                                             ist.st_size, int(ist.st_mtime))).encode()
                          + "|".join(sorted(eval_incidents)).encode()
                          + b"||" + "|".join(sorted(scored_ids)).encode(), digest_size=16).hexdigest()
    if cache_path and pathlib.Path(cache_path).exists():
        try:
            blob = json.loads(pathlib.Path(cache_path).read_text(encoding="utf-8"))
            if blob.get("key") == key:
                return {k: tuple(v) for k, v in blob["flat"].items()}, blob["drawn"]
        except (ValueError, KeyError):
            pass

    print("building the historical analogue pool from %s ..." % sitreps.name, flush=True)
    pool = run_allocation.build_pool(set(eval_incidents))
    flat = {}
    for rows in pool.values():
        for r in rows:
            flat[r["analogue_id"]] = (float(r["today"]), float(r["next"]))
    drawn = {}
    for item_id in sorted(scored_ids):
        it = items.get(item_id)
        if it is not None:
            drawn[item_id] = [r["analogue_id"] for r in run_allocation.analogues(pool, it)]
    if cache_path:
        pathlib.Path(cache_path).write_text(json.dumps({"key": key, "flat": flat, "drawn": drawn}), encoding="utf-8")
    return flat, drawn


# ----------------------------------------------------------------------------------------------------------
# mesogeos
# ----------------------------------------------------------------------------------------------------------

def last_valid(series):
    """The last observed value of a driver window, walking back past days the sensor missed (cloud, mostly)."""
    for k, v in enumerate(reversed(series or [])):
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            return float(v), k
    return None, None


def mesogeos(repo, run_mesogeos):
    task = repo / "task-mesogeos"
    items_path = task / "items.jsonl"
    clim_path = task / "climatology.json"
    ok = need(task, "task-mesogeos directory")
    ok = need(items_path, "mesogeos items, needed for the last-day driver values") and ok
    ok = need(clim_path, "monthly climatology, which is the grounding this baseline isolates") and ok
    if not ok:
        return
    grounded = response_files(task, "grounded")
    bare = response_files(task, "bare")
    if not grounded:
        MISSING.append("%s  (no responses-*-grounded.jsonl; the grounded arm is what this baseline mirrors)"
                       % (task / "responses-<model>-grounded.jsonl"))
        return

    items = {it["item_id"]: it for it in read_jsonl(items_path)}
    clim = json.loads(clim_path.read_text(encoding="utf-8"))
    drivers = [d for d in run_mesogeos.SHOW]

    from sklearn.metrics import average_precision_score

    def z_scores(item_ids, driver):
        """Standardized anomaly of the last observed day against that calendar month's climatology."""
        vals, stale = [], 0
        for i in item_ids:
            it = items.get(i)
            if it is None:
                vals.append(None)
                continue
            month = it["context"]["window_end"][5:7]
            ref = (clim.get(month) or {}).get(driver)
            x, back = last_valid(it["context"]["daily"].get(driver))
            if ref is None or x is None or not ref[1]:
                vals.append(None)
            else:
                vals.append((x - ref[0]) / ref[1])
                stale += back > 0
        return vals, [v is not None for v in vals], stale

    head("MESOGEOS   climatology-only predictor:  standardized anomaly of the last day against the monthly mean")
    print("climatology months on file: %d | drivers: %s" % (len(clim), ", ".join(drivers)))
    flagged = warn_partial("mesogeos grounded",
                           {m + "/grounded": len(read_jsonl(p)) for m, p in grounded.items()})

    out = {}
    for model, path in grounded.items():
        all_rows = read_jsonl(path)
        rows = [r for r in all_rows
                if (r.get("probability") is not None or r.get("call") is not None) and r["item_id"] in items]
        z_t2m, keep, stale = z_scores([r["item_id"] for r in rows], "t2m")
        rows = [r for r, k in zip(rows, keep) if k]
        if not rows:
            print("%-42s  no rows with both a parsed answer and a usable last-day t2m" % (model + "/grounded"))
            continue
        ids_here = [r["item_id"] for r in rows]
        y = np.array([r["label"] for r in rows])
        z_t2m = np.array([v for v, k in zip(z_t2m, keep) if k])
        raw_t2m = np.array([last_valid(items[i]["context"]["daily"]["t2m"])[0] for i in ids_here])

        print("\n%-42s %5s %9s" % ("run", "n", "AP"))
        print("-" * 100)
        counts = {model + "/grounded": len(rows)}
        scored = collections.OrderedDict()
        scored[model + "/grounded"] = run_mesogeos.score(rows, model + "/grounded")["auprc"]
        pair_note = None
        if model in bare:
            answered = lambda r: r.get("probability") is not None or r.get("call") is not None
            brows, both, rel = paired_split(ids_here, read_jsonl(bare[model]), answered)
            bap = lambda lab: run_mesogeos.score([brows[i] for i in sorted(both)], lab)["auprc"]
            if both and rel == "same":
                label = model + "/bare (same items)"
                scored[label], counts[label] = bap(label), len(both)
            elif both and rel == "covered":
                label = model + "/bare (paired, %d items)" % len(both)
                scored[label], counts[label] = bap(label), len(both)
                pair_note = ("bare answered %d items to grounded's %d; the bare row is restricted to "
                             "grounded's items" % (len(brows), len(ids_here)))
            elif both:
                gl, bl = model + "/grounded (paired)", model + "/bare (paired)"
                scored[gl] = run_mesogeos.score([r for r in rows if r["item_id"] in both], gl)["auprc"]
                scored[bl] = bap(bl)
                counts[gl] = counts[bl] = len(both)
                pair_note = ("the arms answered different items (grounded %d, bare %d, both %d), so the arm "
                             "contrast is the paired pair of rows, not the top row"
                             % (len(ids_here), len(brows), len(both)))
        scored["CLIMATOLOGY-ONLY z(t2m) last day"] = float(average_precision_score(y, z_t2m))
        # The monthly mean alone reads none of the item's own drivers: it is the grounding and nothing else.
        clim_mean = np.array([clim[items[i]["context"]["window_end"][5:7]]["t2m"][0] for i in ids_here])
        scored["CLIMATOLOGY-ONLY monthly mean t2m alone"] = float(average_precision_score(y, clim_mean))
        scored["raw last-day t2m, no climatology"] = float(average_precision_score(y, raw_t2m))
        scored["positive rate (no-skill AP)"] = float(np.mean(y))
        for label, ap in scored.items():
            print("%-42s %5d %9s" % (label, counts.get(label, len(rows)), fmt(ap)))
        if pair_note:
            print("   -> %s" % pair_note)
        dropped = len(all_rows) - len(rows)
        if dropped:
            print("   -> dropped %d of %d rows: unparsed answer, item absent, or no usable last-day t2m"
                  % (dropped, len(all_rows)))
        if stale:
            print("   -> %d items had no t2m on the final day; the last observed day of the window was used" % stale)
        g = scored[model + "/grounded"]
        a = scored["CLIMATOLOGY-ONLY z(t2m) last day"]
        verdict("AP", g, a, lower_is_better=False)
        # The climatology rule is the comparator objection B asks for, but it is not the strongest rule in the
        # table. A reader who lifts only the verdict line should see which trivial rule actually wins.
        trivial = {k: v for k, v in scored.items() if k.startswith(("CLIMATOLOGY-ONLY", "raw last-day"))}
        bk = max(trivial, key=trivial.get)
        print("   strongest trivial rule in this table: %s at %.3f  ->  %s by %.3f"
              % (bk, trivial[bk], "model ahead" if g > trivial[bk] else "rule ahead", abs(g - trivial[bk])))
        print("   reference points on the same items: raw last-day t2m, which reads no climatology, scores "
              "%.3f; the\n   month's climatological mean, which reads none of this cell's own drivers, scores "
              "%.3f against a no-skill\n   floor of %.3f. The anomaly, which subtracts the second from the "
              "first, scores %.3f, below both."
              % (scored["raw last-day t2m, no climatology"], scored["CLIMATOLOGY-ONLY monthly mean t2m alone"],
                 scored["positive rate (no-skill AP)"], a))

        # Exploratory scan, printed in both directions so no driver or sign is selected on the outcome.
        print("\n   exploratory: AP of every single-driver last-day anomaly (not a selected baseline)")
        print("   %-14s %7s %7s %6s" % ("driver", "AP(+z)", "AP(-z)", "n"))
        scan = {}
        for d in drivers:
            vals, k, _ = z_scores(ids_here, d)
            sel = [i for i, kk in enumerate(k) if kk]
            if len(sel) < 10 or len(set(y[sel])) < 2:
                continue
            zz = np.array([vals[i] for i in sel])
            up, dn = float(average_precision_score(y[sel], zz)), float(average_precision_score(y[sel], -zz))
            scan[d] = {"ap_pos": up, "ap_neg": dn, "n": len(sel)}
            print("   %-14s %7s %7s %6d" % (d, fmt(up), fmt(dn), len(sel)))
        print()
        out[model] = {"scores": scored, "n": counts, "single_driver_scan": scan,
                      "arms_answered_the_same_items": pair_note is None,
                      "partial_run": model + "/grounded" in flagged}
    print(CLUSTER_NOTE)
    SUMMARY["mesogeos"] = {"partial_runs": flagged, "runs": out}


# ----------------------------------------------------------------------------------------------------------
# figlib
# ----------------------------------------------------------------------------------------------------------

def block_max_absdiff(a, b, block=16):
    """Largest block-mean absolute difference between two same-sized grayscale frames.

    A plume covers a small part of a wide camera view, so a whole-frame mean washes it out; blocking keeps the
    rule trivial while letting a local change register.
    """
    d = np.abs(a - b)
    h, w = d.shape
    d = d[: h - h % block, : w - w % block]
    hb, wb = d.shape[0] // block, d.shape[1] // block
    return float(d.reshape(hb, block, wb, block).mean(axis=(1, 3)).max())


def fire_of(item):
    """The incident a frame belongs to. Several cameras watch one fire, so the sequence is not the fire.

    Falls back to the sequence when items.jsonl carries no fire_name, which makes the held-out cut fall back
    to the leakier sequence-level one rather than silently mislabelling it.
    """
    return item.get("fire_name") or item["sequence"]


def heldout_calls(s, y, groups):
    """Calls from a threshold fitted on every group except the item's own."""
    pred = np.zeros(len(s), dtype=bool)
    for g in sorted(set(groups)):
        held = groups == g
        pred[held] = s[held] >= best_threshold(s[~held], y[~held])
    return pred


def figlib(repo, run_figlib):
    task = repo / "task-figlib"
    items_path = task / "items.jsonl"
    ok = need(task, "task-figlib directory")
    ok = need(items_path, "figlib items, needed to find each sequence's reference frame") and ok
    if not ok:
        return
    grounded = response_files(task, "grounded")
    bare = response_files(task, "bare")
    if not grounded:
        MISSING.append("%s  (no responses-*-grounded.jsonl; the grounded arm is what this baseline mirrors)"
                       % (task / "responses-<model>-grounded.jsonl"))
        return

    items_list = read_jsonl(items_path)
    for it in items_list:
        it["bucket"] = run_figlib.bucket(it["offset_seconds"])
    items = {it["item_id"]: it for it in items_list}
    reference = {}
    for it in items_list:                     # same first-wins-on-ties rule run_figlib uses
        cur = reference.get(it["sequence"])
        if cur is None or it["offset_seconds"] < cur["offset_seconds"]:
            reference[it["sequence"]] = it

    head("FIGLIB   pixel-difference predictor:  largest 16x16 block-mean |frame - reference frame|")

    scored_ids = set()
    for path in grounded.values():
        scored_ids.update(r["item_id"] for r in read_jsonl(path) if r.get("prediction") is not None)
    needed = set()
    for i in scored_ids:
        it = items.get(i)
        if it is not None:
            needed.add(it["image"])
            needed.add(reference[it["sequence"]]["image"])
    if not needed:
        print("no grounded row carries a parsed prediction, so there is nothing to put a baseline beside.")
        SKIPPED.append("figlib pixel-difference baseline: no parsed grounded predictions to score against")
        SUMMARY["figlib"] = {"skipped": "no parsed grounded predictions"}
        return
    absent = [p for p in sorted(needed) if not (repo / p.replace("\\", "/")).exists()]
    if absent:
        line = ("this baseline needs the cached FIgLib frames and %s; %d of %d image files are not on disk. "
                "Skipping it rather than inventing a stand-in. Re-run after fetching the images to "
                "%s to get the number."
                % ("none are present" if len(absent) == len(needed) else "some are missing",
                   len(absent), len(needed), task / "images-1568"))
        print(line)
        SKIPPED.append("figlib pixel-difference baseline: %d of %d cached frames missing under %s"
                       % (len(absent), len(needed), task))
        SUMMARY["figlib"] = {"skipped": line}
        return

    try:
        from PIL import Image
    except ImportError:
        line = ("this baseline needs Pillow to read the cached FIgLib frames; it is not installed in this "
                "interpreter. Skipping it rather than inventing a stand-in.")
        print(line)
        SKIPPED.append("figlib pixel-difference baseline: Pillow not importable")
        SUMMARY["figlib"] = {"skipped": line}
        return

    from sklearn.metrics import average_precision_score, roc_auc_score

    cache = {}

    def gray(rel):
        if rel not in cache:
            with Image.open(repo / rel.replace("\\", "/")) as im:
                cache[rel] = np.asarray(im.convert("L").resize((320, 240), Image.BILINEAR), dtype=np.float32)
        return cache[rel]

    print("reading %d cached frames at 320x240 grayscale ..." % len(needed), flush=True)
    stat = {}
    for i in sorted(scored_ids):
        it = items.get(i)
        if it is None:
            continue
        stat[i] = block_max_absdiff(gray(it["image"]), gray(reference[it["sequence"]]["image"]))

    nseq, nfire = len({i["sequence"] for i in items_list}), len({fire_of(i) for i in items_list})
    print("clustering: %d frames in %d sequences from %d fires; a threshold is fitted holding out one whole "
          "fire" % (len(items_list), nseq, nfire))
    flagged = warn_partial("figlib grounded",
                           {m + "/grounded": len(read_jsonl(p)) for m, p in grounded.items()})

    out = {}
    for model, path in grounded.items():
        rows = [r for r in read_jsonl(path) if r.get("prediction") is not None and r["item_id"] in stat]
        if not rows:
            print("%-42s  no parsed rows with a cached frame pair" % (model + "/grounded"))
            continue
        y = np.array([r["label"] == "smoke" for r in rows])
        s = np.array([stat[r["item_id"]] for r in rows])
        seqs = np.array([r["sequence"] for r in rows])
        fires = np.array([fire_of(items[r["item_id"]]) for r in rows])

        # Leave-one-FIRE-out threshold. Holding out one sequence is not enough: five fires here appear under
        # several cameras at once, so the other sequences of the same fire show the same plume at the same
        # minute, and a cut fitted on them has effectively seen the held-out frames. Fire is the independent
        # unit, which is the whole of the round-1 objection. The sequence-level cut is kept below as the leaky
        # variant, so the size of the difference is visible rather than asserted.
        lofo = heldout_calls(s, y, fires)
        loso = heldout_calls(s, y, seqs)
        orc = s >= best_threshold(s, y)

        model_pred = np.array([bool(r["prediction"]) for r in rows])
        print("\n%-42s %5s %9s %9s %9s" % ("run", "n", "accuracy", "recall", "FPR"))
        print("-" * 100)
        lines = collections.OrderedDict()
        lines[model + "/grounded"] = (model_pred, y)
        pair_note = None
        if model in bare:
            brows, both, rel = paired_split([r["item_id"] for r in rows], read_jsonl(bare[model]),
                                            lambda r: r.get("prediction") is not None)
            pick = lambda rr: (np.array([bool(r["prediction"]) for r in rr]),
                               np.array([r["label"] == "smoke" for r in rr]))
            if both and rel == "same":
                lines[model + "/bare (same items)"] = pick([brows[i] for i in sorted(both)])
            elif both and rel == "covered":
                # The usual FIgLib shape: the grounded arm drops each sequence's reference frame, so bare is a
                # strict superset and restricting it is a true pairing.
                lines[model + "/bare (paired, %d items)" % len(both)] = pick([brows[i] for i in sorted(both)])
                pair_note = ("bare answered %d frames to grounded's %d (the grounded arm drops each sequence's "
                             "reference frame); the bare row is restricted to grounded's frames"
                             % (len(brows), len(rows)))
            elif both:
                # Parse failures on both sides split the sets, and then neither full arm is the comparison.
                lines[model + "/grounded (paired)"] = pick([r for r in rows if r["item_id"] in both])
                lines[model + "/bare (paired)"] = pick([brows[i] for i in sorted(both)])
                pair_note = ("the arms answered different frames (grounded %d, bare %d, both %d), so the arm "
                             "contrast is the paired pair of rows, not the top row"
                             % (len(rows), len(brows), len(both)))
        lines["PIXEL-DIFF, leave-one-FIRE-out cut"] = (lofo, y)
        lines["PIXEL-DIFF, leave-one-sequence-out (leaky)"] = (loso, y)
        lines["PIXEL-DIFF, best cut on all labels"] = (orc, y)
        lines["always call smoke"] = (np.ones(len(rows), dtype=bool), y)
        for label, (p, yy) in lines.items():
            print("%-42s %5d %9s %9s %9s"
                  % (label, len(p), fmt(float(np.mean(p == yy))),
                     fmt(float(np.mean(p[yy]))) if yy.any() else "n/a",
                     fmt(float(np.mean(p[~yy]))) if (~yy).any() else "n/a"))
        auc = float(roc_auc_score(y, s)) if len(set(y)) > 1 else None
        ap = float(average_precision_score(y, s)) if len(set(y)) > 1 else None
        print("   pixel-difference statistic, threshold-free: AUROC %s | AP %s | positive rate %s"
              % (fmt(auc), fmt(ap), fmt(float(np.mean(y)))))
        ga = float(np.mean(model_pred == y))
        ba = float(np.mean(lofo == y))
        verdict("accuracy", ga, ba, lower_is_better=False)
        if pair_note:
            print("   -> %s" % pair_note)
        print("   the leave-one-FIRE-out cut is the comparable one: these %d frames come from %d sequences and "
              "%d fires.%s\n   'Best cut on all labels' is an upper bound the rule cannot reach, since it is "
              "fitted on the labels it is then\n   scored against."
              % (len(rows), len(set(seqs)), len(set(fires)),
                 "" if len(set(seqs)) == len(set(fires)) else
                 "\n   Holding out only the sequence would still fit the cut on other cameras watching the "
                 "same plume at the same minute."))
        out[model] = {"n": len(rows), "model_accuracy": ga, "pixdiff_lofo_accuracy": ba,
                      "pixdiff_loso_accuracy_leaky": float(np.mean(loso == y)),
                      "pixdiff_oracle_accuracy": float(np.mean(orc == y)),
                      "pixdiff_auroc": auc, "pixdiff_ap": ap, "positive_rate": float(np.mean(y)),
                      "sequences": int(len(set(seqs))), "fires": int(len(set(fires))),
                      "arms_answered_the_same_items": pair_note is None,
                      "partial_run": model + "/grounded" in flagged}
        print()
    print(CLUSTER_NOTE)
    SUMMARY["figlib"] = {"statistic": "max 16x16 block-mean |diff| on 320x240 grayscale",
                         "threshold": "leave-one-fire-out", "partial_runs": flagged, "runs": out}


def best_threshold(s, y):
    """Threshold on s maximizing accuracy against y; the median of the tied best cuts, for stability."""
    if len(s) == 0:
        return float("inf")
    cand = np.unique(s)
    cuts = np.concatenate(([cand[0] - 1.0], (cand[:-1] + cand[1:]) / 2.0, [cand[-1] + 1.0])) if len(cand) > 1 \
        else np.array([cand[0] - 1.0, cand[0] + 1.0])
    acc = np.array([float(np.mean((s >= c) == y)) for c in cuts])
    return float(np.median(cuts[acc == acc.max()]))


# ----------------------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", type=pathlib.Path, default=DEFAULT_REPO)
    ap.add_argument("--tasks", nargs="*", default=["allocation", "mesogeos", "figlib"],
                    choices=["allocation", "mesogeos", "figlib"])
    ap.add_argument("--pool-cache", type=pathlib.Path, default=HERE / ".analogue-pool-cache.json",
                    help="cache for the rebuilt ICS-209 analogue pool; always outside the benchmark repo")
    ap.add_argument("--no-pool-cache", action="store_true")
    ap.add_argument("--json-out", type=pathlib.Path, default=HERE / "information_only_baselines.json")
    args = ap.parse_args()

    repo = args.repo.resolve()
    if not need(repo, "AI4Fire repository root"):
        report_and_exit()
    sys.path.insert(0, str(repo))

    mods = {}
    for task, mod in [("allocation", "run_allocation"), ("mesogeos", "run_mesogeos"), ("figlib", "run_figlib")]:
        if task not in args.tasks:
            continue
        if not need(repo / (mod + ".py"), "scoring functions and the grounding construction for %s" % task):
            continue
        try:
            mods[task] = __import__(mod)
        except Exception as exc:                       # an unimportable runner is a missing input, not a crash
            MISSING.append("%s  (imports, but raised: %s)" % (repo / (mod + ".py"), exc))

    print("AI4Fire repository: %s" % repo)
    print("information-only baselines: a trivial rule that sees ONLY what the grounded arm was handed.")
    print("PROVISIONAL: every number below is recomputed from the responses-*.jsonl files present right now.")

    # An unreadable response file stops its own task and is named; the other two still run.
    for name, fn in (("allocation", lambda: allocation(repo, mods["allocation"],
                                                       None if args.no_pool_cache else args.pool_cache)),
                     ("mesogeos", lambda: mesogeos(repo, mods["mesogeos"])),
                     ("figlib", lambda: figlib(repo, mods["figlib"]))):
        if name in mods:
            try:
                fn()
            except InputError as exc:
                print("\n%s: stopped, an input could not be read -- %s" % (name, exc))
                MISSING.append(str(exc))

    if args.json_out:
        SUMMARY["missing_required_inputs"] = list(MISSING)
        SUMMARY["skipped_optional_inputs"] = list(SKIPPED)
        SUMMARY["complete"] = not MISSING
        args.json_out.write_text(json.dumps(SUMMARY, indent=1, default=float), encoding="utf-8")
        print("\nwrote %s%s" % (args.json_out, "" if not MISSING else "  (complete: false)"))
    report_and_exit()


def report_and_exit():
    if SKIPPED:
        print("\nskipped, by design, for want of a declared-optional input:")
        for s in SKIPPED:
            print("  - %s" % s)
    if MISSING:
        print("\nMISSING REQUIRED INPUT:")
        for m in MISSING:
            print("  - %s" % m)
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
