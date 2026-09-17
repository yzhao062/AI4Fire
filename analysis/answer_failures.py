"""Characterise every item where a run produced no usable answer, across all three fire-bench tasks.

    python answer_failures.py
    python answer_failures.py --root C:\\path\\to\\fire-bench --json out.json

Answers the round-1 TMLR objection that the paper reports scores on answered items without reporting who
failed to answer and whether the failures are random. A truncated or unparsable answer is a result: if
failures concentrate on hard items, scoring the remainder inflates the score.

Nothing here is hardcoded to the current response files. Every count, every cap and every item universe is
re-derived from whatever files are on disk, so the script can be re-run unchanged after the pending rerun
overwrites responses-*.jsonl.

Four things are reported per task.

1. Attempted, answered, failed and the failure rate, per responses-*.jsonl file.
2. A cause for each failure: a recorded error field, an output-token-cap hit, an empty body, or a non-empty
   body that would not parse (first 80 characters quoted).
3. Whether failure is independent of item difficulty, by Fisher exact (2x2) or chi-square with a Monte Carlo
   permutation check (larger tables). FIgLib is stratified by offset bucket and by label, Mesogeos by label,
   allocation by whether the filed personnel count moved off persistence. The test is run twice where the two
   differ: once on rows with no usable answer at all, and once on the wider degraded class that adds rows
   missing one required field, which the scorer keeps by imputing it. Testing only the strict class would have
   put every Mesogeos test on two or three events while the class with 43 to 86 events went untested, and that
   is the class where an association actually shows up.
4. The set of items EVERY run answered, which is the denominator the paper should score on, split into the
   shrinkage that comes from the benchmark design and the shrinkage that comes from failures.

Two measurement notes that a naive version of this script gets wrong.

The output cap is inferred per file, not read once from the runner. The runners declare MAX_OUT, but a file on
disk may have been produced under an older value: the current task-figlib files pile up at 200 completion
tokens while run_figlib.py now declares 1536. A cap leaves a spike of rows at one value far above the typical
answer length, so that spike is the evidence used, and the declared MAX_OUT is only the fallback.

completion_tokens includes reasoning tokens on some providers, so "near the cap" alone does not mean the body
was cut. Gemini rows on task-allocation sit within a few tokens of 1536 and still carry complete JSON. The cap
class is therefore only ever assigned to a row that already failed, and answered rows are screened separately
by whether their body is structurally complete.
"""
import argparse
import collections
import json
import math
import os
import re
import statistics
import sys

import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------------------------------------
# Task configuration. answer_fields are the fields a row must carry for the run's scorer to use the row at
# all; a row with every one of them null produced no usable answer. Where a scorer can limp along on a subset
# (Mesogeos imputes a call from the probability and vice versa) the shortfall is reported as "partial".
# ---------------------------------------------------------------------------------------------------------
TASKS = {
    "task-allocation": {
        "answer_fields": ["prediction"],
        "runner": "run_allocation.py",
        "universe_files": ["items-v1.jsonl", "items.jsonl"],
        "strata": ["moved"],
    },
    "task-figlib": {
        "answer_fields": ["prediction"],
        "runner": "run_figlib.py",
        "universe_files": ["items.jsonl"],
        "strata": ["bucket", "label"],
    },
    "task-mesogeos": {
        "answer_fields": ["probability", "call"],
        "runner": "run_mesogeos.py",
        "universe_files": ["items.jsonl"],
        "strata": ["label"],
        # run_mesogeos.score fills a missing call with `probability >= 0.5`, so a row that never answered the
        # question still enters the F1. Whether that substitute agrees with the label more often than a real
        # answer does is the difference between a harmless repair and a score propped up by the repair.
        "imputed_call": True,
    },
}

FAIL_RATE_FLAG = 0.02  # runs above this are called out by name
MOVED_THRESHOLD = 0.10  # matches `moved` in run_allocation.score
PERM_DRAWS = 20000
PERM_SEED = 20260915


# ---------------------------------------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------------------------------------
def read_jsonl(path):
    """Return (rows, n_malformed). A malformed line is itself a kind of failure and is counted, not dropped."""
    rows, bad = [], 0
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    bad += 1
    except OSError as exc:
        print("  cannot read %s: %s" % (path, exc))
    return rows, bad


def declared_cap(root, runner):
    """Pull MAX_OUT out of the runner source. Returns None if the runner or the constant is absent."""
    path = os.path.join(root, runner)
    if not os.path.exists(path):
        return None
    try:
        src = open(path, "r", encoding="utf-8").read()
    except OSError:
        return None
    m = re.search(r"^\s*MAX_OUT\s*=\s*(\d+)", src, re.M)
    return int(m.group(1)) if m else None


def split_run_name(filename):
    """responses-<model>-<arm>[-<variant>].jsonl -> (model, arm, variant)."""
    stem = re.sub(r"\.jsonl$", "", re.sub(r"^responses-", "", filename))
    m = re.search(r"-(bare|grounded)(?:-(.*))?$", stem)
    if not m:
        return stem, "unknown", ""
    return stem[:m.start()], m.group(1), (m.group(2) or "")


def find_runs(task_dir):
    out = []
    if not os.path.isdir(task_dir):
        return out
    for name in sorted(os.listdir(task_dir)):
        if name.startswith("responses-") and name.endswith(".jsonl"):
            out.append(os.path.join(task_dir, name))
    return out


def load_universe(task_dir, candidates):
    """First candidate that exists wins, mirroring what the runner itself evaluates."""
    for name in candidates:
        path = os.path.join(task_dir, name)
        if os.path.exists(path):
            rows, _ = read_jsonl(path)
            if rows:
                return name, {r.get("item_id"): r for r in rows if r.get("item_id")}
    return None, {}


# ---------------------------------------------------------------------------------------------------------
# Row-level judgements
# ---------------------------------------------------------------------------------------------------------
def completion_tokens(row):
    usage = row.get("usage")
    if not isinstance(usage, dict):
        return None
    for key in ("completion_tokens", "output_tokens"):
        val = usage.get(key)
        if isinstance(val, (int, float)):
            return int(val)
    return None


def infer_cap(rows, declared):
    """Infer the output cap this file was produced under.

    A cap shows up as several rows stopping at the same token count, far above the typical answer. That spike
    is trusted over the runner's current MAX_OUT, because a file on disk can predate a change to it.
    """
    cts = [c for c in (completion_tokens(r) for r in rows) if c is not None]
    if not cts:
        return (declared, "declared MAX_OUT" if declared else "unknown", 0)
    top = max(cts)
    at_top = cts.count(top)
    median = statistics.median(cts)
    spike = at_top >= 2 and top >= 2 * max(median, 1)
    if spike:
        # The ceiling itself is read off the data, but a provider stops when the budget runs out and that can
        # land a few tokens short of it: under a 1536 budget the task-allocation Gemini rows stop anywhere in
        # 1520 to 1532. A zero tolerance here would file a row cut at 1530 as an empty or unparsable body
        # rather than as a cap hit. The tolerance only ever widens the cause assigned to a row that already
        # failed, so it cannot turn a complete answer into a truncated one.
        return (top, "observed ceiling (%d rows stop at %d, median answer %d)" % (at_top, top, median),
                max(4, int(round(0.01 * top))))
    if declared:
        # Falling back to the declared value, allow a small margin: providers stop a few tokens short.
        return (declared, "declared MAX_OUT (no ceiling visible in this file)", max(4, int(round(0.02 * declared))))
    return (None, "unknown", 0)


FENCE = re.compile(r"^\s*```[a-zA-Z]*\s*|\s*```\s*$")


def body_complete(text):
    """Does the body look structurally finished? Used to screen ANSWERED rows for silent truncation."""
    if not isinstance(text, str):
        return False
    s = FENCE.sub("", text).strip()
    if not s:
        return False
    if s.count("{") != s.count("}") or s.count("[") != s.count("]"):
        return False
    return s.endswith("}") or s.endswith("]") or s.endswith('"') or s.endswith(".")


def classify(row, fields, cap, tol):
    """Cause of a row that produced no usable answer (or only a partial one).

    Precedence: a recorded error, then the output cap, then an empty body, then a body that finished cleanly
    but left a required field out, then an unparsable non-empty body.

    The cap outranks "empty" on purpose: a null body that burned every one of the cap's tokens is a cap hit
    whose tokens all went to hidden reasoning, and calling it "empty" would hide the cause. The schema class
    is separated from "unparsable" for the same reason. A Mesogeos row reading {"probability": 0.02,
    "reasoning": "..."} is complete, valid JSON that simply omits "fire"; filing it as unparsable would
    describe a truncation that did not happen and point at the wrong fix.
    """
    if row.get("error"):
        return "error_field", str(row.get("error"))[:80]
    ct = completion_tokens(row)
    raw = row.get("raw")
    blank = not isinstance(raw, str) or not raw.strip()
    if cap is not None and ct is not None and ct >= cap - tol:
        return ("output_cap_empty_body" if blank else "output_cap_truncated_text"), (
            "" if blank else raw[:80])
    if blank:
        return "empty_body", ""
    missing = [f for f in fields if row.get(f) is None]
    if body_complete(raw):
        return "schema_field_missing:%s" % ",".join(missing), raw[:80]
    return "unparsable_nonempty", raw[:80]


def answer_state(row, fields):
    """'answered' | 'partial' | 'failed' over the task's answer fields."""
    present = [f for f in fields if row.get(f) is not None]
    if len(present) == len(fields):
        return "answered"
    if present:
        return "partial"
    return "failed"


# ---------------------------------------------------------------------------------------------------------
# Strata
# ---------------------------------------------------------------------------------------------------------
def figlib_bucket(offset):
    """Same rule as run_figlib.bucket, re-derived here so a row without a stored bucket still strata-fies."""
    a = abs(offset)
    return "%s %s" % ("after" if offset > 0 else "before",
                      "0 to 10 min" if a <= 600 else "10 to 25 min" if a <= 1500 else "25 min or more")


def stratum_value(task, name, row, item):
    """Difficulty stratum for one row. Returns None when it cannot be determined."""
    src = {}
    if isinstance(item, dict):
        src.update(item)
    src.update({k: v for k, v in row.items() if v is not None})

    if name == "label":
        val = src.get("label")
        if val is None:
            return None
        if task == "task-mesogeos":
            return "fire" if int(val) == 1 else "no fire"
        return str(val)

    if name == "bucket":
        val = row.get("bucket") or src.get("bucket")
        if val:
            return str(val)
        off = src.get("offset_seconds")
        return figlib_bucket(off) if isinstance(off, (int, float)) else None

    if name == "moved":
        target = src.get("target", src.get("target_personnel"))
        base = src.get("persistence", src.get("baseline_persistence"))
        if not isinstance(target, (int, float)) or not isinstance(base, (int, float)):
            return None
        moved = abs(target - base) / max(base, 1) > MOVED_THRESHOLD
        return "filed count moved" if moved else "filed count held"
    return None


def stratum_order(task, name, values, offsets):
    """Display order: FIgLib buckets run earliest-to-latest by offset, everything else alphabetical."""
    if name == "bucket" and offsets:
        return sorted(values, key=lambda v: (offsets.get(v, 0), v))
    return sorted(values)


# ---------------------------------------------------------------------------------------------------------
# Independence tests
# ---------------------------------------------------------------------------------------------------------
def independence_test(table):
    """table: ordered list of (stratum, n_failed, n_ok). Returns a dict describing the test, or a reason why
    no test applies."""
    rows = [(s, f, o) for s, f, o in table if (f + o) > 0]
    if len(rows) < 2:
        return {"test": "n/a", "reason": "fewer than two non-empty strata"}
    total_fail = sum(f for _, f, _ in rows)
    total_ok = sum(o for _, _, o in rows)
    if total_fail == 0:
        return {"test": "n/a", "reason": "no failures in this run"}
    if total_ok == 0:
        return {"test": "n/a", "reason": "no answered rows in this run"}

    obs = np.array([[f, o] for _, f, o in rows], dtype=float)
    if obs.shape[0] == 2:
        odds, p = stats.fisher_exact(obs.astype(int))
        return {"test": "Fisher exact (2x2)", "p": float(p), "odds_ratio": float(odds),
                "n": int(obs.sum())}

    chi2, p, dof, expected = stats.chi2_contingency(obs)
    out = {"test": "chi-square (%dx2)" % obs.shape[0], "p": float(p), "stat": float(chi2),
           "dof": int(dof), "min_expected": float(expected.min()), "n": int(obs.sum())}
    if expected.min() < 5:
        out["perm_p"] = permutation_p(obs)
        out["note"] = ("min expected cell %.2f < 5, chi-square approximation is unreliable; "
                       "use the permutation p" % expected.min())
    return out


def permutation_p(obs):
    """Monte Carlo p for an r x 2 table: reshuffle which attempts failed, keeping both margins' sizes."""
    counts = obs.sum(axis=1).astype(int)
    n_fail = int(obs[:, 0].sum())
    n = int(counts.sum())
    group = np.repeat(np.arange(len(counts)), counts)
    rng = np.random.default_rng(PERM_SEED)

    def chi2_stat(fail_per_group):
        tbl = np.column_stack([fail_per_group, counts - fail_per_group]).astype(float)
        exp = np.outer(tbl.sum(axis=1), tbl.sum(axis=0)) / tbl.sum()
        with np.errstate(divide="ignore", invalid="ignore"):
            term = np.where(exp > 0, (tbl - exp) ** 2 / exp, 0.0)
        return float(term.sum())

    observed = chi2_stat(obs[:, 0])
    hits = 0
    for _ in range(PERM_DRAWS):
        idx = rng.permutation(n)[:n_fail]
        fail_per_group = np.bincount(group[idx], minlength=len(counts))
        if chi2_stat(fail_per_group) >= observed - 1e-12:
            hits += 1
    return float((hits + 1) / (PERM_DRAWS + 1))


def holm(pvalues):
    """Holm-Bonferroni adjusted p-values, in the order given."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * pvalues[i])
        running = max(running, val)
        adjusted[i] = running
    return adjusted


def best_p(test):
    """The p-value to act on: the permutation one when chi-square was flagged as unreliable."""
    if "perm_p" in test:
        return test["perm_p"]
    return test.get("p")


# ---------------------------------------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------------------------------------
def rule(char="-", width=118):
    print(char * width)


def table(headers, rows, aligns=None):
    if not rows:
        print("  (none)")
        return
    cols = len(headers)
    widths = [len(h) for h in headers]
    for r in rows:
        for i in range(cols):
            widths[i] = max(widths[i], len(str(r[i])))
    aligns = aligns or ["<"] * cols
    flags = ["-" if a == "<" else "" for a in aligns]

    def fmt(vals):
        return "  " + "  ".join(("%%%s%ds" % (flags[i], widths[i])) % str(vals[i]) for i in range(cols))

    print(fmt(headers))
    print("  " + "  ".join("-" * w for w in widths))
    for r in rows:
        print(fmt(r))


def pct(x):
    return "%.2f%%" % (100.0 * x) if x is not None else "n/a"


def pfmt(p):
    if p is None:
        return "n/a"
    return "%.4f" % p if p >= 1e-4 else "%.2e" % p


# ---------------------------------------------------------------------------------------------------------
# Per-file analysis
# ---------------------------------------------------------------------------------------------------------
def analyse_file(task, path, spec, universe, cap_declared):
    rows, malformed = read_jsonl(path)
    model, arm, variant = split_run_name(os.path.basename(path))
    fields = spec["answer_fields"]

    # Duplicate item_ids: keep the last, report the collision.
    by_item, dupes = {}, 0
    for r in rows:
        iid = r.get("item_id")
        if iid is None:
            continue
        if iid in by_item:
            dupes += 1
        by_item[iid] = r
    rows_unique = list(by_item.values())

    cap, cap_note, tol = infer_cap(rows_unique, cap_declared)

    states = {iid: answer_state(r, fields) for iid, r in by_item.items()}
    answered = {i for i, s in states.items() if s == "answered"}
    partial = {i for i, s in states.items() if s == "partial"}
    failed = {i for i, s in states.items() if s == "failed"}

    causes = collections.Counter()
    examples = collections.defaultdict(list)
    for iid in sorted(failed | partial):
        cls, snippet = classify(by_item[iid], fields, cap, tol)
        tag = cls if iid in failed else cls + " (partial)"
        causes[tag] += 1
        if snippet and len(examples[tag]) < 3:
            examples[tag].append((iid, snippet))

    # Answered rows whose body did not finish: scored by the paper, but the text was cut. This test is
    # structural and does not consult the cap, so it stays valid on a provider whose completion_tokens
    # includes hidden reasoning. How many of them sit at the cap is reported alongside, as the cause.
    truncated_but_scored = [i for i in sorted(answered)
                            if not body_complete(by_item[i].get("raw"))]
    cut_at_cap = 0
    if cap is not None:
        for i in truncated_but_scored:
            ct = completion_tokens(by_item[i])
            if ct is not None and ct >= cap - max(tol, 1):
                cut_at_cap += 1

    # How the scorer's repair of a partial row actually scores, against how a real answer scores. Same formula
    # as run_mesogeos.score: call = probability >= 0.5 when the model left "fire" out.
    imp_n = imp_right = ans_n = ans_right = 0
    if spec.get("imputed_call"):
        for iid, r in by_item.items():
            lab = r.get("label")
            if lab is None:
                continue
            truth = int(lab) == 1
            if states[iid] == "partial" and r.get("call") is None and r.get("probability") is not None:
                imp_n += 1
                imp_right += int(((r["probability"] or 0) >= 0.5) == truth)
            elif states[iid] == "answered":
                ans_n += 1
                ans_right += int(bool(r["call"]) == truth)

    attempted = len(by_item)
    fail_rate = (len(failed) / attempted) if attempted else None
    degraded_rate = ((len(failed) + len(partial)) / attempted) if attempted else None

    return {
        "path": path, "file": os.path.basename(path), "task": task,
        "model": model, "arm": arm, "variant": variant,
        "attempted": attempted, "rows_in_file": len(rows), "duplicate_item_ids": dupes,
        "malformed_lines": malformed,
        "answered": len(answered), "partial": len(partial), "failed": len(failed),
        "failure_rate": fail_rate, "degraded_rate": degraded_rate,
        "cap": cap, "cap_note": cap_note, "cap_tolerance": tol,
        "causes": dict(causes), "examples": {k: v for k, v in examples.items()},
        "imputed_rows": imp_n, "imputed_rows_matching_label": imp_right,
        "answered_rows_scored": ans_n, "answered_rows_matching_label": ans_right,
        "truncated_but_scored": len(truncated_but_scored),
        "truncated_but_scored_at_cap": cut_at_cap,
        "truncated_but_scored_ids": truncated_but_scored[:5],
        "_answered_ids": answered, "_failed_ids": failed, "_partial_ids": partial,
        "_attempted_ids": set(by_item), "_rows": by_item,
    }


def stratify(task, spec, res, universe, name, event="failed"):
    """(ordered table) of events per stratum for one run.

    event='failed'   counts only rows with no usable answer at all.
    event='degraded' also counts rows missing one required field, which the scorer keeps by imputing it. Those
                     rows never appear in a failure rate, yet they are items where the model did not answer the
                     question asked, so whether they land at random is the same question the reviewer posed.
    """
    hit = res["_failed_ids"] if event == "failed" else (res["_failed_ids"] | res["_partial_ids"])
    fails, oks, offsets = collections.Counter(), collections.Counter(), collections.defaultdict(list)
    for iid, row in res["_rows"].items():
        val = stratum_value(task, name, row, universe.get(iid))
        if val is None:
            continue
        off = row.get("offset_seconds")
        if off is None and isinstance(universe.get(iid), dict):
            off = universe[iid].get("offset_seconds")
        if isinstance(off, (int, float)):
            offsets[val].append(off)
        if iid in hit:
            fails[val] += 1
        else:
            oks[val] += 1
    med = {k: statistics.median(v) for k, v in offsets.items() if v}
    order = stratum_order(task, name, set(fails) | set(oks), med)
    return [(v, fails[v], oks[v]) for v in order]


# ---------------------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    help="benchmark repository root")
    ap.add_argument("--json", default=None, help="also write the full result as JSON to this path")
    ap.add_argument("--flag-rate", type=float, default=FAIL_RATE_FLAG,
                    help="failure rate above which a run is called out (default 0.02)")
    ap.add_argument("--min-coverage", type=float, default=0.90,
                    help="a run covering less than this fraction of the best coverage in its own arm is "
                         "treated as in-progress or aborted and left out of the common-set arithmetic "
                         "(default 0.90)")
    ap.add_argument("--min-task-coverage", type=float, default=0.50,
                    help="second bar, against the best coverage anywhere in the task, which catches a short "
                         "run that is alone in its arm and so passes --min-coverage trivially (default 0.50; "
                         "a legitimately smaller grounded arm sits near 0.88)")
    args = ap.parse_args()

    root = args.root
    print("=" * 118)
    print("ANSWER FAILURES: who failed to answer, why, and whether the failures are random")
    print("root: %s" % root)
    print("=" * 118)
    print("An item is FAILED when every answer field the scorer needs is null, PARTIAL when some but not all")
    print("are null (the scorer imputes the rest), ANSWERED otherwise. Rates below are over attempted rows.")

    report = {"root": root, "flag_rate": args.flag_rate, "tasks": {}}
    all_results, tests_for_holm = [], []

    for task, spec in TASKS.items():
        task_dir = os.path.join(root, task)
        print()
        rule("=")
        print("TASK  %s" % task)
        rule("=")
        if not os.path.isdir(task_dir):
            print("  directory absent, skipped")
            continue

        uni_name, universe = load_universe(task_dir, spec["universe_files"])
        cap_declared = declared_cap(root, spec["runner"])
        runs = find_runs(task_dir)
        print("  item universe file : %s (%d items)" % (uni_name or "none found", len(universe)))
        print("  runner             : %s (declares MAX_OUT = %s)"
              % (spec["runner"], cap_declared if cap_declared else "not found"))
        print("  answer fields      : %s" % ", ".join(spec["answer_fields"]))
        print("  response files     : %d" % len(runs))
        if not runs:
            print("  no responses-*.jsonl present, nothing to characterise")
            continue

        results = [analyse_file(task, p, spec, universe, cap_declared) for p in runs]
        all_results.extend(results)

        # ---- 1 and 2: counts and causes -------------------------------------------------------------
        print()
        print("  [1] Attempted / answered / failed, and the cause of each failure")
        cause_keys = sorted({k for r in results for k in r["causes"]})
        head = ["model", "arm", "var", "attempt", "answer", "partial", "fail", "fail rate",
                "cut but scored", "cap"] + cause_keys
        body = []
        for r in results:
            body.append([r["model"], r["arm"], r["variant"] or "-", r["attempted"], r["answered"],
                         r["partial"], r["failed"], pct(r["failure_rate"]), r["truncated_but_scored"],
                         r["cap"] if r["cap"] else "?"]
                        + [r["causes"].get(k, 0) for k in cause_keys])
        table(head, body, aligns=["<", "<", "<"] + [">"] * (len(head) - 3))
        print("  'cut but scored' counts ANSWERED rows whose body is structurally unfinished: the run scored")
        print("  them anyway. 'cap' is the output-token ceiling inferred for that file.")
        for r in results:
            if r["truncated_but_scored"]:
                print("  %s: %d of the %d cut-but-scored rows stop at the %s-token cap"
                      % (r["file"], r["truncated_but_scored_at_cap"], r["truncated_but_scored"],
                         r["cap"] if r["cap"] else "?"))
        for r in results:
            if r["malformed_lines"] or r["duplicate_item_ids"]:
                print("  NOTE %s: %d malformed line(s), %d duplicate item_id(s)"
                      % (r["file"], r["malformed_lines"], r["duplicate_item_ids"]))
        for r in results:
            if r["cap"]:
                print("  cap for %s: %s%s" % (r["file"], r["cap_note"],
                                              "" if not r["cap_tolerance"]
                                              else ", tolerance %d tokens" % r["cap_tolerance"]))
        shown = 0
        for r in results:
            for tag, exs in sorted(r["examples"].items()):
                for iid, snip in exs:
                    if shown < 12:
                        print("  example %-28s %-34s %r" % (tag, iid[:34], snip))
                        shown += 1

        # ---- 2b: how the scorer's repair of a partial row actually scores ------------------------------
        if spec.get("imputed_call") and any(r["imputed_rows"] for r in results):
            print()
            print("  [2b] Rows the scorer repaired, and whether the repair agrees with the label more often")
            print("  than a real answer does. A repair that is right more often than the model is a score")
            print("  propped up by the imputation rather than earned by the answer.")
            rows2 = []
            for r in results:
                if not r["imputed_rows"]:
                    continue
                ia = r["imputed_rows_matching_label"] / r["imputed_rows"]
                aa = (r["answered_rows_matching_label"] / r["answered_rows_scored"]
                      if r["answered_rows_scored"] else None)
                rows2.append([r["model"], r["arm"], r["variant"] or "-",
                              "%d" % r["imputed_rows"], pct(ia),
                              "%d" % r["answered_rows_scored"], pct(aa),
                              "%+.1f pts" % (100 * (ia - aa)) if aa is not None else "n/a"])
            table(["model", "arm", "var", "repaired rows", "repair agrees", "answered rows",
                   "answer agrees", "gap"], rows2,
                  aligns=["<", "<", "<", ">", ">", ">", ">", ">"])

        # ---- 3: independence of failure from difficulty ----------------------------------------------
        print()
        print("  [3] Is failure independent of difficulty?")
        task_tests = {}
        # The strict class is rows with no usable answer at all. Where a run also has rows missing one required
        # field, the same stratum is tested a second time on the wider degraded class, because that is the class
        # the scorer silently repairs and it can be two orders of magnitude larger than the strict one. A
        # degraded table identical to the strict one is not re-tested, so the Holm family never carries the same
        # test twice.
        has_partial = any(r["partial"] for r in results)
        for name in spec["strata"]:
            strict_tables = {}
            for event in (["failed", "degraded"] if has_partial else ["failed"]):
                key = name if event == "failed" else "%s (degraded)" % name
                print()
                print("  stratum: %s   event: %s" % (
                    name, "no usable answer at all" if event == "failed"
                    else "failed OR partial, the rows the scorer repairs by imputing the missing field"))
                head = ["model", "arm", "var"]
                strata_names = []
                per_run_tables = {}
                for r in results:
                    tbl = stratify(task, spec, r, universe, name, event)
                    per_run_tables[r["file"]] = tbl
                    for s, _, _ in tbl:
                        if s not in strata_names:
                            strata_names.append(s)
                if not strata_names:
                    print("    stratum not derivable from these rows, skipped")
                    continue
                # keep a single consistent order across runs
                med = {}
                for r in results:
                    for iid, row in r["_rows"].items():
                        val = stratum_value(task, name, row, universe.get(iid))
                        off = row.get("offset_seconds")
                        if off is None and isinstance(universe.get(iid), dict):
                            off = universe[iid].get("offset_seconds")
                        if val is not None and isinstance(off, (int, float)):
                            med.setdefault(val, []).append(off)
                med = {k: statistics.median(v) for k, v in med.items()}
                strata_names = stratum_order(task, name, set(strata_names), med)

                label_word = "fail" if event == "failed" else "degr"
                head += ["%s: %s/n" % (s, label_word) for s in strata_names] + ["test", "p", "note"]
                body = []
                for r in results:
                    tbl = dict((s, (f, o)) for s, f, o in per_run_tables[r["file"]])
                    ordered = [(s, tbl.get(s, (0, 0))[0], tbl.get(s, (0, 0))[1]) for s in strata_names]
                    if event == "failed":
                        strict_tables[r["file"]] = ordered
                    test = independence_test(ordered)
                    task_tests.setdefault(key, {})[r["file"]] = test
                    p = best_p(test)
                    duplicate = event == "degraded" and strict_tables.get(r["file"]) == ordered
                    if p is not None and not duplicate:
                        tests_for_holm.append((task, key, r["file"], p))
                    cells = []
                    for s, f, o in ordered:
                        n = f + o
                        cells.append("%d/%d (%s)" % (f, n, pct(f / n) if n else "n/a"))
                    note = test.get("reason", "")
                    if "perm_p" in test:
                        note = "permutation p=%s (use this one)" % pfmt(test["perm_p"])
                    if duplicate:
                        note = (note + "; " if note else "") + "same table as the strict test, not counted twice"
                    body.append([r["model"], r["arm"], r["variant"] or "-"] + cells
                                + [test["test"], pfmt(test.get("p")), note])
                table(head, body)

                # pooled view
                pooled = collections.Counter()
                for r in results:
                    for s, f, o in per_run_tables[r["file"]]:
                        pooled[(s, "f")] += f
                        pooled[(s, "o")] += o
                ordered = [(s, pooled[(s, "f")], pooled[(s, "o")]) for s in strata_names]
                ptest = independence_test(ordered)
                task_tests.setdefault(key, {})["__pooled__"] = ptest
                cells = " | ".join("%s %d/%d (%s)" % (s, f, f + o, pct(f / (f + o)) if (f + o) else "n/a")
                                   for s, f, o in ordered)
                print("    pooled over all runs: %s" % cells)
                print("    pooled test: %s p=%s%s %s"
                      % (ptest["test"], pfmt(ptest.get("p")),
                         "" if "perm_p" not in ptest else ", permutation p=%s (use this one)" % pfmt(ptest["perm_p"]),
                         ptest.get("reason", "")))
                print("    the pooled row treats the same item under different models as separate draws, so it is")
                print("    a descriptive signal only, not a valid significance test.")

        # ---- 4: the common set ------------------------------------------------------------------------
        print()
        print("  [4] The set every run answered")
        # A run that covers far fewer items than its arm-mates is in progress or was aborted, not a run whose
        # model refused. Including one collapses the intersection to its own length and hides the real
        # failure-driven shrinkage, so it is reported separately rather than folded in.
        # Two bars, because one is not enough. The arm bar catches a half-finished run beside its arm-mates. It
        # cannot catch a run that is alone in its arm, and every run is alone in its arm when a filename does
        # not end in a recognised condition: a three-row probe file lands in the "unknown" arm, is by
        # definition the best coverage there, and drags the intersection down to three items while the loss is
        # reported as arm design. The task bar catches that, and is loose enough that a grounded arm which is
        # legitimately smaller than its bare arm (FIgLib drops 28 reference frames, 87.5%) still passes.
        arm_max = collections.defaultdict(int)
        for r in results:
            arm_max[r["arm"]] = max(arm_max[r["arm"]], r["attempted"])
        task_max = max((r["attempted"] for r in results), default=0)
        for r in results:
            r["coverage"] = (r["attempted"] / arm_max[r["arm"]]) if arm_max[r["arm"]] else 0.0
            r["task_coverage"] = (r["attempted"] / task_max) if task_max else 0.0
            r["complete"] = (r["coverage"] >= args.min_coverage
                             and r["task_coverage"] >= args.min_task_coverage)
        incomplete = [r for r in results if not r["complete"]]
        kept = [r for r in results if r["complete"]]
        usable = kept or results
        if incomplete:
            print("  Excluded as in-progress or aborted (below %.0f%% of the best coverage in the same arm, or"
                  % (100 * args.min_coverage))
            print("  below %.0f%% of the best coverage anywhere in the task):" % (100 * args.min_task_coverage))
            for r in incomplete:
                print("    %-52s %d items: %.1f%% of the %s arm's best (%d), %.1f%% of the task's best (%d)"
                      % (r["file"][:52], r["attempted"], 100 * r["coverage"], r["arm"], arm_max[r["arm"]],
                         100 * r["task_coverage"], task_max))
            if kept:
                print("  The sets below are over the %d remaining run(s)." % len(usable))
            else:
                print("  EVERY run fell below the bar, so excluding them would leave nothing to intersect. The")
                print("  sets below are over ALL %d run(s) and are provisional on those runs finishing."
                      % len(usable))

        attempted_all = set.intersection(*[r["_attempted_ids"] for r in usable]) if usable else set()
        attempted_any = set.union(*[r["_attempted_ids"] for r in usable]) if usable else set()
        answered_all = set.intersection(*[r["_answered_ids"] for r in usable]) if usable else set()
        answered_all_loose = set.intersection(
            *[r["_answered_ids"] | r["_partial_ids"] for r in usable]) if usable else set()
        n_uni = len(universe) if universe else len(attempted_any)
        design_loss = len(attempted_any) - len(attempted_all)
        fail_loss = len(attempted_all) - len(answered_all)
        # Split that loss. A row with no usable answer at all is dropped by the scorer; a row missing one
        # required field is kept and repaired. Calling both "answer failures" turns 5 dropped Mesogeos items
        # into 114 of them, which is the opposite of the honesty this section exists to provide.
        hard_loss = len(attempted_all) - len(answered_all_loose)
        impute_loss = len(answered_all_loose) - len(answered_all)
        rows = [
            ["items in %s" % (uni_name or "n/a"), n_uni, "-"],
            ["attempted by at least one run", len(attempted_any),
             "%s of the item file was never attempted, by the runner's own filter or because these runs "
             "did not reach it (best single run: %d items)"
             % (pct((n_uni - len(attempted_any)) / n_uni), task_max) if n_uni else "-"],
            ["attempted by EVERY run", len(attempted_all),
             "%d lost to arm design (e.g. FIgLib reference frames dropped from the grounded arm)"
             % design_loss],
            ["answered by EVERY run (strict)", len(answered_all),
             "%d lost: %d where some run produced no usable answer at all, %d where some run left a required "
             "field out and the scorer imputed it" % (fail_loss, hard_loss, impute_loss)],
            ["answered or partial by EVERY run", len(answered_all_loose),
             "the widest set the scorer can actually use, since it imputes the missing field"],
        ]
        table(["set", "n", "note"], rows, aligns=["<", ">", "<"])
        if len(attempted_any):
            print("  COMMON SET: %d items. That is %.2f%% of the %d attempted by any run and %.2f%% of the %d"
                  % (len(answered_all), 100.0 * len(answered_all) / len(attempted_any),
                     len(attempted_any), 100.0 * len(answered_all) / max(n_uni, 1), n_uni))
            print("  in the item file. Scoring each run on its own answered subset compares different item sets.")
            print("  Of the %d items outside the common set, %d are outside because a run gave no usable answer"
                  % (len(attempted_all) - len(answered_all), hard_loss))
            print("  and %d because a run left a required field out and the scorer filled it in." % impute_loss)

        # per-arm common set: the cross-arm intersection is capped by design, so report within arm too
        by_arm = collections.defaultdict(list)
        for r in usable:
            by_arm[r["arm"]].append(r)
        arm_rows = []
        for arm, rs in sorted(by_arm.items()):
            att = set.intersection(*[r["_attempted_ids"] for r in rs])
            ans = set.intersection(*[r["_answered_ids"] for r in rs])
            arm_rows.append([arm, len(rs), len(att), len(ans), len(att) - len(ans)])
        table(["arm", "runs kept", "attempted by all", "answered by all", "lost to failures"], arm_rows,
              aligns=["<", ">", ">", ">", ">"])

        # items that more than one run could not answer
        per_item = collections.Counter()
        for r in usable:
            for iid in r["_failed_ids"]:
                per_item[iid] += 1
        multi = [(iid, n) for iid, n in per_item.items() if n >= 2]
        print("  items failed by at least two runs: %d" % len(multi))
        for iid, n in sorted(multi, key=lambda x: (-x[1], x[0]))[:10]:
            hint = ""
            it = universe.get(iid) or {}
            for key in ("label", "bucket", "offset_seconds"):
                if it.get(key) is not None:
                    hint += " %s=%s" % (key, it[key])
            print("    %-56s failed by %d run(s)%s" % (iid[:56], n, hint))

        report["tasks"][task] = {
            "universe_file": uni_name, "universe_n": n_uni,
            "declared_cap": cap_declared,
            "runs": [{k: v for k, v in r.items() if not k.startswith("_")} for r in results],
            "tests": task_tests,
            "attempted_any": len(attempted_any), "attempted_all": len(attempted_all),
            "answered_all": len(answered_all), "answered_all_loose": len(answered_all_loose),
            "design_loss": design_loss, "failure_loss": fail_loss,
            "loss_no_usable_answer": hard_loss, "loss_imputed_field": impute_loss,
            "best_single_run_attempted": task_max,
            "items_failed_by_two_or_more_runs": len(multi),
            "excluded_incomplete_runs": [r["file"] for r in incomplete],
            "runs_in_common_set": [r["file"] for r in usable],
        }

    # ---- multiplicity -------------------------------------------------------------------------------
    print()
    rule("=")
    print("MULTIPLE TESTING")
    rule("=")
    if tests_for_holm:
        adj = holm([t[3] for t in tests_for_holm])
        sig = [(t, a) for t, a in zip(tests_for_holm, adj) if a < 0.05]
        print("  %d independence tests were computable. Holm-adjusted across all of them, %d reach p<0.05."
              % (len(tests_for_holm), len(sig)))
        for (task, name, f, p), a in sig:
            print("    %s %s %s: raw p=%s, Holm p=%s" % (task, name, f, pfmt(p), pfmt(a)))
        if sig:
            print("  Failure is NOT random with respect to difficulty in the run(s) listed above. Scoring the")
            print("  remainder of those runs compares a filtered item set against an unfiltered one.")
        else:
            print("  No stratum-by-failure association survives correction. On these files the failures are")
            print("  consistent with being random with respect to the difficulty strata tested. That is a")
            print("  statement about power as much as about the data: most runs have single-digit failures.")
        report["holm"] = [{"task": t[0], "stratum": t[1], "file": t[2], "p": t[3], "holm_p": a}
                          for t, a in zip(tests_for_holm, adj)]
    else:
        print("  No independence test was computable (no run had both failures and answered rows).")

    # ---- one-line summary per file -------------------------------------------------------------------
    print()
    rule("=")
    print("ONE LINE PER FILE")
    rule("=")
    for r in all_results:
        causes = ", ".join("%s=%d" % (k, v) for k, v in sorted(r["causes"].items())) or "no failures"
        if not r.get("complete", True):
            causes = "[IN PROGRESS OR ABORTED, %.0f%% coverage] " % (100 * r.get("coverage", 0)) + causes
        print("  %-16s %-46s attempted %4d | answered %4d | partial %4d | failed %3d (%7s) | %s"
              % (r["task"].replace("task-", ""), r["file"][:46], r["attempted"], r["answered"],
                 r["partial"], r["failed"], pct(r["failure_rate"]), causes))

    # ---- the plain statement asked for ----------------------------------------------------------------
    print()
    rule("=")
    print("RUNS WHOSE SCORES ARE LEAST TRUSTWORTHY")
    rule("=")
    def tag(r):
        return "" if r.get("complete", True) else "  [IN PROGRESS OR ABORTED, %.0f%% coverage]" % (
            100 * r.get("task_coverage", 0))

    over = [r for r in all_results if (r["failure_rate"] or 0) > args.flag_rate]
    print("  Failure rate above %s (no usable answer at all):" % pct(args.flag_rate))
    if over:
        for r in sorted(over, key=lambda x: -(x["failure_rate"] or 0)):
            print("    %-16s %-46s %7s  (%d of %d items unanswered)%s"
                  % (r["task"].replace("task-", ""), r["file"][:46], pct(r["failure_rate"]),
                     r["failed"], r["attempted"], tag(r)))
    else:
        print("    none. Every run answered more than %s of the items it attempted."
              % pct(1 - args.flag_rate))

    degraded = [r for r in all_results
                if (r["degraded_rate"] or 0) > args.flag_rate and r not in over]
    print("  Degraded rate above %s (failed OR partial, where the scorer imputes the missing field):"
          % pct(args.flag_rate))
    if degraded:
        for r in sorted(degraded, key=lambda x: -(x["degraded_rate"] or 0)):
            print("    %-16s %-46s %7s  (%d failed + %d partial of %d)%s"
                  % (r["task"].replace("task-", ""), r["file"][:46], pct(r["degraded_rate"]),
                     r["failed"], r["partial"], r["attempted"], tag(r)))
    else:
        print("    none beyond those already listed.")

    cut = [r for r in all_results if r["truncated_but_scored"] > 0]
    print("  Runs that scored answers whose body was structurally unfinished (a cut answer that still")
    print("  parsed is counted as a result by the scorer, so it never shows up as a failure at all):")
    if cut:
        for r in sorted(cut, key=lambda x: -x["truncated_but_scored"]):
            print("    %-16s %-46s %d of %d answered rows (%d at the %s-token cap)%s"
                  % (r["task"].replace("task-", ""), r["file"][:46], r["truncated_but_scored"],
                     r["answered"], r["truncated_but_scored_at_cap"], r["cap"] if r["cap"] else "?", tag(r)))
    else:
        print("    none.")

    report["summary"] = {
        "files": len(all_results),
        "over_flag_rate": [r["file"] for r in over],
        "degraded_over_flag_rate": [r["file"] for r in degraded],
        "scored_incomplete_bodies": {r["file"]: r["truncated_but_scored"] for r in cut},
    }

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=1, default=str)
        print()
        print("  wrote %s" % args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
