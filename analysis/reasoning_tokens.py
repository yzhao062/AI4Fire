"""Reasoning tokens the providers report, per model, task, and arm: the reasoning-token table of the
reproducibility appendix (Reasoning Settings and Token Use) and the counts its prose quotes.

No request in this benchmark set a reasoning, thinking, or effort parameter (gw.py), so any reasoning a model did
ran at its serving endpoint's default, and the stored usage records are the only view of it. This script reads
those records from the main arm files of the five tasks (responses-<stem>-<arm>.jsonl, no variant suffix) and
reports, per arm:

  n                          responses in the file
  n_reasoning_field          responses whose usage carries completion_tokens_details.reasoning_tokens
  median_reasoning           median of that field over the responses that carry it (null when none does)
  p90_reasoning              90th percentile of the same values, lower method (not printed in the paper)
  median_completion          median completion tokens; the gateway providers count reasoning inside them
  near_cap                   responses whose completion tokens reach the output cap less --margin tokens
  near_cap_answered          of those, responses the runner still parsed an answer from (its fallbacks included)
  near_cap_reached_answer    of those, responses whose text reaches the answer format the prompt asks for (the
                             ANSWER: line for tool use, the JSON answer keys elsewhere; see ANSWER_FORMAT)
  near_cap_correct           of those, responses scored correct, where the runner stores a correct flag
  near_cap_with_stop_reason  of those, responses whose record stores a provider stop reason
  n_stop_reason              responses that store a stop reason at all; only the Bedrock path keeps one
  stopped_at_cap             responses whose stored stop reason is max_tokens or length
  table_cell                 the paper's cell: the median rounded half up, or n/r unless every response reports

A response without the reasoning field is not reported (n/r), never zero: gemini-3.1-pro's bare smoke file carries
the field on 22 of its 224 records, so that cell is n/r rather than a median of the 22. The paper's table shows the
four gateway models; Bedrock returns no reasoning count, so both open-weight rows are n/r throughout. The tool-use
column is the bare arm. The stored tool-arm files predate run_tooluse._add_usage: the tool loop that produced them
summed only prompt and completion tokens over its calls, so they carry no reasoning count for any model, although
the same gateway endpoint reports one on the bare rows. A sum of calls cannot be held against a per-call cap either,
so the tool arm's near_cap is null.

A response near the cap may still have been cut before its answer. The runners' parsers fall back to the last number
or line when the requested format is missing, so near_cap_answered counts such fallbacks as answers;
near_cap_reached_answer does not. For gemini-3.1-pro's bare tool use, none of the 61 near-cap responses reaches its
ANSWER: line, and the runner's fallback scored 3 of them correct.

The cap is read per response from usage.max_out where the runner recorded it, and otherwise from models.py (1,536
tokens for every core model). With the default margin of 16 tokens, a core response counts as near the cap at 1,520
completion tokens or more.

Inputs: task-*/responses-<stem>-<arm>.jsonl for the selected models (models.py; default the six core models).
Output: analysis/reasoning_tokens.json with the per-arm records, the table rows, and a per-model summary. Stdout
prints the per-arm lines, the table rows in LaTeX, and the summary counts.

    python analysis/reasoning_tokens.py                 # six core models, under a second on this machine
    python analysis/reasoning_tokens.py --tier every    # all 35 registry models, same definitions
"""
import argparse
import decimal
import json
import pathlib
import re
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import answer_failures as af  # noqa: E402
import cluster_uncertainty as cu  # noqa: E402
from models import TASK_CONDITIONS, add_model_args, resolve_models  # noqa: E402
from served_models import TASK_SHORT  # noqa: E402

# Column order of the appendix table. Each column joins its arms with "/"; the tool-use column is the bare arm.
TABLE_COLUMNS = (("allocation", ("bare", "grounded")), ("figlib", ("bare", "grounded")),
                 ("wildfirevqa", ("bare", "grounded")), ("mesogeos", ("bare", "grounded")), ("tooluse", ("bare",)))
# Arms whose stored usage sums several model calls, so no per-call cap applies to their token counts.
MULTI_CALL_ARMS = {("tooluse", "tool")}
CAP_STOPS = ("max_tokens", "length")
# The answer format each runner's prompt asks for, as the runner's own primary pattern matches it (run_allocation,
# run_figlib, run_wildfirevqa, run_mesogeos, and run_tooluse parse or parse_answer, before any fallback).
ANSWER_FORMAT = {
    "allocation": (r'"personnel"\s*:\s*-?[\d.]+',),
    "figlib": (r'"smoke"\s*:\s*(true|false)',),
    "wildfirevqa": (r'"answer"\s*:\s*"[^"]*"', r'"applicability_score"\s*:\s*-?[0-9.]+'),
    "mesogeos": (r'"probability"\s*:\s*[0-9.]+', r'"fire"\s*:\s*(true|false)'),
    "tooluse": (r'(?im)^\s*ANSWER\s*:\s*\S',),
}


def reasoning_tokens(row):
    """usage.completion_tokens_details.reasoning_tokens, or None when the record does not report it."""
    value = ((row.get("usage") or {}).get("completion_tokens_details") or {}).get("reasoning_tokens")
    return value if isinstance(value, (int, float)) else None


def completion_tokens(row):
    value = (row.get("usage") or {}).get("completion_tokens")
    return value if isinstance(value, (int, float)) else None


def near_cap(row, default_cap, margin):
    """Completion tokens reach the output cap less the margin. The cap is usage.max_out where the runner recorded
    it, else the model's registry value."""
    tokens = completion_tokens(row)
    cap = (row.get("usage") or {}).get("max_out") or default_cap
    return tokens is not None and tokens >= cap - margin


def stop_reason(row):
    """The provider's stop reason when the record stores one. gw.bedrock_chat keeps Bedrock's stopReason in
    usage.stop_reason; gw.chat returns the gateway's usage without the choice's finish_reason, so a gateway
    record carries none."""
    return (row.get("usage") or {}).get("stop_reason") or row.get("finish_reason")


def answered(row, task):
    """The runner parsed an answer: every answer field is set (answer_failures.answer_state over that script's
    field list where it has one; the other runners store their parsed answer in prediction), and the answer is not
    an abstention (run_tooluse parses ANSWER: unknown as one)."""
    fields = af.TASKS.get("task-" + task, {}).get("answer_fields", ["prediction"])
    return af.answer_state(row, fields) == "answered" and row.get("abstained") is not True


def reached_answer(row, task):
    """The raw text contains every part of the answer format the prompt asks for (ANSWER_FORMAT), with no fallback."""
    raw = row.get("raw") or ""
    return all(re.search(p, raw, re.I) for p in ANSWER_FORMAT[task])


def half_up(x):
    """Round half up to an integer, as the table prints medians (430.5 prints as 431)."""
    return int(decimal.Decimal(x).to_integral_value(rounding=decimal.ROUND_HALF_UP))


def arm_record(path, task, model, arm, margin):
    """Token statistics of one response file; the fields are listed in the module docstring."""
    rows, bad = cu.read_jsonl(path)
    if bad:
        sys.exit("%s: %d malformed lines; a partial file would change every count" % (path, bad))
    reasoning = [v for v in map(reasoning_tokens, rows) if v is not None]
    completion = [v for v in map(completion_tokens, rows) if v is not None]
    stops = [s for s in map(stop_reason, rows) if s is not None]
    rec = {"task": TASK_SHORT[task], "model": model.label, "arm": arm, "path": path.relative_to(ROOT).as_posix(),
           "n": len(rows), "n_reasoning_field": len(reasoning),
           "median_reasoning": statistics.median(reasoning) if reasoning else None,
           "p90_reasoning": sorted(reasoning)[int(0.9 * (len(reasoning) - 1))] if reasoning else None,
           "median_completion": statistics.median(completion) if completion else None,
           "near_cap": None, "near_cap_answered": None, "near_cap_reached_answer": None, "near_cap_correct": None,
           "near_cap_with_stop_reason": None,
           "n_stop_reason": len(stops), "stopped_at_cap": sum(1 for s in stops if s in CAP_STOPS)}
    if (task, arm) not in MULTI_CALL_ARMS:
        near = [r for r in rows if near_cap(r, model.max_out, margin)]
        rec["near_cap"] = len(near)
        rec["near_cap_answered"] = sum(1 for r in near if answered(r, task))
        rec["near_cap_reached_answer"] = sum(1 for r in near if reached_answer(r, task))
        rec["near_cap_correct"] = (sum(1 for r in near if r.get("correct") is True)
                                   if any("correct" in r for r in rows) else None)
        rec["near_cap_with_stop_reason"] = sum(1 for r in near if stop_reason(r) is not None)
    every_response_reports = bool(rows) and len(reasoning) == len(rows)
    rec["table_cell"] = format(half_up(rec["median_reasoning"]), ",") if every_response_reports else "n/r"
    return rec


def model_summary(model, records, margin):
    """The counts the appendix prose quotes for one model: the range of its printed cells, the cells printed n/r
    with how many responses carry the field, the arms with responses near the cap, and how many responses carry a
    reasoning count or a stop reason."""
    mine = [r for r in records.values() if r["model"] == model.label]
    by_cell = {(r["task"], r["arm"]): r for r in mine}
    values, not_reported = [], {}
    for task, arms in TABLE_COLUMNS:
        for arm in arms:
            rec = by_cell.get((TASK_SHORT[task], arm))
            if rec is None:
                continue
            if rec["table_cell"] == "n/r":
                not_reported["%s|%s" % (rec["task"], arm)] = "%d of %d" % (rec["n_reasoning_field"], rec["n"])
            else:
                values.append(half_up(rec["median_reasoning"]))
    near = [r for r in mine if r["near_cap"]]
    return {"cap": model.max_out, "near_cap_threshold": model.max_out - margin,
            "table_cells": [min(values), max(values)] if values else None,
            "cells_not_reported": not_reported,
            "responses": sum(r["n"] for r in mine),
            "responses_with_reasoning_field": sum(r["n_reasoning_field"] for r in mine),
            "responses_with_stop_reason": sum(r["n_stop_reason"] for r in mine),
            "near_cap": {"%s|%s" % (r["task"], r["arm"]): r["near_cap"] for r in near},
            "near_cap_answered": {"%s|%s" % (r["task"], r["arm"]): r["near_cap_answered"] for r in near},
            "near_cap_reached_answer": {"%s|%s" % (r["task"], r["arm"]): r["near_cap_reached_answer"] for r in near},
            "near_cap_correct": {"%s|%s" % (r["task"], r["arm"]): r["near_cap_correct"] for r in near},
            "near_cap_with_stop_reason": {"%s|%s" % (r["task"], r["arm"]): r["near_cap_with_stop_reason"] for r in near}}


def fmt(x):
    return "--" if x is None else format(x, ",g") if isinstance(x, float) else format(x, ",")


def main():
    ap = argparse.ArgumentParser(description="Reasoning tokens that the providers report, per model, task, and arm, "
                                             "and responses near the output cap.")
    ap.add_argument("--margin", type=int, default=16,
                    help="a response within this many tokens of its output cap counts as near the cap (default 16)")
    ap.add_argument("--out", type=pathlib.Path, default=HERE / "reasoning_tokens.json",
                    help="output JSON (default analysis/reasoning_tokens.json)")
    add_model_args(ap, default_tier="core")
    args = ap.parse_args()
    selected = resolve_models(args)

    records = {}
    for task, _ in TABLE_COLUMNS:
        for m in selected:
            for arm in TASK_CONDITIONS[task]:
                path = ROOT / ("task-" + task) / ("responses-%s-%s.jsonl" % (m.stem, arm))
                if not path.exists():
                    print("%s: not run, skipped" % path.relative_to(ROOT).as_posix(), file=sys.stderr)
                    continue
                records["%s|%s|%s" % (TASK_SHORT[task], m.label, arm)] = arm_record(path, task, m, arm, args.margin)

    def cell(task, label, arm):
        rec = records.get("%s|%s|%s" % (TASK_SHORT[task], label, arm))
        return "--" if rec is None else rec["table_cell"]

    table = {m.label: {TASK_SHORT[task]: "/".join(cell(task, m.label, arm) for arm in arms) for task, arms in TABLE_COLUMNS}
             for m in selected}
    summary = {m.label: model_summary(m, records, args.margin) for m in selected}
    tool_arm = [r for r in records.values() if r["task"] == TASK_SHORT["tooluse"] and r["arm"] == "tool"]
    out = {"near_cap_margin": args.margin, "arms": records, "table": table, "summary": summary,
           "tool_arm": {"responses": sum(r["n"] for r in tool_arm),
                        "responses_with_reasoning_field": sum(r["n_reasoning_field"] for r in tool_arm)}}
    args.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    print("Reasoning tokens per response as the providers report them; n/r: not reported on every response.\n")
    line = "%-12s %-17s %-8s %5s %9s %8s %6s %11s %9s %9s %10s"
    print(line % ("task", "model", "arm", "n", "reported", "median", "p90", "med. compl.", "near cap", "answered",
                  "stop rsn."))
    for r in records.values():
        print(line % (r["task"], r["model"], r["arm"], r["n"], r["n_reasoning_field"], fmt(r["median_reasoning"]),
                      fmt(r["p90_reasoning"]), fmt(r["median_completion"]), fmt(r["near_cap"]),
                      fmt(r["near_cap_answered"]), r["n_stop_reason"]))

    # Like the paper, the LaTeX rows leave out a model with no reported cell; the JSON keeps every row.
    print("\nTable rows (%s; bare/grounded, tool use bare):" % ", ".join(TASK_SHORT[t] for t, _ in TABLE_COLUMNS))
    for label, row in table.items():
        if summary[label]["table_cells"] is not None:
            print("%s & %s \\\\" % (label, " & ".join(row.values())))
    unreported = [label for label in table if summary[label]["table_cells"] is None]
    if unreported:
        print("no reported cell, left out of the rows: %s" % ", ".join(unreported))

    print("\nPer model:")
    for label, s in summary.items():
        cells = "all cells n/r" if s["table_cells"] is None else "cells %s to %s" % tuple(map(fmt, s["table_cells"]))
        partial = ["%s %s" % (k.replace("|", " "), v) for k, v in s["cells_not_reported"].items() if not v.startswith("0 ")]
        print("  %-17s %s%s; reasoning field on %s of %s responses; stop reason on %s" % (
            label, cells, " (n/r, partly reported: %s)" % ", ".join(partial) if partial else "",
            fmt(s["responses_with_reasoning_field"]), fmt(s["responses"]), fmt(s["responses_with_stop_reason"])))
        for k, n in s["near_cap"].items():
            correct = s["near_cap_correct"][k]
            print("  %-17s near the cap (at least %s of %s tokens) on %s: %d, parsed %d, reached the answer format %d,"
                  "%s with a stop reason %d" % (
                      "", fmt(s["near_cap_threshold"]), fmt(s["cap"]), k.replace("|", " "), n, s["near_cap_answered"][k],
                      s["near_cap_reached_answer"][k], "" if correct is None else " scored correct %d," % correct,
                      s["near_cap_with_stop_reason"][k]))
    print("  tool arm: reasoning field on %s of %s responses" % (
        fmt(out["tool_arm"]["responses_with_reasoning_field"]), fmt(out["tool_arm"]["responses"])))
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
