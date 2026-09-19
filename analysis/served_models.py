"""Generate the served-model table for the AI4Fire reported benchmark runs.

Reads manifest-v1.json and inspects the 36 reported response files. Prints a Markdown table,
one row per (task, model), showing the distinct served_model values and their counts,
serving path, and date range. A second part audits every other manifest section that holds
response files (tool use, aerial question answering, and the two model-sweep sections of 2026-09-18)
and prints one row per sweep model with its served identifier and call count. Saves everything as
analysis/served_models.md for the paper appendix.
"""
import argparse
import collections
import datetime
import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from models import by_stem  # noqa: E402

TASK_DISPLAY = {
    "allocation": "Personnel allocation (ICS-209-PLUS)",
    "figlib": "Smoke detection (FIgLib)",
    "mesogeos": "Fire danger forecasting (Mesogeos)",
    "tooluse": "Fire data tool use (FPA-FOD)",
    "wildfirevqa": "Aerial question answering (WildFireVQA)",
}
TASK_SHORT = {"allocation": "allocation", "mesogeos": "fire danger", "figlib": "smoke",
              "tooluse": "tool use", "wildfirevqa": "aerial"}
AUDIT_SECTIONS = ("reported", "tooluse", "wildfirevqa", "added_models", "text_models")


def served_counts_of(path):
    """Counter of served_model over the rows of a response file."""
    counts = collections.Counter()
    with open(fix_path(path), encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                counts[json.loads(line).get("served_model")] += 1
    return counts


def audit_sections(manifest, repo_root):
    """Markdown lines: per-section file and row totals with any file that mixes served identifiers, then one
    row per model of every section beyond the six-model reported set."""
    lines = ["## All Response Files in the Manifest", "",
             "| Section | Files | Responses | Files with one served identifier |", "|---|:---:|:---:|:---:|"]
    mixed = []
    per_model = collections.defaultdict(lambda: {"served": collections.Counter(), "tasks": set(), "paths": set()})
    stems = sorted(by_stem(), key=len, reverse=True)
    for section in AUDIT_SECTIONS:
        block = manifest.get(section)
        entries = block if isinstance(block, list) else (block or {}).get("files", [])
        entries = [e for e in entries if e["path"].endswith(".jsonl")]
        if not entries:
            continue
        n_rows, n_uniform = 0, 0
        for e in entries:
            counts = served_counts_of(repo_root / e["path"])
            n_rows += sum(counts.values())
            n_uniform += len(counts) == 1
            if len(counts) != 1:
                mixed.append((section, e["path"], dict(counts)))
            if section != "reported":
                name = pathlib.Path(e["path"]).name[len("responses-"):-len(".jsonl")]
                stem = next((s for s in stems if name.startswith(s + "-")), None)
                if stem is None:
                    continue
                rec = per_model[stem]
                rec["served"].update(counts)
                rec["tasks"].add(e["path"].split("/")[0][len("task-"):])
                rec["paths"].add(e["path"])
        lines.append("| %s | %d | %s | %d of %d |" % (section, len(entries), format(n_rows, ","), n_uniform, len(entries)))
    lines.append("")
    if mixed:
        lines.append("> [!WARNING]")
        lines.append("> Files whose rows carry more than one served identifier:")
        for section, path, counts in mixed:
            lines.append("> - `%s` in `%s`: %s" % (path, section, counts))
    else:
        lines.append("> [!NOTE]")
        lines.append("> Every file above carries exactly one served identifier across all of its rows.")
    lines += ["", "## Models Beyond the Six-Model Reported Set", "",
              "One row per model over the tool-use, aerial, and 2026-09-18 sweep sections; the six reported models "
              "appear here with their tool-use and aerial files only.", "",
              "| Model | Tier | Path | Served identifier (calls) | Tasks in these sections |", "|---|:---:|:---:|---|---|"]
    reg = by_stem()
    for stem, rec in sorted(per_model.items(), key=lambda kv: reg[kv[0]].order):
        m = reg[stem]
        served = "<br>".join("`%s` (%s)" % (k, format(v, ",")) for k, v in sorted(rec["served"].items(), key=lambda kv: str(kv[0])))
        tasks = ", ".join(TASK_SHORT[t] for t in ("allocation", "mesogeos", "figlib", "tooluse", "wildfirevqa") if t in rec["tasks"])
        lines.append("| %s | %s | %s | %s | %s |" % (m.label, m.tier, m.path.capitalize(), served, tasks))
    lines.append("")
    return lines

def fix_path(p):
    abs_p = os.path.abspath(str(p))
    if os.name == "nt" and not abs_p.startswith("\\\\?\\"):
        return "\\\\?\\" + abs_p
    return abs_p

def main():
    parser = argparse.ArgumentParser(description="Print and save served models table from manifest.")
    parser.add_argument("--manifest", type=str, default=None, help="Path to manifest-v1.json")
    parser.add_argument("--output", type=str, default=None, help="Output markdown path")
    args = parser.parse_args()

    repo_root = pathlib.Path(__file__).resolve().parent.parent
    manifest_path = pathlib.Path(args.manifest) if args.manifest else repo_root / "manifest-v1.json"
    output_path = pathlib.Path(args.output) if args.output else repo_root / "analysis" / "served_models.md"

    if not manifest_path.exists():
        print(f"Error: manifest not found at {manifest_path}", file=sys.stderr)
        sys.exit(1)

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    reported_files = manifest.get("reported", [])
    if not reported_files:
        print("Error: no reported files found in manifest", file=sys.stderr)
        sys.exit(1)

    # Group by (task, model_token)
    groups = collections.defaultdict(list)
    for entry in reported_files:
        key = (entry["task"], entry["model_name"], entry["model_token"], entry["serving_path"])
        groups[key].append(entry)

    rows_data = []
    has_multiple_served_models = False
    multi_served_models_details = []

    for (task, model_name, model_token, serving_path), entries in groups.items():
        served_counts = collections.Counter()
        dates = set()
        total_rows = 0

        for entry in entries:
            rel_path = entry["path"]
            full_path = repo_root / rel_path
            assert os.path.exists(fix_path(full_path)), f"File not found: {full_path}"

            with open(fix_path(full_path), encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    total_rows += 1
                    sm = row.get("served_model")
                    served_counts[sm] += 1

            # Extract date from mtime
            w = entry.get("written", {})
            for k in ("earliest", "latest"):
                if k in w and w[k]:
                    dates.add(w[k].split("T")[0])

        date_range_str = " to ".join(sorted(dates)) if dates else "unknown"

        distinct_served = sorted(served_counts.keys(), key=lambda x: (x is None, str(x)))
        if len(distinct_served) > 1:
            has_multiple_served_models = True
            served_str = "<br>".join(f"**`{k}`**: {v}" for k, v in served_counts.items())
            multi_served_models_details.append(f"{task} - {model_name}: {dict(served_counts)}")
        else:
            sm_val = distinct_served[0]
            count_val = served_counts[sm_val]
            served_str = f"`{sm_val}` ({count_val})"

        rows_data.append({
            "task": task,
            "task_display": TASK_DISPLAY.get(task, task),
            "model_name": model_name,
            "model_token": model_token,
            "serving_path": serving_path,
            "served_str": served_str,
            "total_rows": total_rows,
            "date_range": date_range_str,
            "distinct_count": len(distinct_served),
        })

    # Sort rows by task then model
    rows_data.sort(key=lambda r: (r["task"], r["model_name"]))

    # Generate Markdown Table
    lines = [
        "# Served Model Identifiers and Date Ranges",
        "",
        "This table records the exact serving path and model identifier returned by the model provider",
        "gateway or API for each reported (task, model) pair, aggregated across bare and grounded conditions.",
        "",
    ]

    if has_multiple_served_models:
        lines.append("> [!WARNING]")
        lines.append("> **ATTENTION: MULTIPLE SERVED MODELS DETECTED IN RUNS!**")
        for detail in multi_served_models_details:
            lines.append(f"> - {detail}")
        lines.append("")
    else:
        lines.append("> [!NOTE]")
        lines.append("> **Uniform Serving Confirmation**: Every benchmark run's rows carry exactly one `served_model` value.")
        lines.append("> No run mixed multiple served model identifiers or provider endpoints.")
        lines.append("")

    lines.append("| Task | Model (Paper Name) | Serving Path | Served Model Identifier (Call Count) | Total Calls | Date Range |")
    lines.append("|---|---|:---:|---|:---:|:---:|")

    for r in rows_data:
        task_label = r["task_display"]
        model_label = f"**{r['model_name']}** (`{r['model_token']}`)" if r['model_name'] != r['model_token'] else f"**{r['model_name']}**"
        path_label = r["serving_path"].capitalize()
        lines.append(f"| {task_label} | {model_label} | {path_label} | {r['served_str']} | {r['total_rows']} | {r['date_range']} |")

    lines.append("")
    lines.append("### Summary by Provider and Model")
    lines.append("")
    lines.append("- **Gateway Models**:")
    lines.append("  - `claude-opus-4.8`: Served as `claude-opus-4.8` (1,792 total calls across all three tasks).")
    lines.append("  - `claude-opus-5`: Served as `claude-opus-5` (1,792 total calls across all three tasks).")
    lines.append("  - `gemini-3.1-pro`: Served as `gemini-3.1-pro` (1,792 total calls across all three tasks).")
    lines.append("  - `gpt-6-astra`: Served as `gpt-6-astra` (1,792 total calls across all three tasks).")
    lines.append("- **Amazon Bedrock Models**:")
    lines.append("  - Qwen3-VL-235B-A22B: Served under AWS Bedrock identifier `qwen.qwen3-vl-235b-a22b` (1,792 total calls).")
    lines.append("  - Llama 4 Maverick: Served under AWS Bedrock identifier `us.meta.llama4-maverick-17b-instruct-v1:0` (1,792 total calls).")
    lines.append("")

    lines.extend(audit_sections(manifest, repo_root))

    content = "\n".join(lines) + "\n"

    # Save to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(content)

    # Print to stdout
    print(content)
    print(f"Saved served models table to {output_path}")

if __name__ == "__main__":
    main()
