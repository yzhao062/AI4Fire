"""Run the FPA-FOD fire data tool-use task: questions against the 2.3M fire records in SQLite.

    python run_tooluse.py --dry-run
    python run_tooluse.py --fake --models fake --conditions bare tool
    python run_tooluse.py --fake noisy --models fake --conditions bare tool
    python run_tooluse.py --models claude-opus-5 --conditions bare tool

Bare gives the question alone with no tool; the model answers from memory.
Tool provides access to the read-only SQLite database through the query_fpafod tool.
"""
import argparse
import csv
import json
import pathlib
import re
import sqlite3
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

import gw

S = pathlib.Path(__file__).parent
TASK = S / "task-tooluse"
DB = S / "data" / "tooluse" / "FPA_FOD_20221014.sqlite"
VAR_DESC = S / "data" / "tooluse" / "_variable_descriptions.csv"
NAIVE_CSV = TASK / "naive-baseline.csv"
MAX_TOOL_CALLS = 8  # the budget is eight executed queries, whether the model spreads them over eight turns or issues several in one
MAX_OUT = int(__import__("os").environ.get("AI4FIRE_MAX_OUT", 1536))  # the benchmark cap; AI4FIRE_MAX_OUT raises it for a documented variant run

# -------------------------------------------------------------------------------------------------
# Prompts and Schema
# -------------------------------------------------------------------------------------------------
def build_schema_summary() -> str:
    """Compact one-line description of every column in the Fires table from _variable_descriptions.csv."""
    descriptions = {}
    if VAR_DESC.exists():
        with open(VAR_DESC, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("Table") == "Fires":
                    d = row["Description"].strip()
                    d = re.sub(r"\s*See\s+.*$", "", d)
                    var = row["Variable"]
                    if var == "NWCG_REPORTING_AGENCY":
                        d = "Active NWCG Unit Identifier for agency preparing fire report (BIA, BLM, FS, NPS, ST/C&L, etc.)."
                    elif var == "NWCG_GENERAL_CAUSE":
                        d = "Event or circumstance starting fire (Natural, Arson, Debris burning, Equipment, Fireworks, etc.)."
                    elif var == "FIRE_SIZE_CLASS":
                        d = "Size code based on acres (A:<=0.25, B:0.26-9.9, C:10-99.9, D:100-299, E:300-999, F:1000-4999, G:5000+)."
                    elif var == "FIRE_CODE":
                        d = "Cost tracking code used within interagency wildland fire community."
                    elif var == "FIPS_CODE":
                        d = "Five-digit FIPS county code for representation of counties."
                    elif var == "FIPS_NAME":
                        d = "County name from FIPS code based on nominal designation."
                    elif var == "STATE":
                        d = "Two-letter alphabetic code for state in which fire burned or originated."
                    descriptions[var] = d

    col_types = {}
    if DB.exists():
        con = sqlite3.connect(f"file:{DB.resolve().as_posix()}?mode=ro", uri=True)
        cols_info = con.execute("PRAGMA table_info(Fires)").fetchall()
        con.close()
        for cid, name, c_type, notnull, dflt, pk in cols_info:
            col_types[name] = c_type

    schema_lines = []
    for name, c_type in col_types.items():
        desc = descriptions.get(name, "Wildfire record attribute.")
        desc_clean = " ".join(desc.split())
        schema_lines.append(f"- {name} ({c_type}): {desc_clean}")

    return "\n".join(schema_lines)


TASK_SUMMARY = (
    "You are an expert wildfire database assistant analyzing the US FPA-FOD wildfire records (1992-2020) in table 'Fires'. "
    "Answer each question accurately and concisely based on the data."
)

ANSWER_CONTRACT = (
    "Answer Contract:\n"
    "The last line of your final message must strictly be:\n"
    "ANSWER: <value>\n"
    "where <value> is in the requested format (an integer, a number of acres, a category name, a fire name, or a percentage). "
    "If you cannot answer or verify the question, output 'ANSWER: unknown'."
)

SCHEMA_TEXT = build_schema_summary()
SYSTEM_PROMPT = f"{TASK_SUMMARY}\n\nDatabase Schema for table 'Fires':\n{SCHEMA_TEXT}\n\n{ANSWER_CONTRACT}"


def render_messages(item: dict, condition: str) -> list[dict]:
    lines = [item["prompt"]["question"]]
    if item["prompt"].get("options"):
        lines.append(f"Options: {', '.join(repr(o) for o in item['prompt']['options'])}")
    lines.append(f"Format your final answer as: {item['prompt']['answer_format']}.")
    if condition == "tool":
        lines.append("A SQL tool 'query_fpafod' over the SQLite table 'Fires' is available. You must query the database to determine the answer, and your answer must come from the query results.")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(lines)}
    ]


# -------------------------------------------------------------------------------------------------
# SQL Guard and Tool
# -------------------------------------------------------------------------------------------------
def validate_sql(sql: str) -> tuple[bool, str]:
    """Validate that sql is a single SELECT or WITH ... SELECT statement.
    Returns (True, '') on success, or (False, error_message) on rejection.
    """
    if not sql or not sql.strip():
        return False, "Query is empty"
    s = sql.strip()
    n = len(s)
    i = 0
    tokens = []
    seen_semicolon = False
    while i < n:
        c = s[i]
        if c == '-' and i + 1 < n and s[i + 1] == '-':
            i += 2
            while i < n and s[i] not in ('\r', '\n'):
                i += 1
            continue
        if c == '/' and i + 1 < n and s[i + 1] == '*':
            i += 2
            while i + 1 < n and not (s[i] == '*' and s[i + 1] == '/'):
                i += 1
            i += 2
            continue
        if c == "'":
            i += 1
            while i < n:
                if s[i] == "'":
                    if i + 1 < n and s[i + 1] == "'":
                        i += 2
                    else:
                        i += 1
                        break
                else:
                    i += 1
            continue
        if c == '"':
            i += 1
            while i < n:
                if s[i] == '"':
                    if i + 1 < n and s[i + 1] == '"':
                        i += 2
                    else:
                        i += 1
                        break
                else:
                    i += 1
            continue
        if c == ';':
            seen_semicolon = True
            i += 1
            continue
        if seen_semicolon:
            if not c.isspace():
                return False, "Multiple statements are not permitted"
            i += 1
            continue
        if c.isalnum() or c == '_':
            start = i
            while i < n and (s[i].isalnum() or s[i] == '_'):
                i += 1
            tokens.append(s[start:i].upper())
            continue
        i += 1

    if not tokens:
        return False, "Query contains no SQL statements"

    first_word = tokens[0]
    if first_word not in ("SELECT", "WITH"):
        return False, f"Statement must start with SELECT or WITH, got {first_word}"

    forbidden = {
        "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE",
        "ATTACH", "DETACH", "PRAGMA", "VACUUM", "REPLACE", "REINDEX"
    }
    found = [t for t in tokens if t in forbidden]
    if found:
        return False, f"Forbidden keyword(s) detected: {', '.join(found)}"

    if first_word == "WITH" and "SELECT" not in tokens:
        return False, "WITH statement must contain a SELECT query"

    return True, ""



def _add_usage(total, usage):
    """Sum a call's usage into the item total. Reasoning tokens are added only when the call reported them, so a
    row whose calls never carried the field stays without it (the 2026-09-17 tool-arm rows have that shape)."""
    usage = usage or {}
    total["prompt_tokens"] += usage.get("prompt_tokens", 0) or 0
    total["completion_tokens"] += usage.get("completion_tokens", 0) or 0
    det = usage.get("completion_tokens_details") or {}
    if "reasoning_tokens" in det:
        acc = total.setdefault("completion_tokens_details", {"reasoning_tokens": 0})
        acc["reasoning_tokens"] += det.get("reasoning_tokens") or 0
    if usage.get("tool_protocol"):  # Gemma 3's tool_code wire protocol, flagged by gw.py on every call
        total["tool_protocol"] = usage["tool_protocol"]

def query_fpafod(sql: str, db_path: pathlib.Path = DB, timeout_seconds: float = 15.0, return_meta: bool = False):
    """Execute a single SELECT statement on a read-only connection.

    Enforces single SELECT/WITH guards, a 15-second statement timeout,
    and caps results at 50 rows and 4,000 characters.
    """
    t0 = time.perf_counter()
    ok, err = validate_sql(sql)
    if not ok:
        elapsed = time.perf_counter() - t0
        res = f"Query rejected: {err}"
        meta = {"sql": sql, "rows": 0, "chars": len(res), "truncated": False, "error": res, "seconds": round(elapsed, 4)}
        return (res, meta) if return_meta else res

    uri = f"file:{db_path.resolve().as_posix()}?mode=ro"
    try:
        con = sqlite3.connect(uri, uri=True)
        con.set_progress_handler(lambda: 1 if time.perf_counter() - t0 > timeout_seconds else 0, 2000)
        cur = con.execute(sql)
        col_names = [d[0] for d in cur.description] if cur.description else []
        raw_rows = cur.fetchmany(51)
        con.close()
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        res = f"SQLite error: {exc}"
        meta = {"sql": sql, "rows": 0, "chars": len(res), "truncated": False, "error": res, "seconds": round(elapsed, 4)}
        return (res, meta) if return_meta else res

    truncated = False
    if len(raw_rows) > 50:
        truncated = True
        raw_rows = raw_rows[:50]

    header = " | ".join(str(c) for c in col_names)
    lines = [header]
    for r in raw_rows:
        lines.append(" | ".join(str(v) if v is not None else "NULL" for v in r))
    table_text = "\n".join(lines)

    if len(table_text) > 4000:
        truncated = True
        table_text = table_text[:4000] + "\n[Result truncated at 4000 characters]"
    elif truncated:
        table_text += "\n[Result truncated at 50 rows]"

    elapsed = time.perf_counter() - t0
    meta = {
        "sql": sql,
        "rows": len(raw_rows),
        "chars": len(table_text),
        "truncated": truncated,
        "error": None,
        "seconds": round(elapsed, 4)
    }
    return (table_text, meta) if return_meta else table_text


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "query_fpafod",
            "description": "Execute a single SELECT query against the FPA-FOD SQLite database containing wildfire records in table 'Fires'. Returns a text table capped at 50 rows.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "The SQL SELECT statement to execute."
                    }
                },
                "required": ["sql"]
            }
        }
    }
]


# -------------------------------------------------------------------------------------------------
# Parser and Scoring
# -------------------------------------------------------------------------------------------------
def norm_str(s: any) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^A-Z0-9]", " ", str(s).upper())).strip()


def parse_answer(text: str, item: dict) -> tuple[any, bool, str | None]:
    """Parse the assistant's response text into a prediction.
    Returns (prediction, abstained, failure).
    """
    if not text or not str(text).strip():
        return None, False, "parse"

    raw = str(text).strip()
    ans_type = item.get("answer_type", "string")
    options = (item.get("prompt") or {}).get("options")

    cand = None
    matches = re.findall(r'(?im)^\s*ANSWER\s*:\s*(.+)$', raw)
    if not matches:
        matches = re.findall(r'(?i)ANSWER\s*:\s*(.+)', raw)
    if matches:
        cand = matches[-1].strip()

    if cand is not None:
        cand_norm = re.sub(r"[^\w]", "", cand).lower()
        if cand_norm == "unknown":
            return "unknown", True, None
    elif raw.strip().lower() == "unknown" or re.search(r'(?i)\bANSWER\s*:\s*unknown\b', raw):
        return "unknown", True, None

    def extract_number(s: str, prefer_first=True):
        nums = re.findall(r'[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|[-+]?\d+(?:\.\d+)?', s)
        if not nums:
            return None
        pick = nums[0] if prefer_first else nums[-1]
        return pick.replace(",", "")

    def last_line(s: str) -> str:
        lines = [line.strip() for line in s.splitlines() if line.strip()]
        return lines[-1] if lines else ""

    try:
        if ans_type == "integer":
            val_str = extract_number(cand, prefer_first=True) if cand is not None else extract_number(raw, prefer_first=False)
            if val_str is not None:
                return int(round(float(val_str))), False, None
            return None, False, "parse"

        elif ans_type == "acres":
            val_str = extract_number(cand, prefer_first=True) if cand is not None else extract_number(raw, prefer_first=False)
            if val_str is not None:
                return float(val_str), False, None
            return None, False, "parse"

        elif ans_type == "percent":
            target = cand.replace("%", " ") if cand is not None else raw.replace("%", " ")
            val_str = extract_number(target, prefer_first=(cand is not None))
            if val_str is not None:
                return float(val_str), False, None
            return None, False, "parse"

        elif ans_type == "categorical":
            target_str = cand if cand is not None else last_line(raw)
            cleaned = target_str.strip(" \t\n\r'\"`.:;,*()[]{}")
            if not cleaned:
                return None, False, "parse"
            if cleaned.lower() == "unknown":
                return "unknown", True, None
            if options:
                for opt in options:
                    if cleaned.lower() == opt.lower():
                        return opt, False, None
                for opt in sorted(options, key=len, reverse=True):
                    pattern = r'(?i)(?<!\w)' + re.escape(opt) + r'(?!\w)'
                    if re.search(pattern, target_str):
                        return opt, False, None
            return cleaned, False, None

        elif ans_type == "string":
            target_str = cand if cand is not None else last_line(raw)
            cleaned = target_str.strip(" \t\n\r'\"`.:;,*()[]{}")
            if not cleaned:
                return None, False, "parse"
            if cleaned.lower() == "unknown":
                return "unknown", True, None
            return cleaned, False, None

    except Exception:
        return None, False, "parse"

    return None, False, "parse"


def score_item(item: dict, pred: any) -> bool:
    """Score one prediction under the item's own tolerance rule."""
    if pred is None:
        return False
    rule = item["tolerance"]["rule"]
    gold = item["answer"]
    try:
        if rule == "exact_int":
            return int(round(float(pred))) == int(gold)
        if rule == "relative":
            rel_tol = item["tolerance"].get("rel_tol", 0.005)
            abs_floor = item["tolerance"].get("abs_floor", 1.0)
            return abs(float(pred) - float(gold)) <= max(rel_tol * abs(float(gold)), abs_floor)
        if rule == "absolute":
            abs_tol = item["tolerance"].get("abs_tol", 0.1)
            return abs(float(pred) - float(gold)) <= abs_tol
    except (TypeError, ValueError):
        return False
    if rule == "categorical_exact":
        return str(pred).strip().lower() == str(gold).strip().lower()
    if rule == "string_normalized":
        return norm_str(pred) == norm_str(gold)
    return False


# -------------------------------------------------------------------------------------------------
# Fake Backend Helpers
# -------------------------------------------------------------------------------------------------
def substitute_params(query: str, params: list) -> str:
    """Substitute ? positional parameters into reference query."""
    parts = query.split("?")
    out = []
    for i, p in enumerate(params):
        out.append(parts[i])
        if isinstance(p, int):
            out.append(str(p))
        else:
            s = str(p).replace("'", "''")
            out.append(f"'{s}'")
    out.append(parts[-1])
    return "".join(out)


def perturb_numeric(val: any, item: dict) -> any:
    """Perturb numeric answers by twice the tolerance rule."""
    rule = item["tolerance"]["rule"]
    if rule == "exact_int":
        return int(round(float(val))) + 2
    elif rule == "relative":
        fval = float(val)
        tol = max(item["tolerance"]["rel_tol"] * abs(fval), item["tolerance"]["abs_floor"])
        return fval + 2.5 * tol
    elif rule == "absolute":
        fval = float(val)
        return fval + 2.5 * item["tolerance"]["abs_tol"]
    return val


# -------------------------------------------------------------------------------------------------
# Runner Core
# -------------------------------------------------------------------------------------------------
def load_naive_baseline() -> tuple[dict, float]:
    by_fam = {}
    if NAIVE_CSV.exists():
        with open(NAIVE_CSV, mode="r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                by_fam[row["family"]] = float(row["best_constant_acc"])
    overall = round(statistics.fmean(by_fam.values()), 3) if by_fam else 0.0
    return by_fam, overall


def run_item(item: dict, condition: str, model: str, key: str = None, fake: str = None) -> dict:
    base = {
        "item_id": item["item_id"],
        "family": item["family"],
        "tier": item["tier"],
        "answer_type": item["answer_type"]
    }

    if fake:
        if condition == "bare":
            raw = "ANSWER: unknown"
            usage = {"prompt_tokens": 100, "completion_tokens": 10}
            tool_calls = []
            err = None
            served = "fake"
        else:
            # Fake tool arm: issue substituted reference query as tool call
            sql = substitute_params(item["reference_query"], item["reference_params"])
            table_text, meta = query_fpafod(sql, return_meta=True)
            tool_calls = [meta]
            lines = [l.strip() for l in table_text.splitlines() if l.strip()]
            val = lines[1].split(" | ")[0].strip() if len(lines) > 1 else ""
            if fake == "noisy" and item["answer_type"] in ("integer", "acres", "percent"):
                val = perturb_numeric(val, item)
            raw = f"ANSWER: {val}"
            usage = {"prompt_tokens": 250, "completion_tokens": 25}
            err = None
            served = "fake"

        pred, abstained, fail = parse_answer(raw, item)
        corr = score_item(item, pred) if pred is not None and not abstained else False

        return dict(
            base,
            raw=raw,
            prediction=pred,
            correct=corr,
            abstained=abstained,
            failure=fail,
            tool_calls=tool_calls,
            n_tool_calls=len(tool_calls),
            usage=usage,
            served_model=served,
            error=err
        )

    # Real model execution
    msgs = render_messages(item, condition)
    if condition == "bare":
        try:
            raw, usage, served = gw.call(key, model, msgs, max_tokens=MAX_OUT)
            usage["max_out"] = MAX_OUT  # the cap this run used, as the tool arm and the other runners record it
            pred, abstained, fail = parse_answer(raw, item)
            corr = score_item(item, pred) if pred is not None and not abstained else False
            return dict(
                base,
                raw=raw,
                prediction=pred,
                correct=corr,
                abstained=abstained,
                failure=fail,
                tool_calls=[],
                n_tool_calls=0,
                usage=usage,
                served_model=served,
                error=None
            )
        except Exception as exc:
            return dict(
                base,
                raw="",
                prediction=None,
                correct=None,
                abstained=False,
                failure="error",
                tool_calls=[],
                n_tool_calls=0,
                usage={"prompt_tokens": 0, "completion_tokens": 0},
                served_model=model,
                error=str(exc)[:300]
            )

    # Real tool condition
    tool_calls = []
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "max_out": MAX_OUT}
    served = model
    raw = ""
    err = None

    for step in range(MAX_TOOL_CALLS):
        try:
            msg, usage, served = gw.call_tools(key, model, msgs, tools=TOOLS, max_tokens=MAX_OUT)
        except Exception as exc:
            err = str(exc)[:300]
            break

        _add_usage(total_usage, usage)

        tcs = msg.get("tool_calls")
        if not tcs:
            raw = msg.get("content", "")
            break

        msgs.append(msg)
        budget_spent = False
        for tc in tcs:
            if len(tool_calls) >= MAX_TOOL_CALLS:
                # a turn carrying several calls can reach the budget mid-turn; the rest of that turn is
                # refused rather than executed, so no run exceeds MAX_TOOL_CALLS queries
                budget_spent = True
                msgs.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", ""),
                    "content": "Tool call budget exhausted; answer from the results you already have."
                })
                continue
            fn = tc.get("function", {})
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {"sql": args}
            sql_arg = args.get("sql", "") if isinstance(args, dict) else str(args)
            if not isinstance(sql_arg, str):
                # a call whose sql argument is not a string (Gemma 3 12B passed a dict through the tool_code
                # protocol) is a malformed query: the guard rejects its serialized form and the model reads why
                sql_arg = json.dumps(sql_arg)
            res_text, meta = query_fpafod(sql_arg, return_meta=True)
            tool_calls.append(meta)
            msgs.append({
                "role": "tool",
                "tool_call_id": tc.get("id", ""),
                "content": res_text
            })

        if step == MAX_TOOL_CALLS - 1 or budget_spent or len(tool_calls) >= MAX_TOOL_CALLS:
            # Budget spent; the final call carries no tools
            msgs.append({
                "role": "user",
                "content": "You have reached the maximum number of tool calls. Based on your findings, provide your final answer now. End with ANSWER: <value>."
            })
            try:
                final_msg, final_usage, served = gw.call_tools(key, model, msgs, tools=None, max_tokens=MAX_OUT)
                _add_usage(total_usage, final_usage)
                raw = final_msg.get("content", "")
            except Exception as exc:
                err = str(exc)[:300]
            break

    if err:
        return dict(
            base,
            raw=raw,
            prediction=None,
            correct=None,
            abstained=False,
            failure="error",
            tool_calls=tool_calls,
            n_tool_calls=len(tool_calls),
            usage=total_usage,
            served_model=served,
            error=err
        )

    pred, abstained, fail = parse_answer(raw, item)
    if abstained:
        corr = False
        fail = None
    elif pred is not None:
        corr = score_item(item, pred)
        fail = None
    else:
        corr = False
        if any(tc.get("error") for tc in tool_calls) and (len(tool_calls) >= 8 or all(tc.get("error") for tc in tool_calls)):
            fail = "tool"
        else:
            fail = "parse"

    return dict(
        base,
        raw=raw,
        prediction=pred,
        correct=corr,
        abstained=abstained,
        failure=fail,
        tool_calls=tool_calls,
        n_tool_calls=len(tool_calls),
        usage=total_usage,
        served_model=served,
        error=None
    )


def compute_summary(rows: list[dict], run_key: str) -> dict:
    naive_by_fam, naive_overall = load_naive_baseline()
    n_items = len(rows)
    acc = round(sum(1 for r in rows if r.get("correct") is True) / n_items, 4) if n_items else 0.0

    tiers = sorted(set(r["tier"] for r in rows))
    acc_by_tier = {}
    for t in tiers:
        t_rows = [r for r in rows if r.get("tier") == t]
        acc_by_tier[t] = round(sum(1 for r in t_rows if r.get("correct") is True) / len(t_rows), 4) if t_rows else 0.0

    families = sorted(set(r["family"] for r in rows))
    acc_by_fam = {}
    for f in families:
        f_rows = [r for r in rows if r.get("family") == f]
        acc_by_fam[f] = round(sum(1 for r in f_rows if r.get("correct") is True) / len(f_rows), 4) if f_rows else 0.0

    tool_counts = [r.get("n_tool_calls", 0) for r in rows]
    mean_tc = round(statistics.fmean(tool_counts), 3) if tool_counts else 0.0
    max_tc = max(tool_counts, default=0)

    return {
        "run": run_key,
        "items": n_items,
        "accuracy": acc,
        "accuracy_by_tier": acc_by_tier,
        "accuracy_by_family": acc_by_fam,
        "naive_baseline_by_family": naive_by_fam,
        "naive_baseline_overall": naive_overall,
        "parse_failures": sum(1 for r in rows if r.get("failure") == "parse"),
        "tool_failures": sum(1 for r in rows if r.get("failure") == "tool"),
        "errors": sum(1 for r in rows if r.get("failure") == "error" or r.get("error")),
        "abstentions": sum(1 for r in rows if r.get("abstained") is True),
        "mean_tool_calls": mean_tc,
        "max_tool_calls": max_tc,
        "tokens_in": sum((r.get("usage") or {}).get("prompt_tokens", 0) for r in rows),
        "tokens_out": sum((r.get("usage") or {}).get("completion_tokens", 0) for r in rows),
    }


def update_scores_json(summaries: list[dict]):
    path = TASK / "scores.json"
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    fresh = {s["run"] for s in summaries}
    combined = [s for s in old if s.get("run") not in fresh] + summaries
    path.write_text(json.dumps(combined, indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Run fire data tool-use task on FPA-FOD.")
    ap.add_argument("--models", nargs="*", default=["claude-opus-5"],
                    help="gateway model names or bedrock:<id>")
    ap.add_argument("--conditions", nargs="*", default=["bare", "tool"], choices=["bare", "tool"],
                    help="bare: no tools; tool: SQL query tool")
    ap.add_argument("--workers", type=int, default=6,
                    help="concurrency worker threads")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the system and user prompt of one item per arm and exit")
    ap.add_argument("--limit", type=int, default=None,
                    help="run first N items only")
    ap.add_argument("--retry-errors", action="store_true",
                    help="re-run only the rows whose error field is set in the existing response file; keep the rest")
    ap.add_argument("--fake", nargs="?", const="clean", default=None, choices=["clean", "noisy"],
                    help="run against fake backend (clean or noisy)")
    ap.add_argument("--items", type=pathlib.Path, default=None,
                    help="path to items JSONL file (defaults to task-tooluse/items.jsonl)")
    ap.add_argument("--variant", default=None,
                    help="variant identifier (e.g. p1); if set, uses task-tooluse/items-<variant>.jsonl and writes responses-...-<variant>.jsonl")
    args = ap.parse_args()

    variant_suffix = ""
    if args.items:
        p = pathlib.Path(args.items)
        if p.is_absolute() and p.exists():
            items_path = p
        elif (S / p).exists():
            items_path = S / p
        elif (TASK / p).exists():
            items_path = TASK / p
        elif (TASK / p.name).exists():
            items_path = TASK / p.name
        else:
            items_path = p.resolve()
        m = re.search(r"-(p\d+)$", items_path.stem)
        if m:
            variant_suffix = f"-{m.group(1)}"
        elif args.variant:
            variant_suffix = f"-{args.variant}"
    elif args.variant:
        variant_suffix = f"-{args.variant}"
        items_path = TASK / f"items-{args.variant}.jsonl"
    else:
        items_path = TASK / "items.jsonl"

    allitems = [json.loads(line) for line in items_path.read_text(encoding="utf-8").splitlines() if line]
    items = allitems[:args.limit] if args.limit is not None else allitems

    if args.dry_run:
        it = items[0]
        print("=== SYSTEM PROMPT ===")
        print(SYSTEM_PROMPT)
        print("\n=== BARE USER PROMPT ===")
        print(render_messages(it, "bare")[1]["content"])
        print("\n=== TOOL USER PROMPT ===")
        print(render_messages(it, "tool")[1]["content"])
        print("\n=== ITEM METADATA ===")
        print(f"Item ID: {it['item_id']} | Family: {it['family']} | Tier: {it['tier']} | Answer: {it['answer']}")
        return

    # If --fake was specified and models was left as default, set models to ['fake']
    models = args.models
    if args.fake and models == ["claude-opus-5"]:
        models = ["fake"]

    key = None
    if not args.fake:
        key = gw.load_key()

    summaries = []
    for model in models:
        for cond in args.conditions:
            print(f"\nRunning {model} on {cond} condition ({len(items)} items, workers={args.workers}, fake={args.fake})...")

            def process_one(it):
                return run_item(it, cond, model, key=key, fake=args.fake)

            safe = re.sub(r"[^A-Za-z0-9._-]", "_", model)
            # A fake run is a harness test: it writes under task-tooluse/fake/ and never touches scores.json,
            # so the reported files and the score record hold model runs only.
            out_dir = TASK / "fake" if args.fake else TASK
            out_dir.mkdir(exist_ok=True)
            out_file = out_dir / f"responses-{safe}-{cond}{variant_suffix}.jsonl"

            if args.retry_errors:
                # Keep every row that returned an answer; re-run only the calls that raised.
                kept = {r["item_id"]: r for r in (json.loads(l) for l in out_file.read_text(encoding="utf-8").splitlines() if l.strip())}
                todo = [it for it in items if kept.get(it["item_id"], {}).get("error")]
                print(f"{model}/{cond}: retrying {len(todo)} errored rows of {len(kept)}")
                with ThreadPoolExecutor(max_workers=args.workers) as ex:
                    for r in ex.map(process_one, todo):
                        kept[r["item_id"]] = r
                # keep every recorded row, not only the (possibly --limit) retry slice, in item order
                rows = [kept[it["item_id"]] for it in allitems if it["item_id"] in kept]
            else:
                with ThreadPoolExecutor(max_workers=args.workers) as ex:
                    rows = list(ex.map(process_one, items))
            out_file.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

            run_key = f"{model}/{cond}{variant_suffix}"
            s = compute_summary(rows, run_key)
            summaries.append(s)
            print(json.dumps(s, indent=2))
            if not args.fake:
                update_scores_json(summaries)
            print(f"Saved responses to {out_file}")

    if args.fake:
        print(f"\nCompleted {len(summaries)} fake run(s) under {TASK / 'fake'}; scores.json untouched.")
    else:
        print(f"\nCompleted {len(summaries)} run(s). Updated {TASK / 'scores.json'}.")


if __name__ == "__main__":
    main()
