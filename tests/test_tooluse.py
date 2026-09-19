"""Tests for the FPA-FOD fire data tool-use task (run_tooluse.py)."""
import json
import pathlib
import subprocess
import sys

import pytest

import run_tooluse as rtu

S = pathlib.Path(__file__).parent.parent
TASK = S / "task-tooluse"


def test_tolerance_rules():
    # 1. exact_int
    it_int = {"answer": 480, "tolerance": {"rule": "exact_int"}}
    assert rtu.score_item(it_int, 480) is True      # inside
    assert rtu.score_item(it_int, 480.0) is True    # on the edge
    assert rtu.score_item(it_int, 481) is False     # outside
    assert rtu.score_item(it_int, 479) is False     # outside

    # 2. relative
    it_rel = {"answer": 1000.0, "tolerance": {"rule": "relative", "rel_tol": 0.005, "abs_floor": 1.0}}
    # tolerance = max(0.005 * 1000, 1.0) = 5.0
    assert rtu.score_item(it_rel, 1002.0) is True   # inside (diff 2.0 <= 5.0)
    assert rtu.score_item(it_rel, 1005.0) is True   # on the edge (diff 5.0 <= 5.0)
    assert rtu.score_item(it_rel, 1006.0) is False  # outside (diff 6.0 > 5.0)

    # 2b. relative with abs_floor dominating
    it_floor = {"answer": 10.0, "tolerance": {"rule": "relative", "rel_tol": 0.005, "abs_floor": 1.0}}
    # tolerance = max(0.05, 1.0) = 1.0
    assert rtu.score_item(it_floor, 10.5) is True   # inside (diff 0.5 <= 1.0)
    assert rtu.score_item(it_floor, 11.0) is True   # on the edge (diff 1.0 <= 1.0)
    assert rtu.score_item(it_floor, 11.5) is False  # outside (diff 1.5 > 1.0)

    # 3. absolute
    it_abs = {"answer": 67.87, "tolerance": {"rule": "absolute", "abs_tol": 0.1}}
    assert rtu.score_item(it_abs, 67.90) is True    # inside (diff 0.03 <= 0.1)
    assert rtu.score_item(it_abs, 67.97) is True    # on the edge (diff 0.10 <= 0.1)
    assert rtu.score_item(it_abs, 68.00) is False   # outside (diff 0.13 > 0.1)

    # 4. categorical_exact
    it_cat = {"answer": "Natural", "tolerance": {"rule": "categorical_exact"}}
    assert rtu.score_item(it_cat, "Natural") is True        # inside
    assert rtu.score_item(it_cat, "  natural  ") is True    # on the edge (case / whitespace)
    assert rtu.score_item(it_cat, "Arson") is False         # outside

    # 5. string_normalized
    it_str = {"answer": "CARLA LAKE", "tolerance": {"rule": "string_normalized"}}
    assert rtu.score_item(it_str, "Carla Lake") is True       # inside
    assert rtu.score_item(it_str, " carla-lake! ") is True    # on the edge (punctuation / spacing)
    assert rtu.score_item(it_str, "CARLA RIVER") is False     # outside


def test_sql_guard():
    # Rejects INSERT
    ok, err = rtu.validate_sql("INSERT INTO Fires (STATE) VALUES ('CA')")
    assert not ok and "INSERT" in err

    # Rejects PRAGMA
    ok, err = rtu.validate_sql("PRAGMA table_info(Fires)")
    assert not ok and "PRAGMA" in err

    # Rejects two statements
    ok, err = rtu.validate_sql("SELECT 1; SELECT 2")
    assert not ok and "Multiple statements" in err

    ok, err = rtu.validate_sql("SELECT 1; DROP TABLE Fires;")
    assert not ok and "Multiple statements" in err

    # Accepts WITH
    ok, err = rtu.validate_sql("WITH t AS (SELECT 1 AS val) SELECT * FROM t")
    assert ok and err == ""

    # Accepts single SELECT
    ok, err = rtu.validate_sql("SELECT count(*) FROM Fires WHERE STATE = 'CA'")
    assert ok and err == ""

    # Accepts semicolon inside string literal
    ok, err = rtu.validate_sql("SELECT * FROM Fires WHERE FIRE_NAME = 'FOO;BAR'")
    assert ok and err == ""


def test_sql_guard_rejects_non_string_argument():
    # the runner serializes a dict or list sql argument before the guard; the guard then rejects it as text
    for bad in ({"query": "SELECT 1"}, ["SELECT 1"], 42):
        ok, err = rtu.validate_sql(json.dumps(bad))
        assert ok is False and err


def test_usage_accumulator_keeps_tool_protocol():
    total = {"prompt_tokens": 0, "completion_tokens": 0}
    rtu._add_usage(total, {"prompt_tokens": 5, "completion_tokens": 7})
    assert "tool_protocol" not in total
    rtu._add_usage(total, {"prompt_tokens": 1, "completion_tokens": 2, "tool_protocol": "tool_code"})
    assert total == {"prompt_tokens": 6, "completion_tokens": 9, "tool_protocol": "tool_code"}


def test_parser_ten_strings():
    # 1. Integer with ANSWER: prefix
    p, a, f = rtu.parse_answer("ANSWER: 480", {"answer_type": "integer"})
    assert p == 480 and not a and f is None

    # 2. Integer with comma, no ANSWER prefix
    p, a, f = rtu.parse_answer("The total number of fires is 1,234 in that year.", {"answer_type": "integer"})
    assert p == 1234 and not a and f is None

    # 3. Acres with comma and unit
    p, a, f = rtu.parse_answer("ANSWER: 601,674.5 acres", {"answer_type": "acres"})
    assert abs(p - 601674.5) < 1e-5 and not a and f is None

    # 4. Percent with % sign
    p, a, f = rtu.parse_answer("ANSWER: 67.87%", {"answer_type": "percent"})
    assert abs(p - 67.87) < 1e-5 and not a and f is None

    # 5. Categorical with punctuation and options
    p, a, f = rtu.parse_answer("ANSWER: 'Natural'.", {"answer_type": "categorical", "prompt": {"options": ["Natural", "Arson"]}})
    assert p == "Natural" and not a and f is None

    # 6. String with quotes
    p, a, f = rtu.parse_answer('ANSWER: "CARLA LAKE"', {"answer_type": "string"})
    assert p == "CARLA LAKE" and not a and f is None

    # 7. No ANSWER line, last number fallback
    p, a, f = rtu.parse_answer("After checking the table, the result is 42.", {"answer_type": "integer"})
    assert p == 42 and not a and f is None

    # 8. Explicit abstention
    p, a, f = rtu.parse_answer("ANSWER: unknown", {"answer_type": "integer"})
    assert p == "unknown" and a is True and f is None

    # 9. Parse failure (no answer or number present)
    p, a, f = rtu.parse_answer("I was unable to find any fires in this region.", {"answer_type": "integer"})
    assert p is None and not a and f == "parse"

    # 10. Multi-line response with ANSWER on the last line
    p, a, f = rtu.parse_answer("SQL executed successfully.\nResult found.\nANSWER: 100", {"answer_type": "integer"})
    assert p == 100 and not a and f is None


def test_fake_clean_run():
    # Run fake clean on both bare and tool conditions
    cmd = [sys.executable, "run_tooluse.py", "--fake", "--models", "fake", "--conditions", "bare", "tool"]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(S))
    assert res.returncode == 0, f"run_tooluse failed:\n{res.stderr}"

    # Verify tool arm: 156 / 156 correct
    tool_file = TASK / "fake" / "responses-fake-tool.jsonl"
    assert tool_file.exists()
    tool_rows = [json.loads(line) for line in tool_file.read_text(encoding="utf-8").splitlines() if line]
    assert len(tool_rows) == 156
    assert sum(1 for r in tool_rows if r["correct"] is True) == 156
    assert sum(1 for r in tool_rows if r.get("failure") is not None) == 0
    assert sum(1 for r in tool_rows if r.get("abstained") is True) == 0

    # Verify bare arm: 0 / 156 correct, 156 abstentions
    bare_file = TASK / "fake" / "responses-fake-bare.jsonl"
    assert bare_file.exists()
    bare_rows = [json.loads(line) for line in bare_file.read_text(encoding="utf-8").splitlines() if line]
    assert len(bare_rows) == 156
    assert sum(1 for r in bare_rows if r["correct"] is True) == 0
    assert sum(1 for r in bare_rows if r.get("abstained") is True) == 156


def test_fake_noisy_run():
    # Run fake noisy on both bare and tool conditions
    cmd = [sys.executable, "run_tooluse.py", "--fake", "noisy", "--models", "fake", "--conditions", "bare", "tool"]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(S))
    assert res.returncode == 0, f"run_tooluse failed:\n{res.stderr}"

    tool_file = TASK / "fake" / "responses-fake-tool.jsonl"
    assert tool_file.exists()
    tool_rows = [json.loads(line) for line in tool_file.read_text(encoding="utf-8").splitlines() if line]
    assert len(tool_rows) == 156

    # 9 numeric families must score 0 correct
    numeric_families = {
        "acres_state_year", "count_cause_state_year", "count_county_year",
        "count_season_window", "count_size_class_span", "count_state_year",
        "human_share_state_year", "largest_fire_size_span", "peak_year_state"
    }
    for fam in numeric_families:
        fam_rows = [r for r in tool_rows if r["family"] == fam]
        assert len(fam_rows) == 13
        assert sum(1 for r in fam_rows if r["correct"] is True) == 0, f"Expected 0 correct for numeric family {fam}"

    # 3 non-numeric families score 13 / 13 correct
    non_numeric_families = {"largest_fire_name", "top_cause_state_year", "top_state_for_cause"}
    for fam in non_numeric_families:
        fam_rows = [r for r in tool_rows if r["family"] == fam]
        assert len(fam_rows) == 13
        assert sum(1 for r in fam_rows if r["correct"] is True) == 13, f"Expected 13 correct for non-numeric family {fam}"


def test_run_item_serializes_dict_sql_argument(monkeypatch):
    """Gemma 3 12B passed a dictionary as the sql argument; run_item must serialize it for the guard.

    Before the fix of 2026-09-18 the guard raised AttributeError on the dictionary and the run aborted.
    """
    item = {
        "item_id": "tooluse-count_state_year-01",
        "family": "count_state_year",
        "tier": "single lookup",
        "answer_type": "integer",
        "answer": 216,
        "tolerance": {"rule": "exact_int"},
        "prompt": {"question": "How many fires burned in AK in 1992?", "answer_format": "an integer"},
    }
    calls = []

    def fake_call_tools(key, model, msgs, tools=None, max_tokens=None):
        calls.append(msgs)
        if len(calls) == 1:
            return ({"role": "assistant", "content": "",
                     "tool_calls": [{"id": "c1", "function": {
                         "name": "query_fpafod",
                         "arguments": json.dumps({"sql": {"sql": "SELECT count(*) FROM Fires"}})}}]},
                    {"prompt_tokens": 10, "completion_tokens": 3}, model)
        return ({"role": "assistant", "content": "ANSWER: 216"},
                {"prompt_tokens": 12, "completion_tokens": 4}, model)

    monkeypatch.setattr(rtu.gw, "call_tools", fake_call_tools)
    row = rtu.run_item(item, "tool", "bedrock:google.gemma-3-12b-it", key="unused")

    assert row["n_tool_calls"] == 1
    meta = row["tool_calls"][0]
    assert meta["sql"] == '{"sql": "SELECT count(*) FROM Fires"}'
    assert meta["error"].startswith("Query rejected:")
    assert row["error"] is None
    assert row["prediction"] == 216


def test_tool_budget_counts_calls_not_turns(monkeypatch):
    """A model issuing several calls per turn must still execute at most MAX_TOOL_CALLS queries.

    The loop grants MAX_TOOL_CALLS model turns, so before this guard a model with parallel calls could run
    more queries than the budget. No recorded row exceeded eight, so the guard changes no recorded score.
    """
    item = {
        "item_id": "tooluse-count_state_year-02",
        "family": "count_state_year",
        "tier": "single lookup",
        "answer_type": "integer",
        "answer": 216,
        "tolerance": {"rule": "exact_int"},
        "prompt": {"question": "How many fires burned in AK in 1992?", "answer_format": "an integer"},
    }
    toolless_requests = []

    def fake_call_tools(key, model, msgs, tools=None, max_tokens=None):
        if tools is None:
            toolless_requests.append(list(msgs))
            return ({"role": "assistant", "content": "ANSWER: 216"},
                    {"prompt_tokens": 5, "completion_tokens": 2}, model)
        calls = [{"id": f"c{i}", "function": {"name": "query_fpafod",
                                              "arguments": json.dumps({"sql": "SELECT count(*) FROM Fires"})}}
                 for i in range(3)]
        return ({"role": "assistant", "content": "", "tool_calls": calls},
                {"prompt_tokens": 10, "completion_tokens": 3}, model)

    monkeypatch.setattr(rtu.gw, "call_tools", fake_call_tools)
    row = rtu.run_item(item, "tool", "bedrock:test.parallel-caller", key="unused")

    assert row["n_tool_calls"] == rtu.MAX_TOOL_CALLS
    assert len(toolless_requests) == 1
    assert row["prediction"] == 216


def test_retry_with_limit_keeps_every_recorded_row():
    """--retry-errors with --limit must not rewrite the file with the limited slice.

    Before the fix, rows were rebuilt from the limited item list, so a ten-item retry over a 156-row file
    discarded 146 recorded responses even when it made no call.
    """
    base = [sys.executable, "run_tooluse.py", "--fake", "--models", "fake", "--conditions", "tool"]
    first = subprocess.run(base, capture_output=True, text=True, cwd=str(S))
    assert first.returncode == 0, f"first run failed:\n{first.stderr}"

    out_file = TASK / "fake" / "responses-fake-tool.jsonl"
    before = [json.loads(line) for line in out_file.read_text(encoding="utf-8").splitlines() if line]
    assert len(before) == 156

    retry = subprocess.run(base + ["--retry-errors", "--limit", "10"], capture_output=True, text=True, cwd=str(S))
    assert retry.returncode == 0, f"retry failed:\n{retry.stderr}"

    after = [json.loads(line) for line in out_file.read_text(encoding="utf-8").splitlines() if line]
    assert len(after) == 156
    assert [r["item_id"] for r in after] == [r["item_id"] for r in before]
