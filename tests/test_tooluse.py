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
