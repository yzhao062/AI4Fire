"""Verify the paraphrase item set task-tooluse/items-p1.jsonl mechanically.

Checks:
1. 156 items, item_id set and ordering identical to items.jsonl.
2. Every field except prompt byte-identical to the original, per item.
3. Every prompt differs from the original prompt.
4. Token-level overlap distribution (Jaccard similarity and token recall).
5. No paraphrase contains a database column or table name that its original lacks.
"""
import json
import pathlib
import re
import sqlite3
import sys
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
TASK = ROOT / "task-tooluse"
ORIG_PATH = TASK / "items.jsonl"
P1_PATH = TASK / "items-p1.jsonl"

# Schema discovery: look for sqlite database, fallback to known schema from build_items_tooluse / run_tooluse
DB_PATHS = [
    ROOT / "data" / "tooluse" / "FPA_FOD_20221014.sqlite",
]

FALLBACK_COLUMNS = [
    'OBJECTID', 'Shape', 'FOD_ID', 'FPA_ID', 'SOURCE_SYSTEM_TYPE', 'SOURCE_SYSTEM',
    'NWCG_REPORTING_AGENCY', 'NWCG_REPORTING_UNIT_ID', 'NWCG_REPORTING_UNIT_NAME',
    'SOURCE_REPORTING_UNIT', 'SOURCE_REPORTING_UNIT_NAME', 'LOCAL_FIRE_REPORT_ID',
    'LOCAL_INCIDENT_ID', 'FIRE_CODE', 'FIRE_NAME', 'ICS_209_PLUS_INCIDENT_JOIN_ID',
    'ICS_209_PLUS_COMPLEX_JOIN_ID', 'MTBS_ID', 'MTBS_FIRE_NAME', 'COMPLEX_NAME',
    'FIRE_YEAR', 'DISCOVERY_DATE', 'DISCOVERY_DOY', 'DISCOVERY_TIME',
    'NWCG_CAUSE_CLASSIFICATION', 'NWCG_GENERAL_CAUSE', 'NWCG_CAUSE_AGE_CATEGORY',
    'CONT_DATE', 'CONT_DOY', 'CONT_TIME', 'FIRE_SIZE', 'FIRE_SIZE_CLASS',
    'LATITUDE', 'LONGITUDE', 'OWNER_DESCR', 'STATE', 'COUNTY', 'FIPS_CODE', 'FIPS_NAME'
]
FALLBACK_TABLES = ['Fires']


def get_schema_names():
    """Retrieve all table and column names from the real database or fallback."""
    for db in DB_PATHS:
        if db.exists():
            con = sqlite3.connect(f"file:{db.resolve().as_posix()}?mode=ro", uri=True)
            tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
            cols = []
            if "Fires" in tables:
                cols = [r[1] for r in con.execute("PRAGMA table_info(Fires)").fetchall()]
            con.close()
            return sorted(set(tables + cols))
    return sorted(set(FALLBACK_TABLES + FALLBACK_COLUMNS))


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


def find_schema_entities(text, schema_names):
    """Find schema names that appear as distinct tokens/identifiers in the text."""
    found = set()
    for name in schema_names:
        # Match as an exact identifier token (case-sensitive)
        if re.search(r"\b" + re.escape(name) + r"\b", text):
            found.add(name)
    return found


def main():
    print("Checking paraphrase items against original...")
    assert ORIG_PATH.exists(), f"Original items file missing: {ORIG_PATH}"
    assert P1_PATH.exists(), f"Paraphrase items file missing: {P1_PATH}"

    orig_lines = [l.strip() for l in ORIG_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    p1_lines = [l.strip() for l in P1_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]

    # 1. Item count and item_id set
    assert len(orig_lines) == 156, f"Expected 156 original items, got {len(orig_lines)}"
    assert len(p1_lines) == 156, f"Expected 156 paraphrase items, got {len(p1_lines)}"

    orig_items = [json.loads(l) for l in orig_lines]
    p1_items = [json.loads(l) for l in p1_lines]

    orig_ids = [it["item_id"] for it in orig_items]
    p1_ids = [it["item_id"] for it in p1_items]
    assert p1_ids == orig_ids, "item_id sequence does not match original exactly"
    print(f"PASS [1/5]: 156 items present, item_id sequence identical to original.")

    # 2. Every field except prompt byte-identical
    non_prompt_fields = [k for k in orig_items[0].keys() if k != "prompt"]
    for i, (orig, p1) in enumerate(zip(orig_items, p1_items)):
        iid = orig["item_id"]
        assert set(p1.keys()) == set(orig.keys()), f"{iid}: key sets differ"
        for field in non_prompt_fields:
            orig_val_json = json.dumps(orig[field], sort_keys=True)
            p1_val_json = json.dumps(p1[field], sort_keys=True)
            assert orig_val_json == p1_val_json, f"{iid}: field '{field}' differs:\n  orig: {orig_val_json}\n  p1:   {p1_val_json}"
        # prompt subfields: answer_format and options must match
        assert p1["prompt"].get("answer_format") == orig["prompt"].get("answer_format"), f"{iid}: prompt.answer_format differs"
        assert p1["prompt"].get("options") == orig["prompt"].get("options"), f"{iid}: prompt.options differs"
    print(f"PASS [2/5]: Every field except prompt (including answer & reference_query) is byte-identical.")

    # 3. Every prompt differs from original
    for i, (orig, p1) in enumerate(zip(orig_items, p1_items)):
        iid = orig["item_id"]
        orig_q = orig["prompt"]["question"]
        p1_q = p1["prompt"]["question"]
        assert p1_q != orig_q, f"{iid}: prompt question was not changed!"
        assert len(p1_q.strip()) > 10, f"{iid}: prompt question too short: '{p1_q}'"
    print(f"PASS [3/5]: Every prompt question differs from the original (0/156 unchanged).")

    # 4. Token-level overlap distribution
    jaccards = []
    recalls = []
    for orig, p1 in zip(orig_items, p1_items):
        t_orig = set(tokenize(orig["prompt"]["question"]))
        t_p1 = set(tokenize(p1["prompt"]["question"]))
        intersection = t_orig & t_p1
        union = t_orig | t_p1
        jaccard = len(intersection) / len(union) if union else 0.0
        recall = len(intersection) / len(t_orig) if t_orig else 0.0
        jaccards.append(jaccard)
        recalls.append(recall)

    j = np.array(jaccards)
    r = np.array(recalls)
    print(f"PASS [4/5]: Token-level overlap distribution computed:")
    print(f"  Jaccard similarity (|A ∩ B| / |A ∪ B|):")
    print(f"    min:    {j.min():.3f}")
    print(f"    p25:    {np.percentile(j, 25):.3f}")
    print(f"    median: {np.median(j):.3f}")
    print(f"    p75:    {np.percentile(j, 75):.3f}")
    print(f"    max:    {j.max():.3f}")
    print(f"    mean:   {j.mean():.3f} (std: {j.std():.3f})")
    print(f"  Token overlap / original tokens (|A ∩ B| / |A|):")
    print(f"    min:    {r.min():.3f}")
    print(f"    p25:    {np.percentile(r, 25):.3f}")
    print(f"    median: {np.median(r):.3f}")
    print(f"    p75:    {np.percentile(r, 75):.3f}")
    print(f"    max:    {r.max():.3f}")
    print(f"    mean:   {r.mean():.3f} (std: {r.std():.3f})")

    # 5. Schema entity leak check
    schema_names = get_schema_names()
    print(f"\nChecking schema entity containment across {len(schema_names)} schema entities...")
    leak_count = 0
    for orig, p1 in zip(orig_items, p1_items):
        iid = orig["item_id"]
        orig_q = orig["prompt"]["question"]
        p1_q = p1["prompt"]["question"]
        orig_schema = find_schema_entities(orig_q, schema_names)
        p1_schema = find_schema_entities(p1_q, schema_names)
        leaked = p1_schema - orig_schema
        if leaked:
            print(f"  LEAK in {iid}: paraphrase added schema entities {leaked} not in original!")
            leak_count += 1
    assert leak_count == 0, f"Found {leak_count} items with leaked schema entities!"
    print(f"PASS [5/5]: No paraphrase contains a database column or table name that its original lacks (0 leaks).")

    print("\nALL 5 PARAPHRASE VERIFICATION CHECKS PASSED.")


if __name__ == "__main__":
    main()
