"""Independently recompute every gold answer in pandas, from what each question says rather than from its SQL.

The builder's own self-check re-runs the stored query, which proves the query and the answer agree. It cannot catch
a question whose words describe something the query does not do. This pass recomputes each answer along a separate
path: a pandas filter written from the question's stated semantics, with calendar dates parsed out of DISCOVERY_DATE
rather than read from DISCOVERY_DOY, so the seasonal windows are checked against the text they claim.
"""
import datetime as dt
import json
import pathlib
import re
import sqlite3

import pandas as pd

S = pathlib.Path(__file__).parent
DB = S / "data" / "tooluse" / "FPA_FOD_20221014.sqlite"
ITEMS = [json.loads(l) for l in (S / "task-tooluse" / "items.jsonl").read_text(encoding="utf-8").splitlines() if l]

con = sqlite3.connect(DB)
F = pd.read_sql("SELECT FOD_ID, STATE, FIRE_YEAR, DISCOVERY_DATE, NWCG_GENERAL_CAUSE, NWCG_CAUSE_CLASSIFICATION, "
                "FIRE_SIZE, FIRE_SIZE_CLASS, FIRE_NAME, FIPS_NAME FROM Fires", con)
F["DATE"] = pd.to_datetime(F["DISCOVERY_DATE"], format="%m/%d/%Y", errors="coerce")
print("records:", len(F), "| items:", len(ITEMS))

MONTHS = {"March": 3, "May": 5, "June": 6, "August": 8, "September": 9, "November": 11}


def window_from_text(q, year_lo, year_hi):
    """Read the calendar window out of the question text, e.g. 'between 1 June and 31 August'."""
    m = re.search(r"between (\d+) (\w+) and (\d+) (\w+)", q)
    d1, m1, d2, m2 = int(m.group(1)), MONTHS[m.group(2)], int(m.group(3)), MONTHS[m.group(4)]
    return [(dt.date(y, m1, d1), dt.date(y, m2, d2)) for y in range(year_lo, year_hi + 1)]


def in_windows(sub, wins):
    d = sub["DATE"].dt.date
    keep = pd.Series(False, index=sub.index)
    for a, b in wins:
        keep |= (d >= a) & (d <= b)
    return sub[keep]


bad, checked = [], 0
for it in ITEMS:
    fam, p, gold = it["family"], it["reference_params"], it["answer"]
    y0, y1 = it["dating"]["fire_years_covered"]
    q = it["prompt"]["question"]
    got = None

    if fam == "count_state_year":
        got = len(F[(F["STATE"] == p[0]) & (F["FIRE_YEAR"] == p[1])])
    elif fam == "count_cause_state_year":
        got = len(F[(F["STATE"] == p[0]) & (F["FIRE_YEAR"] == p[1]) & (F["NWCG_GENERAL_CAUSE"] == p[2])])
    elif fam == "count_county_year":
        got = len(F[(F["STATE"] == p[0]) & (F["FIPS_NAME"] == p[1]) & (F["FIRE_YEAR"] == p[2])])
    elif fam == "count_season_window":
        sub = F[(F["STATE"] == p[0]) & (F["FIRE_YEAR"] == p[1])]
        got = len(in_windows(sub, window_from_text(q, y0, y1)))
    elif fam == "acres_state_year":
        got = round(float(F[(F["STATE"] == p[0]) & (F["FIRE_YEAR"] == p[1])]["FIRE_SIZE"].sum()), 1)
    elif fam == "count_size_class_span":
        got = len(F[(F["STATE"] == p[0]) & (F["FIRE_SIZE_CLASS"] == p[1]) & (F["FIRE_YEAR"].between(p[2], p[3]))])
    elif fam == "largest_fire_name":
        sub = F[(F["STATE"] == p[0]) & (F["FIRE_YEAR"] == p[1])]
        got = str(sub.loc[sub["FIRE_SIZE"].idxmax()]["FIRE_NAME"])
    elif fam == "largest_fire_size_span":
        sub = F[(F["STATE"] == p[0]) & (F["FIRE_YEAR"].between(p[1], p[2]))]
        got = float(in_windows(sub, window_from_text(q, y0, y1))["FIRE_SIZE"].max())
    elif fam == "top_cause_state_year":
        sub = F[(F["STATE"] == p[0]) & (F["FIRE_YEAR"] == p[1])]
        got = str(sub["NWCG_GENERAL_CAUSE"].value_counts().idxmax())
    elif fam == "human_share_state_year":
        sub = F[(F["STATE"] == p[0]) & (F["FIRE_YEAR"] == p[1])]
        got = round(100.0 * (sub["NWCG_CAUSE_CLASSIFICATION"] == "Human").sum() / len(sub), 2)
    elif fam == "peak_year_state":
        sub = F[(F["STATE"] == p[0]) & (F["FIRE_YEAR"].between(p[1], p[2]))]
        got = int(sub["FIRE_YEAR"].value_counts().idxmax())
    elif fam == "top_state_for_cause":
        sub = F[(F["NWCG_GENERAL_CAUSE"] == p[0]) & (F["FIRE_YEAR"] == p[1])]
        got = str(sub["STATE"].value_counts().idxmax())

    checked += 1
    ok = abs(float(got) - float(gold)) < 0.051 if isinstance(gold, (int, float)) else str(got) == str(gold)
    if not ok:
        bad.append((it["item_id"], q[:110], got, gold))

print("\nindependently recomputed: %d | agree: %d | disagree: %d" % (checked, checked - len(bad), len(bad)))
for b in bad:
    print("  MISMATCH", b[0], "\n    q:", b[1], "\n    pandas:", b[2], "| stored:", b[3])

print("\nsample items:")
for it in [ITEMS[0], ITEMS[42], ITEMS[70], ITEMS[100], ITEMS[130], ITEMS[150]]:
    print("\n %s [%s, %s, memorization=%s]" % (it["item_id"], it["family"], it["tier"], it["memorization_risk"]))
    print("   Q: %s" % it["prompt"]["question"])
    print("   format: %s | answer: %r | tolerance: %s" % (it["prompt"]["answer_format"], it["answer"],
                                                          it["tolerance"]["rule"]))
    print("   query: %s" % it["reference_query"])
    print("   params: %s" % it["reference_params"])
