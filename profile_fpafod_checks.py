"""The validity checks that chose the builder's guards. Read-only; nothing here writes an item.

Each block answers one question the item design depends on: is a date window expressible without ambiguity, is the
largest fire of a state-year unique and named, is a county count complete, and how concentrated is a categorical
answer before any balancing.
"""
import pathlib
import sqlite3

import pandas as pd

DB = pathlib.Path(__file__).parent / "data" / "tooluse" / "FPA_FOD_20221014.sqlite"
con = sqlite3.connect(DB)
pd.set_option("display.width", 220)

print("== 1. does DISCOVERY_DOY agree with DISCOVERY_DATE, so a day-of-year window is the calendar window ==")
d = pd.read_sql("SELECT STATE, FIRE_YEAR, DISCOVERY_DATE, DISCOVERY_DOY FROM Fires", con)
dt = pd.to_datetime(d["DISCOVERY_DATE"], format="%m/%d/%Y", errors="coerce")
print("unparsed dates:", int(dt.isna().sum()), "| day-of-year mismatches:", int((dt.dt.dayofyear != d["DISCOVERY_DOY"]).sum()))
bad = d[dt.dt.year != d["FIRE_YEAR"]]
print("rows where FIRE_YEAR disagrees with the year inside DISCOVERY_DATE:", len(bad))
print(bad.groupby(["STATE", "FIRE_YEAR"]).size().to_string())
print("-> the builder excludes every (state, year) cell these rows touch, in both readings")

print("\n== 2. is the largest fire of a state-year unique, and does it carry a name ==")
t = pd.read_sql("""
SELECT f.STATE, f.FIRE_YEAR, f.FIRE_NAME, f.FIRE_SIZE FROM Fires f
JOIN (SELECT STATE s, FIRE_YEAR y, MAX(FIRE_SIZE) m FROM Fires GROUP BY 1,2) x
  ON f.STATE = x.s AND f.FIRE_YEAR = x.y AND f.FIRE_SIZE = x.m
""", con)
n_at_max = t.groupby(["STATE", "FIRE_YEAR"]).size()
named = t["FIRE_NAME"].notna() & (t["FIRE_NAME"].astype(str).str.strip() != "")
print("state-years:", len(n_at_max), "| with a unique maximum:", int((n_at_max == 1).sum()),
      "| maximum rows carrying a name:", int(named.sum()), "of", len(t))
print("-> the builder keeps only a unique, named, 1,000-acre-or-larger maximum")

print("\n== 3. is a county count complete, or a silent undercount ==")
f = pd.read_sql("SELECT STATE, AVG(CASE WHEN FIPS_CODE IS NULL THEN 0.0 ELSE 1.0 END) cov, COUNT(*) n "
                "FROM Fires GROUP BY 1 ORDER BY cov DESC", con)
print("states with FIPS_CODE on >= 95%% of records: %d of %d" % (int((f["cov"] >= 0.95).sum()), len(f)))
print(f.head(12).to_string(index=False))
print("-> county questions are asked only in those states")

print("\n== 4. how concentrated is the top-cause answer before any balancing ==")
cs = pd.read_sql("SELECT STATE, FIRE_YEAR, NWCG_GENERAL_CAUSE c, COUNT(*) n FROM Fires GROUP BY 1,2,3", con)
top = cs.sort_values(["STATE", "FIRE_YEAR", "n"], ascending=[True, True, False]).groupby(["STATE", "FIRE_YEAR"]).first()
print(top["c"].value_counts().to_string())
print("modal share across all state-years: %.3f" % top["c"].value_counts(normalize=True).iloc[0])
print("-> one label would carry nearly half the family, so the builder caps any answer at 4 of 13 slots")

print("\n== 5. how concentrated is the peak-year answer ==")
sy = pd.read_sql("SELECT STATE, FIRE_YEAR, COUNT(*) n FROM Fires GROUP BY 1,2", con)
pk = sy.sort_values(["STATE", "n"], ascending=[True, False]).groupby("STATE").first()
print(pk["FIRE_YEAR"].value_counts().head(8).to_string())
print("modal share: %.3f -> no balancing needed on this family" % pk["FIRE_YEAR"].value_counts(normalize=True).iloc[0])
