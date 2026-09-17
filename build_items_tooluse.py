"""Build the version-1 fire data tool-use task from FPA-FOD: questions whose gold answer a reference program computes.

The model under test gets tool access to the same frozen database later. This builder ships the questions, the
reference SQL that produced each answer, and the answer itself. It never calls a language model.

Selection rule, published with the item set:
  - one frozen snapshot: the FPA-FOD 6th Edition SQLite build (Short 2022, RDS-2013-0009.6), 2,303,566 records,
    fire years 1992 to 2020, sha256 of the downloaded archive recorded with every item
  - twelve question families, each one parameterized SQL template; the gold answer is that template's output on
    the snapshot, so no answer is annotated and none is written by hand
  - candidate parameter cells are enumerated exhaustively over the snapshot, then cut by family guards: a floor on
    the cell size so the answer is not degenerate, a unique argmax with a stated margin wherever the answer is an
    argmax, a non-blank FIRE_NAME wherever the answer is a name, leap-free spans wherever the question names a
    calendar window, and exclusion of every (state, year) cell touched by the 38 records whose FIRE_YEAR disagrees
    with the year inside DISCOVERY_DATE, which would give such a question two defensible readings
  - survivors are sorted by a fixed key, reduced to one cell per state at a per-family offset so the years vary
    (two per state in the county family, where only nine states clear the coverage guard), and sampled by an even
    stride, so the item set is a deterministic function of the snapshot and carries no random seed
  - in the two families whose answer is a category, no single gold answer may take more than 4 of the 13 slots;
    without that cap one label carries half the family and the majority-class baseline measures the label
    distribution instead of the model
  - 13 items per family, 156 in total
Every stored query is re-executed at the end and must reproduce its stored answer, so the recorded query is
provably the one that computed the gold.
"""
import datetime as dt
import hashlib
import json
import pathlib
import re
import sqlite3
from collections import Counter

import pandas as pd

S = pathlib.Path(__file__).parent
D = S / "data" / "tooluse"
DB = D / "FPA_FOD_20221014.sqlite"
OUT = S / "task-tooluse"
OUT.mkdir(exist_ok=True)
PER_FAMILY = 13

SOURCE = {
    "dataset": "FPA-FOD 6th Edition (FPA_FOD_20221014)",
    "citation": "Short, Karen C. 2022. Spatial wildfire occurrence data for the United States, 1992-2020 "
                "[FPA_FOD_20221014]. 6th Edition. Fort Collins, CO: Forest Service Research Data Archive. "
                "https://doi.org/10.2737/RDS-2013-0009.6",
    "catalog_page": "https://www.fs.usda.gov/rds/archive/catalog/RDS-2013-0009.6",
    "product_url": "https://www.fs.usda.gov/rds/archive/products/RDS-2013-0009.6/RDS-2013-0009.6_Data_Format4_SQLITE.zip",
    "use_terms": "Collected using funding from the U.S. Government and can be used without additional permissions "
                 "or fees; citation required.",
    "table": "Fires",
}
FETCH = json.loads((D / "fetch.json").read_text(encoding="utf-8"))

# Tolerance rules travel with every item, so a reader can score the task without reading this file.
TOL_INT = {"rule": "exact_int", "detail": "Integer answer. Scored by exact equality after stripping thousands "
                                          "separators and surrounding words."}
TOL_ACRES = {"rule": "relative", "rel_tol": 0.005, "abs_floor": 1.0,
             "detail": "Acreage answer. Correct when |predicted - gold| <= max(0.005 * gold, 1.0) acres."}
TOL_PCT = {"rule": "absolute", "abs_tol": 0.1,
           "detail": "Percentage answer. Correct when |predicted - gold| <= 0.1 percentage points."}
TOL_STR = {"rule": "string_normalized",
           "detail": "String answer. Both sides are uppercased, every character outside A-Z, 0-9 and space becomes "
                     "a space, runs of spaces collapse to one, and the trimmed results must match."}
TOL_CAT = {"rule": "categorical_exact",
           "detail": "Categorical answer. Case-insensitive exact match against one of the listed options."}

SEASONS = [("spring", 3, 1, 5, 31), ("summer", 6, 1, 8, 31), ("autumn", 9, 1, 11, 30)]
MONTH = {3: "March", 5: "May", 6: "June", 8: "August", 9: "September", 11: "November"}
SIZE_CLASS_TEXT = {"E": "E (300 to 999 acres)", "F": "F (1,000 to 4,999 acres)", "G": "G (5,000 acres or more)"}
MISSING_CAUSE = "Missing data/not specified/undetermined"
# Three-year spans that contain no leap year. A leap year shifts the day-of-year of every date after February, so a
# span that straddled one could not carry a single integer day-of-year window matching the calendar range the
# question states.
SPANS3 = [(y, y + 2) for y in (1993, 1997, 2001, 2005, 2009, 2013, 2017)]
SPANS5 = [(y, y + 4) for y in (1995, 2000, 2005, 2010, 2015)]
DECADES = [(1992, 2001), (2002, 2011), (2011, 2020)]

con = sqlite3.connect(DB)
# Two lookup indexes. They add no value to any record and change no answer; they exist so the 156 reference queries
# and the 156 self-check re-runs do not each scan a 958 MB table. The sha256 recorded with every item is the sha256
# of the archive as downloaded from the Forest Service, which this does not touch.
for name, colspec in [("ix_fb_state_year", "STATE, FIRE_YEAR"), ("ix_fb_cause_year", "NWCG_GENERAL_CAUSE, FIRE_YEAR")]:
    con.execute("CREATE INDEX IF NOT EXISTS %s ON Fires(%s)" % (name, colspec))
con.commit()


def q1(sql, params):
    """Run one reference query and return its first row. This function is the reference program."""
    return con.execute(sql, params).fetchone()


print("snapshot:", DB.name, "%.1f MB" % (DB.stat().st_size / 1e6), "| archive sha256:", FETCH["sha256"][:16] + "...")

# Every candidate cell is enumerated in memory. Only the 156 reference queries and their re-runs touch SQL, which
# keeps the builder to minutes rather than hours and keeps the recorded query the single path to each gold answer.
F = pd.read_sql("SELECT FOD_ID, STATE, FIRE_YEAR, DISCOVERY_DATE, DISCOVERY_DOY, NWCG_GENERAL_CAUSE, "
                "NWCG_CAUSE_CLASSIFICATION, FIRE_SIZE, FIRE_SIZE_CLASS, FIRE_NAME, FIPS_NAME, FIPS_CODE, "
                "COMPLEX_NAME, MTBS_ID FROM Fires", con)
n_rows = len(F)
print("records loaded:", n_rows)

# Guard 1: the records whose FIRE_YEAR disagrees with the year inside DISCOVERY_DATE. A question naming a year has
# two defensible readings on any cell those records touch, so every such (state, year) pair is excluded everywhere.
pyear = pd.to_datetime(F["DISCOVERY_DATE"], format="%m/%d/%Y", errors="coerce").dt.year
mism = F[pyear != F["FIRE_YEAR"]]
AMBIG = {(r.STATE, int(r.FIRE_YEAR)) for r in mism.itertuples()} | \
        {(r.STATE, int(y)) for r, y in zip(mism.itertuples(), pyear[mism.index])}
print("date-encoding disagreements:", len(mism), "| (state, year) cells excluded:", len(AMBIG), sorted(AMBIG))


def clean(state, years):
    """True when no year in the range carries a date-encoding disagreement for this state."""
    return all((state, int(y)) not in AMBIG for y in years)


# Guard 2: FIPS coverage, used only by the county family, so a county answer is not a silent undercount.
cov = F.assign(has=F["FIPS_CODE"].notna()).groupby("STATE")["has"].mean()
FIPS_OK = set(cov[cov >= 0.95].index)
print("states with FIPS_NAME coverage >= 0.95:", len(FIPS_OK), sorted(FIPS_OK))

CAUSES = sorted(F["NWCG_GENERAL_CAUSE"].dropna().unique())
STATES = sorted(F["STATE"].dropna().unique())
print("causes:", len(CAUSES), "| states:", len(STATES))

SY = F.groupby(["STATE", "FIRE_YEAR"]).agg(n=("FOD_ID", "size"), acres=("FIRE_SIZE", "sum")).reset_index()
SYC = F.groupby(["STATE", "FIRE_YEAR", "NWCG_GENERAL_CAUSE"], observed=True).size().rename("n").reset_index()


def one_per_state(cands, offset, per_state=1):
    """Up to per_state candidates per state, taken at fixed offsets into that state's sorted list so years vary."""
    by_state = {}
    for c in sorted(cands, key=lambda c: (c["state"], repr(c["key"]))):
        by_state.setdefault(c["state"], []).append(c)
    out = []
    for rank, st in enumerate(sorted(by_state)):
        lst = by_state[st]
        for j in range(min(per_state, len(lst))):
            out.append(lst[(rank * 7 + offset + j * 13) % len(lst)])
    return out


def stride(cands, n):
    """Take n candidates by an even stride over a sorted list. No randomness, so the pick is reproducible."""
    if len(cands) <= n:
        return list(cands)
    step = len(cands) / n
    return [cands[int(i * step)] for i in range(n)]


def stride_capped(pool, n, cap):
    """An even stride over a sorted pool, with no gold answer allowed more than cap of the n slots.

    The cap applies to the two families whose answer is a category. Without it one label carries most of the family
    and the majority-class baseline measures the label distribution rather than the model. A slot whose answer is
    already full advances to the next cell in the same fixed order, so the result stays a function of the pool.
    """
    step = max(len(pool) / n, 1.0)
    first = [int(i * step) for i in range(min(n, len(pool)))]
    used, out = Counter(), []
    for i in first + [j for j in range(len(pool)) if j not in set(first)]:
        if len(out) >= n:
            break
        if used[str(pool[i]["ans"])] >= cap:
            continue
        used[str(pool[i]["ans"])] += 1
        out.append(pool[i])
    return out


def pick(cands, offset, n=PER_FAMILY, per_state=1, cap=None):
    """Reduce to per_state cells per state, order by a fixed key, then take n by an even stride."""
    pool = sorted(one_per_state(cands, offset, per_state), key=lambda c: (c["state"], repr(c["key"])))
    return stride(pool, n) if cap is None else stride_capped(pool, n, cap)


ITEMS = []


def size_memo(acres):
    """Memorization risk for an answer that identifies a single fire, keyed mechanically to that fire's size.

    A fire of 100,000 acres or more draws national coverage a model may well have read. A 1,250-acre fire in
    Delaware does not, so labelling a whole family 'high' would overstate the risk on most of its items.
    """
    if acres >= 100000:
        return "high", ("The answer identifies a fire of %.0f acres. Fires that size are nationally reported, so a "
                        "model may recall this one without any tool." % acres)
    if acres >= 10000:
        return "moderate", ("The answer identifies a fire of %.0f acres. A fire that size is reported regionally "
                            "and sometimes nationally, so partial recall is possible." % acres)
    return "low", ("The answer identifies a fire of %.0f acres. Fires that size are not individually reported "
                   "outside their own state, so recall is an implausible route to this answer." % acres)


MEMO_RECORD_LEVEL = (" FPA-FOD also accounts at the record level: a complex that the public record reports as one "
                     "large incident can appear here as several smaller records, so a remembered answer is often "
                     "the wrong answer for this database.")


def emit(family, tier, cand, question, answer_format, answer, answer_type, tol, sql, params,
         memo_risk, memo_note, options=None, years=None, extra=None):
    ITEMS.append({
        "item_id": "tooluse-%s-%02d" % (family, sum(1 for i in ITEMS if i["family"] == family) + 1),
        "family": family,
        "tier": tier,
        # The fields a model sees. Everything below this block is reference material used only for scoring.
        "prompt": {"question": question, "answer_format": answer_format, "options": options},
        "answer": answer,
        "answer_type": answer_type,
        "tolerance": tol,
        "reference_query": sql,
        "reference_params": list(params),
        "memorization_risk": memo_risk,
        "memorization_note": memo_note,
        "state": cand.get("state"),
        "extra": extra or {},
        "dating": {
            "fire_years_covered": years,
            "data_vintage": "1992-2020",
            "source_release": "2022-10-14 (6th Edition, FPA_FOD_20221014)",
            "source_public_since": "2013 (1st Edition); this edition 2022",
            "snapshot_fetched_utc": FETCH["fetched_utc"],
        },
        "source": dict(SOURCE, snapshot_archive_sha256=FETCH["sha256"], snapshot_records=int(n_rows)),
    })


# --- family 1: count in one state-year. The single-lookup tier. ---------------------------------------------------
SQL1 = "SELECT COUNT(*) AS answer FROM Fires WHERE STATE = ? AND FIRE_YEAR = ?"
c1 = [{"state": r.STATE, "key": int(r.FIRE_YEAR), "year": int(r.FIRE_YEAR)} for r in SY.itertuples()
      if r.n >= 100 and clean(r.STATE, [r.FIRE_YEAR])]
for c in pick(c1, 0):
    p = (c["state"], c["year"])
    emit("count_state_year", "single lookup", c,
         "In the FPA-FOD wildfire database, how many fire records have STATE = '%s' and FIRE_YEAR = %d?" % p,
         "a single integer", int(q1(SQL1, p)[0]), "integer", TOL_INT, SQL1, p,
         "none", "A count over 2.3 million records for one state-year. No published source states this number, so "
                 "a model without the tool can only estimate it.",
         years=[c["year"], c["year"]])

# --- family 2: count by cause within a state-year. ----------------------------------------------------------------
SQL2 = "SELECT COUNT(*) AS answer FROM Fires WHERE STATE = ? AND FIRE_YEAR = ? AND NWCG_GENERAL_CAUSE = ?"
c2 = [{"state": r.STATE, "key": (int(r.FIRE_YEAR), r.NWCG_GENERAL_CAUSE), "year": int(r.FIRE_YEAR),
       "cause": r.NWCG_GENERAL_CAUSE} for r in SYC.itertuples()
      if r.n >= 25 and r.NWCG_GENERAL_CAUSE != MISSING_CAUSE and clean(r.STATE, [r.FIRE_YEAR])]
for c in pick(c2, 3):
    p = (c["state"], c["year"], c["cause"])
    emit("count_cause_state_year", "filtered aggregate", c,
         "In the FPA-FOD wildfire database, how many fire records have STATE = '%s', FIRE_YEAR = %d and "
         "NWCG_GENERAL_CAUSE = '%s'?" % p,
         "a single integer", int(q1(SQL2, p)[0]), "integer", TOL_INT, SQL2, p,
         "none", "A three-way filtered count. The cause breakdown of one state-year is not a published figure.",
         years=[c["year"], c["year"]], extra={"cause": c["cause"]})

# --- family 3: count in one county-year. The narrow-filter tier. --------------------------------------------------
SQL3 = "SELECT COUNT(*) AS answer FROM Fires WHERE STATE = ? AND FIPS_NAME = ? AND FIRE_YEAR = ?"
CNT = F[F["FIPS_NAME"].notna()].groupby(["STATE", "FIPS_NAME", "FIRE_YEAR"], observed=True).size().rename("n").reset_index()
c3 = [{"state": r.STATE, "key": (int(r.FIRE_YEAR), r.FIPS_NAME), "year": int(r.FIRE_YEAR), "county": r.FIPS_NAME}
      for r in CNT.itertuples() if r.n >= 20 and r.STATE in FIPS_OK and clean(r.STATE, [r.FIRE_YEAR])]
# Only nine states clear the FIPS coverage guard, so this family takes two cells per state to reach 13 items.
for c in pick(c3, 5, per_state=2):
    p = (c["state"], c["county"], c["year"])
    emit("count_county_year", "filtered aggregate", c,
         "In the FPA-FOD wildfire database, how many fire records have STATE = '%s', FIPS_NAME = '%s' and "
         "FIRE_YEAR = %d?" % p,
         "a single integer", int(q1(SQL3, p)[0]), "integer", TOL_INT, SQL3, p,
         "none", "A county-year count. Asked only in states where FIPS_NAME is populated on at least 95 percent of "
                 "records, so the answer is not a silent undercount.",
         years=[c["year"], c["year"]], extra={"county": c["county"]})

# --- family 4: count inside a season window. ----------------------------------------------------------------------
SQL4 = "SELECT COUNT(*) AS answer FROM Fires WHERE STATE = ? AND FIRE_YEAR = ? AND DISCOVERY_DOY BETWEEN ? AND ?"
c4 = []
for r in SY.itertuples():
    if r.n < 200 or not clean(r.STATE, [r.FIRE_YEAR]):
        continue
    for name, m1, d1, m2, d2 in SEASONS:
        y = int(r.FIRE_YEAR)
        c4.append({"state": r.STATE, "key": (y, name), "year": y, "season": name,
                   "doy": (dt.date(y, m1, d1).timetuple().tm_yday, dt.date(y, m2, d2).timetuple().tm_yday),
                   "label": "%d %s and %d %s" % (d1, MONTH[m1], d2, MONTH[m2])})
for c in pick(c4, 8):
    p = (c["state"], c["year"], c["doy"][0], c["doy"][1])
    emit("count_season_window", "filtered aggregate", c,
         "In the FPA-FOD wildfire database, how many fire records in STATE = '%s' have a discovery date between "
         "%s %d inclusive?" % (c["state"], c["label"], c["year"]),
         "a single integer", int(q1(SQL4, p)[0]), "integer", TOL_INT, SQL4, p,
         "none", "A seasonal count for one state-year. The window is expressed through the integer DISCOVERY_DOY "
                 "field, whose bounds are computed for this specific year, so a leap year cannot shift it.",
         years=[c["year"], c["year"]], extra={"season": c["season"], "doy_window": list(c["doy"])})

# --- family 5: total acres in a state-year. -------------------------------------------------------------------------
SQL5 = "SELECT ROUND(SUM(FIRE_SIZE), 1) AS answer FROM Fires WHERE STATE = ? AND FIRE_YEAR = ?"
c5 = [{"state": r.STATE, "key": int(r.FIRE_YEAR), "year": int(r.FIRE_YEAR)} for r in SY.itertuples()
      if r.acres >= 5000 and r.n >= 100 and clean(r.STATE, [r.FIRE_YEAR])]
for c in pick(c5, 11):
    p = (c["state"], c["year"])
    emit("acres_state_year", "filtered aggregate", c,
         "In the FPA-FOD wildfire database, what is the sum of the FIRE_SIZE field, in acres, over all fire records "
         "with STATE = '%s' and FIRE_YEAR = %d?" % p,
         "a single number of acres", float(q1(SQL5, p)[0]), "acres", TOL_ACRES, SQL5, p,
         "none", "A sum over every record in a state-year. Other agencies publish state annual acreage on other "
                 "definitions, so a remembered figure will not match this one.",
         years=[c["year"], c["year"]])

# --- family 6: count of fires in a size class over a five-year span. --------------------------------------------------
SQL6 = "SELECT COUNT(*) AS answer FROM Fires WHERE STATE = ? AND FIRE_SIZE_CLASS = ? AND FIRE_YEAR BETWEEN ? AND ?"
c6 = []
for (y0, y1) in SPANS5:
    sub = F[F["FIRE_YEAR"].between(y0, y1)]
    g = sub.groupby(["STATE", "FIRE_SIZE_CLASS"], observed=True).size().rename("n").reset_index()
    for r in g.itertuples():
        # The floor is 20 rather than the guard's natural 10, so no gold answer sits on the guard boundary where a
        # constant equal to the floor would pick up items for free.
        if r.n >= 20 and r.FIRE_SIZE_CLASS in SIZE_CLASS_TEXT and clean(r.STATE, range(y0, y1 + 1)):
            c6.append({"state": r.STATE, "key": (y0, r.FIRE_SIZE_CLASS), "y0": y0, "y1": y1,
                       "cls": r.FIRE_SIZE_CLASS})
for c in pick(c6, 2):
    p = (c["state"], c["cls"], c["y0"], c["y1"])
    emit("count_size_class_span", "filtered aggregate", c,
         "In the FPA-FOD wildfire database, how many fire records have STATE = '%s', FIRE_SIZE_CLASS = '%s' and "
         "FIRE_YEAR between %d and %d inclusive? Size class %s is %s."
         % (c["state"], c["cls"], c["y0"], c["y1"], c["cls"], SIZE_CLASS_TEXT[c["cls"]].split(" ", 1)[1].strip("()")),
         "a single integer", int(q1(SQL6, p)[0]), "integer", TOL_INT, SQL6, p,
         "none", "A count of large fires over a five-year span, compositional over two filters and a range.",
         years=[c["y0"], c["y1"]], extra={"size_class": c["cls"]})

# --- family 7: the name of the largest fire in a state-year. ------------------------------------------------------
SQL7 = ("SELECT FIRE_NAME AS answer FROM Fires WHERE STATE = ? AND FIRE_YEAR = ? "
        "ORDER BY FIRE_SIZE DESC, FOD_ID ASC LIMIT 1")
mx = F.loc[F.groupby(["STATE", "FIRE_YEAR"])["FIRE_SIZE"].idxmax()]
ties = F.merge(mx[["STATE", "FIRE_YEAR", "FIRE_SIZE"]], on=["STATE", "FIRE_YEAR", "FIRE_SIZE"]) \
         .groupby(["STATE", "FIRE_YEAR"]).size().rename("n_at_max").reset_index()
mx = mx.merge(ties, on=["STATE", "FIRE_YEAR"])
c7 = [{"state": r.STATE, "key": int(r.FIRE_YEAR), "year": int(r.FIRE_YEAR), "size": float(r.FIRE_SIZE),
       "complex": None if pd.isna(r.COMPLEX_NAME) else str(r.COMPLEX_NAME),
       "mtbs": None if pd.isna(r.MTBS_ID) else str(r.MTBS_ID)} for r in mx.itertuples()
      if r.n_at_max == 1 and isinstance(r.FIRE_NAME, str) and r.FIRE_NAME.strip() and r.FIRE_SIZE >= 1000
      and clean(r.STATE, [r.FIRE_YEAR])]
for c in pick(c7, 6):
    p = (c["state"], c["year"])
    risk, note = size_memo(c["size"])
    emit("largest_fire_name", "filtered aggregate", c,
         "In the FPA-FOD wildfire database, among fire records with STATE = '%s' and FIRE_YEAR = %d, find the one "
         "with the largest FIRE_SIZE. Report the value of its FIRE_NAME field." % p,
         "the fire name as stored in the database", str(q1(SQL7, p)[0]), "string", TOL_STR, SQL7, p,
         risk, note + MEMO_RECORD_LEVEL,
         years=[c["year"], c["year"]],
         extra={"answer_fire_size_acres": c["size"], "answer_complex_name": c["complex"],
                "answer_mtbs_id": c["mtbs"]})

# --- family 8: the size of the largest fire in a season across a three-year span. -------------------------------------
SQL8 = ("SELECT MAX(FIRE_SIZE) AS answer FROM Fires WHERE STATE = ? AND FIRE_YEAR BETWEEN ? AND ? "
        "AND DISCOVERY_DOY BETWEEN ? AND ?")
c8 = []
for (y0, y1) in SPANS3:
    yrs = list(range(y0, y1 + 1))
    sub = F[F["FIRE_YEAR"].between(y0, y1)]
    for name, m1, d1, m2, d2 in SEASONS:
        los = {dt.date(y, m1, d1).timetuple().tm_yday for y in yrs}
        his = {dt.date(y, m2, d2).timetuple().tm_yday for y in yrs}
        assert len(los) == 1 and len(his) == 1, "span %d-%d is not leap-free" % (y0, y1)
        lo, hi = los.pop(), his.pop()
        w = sub[sub["DISCOVERY_DOY"].between(lo, hi)]
        for st, mval in w.groupby("STATE")["FIRE_SIZE"].max().items():
            # 5,000 acres rather than 1,000, so no gold answer sits on the guard boundary and the family is about
            # fires a state would actually mount a campaign on.
            if mval >= 5000 and clean(st, yrs):
                c8.append({"state": st, "key": (y0, name), "y0": y0, "y1": y1, "season": name, "doy": (lo, hi),
                           "label": "%d %s and %d %s" % (d1, MONTH[m1], d2, MONTH[m2])})
for c in pick(c8, 9):
    p = (c["state"], c["y0"], c["y1"], c["doy"][0], c["doy"][1])
    a = float(q1(SQL8, p)[0])
    risk, note = size_memo(a)
    emit("largest_fire_size_span", "multi-step", c,
         "In the FPA-FOD wildfire database, consider fire records with STATE = '%s' whose FIRE_YEAR is between %d "
         "and %d inclusive and whose discovery date falls between %s of those years. What is the largest value of "
         "FIRE_SIZE among them, in acres?" % (c["state"], c["y0"], c["y1"], c["label"]),
         "a single number of acres", a, "acres", TOL_ACRES, SQL8, p,
         risk, note + " Recall would still have to survive the season filter and the three-year span, which select "
                      "a fire the reader has no reason to expect." + MEMO_RECORD_LEVEL,
         years=[c["y0"], c["y1"]], extra={"season": c["season"], "doy_window": list(c["doy"])})

# --- family 9: which cause tops a state-year. The categorical argmax. -------------------------------------------------
SQL9 = ("SELECT NWCG_GENERAL_CAUSE AS answer, COUNT(*) n FROM Fires WHERE STATE = ? AND FIRE_YEAR = ? "
        "GROUP BY 1 ORDER BY n DESC, 1 ASC LIMIT 1")
c9 = []
for (st, yr), g in SYC.groupby(["STATE", "FIRE_YEAR"]):
    if g["n"].sum() < 200 or not clean(st, [yr]) or len(g) < 2:
        continue
    g = g.sort_values(["n", "NWCG_GENERAL_CAUSE"], ascending=[False, True])
    top, second = g.iloc[0], g.iloc[1]
    if top["NWCG_GENERAL_CAUSE"] == MISSING_CAUSE or top["n"] < 1.1 * second["n"]:
        continue
    c9.append({"state": st, "key": int(yr), "year": int(yr), "margin": float(top["n"] / second["n"]),
               "ans": top["NWCG_GENERAL_CAUSE"]})
for c in pick(c9, 4, cap=4):
    p = (c["state"], c["year"])
    emit("top_cause_state_year", "multi-step", c,
         "In the FPA-FOD wildfire database, among fire records with STATE = '%s' and FIRE_YEAR = %d, which value of "
         "NWCG_GENERAL_CAUSE appears on the most records?" % p,
         "one value from the options list", str(q1(SQL9, p)[0]), "categorical", TOL_CAT, SQL9, p,
         "partial", "The answer is one of thirteen closed categories and the national prior is informative, so a "
                    "model can score above chance with no tool at all. The majority-class baseline printed below is "
                    "the number this family has to be read against.",
         options=CAUSES, years=[c["year"], c["year"]], extra={"argmax_margin": round(c["margin"], 3)})

# --- family 10: the human-caused share of a state-year. ---------------------------------------------------------------
SQL10 = ("SELECT ROUND(100.0 * SUM(CASE WHEN NWCG_CAUSE_CLASSIFICATION = 'Human' THEN 1 ELSE 0 END) / COUNT(*), 2) "
         "AS answer FROM Fires WHERE STATE = ? AND FIRE_YEAR = ?")
c10 = [{"state": r.STATE, "key": int(r.FIRE_YEAR), "year": int(r.FIRE_YEAR)} for r in SY.itertuples()
       if r.n >= 200 and clean(r.STATE, [r.FIRE_YEAR])]
for c in pick(c10, 1):
    p = (c["state"], c["year"])
    emit("human_share_state_year", "multi-step", c,
         "In the FPA-FOD wildfire database, among all fire records with STATE = '%s' and FIRE_YEAR = %d, what "
         "percentage carry NWCG_CAUSE_CLASSIFICATION = 'Human'? The denominator is every record in that state-year, "
         "including those classified as missing or undetermined." % p,
         "a percentage to two decimal places", float(q1(SQL10, p)[0]), "percent", TOL_PCT, SQL10, p,
         "none", "A ratio of two counts inside one state-year. The denominator is stated in the question, so the "
                 "only ambiguity a model faces is in its own query, not in the wording.",
         years=[c["year"], c["year"]])

# --- family 11: which year in a decade holds the most records for a state. --------------------------------------------
SQL11 = ("SELECT FIRE_YEAR AS answer, COUNT(*) n FROM Fires WHERE STATE = ? AND FIRE_YEAR BETWEEN ? AND ? "
         "GROUP BY 1 ORDER BY n DESC, 1 ASC LIMIT 1")
c11 = []
for st in STATES:
    for (y0, y1) in DECADES:
        g = SY[(SY["STATE"] == st) & (SY["FIRE_YEAR"].between(y0, y1))].sort_values(
            ["n", "FIRE_YEAR"], ascending=[False, True])
        if len(g) < 10 or g["n"].sum() < 500 or not clean(st, range(y0, y1 + 1)):
            continue
        if g.iloc[0]["n"] < 1.05 * g.iloc[1]["n"]:
            continue
        c11.append({"state": st, "key": y0, "y0": y0, "y1": y1, "margin": float(g.iloc[0]["n"] / g.iloc[1]["n"])})
for c in pick(c11, 10):
    p = (c["state"], c["y0"], c["y1"])
    emit("peak_year_state", "multi-step", c,
         "In the FPA-FOD wildfire database, among fire records with STATE = '%s' and FIRE_YEAR between %d and %d "
         "inclusive, which single FIRE_YEAR holds the most records?" % p,
         "a single four-digit year", int(q1(SQL11, p)[0]), "integer", TOL_INT, SQL11, p,
         "none", "An argmax over ten annual counts for one state. The answer is a year, so a model can guess inside "
                 "a ten-value range; the majority-class baseline printed below bounds how far that gets.",
         years=[c["y0"], c["y1"]], extra={"argmax_margin": round(c["margin"], 3)})

# --- family 12: which state tops a cause-year. The cross-state argmax. -------------------------------------------------
SQL12 = ("SELECT STATE AS answer, COUNT(*) n FROM Fires WHERE NWCG_GENERAL_CAUSE = ? AND FIRE_YEAR = ? "
         "GROUP BY 1 ORDER BY n DESC, 1 ASC LIMIT 1")
c12 = []
for cause in CAUSES:
    if cause == MISSING_CAUSE:
        continue
    for yr in range(1992, 2021):
        g = SYC[(SYC["NWCG_GENERAL_CAUSE"] == cause) & (SYC["FIRE_YEAR"] == yr)].sort_values(
            ["n", "STATE"], ascending=[False, True])
        if len(g) < 5 or g["n"].sum() < 500 or g.iloc[0]["n"] < 1.1 * g.iloc[1]["n"]:
            continue
        if not clean(g.iloc[0]["STATE"], [yr]):
            continue
        c12.append({"state": g.iloc[0]["STATE"], "key": (cause, yr), "cause": cause, "year": int(yr),
                    "margin": float(g.iloc[0]["n"] / g.iloc[1]["n"]), "ans": g.iloc[0]["STATE"]})
# This family is keyed by (cause, year) rather than by state, so the per-state reduction does not apply. The cap on
# how many of the thirteen slots one answer may take does.
for c in stride_capped(sorted(c12, key=lambda c: repr(c["key"])), PER_FAMILY, 4):
    p = (c["cause"], c["year"])
    emit("top_state_for_cause", "multi-step", c,
         "In the FPA-FOD wildfire database, among fire records with NWCG_GENERAL_CAUSE = '%s' and FIRE_YEAR = %d, "
         "which STATE holds the most records?" % p,
         "a two-letter state code", str(q1(SQL12, p)[0]), "categorical", TOL_CAT, SQL12, p,
         "partial", "The answer is one of 52 state codes and a model can guess the biggest fire state with no tool. "
                    "The majority-class baseline printed below says how far that guess gets.",
         options=STATES, years=[c["year"], c["year"]],
         extra={"cause": c["cause"], "argmax_margin": round(c["margin"], 3)})

# -----------------------------------------------------------------------------------------------------------------
# Self-check: re-run every stored query and confirm it reproduces the stored answer. A query that had drifted from
# its answer would make the item unreproducible, which is the one failure this task cannot ship with.
bad = 0
for it in ITEMS:
    got, exp = q1(it["reference_query"], tuple(it["reference_params"]))[0], it["answer"]
    ok = abs(float(got) - float(exp)) < 1e-6 if isinstance(exp, (int, float)) else str(got) == str(exp)
    if not ok:
        bad += 1
        print("  MISMATCH", it["item_id"], got, exp)
print("\nself-check: %d of %d stored queries reproduce their stored answer" % (len(ITEMS) - bad, len(ITEMS)))
assert bad == 0


# -----------------------------------------------------------------------------------------------------------------
# Naive baseline. The trivial strategy on a short-answer set is a constant: one answer repeated across a family. The
# constant is chosen on the item set itself, which makes this an upper bound on constant guessing rather than an
# estimate of it. Any model score has to be read against these numbers.
def norm_str(s):
    return re.sub(r"\s+", " ", re.sub(r"[^A-Z0-9]", " ", str(s).upper())).strip()


def correct(item, pred):
    """Score one prediction under the item's own tolerance rule. This scorer ships with the items."""
    rule, gold = item["tolerance"]["rule"], item["answer"]
    try:
        if rule == "exact_int":
            return int(round(float(pred))) == int(gold)
        if rule == "relative":
            return abs(float(pred) - float(gold)) <= max(item["tolerance"]["rel_tol"] * abs(float(gold)),
                                                         item["tolerance"]["abs_floor"])
        if rule == "absolute":
            return abs(float(pred) - float(gold)) <= item["tolerance"]["abs_tol"]
    except (TypeError, ValueError):
        return False
    if rule == "categorical_exact":
        return str(pred).strip().lower() == str(gold).strip().lower()
    return norm_str(pred) == norm_str(gold)


def best_constant(items):
    """The best single answer repeated over these items, and the share of items it gets right."""
    best, score = None, -1.0
    for cand in [i["answer"] for i in items]:
        s = sum(1 for i in items if correct(i, cand)) / len(items)
        if s > score:
            best, score = cand, s
    return best, score


df = pd.DataFrame([{"item_id": i["item_id"], "family": i["family"], "tier": i["tier"],
                    "answer_type": i["answer_type"], "memorization_risk": i["memorization_risk"],
                    "answer": i["answer"], "years": "%d-%d" % tuple(i["dating"]["fire_years_covered"]),
                    "state": i["state"]} for i in ITEMS])
print("\nitems: %d | families: %d" % (len(ITEMS), df["family"].nunique()))
print("\nitems by tier:\n", df.groupby("tier").size().to_string())
print("\nitems by answer type:\n", df.groupby("answer_type").size().to_string())
print("\nitems by memorization risk:\n", df.groupby("memorization_risk").size().to_string())
print("\nfire years covered: %d to %d | distinct states: %d | most items on one state: %d"
      % (min(i["dating"]["fire_years_covered"][0] for i in ITEMS),
         max(i["dating"]["fire_years_covered"][1] for i in ITEMS),
         df["state"].nunique(), df["state"].value_counts().max()))

print("\nnaive baseline, best constant per family (an upper bound on constant guessing):")
rows = []
for fam in sorted(df["family"].unique()):
    fitems = [i for i in ITEMS if i["family"] == fam]
    b, s = best_constant(fitems)
    mode = Counter(str(i["answer"]) for i in fitems).most_common(1)[0]
    rows.append({"family": fam, "n": len(fitems), "answer_type": fitems[0]["answer_type"],
                 "best_constant": str(b)[:32], "best_constant_acc": round(s, 3),
                 "modal_answer": mode[0][:26], "modal_share": round(mode[1] / len(fitems), 3)})
base = pd.DataFrame(rows).sort_values("best_constant_acc", ascending=False)
print(base.to_string(index=False))

overall_b, overall_s = best_constant(ITEMS)
weighted = sum(r["n"] * r["best_constant_acc"] for r in rows) / len(ITEMS)
print("\none constant over the whole set: %r -> %.3f" % (overall_b, overall_s))
print("best constant chosen per family, averaged over items: %.3f" % weighted)
hi = base[base["best_constant_acc"] >= 0.4]
print("FLAG, families a constant already reaches 0.40 or better: %s"
      % (", ".join("%s (%.2f)" % (r.family, r.best_constant_acc) for r in hi.itertuples()) if len(hi) else "none"))

print("\nclass balance on the two categorical families:")
for fam in ("top_cause_state_year", "top_state_for_cause"):
    vals = Counter(i["answer"] for i in ITEMS if i["family"] == fam)
    print("  %-22s %d distinct answers over %d items | %s"
          % (fam, len(vals), sum(vals.values()), ", ".join("%s x%d" % (k[:26], v) for k, v in vals.most_common(5))))

print("\nnumeric answer spread, which is why no constant covers a family:")
for fam in sorted(df[df["answer_type"].isin(["integer", "acres", "percent"])]["family"].unique()):
    v = pd.to_numeric(df[df["family"] == fam]["answer"])
    print("  %-24s min %11.1f | median %11.1f | max %13.1f" % (fam, v.min(), v.median(), v.max()))

print("\nwhich items a model could answer from memory, with no tool:")
for risk in ("high", "moderate", "partial", "low", "none"):
    g = df[df["memorization_risk"] == risk]
    if len(g):
        print("  %-9s %3d items | %s" % (risk, len(g), ", ".join("%s x%d" % (k, v) for k, v in
                                                                 g["family"].value_counts().items())))
exposed = df[df["memorization_risk"].isin(["high", "moderate", "partial"])]
print("  items where recall is a plausible route to the answer: %d of %d (%.0f%%)"
      % (len(exposed), len(df), 100 * len(exposed) / len(df)))
print("  items where it is not: %d (%.0f%%)" % (len(df) - len(exposed), 100 * (1 - len(exposed) / len(df))))
c7i = [i for i in ITEMS if i["family"] == "largest_fire_name"]
div = sum(1 for i in c7i if i["extra"].get("answer_complex_name") or not i["extra"].get("answer_mtbs_id"))
print("  of the %d largest_fire_name items, %d have an answer row that belongs to a complex or carries no MTBS_ID, "
      "so the database's answer need not be the incident the public record names" % (len(c7i), div))

(OUT / "items.jsonl").write_text("\n".join(json.dumps(r) for r in ITEMS) + "\n", encoding="utf-8")
df.to_csv(OUT / "items-index.csv", index=False)
base.to_csv(OUT / "naive-baseline.csv", index=False)
print("\nwrote", OUT / "items.jsonl", (OUT / "items.jsonl").stat().st_size, "bytes")
print("items.jsonl sha256:", hashlib.sha256((OUT / "items.jsonl").read_bytes()).hexdigest()[:32])
