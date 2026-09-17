"""Profile the FPA-FOD SQLite build: tables, columns, row counts, and the distributions the tool-use questions need."""
import pathlib
import sqlite3
import zipfile

import pandas as pd

D = pathlib.Path(__file__).parent / "data" / "tooluse"
DB = D / "FPA_FOD_20221014.sqlite"

VD = D / "_variable_descriptions.csv"
if not VD.exists():
    with zipfile.ZipFile(D / "RDS-2013-0009.6_Data_Format4_SQLITE.zip") as z:
        with z.open("Data/_variable_descriptions.csv") as src:
            VD.write_bytes(src.read())

con = sqlite3.connect(DB)
tabs = pd.read_sql("SELECT name, type FROM sqlite_master WHERE type IN ('table','view')", con)
print("tables and views:")
print(tabs.to_string())

for t in tabs[tabs["type"] == "table"]["name"]:
    try:
        n = pd.read_sql("SELECT COUNT(*) AS n FROM '%s'" % t, con)["n"][0]
    except Exception as exc:
        n = "err: %s" % exc
    print("  %-30s rows=%s" % (t, n))

MAIN = "Fires"
cols = pd.read_sql("PRAGMA table_info('%s')" % MAIN, con)
print("\n%s columns: %d" % (MAIN, len(cols)))
print(cols[["name", "type"]].to_string())

print("\nvariable descriptions:")
vd = pd.read_csv(VD)
print(vd.head(60).to_string())

print("\nkey distributions")
for q, label in [
    ("SELECT MIN(FIRE_YEAR), MAX(FIRE_YEAR), COUNT(*) FROM Fires", "year range and total"),
    ("SELECT NWCG_GENERAL_CAUSE, COUNT(*) n FROM Fires GROUP BY 1 ORDER BY n DESC", "general cause"),
    ("SELECT FIRE_SIZE_CLASS, COUNT(*) n FROM Fires GROUP BY 1 ORDER BY 1", "size class"),
    ("SELECT COUNT(DISTINCT STATE) FROM Fires", "distinct states"),
    ("SELECT COUNT(*) FROM Fires WHERE FIRE_SIZE IS NULL", "null fire size"),
    ("SELECT COUNT(*) FROM Fires WHERE DISCOVERY_DATE IS NULL", "null discovery date"),
    ("SELECT COUNT(*) FROM Fires WHERE CONT_DATE IS NULL", "null containment date"),
    ("SELECT COUNT(*) FROM Fires WHERE FIRE_NAME IS NULL OR TRIM(FIRE_NAME)=''", "null fire name"),
    ("SELECT typeof(DISCOVERY_DATE), COUNT(*) n FROM Fires GROUP BY 1", "discovery date storage type"),
    ("SELECT DISCOVERY_DATE, DISCOVERY_DOY, FIRE_YEAR FROM Fires LIMIT 5", "discovery date sample"),
    ("SELECT NWCG_REPORTING_AGENCY, COUNT(*) n FROM Fires GROUP BY 1 ORDER BY n DESC LIMIT 15", "reporting agency"),
    ("SELECT STATE, COUNT(*) n, ROUND(SUM(FIRE_SIZE)) acres FROM Fires GROUP BY 1 ORDER BY n DESC LIMIT 15", "state"),
    ("SELECT FIRE_NAME, STATE, FIRE_YEAR, ROUND(FIRE_SIZE) FROM Fires ORDER BY FIRE_SIZE DESC LIMIT 15", "largest"),
]:
    print("\n--", label)
    try:
        print(pd.read_sql(q, con).to_string())
    except Exception as exc:
        print("   query failed:", exc)
