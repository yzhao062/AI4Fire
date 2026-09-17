"""Extract ICS-209-PLUS, then profile the fields the allocation task needs: personnel, cost, size, dates, continuity."""
import pathlib
import zipfile

import pandas as pd

D = pathlib.Path(__file__).parent / "data" / "ics209"
SIT = D / "ics209plus-wildfire" / "ics209-plus-wf_sitreps_1999to2020.csv"
INC = D / "ics209plus-wildfire" / "ics209-plus-wf_incidents_1999to2020.csv"

if not SIT.exists():
    with zipfile.ZipFile(D / "wildfire.zip") as z:
        for n in z.namelist():
            if n.endswith(".csv") and not n.startswith("__MACOSX"):
                z.extract(n, D)
    with zipfile.ZipFile(D / "reference.zip") as z:
        for n in z.namelist():
            if n.endswith(".csv") and not n.startswith("__MACOSX"):
                z.extract(n, D)

head = pd.read_csv(SIT, nrows=5, low_memory=False)
print("sitrep columns:", len(head.columns))
want = [c for c in head.columns if any(k in c.upper() for k in
        ("INCIDENT_ID", "REPORT_TO_DATE", "REPORT_FROM_DATE", "TOTAL_PERSONNEL", "IM_COST", "ACRES", "PCT_CONTAINED",
         "EVACUATION", "STR_DESTROYED", "INJURIES", "FATALITIES", "CAUSE", "POO_STATE", "INCIDENT_NAME", "START_YEAR",
         "FUEL", "WEATHER", "GROWTH_POTENTIAL", "TERRAIN", "COMPLEX", "TOTAL_AERIAL", "CRS_", "RESOURCE"))]
print("selected:", len(want))
sit = pd.read_csv(SIT, usecols=want, low_memory=False)
print("sitreps:", len(sit), "| incidents:", sit["INCIDENT_ID"].nunique())
for c in ("TOTAL_PERSONNEL", "EST_IM_COST_TO_DATE", "ACRES", "PCT_CONTAINED"):
    if c in sit:
        s = pd.to_numeric(sit[c], errors="coerce")
        print("%-22s filled %5.1f%% | median %10.1f | p90 %12.1f | max %12.1f" %
              (c, 100 * s.notna().mean(), s.median(), s.quantile(0.9), s.max()))
print("\ncolumns kept:", sorted(sit.columns.tolist()))

sit["day"] = pd.to_datetime(sit["REPORT_TO_DATE"], errors="coerce")
inc = pd.read_csv(INC, low_memory=False)
print("\nincident columns:", len(inc.columns))
print("incident sample fields:", [c for c in inc.columns if any(k in c.upper() for k in
      ("INCIDENT_ID", "NAME", "START", "END", "STATE", "FINAL_ACRES", "COST", "PERSONNEL", "COMPLEX"))][:25])

runs = (sit.dropna(subset=["TOTAL_PERSONNEL"]).groupby("INCIDENT_ID")
           .agg(days=("day", "nunique"), first=("day", "min"), last=("day", "max"),
                peak=("TOTAL_PERSONNEL", "max"), state=("POO_STATE", "first")))
runs["year"] = runs["first"].dt.year
print("\nincidents with any personnel series:", len(runs))
print("with 10+ days:", (runs["days"] >= 10).sum(), "| 20+ days:", (runs["days"] >= 20).sum())
outside = runs[(runs["days"] >= 15) & (runs["state"] != "CA")]
print("15+ days and outside California:", len(outside))
print("\nby year, 15+ days, outside CA:")
print(outside.groupby("year").size().to_string())
print("\nlargest such fires, 2015 onward:")
print(outside[outside["year"] >= 2015].sort_values("peak", ascending=False).head(12).to_string())
