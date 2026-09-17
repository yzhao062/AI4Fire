"""Build the version-1 allocation task from ICS-209-PLUS: predict the next day's filed personnel count.

Selection rule, published with the item set:
  - wildfire incidents whose point of origin is outside California, to keep distance from the most reported fires
  - start year 2015 to 2020
  - a run of at least 10 consecutive calendar days of situation reports, each with a positive personnel count
  - at most 6 items per incident, evenly spaced over the run, so that a long fire does not dominate
The target is TOTAL_PERSONNEL on day t+1, which is what incident command filed, not what the fire needed.
"""
import json
import pathlib

import numpy as np
import pandas as pd

D = pathlib.Path(__file__).parent / "data" / "ics209" / "ics209plus-wildfire"
OUT = pathlib.Path(__file__).parent / "task-allocation"
OUT.mkdir(exist_ok=True)
COLS = ["INCIDENT_ID", "INCIDENT_NAME", "POO_STATE", "START_YEAR", "REPORT_TO_DATE", "TOTAL_PERSONNEL", "ACRES",
        "NEW_ACRES", "PCT_CONTAINED_COMPLETED", "EST_IM_COST_TO_DATE", "PROJECTED_FINAL_IM_COST", "GROWTH_POTENTIAL",
        "TERRAIN", "FUEL_MODEL", "WEATHER_CONCERNS_NARR", "STR_DESTROYED", "EVACUATION_IN_PROGRESS", "TOTAL_AERIAL",
        "COMPLEXITY_LEVEL_NARR", "CAUSE", "COMPLEX"]

sit = pd.read_csv(D / "ics209-plus-wf_sitreps_1999to2020.csv", usecols=COLS, low_memory=False)
sit["day"] = pd.to_datetime(sit["REPORT_TO_DATE"], errors="coerce", format="mixed")
sit = sit.dropna(subset=["day", "TOTAL_PERSONNEL"])
sit["TOTAL_PERSONNEL"] = pd.to_numeric(sit["TOTAL_PERSONNEL"], errors="coerce")
sit = sit[sit["TOTAL_PERSONNEL"] > 0]
sit["date"] = sit["day"].dt.normalize()
sit = sit.sort_values(["INCIDENT_ID", "date"]).groupby(["INCIDENT_ID", "date"], as_index=False).last()

pool = sit[(sit["START_YEAR"].between(2015, 2020)) & (sit["POO_STATE"] != "CA")].copy()
print("reports in pool:", len(pool), "| incidents:", pool["INCIDENT_ID"].nunique())

items, runs_kept = [], 0
for iid, g in pool.groupby("INCIDENT_ID"):
    g = g.sort_values("date").reset_index(drop=True)
    gap = g["date"].diff().dt.days.fillna(1)
    g["run"] = (gap != 1).cumsum()
    for _, run in g.groupby("run"):
        if len(run) < 10:
            continue
        runs_kept += 1
        run = run.reset_index(drop=True)
        picks = np.linspace(3, len(run) - 2, num=min(6, len(run) - 4)).astype(int)
        for t in sorted(set(picks)):
            row, nxt = run.loc[t], run.loc[t + 1]
            hist = run.loc[max(0, t - 3):t]
            items.append({
                "item_id": "alloc-%s-%s" % (iid, row["date"].date()),
                "incident_id": iid,
                "incident_name": str(row["INCIDENT_NAME"]),
                "state": row["POO_STATE"],
                "start_year": int(row["START_YEAR"]),
                "report_date": str(row["date"].date()),
                "target_date": str(nxt["date"].date()),
                # The source averages multiple filings on one day, so a value can be fractional. The task rounds to
                # whole people, which is what a filing reports, and the rounding rule travels with the item set.
                "target_personnel": float(round(nxt["TOTAL_PERSONNEL"])),
                "baseline_persistence": float(round(row["TOTAL_PERSONNEL"])),
                "day_of_run": int(t + 1),
                "run_length": int(len(run)),
                "context": {
                    "acres": None if pd.isna(row["ACRES"]) else float(row["ACRES"]),
                    "new_acres": None if pd.isna(row["NEW_ACRES"]) else float(row["NEW_ACRES"]),
                    "percent_contained": None if pd.isna(row["PCT_CONTAINED_COMPLETED"]) else float(row["PCT_CONTAINED_COMPLETED"]),
                    "personnel_today": float(round(row["TOTAL_PERSONNEL"])),
                    "personnel_last_days": [float(round(x)) for x in hist["TOTAL_PERSONNEL"]],
                    "acres_last_days": [None if pd.isna(x) else float(x) for x in hist["ACRES"]],
                    "aerial_resources": None if pd.isna(row["TOTAL_AERIAL"]) else float(row["TOTAL_AERIAL"]),
                    "cost_to_date": None if pd.isna(row["EST_IM_COST_TO_DATE"]) else float(row["EST_IM_COST_TO_DATE"]),
                    "projected_final_cost": None if pd.isna(row["PROJECTED_FINAL_IM_COST"]) else float(row["PROJECTED_FINAL_IM_COST"]),
                    "growth_potential": None if pd.isna(row["GROWTH_POTENTIAL"]) else str(row["GROWTH_POTENTIAL"]),
                    "terrain": None if pd.isna(row["TERRAIN"]) else str(row["TERRAIN"]),
                    "fuel_model": None if pd.isna(row["FUEL_MODEL"]) else str(row["FUEL_MODEL"]),
                    "weather_concerns": None if pd.isna(row["WEATHER_CONCERNS_NARR"]) else str(row["WEATHER_CONCERNS_NARR"])[:600],
                    "structures_destroyed": None if pd.isna(row["STR_DESTROYED"]) else float(row["STR_DESTROYED"]),
                    "evacuation_in_progress": None if pd.isna(row["EVACUATION_IN_PROGRESS"]) else str(row["EVACUATION_IN_PROGRESS"]),
                    "cause": None if pd.isna(row["CAUSE"]) else str(row["CAUSE"]),
                },
            })

df = pd.DataFrame(items)
print("runs kept:", runs_kept, "| items:", len(df), "| incidents:", df["incident_id"].nunique(),
      "| states:", df["state"].nunique())
print("\nitems by year:\n", df.groupby("start_year").size().to_string())
print("\nitems by state (top 10):\n", df.groupby("state").size().sort_values(ascending=False).head(10).to_string())

y, p = df["target_personnel"].to_numpy(), df["baseline_persistence"].to_numpy()
scale = df.groupby("incident_id")["target_personnel"].transform("mean").to_numpy()
print("\ntarget personnel: median %.0f | p90 %.0f | max %.0f" % (np.median(y), np.quantile(y, 0.9), y.max()))
print("day-over-day change: median abs %.0f | share of days with a change over 10%%: %.2f"
      % (np.median(np.abs(y - p)), float(np.mean(np.abs(y - p) / np.maximum(p, 1) > 0.1))))
print("persistence baseline: MAE %.1f | MAE normalised per fire %.3f | within 25%%: %.3f | median abs log ratio %.3f"
      % (np.mean(np.abs(y - p)), np.mean(np.abs(y - p) / scale), float(np.mean(np.abs(y - p) / np.maximum(y, 1) <= 0.25)),
         float(np.median(np.abs(np.log((p + 1) / (y + 1)))))))

(OUT / "items.jsonl").write_text("\n".join(json.dumps(r) for r in items) + "\n", encoding="utf-8")
df.drop(columns=["context"]).to_csv(OUT / "items-index.csv", index=False)
print("\nwrote", OUT / "items.jsonl", (OUT / "items.jsonl").stat().st_size, "bytes")
