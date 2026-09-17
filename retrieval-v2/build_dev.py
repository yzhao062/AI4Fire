"""Build the development set for the allocation retrieval-rule experiment.

Development fire-days come from the population run_allocation.build_pool draws analogues from, fires that
started before 2015 or in California, so no development incident is an evaluation incident. Items are built
exactly as build_items_ics209.py builds evaluation items (runs of at least 10 consecutive reports with a
positive personnel count, up to six evenly spaced picks per run, the same context fields and rounding), then
sampled to 600 with at most two per incident, stratified by start year, under a fixed seed.

Each development item also carries, in `pool_rows`, nothing: the pool is rebuilt by dev_eval.py, which
excludes the item's own incident when drawing for it.

Usage: python build_dev.py   (writes dev-items.jsonl beside this file)
"""
import collections
import json
import pathlib
import random
import sys

import numpy as np
import pandas as pd

FIRE_BENCH = pathlib.Path(r"C:\Users\yuezh\PycharmProjects\fire-bench")
HERE = pathlib.Path(__file__).resolve().parent
D = FIRE_BENCH / "data" / "ics209" / "ics209plus-wildfire"
COLS = ["INCIDENT_ID", "INCIDENT_NAME", "POO_STATE", "START_YEAR", "REPORT_TO_DATE", "TOTAL_PERSONNEL", "ACRES",
        "NEW_ACRES", "PCT_CONTAINED_COMPLETED", "EST_IM_COST_TO_DATE", "PROJECTED_FINAL_IM_COST", "GROWTH_POTENTIAL",
        "TERRAIN", "FUEL_MODEL", "WEATHER_CONCERNS_NARR", "STR_DESTROYED", "EVACUATION_IN_PROGRESS", "TOTAL_AERIAL",
        "CAUSE"]
SEED = 20260916
N_DEV = 600


def main():
    eval_items = [json.loads(l) for l in (FIRE_BENCH / "task-allocation" / "items-v1.jsonl").read_text(encoding="utf-8").splitlines()]
    eval_incidents = {i["incident_id"] for i in eval_items}

    sit = pd.read_csv(D / "ics209-plus-wf_sitreps_1999to2020.csv", usecols=COLS, low_memory=False)
    sit["day"] = pd.to_datetime(sit["REPORT_TO_DATE"], errors="coerce", format="mixed")
    sit = sit.dropna(subset=["day", "TOTAL_PERSONNEL"])
    sit["TOTAL_PERSONNEL"] = pd.to_numeric(sit["TOTAL_PERSONNEL"], errors="coerce")
    sit = sit[sit["TOTAL_PERSONNEL"] > 0]
    sit["date"] = sit["day"].dt.normalize()
    sit = sit.sort_values(["INCIDENT_ID", "date"]).groupby(["INCIDENT_ID", "date"], as_index=False).last()
    pop = sit[((sit["START_YEAR"] < 2015) | (sit["POO_STATE"] == "CA")) & (~sit["INCIDENT_ID"].isin(eval_incidents))].copy()
    print("development population: reports %d | incidents %d" % (len(pop), pop["INCIDENT_ID"].nunique()))

    items = []
    fire_days = collections.defaultdict(list)  # incident -> every eligible day's next-day count, for fire_mean
    for iid, g in pop.groupby("INCIDENT_ID"):
        g = g.sort_values("date").reset_index(drop=True)
        gap = g["date"].diff().dt.days.fillna(1)
        g["run"] = (gap != 1).cumsum()
        for _, run in g.groupby("run"):
            if len(run) < 10:
                continue
            run = run.reset_index(drop=True)
            for t in range(len(run) - 1):
                fire_days[iid].append(float(round(run.loc[t + 1, "TOTAL_PERSONNEL"])))
            picks = np.linspace(3, len(run) - 2, num=min(6, len(run) - 4)).astype(int)
            for t in sorted(set(picks)):
                row, nxt = run.loc[t], run.loc[t + 1]
                hist = run.loc[max(0, t - 3):t]
                f = lambda v: None if pd.isna(v) else float(v)
                s = lambda v: None if pd.isna(v) else str(v)
                items.append({
                    "item_id": "dev-%s-%s" % (iid, row["date"].date()),
                    "incident_id": iid, "incident_name": str(row["INCIDENT_NAME"]), "state": row["POO_STATE"],
                    "start_year": int(row["START_YEAR"]),
                    "report_date": str(row["date"].date()), "target_date": str(nxt["date"].date()),
                    "target_personnel": float(round(nxt["TOTAL_PERSONNEL"])),
                    "baseline_persistence": float(round(row["TOTAL_PERSONNEL"])),
                    "day_of_run": int(t + 1), "run_length": int(len(run)),
                    "context": {
                        "acres": f(row["ACRES"]), "new_acres": f(row["NEW_ACRES"]),
                        "percent_contained": f(row["PCT_CONTAINED_COMPLETED"]),
                        "personnel_today": float(round(row["TOTAL_PERSONNEL"])),
                        "personnel_last_days": [float(round(x)) for x in hist["TOTAL_PERSONNEL"]],
                        "acres_last_days": [f(x) for x in hist["ACRES"]],
                        "aerial_resources": f(row["TOTAL_AERIAL"]), "cost_to_date": f(row["EST_IM_COST_TO_DATE"]),
                        "projected_final_cost": f(row["PROJECTED_FINAL_IM_COST"]),
                        "growth_potential": s(row["GROWTH_POTENTIAL"]), "terrain": s(row["TERRAIN"]),
                        "fuel_model": s(row["FUEL_MODEL"]),
                        "weather_concerns": None if pd.isna(row["WEATHER_CONCERNS_NARR"]) else str(row["WEATHER_CONCERNS_NARR"])[:600],
                        "structures_destroyed": f(row["STR_DESTROYED"]),
                        "evacuation_in_progress": s(row["EVACUATION_IN_PROGRESS"]), "cause": s(row["CAUSE"]),
                    },
                })
    print("candidate development items: %d from %d incidents" % (len(items), len({i["incident_id"] for i in items})))

    # Stratified sample: N_DEV items spread over start years in proportion to candidates, at most two per incident.
    rng = random.Random(SEED)
    by_year = collections.defaultdict(list)
    for it in items:
        by_year[it["start_year"]].append(it)
    total = len(items)
    picked = []
    for year in sorted(by_year):
        pool = by_year[year]
        quota = max(1, round(N_DEV * len(pool) / total))
        rng.shuffle(pool)
        seen = collections.Counter()
        for it in pool:
            if len([p for p in picked if p["start_year"] == year]) >= quota:
                break
            if seen[it["incident_id"]] >= 2:
                continue
            picked.append(it)
            seen[it["incident_id"]] += 1
    for it in picked:
        it["fire_mean"] = float(np.mean(fire_days[it["incident_id"]]))
    picked.sort(key=lambda i: i["item_id"])
    assert not ({i["incident_id"] for i in picked} & eval_incidents)
    (HERE / "dev-items.jsonl").write_text("\n".join(json.dumps(r) for r in picked) + "\n", encoding="utf-8")
    y = np.array([i["target_personnel"] for i in picked]); b = np.array([i["baseline_persistence"] for i in picked])
    moved = np.abs(y - b) / np.maximum(b, 1) > 0.1
    print("development items: %d | incidents %d | states %d | years %s..%s | moving days %.3f" % (
        len(picked), len({i["incident_id"] for i in picked}), len({i["state"] for i in picked}),
        min(i["start_year"] for i in picked), max(i["start_year"] for i in picked), float(moved.mean())))
    print("wrote", HERE / "dev-items.jsonl")


if __name__ == "__main__":
    main()
