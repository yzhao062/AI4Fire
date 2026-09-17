"""Run the ICS-209-PLUS allocation task: sample the version-1 items, render bare and grounded prompts, call the
gateway, and score against the filed next-day personnel count.

    python run_allocation.py --dry-run                 renders one prompt per condition and prints the baselines
    python run_allocation.py --models claude-opus-5    runs the named models, both conditions

The grounded condition adds retrieved analogues from a historical pool that shares no incident with the evaluation
set: fires that started before 2015, or in California. Retrieval keys on acres, containment, and personnel bands.
"""
import argparse
import collections
import hashlib
import json
import math
import pathlib
import random
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

import gw

S = pathlib.Path(__file__).parent
MAX_OUT = 1536  # see the note in the module docstring on why this is not a few hundred
TASK = S / "task-allocation"
SIT = S / "data" / "ics209" / "ics209plus-wildfire" / "ics209-plus-wf_sitreps_1999to2020.csv"
PER_YEAR = 50
SEED = 20260915


def band(value, edges):
    for i, e in enumerate(edges):
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return "unknown"
        if value < e:
            return i
    return len(edges)


ACRE_EDGES = [100, 1000, 5000, 20000, 100000]
PCT_EDGES = [10, 30, 60, 90]
PERS_EDGES = [25, 75, 200, 500]


def key_of(acres, pct, pers):
    return (band(acres, ACRE_EDGES), band(pct, PCT_EDGES), band(pers, PERS_EDGES))


def sample_items():
    items = [json.loads(l) for l in (TASK / "items.jsonl").read_text(encoding="utf-8").splitlines()]
    rng = random.Random(SEED)
    by_year = collections.defaultdict(list)
    for it in items:
        by_year[it["start_year"]].append(it)
    picked = []
    for year in sorted(by_year):
        pool = by_year[year]
        seen = collections.Counter()
        rng.shuffle(pool)
        for it in pool:
            if len(picked) % PER_YEAR == 0 and seen and sum(seen.values()) >= PER_YEAR:
                break
            if seen[it["incident_id"]] >= 2:
                continue
            picked.append(it)
            seen[it["incident_id"]] += 1
            if sum(seen.values()) >= PER_YEAR:
                break
    (TASK / "items-v1.jsonl").write_text("\n".join(json.dumps(r) for r in picked) + "\n", encoding="utf-8")
    return picked


def build_pool(eval_incidents):
    """Historical analogues: fires that started before 2015 or in California, so no evaluation incident appears.

    Sharing no incident with the evaluation set is not enough. The California branch admits fires that started
    as late as 2020, so a row here can post-date the day an item asks about. Each row therefore carries its own
    date, and analogues() drops the rows whose outcome day is on or after the item's report day.
    """
    cols = ["INCIDENT_ID", "POO_STATE", "START_YEAR", "REPORT_TO_DATE", "TOTAL_PERSONNEL", "ACRES", "PCT_CONTAINED_COMPLETED"]
    sit = pd.read_csv(SIT, usecols=cols, low_memory=False)
    sit["date"] = pd.to_datetime(sit["REPORT_TO_DATE"], errors="coerce", format="mixed").dt.normalize()
    sit["TOTAL_PERSONNEL"] = pd.to_numeric(sit["TOTAL_PERSONNEL"], errors="coerce")
    sit = sit.dropna(subset=["date", "TOTAL_PERSONNEL"])
    sit = sit[(sit["TOTAL_PERSONNEL"] > 0) & (~sit["INCIDENT_ID"].isin(eval_incidents))]
    sit = sit[(sit["START_YEAR"] < 2015) | (sit["POO_STATE"] == "CA")]
    sit = sit.sort_values(["INCIDENT_ID", "date"]).groupby(["INCIDENT_ID", "date"], as_index=False).last()
    pool = collections.defaultdict(list)
    for iid, g in sit.groupby("INCIDENT_ID"):
        g = g.sort_values("date").reset_index(drop=True)
        gaps = g["date"].diff().dt.days.fillna(1)
        for t in range(len(g) - 1):
            if gaps.iloc[t + 1] != 1:
                continue
            row, nxt = g.loc[t], g.loc[t + 1]
            pool[key_of(row["ACRES"], row["PCT_CONTAINED_COMPLETED"], row["TOTAL_PERSONNEL"])].append(
                {"analogue_id": "%s@%s" % (iid, row["date"].date()),
                 "date": row["date"],
                 "today": float(row["TOTAL_PERSONNEL"]), "next": float(nxt["TOTAL_PERSONNEL"]),
                 "acres": float(row["ACRES"]) if not pd.isna(row["ACRES"]) else None,
                 "pct": float(row["PCT_CONTAINED_COMPLETED"]) if not pd.isna(row["PCT_CONTAINED_COMPLETED"]) else None})
    return pool


def analogues(pool, item, k=6):
    """Draw up to k analogues whose two days both precede the item's report day, under a seed that does not move.

    Each analogue is a consecutive-day pair, and the second day is the outcome the prompt shows. The first
    version of this rule only required the input day to precede the target day, which let one analogue show an
    outcome filed on the target day itself (round-2 review, 2026-09-16). The rule now requires the outcome day,
    which is the input day plus one, to precede the report day the forecast is made from.

    Python's hash() is salted per process, so seeding on it made the drawn set unreproducible from one run to
    the next. blake2b is stable across processes and machines.
    """
    c = item["context"]
    report = pd.Timestamp(item["report_date"])
    hits = [r for r in pool.get(key_of(c["acres"], c["percent_contained"], c["personnel_today"]), [])
            if r["date"] + pd.Timedelta(days=1) < report]
    seed = int.from_bytes(hashlib.blake2b(item["item_id"].encode("utf-8"), digest_size=8).digest(), "big")
    return random.Random(seed).sample(hits, min(k, len(hits)))


SYSTEM = ("You forecast wildfire resource filings. You answer with one JSON object and nothing else: "
          '{"personnel": <integer>, "reasoning": "<one sentence>"}.')


def render(item, extra=None):
    c = item["context"]
    lines = [
        "A wildfire incident filed an ICS-209 situation report for %s." % item["report_date"],
        "Report fields:",
        "- acres burned: %s" % c["acres"],
        "- new acres in the last period: %s" % c["new_acres"],
        "- percent contained: %s" % c["percent_contained"],
        "- total personnel assigned today: %s" % c["personnel_today"],
        "- personnel on the last reported days, oldest first: %s" % c["personnel_last_days"],
        "- acres on those days: %s" % c["acres_last_days"],
        "- aerial resources: %s" % c["aerial_resources"],
        "- estimated cost to date: %s" % c["cost_to_date"],
        "- projected final cost: %s" % c["projected_final_cost"],
        "- growth potential: %s" % c["growth_potential"],
        "- terrain difficulty: %s" % c["terrain"],
        "- fuel model: %s" % c["fuel_model"],
        "- structures destroyed so far: %s" % c["structures_destroyed"],
        "- evacuation in progress: %s" % c["evacuation_in_progress"],
        "- weather concerns: %s" % (c["weather_concerns"] or "none recorded"),
        "",
        "Question: how many total personnel will this incident report as assigned on %s, the next day?" % item["target_date"],
    ]
    if extra:
        lines[16:16] = extra
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n".join(lines)}]


def grounded_block(rows):
    if not rows:
        return ["", "Historical analogues: none matched this band."]
    out = ["", "Historical analogues from earlier incidents in the same size, containment, and staffing band.",
           "Each line gives personnel today, personnel the next day, and the ratio."]
    for r in rows:
        today, nxt = r["today"], r["next"]
        out.append("- %d -> %d (ratio %.2f; acres %s, contained %s)" % (today, nxt, nxt / max(today, 1), r["acres"], r["pct"]))
    ratios = [r["next"] / max(r["today"], 1) for r in rows]
    out.append("- median ratio in this band: %.2f" % float(np.median(ratios)))
    return out


def parse(text):
    if text is None:
        return None
    m = re.search(r'"personnel"\s*:\s*(-?[\d.]+)', text)
    if not m:
        m = re.search(r"(-?\d[\d,]*)", text.replace(",", ""))
    try:
        return float(m.group(1).replace(",", "")) if m else None
    except (ValueError, AttributeError):
        return None


def score(rows, label):
    ok = [r for r in rows if r["prediction"] is not None]
    if not ok:
        return {"run": label, "items": len(rows), "parsed": 0}
    y = np.array([r["target"] for r in ok])
    p = np.array([r["prediction"] for r in ok])
    base = np.array([r["persistence"] for r in ok])
    scale = np.array([r["fire_mean"] for r in ok])
    moved = np.abs(y - base) / np.maximum(base, 1) > 0.1
    return {"run": label, "items": len(rows), "parsed": len(ok),
            "mae": float(np.mean(np.abs(y - p))),
            "mae_norm": float(np.mean(np.abs(y - p) / scale)),
            "within_25pct": float(np.mean(np.abs(y - p) / np.maximum(y, 1) <= 0.25)),
            "median_abs_log_ratio": float(np.median(np.abs(np.log((p + 1) / (y + 1))))),
            "mae_norm_on_moving_days": float(np.mean((np.abs(y - p) / scale)[moved])) if moved.any() else None,
            "beats_persistence_share": float(np.mean(np.abs(y - p) < np.abs(y - base)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["claude-opus-5", "claude-opus-4.8"])
    ap.add_argument("--conditions", nargs="*", default=["bare", "grounded"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    items = sample_items()
    if args.limit:
        items = items[:args.limit]
    # Normalise each error by the incident's mean staffing over its whole run, not over the sampled days, so that
    # the scale does not depend on how many days of a fire the sample happened to take.
    fire_mean = collections.defaultdict(list)
    for line in (TASK / "items.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        fire_mean[row["incident_id"]].append(row["target_personnel"])
    fire_mean = {k: float(np.mean(v)) for k, v in fire_mean.items()}
    print("evaluation items: %d | incidents: %d | states: %d" %
          (len(items), len({i["incident_id"] for i in items}), len({i["state"] for i in items})))

    pool = build_pool({i["incident_id"] for i in items})
    print("historical analogue bands: %d | analogue day pairs: %d" % (len(pool), sum(len(v) for v in pool.values())))
    ana = {it["item_id"]: analogues(pool, it) for it in items}
    print("items with at least three analogues: %.2f" % float(np.mean([len(ana[i["item_id"]]) >= 3 for i in items])))

    base_rows = [{"target": it["target_personnel"], "prediction": it["baseline_persistence"],
                  "persistence": it["baseline_persistence"], "fire_mean": fire_mean[it["incident_id"]]} for it in items]
    print("\nbaseline:", json.dumps(score(base_rows, "persistence"), indent=None))

    if args.dry_run:
        it = items[0]
        print("\n--- bare prompt\n" + render(it)[1]["content"])
        print("\n--- grounded addition\n" + "\n".join(grounded_block(ana[it["item_id"]])))
        print("\ntarget personnel:", it["target_personnel"], "| persistence:", it["baseline_persistence"])
        return

    key = gw.load_key()
    summaries = []
    for model in args.models:
        for cond in args.conditions:
            def one(it):
                msgs = render(it, grounded_block(ana[it["item_id"]]) if cond == "grounded" else None)
                try:
                    text, usage, served = gw.call(key, model, msgs, max_tokens=MAX_OUT)
                except Exception as exc:  # a failed call is recorded, never silently dropped
                    return {"item_id": it["item_id"], "error": str(exc)[:200], "prediction": None,
                            "target": it["target_personnel"], "persistence": it["baseline_persistence"],
                            "fire_mean": fire_mean[it["incident_id"]]}
                return {"item_id": it["item_id"], "raw": text, "usage": usage, "served_model": served,
                        "prediction": parse(text), "target": it["target_personnel"],
                        "persistence": it["baseline_persistence"], "fire_mean": fire_mean[it["incident_id"]],
                        # The drawn analogues are part of the input, so the record keeps them; without this the
                        # grounded prompt of a past run cannot be rebuilt.
                        "analogue_ids": [r["analogue_id"] for r in ana[it["item_id"]]] if cond == "grounded" else []}

            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                rows = list(ex.map(one, items))
            label = "%s/%s" % (model, cond)
            safe = re.sub(r"[^A-Za-z0-9._-]", "_", model)  # Bedrock ids carry colons, which Windows rejects in a path
            out = TASK / ("responses-%s-%s.jsonl" % (safe, cond))
            out.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
            s = score(rows, label)
            s["errors"] = sum(1 for r in rows if r.get("error"))
            s["tokens_in"] = sum((r.get("usage") or {}).get("prompt_tokens", 0) for r in rows)
            s["tokens_out"] = sum((r.get("usage") or {}).get("completion_tokens", 0) for r in rows)
            summaries.append(s)
            print(json.dumps(s))
    summaries.append(score(base_rows, "persistence"))
    # Keep every run ever scored in one file: a later run of one model must not erase the others.
    path = TASK / "scores.json"
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    fresh = {s["run"] for s in summaries}
    merged = [s for s in old if s["run"] not in fresh] + summaries
    path.write_text(json.dumps(merged, indent=1), encoding="utf-8")
    print("\nwrote", TASK / "scores.json")


if __name__ == "__main__":
    main()
