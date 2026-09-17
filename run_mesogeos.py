"""Run the Mesogeos fire danger task: 30 days of drivers for one cell, does a fire of 30 hectares or more start next day.

    python run_mesogeos.py --dry-run
    python run_mesogeos.py --models claude-opus-5 --folds 0 1

Bare gives the driver window alone. Grounded adds the climatology of each driver over the training years, so the model
can see whether today's values are unusual for the season rather than judging raw numbers cold. Scored by AUPRC on the
fire class and per-class F1, the metrics the source paper reports, against its published LSTM, Transformer, and GTN.
"""
import argparse
import json
import pathlib
import re
import statistics
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from sklearn.metrics import average_precision_score, f1_score

import gw

S = pathlib.Path(__file__).parent
MAX_OUT = 1536  # see the note in the module docstring on why this is not a few hundred
TASK = S / "task-mesogeos"
PUBLISHED = {"LSTM": 0.853, "Transformer": 0.856, "GTN": 0.858}
SHOW = ["t2m", "d2m", "tp", "sp", "wind_speed", "rh", "lai", "ndvi", "smi", "lst_day", "lst_night", "ssrd"]
SYSTEM = ('You judge wildfire danger from daily driver values. Answer with one JSON object and nothing else: '
          '{"probability": <number between 0 and 1>, "fire": true or false, "reasoning": "<one short sentence>"}.')


def summarise(series, k=6):
    """Show the last k days in full and the window statistics, so the prompt stays short without hiding the trend."""
    vals = [v for v in series if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not vals:
        return "no observations"
    tail = ", ".join("missing" if v is None or (isinstance(v, float) and np.isnan(v)) else "%.4g" % v
                     for v in series[-k:])
    return "last %d days %s | window mean %.4g, min %.4g, max %.4g" % (k, tail, statistics.fmean(vals), min(vals), max(vals))


def render(item, clim=None):
    c = item["context"]
    lines = ["A 1 km cell in the Mediterranean, at longitude %.4f, latitude %.4f." % (c["longitude"], c["latitude"]),
             "Daily fire drivers over the %d days ending %s:" % (c["history_days"], c["window_end"])]
    for name in SHOW:
        if name in c["daily"]:
            lines.append("- %s: %s" % (name, summarise(c["daily"][name])))
    static = c.get("static") or {}
    if static:
        lines.append("Static drivers: " + ", ".join("%s %.4g" % (k, v) for k, v in static.items()
                                                    if isinstance(v, (int, float))))
    units = c.get("units") or {}
    if units:
        lines.append("Units: " + "; ".join("%s is %s" % (k, v) for k, v in units.items() if k in SHOW))
    if clim:
        lines.append("")
        lines.append("Climatology for this calendar month over the training years, as mean and standard deviation:")
        for name in SHOW:
            if name in clim:
                lines.append("- %s: mean %.4g, sd %.4g" % (name, clim[name][0], clim[name][1]))
    lines += ["", "Question: does a wildfire that burns at least %d hectares start in this cell on %s?"
              % (c["size_class_hectares"], c["target_date"])]
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n".join(lines)}]


def climatology():
    """Monthly mean and standard deviation per driver, computed from the 2006 to 2019 training rows of the source
    files by build_climatology, so no test value informs the grounding."""
    path = TASK / "climatology.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def parse(text):
    if not text:
        return None, None
    p = re.search(r'"probability"\s*:\s*([0-9.]+)', text)
    f = re.search(r'"fire"\s*:\s*(true|false)', text, re.I)
    return (float(p.group(1)) if p else None), (f.group(1).lower() == "true" if f else None)


def score(rows, label):
    ok = [r for r in rows if r["probability"] is not None or r["call"] is not None]
    if not ok:
        return {"run": label, "items": len(rows), "parsed": 0}
    y = np.array([r["label"] for r in ok])
    prob = np.array([r["probability"] if r["probability"] is not None else (1.0 if r["call"] else 0.0) for r in ok])
    call = np.array([r["call"] if r["call"] is not None else (r["probability"] or 0) >= 0.5 for r in ok])
    return {"run": label, "items": len(rows), "parsed": len(ok),
            "auprc": float(average_precision_score(y, prob)),
            "f1_fire": float(f1_score(y, call, pos_label=1)),
            "f1_no_fire": float(f1_score(y, call, pos_label=0)),
            "positive_rate_called": float(np.mean(call)), "positive_rate_true": float(np.mean(y)),
            "published_auprc": PUBLISHED}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["claude-opus-5"])
    ap.add_argument("--conditions", nargs="*", default=["bare", "grounded"])
    ap.add_argument("--folds", nargs="*", type=int, default=[0])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    allitems = [json.loads(l) for l in (TASK / "items.jsonl").read_text(encoding="utf-8").splitlines()]
    clim = climatology()
    items = [i for i in allitems if i["split"] == "test" and i.get("fold", 0) in args.folds]
    print("test items in folds %s: %d | positive rate %.3f | climatology months %d"
          % (args.folds, len(items), np.mean([i["label"] for i in items]), len(clim)))

    if args.dry_run:
        it = items[0]
        print("\n--- bare\n" + render(it)[1]["content"][:1500])
        month = it["context"]["window_end"][5:7]
        print("\n--- grounded tail\n" + render(it, clim.get(month))[1]["content"][-700:])
        print("\nlabel:", it["label"], it["answer"])
        return

    key = gw.load_key()
    summaries = []
    for model in args.models:
        for cond in args.conditions:
            def one(it):
                month = it["context"]["window_end"][5:7]
                msgs = render(it, clim.get(month) if cond == "grounded" else None)
                base = {"item_id": it["item_id"], "label": it["label"], "target_date": it["target_date"]}
                try:
                    text, usage, served = gw.call(key, model, msgs, max_tokens=MAX_OUT)
                except Exception as exc:
                    return dict(base, error=str(exc)[:200], probability=None, call=None)
                p, f = parse(text)
                return dict(base, raw=text, usage=usage, served_model=served, probability=p, call=f)

            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                rows = list(ex.map(one, items))
            safe = re.sub(r"[^A-Za-z0-9._-]", "_", model)
            (TASK / ("responses-%s-%s.jsonl" % (safe, cond))).write_text(
                "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
            s = score(rows, "%s/%s" % (model, cond))
            s["errors"] = sum(1 for r in rows if r.get("error"))
            s["tokens_in"] = sum((r.get("usage") or {}).get("prompt_tokens", 0) for r in rows)
            summaries.append(s)
            print(json.dumps(s))

    path = TASK / "scores.json"
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    fresh = {s["run"] for s in summaries}
    path.write_text(json.dumps([s for s in old if s["run"] not in fresh] + summaries, indent=1), encoding="utf-8")
    print("\nwrote", path)


if __name__ == "__main__":
    main()
