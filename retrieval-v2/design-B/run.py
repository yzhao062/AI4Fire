"""Evaluate every family-B variant and v1 on the development items with the shared harness, in one process.

    python run.py                 builds the pool once, checks the frozen scales, scores v1 and every variant,
                                  prints the harness table, applies the prespecified selection rule, and writes
                                  results.json and results.txt beside this file.

Nothing here touches the 300 test items: only dev-items.jsonl is scored.
"""
import json
import pathlib
import random
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import dev_eval  # noqa: E402
import fit_scales  # noqa: E402
import rule as rulemod  # noqa: E402

AUC_MARGIN = 0.05


def select(results, ref):
    """The brief's prespecified rule, applied within this family. Returns (name, bar_met)."""
    def n_params(name):
        r = rulemod.RULES[name]
        w = rulemod.WEIGHTS[r.weights]
        return len(rulemod.FEATURES) * 2 + sum(1 for x in w if x != 1.0) + (0 if r.cap is None else 1)

    cleared = [r for r in results if r["move_auc"] >= ref["move_auc"] + AUC_MARGIN and r["false_move"] < ref["false_move"]]
    if cleared:
        best = min(cleared, key=lambda r: r["nmae"])
        tied = [r for r in cleared if r["nmae"] <= best["nmae"] + 0.002]
        pick = min(tied, key=lambda r: (n_params(r["name"]), r["nmae"]))
        return pick["name"], True
    lowered = [r for r in results if r["nmae"] < ref["nmae"]]
    if not lowered:
        return None, False
    best = max(lowered, key=lambda r: r["move_auc"])
    return best["name"], False


def diagnose(pool, items, ref, out):
    """Where the frozen rule's gain comes from: by report period, and what its draws look like."""
    by_id = {r["analogue_id"]: r for r in pool}
    err_ref, err = np.array(ref["errors"]), np.array(out["errors"])
    years = np.array([int(it["report_date"][:4]) for it in items])
    lines = []
    for label, m in [("report day before 2015", years < 2015), ("report day 2015 or later (California)", years >= 2015)]:
        lines.append("  %-40s n=%3d  v1 nMAE %.3f  %s nMAE %.3f  diff %+.3f" % (
            label, int(m.sum()), float(err_ref[m].mean()), out["name"], float(err[m].mean()), float((err - err_ref)[m].mean())))
    in_band, n_inc, same_pers, n_rows = [], [], [], 0
    for it in items:
        ids = out["drawn_ids"][it["item_id"]]
        if not ids:
            continue
        c = it["context"]
        key = rulemod.key_of(c["acres"], c["percent_contained"], c["personnel_today"])
        rows = [by_id[d] for d in ids]
        in_band.append(np.mean([r["key"] == key for r in rows]))
        same_pers.append(np.mean([r["key"][2] == key[2] for r in rows]))
        n_inc.append(len({r["incident_id"] for r in rows}))
        n_rows += len(rows)
    lines.append("  %s draws: %.2f distinct incidents per item, %.0f%% of drawn rows in the item's v1 band, %.0f%% in its personnel band" % (
        out["name"], float(np.mean(n_inc)), 100 * float(np.mean(in_band)), 100 * float(np.mean(same_pers))))
    return {"lines": lines, "distinct_incidents": float(np.mean(n_inc)), "share_in_v1_band": float(np.mean(in_band)),
            "share_in_pers_band": float(np.mean(same_pers))}


def main():
    t0 = time.time()
    pool = dev_eval.load_pool()
    items = dev_eval.load_items()
    print("pool rows: %d | dev items: %d | pool built in %.1fs" % (len(pool), len(items), time.time() - t0))

    # the frozen scales must be exactly what a fresh fit on the pre-2015 rows gives
    fresh = fit_scales.fit(pool)
    for f in rulemod.FEATURES:
        assert abs(fresh["mean"][f] - rulemod.SCALES["mean"][f]) < 1e-9 and abs(fresh["std"][f] - rulemod.SCALES["std"][f]) < 1e-9, f
    print("frozen scales in scales.json match a fresh fit on %s" % fresh["fitted_on"])

    t0 = time.time()
    rulemod.prepare(pool)
    print("standardized %d pool rows in %.1fs" % (len(rulemod.REG.ids), time.time() - t0))

    print(dev_eval.HEADER)
    ref = dev_eval.evaluate(dev_eval.rule_v1, pool, items, name="v1 (paper)")
    results, timings = [], {}
    for name, fn in rulemod.RULES.items():
        t0 = time.time()
        out = dev_eval.evaluate(fn, pool, items, name=name, reference=ref)
        timings[name] = time.time() - t0
        results.append(out)

    # determinism and rng independence of the frozen rule: same draws under two different rng seeds and on a rerun
    frozen = rulemod.RULES[rulemod.FROZEN]
    for it in items[:25]:
        cands = dev_eval.candidates(pool, it)
        a = [r["analogue_id"] for r in frozen(cands, it, random.Random(1))]
        b = [r["analogue_id"] for r in frozen(cands, it, random.Random(2))]
        c = [r["analogue_id"] for r in frozen(cands, it, random.Random(dev_eval.item_seed(it["item_id"])))]
        assert a == b == c, it["item_id"]
    print("frozen rule %s: identical draws under three different rng seeds on 25 items (rng unused)" % rulemod.FROZEN)

    pick, bar_met = select(results, ref)
    print("prespecified selection within family B: %s (%s)" % (
        pick, "both bars cleared" if bar_met else "bar not met; highest AUC among nMAE < v1"))
    print("FROZEN = %s" % rulemod.FROZEN)
    slow = {k: v for k, v in timings.items() if v > 60}
    print("timing: slowest variant %.1fs (harness loop included)%s" % (max(timings.values()), "" if not slow else "; over a minute: %s" % slow))

    frozen_out = next(r for r in results if r["name"] == rulemod.FROZEN)
    diagnostics = diagnose(pool, items, ref, frozen_out)
    for line in diagnostics["lines"]:
        print(line)

    table = [dev_eval.HEADER, dev_eval.fmt_row(ref)] + [dev_eval.fmt_row(r) for r in results]
    (HERE / "results.txt").write_text("\n".join(table) + "\n", encoding="utf-8")
    slim = lambda o: {k: v for k, v in o.items() if k not in ("errors", "drawn_ids")}
    json.dump({"reference": slim(ref), "variants": [slim(r) for r in results], "frozen": rulemod.FROZEN,
               "selection": {"pick": pick, "bar_met": bar_met}, "timings_s": timings,
               "frozen_diagnostics": {k: v for k, v in diagnostics.items() if k != "lines"},
               "frozen_drawn_ids": frozen_out["drawn_ids"]},
              open(HERE / "results.json", "w"), indent=1)
    print("wrote", HERE / "results.txt", "and", HERE / "results.json")


if __name__ == "__main__":
    main()
