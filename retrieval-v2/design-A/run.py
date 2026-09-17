"""Evaluate every Family A variant and v1 on the development items with the shared harness.

Builds the pool once, scores v1 as the reference, then every variant in rule.RULES with the paired interval
against v1, and prints the harness table. Writes results.json (all rows, without per-item errors) beside this file.
Usage: python run.py
"""
import json
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))   # dev_eval.py lives one level up
sys.path.insert(0, str(HERE))

from dev_eval import HEADER, evaluate, fmt_row, load_items, load_pool, rule_v1   # noqa: E402
import rule as R                                                                # noqa: E402


def main():
    t0 = time.time()
    pool = load_pool()
    items = load_items()
    print("pool rows: %d | dev items: %d | pool built in %.1fs" % (len(pool), len(items), time.time() - t0))
    print(HEADER)
    ref = evaluate(rule_v1, pool, items, name="v1 (paper)")
    results = [ref]
    for name, fn in R.RULES.items():
        t1 = time.time()
        out = evaluate(fn, pool, items, name=name, reference=ref, quiet=True)
        out["seconds"] = time.time() - t1
        print(fmt_row(out) + "  (%.0fs)" % out["seconds"])
        results.append(out)
    print("\nfrozen: %s" % R.FROZEN)
    slim = [{k: v for k, v in o.items() if k not in ("errors", "drawn_ids")} for o in results]
    (HERE / "results.json").write_text(json.dumps(slim, indent=1), encoding="utf-8")
    # selection-rule view: the judge's bars against v1
    print("\nselection view (bars: mAUC >= v1 + 0.05 = %.3f, false move < v1 = %.2f, nMAE < v1 = %.3f)" % (
        ref["move_auc"] + 0.05, ref["false_move"], ref["nmae"]))
    for o in slim[1:]:
        clears = o["move_auc"] >= ref["move_auc"] + 0.05 and o["false_move"] < ref["false_move"]
        print("  %-16s nMAE %.3f  mAUC %.3f  falsemv %.2f  cov3 %.2f  %s" % (
            o["name"], o["nmae"], o["move_auc"], o["false_move"], o["coverage_3"],
            "clears both bars" if clears else ("lowers nMAE only" if o["nmae"] < ref["nmae"] else "-")))


if __name__ == "__main__":
    main()
