"""Render the variant table for report.md from results.json and rule.RULES (no transcription by hand)."""
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import rule as R  # noqa: E402

rs = json.load(open(HERE / "results.json"))
ref = rs[0]
auc_bar, fm_bar = ref["move_auc"] + 0.05, ref["false_move"]
print("| variant | ladder (most specific first) | draw | cov3 | nMAE | stable | moving | false move | missed move | mAUC | Brier | diff vs v1 [95%] | bars |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for o in rs:
    name = o["name"]
    if name in R.RULES:
        fn = R.RULES[name]
        ladder = " > ".join(["+".join(fn.components[:n]) for n in range(len(fn.components), 0, -1)])
        draw = "fill to 6" if fn.fill else "relax if < %d" % fn.min_rows
    else:
        ladder, draw = "band", "sample 6"
    diff = "%+.3f [%+.3f, %+.3f]" % (o["diff_vs_reference"], o["diff_ci"][0], o["diff_ci"][1]) if "diff_ci" in o else "reference"
    if name == ref["name"]:
        bars = "reference"
    elif o["move_auc"] >= auc_bar and o["false_move"] < fm_bar:
        bars = "both"
    elif o["nmae"] < ref["nmae"]:
        bars = "nMAE only"
    else:
        bars = "none"
    print("| %s | %s | %s | %.2f | %.3f | %.3f | %.3f | %.2f | %.2f | %.3f | %.3f | %s | %s |" % (
        name.replace(">", "\\>"), ladder, draw, o["coverage_3"], o["nmae"], o["stable"], o["moving"], o["false_move"],
        o["missed_move"], o["move_auc"], o["move_brier"], diff, bars))
