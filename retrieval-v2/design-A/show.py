"""Print the exact harness numbers for v1 and the leading variants from results.json."""
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
rs = json.load(open(HERE / "results.json"))
ref = rs[0]
print("bars: auc >= %.4f, false_move < %.4f, nmae < %.4f" % (ref["move_auc"] + 0.05, ref["false_move"], ref["nmae"]))
for o in rs:
    if o["name"] in ("v1 (paper)", "A-b5-t3-p", "A-b5-p-t3", "A-b5-t3-p-fill", "A-b5-p-t3-min6", "A-t3", "A-b5-p"):
        print("%-16s nmae %.4f  auc %.4f  false %.4f  cov3 %.4f  cov1 %.4f  drawn %.3f  stable %.4f moving %.4f  brier %.4f  w25 %.3f  diff %s ci %s" % (
            o["name"], o["nmae"], o["move_auc"], o["false_move"], o["coverage_3"], o["coverage_1"], o["mean_drawn"],
            o["stable"], o["moving"], o["move_brier"], o["within_25"],
            ("%+.4f" % o["diff_vs_reference"]) if "diff_vs_reference" in o else "-",
            [round(x, 4) for x in o["diff_ci"]] if "diff_ci" in o else "-"))
clear = [o for o in rs[1:] if o["move_auc"] >= ref["move_auc"] + 0.05 and o["false_move"] < ref["false_move"]]
clear.sort(key=lambda o: o["nmae"])
print("clearing both bars, by nMAE:", [(o["name"], round(o["nmae"], 4)) for o in clear[:6]])
