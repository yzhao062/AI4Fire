import json, pathlib
R = json.load(open(pathlib.Path(__file__).resolve().parent / "results.json"))
ref = R["reference"]
print("%-20s %8s %8s %8s %8s %8s %8s %8s  %s" % ("name", "nmae", "auc", "falsemv", "missmv", "stable", "moving", "brier", "diff [95%]"))
print("%-20s %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f" % ("v1", ref["nmae"], ref["move_auc"], ref["false_move"], ref["missed_move"], ref["stable"], ref["moving"], ref["move_brier"]))
for r in sorted(R["variants"], key=lambda r: r["nmae"]):
    print("%-20s %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f  %+.4f [%+.4f, %+.4f]" % (r["name"], r["nmae"], r["move_auc"], r["false_move"], r["missed_move"], r["stable"], r["moving"], r["move_brier"], r["diff_vs_reference"], r["diff_ci"][0], r["diff_ci"][1]))
print({k: round(v, 1) for k, v in R["timings_s"].items()})
