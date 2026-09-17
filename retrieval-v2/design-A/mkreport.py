"""Write report.md for design A (the brief's deliverable) from the final results.json and diag-output.txt.

The variant table is rendered from results.json so that no number is transcribed by hand; the prose is below.
Usage: python mkreport.py
"""
import json
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
PY = sys.executable

rs = {o["name"]: o for o in json.load(open(HERE / "results.json"))}
ref, fz = rs["v1 (paper)"], rs["A-b5-t3-p"]
table = subprocess.run([PY, str(HERE / "table.py")], capture_output=True, text=True, check=True).stdout.strip()
table = table.replace("| A-b5-t3-p |", "| **A-b5-t3-p** (frozen) |")

clear = sorted([o for o in rs.values() if o["name"] != ref["name"] and o["move_auc"] >= ref["move_auc"] + 0.05
                and o["false_move"] < ref["false_move"]], key=lambda o: o["nmae"])
clear_txt = ", ".join("`%s` %.4f" % (o["name"], o["nmae"]) for o in clear[:6])

HEAD = """# Design A: recent-dynamics matching

Frozen variant: `A-b5-t3-p` (`rule.FROZEN`). Development numbers as the harness printed them: cov3 {cov3:.2f}, nMAE {nmae:.3f} (v1 {rnmae:.3f}, persistence {pers:.3f}), stable-day error {stable:.3f} (v1 {rstable:.3f}), moving-day error {moving:.3f} (v1 {rmoving:.3f}), false move {fm:.2f} (v1 {rfm:.2f}), missed move {mm:.2f}, direction {dr:.2f}, within-25 {w25:.2f}, movement Brier {brier:.3f}, movement AUC {auc:.3f} (v1 {rauc:.3f}), median-ratio IQR [{iq0:.2f}, {iq1:.2f}] against actual [{aq0:.2f}, {aq1:.2f}], paired difference in nMAE against v1 {diff:+.3f} [{ci0:+.3f}, {ci1:+.3f}]. It clears both bars of the prespecified rule (AUC bar {aucbar:.3f}, false-move bar {fmbar:.2f}) and has the lowest development nMAE among the variants that do.
""".format(cov3=fz["coverage_3"], nmae=fz["nmae"], rnmae=ref["nmae"], pers=fz["nmae_persistence"], stable=fz["stable"],
           rstable=ref["stable"], moving=fz["moving"], rmoving=ref["moving"], fm=fz["false_move"], rfm=ref["false_move"],
           mm=fz["missed_move"], dr=fz["direction_right"], w25=fz["within_25"], brier=fz["move_brier"], auc=fz["move_auc"],
           rauc=ref["move_auc"], iq0=fz["median_ratio_iqr"][0], iq1=fz["median_ratio_iqr"][1], aq0=fz["actual_ratio_iqr"][0],
           aq1=fz["actual_ratio_iqr"][1], diff=fz["diff_vs_reference"], ci0=fz["diff_ci"][0], ci1=fz["diff_ci"][1],
           aucbar=ref["move_auc"] + 0.05, fmbar=ref["false_move"])

BODY = """
## The family

v1 draws six earlier fire-days from other incidents in the same acres, containment, and personnel band. A band says nothing about whether the count is moving, which is the paper's diagnosis: the drawn set implies movement on days that hold flat. Family A keeps the band and adds the item's recent personnel dynamics, which the report already shows the model (`personnel_last_days`, the last four counts ending today), matched against the same dynamics on the pool rows. The exploration on pool rows dated before 2015 (`explore.py`) shows why this should help: the probability that the count moves by more than a tenth the next day is 0.36 after a day on which it held exactly, 0.42 after a change within a tenth, 0.61 after a fall of more than a tenth, and 0.63 after a rise of more than a quarter; after two days of exact hold it is 0.24 against 0.45 after one. The next-day median ratio also follows the sign of the last change (0.88 after a fall of a tenth to a quarter, 1.04 after a rise of more than a quarter, 1.00 after a hold). Acres growth carries no movement signal in the pool (0.50 against 0.50), and containment change (0.56 when containment rose, 0.45 when it held) cannot be matched because the item context carries no containment history, only today's percent contained.

Every variant is a ladder over a tuple of components, most specific first. If fewer than three rows match the full tuple, the last component is dropped and the match is retried, down to the v1 band alone, so coverage stays at v1's 0.99. Six rows are then sampled at random with the item rng, as v1 does. The components:

- `band`: v1's (acres band, containment band, personnel band), always first.
- `p3`, `p5`: the one-day personnel change, today against the day before (item: `personnel_last_days[-2]` to `personnel_today`; pool row: `prev` to `today`), three ways (fell by more than a tenth, held within a tenth, rose by more than a tenth) or five ways (the fall and the rise each split at a quarter).
- `p4x`, `p6x`: as `p3` and `p5` with the exact hold (count unchanged) split off from a small change.
- `p5s`, `p7s`: as `p4x` and `p6x` with the exact hold split by whether it has lasted two days.
- `t3`, `t5`: the two-day change, today against two days before (item: `personnel_last_days[-3]`; pool row: the `prev` of the same incident's row for the day before, looked up among the candidate rows the harness already filtered), three or five ways with the same edges.
- `a`: whether acres grew over the last day (item: `acres_last_days[-2]` to `[-1]`, falling back to `new_acres > 0`; pool row: `acres_prev` to `acres`, same fallback).

The "fill" variants take every row at the most specific level and top up from the next levels until six are drawn; the "min6" variants relax when fewer than six rows match instead of three. Nothing is fitted: the edges are a tenth (the harness's own definition of a moving day) and a quarter, fixed before any run.

## Every variant

Numbers from `run-output.txt` (the harness table; `results.json` holds the same rows with full precision). "bars" says whether the variant clears both bars of the selection rule (movement AUC at least v1 + 0.05 = {aucbar:.3f} and false move below v1's {fmbar:.2f}), only lowers nMAE below v1's {rnmae:.3f}, or neither. Coverage with at least three analogues is 0.99 for every variant because the ladder ends at the v1 band.

{table}

What the table says, in the order the variants were tried:

1. The one-day change alone (`A-b3-p` to `A-b6x-p`) lowers nMAE by 0.006 to 0.010 with intervals that include zero, and the finer five-way split raises the movement AUC the most (0.611 for `A-b5-p`, the highest in the family). The AUC gain comes from the size of the last change, which the three-way split blurs.
2. Acres growth as a further level (`-pa`) hurts: it splits the cells without adding signal, and for the three- and five-way bucketings it raises the stable-day error from 0.11 to 0.13. Adding acres first (`A-b4x-ap`) is no better than leaving it out.
3. The two-day hold streak (`p5s`, `p7s`) and the coarsening ladders (`A-p3>5...`) change little against their one-day counterparts.
4. The two-day change as a second dynamics level is the step that matters: every `-t3` variant lowers nMAE to 0.201 to 0.208 with an interval that excludes zero, and the stable-day error falls from 0.11 to between 0.08 and 0.09, because a fire whose count has not changed in two days now draws analogues with the same flat history. On its own (`A-t3`) the two-day change lowers nMAE to 0.201 but leaves the AUC at 0.573, below the bar; the one-day five-way split is what carries the AUC over it.
5. Ordering: `A-b5-t3-p` and `A-b5-p-t3` match the same full key, so 523 of 599 items (87 percent) draw from identical sets. They differ only where the full key has fewer than three rows: the frozen ordering falls back to the two-day three-way cell (larger, more stable), the other to the one-day five-way cell (smaller). The 46 items served at that level account for the whole 0.009 nMAE gap between the two.
6. Fill and min-six thresholds move nMAE by at most 0.004 either way; they lower the false-move rate slightly (0.24) at the cost of AUC or nMAE. None beats the plain ladder on the selection criterion.

## The frozen choice and why

The prespecified rule picks, among the variants clearing both bars, the one with the lowest development nMAE. {nclear} variants clear both bars; the six lowest are {clear_txt}. No other clearing variant is within the 0.002 tie band of the leader, so the tie-break does not arise; every variant in the family has zero fitted parameters in any case. `A-b5-t3-p` is therefore frozen: ladder band+t3+p5, then band+t3, then band; relax when fewer than three rows match; sample six with the item rng.

Its exact development numbers (`results.json`): nMAE {nmae4:.4f} against v1's {rnmae4:.4f}, difference {diff4:+.4f} with the paired incident-cluster bootstrap interval [{ci04:+.4f}, {ci14:+.4f}]; movement AUC {auc4:.4f} against the bar of {aucbar4:.4f}; false move {fm4:.4f} against the bar of {fmbar4:.4f}; coverage with at least three analogues {cov34:.4f} (v1 {rcov34:.4f}; the same items lack a band match in both), mean rows drawn {drawn:.2f}; movement Brier {brier4:.4f}; beats persistence {beats:.2f}; within 25 percent {w254:.3f}. The diagnostics (`diag-output.txt`) show where the gain sits: on items whose count held within a tenth over the last day (305 of 599), the stable-day error falls from 0.112 to 0.061; on items whose count fell by a tenth to a quarter, nMAE falls from 0.200 to 0.170 because the drawn analogues now continue the fall. The one group that gets worse is the 29 stable days after a rise of more than a quarter (stable-day error 0.180 against 0.140), where the matched analogues keep rising and the item does not. The ladder served 523 items at the full key, 46 at band+t3, 26 at the band alone, and 4 with no analogue.

The rule's own median-ratio IQR is [{iq0:.2f}, {iq1:.2f}] against the actual [{aq0:.2f}, {aq1:.2f}] and v1's [{riq0:.2f}, {riq1:.2f}]: it moves the analogue prediction toward the falls that dominate the development set and, through the flat-history match, stops predicting movement on days that hold.

## What information the frozen rule uses

Item side, from `context` only: `personnel_last_days` (the last three of the four counts), `personnel_today`, `acres`, `percent_contained` (the v1 band). Pool side: `key`, `today`, `prev`, and the `prev` of the same incident's row for the day before, taken from the candidate rows the harness passes, which already exclude the item's incident and any row whose outcome day is on or after the report day; that earlier row is always at least two days before the report day. `acres_last_days`, `new_acres`, and `acres_prev` are read by the `a` component only, which the frozen rule does not include. No fitted constants: HOLD = 0.10 and BIG = 0.25 are fixed in `rule.py`. Deterministic given the item rng: the candidate rows are grouped by key in first-appearance order and sampled with `rng.sample`, as v1 does. Runtime is 14 seconds for the 599 development items including the harness's own candidate filter, of which the rule accounts for about 25 milliseconds per item. The rule caches derived features on the pool row dicts under keys beginning with an underscore (`_dyn`, `_dyn2`, `_prev2`, `_kA:...`); the harness reads none of them.

## What would break it

- Selection optimism. Forty variants were scored on 599 items and the frozen one is the minimum; its paired interval is computed for that one variant after the fact and does not account for the search. The direction of the gain is consistent across the whole `-t3` group (nine variants, all with intervals excluding zero), so the effect is unlikely to be a lucky draw, but the test gain should be expected to be smaller than 0.028.
- The AUC margin is thin. {auc4:.4f} clears the {aucbar4:.4f} bar by {margin:.4f}. The AUC is computed from the share of six drawn rows whose ratio moved, a coarse statistic; a different seed or a different pool would move it either way. The false-move margin ({fm:.2f} against {rfm:.2f}) is wide.
- The development set moves more often than the test set (0.53 against 0.39 of days). The frozen rule's gains split between moving days (continuing a fall) and stable days (matching flat histories); on the test set the stable-day half should count for more and the moving-day half for less, and persistence itself is a stronger baseline there (0.146 against the development 0.210).
- The advantage over `A-b5-p-t3` rests on 46 items served at the relaxed level. If the judge prefers the ordering that keeps the one-day change longer, `A-b5-p-t3` (nMAE 0.207, AUC 0.600, false move 0.25) also clears both bars, and `A-b5-p-t3-min6` (0.204, 0.602, 0.24) is the safest on the AUC bar.
- Items whose history has fewer than three counts would land in the "unknown" two-day bucket, which matches pool rows on the first or second day of their run, a population that moves more often (0.59). Every development and evaluation item carries four counts, so this cannot happen in the harness, but a different item builder could trigger it.
- A one-day count change caused by a corrected filing rather than a real staffing change is read as a move and pulls analogues that continue it.
- The prompt still introduces the block as analogues "in the same size, containment, and staffing band". The brief fixes the prompt, so the rule cannot say that the analogues also share the recent trend; the paper's description of the rule must say so instead.

## Files

- `rule.py`: `RULES` (40 variants), `FROZEN = "A-b5-t3-p"`, the component and ladder code. Imports `K` and `key_of` from `dev_eval`.
- `run.py`: builds the pool once, evaluates v1 and every variant, prints the harness table and the selection view, writes `results.json`. Output of the final run in `run-output.txt` (about 10 minutes for 40 variants).
- `diag.py [variant]`: ladder-level usage and per-bucket errors for one variant; `diag-output.txt` holds the frozen rule's.
- `explore.py`: the pre-2015 pool exploration behind the design; `table.py` renders the table above from `results.json`; `mkreport.py` writes this file.
""".format(aucbar=ref["move_auc"] + 0.05, fmbar=ref["false_move"], rnmae=ref["nmae"], table=table, nclear=len(clear),
           clear_txt=clear_txt, nmae4=fz["nmae"], rnmae4=ref["nmae"], diff4=fz["diff_vs_reference"], ci04=fz["diff_ci"][0],
           ci14=fz["diff_ci"][1], auc4=fz["move_auc"], aucbar4=ref["move_auc"] + 0.05, fm4=fz["false_move"],
           fmbar4=ref["false_move"], cov34=fz["coverage_3"], rcov34=ref["coverage_3"], drawn=fz["mean_drawn"],
           brier4=fz["move_brier"], beats=fz["beats_persistence"], w254=fz["within_25"], iq0=fz["median_ratio_iqr"][0],
           iq1=fz["median_ratio_iqr"][1], aq0=fz["actual_ratio_iqr"][0], aq1=fz["actual_ratio_iqr"][1],
           riq0=ref["median_ratio_iqr"][0], riq1=ref["median_ratio_iqr"][1], margin=fz["move_auc"] - ref["move_auc"] - 0.05,
           fm=fz["false_move"], rfm=ref["false_move"])

(HERE / "report.md").write_text(HEAD + BODY, encoding="utf-8")
print("wrote", HERE / "report.md", len(HEAD + BODY), "chars")
