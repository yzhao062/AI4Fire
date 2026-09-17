# Design A: recent-dynamics matching

Frozen variant: `A-b5-t3-p` (`rule.FROZEN`). Development numbers as the harness printed them: cov3 0.99, nMAE 0.197 (v1 0.225, persistence 0.210), stable-day error 0.077 (v1 0.110), moving-day error 0.306 (v1 0.329), false move 0.25 (v1 0.32), missed move 0.60, direction 0.61, within-25 0.69, movement Brier 0.274, movement AUC 0.596 (v1 0.544), median-ratio IQR [0.91, 1.01] against actual [0.77, 1.03], paired difference in nMAE against v1 -0.028 [-0.049, -0.010]. It clears both bars of the prespecified rule (AUC bar 0.594, false-move bar 0.32) and has the lowest development nMAE among the variants that do.

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

Numbers from `run-output.txt` (the harness table; `results.json` holds the same rows with full precision). "bars" says whether the variant clears both bars of the selection rule (movement AUC at least v1 + 0.05 = 0.594 and false move below v1's 0.32), only lowers nMAE below v1's 0.225, or neither. Coverage with at least three analogues is 0.99 for every variant because the ladder ends at the v1 band.

| variant | ladder (most specific first) | draw | cov3 | nMAE | stable | moving | false move | missed move | mAUC | Brier | diff vs v1 [95%] | bars |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v1 (paper) | band | sample 6 | 0.99 | 0.225 | 0.110 | 0.329 | 0.32 | 0.60 | 0.544 | 0.285 | reference | reference |
| A-b3-p | band+p3 > band | relax if < 3 | 0.99 | 0.215 | 0.112 | 0.308 | 0.30 | 0.60 | 0.575 | 0.281 | -0.010 [-0.026, +0.006] | nMAE only |
| A-b5-p | band+p5 > band | relax if < 3 | 0.99 | 0.217 | 0.106 | 0.316 | 0.29 | 0.58 | 0.611 | 0.266 | -0.009 [-0.026, +0.008] | both |
| A-b4x-p | band+p4x > band | relax if < 3 | 0.99 | 0.218 | 0.111 | 0.314 | 0.29 | 0.60 | 0.564 | 0.289 | -0.007 [-0.021, +0.008] | nMAE only |
| A-b6x-p | band+p6x > band | relax if < 3 | 0.99 | 0.219 | 0.105 | 0.322 | 0.27 | 0.59 | 0.602 | 0.272 | -0.006 [-0.022, +0.009] | both |
| A-b3-pa | band+p3+a > band+p3 > band | relax if < 3 | 0.99 | 0.233 | 0.135 | 0.321 | 0.32 | 0.61 | 0.562 | 0.284 | +0.008 [-0.009, +0.026] | none |
| A-b5-pa | band+p5+a > band+p5 > band | relax if < 3 | 0.99 | 0.233 | 0.128 | 0.327 | 0.29 | 0.59 | 0.587 | 0.277 | +0.008 [-0.010, +0.027] | none |
| A-b4x-pa | band+p4x+a > band+p4x > band | relax if < 3 | 0.99 | 0.221 | 0.104 | 0.326 | 0.27 | 0.57 | 0.580 | 0.277 | -0.004 [-0.018, +0.010] | nMAE only |
| A-b6x-pa | band+p6x+a > band+p6x > band | relax if < 3 | 0.99 | 0.222 | 0.098 | 0.333 | 0.25 | 0.57 | 0.603 | 0.270 | -0.003 [-0.018, +0.011] | both |
| A-b4x-ap | band+a+p4x > band+a > band | relax if < 3 | 0.99 | 0.217 | 0.099 | 0.324 | 0.26 | 0.58 | 0.586 | 0.273 | -0.008 [-0.022, +0.006] | nMAE only |
| A-b5s-p | band+p5s > band | relax if < 3 | 0.99 | 0.217 | 0.109 | 0.314 | 0.29 | 0.60 | 0.564 | 0.290 | -0.009 [-0.022, +0.005] | nMAE only |
| A-b5s-pa | band+p5s+a > band+p5s > band | relax if < 3 | 0.99 | 0.222 | 0.106 | 0.327 | 0.27 | 0.59 | 0.577 | 0.277 | -0.003 [-0.018, +0.012] | nMAE only |
| A-b7s-p | band+p7s > band | relax if < 3 | 0.99 | 0.217 | 0.102 | 0.321 | 0.28 | 0.60 | 0.600 | 0.274 | -0.008 [-0.023, +0.008] | both |
| A-b7s-pa | band+p7s+a > band+p7s > band | relax if < 3 | 0.99 | 0.222 | 0.099 | 0.334 | 0.25 | 0.58 | 0.599 | 0.271 | -0.003 [-0.019, +0.013] | both |
| A-b5s-pa-min6 | band+p5s+a > band+p5s > band | relax if < 6 | 0.99 | 0.220 | 0.104 | 0.324 | 0.28 | 0.60 | 0.571 | 0.278 | -0.006 [-0.020, +0.008] | nMAE only |
| A-b5s-pa-fill | band+p5s+a > band+p5s > band | fill to 6 | 0.99 | 0.219 | 0.104 | 0.322 | 0.27 | 0.58 | 0.579 | 0.273 | -0.006 [-0.020, +0.008] | nMAE only |
| A-b5s-p-min6 | band+p5s > band | relax if < 6 | 0.99 | 0.217 | 0.110 | 0.314 | 0.29 | 0.60 | 0.566 | 0.288 | -0.008 [-0.022, +0.006] | nMAE only |
| A-b5s-p-fill | band+p5s > band | fill to 6 | 0.99 | 0.218 | 0.112 | 0.314 | 0.30 | 0.59 | 0.563 | 0.291 | -0.007 [-0.021, +0.007] | nMAE only |
| A-p3\>5 | band+p3+p5 > band+p3 > band | relax if < 3 | 0.99 | 0.217 | 0.108 | 0.315 | 0.30 | 0.59 | 0.604 | 0.269 | -0.008 [-0.025, +0.008] | both |
| A-p3\>5\>6x | band+p3+p5+p6x > band+p3+p5 > band+p3 > band | relax if < 3 | 0.99 | 0.222 | 0.111 | 0.321 | 0.29 | 0.59 | 0.593 | 0.277 | -0.003 [-0.020, +0.014] | nMAE only |
| A-p3\>5\>6x\>7s | band+p3+p5+p6x+p7s > band+p3+p5+p6x > band+p3+p5 > band+p3 > band | relax if < 3 | 0.99 | 0.220 | 0.109 | 0.321 | 0.29 | 0.60 | 0.593 | 0.279 | -0.005 [-0.022, +0.012] | nMAE only |
| A-b5-p-t3 | band+p5+t3 > band+p5 > band | relax if < 3 | 0.99 | 0.207 | 0.091 | 0.311 | 0.25 | 0.60 | 0.600 | 0.273 | -0.019 [-0.034, -0.004] | both |
| A-p3\>5-t3 | band+p3+p5+t3 > band+p3+p5 > band+p3 > band | relax if < 3 | 0.99 | 0.207 | 0.093 | 0.309 | 0.26 | 0.60 | 0.592 | 0.277 | -0.019 [-0.033, -0.004] | nMAE only |
| A-b3-p-t3 | band+p3+t3 > band+p3 > band | relax if < 3 | 0.99 | 0.208 | 0.099 | 0.306 | 0.30 | 0.57 | 0.590 | 0.275 | -0.017 [-0.032, -0.003] | nMAE only |
| A-b5-p-fill | band+p5 > band | fill to 6 | 0.99 | 0.216 | 0.106 | 0.316 | 0.29 | 0.59 | 0.608 | 0.267 | -0.009 [-0.025, +0.006] | both |
| A-b5-p-min6 | band+p5 > band | relax if < 6 | 0.99 | 0.216 | 0.102 | 0.318 | 0.29 | 0.58 | 0.613 | 0.264 | -0.009 [-0.026, +0.006] | both |
| A-p3\>5-fill | band+p3+p5 > band+p3 > band | fill to 6 | 0.99 | 0.213 | 0.103 | 0.313 | 0.30 | 0.58 | 0.603 | 0.269 | -0.012 [-0.028, +0.003] | both |
| A-p3\>5-min6 | band+p3+p5 > band+p3 > band | relax if < 6 | 0.99 | 0.214 | 0.104 | 0.314 | 0.29 | 0.59 | 0.605 | 0.267 | -0.011 [-0.027, +0.004] | both |
| A-b5-p-t5 | band+p5+t5 > band+p5 > band | relax if < 3 | 0.99 | 0.206 | 0.097 | 0.305 | 0.27 | 0.58 | 0.593 | 0.277 | -0.019 [-0.034, -0.004] | nMAE only |
| **A-b5-t3-p** (frozen) | band+t3+p5 > band+t3 > band | relax if < 3 | 0.99 | 0.197 | 0.077 | 0.306 | 0.25 | 0.60 | 0.596 | 0.274 | -0.028 [-0.049, -0.010] | both |
| A-b6x-p-t3 | band+p6x+t3 > band+p6x > band | relax if < 3 | 0.99 | 0.202 | 0.082 | 0.311 | 0.26 | 0.59 | 0.590 | 0.277 | -0.023 [-0.040, -0.007] | nMAE only |
| A-b5-p-t3-a | band+p5+t3+a > band+p5+t3 > band+p5 > band | relax if < 3 | 0.99 | 0.212 | 0.095 | 0.317 | 0.26 | 0.63 | 0.578 | 0.282 | -0.013 [-0.029, +0.002] | nMAE only |
| A-b5-p-t3-fill | band+p5+t3 > band+p5 > band | fill to 6 | 0.99 | 0.205 | 0.086 | 0.311 | 0.24 | 0.59 | 0.599 | 0.273 | -0.020 [-0.034, -0.007] | both |
| A-b5-p-t3-min6 | band+p5+t3 > band+p5 > band | relax if < 6 | 0.99 | 0.204 | 0.080 | 0.317 | 0.24 | 0.60 | 0.602 | 0.273 | -0.021 [-0.038, -0.005] | both |
| A-t3 | band+t3 > band | relax if < 3 | 0.99 | 0.201 | 0.087 | 0.304 | 0.29 | 0.60 | 0.573 | 0.280 | -0.024 [-0.045, -0.007] | nMAE only |
| A-t5 | band+t5 > band | relax if < 3 | 0.99 | 0.209 | 0.098 | 0.310 | 0.33 | 0.57 | 0.566 | 0.287 | -0.016 [-0.031, -0.002] | nMAE only |
| A-t3-p3 | band+t3+p3 > band+t3 > band | relax if < 3 | 0.99 | 0.201 | 0.085 | 0.306 | 0.29 | 0.57 | 0.585 | 0.278 | -0.024 [-0.046, -0.006] | nMAE only |
| A-t5-p5 | band+t5+p5 > band+t5 > band | relax if < 3 | 0.99 | 0.203 | 0.090 | 0.305 | 0.29 | 0.59 | 0.570 | 0.289 | -0.022 [-0.041, -0.005] | nMAE only |
| A-b5-t3-p-fill | band+t3+p5 > band+t3 > band | fill to 6 | 0.99 | 0.201 | 0.076 | 0.314 | 0.24 | 0.59 | 0.595 | 0.274 | -0.024 [-0.045, -0.008] | both |
| A-b5-t3-p-min6 | band+t3+p5 > band+t3 > band | relax if < 6 | 0.99 | 0.204 | 0.085 | 0.310 | 0.25 | 0.59 | 0.595 | 0.276 | -0.022 [-0.039, -0.006] | both |

What the table says, in the order the variants were tried:

1. The one-day change alone (`A-b3-p` to `A-b6x-p`) lowers nMAE by 0.006 to 0.010 with intervals that include zero, and the finer five-way split raises the movement AUC the most (0.611 for `A-b5-p`, the highest in the family). The AUC gain comes from the size of the last change, which the three-way split blurs.
2. Acres growth as a further level (`-pa`) hurts: it splits the cells without adding signal, and for the three- and five-way bucketings it raises the stable-day error from 0.11 to 0.13. Adding acres first (`A-b4x-ap`) is no better than leaving it out.
3. The two-day hold streak (`p5s`, `p7s`) and the coarsening ladders (`A-p3>5...`) change little against their one-day counterparts.
4. The two-day change as a second dynamics level is the step that matters: every `-t3` variant lowers nMAE to 0.201 to 0.208 with an interval that excludes zero, and the stable-day error falls from 0.11 to between 0.08 and 0.09, because a fire whose count has not changed in two days now draws analogues with the same flat history. On its own (`A-t3`) the two-day change lowers nMAE to 0.201 but leaves the AUC at 0.573, below the bar; the one-day five-way split is what carries the AUC over it.
5. Ordering: `A-b5-t3-p` and `A-b5-p-t3` match the same full key, so 523 of 599 items (87 percent) draw from identical sets. They differ only where the full key has fewer than three rows: the frozen ordering falls back to the two-day three-way cell (larger, more stable), the other to the one-day five-way cell (smaller). The 46 items served at that level account for the whole 0.009 nMAE gap between the two.
6. Fill and min-six thresholds move nMAE by at most 0.004 either way; they lower the false-move rate slightly (0.24) at the cost of AUC or nMAE. None beats the plain ladder on the selection criterion.

## The frozen choice and why

The prespecified rule picks, among the variants clearing both bars, the one with the lowest development nMAE. 16 variants clear both bars; the six lowest are `A-b5-t3-p` 0.1975, `A-b5-t3-p-fill` 0.2010, `A-b5-t3-p-min6` 0.2036, `A-b5-p-t3-min6` 0.2042, `A-b5-p-t3-fill` 0.2047, `A-b5-p-t3` 0.2066. No other clearing variant is within the 0.002 tie band of the leader, so the tie-break does not arise; every variant in the family has zero fitted parameters in any case. `A-b5-t3-p` is therefore frozen: ladder band+t3+p5, then band+t3, then band; relax when fewer than three rows match; sample six with the item rng.

Its exact development numbers (`results.json`): nMAE 0.1975 against v1's 0.2252, difference -0.0277 with the paired incident-cluster bootstrap interval [-0.0494, -0.0103]; movement AUC 0.5964 against the bar of 0.5936; false move 0.2465 against the bar of 0.3169; coverage with at least three analogues 0.9866 (v1 0.9866; the same items lack a band match in both), mean rows drawn 5.71; movement Brier 0.2744; beats persistence 0.41; within 25 percent 0.686. The diagnostics (`diag-output.txt`) show where the gain sits: on items whose count held within a tenth over the last day (305 of 599), the stable-day error falls from 0.112 to 0.061; on items whose count fell by a tenth to a quarter, nMAE falls from 0.200 to 0.170 because the drawn analogues now continue the fall. The one group that gets worse is the 29 stable days after a rise of more than a quarter (stable-day error 0.180 against 0.140), where the matched analogues keep rising and the item does not. The ladder served 523 items at the full key, 46 at band+t3, 26 at the band alone, and 4 with no analogue.

The rule's own median-ratio IQR is [0.91, 1.01] against the actual [0.77, 1.03] and v1's [0.94, 1.03]: it moves the analogue prediction toward the falls that dominate the development set and, through the flat-history match, stops predicting movement on days that hold.

## What information the frozen rule uses

Item side, from `context` only: `personnel_last_days` (the last three of the four counts), `personnel_today`, `acres`, `percent_contained` (the v1 band). Pool side: `key`, `today`, `prev`, and the `prev` of the same incident's row for the day before, taken from the candidate rows the harness passes, which already exclude the item's incident and any row whose outcome day is on or after the report day; that earlier row is always at least two days before the report day. `acres_last_days`, `new_acres`, and `acres_prev` are read by the `a` component only, which the frozen rule does not include. No fitted constants: HOLD = 0.10 and BIG = 0.25 are fixed in `rule.py`. Deterministic given the item rng: the candidate rows are grouped by key in first-appearance order and sampled with `rng.sample`, as v1 does. Runtime is 14 seconds for the 599 development items including the harness's own candidate filter, of which the rule accounts for about 25 milliseconds per item. The rule caches derived features on the pool row dicts under keys beginning with an underscore (`_dyn`, `_dyn2`, `_prev2`, `_kA:...`); the harness reads none of them.

## What would break it

- Selection optimism. Forty variants were scored on 599 items and the frozen one is the minimum; its paired interval is computed for that one variant after the fact and does not account for the search. The direction of the gain is consistent across the whole `-t3` group (nine variants, all with intervals excluding zero), so the effect is unlikely to be a lucky draw, but the test gain should be expected to be smaller than 0.028.
- The AUC margin is thin. 0.5964 clears the 0.5936 bar by 0.0028. The AUC is computed from the share of six drawn rows whose ratio moved, a coarse statistic; a different seed or a different pool would move it either way. The false-move margin (0.25 against 0.32) is wide.
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
