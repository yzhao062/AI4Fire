# Design B: nearest-neighbour analogue retrieval

## The family

Every variant keeps the v1 pool, the v1 eligibility filter (other incidents only, outcome day before the
report day), six analogues, and the same displayed fields. What changes is which six. Instead of a random
draw from the band, the rule ranks the eligible rows by a weighted Euclidean distance over seven standardized
features that are known on the report day and takes the six nearest. Ties are broken by `analogue_id`
ascending. A nearest-neighbour draw has no random step, so the rule never touches the `rng` the harness
passes; `run.py` checks that the frozen rule returns identical draws under three different seeds.

The seven features, item side and pool side:

| # | feature | item | pool row |
|---|---------|------|----------|
| 0 | pers | log(max(personnel_today, 1)) | log(max(today, 1)) |
| 1 | acres | log1p(acres) | log1p(acres) |
| 2 | pct | percent_contained clipped to 0..100 | pct clipped to 0..100 |
| 3 | day | log(day_of_run) | log(day_of_run) |
| 4 | change | log(personnel_today / personnel_last_days[-2]) | log(max(today, 1) / max(prev, 1)) |
| 5 | new | log1p(new_acres) | log1p(new_acres) |
| 6 | aerial | log1p(aerial_resources) | log1p(aerial) |

Each feature is standardized as (x - mean) / std. The fourteen constants were fitted once by `fit_scales.py`
on the 54,566 pool rows dated before 2015-01-01 and are frozen in `scales.json`; `run.py` refits them from
the pool and asserts that the file matches to 1e-9 before scoring anything. Containment is clipped because
2 percent of pool rows carry values in the millions (v1 puts those in its top band, which is the same
treatment). A pool row that lacks a feature sits at the fitted mean (z = 0); this affects `change` on the
first day of a run (22 percent of rows, all with `day_of_run` = 1) and `pct` on 7 percent of rows. An item
that lacks a feature drops that feature from its distance (containment is missing on 8.5 percent of
development items and on no test item; every item has four days of personnel history, so `change` is
always present).

Three scopes restrict which rows are ranked: `band` is the v1 band exactly (same acres, containment, and
personnel band); `pers` is the same personnel band only; `all` is every eligible row. Four weightings
(order pers, acres, pct, day, change, new, aerial): `eq` = all ones; `chg` = change 4, new 3, others 1;
`dyn` = day 3, change 4, new 3, others 1; `sta` = pers 3, acres 2, pct 2, others 1. Four `cap2` variants
allow at most two rows from any one incident, taken in distance order. Two ablations say what does the
work: `nochg` is `eq` with the change feature removed; `band3` ranks on the three v1 band variables only.
Twenty variants in all, every one in `RULES`.

## Every variant on the 599 development items (harness output, `results.txt`)

```
rule                     cov3   nMAE  beats stable moving falsmv missmv  dirok    w25 mBrier   mAUC med-ratio IQR | actual IQR | diff vs ref [95%]
v1 (paper)               0.99  0.225   0.38  0.110  0.329   0.32   0.60   0.60   0.66  0.285  0.544 [0.94, 1.03] | [0.77, 1.03]
nn_band_eq               0.99  0.206   0.44  0.086  0.315   0.26   0.55   0.65   0.68  0.268  0.611 [0.89, 1.01] | [0.77, 1.03] | -0.019 [-0.041, -0.000]
nn_band_chg              0.99  0.203   0.42  0.093  0.302   0.27   0.58   0.64   0.68  0.266  0.608 [0.89, 1.01] | [0.77, 1.03] | -0.022 [-0.038, -0.007]
nn_band_dyn              0.99  0.201   0.39  0.092  0.300   0.29   0.61   0.60   0.68  0.270  0.601 [0.90, 1.00] | [0.77, 1.03] | -0.024 [-0.040, -0.009]
nn_band_sta              0.99  0.217   0.42  0.087  0.334   0.28   0.57   0.65   0.68  0.263  0.616 [0.90, 1.01] | [0.77, 1.03] | -0.008 [-0.031, +0.013]
nn_pers_eq               1.00  0.196   0.43  0.073  0.307   0.24   0.58   0.64   0.70  0.256  0.631 [0.90, 1.00] | [0.77, 1.03] | -0.029 [-0.049, -0.012]
nn_pers_chg              1.00  0.191   0.45  0.073  0.298   0.23   0.58   0.70   0.70  0.253  0.632 [0.91, 1.00] | [0.77, 1.03] | -0.034 [-0.055, -0.016]
nn_pers_dyn              1.00  0.196   0.43  0.077  0.303   0.25   0.57   0.67   0.69  0.257  0.625 [0.90, 1.01] | [0.77, 1.03] | -0.029 [-0.050, -0.012]
nn_pers_sta              1.00  0.205   0.43  0.086  0.313   0.27   0.55   0.64   0.68  0.259  0.629 [0.89, 1.01] | [0.77, 1.03] | -0.020 [-0.042, -0.000]
nn_all_eq                1.00  0.193   0.43  0.076  0.298   0.22   0.57   0.64   0.68  0.258  0.633 [0.90, 1.00] | [0.77, 1.03] | -0.032 [-0.054, -0.014]
nn_all_chg               1.00  0.195   0.45  0.081  0.298   0.25   0.57   0.68   0.70  0.255  0.634 [0.90, 1.00] | [0.77, 1.03] | -0.030 [-0.053, -0.011]
nn_all_dyn               1.00  0.193   0.44  0.072  0.301   0.24   0.55   0.67   0.68  0.252  0.640 [0.89, 1.01] | [0.77, 1.03] | -0.033 [-0.055, -0.014]
nn_all_sta               1.00  0.199   0.42  0.082  0.304   0.25   0.58   0.63   0.69  0.266  0.615 [0.90, 1.01] | [0.77, 1.03] | -0.026 [-0.048, -0.008]
nn_pers_chg_cap2         1.00  0.190   0.44  0.071  0.297   0.23   0.56   0.68   0.70  0.251  0.632 [0.90, 1.00] | [0.77, 1.03] | -0.035 [-0.056, -0.017]
nn_pers_dyn_cap2         1.00  0.197   0.42  0.078  0.303   0.28   0.56   0.66   0.69  0.259  0.612 [0.89, 1.01] | [0.77, 1.03] | -0.029 [-0.049, -0.011]
nn_all_chg_cap2          1.00  0.195   0.44  0.081  0.298   0.26   0.56   0.67   0.70  0.254  0.629 [0.89, 1.01] | [0.77, 1.03] | -0.030 [-0.053, -0.011]
nn_all_dyn_cap2          1.00  0.193   0.44  0.073  0.301   0.25   0.54   0.67   0.69  0.257  0.624 [0.89, 1.00] | [0.77, 1.03] | -0.032 [-0.055, -0.014]
nn_pers_nochg            1.00  0.206   0.40  0.086  0.315   0.26   0.55   0.62   0.68  0.270  0.591 [0.90, 1.01] | [0.77, 1.03] | -0.019 [-0.040, -0.001]
nn_pers_band3            1.00  0.229   0.38  0.118  0.329   0.33   0.58   0.64   0.64  0.286  0.546 [0.91, 1.03] | [0.77, 1.03] | +0.004 [-0.018, +0.023]
nn_all_nochg             1.00  0.197   0.44  0.085  0.298   0.25   0.56   0.65   0.69  0.270  0.598 [0.90, 1.01] | [0.77, 1.03] | -0.028 [-0.050, -0.010]
nn_all_band3             1.00  0.230   0.39  0.120  0.328   0.33   0.57   0.65   0.65  0.284  0.547 [0.91, 1.03] | [0.77, 1.03] | +0.004 [-0.017, +0.023]
```

One line per variant, in words:

- `nn_band_eq`: nearest six within the v1 band, equal weights. nMAE 0.206, AUC 0.611, false move 0.26.
- `nn_band_chg`: same scope, change-heavy weights. nMAE 0.203, AUC 0.608, false move 0.27.
- `nn_band_dyn`: same scope, change and day heavy. nMAE 0.201, AUC 0.601, false move 0.29.
- `nn_band_sta`: same scope, state-heavy weights. nMAE 0.217, AUC 0.616, false move 0.28; the interval against v1 spans zero.
- `nn_pers_eq`: personnel band only, equal weights. nMAE 0.196, AUC 0.631, false move 0.24.
- `nn_pers_chg` (frozen): personnel band only, change-heavy weights. nMAE 0.191, AUC 0.632, false move 0.23.
- `nn_pers_dyn`: personnel band only, change and day heavy. nMAE 0.196, AUC 0.625, false move 0.25.
- `nn_pers_sta`: personnel band only, state-heavy weights. nMAE 0.205, AUC 0.629, false move 0.27.
- `nn_all_eq`: whole candidate set, equal weights. nMAE 0.193, AUC 0.633, false move 0.22 (the lowest false-move rate).
- `nn_all_chg`: whole set, change-heavy. nMAE 0.195, AUC 0.634, false move 0.25.
- `nn_all_dyn`: whole set, change and day heavy. nMAE 0.193, AUC 0.640 (the highest AUC), false move 0.24.
- `nn_all_sta`: whole set, state-heavy. nMAE 0.199, AUC 0.615, false move 0.25.
- `nn_pers_chg_cap2`: the frozen rule with at most two rows per incident. nMAE 0.190 (the lowest), AUC 0.632, false move 0.23.
- `nn_pers_dyn_cap2`: nMAE 0.197, AUC 0.612, false move 0.28.
- `nn_all_chg_cap2`: nMAE 0.195, AUC 0.629, false move 0.26.
- `nn_all_dyn_cap2`: nMAE 0.193, AUC 0.624, false move 0.25.
- `nn_pers_nochg` (ablation): equal weights without the recent personnel change. nMAE 0.206, AUC 0.591, false move 0.26.
- `nn_pers_band3` (ablation): nearest on acres, containment, and personnel only. nMAE 0.229, AUC 0.546, false move 0.33: v1's numbers.
- `nn_all_nochg` (ablation): nMAE 0.197, AUC 0.598, false move 0.25.
- `nn_all_band3` (ablation): nMAE 0.230, AUC 0.547, false move 0.33: v1's numbers.

## The frozen choice: `nn_pers_chg`

The brief's selection rule, applied within this family (`run.py` prints the pick). Every one of the sixteen
non-ablation variants clears both bars: AUC at least 0.594 (v1's 0.544 plus 0.05) and a false-move rate below
v1's 0.32. Among them the lowest nMAE is `nn_pers_chg_cap2` at 0.1900, and `nn_pers_chg` at 0.1911 is inside
the 0.002 tie window; the tie goes to the simpler rule, which is the one without the per-incident cap. The
next variant, `nn_all_dyn` at 0.1925, is outside the window. So the frozen rule is: nearest six within the
item's personnel band, weights (1, 1, 1, 1, 4, 3, 1), no cap, ties by `analogue_id`.

Its development numbers, as the harness printed them: coverage-3 1.00, nMAE 0.191 (persistence 0.210, v1
0.225), beats persistence 0.45, stable-day error 0.073, moving-day error 0.298, false move 0.23, missed move
0.58, direction 0.70, within-25 0.70, movement Brier 0.253, movement AUC 0.632, median-ratio IQR
[0.91, 1.00] against the actual [0.77, 1.03]. Paired incident-cluster bootstrap against v1 on the difference
in normalized error: -0.034 [-0.055, -0.016]. It is the first rule in this table whose analogue-only
prediction beats persistence on development (0.191 against 0.210); v1's did not (0.225).

Two readings of "simpler" are possible. Every variant has the same fourteen fitted constants, so on a strict
count of fitted parameters the tie is unresolved and the lowest nMAE, `nn_pers_chg_cap2`, would win. The
difference between the two is 0.001 nMAE, and their AUC and false-move rates are identical to two decimals,
so the judge can take either without a re-run; `RULES["nn_pers_chg_cap2"]` is in the module. The cap version
has one property worth knowing for the model run: it guarantees the six lines come from at least three
incidents, where the frozen rule's draws span 4.85 distinct incidents on average.

What does the work, from the ablations. Ranking by distance on the three v1 band variables alone reproduces
v1 (nMAE 0.229 and 0.230, AUC 0.546 and 0.547), so nearest-instead-of-random adds nothing by itself. The
four added features carry the gain, and the recent personnel change carries about half of it: removing it
from the equal-weight rule moves nMAE from 0.196 to 0.206 and AUC from 0.631 to 0.591 in the personnel-band
scope. This is the mechanism the paper's finding predicts: a move yesterday raises the chance of a move
today from 0.39 to 0.59 in the pool, and v1's band ignores it. Weighting the change features more heavily
helps a little (0.196 to 0.191); weighting the state features more heavily hurts (0.205), because that
pulls the rule back toward v1. Widening the scope from the band to the personnel band helps every
weighting by about 0.01; widening further to the whole set changes little either way.

Where the gain sits: on the 526 development items with a report day before 2015 the frozen rule's nMAE is
0.191 against v1's 0.228 (-0.037); on the 73 items from 2015 or later, all California fires and the closest
proxy for the test period, it is 0.190 against 0.205 (-0.015). The smaller later-period gain rests on 73
items and is inside the paired interval, so it is a caution and not a contradiction. The false-move rate
drops from 0.32 to 0.23 and the stable-day error from 0.110 to 0.073, so most of the gain is the intended
one: the displayed median stops implying movement on days that hold flat (median-ratio IQR [0.91, 1.00]
against v1's [0.94, 1.03]). The missed-move rate barely changes (0.60 to 0.58): the rule says "flat" better
than it says "move".

## Information the frozen rule uses

From the item, all of it displayed in the bare prompt already: `personnel_today`, `personnel_last_days`
(only the last two entries, for the one-day change), `acres`, `percent_contained`, `new_acres`,
`aerial_resources`, and `day_of_run`. From each candidate row, the same seven quantities on the row's own
day (`today`, `prev`, `acres`, `pct`, `new_acres`, `aerial`, `day_of_run`); the row's `next` is never read by
the rule, only displayed afterwards. Nothing dated on or after the report day is touched: the harness's
`candidates()` filters the pool before the rule sees it, and the harness asserts eligibility on every drawn
row. The fitted constants come from pool rows dated before 2015-01-01, which precedes every test report day.

Constraint checklist from the brief: (1) six analogues, same fields, the rule only decides which six; (2)
report-day information only, as above; (3) fourteen standardization constants fitted on pre-2015 rows and
frozen in `scales.json`, verified against a refit on every run; (4) deterministic, and independent of `rng`
(checked under three seeds on 25 items, and redrawn in a fresh process without `prepare()` on 86 items with
identical results, `check_frozen.py`); (5) coverage-3 is 1.00 on development, above the 0.90 bar; (6) the
slowest variant takes 15 seconds for 599 items once the pool is built, harness loop included.

## What would break it

- The prompt's header line. The grounded block says the analogues come "from earlier incidents in the same
  size, containment, and staffing band". Under the frozen rule every drawn row is in the item's staffing
  band, but only 44 percent of drawn rows are in its full v1 band; size and containment are matched by
  distance instead. Each analogue line already prints its acres and containment, so the model can see the
  actual match, but the header sentence is no longer exactly true. Whether to reword it (for example, "from
  earlier incidents in the same staffing band, nearest in size, containment, day of incident, and recent
  change") is the coordinator's call; the brief fixes the format, so this is flagged, not changed.
- Shift in the raw fields. The scales were fitted on 1999 to 2014 reports. A test-period drift in how
  agencies file `TOTAL_AERIAL` or `NEW_ACRES` would move items away from the pool in those two dimensions
  and quietly reweight the distance. The rule still returns six rows, just less well matched.
- Data-entry glitches in the personnel history. The change feature reacts to the last two personnel
  values; a spurious value such as 564 to 5 (a real development item) makes the rule fetch other rows with
  the same glitch, whose medians then say "rebound". On that item the rebound was right (the filed count was
  117), but a glitch that does not rebound would be answered badly, where v1 would have shrugged.
- Missing history. If a future item were built with fewer than two entries in `personnel_last_days`, the
  rule silently drops the change feature and falls back to something close to `nn_pers_nochg` (nMAE 0.206).
  No development or test item has this; all carry four entries.
- Neighbour clustering. The six nearest rows come from 4.85 distinct incidents on average, and consecutive
  days of one fire can occupy two or three slots, which makes the displayed median rest on fewer independent
  outcomes than six. The `cap2` variant removes this at no cost on development (nMAE 0.190) and is the
  natural substitute if the judge prefers it.
- Selection on development. Twenty variants were scored on the same 599 items and the winner was picked by
  nMAE, so the frozen number is an optimistic estimate of the test-period gain by the usual amount for a
  pick among near-ties; the spread among the top eight variants is 0.006 nMAE, well inside the paired
  interval's half-width of 0.02, so the choice among them is not what carries the result. The ablations
  and the pre-2015 versus 2015-plus split are the evidence that the gain is the change feature, not the
  pick.

## Files

- `rule.py`: the family, `RULES` (twenty variants), `FROZEN = "nn_pers_chg"`, `prepare(pool)`.
- `scales.json`: the fourteen frozen constants; `fit_scales.py` regenerates them from pre-2015 pool rows.
- `run.py`: builds the pool once, verifies the frozen scales, scores v1 and every variant in one process,
  applies the selection rule, writes `results.txt`, `results.json` (with the frozen rule's draws per item),
  and `run-output.txt`.
- `check_frozen.py`: fresh-process reproduction of the frozen draws without `prepare()`.
- `explore.py`, `explore2.py`, `examples.py`, `show.py`: the structural checks and example draws quoted above.
