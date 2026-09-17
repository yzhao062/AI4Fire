# Design C: stability-aware stratified draw

## Family

The paper's diagnosis is that a set matched on acres, containment, and staffing is not matched on the probability that the count moves at all. Family C targets that directly. For each item it estimates P(move) from the report-day context (p_hat), where a move is a next-day count that changes by more than a tenth, and then draws the six analogues so that the drawn set carries that probability: n_move = round(6 * p_hat) rows whose own next-day ratio moved, and 6 - n_move rows that held. Both strata come from the v1 band (same acres, containment, and personnel band) and are topped up from the personnel band when a stratum is thin. Each stratum is sampled with the item rng, and the six rows are shuffled with it. The prompt, the displayed fields, and the six-row budget do not change.

The analogue-only prediction is persistence times the median drawn ratio. With three or fewer moved rows in a consistent direction the median stays inside the held rows, so the prediction holds; with four or more it lands among the moved rows. The drawn set therefore behaves like a discretised conditional median: it moves only when the context makes a move in one direction more likely than not, and the displayed share of moving analogues is the estimate of P(move) itself, which is what the harness's movement AUC scores.

Three sources of p_hat were tried, each with three ways of handling the direction of the moved rows:

- p_hat sources: a logistic model on seven report-day quantities (`model`), a table of movement rates by day-of-run bucket and containment band (`phase`), and a gradient-boosting classifier on the same quantities (`hgb`).
- direction: moved rows drawn regardless of direction (`mixed`, the plain variant name), moved rows matched to the sign of the item's most recent personnel change (`-trend`), or moved rows split into round(n_move * p_up) up-moves and the rest down-moves, with p_up from a second logistic model of P(up | move) (`-dir`).

## Fitted components

All fits use pool rows dated before 2015-01-01 that have a previous day (day_of_run >= 2) and belong to a run of at least ten reports, the population the items are built from: 25,070 rows from 1,316 incidents, move rate 0.462, up-share among moves 0.428. The constants are written to `model.json` by `fit.py`; the two logistic models are stored as intercept plus 17 raw-unit weights each (standardised during fitting, then folded back). The phase table has 54 cells with a pseudo-count of 20 toward the global rate. The gradient-boosting model is pickled to `hgb.pkl`; the frozen variant does not use it.

Features of the logistic models, all from the report day: log personnel, log day of run, the recent change log(today / prev) clipped to [-1.5, 1.5], its absolute value, an indicator for an exactly flat previous day, log1p new acres, log1p acres, log1p aerial resources (each with a missing indicator), and the five containment bands plus unknown as indicators. The item's `prev` is the second-to-last entry of `personnel_last_days`, which ends with today's count.

The models' own AUC for P(move) on the 599 development items (dev move rate 0.528):

| fit | logistic | phase | hgb | direction AUC given move | direction accuracy given move |
|---|---|---|---|---|---|
| frozen (all pre-2015 rows) | 0.647 | 0.601 | 0.718 | 0.814 | 0.766 |
| refit leaving the development incidents out | 0.647 | 0.591 | 0.696 | 0.813 | 0.769 |
| grouped 5-fold CV on the training rows | 0.654 | 0.615 | 0.693 | 0.851 | 0.786 |

The development incidents are in the pool, so the frozen fit has seen the development items' own rows; the refit row shows that this costs the logistic model nothing and the gradient-boosting model about 0.02 of AUC. Movement is driven by the size of yesterday's change (weight +0.94 on |recent change|), by day of run (-0.34 on its log), by crew size (+0.22 on log personnel), by aerial resources (-0.22), and by containment in a U shape (moves are more likely below 10 percent and above 90 percent contained). Direction is driven by containment (below 30 percent contained pushes up, above 60 percent pushes down) and by the sign of yesterday's change (+0.52). Training on all pre-2015 rows without the run-length filter gives development AUCs of 0.641 (logistic), 0.586 (phase), and 0.711 (hgb); the filter was chosen because it matches the item population, and it changes nothing that follows.

## Every variant on development (harness output, `run.py`)

| rule | cov3 | nMAE | beats | stable | moving | false move | missed move | dir ok | w25 | mBrier | mAUC | median-ratio IQR | diff vs v1 [95%] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v1 (paper) | 0.99 | 0.225 | 0.38 | 0.110 | 0.329 | 0.32 | 0.60 | 0.60 | 0.66 | 0.285 | 0.544 | [0.94, 1.03] | |
| C-model | 1.00 | 0.207 | 0.39 | 0.075 | 0.326 | 0.23 | 0.64 | 0.57 | 0.67 | 0.241 | 0.618 | [0.92, 1.01] | -0.018 [-0.041, +0.000] |
| C-model-trend | 1.00 | 0.216 | 0.40 | 0.097 | 0.322 | 0.35 | 0.52 | 0.65 | 0.68 | 0.241 | 0.618 | [0.90, 1.07] | -0.010 [-0.033, +0.012] |
| **C-model-dir** | 1.00 | **0.193** | 0.45 | 0.056 | 0.316 | **0.17** | 0.65 | 0.64 | 0.70 | 0.241 | **0.618** | [0.92, 1.01] | **-0.032 [-0.054, -0.014]** |
| C-phase | 1.00 | 0.203 | 0.41 | 0.059 | 0.332 | 0.19 | 0.71 | 0.58 | 0.68 | 0.248 | 0.577 | [0.93, 1.01] | -0.023 [-0.044, -0.004] |
| C-phase-trend | 1.00 | 0.210 | 0.43 | 0.074 | 0.332 | 0.30 | 0.62 | 0.65 | 0.68 | 0.248 | 0.577 | [0.91, 1.06] | -0.015 [-0.037, +0.003] |
| C-phase-dir | 1.00 | 0.195 | 0.45 | 0.046 | 0.330 | 0.15 | 0.76 | 0.63 | 0.69 | 0.248 | 0.577 | [0.94, 1.01] | -0.030 [-0.051, -0.013] |
| C-hgb | 1.00 | 0.199 | 0.40 | 0.069 | 0.316 | 0.18 | 0.60 | 0.59 | 0.69 | 0.215 | 0.714 | [0.92, 1.01] | -0.026 [-0.049, -0.006] |
| C-hgb-dir | 1.00 | 0.192 | 0.46 | 0.063 | 0.308 | 0.16 | 0.64 | 0.64 | 0.70 | 0.215 | 0.714 | [0.93, 1.01] | -0.033 [-0.057, -0.014] |

Persistence scores nMAE 0.210 on these items; the actual-ratio IQR is [0.77, 1.03] for every row. Every variant covers at least three analogues on 100 percent of items and runs in 7 to 19 seconds for the 599 items once the pool is built.

One line per variant:

- C-model: logistic p_hat, direction-blind moved rows. nMAE 0.207, mAUC 0.618, false move 0.23. Clears both bars; the mixed directions cancel in the median, so it mostly holds.
- C-model-trend: logistic p_hat, moved rows in the direction of yesterday's change. nMAE 0.216, mAUC 0.618, false move 0.35. Fails the false-move bar: the trend is right on 77 percent of down days but only 63 percent of up days, and it forces a consistent direction whenever n_move >= 3.
- C-model-dir: logistic p_hat, moved rows split by the fitted P(up | move). nMAE 0.193, mAUC 0.618, false move 0.17. Clears both bars; lowest nMAE among the simple rules and below persistence.
- C-phase: table p_hat, direction-blind. nMAE 0.203, mAUC 0.577, false move 0.19. Fails the AUC bar (0.594).
- C-phase-trend: table p_hat, trend direction. nMAE 0.210, mAUC 0.577, false move 0.30. Fails the AUC bar.
- C-phase-dir: table p_hat, fitted direction split. nMAE 0.195, mAUC 0.577, false move 0.15. Fails the AUC bar, although its error is within 0.002 of the winner: the direction model does most of the work.
- C-hgb: boosting p_hat, direction-blind. nMAE 0.199, mAUC 0.714, false move 0.18. Clears both bars.
- C-hgb-dir: boosting p_hat, fitted direction split. nMAE 0.192, mAUC 0.714, false move 0.16. Clears both bars with the lowest nMAE, by 0.001 over C-model-dir.

## Frozen choice: `C-model-dir`

Applying the prespecified rule: four variants clear both bars (C-model, C-model-dir, C-hgb, C-hgb-dir). The lowest nMAE is C-hgb-dir at 0.192, with C-model-dir at 0.193; the difference is within the 0.002 tie band, so the simpler rule wins. C-model-dir has 36 fitted coefficients (two logistic models of 17 weights plus an intercept each) against 300 boosted trees plus the same direction model; it depends on nothing but `math` and `json` at run time, and its AUC does not change when the development incidents are left out of the fit (0.647 both ways), whereas the boosting model's does (0.718 to 0.696). Its numbers, as the harness printed them: cov3 1.00, nMAE 0.193, beats-persistence 0.45, stable 0.056, moving 0.316, false move 0.17, missed move 0.65, direction 0.64, within-25 0.70, move Brier 0.241, move AUC 0.618, median-ratio IQR [0.92, 1.01], difference against v1 -0.032 [-0.054, -0.014].

Under the harness seeds the draw is fixed; under five alternative item seeds (`diag.py`) C-model-dir scores nMAE 0.190 to 0.196 (mean 0.192) with false move 0.15 to 0.20, against v1's 0.219 to 0.235 (mean 0.228). The movement AUC is seed-independent because n_move is a deterministic function of the context. On the 599 items the frozen variant draws n_move of 1 on 24 items, 2 on 166, 3 on 291, 4 on 98, and 5 on 20; the analogue-only prediction moves on 160 items and holds on 439; the personnel-band top-up is used on 25 items.

Two cautions on the numbers. The bootstrap interval excludes zero, but the selection was made on the same 599 items, so the development nMAE is the best of eight variants and the paired interval does not carry that selection. The development population moves on 53 percent of days and the test population on 39 percent; the model's mean p_hat on development is 0.48, a little below the actual rate, so on a calmer test population n_move should fall and the false-move rate should not rise, but the base-rate shift is the main thing the test run will check.

## Information used

Per item: `context.personnel_today`, `context.personnel_last_days` (its second-to-last entry as yesterday's count), `context.percent_contained`, `context.new_acres`, `context.acres`, `context.aerial_resources`, and the item's `day_of_run`. All are on the report day or before. From the candidate rows the harness passes (other incidents, outcome day before the report day): acres, containment, and personnel on the analogue's own day for the band, and its next-day count for the moved-or-held and up-or-down strata; those next-day counts are earlier fire-days of other incidents, already filtered by the harness. The fitted constants come only from pool rows dated before 2015-01-01. Nothing from the target, nothing dated on or after the report day. The item rng is the only source of randomness (two to three `sample` calls and one `shuffle`).

For the development items dated before 2015 the frozen constants were fitted on rows from other incidents that postdate the item, which the brief permits for the design phase; the leave-development-incidents-out refit shows the item's own rows do not matter to the logistic fit.

## What would break it

- A different item format. The rule reads yesterday's count from `personnel_last_days[-2]`; if that list were absent or shorter than two entries, the recent change becomes zero and the flat-yesterday indicator fires, which pushes p_hat down and makes the rule hold more often. It needs `day_of_run` on the item.
- A base-rate shift beyond the features. If test days move for reasons the seven features do not see (for instance a reporting convention that files unchanged counts for days and then jumps), p_hat will be miscalibrated and n_move wrong in the same direction on every item.
- The 0.42 to 0.58 boundary. Items with n_move = 3 (about half of them) get a half-move when the direction model is confident; the false-move rate lives there. A test population with many stable days at moderate p_hat will show a higher false-move rate than 0.17.
- Thin bands. When a stratum is thin the top-up comes from the personnel band regardless of acres and containment, so the displayed rows can differ in size from the item; this touched 25 of 599 development items.
- The gradient-boosting variants depend on the scikit-learn version that wrote `hgb.pkl`; the frozen variant does not.

## Files

- `rule.py`: `RULES` with all eight variants, `FROZEN = "C-model-dir"`, the feature functions, and the draw.
- `model.json`: the frozen constants (two logistic models, phase table, fit details). `hgb.pkl`: the boosting model for the two `hgb` variants only.
- `fit.py`: fits and writes the constants from pre-2015 pool rows; `run.py` calls it before evaluating.
- `run.py`: builds the pool once, fits, evaluates v1 and every variant with the harness, writes `results.json`.
- `diag.py`: the seed-robustness and draw-composition numbers quoted above.
