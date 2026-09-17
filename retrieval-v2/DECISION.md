# Decision: the frozen rule is family B's `nn_pers_chg`

Applied on 2026-09-17 by the coordinator, after the Workflow's judge agent was killed by a shutdown while reading the reports. The brief's selection rule is mechanical, so the decision is reproduced here from the three `results.json` files rather than re-judged.

## The rule, as the brief states it

Among candidates satisfying the constraints, the winner is the one with the lowest development normalized MAE (analogue-only prediction) among those whose movement AUC exceeds v1's 0.544 by at least 0.05 and whose false-move rate on stable days is below v1's 0.32. Ties within 0.002 nMAE go to the simpler rule (fewer fitted parameters).

## Applied

Every variant of the three families, sorted by development nMAE, with the two bars (AUC >= 0.594, false move < 0.32, coverage-3 >= 0.90):

| family | variant | nMAE | move AUC | false move | clears |
|---|---|---|---|---|---|
| B | nn_pers_chg_cap2 | 0.1900 | 0.632 | 0.225 | yes |
| B | nn_pers_chg | 0.1911 | 0.632 | 0.225 | yes |
| C | C-hgb-dir | 0.1917 | 0.714 | 0.158 | yes |
| B | nn_all_dyn | 0.1925 | 0.640 | 0.236 | yes |
| C | C-model-dir | 0.1928 | 0.618 | 0.173 | yes |
| A | A-b5-t3-p (best of A) | 0.1975 | 0.596 | 0.246 | yes |
| | v1 (paper) | 0.2252 | 0.544 | 0.317 | |

The lowest nMAE among bar-clearers is `nn_pers_chg_cap2` at 0.1900. Two variants sit inside the 0.002 tie window: `nn_pers_chg` (0.1911) and `C-hgb-dir` (0.1917). The tie goes to the simpler rule. The B variants carry fourteen fitted constants (the mean and standard deviation of seven features on the 54,566 pool rows dated before 2015); `C-hgb-dir` carries a 300-tree boosting classifier plus a 17-weight logistic direction model, so it loses the tie. Between the two B variants the fitted constants are identical, and the per-incident cap is one rule component more, so the frozen rule is the one without it, as family B's own report also concluded.

**FROZEN: `nn_pers_chg`.** Nearest six eligible pool rows within the item's personnel band, ranked by weighted squared Euclidean distance over seven standardized report-day features (log personnel, log1p acres, containment clipped to 0 to 100, log day of run, one-day log personnel change, log1p new acres, log1p aerial resources) with weights (1, 1, 1, 1, 4, 3, 1); ties by `analogue_id`; no random step. Development numbers as the harness printed them and as the coordinator reproduced them on 2026-09-17 (`design-B/verify-run.txt`): coverage-3 1.00, nMAE 0.191 (v1 0.225, persistence 0.210), beats persistence 0.45, stable 0.073, moving 0.298, false move 0.23, missed move 0.58, direction 0.70, within-25 0.70, movement Brier 0.253, movement AUC 0.632, median-ratio IQR [0.91, 1.00] against actual [0.77, 1.03], paired incident bootstrap against v1 -0.034 [-0.055, -0.016].

## Checks made before freezing

- Constraints: the item side reads `personnel_today`, `personnel_last_days[-2]`, `acres`, `percent_contained`, `new_acres`, `aerial_resources`, and `day_of_run`, all on the report day; the pool side reads the same seven quantities on the row's own day and never the row's `next`; the harness's candidate filter (other incidents, outcome day before the report day) is unchanged; the fourteen constants are fitted on pool rows dated before 2015-01-01 and `run.py` refits and asserts them; the rule ignores `rng` and `run.py` shows identical draws under three seeds.
- Reproduction: `python retrieval-v2/design-B/run.py` in the repository copy reproduces every row of the table and the frozen pick.
- Two cautions the report itself raises, carried into the paper: the selection was made on the same 599 items across the three families' recorded variants (39 in A, 20 in B including four ablations, 8 in C), so 0.191 is a selected value and the paired interval does not carry that selection; the gain is smaller on the 73 development items from 2015 or later (0.205 to 0.190) than on the 526 earlier ones (0.228 to 0.191). Only 44 percent of the drawn rows on the development items, and 45 percent on the 300 evaluation items, fall in the item's v1 band (all fall in its personnel band), so the grounded block's header line, which said "same size, containment, and staffing band" under v1, says "same staffing band" under v2; everything else in the prompt is byte-identical.

## What follows

`run_allocation.py --rule v2` implements the frozen rule inside the runner with the constants in `task-allocation/rule-v2-scales.json`, writes `responses-<model>-grounded-v2.jsonl`, and records the draws in `task-allocation/rule-v2-draws.jsonl`; `analysis/retrieval_v2.py` scores the six models beside bare and rule v1.
