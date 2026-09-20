# Brief: a movement-conditioned analogue-retrieval rule for the AI4Fire allocation task

## The problem

The allocation task asks a model to predict tomorrow's filed personnel count on a wildfire from today's
ICS-209 situation report (300 evaluation items from 245 incidents, 2015 to 2020, outside California). The
grounded arm adds up to six "analogues": earlier fire-days from other incidents in the same acres,
containment, and personnel band, each shown with its own next-day count and ratio, plus the median
ratio. The paper's finding: this grounding raises error on five of six models, the open-weight models
copy the displayed median, and the reason is that a set matched on size, containment, and staffing is not
matched on the probability that the count moves at all. Across items the median analogue ratio spans
0.93 to 1.07 (quartiles) while the filed ratio spans 0.93 to 1.02; the analogues imply movement on days
that hold flat. The analogue-only prediction (today times the displayed median) scores normalized MAE
0.249 on the test items against 0.146 for persistence.

Two reviewers named the same single experiment as the one that would most raise the paper's soundness:
design and freeze an alternative retrieval rule on development incidents disjoint from the test set,
using only information available before each report, such that the analogue-only prediction it hands the
model predicts movement better on development; then run the six models with that rule on the same 300
items, same prompt format, same information budget (six analogues, same displayed fields), and report
the paired contrast against the current rule. This workflow does the design and freezing. The model
runs happen afterwards, by the coordinator, and cost money, so the frozen rule must be worth running.

## What is already built (do not modify these files)

Directory `<scratch>\<scratch-session>\5298f142-9cba-41ca-a975-67b31eb2587b\scratchpad\exp\wf\`:

- `dev-items.jsonl`: 599 development items from 520 incidents that started before 2015 or in
  California, built exactly like the evaluation items (`build_dev.py` shows how), each with the
  same `context` fields the evaluation items carry plus `fire_mean` (the incident's mean next-day
  count over all its eligible days). No development incident is an evaluation incident. Development
  has more moving days than test (0.53 against 0.39), so absolute numbers differ; the comparison
  that matters is a candidate rule against v1 on the same development items.
- `dev_eval.py`: the harness. `load_pool()` builds the 60,428 analogue rows (consecutive-day pairs
  from other incidents in the same population; each row has today, next, prev (the day before, or
  None), acres, acres_prev, new_acres, pct, pct_prev, aerial, growth_potential, terrain, cause,
  structures, day_of_run, state, start_year, date, and the v1 band `key`). `candidates(pool, item)`
  returns the rows an item may draw from: other incidents only, outcome day strictly before the
  report day. `rule_v1` is the paper's rule. `evaluate(rule, pool, items, name, reference=...)`
  scores the analogue-only prediction (persistence times the median ratio of the drawn rows) and
  prints one row: coverage with at least three analogues, normalized MAE, beats-persistence share,
  stable-day and moving-day error, false-move and missed-move rates, direction accuracy, the
  within-25-percent share, and two movement-prediction numbers: the Brier score and the AUC of
  "share of drawn analogues whose ratio moved by more than a tenth" against whether the item's
  count actually moved. With `reference=` it adds a paired incident-cluster bootstrap interval on
  the difference in normalized error. Read the module docstring for the rule interface.
- `v1-dev-reference.json` and the harness's own output for v1 on development:
  cov3 0.99, nMAE 0.225 (persistence 0.210), beats 0.38, stable 0.110, moving 0.329,
  false move 0.32, missed move 0.60, direction 0.60, within-25 0.66, move Brier 0.285,
  move AUC 0.544, median-ratio IQR [0.94, 1.03] against actual [0.77, 1.03].
  On the 300 test items the harness reproduces the paper exactly (nMAE 0.249, the two IQRs, and all
  300 stored draws), so a rule that scores here will transfer.

The AI4Fire repository is at `<repo>` (read it, never write to it):
`run_allocation.py` holds the pool, the rule, the prompt, and the scorer; `task-allocation/items-v1.jsonl`
the 300 test items. Do not evaluate any candidate on the 300 test items; the judge and the
implementer will, once, for the frozen rule only.

Python: `<home>\miniforge3\envs\py312\python.exe` (numpy, pandas, scikit-learn). A harness run
takes about 25 seconds (pool construction dominates; build the pool once and evaluate many rules).

## Constraints every candidate must satisfy

1. Draws at most six analogues per item and shows the same fields; the prompt format does not change.
   The rule decides which six, nothing else.
2. Uses only information available on the report day: the item's `context` and `day_of_run`, and pool
   rows the harness already filtered (other incidents, outcome day before the report day). Never the
   target, never anything dated on or after the report day.
3. Any fitted component (a prior for the probability of movement, feature standardization, a
   nearest-neighbour scale) is fitted only on pool rows dated before 2015-01-01, which precedes every
   test report day, and the fit is frozen as constants or a small saved object.
4. Deterministic given the item seed the harness passes (`rng`); no other randomness.
5. Coverage: at least three analogues on at least 90 percent of development items.
6. Runs in under a minute for 600 items once the pool is built.

## The prespecified selection rule (the judge applies this; design to it)

Among candidates satisfying the constraints, the winner is the one with the lowest development
normalized MAE (analogue-only prediction) among those whose movement AUC exceeds v1's 0.544 by at least
0.05 and whose false-move rate on stable days is below v1's 0.32. If no candidate clears both bars, the
winner is the candidate with the highest movement AUC that still lowers normalized MAE below v1's 0.225,
and the decision says the bar was not met. Ties within 0.002 nMAE go to the simpler rule (fewer fitted
parameters). The paired interval against v1 is reported but does not decide.

## Deliverables per design agent

A directory `<wf>/design-<letter>/` containing `rule.py` (a module exposing `RULES = {name: fn}` with
every variant tried, and `FROZEN = "<name>"` naming the one you submit, with any fitted constants
written into the file or a small JSON beside it), `run.py` (builds the pool once, evaluates every
variant and v1 with the harness, prints the table), and `report.md` (the family, each variant in one
line with its numbers, the frozen choice and why, what information it uses, and what would break it).
Return the frozen variant's numbers in the structured output.
