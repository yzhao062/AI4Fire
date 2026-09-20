# AI4Fire

**Task metadata for a five-task wildfire benchmark for language models and agents.**

![AI4Fire: five wildfire tasks, run bare and grounded across a registry of 35 models from twelve vendors, each task scored against a non-LLM comparator.](https://raw.githubusercontent.com/yzhao062/AI4Fire/main/docs/hero.png)

> This package carries the benchmark's task inventory so it can be queried without cloning the
> repository. **It does not run the evaluation.** The prompts, the stored model responses, the
> scoring code, and the offline reproduction live at
> [github.com/yzhao062/AI4Fire](https://github.com/yzhao062/AI4Fire).

## What the Benchmark Is

Five wildfire tasks that score without a human in the loop, against a released answer key. Every
model answers every item twice, once from the task prompt alone and once with that task's
grounding material added. Every task is scored beside at least one non-LLM comparator.

| Task | Source | Items | Grounding Added | Non-LLM Comparator |
|---|---|---|---|---|
| Daily personnel allocation | ICS-209-PLUS | 300 fire-days | Retrieved analogues | Persistence, trained regressor |
| Wildfire smoke detection | FIgLib | 224 frames, 196 paired | Reference frame | Two constants, frame-difference detector |
| Fire danger forecasting | Mesogeos Track A | 386 cells | Monthly climatology | Calendar-month prior, temperature rule, trained classifier |
| Temperature-grounded aerial QA | WildFireVQA over FLAME 3 | 408 items, 390 frames | Thermal summary block | Held-out majority, closed-form thermal rule |
| Fire data tool use | FPA-FOD 6th edition | 156 items | Read-only SQL tool | Best constant per family |

The five grounding interventions are not comparable to each other, because each appears on
exactly one task. Results are therefore task-conditional, and there is no single average
grounding effect to report.

## Usage

```bash
pip install ai4fire
ai4fire            # print the task inventory
ai4fire --json     # the same, as JSON
```

```python
from ai4fire import TASKS, REPOSITORY

for task in TASKS:
    print(task["key"], task["items"], task["comparator"])
```

## Reproducing the Results

Clone the repository. One command rebuilds every primary table from the stored responses,
offline, with no credentials and no network.

```bash
git clone https://github.com/yzhao062/AI4Fire.git
cd AI4Fire
pip install -r requirements.txt
python reproduce_tables.py
```

## Data Licensing

Each upstream source keeps its own terms, which this package does not replace. ICS-209-PLUS and
Mesogeos are CC BY 4.0. FIgLib is CC BY-NC-ND 4.0, whose NoDerivatives term is why neither the
camera frames nor features derived from them are redistributed. FPA-FOD is US Government public
domain. WildFireVQA carries an unresolved discrepancy between Apache-2.0 metadata and a CC BY 4.0
dataset card. The repository documents each in full.

## License

BSD 2-Clause for the code and our original contributions.
