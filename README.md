# AI4Fire

Work in progress: two of the five tasks have not run, and the benchmark is still being extended.

The execution record behind *AI4Fire: Large Language Models and Agents on Fire Tasks*, a survey and benchmark: the version-1 task builders, the item manifests, every stored model response, and the scripts that turn those responses into the numbers the paper reports.

Version 1 specifies five wildfire tasks that score without a human in the loop. Three have run, bare and grounded, on six models; two are built and waiting.

| Task | Source | Items | Status |
|---|---|---|---|
| Daily personnel allocation | ICS-209-PLUS (St. Denis et al., 2023), CC BY 4.0 | 300 fire-days | run |
| Wildfire smoke detection and time to detection | FIgLib, the HPWREN fire ignition image library | 224 frames | run |
| Fire danger forecasting | Mesogeos Track A (Kondylatos et al., 2023), CC BY 4.0 | 386 cells | run |
| Temperature-grounded aerial question answering | WildFireVQA over FLAME 3 imagery | 408 items | built, imagery waits on an account |
| Fire data tool use | FPA-FOD 6th edition (Short, 2022) | 156 items | built, not run |

Models: `claude-opus-5`, `claude-opus-4.8`, `gemini-3.1-pro`, and `gpt-6-astra` through one gateway; `Qwen3-VL-235B-A22B` and `Llama 4 Maverick` through Amazon Bedrock. Every run uses the same prompts and one output cap of 1,536 tokens. Five models ran at temperature zero; the `gpt-6-astra` endpoint accepts only its default temperature of 1 and takes the cap as `max_completion_tokens`, which counts reasoning tokens (`gw.py`). Its bare allocation repeat is under `repeat/`.

## Layout

```
task-<name>/           items.jsonl (the full pool), items-v1.jsonl or items-index.csv (the scored set),
                       responses-<model>-<condition>.jsonl (one row per item: raw text, parsed answer, usage,
                       served model, and for grounded allocation the analogue ids the prompt carried), scores.json
task-allocation/b4-rerun/   the one grounded item re-queried after the round-2 review, with the rows it replaced
repeat/                repeat runs used for the run-to-run variation numbers
archive-2026-09-15-*/  earlier runs kept as evidence: the output-cap defect (cap200) and the first analogue pool
logs/                  runner logs; root-level run-*.log and .err files are the gateway runs
analysis/              interval, comparator, pairing, and failure scripts (see below)
figures/               the paper's data figures: each make_<name>.py draws one from the response files and the
                       analysis outputs, prints its numbers beside the paper's, and writes <name>.pdf and .png;
                       figstyle.py holds the shared palette and type sizes
survey/                the two search passes (138 kept works with the 31 re-check records), the scoping summary
                       they re-checked, the 34-source data availability check, and the script behind Appendix A's counts
build_items_*.py       task builders; fetch_*.py downloads the two sources that allow it
run_*.py               the runners; gw.py is the only place a model is called
```

`data/` and the FIgLib frames under `task-figlib/images*` are not in this repository. FIgLib is served as-is with no formal license, so the item manifest carries a `source_url` per frame and `build_items_figlib.py` re-downloads them. `fetch_mesogeos.py` and `fetch_fpafod.py` download their sources; ICS-209-PLUS and WildFireVQA are downloaded by hand from the pages the builders name.

## Reproducing the paper's numbers

Every table starts from the stored responses, so nothing below calls a model.

```
python table_rows_allocation.py            # allocation table: MAE, normalized error, split columns, within-25% share
python analyze_allocation_all.py           # stable and moving days, false move, missed move, direction
python copy_agreement.py                   # how often a grounded prediction equals the displayed analogue rule
python score_mesogeos_all.py               # AUPRC, fire-class F1, call rate, omitted answers, per run
python month_prior.py                      # the calendar-month prior fit on the training years
python repeat_stats.py                     # repeat runs against their first runs
python analysis/figlib_paired.py           # smoke detection on the 196 items both conditions answered
python analysis/cluster_uncertainty.py     # 95 percent cluster bootstrap intervals (20,000 resamples, seed 20260915)
python analysis/information_only_baselines.py   # persistence, analogue-only, climatology, and last-day rules
python analysis/answer_failures.py         # rows with no usable answer, by task and run
python analysis/served_models.py           # table of distinct served_model values, counts, and date ranges
python analysis/calibration_mesogeos.py    # decision consistency, ECE, Brier and its Murphy terms, reliability tables, per run
python analysis/prompt_sensitivity.py      # the two prompt paraphrases against the paper's prompt, paired block bootstrap
python analysis/retrieval_v2.py            # grounded runs under analogue rule v2 beside bare and rule v1, paired by incident
python baselines/mesogeos_trained.py       # boosted and logistic classifiers on the prompt's numbers; writes responses-baseline-*.jsonl
python baselines/allocation_trained.py     # boosted and ridge regressors on the report fields; writes responses-baseline-*.jsonl
python build_manifest.py --check           # verify every file in manifest-v1.json against its recorded checksum
python figures/make_grounding_effects.py   # grounded minus bare on the three tasks, with the cluster intervals
python figures/make_survey_landscape.py    # the 138 kept works by kind and task category, and models tested
python figures/make_allocation_analogues.py   # copies of the displayed analogue median; spread of the next-day ratio
python figures/make_figlib_timeline.py     # smoke detection accuracy by time since the plume, bare and grounded
python figures/make_calibration.py         # reliability diagrams per model beside the calendar-month prior
python figures/make_prompt_sensitivity.py  # stated probabilities under the paper's prompt against two paraphrases
```

Each figure script reads the same files as the analysis script it follows. Before writing the figure it prints every number it draws beside the paper's, with a match flag per row. `make_allocation_analogues.py` rebuilds the analogue pool from the SIT record, which takes about half a minute.

### Reported runs

The 36 reported response files (three tasks x six models x two conditions) are listed in `manifest-v1.json` at the repository root with their row counts, SHA-256 checksums, distinct `served_model` values, and serving pathways. The analysis scripts (`analysis/cluster_uncertainty.py`, `analysis/answer_failures.py`, `analysis/figlib_paired.py`, `score_mesogeos_all.py`, `table_rows_allocation.py`, and `copy_agreement.py`) support a `--manifest manifest-v1.json` flag that restricts evaluation strictly to these reported files, filtering out exploratory probes and historic resolution arms. For `cluster_uncertainty.py`, the flag also automatically silences the cross-run skew guard for the five grounded allocation pairs marked with `skew_override: true` from the analogue-date repair.

`build_manifest.py` regenerates the manifest from the tree: membership, sections, and reasons come from the existing file; row counts, checksums, served identifiers, and write times are recomputed. Beside the 36 reported files it lists the trained-baseline response files (`baselines` section), the prompt-paraphrase runs (`prompt_variants`), and the rule-v2 grounded runs (`retrieval_v2`). `build_manifest.py --check` exits 1 if any recorded checksum differs from the file on disk.

### Trained baselines, prompt paraphrases, and the second analogue rule

`baselines/mesogeos_trained.py` fits a histogram gradient-boosted classifier and a logistic regression on the 2006 to 2019 training years, on two feature sets: the 121 numbers the bare prompt prints, at the four significant figures it prints them (last six daily values plus window mean, minimum, and maximum per driver, the static fields, and the month; the script asserts the 386 evaluation rows against the item file) and all 30 daily values per driver at source precision. The learning rate and iteration count are chosen on 2020, and the fits are scored on the 386 items and on the full 2021 to 2022 holdout; `baselines/mesogeos_trained.json` holds every number. `baselines/allocation_trained.py` fits a gradient-boosted regressor of the next-day personnel ratio (and count and ridge variants) on 30,869 fire-days from 1,499 incidents disjoint from the evaluation incidents, reading the 23 report fields of the bare prompt, with hyperparameters chosen by five-fold cross-validation grouped by incident; its analogue ablation draws the six analogues with the runner's own rule and asserts the draw against the saved prompts on all 300 items; `baselines/allocation_trained.json` holds the cross-validation table, the 300-item scores, and the incident-clustered intervals. Both scripts write `responses-baseline-<fit>-bare.jsonl` in the task directory in the runner's row format, so the paper's scorers read them like a model run; `cluster_uncertainty.py` treats a `baseline-` arm as legitimately unpaired.

`run_mesogeos.py --variant p1` and `--variant p2` run two paraphrases of the fire danger prompt that carry the same numbers and the same answer schema (an analyst framing with the drivers named in words, and a question-first table layout). They write `responses-<model>-bare-p1.jsonl` and `-p2.jsonl` and never replace the reported `p0` files; `analysis/prompt_sensitivity.py` scores them against `p0` with a paired block-by-month bootstrap.

`run_allocation.py --rule v2` draws the six analogues under the frozen nearest-neighbour rule of `retrieval-v2/DECISION.md` (family B's `nn_pers_chg`, designed on 599 development items from incidents disjoint from the evaluation set and selected by the rule fixed in `retrieval-v2/BRIEF.md`), with its fourteen standardization constants in `task-allocation/rule-v2-scales.json`. It writes `responses-<model>-grounded-v2.jsonl` and the draws to `task-allocation/rule-v2-draws.jsonl`, and never touches a v1 file; `retrieval-v2/check_v2_transfer.py` confirms that the v1 draws are unchanged and that the runner's v2 draws equal the harness's on all 300 items; `analysis/retrieval_v2.py` scores the v2 runs beside bare and rule v1 with incident-paired intervals, including the two analogue-only rules against persistence and against each other, all joined by item id.

`cluster_uncertainty.py` flags a bare-and-grounded pair whose files were written more than 60 minutes apart. The grounded allocation files of the five models that ran before the analogue-date repair were rewritten on 2026-09-16 to replace one repaired item (below; `gpt-6-astra` ran under the final rule), so for allocation pass `--max-skew-min 100000`; the pairing is by item id and was checked.

## Running a model

`gw.py` reads `NAIRR_GATEWAY_KEY` for the gateway and uses the standard AWS credential chain, or `AWS_BEARER_TOKEN_BEDROCK`, for Bedrock. No key appears in this repository or in any log.

```
python run_allocation.py --models claude-opus-5 claude-opus-4.8
python run_figlib.py --models gemini-3.1-pro --workers 2
python run_mesogeos.py --models bedrock:qwen.qwen3-vl-235b-a22b
python run_all.py bedrock:us.meta.llama4-maverick-17b-instruct-v1:0            # all three tasks, one after another
python run_all.py gpt-6-astra
python requery_figlib_item.py gpt-6-astra bare <item_id>                       # re-query one item after a transport error; logged in task-figlib/requery-log.jsonl
```

A runner rewrites one condition file at the end of that condition, so a killed run loses everything since the last write, and a rerun of one condition leaves the other condition's file older; that is what the 60-minute check above is for.

## The analogue-date repair

The grounded allocation prompt shows retrieved analogues, each a consecutive pair of fire-days from another incident with the staffing on both days. The first version of the eligibility rule required only the analogue's input day to precede the item's target day, which let one item of 300 see an outcome filed on the item's own target day. The rule now requires both days to precede the item's report day (`analogues()` in `run_allocation.py`). `check_b4_draws.py` shows that the corrected rule changes exactly one item's draw; `rerun_allocation_item.py` re-queried that item on all five models and `splice_b4.py` moved the rows into the response files. The replaced rows are in `task-allocation/b4-rerun/replaced-rows.jsonl`.

## License

The code is under the BSD 2-Clause License (`LICENSE`). The stored responses are model outputs over items derived from the sources above, each under its own terms; the item manifests carry the identifiers needed to rebuild every prompt from the source data.
