# FireAIBench

The execution record behind *Large Language Models and AI Agents on Fire Tasks: A Survey and Benchmark*: the version-1 task builders, the item manifests, every stored model response, and the scripts that turn those responses into the numbers the paper reports.

Version 1 specifies five wildfire tasks that score without a human in the loop. Three have run, bare and grounded, on five models; two are built and waiting.

| Task | Source | Items | Status |
|---|---|---|---|
| Daily personnel allocation | ICS-209-PLUS (St. Denis et al., 2023), CC BY 4.0 | 300 fire-days | run |
| Wildfire smoke detection and time to detection | FIgLib, the HPWREN fire ignition image library | 224 frames | run |
| Fire danger forecasting | Mesogeos Track A (Kondylatos et al., 2023), CC BY 4.0 | 386 cells | run |
| Temperature-grounded aerial question answering | WildFireVQA over FLAME 3 imagery | 408 items | built, imagery waits on an account |
| Fire data tool use | FPA-FOD 6th edition (Short, 2022) | 156 items | built, not run |

Models: `claude-opus-5`, `claude-opus-4.8`, and `gemini-3.1-pro` through one gateway; `Qwen3-VL-235B-A22B` and `Llama 4 Maverick` through Amazon Bedrock. Every run uses the same prompts, temperature zero, and one output cap of 1,536 tokens.

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
```

`cluster_uncertainty.py` flags a bare-and-grounded pair whose files were written more than 60 minutes apart. The five grounded allocation files were rewritten on 2026-09-16 to replace one repaired item (below), so for allocation pass `--max-skew-min 100000`; the pairing is by item id and was checked.

## Running a model

`gw.py` reads `NAIRR_GATEWAY_KEY` for the gateway and uses the standard AWS credential chain, or `AWS_BEARER_TOKEN_BEDROCK`, for Bedrock. No key appears in this repository or in any log.

```
python run_allocation.py --models claude-opus-5 claude-opus-4.8
python run_figlib.py --models gemini-3.1-pro --workers 2
python run_mesogeos.py --models bedrock:qwen.qwen3-vl-235b-a22b
python run_bedrock_all.py bedrock:us.meta.llama4-maverick-17b-instruct-v1:0    # all three tasks, one after another
```

A runner rewrites one condition file at the end of that condition, so a killed run loses everything since the last write, and a rerun of one condition leaves the other condition's file older; that is what the 60-minute check above is for.

## The analogue-date repair

The grounded allocation prompt shows retrieved analogues, each a consecutive pair of fire-days from another incident with the staffing on both days. The first version of the eligibility rule required only the analogue's input day to precede the item's target day, which let one item of 300 see an outcome filed on the item's own target day. The rule now requires both days to precede the item's report day (`analogues()` in `run_allocation.py`). `check_b4_draws.py` shows that the corrected rule changes exactly one item's draw; `rerun_allocation_item.py` re-queried that item on all five models and `splice_b4.py` moved the rows into the response files. The replaced rows are in `task-allocation/b4-rerun/replaced-rows.jsonl`.

## License

The code is under the BSD 2-Clause License (`LICENSE`). The stored responses are model outputs over items derived from the sources above, each under its own terms; the item manifests carry the identifiers needed to rebuild every prompt from the source data.
