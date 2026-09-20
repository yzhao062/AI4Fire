# AI4Fire

Work in progress: the benchmark is still being extended.

The execution record behind *AI4Fire: Large Language Models and Agents on Fire Tasks*, a survey and benchmark: the version-1 task builders, the item manifests, every stored model response, and the scripts that turn those responses into the numbers the paper reports.

Version 1 specifies five wildfire tasks that score without a human in the loop. All five have run on the six reported models, bare and grounded (bare and with the tool on the tool-use task). The model sweep of 2026-09-18 adds 29 Bedrock models.

| Task | Source | Items | Status |
|---|---|---|---|
| Daily personnel allocation | ICS-209-PLUS (St. Denis et al., 2023), CC BY 4.0 | 300 fire-days | run |
| Wildfire smoke detection and time to detection | FIgLib, the HPWREN fire ignition image library | 224 frames | run |
| Fire danger forecasting | Mesogeos Track A (Kondylatos et al., 2023), CC BY 4.0 | 386 cells | run |
| Temperature-grounded aerial question answering | WildFireVQA (Habibpour et al., 2026), apache-2.0, over FLAME 3 imagery | 408 items over 390 frames | run |
| Fire data tool use | FPA-FOD 6th edition (Short, 2022) | 156 items | run |

## Models

The six reported models are `claude-opus-5`, `claude-opus-4.8`, `gemini-3.1-pro`, and `gpt-6-astra` through one gateway, and `Qwen3-VL-235B-A22B` and `Llama 4 Maverick` through Amazon Bedrock. The model sweep of 2026-09-18 added 29 Bedrock models in two groups, chosen by a capability probe (`probe_bedrock_caps.py`; results in `probes/bedrock-caps-2026-09-18.json`). Ten models that accept images and tools ran all five tasks. Nineteen text-only models ran the three text tasks; the two whose tool probe failed ran allocation and fire danger only. The paired analyses, the calibration analysis, the sweep tables, and the model figures read the registry `models.py`. Its tiers are `core`, `added`, `all`, `full`, `text`, and `every`: `all` and its alias `full` select the sixteen full-capability models, and `every` selects all 35. The legacy cluster and table scripts keep their own file discovery or manifest rules. `docs/MODEL-SWEEP-2026-09-18.md` records the probe, the groups, the pilots, the launch scripts, and the outcome.

| Model | Identifier (`gw.py`) | Vendor | Weights | Group | Tasks | Cap |
|---|---|---|---|---|---|---|
| claude-opus-4.8 | `claude-opus-4.8` | Anthropic | proprietary | reported six | all five | 1,536 |
| claude-opus-5 | `claude-opus-5` | Anthropic | proprietary | reported six | all five | 1,536 |
| gemini-3.1-pro | `gemini-3.1-pro` | Google | proprietary | reported six | all five | 1,536 |
| gpt-6-astra | `gpt-6-astra` | OpenAI | proprietary | reported six | all five | 1,536 |
| Nova Lite | `bedrock:amazon.nova-lite-v1:0` | Amazon | proprietary | added, full capability | all five | 1,536 |
| Nova Pro | `bedrock:amazon.nova-pro-v1:0` | Amazon | proprietary | added, full capability | all five | 1,536 |
| Nova 2 Lite | `bedrock:us.amazon.nova-2-lite-v1:0` | Amazon | proprietary | added, full capability | all five | 1,536 |
| Qwen3-VL | `bedrock:qwen.qwen3-vl-235b-a22b` | Alibaba | open | reported six | all five | 1,536 |
| Llama 4 Maverick | `bedrock:us.meta.llama4-maverick-17b-instruct-v1:0` | Meta | open | reported six | all five | 1,536 |
| Llama 4 Scout | `bedrock:us.meta.llama4-scout-17b-instruct-v1:0` | Meta | open | added, full capability | all five | 1,536 |
| Mistral Large 3 | `bedrock:mistral.mistral-large-3-675b-instruct` | Mistral | open | added, full capability | all five | 1,536 |
| Ministral 3 8B | `bedrock:mistral.ministral-3-8b-instruct` | Mistral | open | added, full capability | all five | 1,536 |
| Kimi K2.5 | `bedrock:moonshotai.kimi-k2.5` | Moonshot | open | added, full capability | all five | 1,536 |
| Gemma 3 27B | `bedrock:google.gemma-3-27b-it` | Google | open | added, full capability | all five | 1,536 |
| Gemma 3 12B | `bedrock:google.gemma-3-12b-it` | Google | open | added, full capability | all five | 1,536 |
| Gemma 3 4B | `bedrock:google.gemma-3-4b-it` | Google | open | added, full capability | all five | 1,536 |
| Nova Micro | `bedrock:amazon.nova-micro-v1:0` | Amazon | proprietary | text-only sweep | allocation, fire danger, tool use | 1,536 |
| Llama 3.3 70B | `bedrock:us.meta.llama3-3-70b-instruct-v1:0` | Meta | open | text-only sweep | allocation, fire danger, tool use | 1,536 |
| Llama 3.1 70B | `bedrock:us.meta.llama3-1-70b-instruct-v1:0` | Meta | open | text-only sweep | allocation, fire danger, tool use | 1,536 |
| Mistral Small 2402 | `bedrock:mistral.mistral-small-2402-v1:0` | Mistral | open | text-only sweep | allocation, fire danger, tool use | 1,536 |
| Devstral 2 123B | `bedrock:mistral.devstral-2-123b` | Mistral | open | text-only sweep | allocation, fire danger, tool use | 1,536 |
| Qwen3 32B | `bedrock:qwen.qwen3-32b-v1:0` | Alibaba | open | text-only sweep | allocation, fire danger, tool use | 1,536 |
| Qwen3 Next 80B | `bedrock:qwen.qwen3-next-80b-a3b` | Alibaba | open | text-only sweep | allocation, fire danger, tool use | 1,536 |
| Qwen3 Coder 30B | `bedrock:qwen.qwen3-coder-30b-a3b-v1:0` | Alibaba | open | text-only sweep | allocation, fire danger, tool use | 1,536 |
| DeepSeek V3.2 | `bedrock:deepseek.v3.2` | DeepSeek | open | text-only sweep | allocation, fire danger, tool use | 1,536 |
| GPT-OSS 120B | `bedrock:openai.gpt-oss-120b-1:0` | OpenAI | open | text-only sweep | allocation, fire danger, tool use | 1,536 |
| GPT-OSS 20B | `bedrock:openai.gpt-oss-20b-1:0` | OpenAI | open | text-only sweep | allocation, fire danger, tool use | 1,536 |
| GLM 5 | `bedrock:zai.glm-5` | Z.AI | open | text-only sweep | allocation, fire danger, tool use | 1,536 |
| GLM 4.7 | `bedrock:zai.glm-4.7` | Z.AI | open | text-only sweep | allocation, fire danger, tool use | 1,536 |
| GLM 4.7 Flash | `bedrock:zai.glm-4.7-flash` | Z.AI | open | text-only sweep | allocation, fire danger, tool use | 1,536 |
| MiniMax M2.5 | `bedrock:minimax.minimax-m2.5` | MiniMax | open | text-only sweep | allocation, fire danger, tool use | 8,192 |
| Kimi K2 Thinking | `bedrock:moonshot.kimi-k2-thinking` | Moonshot | open | text-only sweep | allocation, fire danger, tool use | 8,192 |
| Nemotron Super 3 120B | `bedrock:nvidia.nemotron-super-3-120b` | NVIDIA | open | text-only sweep | allocation, fire danger, tool use | 1,536 |
| Llama 3.1 8B | `bedrock:us.meta.llama3-1-8b-instruct-v1:0` | Meta | open | text-only sweep | allocation, fire danger | 1,536 |
| DeepSeek R1 | `bedrock:us.deepseek.r1-v1:0` | DeepSeek | open | text-only sweep | allocation, fire danger | 8,192 |

Every run uses the same prompts and the output cap of 1,536 tokens. The three reasoning models marked 8,192 are the exception. MiniMax M2.5 and Kimi K2 Thinking parsed 2 of 10 bare allocation pilot items at the standard cap, and DeepSeek R1 parsed 6. Their runs record the 8,192-token cap in `usage.max_out`, and `models.py` records the configured cap of every model; rows written before that field existed, the core six included, do not carry it. All models ran at temperature zero except `gpt-6-astra`, whose endpoint accepts only its default temperature of 1 and takes the cap as `max_completion_tokens`, which counts reasoning tokens (`gw.py`). Its bare allocation repeat is under `repeat/`. Gemma 3 on Bedrock does not use Converse tool blocks; it calls tools through Python calls in fenced `tool_code` blocks. `gw.py` therefore renders the tool declarations into the system prompt and parses the call blocks from the assistant turns (`parse_tool_code_fences`). The runner sees the same tool interface over a different wire protocol. Such rows carry `tool_protocol: "tool_code"` in the usage dictionary. The full Llama 3.1 8B run supersedes its 10-item allocation probe of 2026-09-16. That probe verified the Bedrock credentials, and its rows remain in the history at commit 6e37924.

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
build_items_*.py       task builders; fetch_*.py downloads the three sources that allow it
match_flame3.py        maps each aerial item to its FLAME 3 frame by thermal fingerprint and renders the thermal view
run_*.py               the runners; gw.py is the only place a model is called
models.py              the model registry: label, identifier, vendor, weights, serving path, tier, cap, tool support
probe_bedrock_caps.py  the text, image, and tool probe behind the sweep groups; probes/ holds its results
run_tier1.ps1          one detached five-task chain per added full-capability model (2026-09-18 sweep)
run_tier2.ps1          the text-only sweep: one detached three-task chain per model, throttled; -Pilot for the ten-item pilots
docs/                  the sweep record
tests/                 pytest: the tool-use harness against the reference queries, and the Gemma tool_code adapter
```

`data/` and the FIgLib frames under `task-figlib/images*` are not in this repository. FIgLib is served as-is with no formal license, so the item manifest carries a `source_url` per frame and `build_items_figlib.py` re-downloads them. `fetch_mesogeos.py` and `fetch_fpafod.py` download their sources; ICS-209-PLUS and the WildFireVQA question release are downloaded by hand from the pages the builders name. `fetch_flame3.py` downloads the FLAME 3 computer-vision subset (Sycan Marsh) from its Kaggle mirror with the Kaggle API credentials read from the environment or a local `.env` (`KAGGLE_USERNAME`, `KAGGLE_KEY`; the IEEE DataPort original needs an institutional login), and `match_flame3.py` then writes `task-wildfirevqa/image-map.json` and the inferno-rendered thermal images under `data/flame3/rendered/`.

## Reproducing the paper's numbers

### Quick Start: Reproduce Primary Tables in One Command

The primary results tables of the paper across all five wildfire tasks can be regenerated in a single command directly from the tracked model responses (`task-*/responses-*.jsonl`). No API keys, credentials, GPU, or raw data downloads are required:

```bash
python reproduce_tables.py
```

This runs offline in under 3 seconds and reproduces the numbers for all 36 reported core model runs across:
- **Daily Personnel Allocation:** Table MAE, normalized error, share within 25%, and fraction beating persistence comparator (13.76 MAE, 0.1465 norm MAE).
- **Wildfire Smoke Detection:** Frame accuracy, recall, false positive rate, and sequences detected (out of 28 positive sequences).
- **Fire Danger Forecasting:** AUPRC, fire-class F1, call rate, and omitted answer counts.
- **Aerial Question Answering:** Overall accuracy, closed-form 48-item subset, other 360 items, and non-LLM majority (0.6275) and hybrid (0.6642) comparators.
- **Fire Data Tool Use:** Accuracy, abstentions, and mean tool calls for bare and tool-augmented arms.

To reproduce results for the expanded 16 full-capability models from the Bedrock sweep:
```bash
python reproduce_tables.py --tier all
# or
python table_rows_new_models.py --tier all
```

### Detailed Analysis Scripts (Offline, No Raw Data Downloads)

The following scripts evaluate stored model responses without calling any external models or requiring raw source downloads:

```bash
# Task-specific scoring and table generation
python table_rows_allocation.py            # allocation table: MAE, normalized error, split columns, within-25% share
python analyze_allocation_all.py           # stable and moving days, false move, missed move, direction
python copy_agreement.py                   # how often a grounded prediction equals the displayed analogue rule
python score_mesogeos_all.py               # AUPRC, fire-class F1, call rate, omitted answers, per run
python repeat_stats.py                     # run-to-run variation across repeat runs
python table_rows_new_models.py --tier all # results tables for all sixteen full-capability models

# Paired and bootstrap uncertainty analyses (all use stable seed 20260915, 20,000 resamples)
python analysis/figlib_paired.py           # smoke detection paired analysis (196 common items)
python analysis/tooluse_paired.py          # tool use paired tool-minus-bare accuracy with 12-family bootstrap
python analysis/wildfirevqa_paired.py      # aerial QA paired grounded-vs-bare with 390-frame bootstrap
python analysis/wildfirevqa_comparators.py # aerial models against majority and hybrid comparators
python analysis/calibration_mesogeos.py    # ECE, Brier score, Murphy reliability and resolution terms
python analysis/prompt_sensitivity.py      # fire danger prompt paraphrases (p1, p2) against p0
python analysis/cluster_uncertainty.py     # 95% cluster bootstrap intervals across tasks
python analysis/retrieval_v2.py            # grounded runs under analogue rule v2 beside bare and rule v1
python analysis/answer_failures.py         # unparseable/omitted answer counts by task and model
python analysis/served_models.py           # audit of distinct served_model strings and timestamps

# Integrity checks and tests
python build_manifest.py --check           # verify all 330+ files against SHA-256 hashes in manifest-v1.json
python -m pytest tests                     # test tool-use harness and Gemma tool_code adapter (25 passed, 2 skipped)

# Figures (each prints its numbers beside paper values and renders PDF/PNG)
python figures/make_grounding_effects.py   # Figure: grounded minus bare across four tasks with cluster CIs
python figures/make_survey_landscape.py    # Figure: 138 kept literature works by taxonomy category
python figures/make_allocation_analogues.py# Figure: analogue copy rates and next-day staffing ratios
python figures/make_figlib_timeline.py     # Figure: smoke detection accuracy by minutes since plume
python figures/make_calibration.py         # Figure: fire danger reliability diagrams vs month prior
python figures/make_prompt_sensitivity.py  # Figure: stated probabilities under prompt paraphrases
```

*Note on offline execution*: `copy_agreement.py`, `analysis/retrieval_v2.py`, and `figures/make_allocation_analogues.py` utilize the precomputed analogue pool cache shipped at `analysis/.analogue-pool-cache.json`, enabling immediate offline execution without re-indexing the raw historical ICS-209-PLUS database.

### Scripts Requiring Raw Upstream Data (`data/`)

The following specialized scripts train new non-LLM baseline models or compute empirical priors from raw multi-year historical records. They require raw upstream data to be fetched into `data/` using the respective fetch scripts:

```bash
python month_prior.py                      # fits calendar-month prior on raw 2006-2019 NetCDF files (data/mesogeos/)
python analysis/information_only_baselines.py # fits historical persistence and climatology rules
python baselines/mesogeos_trained.py       # trains boosted trees & logistic regression on raw Mesogeos drivers
python baselines/allocation_trained.py     # trains boosted regressors on raw 30,869 ICS-209-PLUS fire-days
```

### Reported runs

The 36 reported response files (three tasks x six models x two conditions) are listed in `manifest-v1.json` at the repository root with their row counts, SHA-256 checksums, distinct `served_model` values, and serving pathways. The analysis scripts (`analysis/cluster_uncertainty.py`, `analysis/answer_failures.py`, `analysis/figlib_paired.py`, `score_mesogeos_all.py`, `table_rows_allocation.py`, and `copy_agreement.py`) support a `--manifest manifest-v1.json` flag that restricts evaluation strictly to these reported files, filtering out exploratory probes and historic resolution arms. For `cluster_uncertainty.py`, the flag also automatically silences the cross-run skew guard for the five grounded allocation pairs marked with `skew_override: true` from the analogue-date repair.

`build_manifest.py` regenerates the manifest from the tree: membership, sections, and reasons come from the existing file; row counts, checksums, served identifiers, and write times are recomputed. Beside the 36 reported files it lists the trained-baseline response files (`baselines`), the prompt-paraphrase runs (`prompt_variants`), the rule-v2 grounded runs (`retrieval_v2`), and the tool-use and aerial runs of every model (`tooluse`, `wildfirevqa`). The `added_models` and `text_models` sections list the 2026-09-18 sweep files on the other tasks, selected through the tiers in `models.py`. `build_manifest.py --check` exits 1 if any recorded checksum differs from the file on disk.

### Trained baselines, prompt paraphrases, the second analogue rule, and the tool-use task

`baselines/mesogeos_trained.py` fits a histogram gradient-boosted classifier and a logistic regression on the 2006 to 2019 training years, on two feature sets: the 121 numbers the bare prompt prints, at the four significant figures it prints them (last six daily values plus window mean, minimum, and maximum per driver, the static fields, and the month; the script asserts the 386 evaluation rows against the item file) and all 30 daily values per driver at source precision. The learning rate and iteration count are chosen on 2020, and the fits are scored on the 386 items and on the full 2021 to 2022 holdout; `baselines/mesogeos_trained.json` holds every number. `baselines/allocation_trained.py` fits a gradient-boosted regressor of the next-day personnel ratio (and count and ridge variants) on 30,869 fire-days from 1,499 incidents disjoint from the evaluation incidents, reading the 23 report fields of the bare prompt, with hyperparameters chosen by five-fold cross-validation grouped by incident; its analogue ablation draws the six analogues with the runner's own rule and asserts the draw against the saved prompts on all 300 items; `baselines/allocation_trained.json` holds the cross-validation table, the 300-item scores, and the incident-clustered intervals. Both scripts write `responses-baseline-<fit>-bare.jsonl` in the task directory in the runner's row format, so the paper's scorers read them like a model run; `cluster_uncertainty.py` treats a `baseline-` arm as legitimately unpaired.

`run_mesogeos.py --variant p1` and `--variant p2` run two paraphrases of the fire danger prompt that carry the same numbers and the same answer schema (an analyst framing with the drivers named in words, and a question-first table layout). They write `responses-<model>-bare-p1.jsonl` and `-p2.jsonl` and never replace the reported `p0` files; `analysis/prompt_sensitivity.py` scores them against `p0` with a paired block-by-month bootstrap.

`run_allocation.py --rule v2` draws the six analogues under the frozen nearest-neighbour rule of `retrieval-v2/DECISION.md` (family B's `nn_pers_chg`, designed on 599 development items from incidents disjoint from the evaluation set and selected by the rule fixed in `retrieval-v2/BRIEF.md`), with its fourteen standardization constants in `task-allocation/rule-v2-scales.json`. It writes `responses-<model>-grounded-v2.jsonl` and the draws to `task-allocation/rule-v2-draws.jsonl`, and never touches a v1 file; `retrieval-v2/check_v2_transfer.py` confirms that the v1 draws are unchanged and that the runner's v2 draws equal the harness's on all 300 items; `analysis/retrieval_v2.py` scores the v2 runs beside bare and rule v1 with incident-paired intervals, including the two analogue-only rules against persistence and against each other, all joined by item id.

`run_tooluse.py --models <ids> --conditions bare tool` runs the FPA-FOD tool-use task. The bare arm answers from memory; the tool arm exposes one function, `query_fpafod(sql)`, which accepts a single `SELECT` or `WITH` statement, returns at most 50 rows and 4,000 characters, and may be called up to eight times before the harness asks for the answer without the tool (`gw.call_tools` routes the tool loop to Bedrock, to the gateway's chat completions, or, for the OpenAI reasoning models, to the gateway's Responses API, because the Azure endpoint rejects function tools on chat completions unless reasoning is switched off; the raw output items of each response travel back verbatim so the reasoning items stay with their function calls). It writes `task-tooluse/responses-<model>-{bare,tool}.jsonl` and summarizes into `task-tooluse/scores.json`; `--fake` and `--fake noisy` exercise the harness against the reference queries and write under `task-tooluse/fake/`, which is ignored. `analysis/tooluse_paired.py` scores the arms, reports the paired tool-minus-bare accuracy with a 12-cluster family bootstrap, and counts the failure shapes (the unstated `DISCOVERY_DATE` format behind the calendar-window errors, and answers written as code). All six models ran on 2026-09-17: the open-weight pair through Bedrock (0.083 and 0.071 bare, 0.891 and 0.885 with the tool) and the proprietary four through the gateway (`analysis/tooluse_paired.json` holds every number). The sweep models ran on 2026-09-18. The call after the eighth tool call carries no tool declaration, and Bedrock rejects a history that holds tool blocks without one, so `gw.py` renders that history as text for the final call. None of the six reported models ever reached the eighth call. The 19 rows of three added models that had failed there before the fix were re-queried with `run_tooluse.py --retry-errors`.

`cluster_uncertainty.py` flags a bare-and-grounded pair whose files were written more than 60 minutes apart. The grounded allocation files of the five models that ran before the analogue-date repair were rewritten on 2026-09-16 to replace one repaired item (below; `gpt-6-astra` ran under the final rule), so for allocation pass `--max-skew-min 100000`; the pairing is by item id and was checked.

## Running a model

Gateway models require two environment variables:
- `NAIRR_GATEWAY_URL`: the base URL of the NAIRR gateway endpoint (e.g. `http://<gateway-host>:<port>/v1`). A run that needs the gateway and lacks this variable will fail with a clear message naming the variable.
- `NAIRR_GATEWAY_KEY`: the authorization bearer key for the gateway.

Bedrock models use the standard AWS credential chain, or `AWS_BEARER_TOKEN_BEDROCK`. No key or internal endpoint appears in this repository or in any log.

Running offline analyses, tests, and table reproduction from stored responses does not require either variable.

```bash
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

## Data Sources, Licensing, and Withheld Data Rationale

### Upstream Sources and Terms

AI4Fire benchmarks models across five wildfire data sources. Each source was verified directly on its distribution portal:

1. **Daily Personnel Allocation — ICS-209-PLUS**:
   - *Source*: St. Denis et al. (2023), archived on figshare (DOI: [10.6084/m9.figshare.22303135](https://doi.org/10.6084/m9.figshare.22303135)).
   - *License*: **CC BY 4.0**.
   - *Contents*: Daily situation report filings (1999–2020) filtered by strict temporal precedence.

2. **Wildfire Smoke Detection — FIgLib**:
   - *Source*: HPWREN (High Performance Wireless Research and Education Network), UC San Diego (Dewangan et al., 2022).
   - *License*: **CC BY-NC-ND 4.0**, per HPWREN's data-use conditions (<https://www.hpwren.ucsd.edu/cc.html>). The NonCommercial term requires a separate licence from UC San Diego for commercial use, and the NoDerivatives term covers derived image products, so this repository redistributes neither the frames nor features computed from them.
   - *Release mechanism*: Raw image files are withheld from this repository. The repository provides sequence identifiers, frame metadata, labels, and exact download URLs in the item manifest (`task-figlib/items.jsonl`), alongside an automated fetcher (`build_items_figlib.py`).

3. **Fire Danger Forecasting — Mesogeos Track A**:
   - *Source*: Kondylatos et al. (2023), archived on Zenodo (DOI: [10.5281/zenodo.7473331](https://doi.org/10.5281/zenodo.7473331)).
   - *License*: **CC BY 4.0**.
   - *Contents*: Evaluated on the published 2021–2022 temporal holdout using 24 runnable weather, vegetation, and topography features. Downloadable via `fetch_mesogeos.py`.

4. **Temperature-Grounded Aerial QA — WildFireVQA & FLAME 3**:
   - *Source*: Habibpour et al. (2026), hosted on Hugging Face (`mobiiin/WildFire_VQA`), paired with FLAME 3 aerial imagery.
   - *License*: **Apache-2.0 / CC BY 4.0**. Notice of license discrepancy: the Hugging Face repository metadata records `Apache-2.0` (`"license": "https://choosealicense.com/licenses/apache-2.0/"`), while the dataset card prose under Dataset Summary explicitly specifies `CC-BY-4.0` (`License: CC-BY-4.0`). FLAME 3 thermal and RGB imagery is licensed under **CC BY 4.0** via IEEE DataPort open access and Kaggle mirror. Downloadable via `fetch_flame3.py` and `match_flame3.py`.

5. **Fire Data Tool Use — FPA-FOD**:
   - *Source*: USDA Forest Service Research Data Archive (Short, 2022, 6th Edition, DOI: [10.2737/RDS-2013-0009.6](https://doi.org/10.2737/RDS-2013-0009.6)).
   - *License*: **US Government Public Domain / Open Data** ("can be used without additional permissions or fees; citation is required"). Downloadable via `fetch_fpafod.py` into a 214 MB SQLite database.

### Why Raw Data is Withheld from Git

Raw source datasets (the `data/` directory and raw image folders under `task-figlib/images*`) are intentionally omitted from this Git repository:
- **Redistribution Terms**: FIgLib frames are CC BY-NC-ND 4.0, so third-party redistribution of the raw camera JPEGs and of derivatives of them is restricted; users download frames directly from UCSD HPWREN using the URLs in the item manifest.
- **Repository Size**: Raw sources exceed 50 GB across full Mesogeos NetCDF rasters, FLAME 3 aerial video/thermal frames, and multi-decade FPA-FOD/ICS-209-PLUS databases.
- **Self-Contained Evaluation**: Every evaluation prompt, question, reference answer, and item metadata is fully preserved in the tracked `task-*/items.jsonl` files. All model outputs are tracked in `task-*/responses-*.jsonl`. Furthermore, `analysis/.analogue-pool-cache.json` is shipped so that analogue-based analysis and figures run immediately without needing the multi-gigabyte raw ICS-209-PLUS database. For researchers seeking to retrain baselines or download original rasters from scratch, automated fetch scripts (`fetch_*.py`) and item builders (`build_items_*.py`) are provided.

## License

- **Benchmark Code & Harness**: Released under the **BSD 2-Clause License** (see [LICENSE](LICENSE)). This covers the evaluation harnesses, prompt construction scripts, scoring scripts, analysis pipelines, and figure generators.
- **Task Manifests & Prompts**: BSD 2-Clause covers our original contributions: the prompt templates, the item construction, and the manifest format. Questions, options, and other content imported from a source dataset retain that source's terms and are not relicensed here. WildFireVQA is the unresolved case: its Hugging Face metadata records Apache-2.0 while its dataset card specifies CC BY 4.0, and our BSD grant replaces neither.
- **Stored Model Responses**: Stored outputs from proprietary and open-weight models are released for academic research and reproducibility verification under the terms of their respective upstream input datasets.
