# Model Sweep of 2026-09-18

The benchmark record at commit 6e37924 holds six models on five tasks. This sweep adds 29 Amazon Bedrock models, run on the lab's Bedrock credits with the same prompts, items, and scorers. It changes no reported number of the six-model record. Every new file sits beside the old ones and is listed in its own manifest section.

## Why the Set Grew

The positioning table of the paper compares AI4Fire with the evaluations the survey found: task coverage, models tested, and model families tested. Six models on five tasks led the task column and trailed the largest model sweep, DisasterBench with 22 models. Bedrock serves a wide set of open-weight and proprietary models under one API. The cheapest way to widen the model column was therefore a probe of every chat model the account can reach. The ones whose capabilities fit the tasks then ran through the unchanged harness.

## Capability Probe

`probe_bedrock_caps.py` sends three requests per model through the Converse API in `us-east-1`; the results are in `probes/bedrock-caps-2026-09-18.json`. The three requests are a plain text request expecting a one-word answer, a 64 by 64 red JPEG with a colour question, and one mock tool `get_number` with an instruction to call it. A model passes the tool probe only if the response carries a `toolUse` block. The 33 probed models:

| Model identifier | Text | Image | Tool | Note |
|---|:---:|:---:|:---:|---|
| `amazon.nova-micro-v1:0` | ok | no | ok |  |
| `amazon.nova-lite-v1:0` | ok | ok | ok |  |
| `amazon.nova-pro-v1:0` | ok | ok | ok |  |
| `us.amazon.nova-2-lite-v1:0` | ok | ok | ok |  |
| `us.meta.llama3-3-70b-instruct-v1:0` | ok | no | ok |  |
| `us.meta.llama4-scout-17b-instruct-v1:0` | ok | ok | ok |  |
| `us.meta.llama4-maverick-17b-instruct-v1:0` | ok | ok | ok | one of the six reported models |
| `us.meta.llama3-1-70b-instruct-v1:0` | ok | no | ok |  |
| `us.meta.llama3-1-8b-instruct-v1:0` | ok | no | no | accepts the tool declaration and answers with Python source instead of a call |
| `mistral.mistral-large-3-675b-instruct` | ok | ok | ok |  |
| `mistral.mistral-small-2402-v1:0` | ok | no | ok |  |
| `mistral.ministral-3-8b-instruct` | ok | ok | ok |  |
| `mistral.mixtral-8x7b-instruct-v0:1` | ok | no | no | rejects tool use and system messages; out of the sweep |
| `mistral.devstral-2-123b` | ok | no | ok |  |
| `qwen.qwen3-vl-235b-a22b` | ok | ok | ok | one of the six reported models |
| `qwen.qwen3-32b-v1:0` | ok | no | ok |  |
| `qwen.qwen3-next-80b-a3b` | ok | no | ok |  |
| `qwen.qwen3-coder-30b-a3b-v1:0` | ok | no | ok |  |
| `deepseek.v3.2` | ok | no | ok |  |
| `us.deepseek.r1-v1:0` | ok | no | no | rejects tool use |
| `openai.gpt-oss-120b-1:0` | ok | no | ok |  |
| `openai.gpt-oss-20b-1:0` | ok | no | ok |  |
| `google.gemma-3-27b-it` | ok | ok | no | answers with a fenced `tool_code` block (`print(get_number(name='alpha'))`) instead of a `toolUse` block |
| `google.gemma-3-12b-it` | ok | ok | no | answers with a fenced `tool_code` block (`print(get_number(name='alpha'))`) instead of a `toolUse` block |
| `google.gemma-3-4b-it` | ok | ok | no | answers with a fenced `tool_code` block (`print(get_number(name='alpha'))`) instead of a `toolUse` block |
| `zai.glm-5` | ok | no | ok |  |
| `zai.glm-4.7` | ok | no | ok |  |
| `zai.glm-4.7-flash` | ok | no | ok |  |
| `minimax.minimax-m2.5` | ok | no | ok |  |
| `moonshotai.kimi-k2.5` | ok | ok | ok |  |
| `moonshot.kimi-k2-thinking` | ok | no | ok |  |
| `nvidia.nemotron-super-3-120b` | ok | no | ok |  |
| `writer.palmyra-vision-7b` | ok | ok | no | rejects any tool declaration with a validation error; not run |

Every "no" in the Image column is the same `ValidationException` (the model does not support the image content block).

## Groups

`models.py` records every model with its tier, cap, and tool support. The analysis, table, and figure scripts read it (`--tier core|added|all|text|every`).

1. **Reported six** (tier `core`): `claude-opus-4.8`, `claude-opus-5`, `gemini-3.1-pro`, `gpt-6-astra` through the gateway; Qwen3-VL-235B-A22B and Llama 4 Maverick through Bedrock. Unchanged.
2. **Added full-capability models** (tier `added`, ten models, all five tasks). Seven passed all three probes: Nova Lite, Nova Pro, Nova 2 Lite, Llama 4 Scout, Mistral Large 3, Ministral 3 8B, Kimi K2.5. Gemma 3 27B, 12B, and 4B passed the image probe and use tools through the `tool_code` adapter below.
3. **Text-only sweep** (tier `text`, nineteen models): allocation and fire danger for all, plus tool use for the seventeen whose tool probe passed. The seventeen: Nova Micro, Llama 3.3 70B, Llama 3.1 70B, Mistral Small 2402, Devstral 2 123B, Qwen3 32B, Qwen3 Next 80B, Qwen3 Coder 30B, DeepSeek V3.2, GPT-OSS 120B, GPT-OSS 20B, GLM 5, GLM 4.7, GLM 4.7 Flash, MiniMax M2.5, Kimi K2 Thinking, Nemotron Super 3 120B. Llama 3.1 8B and DeepSeek R1 ran without the tool. The two image tasks (smoke detection, aerial question answering) are out of reach for this group.

With the added group the full-capability set is sixteen models from eight vendors (nine model families: Claude, Gemini, GPT, Nova, Llama, Mistral, Kimi, Qwen, Gemma). With the text-only group the whole record holds 35 models from twelve vendors.

## Harness Changes

- **Gemma 3 tool protocol.** Gemma 3 on Bedrock ignores the Converse tool configuration. When told about tools in the prompt, it calls them through Python calls inside fenced `tool_code` blocks, the Gemma function-calling convention. `gw.bedrock_call_tools` detects the model (`is_gemma_model`) and renders the tool declarations into the system prompt (`render_gemma_tool_specs`). It parses the call from the assistant turn with `ast` (`parse_tool_code_fences`) and returns the result in a `tool_output` block. The runner therefore sees the same tool interface and the same eight-call budget. The usage dictionary of such rows carries `tool_protocol: "tool_code"`. The parser accepts a positional or keyword call of a declared tool, optionally wrapped in `print(...)`; an argument that is not a literal reaches the SQL guard as text. An undeclared attribute call and a JSON-only fence are not calls. A Gemma size that drifts from the convention scores as a protocol failure rather than being repaired by the harness. `tests/test_gemma_toolcode.py` covers the rendering and the parser.
- **A dictionary where a query string belongs.** Gemma 3 12B passed a serialized dictionary as the query on three items of the recorded run, and in the aborted first attempt the runner's SQL guard raised on the dictionary and stopped the run. The runner now serializes a non-string argument before the guard, which rejects it with a message the model can read (`tests/test_tooluse.py`). The three Gemma tool arms were re-run after this fix and after the `tool_protocol` flag reached the row usage.
- **Cap path of the tool loop.** After the eighth tool call the runner asks for the answer without the tool. On Bedrock that final request carried a history with `toolUse` and `toolResult` blocks and no `toolConfig`, which the API rejects. None of the six reported models ever reached the eighth call (`n_tool_calls >= 8` is zero in every recorded tool file), so the record was never affected. Three added models did: Ministral 3 8B (5 rows), Nova 2 Lite (12 rows), and Kimi K2.5 (2 rows). `gw._tool_blocks_as_text` now renders the tool exchange as text blocks for that final call. `run_tooluse.py --retry-errors` re-queried exactly those 19 rows and left every other row untouched.
- **Tool budget counted in calls.** The tool loop grants eight model turns, and a turn may carry several calls, so a model issuing parallel calls could once have run more than eight queries. `run_tooluse.MAX_TOOL_CALLS` now caps executed queries at eight and refuses the rest of a turn that reaches the cap, then asks for the final answer without the tool. The maximum across every recorded tool arm is exactly eight calls, so the guard changes no recorded score (`tests/test_tooluse.py::test_tool_budget_counts_calls_not_turns`).
- **Output cap as a documented variant.** `run_allocation.py`, `run_mesogeos.py`, and `run_tooluse.py` read `AI4FIRE_MAX_OUT` (default 1,536, the benchmark cap). The three reasoning-model runs record their 8,192-token cap in `usage.max_out`, and `models.py` records the configured cap of every model; rows written before that field existed, the core six included, do not carry it. In the ten-item allocation pilots at 1,536, MiniMax M2.5 parsed 2 of 10 bare and 8 of 10 grounded items, Kimi K2 Thinking 2 and 3, and DeepSeek R1 6 and 6. The rest of the output was reasoning that hit the cap before the answer. These three ran the sweep at 8,192, and their rows say so. Every other model kept 1,536.
- **Registry.** The paired analyses, the calibration analysis, the sweep tables, and the model figures read `models.py`. The legacy cluster and table scripts keep their own file discovery, and `analysis/served_models.py` audits manifest membership by design. The tier names are `core` (the six reported models), `added` (the ten sweep models on all five tasks), `all` with the alias `full` (those sixteen), `text` (the nineteen text-only models), and `every` (all 35). The six-model outputs were checked unchanged after the refactor: analysis JSONs equal to the archived ones, and every figure regenerates pixel-identically. `figures/check_reproduction.py` runs that check, calling each generator the way the paper's figures are written (both `--out-pdf` and `--out-png`, the path `fs.savefig` takes at dpi 200) and comparing the result with the copy in `figures/`; all eight figures report identical. A PNG-only invocation writes a review image at dpi 300 with a tight bounding box, which is a different image by design. The FIgLib analysis JSON matches numerically and differs in its generation timestamp.
- **Sixteen-model figure.** `figures/make_grounding_effects.py --tier all` writes `figures/grounding_effects_sweep.{pdf,png}` from a cluster-bootstrap run over every response file (`analysis/cluster-all.stdout.txt`). Two fixes came with it. The interval parser accepted only negative lower bounds, a latent bug the six models never triggered. A point beyond the shared axis, such as Nova 2 Lite's allocation delta, is now clipped at the edge and labelled.

## Pilots

Ten-item pilots preceded every full run. The seven probe-passed models ran allocation, smoke detection, tool use, and aerial question answering (`pilot_tier1.ps1`); all parsed 10 of 10 with no truncation. The twenty text-only candidates ran allocation (`run_tier2.ps1 -Pilot`; `logs/pilot-tier2-2026-09-18.log`). Mixtral 8x7B failed all 20 pilot calls because the endpoint rejects system messages, and left the sweep. The aerial pilots with the `pilot` suffix are under `task-wildfirevqa/pilot/` (ignored by git); the other pilots were overwritten by the full runs.

## Launch

- `run_tier1.ps1`: one detached chain per added model (`run_all.py` for the three original tasks, then `run_tooluse.py`, then `run_wildfirevqa.py`, four workers each). Logs are under `logs/tier1-<model>-2026-09-18.log`. Launched 14:19 for the seven probe-passed models and 14:56 for the three Gemma sizes.
- `run_tier2.ps1`: one detached three-task chain per text-only model, at most seven at once, with `AI4FIRE_MAX_OUT=8192` set for the three reasoning models only. Logs are under `logs/tier2-<model>-2026-09-18.log` and the launcher log is `logs/tier2-launcher-2026-09-18.log`. Launched 14:56. The two reasoning models were the slowest: their chain shells were replaced at 16:50 by per-task processes so the remaining arms could run in parallel, and the last one (MiniMax M2.5, fire danger grounded) finished at 18:56.
- Neither script holds a credential; `gw.py` reads `AWS_BEARER_TOKEN_BEDROCK` from the environment.

## Outcome

`analysis/model_sweep.py` scores every model with the definitions of the paper's main tables and writes `analysis/model_sweep.json`. All 35 models completed every task of their group with zero transport errors after the retries above. The table holds point estimates; the paper's sixteen-model figure adds cluster-bootstrap intervals.

Truncation at the output cap, by model and row count (every row whose `usage.stop_reason` is `max_tokens`, some of which still carry a parsed answer): MiniMax M2.5 261, GPT-OSS 20B 17, Llama 3.1 8B 14, Kimi K2 Thinking 7, Mistral Large 3 5, Qwen3 Next 80B 3, Nemotron Super 3 120B 3, Nova 2 Lite 1, Qwen3-VL 1, GLM 4.7 Flash 1.

MiniMax M2.5 is the outlier even at 8,192 tokens. Its 261 truncated responses are 22 bare and 4 grounded allocation responses, 74 bare and 86 grounded fire-danger responses, and 75 bare tool-use responses; the bare tool-use arm has 78 unanswered responses in all, the three that ended normally included. Its allocation error therefore rests on the 278 bare and 296 grounded predictions that parsed, and its fire-danger AUPRC and fire-call rates on the 312 bare and 300 grounded answers, so those scores can sit above or below the score the complete item set would give. Persistence on the two allocation subsets is 0.143 and 0.146. Tool accuracy counts all 156 items and scores an unanswered response wrong. Every other model marked at most 17 responses as truncated.

| Model | Tier | Cap | Alloc nMAE bare / grd | Smoke recall bare / grd | AUPRC bare / grd | Fire-call bare / grd | Tool bare / tool | Aerial bare / grd |
|---|---|---:|---|---|---|---|---|---|
| claude-opus-4.8 | core | 1,536 | 0.157 / 0.170 | 0.554 / 0.616 | 0.613 / 0.581 | 0.215 / 0.249 | 0.103 / 1.000 | 0.642 / 0.642 |
| claude-opus-5 | core | 1,536 | 0.165 / 0.173 | 0.607 / 0.643 | 0.688 / 0.682 | 0.065 / 0.109 | 0.160 / 1.000 | 0.650 / 0.657 |
| gemini-3.1-pro | core | 1,536 | 0.179 / 0.177 | 0.545 / 0.732 | 0.697 / 0.682 | 0.207 / 0.199 | 0.160 / 1.000 | 0.632 / 0.647 |
| gpt-6-astra | core | 1,536 | 0.161 / 0.220 | 0.643 / 0.741 | 0.620 / 0.590 | 0.044 / 0.078 | 0.000 / 0.994 | 0.623 / 0.635 |
| Nova Lite | added | 1,536 | 0.272 / 0.250 | 0.446 / 0.527 | 0.417 / 0.377 | 0.205 / 0.085 | 0.019 / 0.885 | 0.527 / 0.556 |
| Nova Pro | added | 1,536 | 0.171 / 0.242 | 0.589 / 0.652 | 0.477 / 0.479 | 0.192 / 0.114 | 0.090 / 0.827 | 0.554 / 0.554 |
| Nova 2 Lite | added | 1,536 | 0.163 / 1.687 | 0.321 / 0.223 | 0.442 / 0.478 | 0.137 / 0.119 | 0.058 / 0.865 | 0.488 / 0.507 |
| Qwen3-VL | core | 1,536 | 0.161 / 0.248 | 0.545 / 0.616 | 0.485 / 0.536 | 0.684 / 0.710 | 0.083 / 0.891 | 0.618 / 0.618 |
| Llama 4 Maverick | core | 1,536 | 0.193 / 0.246 | 0.509 / 0.554 | 0.528 / 0.515 | 0.256 / 0.303 | 0.071 / 0.885 | 0.534 / 0.544 |
| Llama 4 Scout | added | 1,536 | 0.230 / 0.246 | 0.384 / 0.482 | 0.473 / 0.460 | 0.648 / 0.642 | 0.032 / 0.910 | 0.588 / 0.566 |
| Mistral Large 3 | added | 1,536 | 0.260 / 0.225 | 0.414 / 0.577 | 0.473 / 0.442 | 0.837 / 0.837 | 0.083 / 0.923 | 0.515 / 0.559 |
| Ministral 3 8B | added | 1,536 | 0.208 / 0.250 | 0.384 / 0.491 | 0.467 / 0.420 | 0.772 / 0.821 | 0.071 / 0.833 | 0.515 / 0.515 |
| Kimi K2.5 | added | 1,536 | 0.173 / 0.254 | 0.420 / 0.545 | 0.518 / 0.529 | 0.446 / 0.427 | 0.115 / 0.942 | 0.583 / 0.613 |
| Gemma 3 27B | added | 1,536 | 0.169 / 0.252 | 0.554 / 0.688 | 0.532 / 0.466 | 0.909 / 0.948 | 0.058 / 0.827 | 0.480 / 0.532 |
| Gemma 3 12B | added | 1,536 | 0.177 / 0.270 | 0.589 / 0.598 | 0.446 / 0.389 | 1.000 / 1.000 | 0.058 / 0.000 | 0.429 / 0.453 |
| Gemma 3 4B | added | 1,536 | 0.147 / 0.175 | 0.536 / 0.607 | 0.347 / 0.339 | 1.000 / 1.000 | 0.058 / 0.000 | 0.392 / 0.404 |
| Nova Micro | text | 1,536 | 0.166 / 0.289 |  | 0.339 / 0.332 | 0.000 / 0.000 | 0.019 / 0.821 |  |
| Llama 3.3 70B | text | 1,536 | 0.927 / 0.248 |  | 0.413 / 0.458 | 0.446 / 0.440 | 0.090 / 0.288 |  |
| Llama 3.1 70B | text | 1,536 | 0.205 / 0.257 |  | 0.400 / 0.427 | 0.145 / 0.202 | 0.006 / 0.000 |  |
| Mistral Small 2402 | text | 1,536 | 0.491 / 1.958 |  | 0.401 / 0.407 | 0.858 / 0.671 | 0.058 / 0.038 |  |
| Devstral 2 123B | text | 1,536 | 0.179 / 0.237 |  | 0.522 / 0.522 | 0.681 / 0.674 | 0.096 / 0.872 |  |
| Qwen3 32B | text | 1,536 | 0.147 / 0.247 |  | 0.429 / 0.458 | 0.782 / 0.821 | 0.077 / 0.840 |  |
| Qwen3 Next 80B | text | 1,536 | 0.153 / 0.226 |  | 0.416 / 0.391 | 0.811 / 0.808 | 0.083 / 0.853 |  |
| Qwen3 Coder 30B | text | 1,536 | 0.151 / 0.240 |  | 0.383 / 0.409 | 0.974 / 0.943 | 0.045 / 0.917 |  |
| DeepSeek V3.2 | text | 1,536 | 0.167 / 0.255 |  | 0.469 / 0.439 | 0.668 / 0.653 | 0.103 / 1.000 |  |
| GPT-OSS 120B | text | 1,536 | 0.166 / 0.242 |  | 0.448 / 0.460 | 0.606 / 0.435 | 0.051 / 0.936 |  |
| GPT-OSS 20B | text | 1,536 | 0.166 / 0.271 |  | 0.438 / 0.392 | 0.412 / 0.252 | 0.058 / 0.917 |  |
| GLM 5 | text | 1,536 | 0.189 / 0.270 |  | 0.570 / 0.566 | 0.280 / 0.311 | 0.103 / 1.000 |  |
| GLM 4.7 | text | 1,536 | 0.184 / 0.239 |  | 0.494 / 0.496 | 0.500 / 0.495 | 0.128 / 0.968 |  |
| GLM 4.7 Flash | text | 1,536 | 0.146 / 0.270 |  | 0.444 / 0.445 | 0.259 / 0.345 | 0.077 / 0.840 |  |
| MiniMax M2.5 | text | 8,192 | 0.156 / 0.244 |  | 0.574 / 0.467 | 0.638 / 0.577 | 0.026 / 0.885 |  |
| Kimi K2 Thinking | text | 8,192 | 0.193 / 0.253 |  | 0.560 / 0.511 | 0.650 / 0.507 | 0.058 / 0.891 |  |
| Nemotron Super 3 120B | text | 1,536 | 0.150 / 0.243 |  | 0.529 / 0.490 | 0.306 / 0.508 | 0.096 / 0.910 |  |
| Llama 3.1 8B | text | 1,536 | 0.175 / 0.300 |  | 0.334 / 0.325 | 0.000 / 0.000 |  |  |
| DeepSeek R1 | text | 8,192 | 0.173 / 0.260 |  | 0.472 / 0.463 | 0.899 / 0.868 |  |  |

Tallies over the sixteen full-capability models (the paper's Section 4.6):

- Allocation: no model beats persistence (0.146) bare or grounded; bare error runs from 0.147 to 0.272, and analogues raise it on 13 of 16. Nova 2 Lite's grounded 1.687 comes from one item, a displayed analogue of 866 personnel copied onto an incident staffed by two (0.248 without it).
- Smoke detection: the reference frame raises recall on 15 of 16 and lowers it on 1 (Nova 2 Lite); the false-positive rate rises on 10.
- Fire danger: bare AUPRC runs from 0.347 to 0.697; 2 models outrank the temperature rule (0.654); climatology raises AUPRC on 4 of 16. Bare fire-call rates run from 0.044 to 1.000 against the 0.339 base rate; Gemma 3 12B and 4B call fire on every item.
- Tool use: the tool raises accuracy on 14 of 16; 10 reach 0.88 or above. Gemma 3 12B and 4B stay at 0.000 because neither executed a valid database query under the declared interface. Of 12B's 148 zero-call items, 118 mention the undeclared `query_fpafod.query`; its other eight items carry 24 rejected calls, seven serialized dictionaries on three items and 17 literal `sql` arguments on five. Gemma 3 4B writes JSON inside the fence on all 156.
- Aerial question answering: 3 models bare and 4 grounded exceed the 0.627 majority baseline, all among the four gateway models; the thermal block raises accuracy on 11 of 16.

Tallies over the nineteen text-only models (the paper's Appendix D.5):

- Allocation: bare error runs from 0.146 to 0.927; analogues raise it on 18 of 19. Below persistence bare: GLM 4.7 Flash (persistence copied on nearly every item; see below). Llama 3.3 70B's bare 0.927 comes from one prediction of 412 for an incident staffed by two (0.244 without it). Mistral Small 2402's grounded 1.958 comes from the same 866-personnel analogue that caught Nova 2 Lite (0.520 without it).
- Fire danger: bare AUPRC runs from 0.334 to 0.574, with 0 models above the temperature rule. Bare fire-call rates run from 0.000 to 0.974; Nova Micro and Llama 3.1 8B never call fire.
- Tool use: the tool raises accuracy on 15 of 17; 9 reach 0.88 or above. Below 0.5 with the tool: Llama 3.3 70B 0.288, Llama 3.1 70B 0.000, Mistral Small 2402 0.038. Llama 3.1 70B never called the tool and answered from memory. Llama 3.3 70B wrote its call as JSON text in the message on 108 of 156 items, and Mistral Small 2402 wrote the SQL in prose beside a placeholder answer.
- Persistence copying: across all 35 models, five return the previous day's count on at least 290 of 300 bare allocation items (Qwen3 Coder 30B 299, Gemma 3 4B 299, Qwen3 32B 297, Nemotron Super 3 120B 294, GLM 4.7 Flash 294). Their bare error therefore sits within a few thousandths of persistence.

## Verification Analyses Added After the Round-10 Review

- `analysis/wildfirevqa_intervals.py` writes `analysis/wildfirevqa_intervals.json`: frame-clustered (390 frames) and
  question-clustered (34 families) 95 percent intervals for the sixteen full-capability models on aerial question
  answering, 20,000 resamples at seed 20260915. The core six reproduce `analysis/wildfirevqa_paired.py --tier core`
  to every printed digit. Frame clustering is the unit of dependence, since items on one frame share altitude,
  viewpoint, thermal calibration, and smoke occlusion.
- `analysis/text_sweep_intervals.py` writes `analysis/text_sweep_intervals.json`: clustered intervals for all three
  arms of the nineteen text-only models, with the units of the main tables (incidents, one-degree cell by month
  blocks, question families).
- `analysis/sweep_meta.py` writes `analysis/sweep_meta.json` and `figures/sweep_meta.{pdf,png}`: the cross-model
  questions a six-model record cannot answer, including the correlation between persistence copying and bare
  allocation error, the grounding effect against bare skill, within-family size contrasts, and a rank test between
  open-weight and proprietary models. The figure is a code-record artifact; the paper does not include it.
- An independent audit unit recomputed every number of Table 12, Table 19, and the two new subsections from the
  response files, without using `analysis/model_sweep.py`: 465 of 466 claims matched, and the one mismatch was the
  headline total of scored responses (77,644 under the paired smoke rule, not 77,640).

## Round-11 Repairs in the Code Record

- **A limited retry kept only the limited slice.** `run_tooluse.py --retry-errors --limit N` rebuilt the row list
  from the truncated item list, so a ten-item retry over a 156-row file would have written ten rows. The runner now
  keeps every recorded row in item order (`tests/test_tooluse.py::test_retry_with_limit_keeps_every_recorded_row`).
  No recorded file was affected: the 19-row retry of 2026-09-18 ran without `--limit`, and `build_manifest.py --check`
  reports zero changed files.
- **`--tier text` crashed.** The tally block called `min` on an empty sequence for the two image tasks. Each task now
  reports `{"models": 0}` when the tier holds none of them.
- **The meta-analysis needed a gitignored cache.** `analysis/.analogue-pool-cache.json` is ignored, so a fresh clone
  could not run `analysis/sweep_meta.py`. It now rebuilds the analogue pool from the recorded task sources, the way
  `copy_agreement.py` does, and writes the cache back. With the cache removed, the regenerated JSON is identical.
- **The permutation null.** `question_1_grounding_vs_skill` now reports a shared-baseline null beside each
  correlation: the same statistic with the grounded scores permuted across models, 5,000 draws at seed 20260915. For
  allocation over 35 models the null median is -0.620 with a central range of [-0.803, -0.389], which contains the
  observed -0.430, so that correlation carries no evidence about which models retrieval helps.
- **Per-row cap metadata.** 44,476 of the 78,092 canonical rows lack `usage.max_out`, the core six included, because
  the field arrived with the sweep. The README and this record now say what the rows actually carry.
- **Generated comments.** `table_rows_new_models.py` dates each row by its model's tier rather than one hard-coded
  date, and no longer calls every smoke pair 196 items.

## Round-12 Repairs in the Code Record

- **The expanded table generator read stale cached scores.** `table_rows_new_models.py` took MAE, normalized
  error, and the persistence share from `task-allocation/scores.json` when an entry existed, and those entries
  predate the current scorer. After the generator gained the core roster, its rows disagreed with the paper
  (Opus 4.8 bare printed 14.50 and 0.160 against the recorded 14.31 and 0.157). Every column now comes from the
  responses, and the generator reproduces the paper's allocation table.
- **The figure checker states its scope.** `figures/check_reproduction.py` requires each generator to write the
  PDF the paper includes, reports that it compares PNG pixels, and accepts `--paper-figures <dir>` to compare
  the paper's copy of each figure as well. All eight figures are identical in both repositories.
- **Matched comparators are exported.** `analysis/text_sweep_intervals.py` scores each model on the items parsed
  in both arms, so a comparison against the full-set persistence value or the full-set temperature rule mixes
  populations. It now exports `allocation_paired_bare_minus_persistence` and
  `mesogeos_bare_vs_matched_temperature_rule`, the latter with the rule recomputed on each model's scored subset
  (`temperature_rule_ap`, the raw last-day t2m ranking that scores 0.654 on all 386 items). The fixed-reference
  tallies remain under `*_vs_fixed_*` keys. `cluster_uncertainty.prepare_mesogeos` now also returns `kept`, the
  item ids it scored, which is what makes the matched comparator possible.

## Review Outcome

Four `/vet` rounds covered this sweep. Round 10 returned BLOCK with twelve findings, round 11 BLOCK from Codex
and PASS from Antigravity, round 12 BLOCK on one table-generator defect, and round 13 PASS with no findings and
nothing left open. The round-13 reviewer reproduced the core allocation table from the responses, scored the
matched temperature comparator independently, and confirmed that no reported number moved: the 36 reported and
17 supporting manifest entries and all 60 core response files are byte-identical to the previous commit.

## Archive

Before the sweep touched the tree, `git archive` of commit 6e37924 was written to `fire-bench-archive/AI4Fire-6e37924-six-models-2026-09-18.zip` outside the repository (375 entries, 115 response files, 9 scores files). The 53 manifest checksums verified against it with 0 problems. The seven gitignored analysis JSONs of the six-model record sit beside it under `analysis-json-2026-09-18/`. The six-model record is also the committed history itself.
