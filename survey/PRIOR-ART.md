# Prior Art: Does a Survey-Plus-Benchmark of AI on Fire Tasks Exist?

The goal being checked is a work that does two things. It surveys what is already known about
large language models (LLMs) and AI agents on fire tasks. It also evaluates current LLMs and agents
across many of those tasks, with per-task results.

Two search passes ran on 2026-09-13, and every listed work was confirmed by fetching its page.

- **Pass 1** looked at AI on fire tasks broadly. Two independent panels (Claude and Gemini) each
  worked four angles. The 102 merged works are in `prior-art-2026-09-13.json`.
- **Pass 2** looked only at LLMs and agents on fire tasks, first public in 2025 or 2026. Thirteen
  Agy (Gemini) search units covered it, and three more units re-fetched the 31 records this file
  cites. The 138 merged works are in `prior-art-r2-2026-09-13.json`.

## Answer

**Not as a single piece of work, after both passes.** Pass 1 found no survey plus benchmark of AI
on fire tasks. Pass 2 narrowed the question to LLMs and agents, searched 13 angles, and all 13
units reached the same answer. The nearest works either evaluate one task family, evaluate many
tasks without surveying fire work, or survey without evaluating.

## Pass 1: AI Broadly

**Surveys are plentiful and run no experiments.** Jain et al. 2020 (Environmental Reviews) remains
the reference taxonomy across six wildfire-ML domains. Later reviews cover decision support, UAVs,
risk prediction, spread models, and a 2026 global review of spatial-AI prediction. None of them tests
a model.

**Benchmarks are plentiful and cover one to three tasks.** Mesogeos (NeurIPS 2023 Datasets and
Benchmarks) covers fire danger and burned area. TS-SatFire (Scientific Data 2025) covers detection,
burned area, and next-day progression with 11 models. SeasFire, WildfireSpreadTS, and the
EO4WildFires severity study sit in the same satellite-perception family. None surveys the wider task
landscape.

**LLM and agent evaluations are narrow.** Chen et al. 2025 (arXiv:2510.12061) grounds about 14 LLM
configurations in geospatial data to predict daily personnel and budget for California wildfires.
SmokeBench evaluates multimodal LLMs on smoke detection. CREW-Wildfire is a multi-agent
collaboration benchmark in a procedurally generated environment. Each covers one or two tasks.

| Work | What it does | What it leaves undone |
|---|---|---|
| WILDFIRE-FM, Xu et al. 2026 (arXiv:2605.18911) | Pretrains a wildfire foundation model and compares it with ten Earth foundation models across occupancy, spread, retrieval, and regression tasks under a fixed evaluation contract | Only Earth foundation models; no LLMs or agents; forecasting tasks only; no landscape survey |
| Land8Fire, Tran et al. 2025 | Pairs a comprehensive review with a new dataset and extensive benchmarking | One task, segmentation |
| TS-SatFire, Zhao et al. 2025 | Three satellite tasks, 11 models, a landscape section | Satellite perception only; no decision, retrieval, or language tasks |
| Chen et al. 2025 (arXiv:2510.12061) | Many LLMs on wildfire resource prediction | Two related tasks, California wildfires only |

WILDFIRE-FM was confirmed directly on arXiv (submitted 2026-05-14). It is the nearest competitor on
the benchmark half for non-LLM models. It also argues that transfer conclusions depend strongly on
evaluation design, which any cross-task comparison will have to address.

## Pass 2: LLMs and Agents, 2025 and 2026

### How It Was Run

Eight units searched by topic. Three covered text LLMs on fire knowledge and documents,
vision-language models on fire imagery, and agents for decision support and operations. The other
five covered simulation-coupled agents, geospatial and Earth-observation LLMs, crisis
communication, structure fire and the fire service, and surveys of LLMs in disaster management.
Five more units swept sources instead: machine learning venues; NLP, vision, and climate workshops;
domain journals and ISCRAM; regional and non-English literature; and government, industry, and grey
literature.

Every unit worked from one written contract. It had to fetch a confirming page for each work and
could not fill authors, dates, venues, model names, or numbers from memory. It tagged each work
with the task categories it evaluates and scored its closeness to this project from 0 to 3. The
machine learning venue unit ended without output on its first run and was dispatched again.

Afterward, three re-check units fetched the pages of the 31 records cited below. For each record
they checked every claim against the page and recorded whether it included a fire task.

### The Closest Works

| Work | What it does | What it leaves undone |
|---|---|---|
| CREW-Wildfire, Hyun et al. 2025 (arXiv:2507.05178) | Benchmarks four LLM multi-agent frameworks, run on GPT-4o, in a procedurally generated wildfire environment | One task family, operations in simulation; one base model; no survey |
| DisasterBench, Zhang et al. 2026 (arXiv:2606.06217) | Evaluates 21 multimodal LLMs on UAV disaster-response tasks, including wildfire propagation | Disasters in general and UAV imagery only; no survey of fire work |
| Chen et al. 2025 (arXiv:2510.12061) | About 14 LLM configurations on wildfire personnel and cost allocation, with and without geospatial grounding | Two related decision tasks |
| OGC engineering reports D-123 (Tricomi, 2025) and 25-012 (Tricomi, 2026) | Map generative AI and agent workflows onto wildland fire management, and assess more than 200 Canadian wildfire datasets for use with generative AI | No model evaluation across tasks |

### What the 2025 and 2026 Literature Looks Like

The units returned 138 fetch-confirmed works, and 117 of them were absent from pass 1. By kind,
75 are systems with an evaluation, 21 are evaluations, 14 are benchmarks, 14 are surveys, 8 are
position papers, 5 are datasets, and 1 is a system without evaluation. A keyword screen finds an
LLM, a vision-language model, or an agentic LLM framework in 109 of the 138. Most of the other 29
are reinforcement learning, CNN, or Earth-observation foundation model work, or surveys of AI in
general, which the units included despite the contract. The screen also misses a few LLM systems
whose records never name a model.

**Most evaluations test one system.** Of the 90 LLM works that carry an evaluation, the units could
not determine how many models 34 of them tested. By the units' counts, 12 test one model and 26
test two or three. Only 16 test four to nine, and 2 test ten or more. The usual design is one
proposed system against baselines its authors chose.

**The work is spread unevenly across task categories.** A work can carry several tags. Decision
support and operations carry 71, detection and perception 52, geospatial analysis 37, and document
understanding 35. Simulation-coupled agents and communication each carry 30, knowledge question
answering 26, forecasting 20, and data retrieval and tool use 10. The units assigned these tags,
and nobody re-checked them.

**Structure fire and wildfire are separate literatures.** Code compliance, fire protection
specifications, and fire investigation are almost entirely structure-fire work, much of it from
Chinese and Korean groups. Imagery, operations, and forecasting are almost entirely wildfire
work. In the unit records, no LLM evaluation covers both.

**The surveys split along the same line as the project's two halves.** The 14 surveys and reviews
fall into two groups. Surveys of LLMs in disaster management treat fire as one hazard among
several. Reviews centered on fire, such as those on wildland-urban interface management, wildfire
prediction, and building code compliance, cover AI broadly, and LLMs are absent or minor in them.
By the units' reading, none of the 14 runs its own model evaluation.

The machine learning venue sweep alone found three NeurIPS 2025 Datasets and Benchmarks entries on
fire perception: DetectiumFire, Fire360, and RSCC. No topic unit found them, and they were not
re-checked.

### What Has Been Found

These patterns rest only on the re-checked records, and every number below was read from a
fetched page.

**Grounding helps wherever it has been tried.** Direct LLM scoring of wildfire risk sub-criteria
correlated with expert rankings at R = 0.009, and indicator-guided retrieval prompting raised the
correlation to R = 0.598 (Cheng et al. 2026, Fire). On post-wildfire damage assessment, GPT-4o and
LLaVA alone scored 0%, while IC-EO, an agent that writes code calling Earth-observation models,
scored 50% (arXiv:2602.00117). In Chen et al. 2025, text-only agents over-allocated personnel two
to three times and under-allocated cost by more than four times, and geospatial grounding beat
LSTM and physical baselines. For fire codes, RegCheck-Hybrid reached 89% F1 by applying rules first
and calling an LLM only as a fallback. Sun et al. 2026 (Fire Technology) removed hallucinated
numerical thresholds by aligning regulations to structured records before summarizing. Each of
these compares the authors' own system with baselines they chose. No study compares grounding
strategies across tasks on one model set.

**Quantitative forecasting is the weakest case measured.** On fire radiative power prediction, the
retrieval-augmented agent WildfireGPT reached an MAE of 14.849 and an R² of 0. TabNet, a tabular
deep model, reached an MAE of 0.055 and an R² of 0.405 (Ramesh et al. 2025, Natural Hazards).

**Recognition is close to solved, while localization and physical measurement are not.** Zero-shot
active-fire classification on 500 satellite images reached 0.98 accuracy with Gemini 2.5 Flash,
0.966 with Llama 3.2-Vision, and 0.728 with LLaVA (Kohli et al. 2025, TechRxiv). In DisasterVQA
(ICWSM 2026), seven vision-language models scored 0.93 to 1.00 on yes-or-no fire questions but made
errors and hallucinated on quantitative estimates. SmokeBench (WACV 2026) found that multimodal
LLMs classify large plumes well and localize early-stage smoke poorly, with performance tracking
smoke volume more than contrast. WildFireVQA (CVPR 2026 workshops) posed 207,298 aerial RGB and
thermal questions to four multimodal LLMs and found clear deficits in temperature-grounded
reasoning.

**Small tuned models can match general frontier models on narrow perception.** ForestFireVLM-7B, a
fine-tuned Qwen2.5-VL-7B, described UAV forest fire scenes with 76.6% accuracy against 65.5% for
Gemini 2.0 Pro (Seidel et al. 2025, Drones). In DisasterBench, a 2B domain-tuned model matched the
reasoning accuracy of GPT-4o at 72.60%.

**Multi-agent coordination breaks down as tasks grow.** In CREW-Wildfire, simple tasks succeeded,
but large suppression and rescue tasks often scored below a do-nothing baseline because plans did
not adapt and agents were lost. In the MOASEI wildfire suppression track at AAMAS 2025, an LLM
meta-optimization loop adapted policies through feedback, and GNN and neural policies still earned
higher cumulative reward.

**LLMs already work as analysts over fire records.** A prompted GPT-4 sorted 24,254 text entries
from 6,630 incidents in the Wildland Fire Decision Support System into 13 barrier types (Epstein et
al. 2025, International Journal of Wildland Fire). Roads (42%) and burn scars (26%) were the most
common.
Task-state monitoring over firefighter incident-command transcripts stayed strong when run
incrementally, and its errors concentrated in unit-assignment timing and completion outcomes
(Grünert et al. 2026, SwissText). MAGR-FI (ISCRAM 2026), a three-agent game-theoretic reasoning
framework, raised fire investigation causal reasoning scores from 6.95 to 8.51 over 1,051 real
cases.

**Model choice often trails the frontier.** WildfireVLM, posted in February 2026, compared GPT-4o
with Claude 3.5 Sonnet (arXiv:2602.13305). WildFireVQA tested only open models of 8B parameters or
fewer. Four of the re-checked fire works include GPT-5-class models: Chen et al. 2025, FireScope
(CVPR 2026), DisasterBench, and Grünert et al. 2026. A weakness measured on an older or smaller
model says little about a current frontier model, so results in this literature date quickly.

## The Gap

No work surveys what is known about LLMs and agents across fire tasks and then evaluates one set of
current models across those tasks under one protocol. Pass 2 sharpens that gap in three places.
Nearly every system paper reports that grounding helps, yet no study measures bare and grounded
versions of the same models across tasks. No evaluation spans structure fire and wildfire.
Forecasting and data retrieval carry the least LLM work, and the one direct forecasting comparison
found the LLM agent unreliable.

## How Much to Trust These Results

**Neither pass is saturated.** In pass 1 the two panels overlapped on only 22 of 102 works. In pass
2, 117 of 138 works were new relative to pass 1, and only 50 of the 138 surfaced in two or more
units. The machine learning venue sweep also found benchmarks that no topic unit reached.

**The negative answer is better supported than any count.** Thirteen angles agree, and the closest
works surfaced in more than one unit: CREW-Wildfire in two, DisasterBench in two, and SmokeBench in
five.

**Per-work details in the JSON are leads until re-checked.** All 31 re-fetched records exist, but 10
had a problem. Four carried a claim the page contradicted or did not contain. Five were only partly
supported, for example a wrong model list or a wrong description of the method. Six turned out to
include no fire task at all, and these groups overlap. The re-check made one error of its own, an
undercounted model list that a second fetch caught. Works carrying a `recheck` object in
`prior-art-r2-2026-09-13.json` hold the corrected values. The category tags and closeness scores are
unit judgments that nobody re-checked.
