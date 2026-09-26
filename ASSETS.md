# Data Assets and Licensing in AI4Fire

This document records, for every data source the AI4Fire benchmark uses or audited, the upstream terms, what this repository redistributes, and how the rest can be fetched.

---

## 1. Asset-Level License Table

| Source Name | Upstream Terms & Source URL | Derived Files Redistributed | Terms for Derived Redistribution | Recommendation |
|---|---|---|---|:---:|
| **FIgLib** (Wildfire Smoke Detection) | **CC BY-NC-ND 4.0**, listed for HPWREN program data on HPWREN's [data-use page](https://www.hpwren.ucsd.edu/cc.html); the FIgLib page itself names no license and says "its use requires a credit reference to https://www.hpwren.ucsd.edu/ in derivative work". [HPWREN FIgLib Portal](https://www.hpwren.ucsd.edu/HPWREN-FIgLib/) / [Data Index](https://cdn.hpwren.ucsd.edu/HPWREN-FIgLib-Data/index.html) | `task-figlib/items.jsonl` (224 frames, 196 paired), `task-figlib/responses-*.jsonl`, `task-figlib/scores.json`, `task-figlib/figlib-paired.json` | Source terms (CC BY-NC-ND 4.0) with credit to HPWREN / UC San Diego; contains sequence metadata, timing offsets, labels, and model answers (no raw image pixels and no features computed from them). | **Ship metadata & responses; exclude raw images; ship fetch script** (`build_items_figlib.py`) |
| **WildFireVQA & FLAME 3** (Aerial Question Answering) | **Conflicting upstream terms on WildFireVQA**: Hugging Face metadata declares `Apache-2.0`, while the dataset card text declares `CC-BY-4.0`. [WildFireVQA Hugging Face](https://huggingface.co/datasets/mobiiin/WildFire_VQA). **FLAME 3 Imagery**: `CC BY 4.0` via [IEEE DataPort](https://doi.org/10.21227/w0mz-aq48) (requires free IEEE login) and [Kaggle Mirror](https://www.kaggle.com/datasets/brycehopkins/flame-3-computer-vision-subset-sycan-marsh). | `task-wildfirevqa/items.jsonl` (408 QA items with radiometric thermal statistics), `task-wildfirevqa/image-map.json`, `task-wildfirevqa/responses-*.jsonl`, `task-wildfirevqa/scores.json`, `task-wildfirevqa/source-prompt-template.txt` | Apache-2.0 / CC BY 4.0 with attribution to Habibpour et al. (2026) and FLAME 3 (IEEE DataPort). Contains prompt texts, derived thermal summaries, and model responses (no raw aerial images). | **Ship QA items & responses; exclude raw imagery; ship fetch script** (`fetch_flame3.py`) |
| **Mesogeos** (Fire Danger Forecasting) | **CC BY 4.0** on [Zenodo](https://doi.org/10.5281/zenodo.7473331) and [GitHub](https://github.com/Orion-AI-Lab/mesogeos); data hosted on [Google Drive](https://drive.google.com/drive/folders/1aRXQXVvw6hz0eYgtJDoixjPQO-_bRKz9). Note: Website text carries CC BY-SA 4.0. | `task-mesogeos/items.jsonl` (the 4,120-sample 2021–2022 holdout that contains the 386 scored items), `task-mesogeos/responses-*.jsonl`, `task-mesogeos/scores.json`, `mesogeos-common-items.json` | CC BY 4.0 with attribution to Kondylatos et al. (2023). Contains derived text prompt formulations and model outputs. | **Ship items & responses; exclude 648 GB datacube; ship fetch script** (`fetch_mesogeos.py`) |
| **ICS-209-PLUS** (Daily Personnel Allocation) | **CC BY 4.0** on [figshare](https://doi.org/10.6084/m9.figshare.19858927.v3); supporting GAL code is MIT on [GitHub](https://github.com/defene/GAL). | `task-allocation/items.jsonl` (the 4,320-fire-day pool), `task-allocation/items-v1.jsonl` (the 300 scored fire-days), `task-allocation/responses-*.jsonl`, `task-allocation/scores.json`, `baselines/allocation_trained.json`, `retrieval-v2/` (analogues under rule v2) | CC BY 4.0 with attribution to St. Denis et al. (2023). Contains incident metadata, staffing figures, and retrieved historical analogues. | **Ship items, analogues, & responses; exclude 48 MB raw zip; ship build script** (`build_items_ics209.py`) |
| **FPA-FOD** (Fire Data Tool Use) | **US Government Open Data / Public Domain** ("collected using funding from the U.S. Government and can be used without additional permissions or fees", citation required). [Forest Service Research Data Archive](https://doi.org/10.2737/RDS-2013-0009.6). | `task-tooluse/items.jsonl` (156 natural language database questions across 12 families, reference SQL queries, gold answers), `task-tooluse/responses-*.jsonl` (including tool call traces), `task-tooluse/scores.json` | Public Domain / US Government Open Data terms with required citation to Short (2022). | **Ship questions & traces; exclude 214 MB SQLite binary; ship fetch script** (`fetch_fpafod.py`) |
| **Fire360** (Withdrawn Structure-Fire Task) | **Disputed terms**: Paper claims a custom "research-only MIT license with added restrictions on surveillance, enforcement, behavioral profiling, or nonconsensual monitoring". Shipped `LICENSE.txt` on [Box](https://uofi.box.com/v/fire360dataset) is standard MIT (copyright 2025 Aditi Tiwari). | `task-fire360/items.jsonl`, `task-fire360/items-index.csv`, `task-fire360/baselines.json`, `build_items_fire360.py` | Withdrawn task. Shipped prompts provide no mapping to visual evaluations (doctrine trivia only). | **Exclude completely from anonymized release** (drop `task-fire360/` and `build_items_fire360.py`) |

---

## 2. In-Depth Source Verification and Distribution Rationale

### 2.1 FIgLib (Wildfire Smoke Detection)
- **Upstream Location**: Hosted by the High Performance Wireless Research and Education Network (HPWREN) at UC San Diego (`https://www.hpwren.ucsd.edu/HPWREN-FIgLib/` and `https://cdn.hpwren.ucsd.edu/HPWREN-FIgLib-Data/`).
- **License Status**: HPWREN's data-use page (<https://www.hpwren.ucsd.edu/cc.html>) says that "UC San Diego HPWREN program data is copyrighted under a CC BY-NC-ND 4.0 Creative Commons License" and grants no commercial use. The FIgLib page itself names no license and states two conditions:
  1. *"This data is provided as-is, no guarantees for anything is included."*
  2. *"While we are making the data publicly available, its use requires a credit reference to https://www.hpwren.ucsd.edu/ in derivative work."*
- **Redistribution Analysis**: CC BY-NC-ND 4.0 allows non-commercial sharing of unmodified material with attribution and forbids sharing adapted material. As the README states, this repository redistributes neither the frames nor features computed from them.
- **Release Strategy**:
  - **Do Not Ship Raw Images**: No frame JPEGs are tracked in the repository or shipped in the artifact export.
  - **Ship Metadata & Labels**: `task-figlib/items.jsonl` distributes only derived metadata: camera name, sequence ID, timestamp, offset seconds from ignition, and the ground-truth binary label (`smoke` vs. `no smoke`). These fields follow the source's terms, with credit to HPWREN / UC San Diego.
  - **Ship Stored Responses**: `task-figlib/responses-*.jsonl` contain model prediction outputs and tokens, which are essential for offline reproduction.
  - **Ship Fetch Script**: `build_items_figlib.py` programmatically downloads the required frames directly from HPWREN's CDN for users who wish to run new evaluations.

### 2.2 WildFireVQA & FLAME 3 (Aerial Question Answering)
- **Upstream Location**:
  - Questions and annotations: Hugging Face dataset `mobiiin/WildFire_VQA` (`https://huggingface.co/datasets/mobiiin/WildFire_VQA`).
  - Imagery: FLAME 3 UAV dataset on IEEE DataPort (DOI: 10.21227/w0mz-aq48) and Kaggle mirror (`brycehopkins/flame-3-computer-vision-subset-sycan-marsh`).
- **Conflict Verification**:
  - The Hugging Face dataset metadata tag specifies `license: apache-2.0`.
  - The dataset card prose (`README.md`) on Hugging Face states `CC-BY-4.0`.
  - The underlying FLAME 3 imagery on IEEE DataPort records `CC BY 4.0` in its metadata.
  - *Resolution*: Both Apache-2.0 and CC BY 4.0 are permissive open licenses permitting derivative research datasets, commercial and non-commercial reuse, and redistribution subject to proper attribution.
- **Release Strategy**:
  - **Ship Derived Items**: `task-wildfirevqa/items.jsonl` (408 items) provides question text, options, answers, radiometric thermal summaries (`temp_summary`), and applicability scores with clear attribution.
  - **Ship Model Responses**: `task-wildfirevqa/responses-*.jsonl` contains the full set of bare and grounded model answers.
  - **Exclude Raw Aerial Imagery**: The multi-gigabyte FLAME 3 raw RGB and thermal TIFF archives are excluded from the repository.
  - **Ship Fetch Script**: `fetch_flame3.py` downloads the FLAME 3 computer-vision subset via Kaggle API / IEEE DataPort for users requiring image pixels.

### 2.3 Mesogeos (Fire Danger Forecasting)
- **Upstream Location**: Zenodo record (DOI: 10.5281/zenodo.7473331), GitHub (`Orion-AI-Lab/mesogeos`), Google Drive folder `1aRXQXVvw6hz0eYgtJDoixjPQO-_bRKz9`.
- **License Status**: `CC BY 4.0` confirmed on Zenodo, arXiv (2306.05144), and repository README. The project website carries a CC BY-SA 4.0 notice for the website presentation.
- **Redistribution Analysis**: CC BY 4.0 explicitly authorizes creating and redistributing derivative works with attribution.
- **Release Strategy**:
  - **Ship Derived Benchmark Items**: `task-mesogeos/items.jsonl` holds the 4,120 samples of the 2021–2022 temporal holdout, which contain the 386 scored items; each item formats the 24 runnable driver features into prompt text.
  - **Ship Stored Responses**: `task-mesogeos/responses-*.jsonl` stores all raw predictions and stated probabilities.
  - **Exclude Raw Datacube**: The 648 GB Zarr datacube is excluded.
  - **Ship Fetch Script**: `fetch_mesogeos.py` provides automated download of the 0.9 GB Track A tabular CSV files from Google Drive.

### 2.4 ICS-209-PLUS (Daily Personnel Allocation)
- **Upstream Location**: figshare repository (DOI: 10.6084/m9.figshare.19858927.v3), St. Denis et al. (2023), Scientific Data.
- **License Status**: `CC BY 4.0` confirmed via figshare API. The accompanying GAL reference modeling repository on GitHub is licensed under the MIT License.
- **Redistribution Analysis**: Fully permissive for derived items and benchmarks under attribution.
- **Release Strategy**:
  - **Ship Derived Items & Analogues**: `task-allocation/items.jsonl` (4,320 fire-day situation records, including the 300 scored in `items-v1.jsonl`) and `retrieval-v2/` (analogues under rule v2) are redistributed under CC BY 4.0.
  - **Ship Model Responses**: `task-allocation/responses-*.jsonl` contains all bare and grounded predictions.
  - **Exclude Raw Bulk Archive**: The 48.7 MB raw `ics209plus-wildfire.zip` archive is excluded.
  - **Ship Build Script**: `build_items_ics209.py` documents the deterministic extraction and filtering rules from upstream files.

### 2.5 FPA-FOD (Fire Data Tool Use)
- **Upstream Location**: USDA Forest Service Research Data Archive (Short 2022, 6th Edition, DOI: 10.2737/RDS-2013-0009.6).
- **License Status**: U.S. Government Open Data / Public Domain. The archive terms explicitly state: *"These data were collected using funding from the U.S. Government and can be used without additional permissions or fees"*, requiring standard scientific citation.
- **Release Strategy**:
  - **Ship Question Suite & Traces**: `task-tooluse/items.jsonl` (156 natural language questions, gold SQL statements, ground-truth results) and `task-tooluse/responses-*.jsonl` (full SQL tool-call execution traces) are shipped.
  - **Exclude 214 MB Database Binary**: The raw SQLite database file is excluded from git tracking.
  - **Ship Fetch Script**: `fetch_fpafod.py` downloads the official SQLite database from the Forest Service archive and records its SHA-256.

### 2.6 Fire360 (Withdrawn Structure-Fire Task)
- **Upstream Location**: University of Illinois Box folder `https://uofi.box.com/v/fire360dataset`, Tiwari et al. (2025), NeurIPS 2025 Datasets and Benchmarks track.
- **Status & Defects**:
  - Withdrawn from the AI4Fire evaluation suite (documented in the paper's appendix on defects found while building the tasks).
  - While the paper describes an evaluation grounded in 360-degree training video, the released file (`prompt_templates.json`) contains only 50 text trivia questions and 50 free-text duplicates without any timestamps, frame IDs, or visual grounding.
  - Upstream licensing statements conflict: the paper asserts a custom "research-only MIT license with added restrictions on surveillance, enforcement, behavioral profiling, or nonconsensual monitoring", whereas the Box folder's `LICENSE.txt` is standard MIT.
- **Release Strategy**:
  - **Exclude from Anonymized Release**: The `task-fire360/` directory and `build_items_fire360.py` are strictly excluded from the anonymized benchmark release. They form no part of the five active wildfire tasks, are excluded from `manifest-v1.json`, and do not participate in offline reproduction.

---

## 3. Summary of Release Decisions

| Component Type | Included in Anonymized Release | Omitted / Shipped via Fetch Script |
|---|---|---|
| **Benchmark Manifests & Items** | Yes (`task-*/items.jsonl`, `retrieval-v2/`) | None |
| **Stored Model Responses** | Yes (`task-*/responses-*.jsonl`, 77,644 scored responses) | None |
| **Raw Visual Assets (JPEGs / TIFFs)** | No (excluded to respect upstream terms and repository size limits) | Available via `build_items_figlib.py` and `fetch_flame3.py` |
| **Large Databases & Cubes** | No (FPA-FOD 214 MB SQLite, Mesogeos 648 GB Zarr, ICS-209 48 MB zip) | Available via `fetch_fpafod.py`, `fetch_mesogeos.py`, `build_items_ics209.py` |
| **Withdrawn Task Files** | No (`task-fire360/` and `build_items_fire360.py` excluded) | Audited defect recorded in the paper's appendix on defects found while building the tasks |
| **Internal Endpoints & Credentials** | No (secrets excluded, `.env` gitignored, endpoints sanitized) | Not required for offline reproduction |
