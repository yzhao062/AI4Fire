# Data check digest, 2026-09-15

One entry per candidate task or source. Full detail, including every URL fetched, is in
`benchmark-data-check-2026-09-15.md`. Verdicts: ready, needs-work, not-feasible-now.

## Communication and Alerts + Data Retrieval and Tool Use

**[not-feasible-now] Multilingual wildfire evacuation guidance dialog** (Communication and Alerts, wildfire)
- data: None found as a released dataset. BEACON (kulkarni2026beacon, arXiv 2609.03301) is a system demonstration that ingests third-party live feeds: Watch Duty (fire perimeters, evacuation order status, shelters), NOAA HRRR (f
- page: https://arxiv.org/abs/2609.03301
- access: none found for the BEACON evaluation data. The underlying feeds split three ways. Watch Duty: effectively closed. Its terms of service state "You must not use any robot, spider, or other automatic device, process, or means to access the Platform for any purpos
- license: The BEACON preprint itself is CC BY 4.0, but that covers the paper, not data it never released. Watch Duty: research use is not granted and redistribution of derived items is prohibited without written consent; the terms reference a Creative Commons Attributio
- size: Zero released evaluation items. For the substitute sources: the NIFC WFIGS year-to-date perimeter layer returned {"count":7027} on 2026-09-15 under a definition query of attr_FireD
- labels: None. The BEACON paper is explicit that "The results below are qualitative observations of the end-to-end system behaviors": no metric, no test-case count, no ground truth, no baseline, no released evaluation set. It reports observations on
- contamination: Excellent on the substitute sources, absent on the motivating work. The BEACON preprint was submitted 2026-09-03 and is under review for the IEEE SpatialConnect Workshop 2026, so it postdates most cur
- build: Build it, from three verified public sources, and split the task so that the scorable part is separated from the part that needs human reference. (1) Factual spine: harvest the NIFC WFIGS services at https://services3.arcgis.com/T4QMspbfLg3
- effort: High, and concentrated in one place. Two to four weeks of engineering for a 300 to 600 item English factual set: a snapshot harness that fre

**[needs-work] Crisis translation and urgency classification with fire content** (Communication and Alerts, both (the fire-beari)
- data: No corpus exists that is multilingual AND carries fire items AND carries urgency labels. Four public corpora each have two of the three: HumAID (English, wildfire, humanitarian labels); CrisisMMD (English, 2017 Californi
- page: https://github.com/rmunro/disaster_response_messages
- access: public download for three of the four, with a licensing catch on two. Multilingual Disaster Response Messages: direct download of three CSVs from GitHub, no registration. HumAID: direct download from https://crisisnlp.qcri.org/humaid_dataset.html and a mirror 
- license: Multilingual Disaster Response Messages: CC BY 4.0, confirmed in the repository LICENSE file. This is the only fire-bearing multilingual source found whose license plainly permits research use and redistribution of derived benchmark items, subject to attributi
- size: Measured directly, not quoted. Multilingual Disaster Response Messages: 23,850 rows total (19,116 train, 2,337 validation, 2,397 test), of which exactly 266 carry fire=1 (224, 31, 
- labels: Categorical labels exist, but none of them is urgency on a fire item in a non-English language. Multilingual Disaster Response Messages: 38 binary category columns, with fire documented as "whether the message is related to fire, including 
- contamination: Poor on everything that exists, good on the build route. Multilingual Disaster Response Messages covers 2010-2012 events and has been on GitHub, on Kaggle and inside a widely taken Udacity nanodegree 
- build: Build it from NWS CAP alerts, which solve the label problem and the license problem at once. Harvest fire-related alerts from https://api.weather.gov/alerts, restricted to the five verified fire and evacuation event types (Red Flag Warning,
- effort: Low to moderate, and much cheaper than item 1. A few days of scripting to harvest, filter and normalize CAP alerts and to run NLLB-200 trans

**[needs-work] Agents that query fire databases, APIs, or tools** (Data Retrieval and Tool Use, both, but asymmetric)
- data: No fire tool-use benchmark exists; a search for one surfaced only FireBench (arXiv 2603.04857), which is enterprise instruction-following and has nothing to do with fire. What does exist is a set of stable public fire da
- page: https://www.fs.usda.gov/rds/archive/catalog/RDS-2013-0009.6
- access: public download or free API for all five wildfire sources. FPA-FOD: direct download from the Forest Service Research Data Archive, no registration, in ACCDB, GDB, GPKG and SQLite. FIRMS: free MAP_KEY issued by email, with the documented limit "MAP_KEY limit is
- license: Clean for research use and derived items on the wildfire side. FPA-FOD: the data "were collected using funding from the U.S. Government and can be used without additional permissions or fees", with citation required (Short, Karen C. 2022, 6th Edition, https://
- size: FPA-FOD: "2.3 million geo-referenced wildfire records, representing a total of 180 million acres burned", 1992-2020, SQLite build 213.96 MB (ACCDB 172.18 MB, GDB 136.26 MB, GPKG 21
- labels: This is the category's strongest property: gold answers are computed, not annotated. A reference program runs the intended query against a frozen snapshot (an FPA-FOD SQLite file, or a dated dump of WFIGS and FIRMS responses) and its output
- contamination: The best of the three items. NIFC's year-to-date layer is filtered to fires discovered after 2026-01-01 and held 7,027 perimeters on 2026-09-15, all of which postdate current model training cutoffs, w
- build: not needed as a rescue, but the task itself must be constructed, since no fire tool-use benchmark exists. Design: freeze a snapshot of each source with a recorded fetch timestamp; write a reference program that answers each question against
- effort: Moderate, and the lowest-risk of the three. Roughly one week for a snapshot and replay harness plus the reference query program over FPA-FOD

## Forecasting and Prediction (wildfire), plus post-wildfire Geospatial Analysis and the public wildfir

**[needs-work] Quantitative fire radiative power (FRP) forecasting: the model is given a day's fire-weather and vegetation dr** (Forecasting and Prediction, wildfire)
- data: No released dataset behind the motivating work. ramesh2025wildfiregpt used a table of 61 features and 619,373 rows that it says was "primarily sourced from the NASA Fire Energetics and Emissions Research platfrom (Davies
- page: https://firms.modaps.eosdis.nasa.gov/download/
- access: The paper's own table: none found. There is no data availability statement, no repository link, and no supplementary file; the article's back matter carries only Funding, Conflict of Interest, and Ethical approval sections. FIRMS: public but behind a light gat
- license: The article is CC BY 4.0 (Crossref records https://creativecommons.org/licenses/by/4.0 for both the TDM and VOR versions, effective 2025-05-28), but that covers the text and not an unreleased table. FIRMS data fall under NASA's Earth Science Data and Informati
- size: Paper's table (unreleased): 61 features, 619,373 rows, with a 74,550-row subset used for model selection. FIRMS archive (usable): MODIS Collection 6.1 from 2000-11-01 to present, V
- labels: Yes, once we build them. Each FIRMS detection row is a reference answer: a scalar FRP in megawatts attached to a latitude, longitude, acquisition date, and acquisition time in UTC. Scoring follows what the motivating paper reports: MAE, MSE
- contamination: Item-level dating is exact. Every FIRMS row carries acq_date and an acq_time "of acquisition/overpass of the satellite (in UTC)", so a post-cutoff window can be cut to the minute. Near-real-time rows 
- build: Pull a dated window from the FIRMS archive (https://firms.modaps.eosdis.nasa.gov/download/) or the Area API (https://firms.modaps.eosdis.nasa.gov/api/area/) after requesting a free MAP_KEY (https://firms.modaps.eosdis.nasa.gov/api/map_key/)
- effort: Two to three weeks of one person's work. Most of it is the spatial and temporal join (gridding point detections, aligning POWER cells to FIR

**[needs-work] Wildfire risk raster prediction with an out-of-region test: train or prompt on United States tiles with expert** (Forecasting and Prediction, wildfire)
- data: FireScope-Bench, the dataset released with markov2026firescope (CVPR 2026). It pairs Sentinel-2 imagery with climate data and expert-defined wildfire risk rasters across the USA, plus real wildfire events in Europe. The 
- page: https://huggingface.co/datasets/INSAIT-Institute/FireScope-Bench
- access: Public download. The Hugging Face dataset page shows no gating and no terms-acceptance step, and the code repository README points at the same URL with no registration mentioned. Recorded downloads at the time of checking: 2,491.
- license: cc-by-4.0 on the Hugging Face dataset card, with the added condition that "Users should also comply with the licensing and attribution requirements of the original data providers" (Copernicus Sentinel-2, NASA POWER, MTBS, EFFIS) and that users cite the CVPR 20
- size: 127 GB, 20,296 rows, split train 12k / validation 4.18k / test 4.14k. The paper reports coverage of over 5.7M km2 across 55K regions and 6.3B pixels. Root files are a 30.2 MB clima
- labels: Yes for the native task, no for a text task. The reference is a continuous 682x682 risk raster for the US tiles and a binary burn mask (MTBS for the USA, EFFIS for Europe) with paired control masks for the event tiles. That scores automatic
- contamination: Tiles are dated at the year level; the event folders are organized by year. The dataset became public on Hugging Face with a last-modified timestamp of 2026-05-12, and the paper was posted to arXiv on
- build: not needed
- effort: One to two weeks to derive text-answerable items. The download is the largest single cost at 127 GB; a usable subset can be pulled with the 

**[needs-work] Wildfire risk sub-criteria scoring against expert rankings: the model assigns importance weights to 16 wildfir** (Forecasting and Prediction, wildfire)
- data: The materials behind cheng2026retrieval (Fire 2026, 9, 143). Both halves asked about are released, in three places. (1) The 16 sub-criteria are named in the article body: species composition, development stage, stand cro
- page: https://github.com/Yuheng-Cheng/Large-Language-Model-Augmented-Decision-Strategies-for-Hierarchical-Wildfire-Risk-Evaluation
- access: Public download, no registration. The article's Data Availability Statement reads: "Data will be made available: https://github.com/Yuheng-Cheng/Large-Language-Model-Augmented-Decision-Strategies-for-Hierarchical-Wildfire-Risk-Evaluation/tree/main (accessed on
- license: Split, and this is the problem. The article is CC BY 4.0: Crossref records https://creativecommons.org/licenses/by/4.0/ for the VOR, and the PDF carries "Copyright: 2026 by the authors. Licensee MDPI, Basel, Switzerland. This article is an open access article 
- size: 16 items. That is the whole task. Five LLMs were evaluated (ChatGPT-4o, Gemini-2.0, Baichuan-3, Kimi-1.5, ChatGLM-4) across four strategies (DLS, MDS, FDP, IGP), all through their 
- labels: Yes, and in a directly scorable form: a 16-element integer rank vector. The paper's own metrics are Pearson's R, Spearman's rho, and Kendall's tau against the expert baseline, reporting DLS at R = 0.009 +/- 0.070, MDS at R = 0.181, FDP at R
- contamination: The article is dated precisely: received 2026-02-03, revised 2026-03-20, accepted 2026-03-24, published 2026-03-26. The repository was accessed by the authors on 2026-03-09. Contamination risk is high
- build: not needed for reuse, but a contamination-clean version would need a new baseline. Build it by taking a different published AHP wildfire risk study with a full expert weight table that postdates the model cutoffs, or by running a small expe
- effort: Two to three days to reproduce the task exactly as published, because the criteria, the prompts, and the reference vector are all in hand. T

**[needs-work] Post-wildfire burn-scar and damage assessment: given post-event Earth-observation imagery, the model answers q** (Geospatial Analysis, wildfire)
- data: No usable dataset from the motivating work. lahouel2026iceo built its own test set rather than using a named public one: "we built 12 questions" for post-wildfire assessment, plus 12 simpler object-presence questions, 24
- page: https://huggingface.co/datasets/ibm-nasa-geospatial/hls_burn_scars
- access: IC-EO's 12 questions: none found. The arXiv page lists no code or dataset URL, and the paper carries no data availability or code availability statement. A request to the authors is the only route, and the paper is a preprint (submitted 2026-01-27, cs.CV and c
- license: IC-EO: the arXiv posting is under the arXiv non-exclusive distribution license 1.0, which covers the paper and confers no rights to any data. HLS Burn Scars: cc-by-4.0, which allows research use and the release of derived questions with attribution; the card a
- size: IC-EO: 12 burn-scar questions plus 12 object-presence questions. HLS Burn Scars: 2.65 GB, 1,068 rows covering 804 scenes of 512x512 pixels, split randomly two thirds training and o
- labels: Not for IC-EO. Twelve items with an undescribed annotation procedure will not support a claim a TMLR reader can check, and 50% on 12 items is 6 correct answers, which no confidence interval can separate from chance on a binary question. HLS
- contamination: IC-EO items: undatable, since they are not released and the paper does not say which scenes they came from. HLS Burn Scars: scenes are drawn from HLS v1.4 imagery over the contiguous United States for
- build: Build from HLS Burn Scars (https://huggingface.co/datasets/ibm-nasa-geospatial/hls_burn_scars), CC BY 4.0, 804 labeled scenes. Reduce each mask to a text-answerable reference: the burned fraction of the scene as a percentage, a binary burn-
- effort: Two to three weeks. The dataset download is small (2.65 GB), so the work is the item derivation, the band-to-image rendering, and the tool h

**[needs-work] Next-day wildfire spread prediction from a multi-modal daily time series: given the recent daily observations ** (Forecasting and Prediction, wildfire)
- data: WildfireSpreadTS (gerard2023wildfirespreadts), NeurIPS 2023 Datasets and Benchmarks Track. A multi-temporal, multi-modal remote-sensing dataset for predicting wildfire spread at 24-hour resolution, giving for each fire e
- page: https://zenodo.org/records/8006177
- access: Public download, open access without login. Two files on Zenodo: WildfireSpreadTS.zip and WildfireSpreadTS_Documentation.pdf. The data loader, Lightning data module, and experiment code are at github.com/SebastianGer/WildfireSpreadTS under MIT.
- license: CC-BY-4.0 on the Zenodo record, with no access conditions or restrictions stated beyond the attribution CC BY requires. Research use and release of derived items are both allowed. The code is separately MIT licensed.
- size: 48.4 GB (WildfireSpreadTS.zip), plus a 7.8 MB documentation PDF. 13,607 images across 607 fire events in the United States, January 2018 to October 2021.
- labels: Yes. The label is the next day's active-fire mask, so a per-cell binary reference exists for every fire-day pair. That is already a scorable target for a raster model; a text-answerable version needs a reduction step (a burn-or-not call on 
- contamination: Item-level dating is exact, since the dataset is a daily time series and each fire event is bounded in time. The window is January 2018 to October 2021, and the dataset itself has been public on Zenod
- build: not needed
- effort: One to two weeks to derive text-answerable items and a scoring harness, dominated by the 48.4 GB download and by writing the cell-selection 

**[needs-work] Multi-task satellite image time-series wildfire tasks: active fire detection, daily burned-area mapping, and n** (Forecasting and Prediction, wildfire)
- data: TS-SatFire (zhao2025tssatfire), Scientific Data 12, 1817 (2025). Surface reflectance image time series with auxiliary weather, topography, land cover, and fuel data, covering each wildfire's lifecycle, with active fire a
- page: https://www.kaggle.com/datasets/z789456sx/ts-satfire
- access: Public download through Kaggle, with an account. The article states "The TS-SatFire dataset is available on Kaggle, accessible via the link (https://www.kaggle.com/datasets/z789456sx/ts-satfire)". An unauthenticated request to the Kaggle download endpoint retu
- license: Creative Commons Attribution 4.0 International. The Kaggle record's licenseName is "Attribution 4.0 International (CC BY 4.0)" and the article carries the same. Research use and release of derived items are both allowed with attribution.
- size: 76,314,754,827 bytes (about 76.3 GB) on Kaggle; the article states 71 GB. 3,552 surface reflectance images across 179 wildfire events. Splits: 125 training events (2017 to 2020), 1
- labels: Yes, three label types, all pixel-level: active fire detection as pixel-wise binary classification (fire or non-fire), daily burned area mapping as pixel-wise binary segmentation (burned or unburned), and wildfire progression prediction as 
- contamination: January 2017 to October 2021 over the contiguous United States, with the active fire detection test set drawn from 2018 to 2022 across several continents, so every item is datable to its acquisition. 
- build: not needed
- effort: Two weeks, similar to WildfireSpreadTS: the 76 GB download plus an item-derivation pass. The progression task is the one worth the effort he

**[needs-work] Wildfire segmentation on multispectral Landsat 8 imagery, posed as text-answerable questions about fire presen** (Detection and Perception, wildfire)
- data: Land8Fire (tran2025land8fire), Remote Sensing 2025, 17, 2776. A large-scale wildfire segmentation dataset of over 20,000 multispectral image patches derived from Landsat 8 and manually annotated for high-quality fire mas
- page: https://github.com/UARK-AICV/Land8Fire
- access: Public download, no registration, but through an unversioned host. The repository ships two scripts, land8fire/kfolds_downloader.py for the k-fold subdivided data and land8fire/downloader.py for the full-resolution 7k x 7k TIF images, both of which pull from G
- license: Split, and the redistribution question is open. The article is CC BY 4.0: the ScholarWorks record states "This work is licensed under a Creative Commons Attribution 4.0 International License." The GitHub repository has no license at all (the GitHub API returns
- size: images.zip is 55 GB on Google Drive ("[images.zip] (55G) is too large for Google to scan for viruses"), plus a separate masks.zip. The article reports over 20,000 multispectral ima
- labels: Yes for segmentation: manually annotated binary fire masks with predefined 5-fold splits. There is no text-answerable target in the release, so an item would have to be derived the same way as for HLS Burn Scars.
- contamination: Weak. The article and the release do not state the acquisition dates of the patches in anything I could fetch, and the source is the ActiveFire dataset, which draws on Landsat 8 scenes over South Amer
- build: not needed for the segmentation task; a text-answerable version would follow the HLS Burn Scars route (reduce a mask to a burned fraction, a binary call, or an ordering).
- effort: Two to three weeks, and the effort estimate is soft because the release is incomplete. The 55 GB download plus an item-derivation pass is th

**[needs-work] Short-term wildfire danger forecasting: given the recent daily values of the fire drivers for a 1 km Mediterra** (Forecasting and Prediction, wildfire)
- data: Mesogeos (kondylatos2023mesogeos), NeurIPS 2023 Datasets and Benchmarks Track. A spatio-temporal datacube for the Mediterranean that harmonizes wildfire drivers (meteorology from ERA5-Land, MODIS vegetation and temperatu
- page: https://github.com/Orion-AI-Lab/mesogeos
- access: Public download, no registration. The datacube and the extracted tracks are in a publicly viewable Google Drive folder (https://drive.google.com/drive/folders/1aRXQXVvw6hz0eYgtJDoixjPQO-_bRKz9), which holds mesogeos_cube.zarr, ml_tracks, auxilliary, notebooks,
- license: CC BY 4.0. The Zenodo record shows Creative Commons Attribution 4.0 International, the arXiv posting is CC-BY-4.0, and the repository README states "Creative Commons Attribution v4". Research use and release of derived items are both allowed with attribution. 
- size: The datacube "occupies a storage space of 648 GB" under default Zarr compression, and loading it needs about 3.2 TB of memory at 32-bit floats. The extracted ML tracks are far smal
- labels: Yes, and this is the best-shaped reference in the group. Track A is already a binary classification label per sample, with published metrics and baselines (LSTM, a Transformer encoder, and a Gated Transformer Network), and it ships with a d
- contamination: Item-level dating is exact, since every sample is a cell on a specific day. The published split is already a time holdout: 2006 to 2019 train, 2020 validation, 2021 to 2022 test, which is the structur
- build: not needed
- effort: One week, the lowest in this group. The ML tracks are already extracted, tabular, labeled, split, and paired with baselines, so the work is 

**[needs-work] Global sub-seasonal to seasonal wildfire modeling: given the recent driver values for a 0.25 degree cell, the ** (Forecasting and Prediction, wildfire)
- data: SeasFire cube (karasante2025seasfire), Scientific Data 12, 368 (2025). A curated global spatio-temporal datacube for sub-seasonal to seasonal wildfire modeling, with 59 variables spanning climate, vegetation, oceanic ind
- page: https://zenodo.org/records/13834057
- access: Public download, open access without login. Version 0.4 is on Zenodo as seasfire_v0.4.zip with a changelog PDF; earlier versions (v0.3 at 10.5281/zenodo.8055879) are also open. The cube is in .zarr format, readable with xarray in Python or YAXArray.jl in Julia
- license: Creative Commons Attribution 4.0 International, stated both on the Zenodo record (CC-BY-4.0) and in the article ("Creative Commons License 4.0 International"). Research use and release of derived items are both allowed with attribution.
- size: 43.9 GB for seasfire_v0.4.zip (v0.3 was 44.0 GB). 59 variables at 0.25 degree spatial resolution and 8-day temporal resolution, 46 datetimes per year, covering 2001 to 2021 (21 yea
- labels: Yes in principle, and this is the source that needs the most derivation work. The cube carries burned-area and fire variables as data layers rather than as a packaged task with a fixed split, so unlike Mesogeos there is no pre-extracted mac
- contamination: Item-level dating is exact at the 8-day step, and the coverage is 2001 to 2021. The article was published 2025-03-03 and the v0.4 Zenodo record 2024-09-26, with earlier versions public since 2022. Eve
- build: not needed for the data; the task definition has to be built. Extract a target (burned area per cell per 8-day step, or a binary above-threshold call), define a time split, and write a non-LLM baseline before any LLM number means anything. 
- effort: Two to three weeks, most of it task definition rather than data handling: choosing the target, the threshold, the aggregation region, and th

**[not-feasible-now] Wildfire foundation-model transfer tasks (occupancy, spread, analog retrieval, burned-area regression) under a** (Forecasting and Prediction, wildfire)
- data: No released dataset. xu2026wildfirefm introduces WILDFIRE-FM, a foundation model pretrained for wildfire prediction, and a fixed-contract evaluation framework with a fixed-output check for matching-rule effects and a fix
- page: https://arxiv.org/abs/2605.18911
- access: None found. The paper is a preprint (submitted 2026-05-14, cs.LG and cs.AI, no venue named) and no dataset is released. The two links it gives both fail to yield anything: the code link is an anonymous.4open.science URL, which returns only the Anonymous Github
- license: The arXiv posting is under Creative Commons Zero 1.0 Public Domain Dedication, which covers the paper only. No license is stated or reachable for the code, the weights, or the assembled California dataset. The upstream ingredients are individually permissive (
- size: Not stated for the assembled dataset. The reported splits are temporal: June to August 2024 for training, September 2024 for validation, and October 2024 for testing, over a Califo
- labels: Yes in the paper, no to us. The four tasks have well-defined targets: occupancy as gridded binary fire presence, fire spread as burned raster patches scored with spatial-overlap metrics, analog retrieval scored with nDCG@10, and regression 
- contamination: The evaluation window is June to October 2024, which is datable and recent, and it would be a genuinely useful post-cutoff window for older models. It is not post-cutoff for a 2026 frontier model. The
- build: Rebuildable in principle from public sources, and expensive. Every ingredient is individually reachable: NASA FIRMS for active-fire supervision (free MAP_KEY, https://firms.modaps.eosdis.nasa.gov/api/map_key/), NOAA HRRR for weather, LANDFI
- effort: Two to four months if rebuilt from scratch, which I do not recommend for this benchmark. The cheaper path is a request to the authors, since

## Structure-fire coverage and document-understanding tasks (items 1-7)

**[needs-work] Building fire code compliance checking on tables and clauses** (Knowledge Question Answering, structure fire)
- data: "Real-world fire audit dataset" used by RegCheck-Hybrid; never named, sized or released
- page: https://api.semanticscholar.org/graph/v1/paper/DOI:10.1109/MEAI68126.2025.11406521
- access: none found. The paper is a paywalled IEEE conference paper (MEAI 2025, DOI 10.1109/MEAI68126.2025.11406521, published 2025-12-05, Zi-Rui Chen and Wei Zhou). Semantic Scholar reports no open-access PDF. The abstract is the only text I could read and it says onl
- license: Dataset: unknown, none stated. Paper: IEEE copyright. Separately, on the underlying code text: England's Approved Document B (fire safety) is published on GOV.UK under the Open Government Licence v3.0, which permits copying, publishing, distributing, adapting 
- size: unknown; the abstract gives no item count, no country and no source agency
- labels: Not released. The paper reports an 89% F1-score, which implies per-clause compliant / non-compliant labels exist internally, but nothing is published.
- contamination: Paper published 2025-12-05. The audit records themselves are undated, so contamination risk for a 2026 model cannot be assessed at all.
- build: Yes, and it is the strongest route in this group. CODE-ACCORD (Zenodo, https://zenodo.org/doi/10.5281/zenodo.10210022, v1.0.0, published 2023-11-27, CC BY 4.0) ships both the raw regulatory text and annotations. Its source table lists Engla
- effort: Roughly three to five person-weeks. Pull the numeric-threshold and tabular clauses out of Approved Document B volumes 1 and 2 (384 pages, al

**[needs-work] Fire science and safety question answering** (Knowledge Question Answering, both)
- data: none found as an off-the-shelf benchmark; four public source corpora verified as usable inputs
- page: https://www.gov.uk/government/collections/approved-documents
- access: public download for every source below. (a) England's Approved Documents including Part B fire safety, direct PDF download from GOV.UK, no registration. (b) CODE-ACCORD at https://zenodo.org/doi/10.5281/zenodo.10210022, single 101.3 MB zip, no registration. (c
- license: (a) Open Government Licence v3.0: permits copying, publishing, distributing, adapting and commercial exploitation, attribution required, no sub-licensing right stated. Research use and release of derived questions are both allowed. (b) CC BY 4.0 on the Zenodo 
- size: Approved Document B: 2 volumes, 384 pages. CODE-ACCORD: 862 annotated sentences, 4,297 entities, 4,329 relations, over a 33-document 1,688-page source corpus. NFIRS Public Data Rel
- labels: None exist ready-made; they would be constructed. The tractable form is clause-grounded: the reference answer is a numeric threshold or a category plus the clause identifier it comes from, which scores by exact match with a numeric toleranc
- contamination: Approved Document B: 2019 edition incorporating 2020 and 2022 amendments, collection page last updated 2025-03-11, and GOV.UK also carries a volume 1 PDF collated with 2026 and 2029 amendments, which 
- build: Build from the four sources above. Structure fire: numeric and tabular threshold questions from Approved Document B, using CODE-ACCORD's already-sentence-segmented English_Regulations folder as the starting text so the extraction step is sk
- effort: Roughly four to six person-weeks for 800 to 1,200 items across the two domains, including a two-reviewer adjudication pass on a 10% sample. 

**[not-feasible-now] Categorizing barriers to fire spread from incident text** (Document Understanding, wildfire)
- data: Wildland Fire Decision Support System (WFDSS) free-text incident entries
- page: https://connectsci.au/wf/article/34/9/WF25051/200681/Learning-from-Wildfire-Decision-Support-large
- access: none. The paper's data availability statement says "WFDSS data are not publicly available owing to the sensitive nature of emergency response decisions", adding that "the federal agencies responsible for the WFDSS intend to provide a reporting interface in the
- license: Article: CC BY-NC-ND 4.0. Data: no license, access restricted. Neither research use nor release of derived items is available to us.
- size: 24,254 text entries across 6,630 WFDSS incidents, 2011 to 2023
- labels: A 13-category barrier taxonomy exists and would score as multi-label classification, but the labels are not released. Validation used 260 randomly selected WFDSS entries independently reviewed by two experienced wildland firefighters drawn 
- contamination: Incidents span 2011 to 2023; the article appeared in 2025. Contamination risk is low precisely because the text was never public, which is the same reason we cannot use it.
- build: Not needed, because nothing equivalent exists. I looked for a substitute corpus of wildland fire decision or operational narrative text and found none. The NFIRS Public Data Release is the obvious candidate and it fails: the release "consis
- effort: Not applicable. The only path is an interagency data agreement with the WFDSS owners, which is a months-long institutional negotiation with 

**[needs-work] Incident-command transcript task-state monitoring** (Document Understanding, structure fire)
- data: Source-grounded synthetic German firefighter incident-command scenarios released with the SwissText 2026 paper
- page: https://github.com/zhaw-iwi/swisstext26_pub
- access: public download, no registration. The repository is public, created 2026-05-05, last pushed 2026-05-07, about 21.7 MB. It holds the scenario JSON files, the dataset-generation and validation prompts, the source notes, the evaluation prompt payloads and the per
- license: The GitHub API reports the repository's license field as null, so there is no LICENSE file and no grant of reuse rights. The paper carries the ACL proceedings line "June 10, 2026 (c)2026 Association for Computational Linguistics"; the Anthology landing page di
- size: Five German-language scenarios, 102 ordered radio messages, 15 predefined tasks, 15 task-state traces. Of the 15 tasks, all are explicitly assigned in the transcript evidence, 12 a
- labels: Yes, and in the best form of any item in this group. Each gold entry is a fixed-schema JSON tuple y(k) = (assigned, assigned_unit, completed, completion_outcome) over the current transcript prefix k, plus the message id at which assignment 
- contamination: Public since 2026-05-05, which is after the training cutoff of most models we would run, so verbatim contamination risk is genuinely low. The offsetting hazard is different in kind: the scenarios were
- build: Not needed for the released items, but the group's structure-fire coverage cannot lean on five scenarios. I searched for a public corpus of fire incident-command communication and found none. The nearest verified thing is RescueSpeech (arXi
- effort: Low to run as released: a day to wire the prefix-level evaluation, since the authors ship the pipeline. Two to three weeks if we extend it, 

**[not-feasible-now] Damage assessment reports from bushfire imagery** (Document Understanding, wildfire)
- data: Bushfire photographs reproduced in a Queensland Fire Department magazine, used only in the paper's Supplementary Information
- page: https://pmc.ncbi.nlm.nih.gov/articles/PMC12886768/
- access: public download for the underlying files, but there is no dataset. The data availability statement reads in full: "The crisis image dataset in the main text is released by Qatar Computing Research Institute for humanitarian computing research, and can be acces
- license: Article: CC BY-NC-ND 4.0. Code repository github.com/zche3016/DisasTeller carries a LICENSE file reading "Attribution-NonCommercial-NoDerivatives 4.0 International", Copyright (c) 2024 zche3016. The NoDerivatives clause blocks exactly what we would need to do,
- size: not stated anywhere I could fetch. No bushfire image count appears in the article, the PMC full text or the repository README. The repository does contain a Bushfire_images folder,
- labels: None usable. Outputs were scored two ways on the same rubric: by GPT-4o acting as "EvaluatorGPT" and by human assessors with civil engineering backgrounds, both rating coherence, consistency and accuracy on 0-10 scales across five independe
- contamination: The Queensland magazine issue is December 2019 and the QCRI crisis image set is from 2017, so both have been public for six to nine years. A 2026 model has almost certainly seen the Wajima City covera
- build: No route I can defend from a source I fetched. Public bushfire damage imagery with damage labels does exist in the building-damage assessment literature, but I did not fetch any such dataset in this pass and I will not name one from memory.
- effort: Prohibitive as scoped. Writing reference damage-assessment reports for even 100 bushfire scenes needs a structural or bushfire assessor for 

**[not-feasible-now] Drafting fire protection system specifications** (Document Understanding, structure fire)
- data: none released. Two Korean public standards collections identified as possible inputs: the national fire safety standards NFPC/NFTC, and the KCS 31 series standard construction specifications
- page: https://www.law.go.kr/lbook/lbInfoR.do?lbookSeq=100443&FSort=100
- access: The motivating paper releases nothing and is paywalled: Kim Jong-Cheol and Shin Dong-il (Myongji University), "Performance Comparison and Improvement of LLM Models for Automated Generation of Fire Protection System Construction Specifications", Korean Journal 
- license: Motivating paper: DBpia paywall, no data license. Korean standards: no copyright notice or terms-of-use statement appeared on the Law Information Center page I fetched, so the redistribution license is unverified and I will not assume Korean law is reusable wi
- size: Motivating study: item count never stated on any page I could fetch; the abstract says only that the study covers fire-service-designated objects such as factories, warehouses and 
- labels: None exist. There is no reference specification corpus, and the source paper's own evaluation was expert review, which it reports as essential, noting that the models "generate basic specification structures well but struggle with building-
- contamination: The Korean standards are current law with real-time automatic updates, so individual clauses are datable by amendment date and a post-cutoff slice could in principle be cut. The collections themselves
- build: Only for a reframed task. Drafting a specification and scoring it automatically has no honest route, because a specification is long-form prose with no single correct wording and no public reference set. What can be built is a fill-in task:
- effort: Two to four person-weeks for the reframed fill-in version, reusing the Approved Document B extraction already needed for item 1, so the marg

**[needs-work] Fire investigation causal reasoning on real cases** (Document Understanding, structure fire)
- data: 1,051 real-world fire investigation cases used by MAGR-FI; not released. Public substitute: NIOSH Fire Fighter Fatality Investigation and Prevention Program reports, plus the USFA Technical Report Series
- page: https://ojs.iscram.org/index.php/Proceedings/article/view/227
- access: Motivating dataset: none found. The ISCRAM 2026 article page carries the abstract and a free PDF, but gives no data availability statement, no repository, no country, no date range and no license line, and the PDF did not supply them either. A request to the a
- license: Motivating dataset: unknown, none stated. ISCRAM proceedings: I could not find a license or open-access statement on the journal's reader-information or submission pages, although the PDFs download freely. NIOSH FFFIPP reports: US federal public domain, no cop
- size: MAGR-FI test set: 1,051 real-world cases, with no further breakdown published. NIOSH FFFIPP: reports from 1998 to present; the program page gives no total and I could not verify a 
- labels: Motivating work: a graded quality score, not a label. MAGR-FI reports average scores rising from 6.95 to 8.51 with a 120% gain on complex cases, which is a rubric out of 10; no page I could read says who or what produced those scores, so it
- contamination: MAGR-FI: paper in the ISCRAM 2026 proceedings, 23rd conference, The Hague, 2026-05-31 to 2026-06-03; the cases themselves are undated. NIOSH FFFIPP: reports public since 1998, so heavily indexed and a
- build: Yes, from NIOSH FFFIPP reports at https://www.cdc.gov/niosh/firefighters/fffipp/index.html, with CDC Stacks holding the pre-2017 archive (example verified: https://stacks.cdc.gov/view/cdc/163994, a free 698 KB PDF with no copyright restrict
- effort: Roughly four to six person-weeks for 200 to 400 items. The work is not collection, which is a scripted download, but extraction and redactio

## Decision Support and Operations (wildfire) plus Simulation-Coupled Agents (wildfire)

**[needs-work] Daily personnel and cost allocation on held-out California wildfires** (Decision Support and Operations, wildfire)
- data: Derived per-fire-day personnel and cost series for 14 large 2020 California wildfires (9 historical corpus, 5 held out), built by the authors from ICS-209-Plus daily situation reports, plus a geospatial context layer ass
- page: https://arxiv.org/abs/2510.12061
- access: Public download for the code only (https://github.com/defene/GAL, MIT). No release of the derived evaluation data. The repository README states: "This repository contains the code, but not the full underlying geospatial assets or generated experiment outputs."
- license: GAL code: MIT. ICS-209-PLUS (the ground-truth source): CC BY 4.0, confirmed via the figshare API for DOI 10.6084/m9.figshare.19858927.v3, which permits research use and redistribution of derived items with attribution. The Scientific Data paper describing it i
- size: Held-out evaluation: 5 fires with durations of 29, 30, 35, 35 and 83 days, so roughly 212 fire-days before exclusions (CZU Aug Lightning, Aug 17, 35 days; El Dorado, Sep 5, 35 days
- labels: Yes, numeric and per fire-day. ICS-209-PLUS daily situation reports carry TOTAL_PERSONNEL ("Total number of personnel resources summed across all agencies"), EST_IM_COST_TO_DATE ("Estimated Incident Management Costs to Date") and PROJECTED_
- contamination: Items are datable to the day, since every sitrep is a dated filing. ICS-209-PLUS v3 went public on figshare 2023-01-10 and covers 1999 to 2020; a 1999 to 2014 precursor has been public since 2019. The
- build: Download ics209plus-wildfire.zip from https://figshare.com/articles/dataset/All-hazards_dataset_mined_from_the_US_National_Incident_Management_System_1999-2020/19858927 (CC BY 4.0), join the daily sitreps to the wildfire incident summary ta
- effort: The ICS-209 half is 1 to 2 days: one join on INCIDENT_ID, a cleaning rule, and an item generator over fire-days. The geospatial half is the 

**[not-feasible-now] Operations tasks grouped by the incident-command phases that an OGC report maps agent workflows onto** (Decision Support and Operations, wildfire)
- data: none found. OGC Public Engineering Report 24-071 (D-123) is a requirements and governance document with no evaluation items, no questions, no reference answers and no model results.
- page: https://www.ogc.org/wp-content/uploads/2025/07/24-071_OGC_CLIMATE_AND_DISASTER_RESILIENCE_PILOT_IV_D-123_GENERATIVE_AI_IN_WILDLAND_FIRE_MANA
- access: Public download, no registration. The PDF is free on ogc.org and the document carries an external identifier at https://zenodo.org/uploads/12721058. It opened only after an alternate read path: the first read reported it as password-protected, and pypdf extrac
- license: OGC document license at https://www.ogc.org/license, which grants the right to "deal in the Intellectual Property without restriction ... including without limitation the rights to implement, use, copy, modify, merge, publish, distribute, and/or sublicense cop
- size: 48 pages. Submission date 2024-07-10, approval and publication date 2025-01-02, PDF creation date 2024-07-26. Editors: Matt Tricomi, Kevin Hope, Xentity Corporation. Contents: 8 nu
- labels: None. The report contains no questions, no prompts, no worked examples, no scored scenarios and no model evaluation. Zero models are named on the page. What it does contain is Table 3 ("Generative AI Potential Solution Topics across the Dis
- contamination: The report was submitted 2024-07-10 and published 2025-01-02, well before any 2026 model cutoff, and the PDF has been on ogc.org since at least July 2025. If items were built from its text they would 
- build: Build the items from NWCG publications, which are explicitly public domain. The front matter of the Incident Response Pocket Guide states: "Publications and training materials produced by the National Wildfire Coordinating Group (NWCG) are 
- effort: About one week to hand-build 100 to 200 doctrine items from the IRPG with a key, plus an expert spot-check pass. Effort is not the constrain

**[not-feasible-now] Retrieval-augmented answers for wildfire risk and insurance workflows** (Decision Support and Operations, wildfire)
- data: none found as a task dataset. OGC Engineering Report 25-012 is an inventory and use-case document. Its Annex B does name individual Canadian sources with URLs: Table B.1 Apps/Systems List (30 rows) and Table B.2 Data Sou
- page: https://docs.ogc.org/per/25-012.html
- access: Public download of the report, no registration: HTML at docs.ogc.org/per/25-012.html and PDF at docs.ogc.org/per/25-012.pdf (66 pages). The listed Canadian datasets are separately and freely downloadable from their custodians. Checked directly: the Canadian Na
- license: The report: OGC document license (see item 2), copyright OGC 2026, which permits research use and derived works with a modification notice. The report states no license for any inventoried dataset. Checked independently: the Canadian National Fire Database (NF
- size: Report: 66 pages, published 2026-04-23, editors Matt Tricomi and Connor Miller. The inventory is "over 200 Canadian wildfire-related data sources" counted by information class in T
- labels: None. This is the clearest instance of the group note. I searched the extracted 66-page text for prompts, questions, benchmarks, accuracy figures and ground truth and found none. The "Data Sources and FAIR Evaluation" section is a readiness
- contamination: The report was published 2026-04-23, after most 2026 training cutoffs, but it is a document rather than a test set, so that date protects nothing on its own. The inventoried Canadian data has been pub
- build: Two routes, both with a key. (a) Numeric retrieval question answering over the Canadian National Fire Database: download NFDB_poly_1972to2024 from https://cwfis.cfs.nrcan.gc.ca/downloads/nfdb/fire_poly/current_version/ under Open Government
- effort: Route (a): 3 to 5 days, including the question generator, the per-agency attribution block and the numeric tolerance rules. Route (b): 2 to 

**[ready] Multi-agent wildfire suppression and rescue** (Simulation-Coupled Agents, wildfire)
- data: CREW-Wildfire, a procedurally generated wildfire response simulation benchmark with 12 scored levels, built on the CREW human-AI teaming platform and the Crew Dojo Unity framework. Four heterogeneous agent types (firefig
- page: https://arxiv.org/abs/2507.05178
- access: Public download. Code at https://github.com/generalroboticslab/CREW (39 stars, 16 commits on main); the wildfire component is at crew-algorithms/crew_algorithms/wildfire_alg with its own README naming the four frameworks implemented (CAMON, COELA, Embodied, HM
- license: Code repository: Apache-2.0. The arXiv posting of the paper carries CC BY-NC-ND 4.0, which forbids derivatives and commercial use of the paper text. Flag the mismatch rather than paper over it: we would build on the Apache-2.0 code, which permits research use 
- size: 12 benchmark levels across 16 level configurations. Team compositions range from 2 to 12 agents per level (for example "10 F, 1 B, 2 D, 2 H" for the Full Environment level). The en
- labels: Yes, and of the right kind for this group. The simulator computes the score, so no human is in the loop and no rubric is needed. Each level has an explicit score function: Cut Trees (sparse and lines, small and large) scores trees cut in la
- contamination: Scenarios are procedurally generated, so the specific instances we evaluate on did not exist before we generate them. This is the lowest contamination risk of the five items. The residual risk is that
- build: not needed. The environment and the scoring are released.
- effort: About one week to stand up, then the token budget. Install is heavier than a pip command: conda for the base environment, Poetry for crew-al

**[needs-work] Policy adaptation as the set of agents and tasks changes** (Simulation-Coupled Agents, wildfire)
- data: The MOASEI wildfire track runs on free-range-zoo, a PyTorch multi-agent environment suite with wildfire, rideshare and cybersecurity domains, built for agent, task and frame openness. The wildfire domain is a grid world 
- page: https://arxiv.org/abs/2507.05469
- access: Public download. Environment at https://github.com/oasys-mas/free-range-zoo (9 stars, 625 commits), with a frozen evaluation release tagged moasei2026v1.0 dated 2026-01-05, described as "the frozen version of free-range-zoo which will be used to evaluate resul
- license: free-range-zoo: MIT, which permits research use and redistribution of derived items. moasei-starter: AGPL-3.0, which is copyleft and would require source release if we distributed a derived starter, so prefer building against free-range-zoo directly. The 2025 
- size: Three wildfire scenarios plus a bonus, on 3x3 grids with varying initial fire positions and agent placements. The 2025 evaluation used three configurations (WS1, WS2, WS3) of incre
- labels: Yes, computed by the simulator. The 2025 report uses "average cumulative episodic reward of multiple runs as the primary performance metric across all tracks", with significance at p<0.05. The 2026 edition "expanded its reporting metrics to
- contamination: Configurations are drawn from a distribution and the 2026 held-out test set is "drawn from the same distribution but are not guaranteed to be identical to the training set", so instance-level contamin
- build: not needed for the environment. The LLM-in-the-loop harness does need building; see effort.
- effort: About 1 to 2 weeks. Install is straightforward (Python 3.12, Rust/Cargo, Poetry). The bulk of the work is a decision-time wrapper that rende

## Detection and Perception: data availability for the six candidate perception tasks (items 1-6), wild

**[not-feasible-now] Zero-shot active wildfire satellite classification (binary yes/no active fire from a satellite image)** (Detection and Perception, wildfire)
- data: "Satellite Wildfire detection", a Roboflow Universe object-detection project by HTW Berlin. This is the dataset the paper actually used; the paper's full text names it explicitly and it is not an author-built set.
- page: https://universe.roboflow.com/htw-berlin-xv7eo/satellite-wildfire-detection
- access: Public download. The Roboflow Universe page is publicly listed and its schema.org record carries "isAccessibleForFree":true. Downloading or forking a version goes through Roboflow and needs a free account plus an API key; the page renders "Sign In or Sign Up" 
- license: CC BY 4.0. The page states "License: CC BY 4.0" and the per-image schema.org records carry "license":"https://creativecommons.org/licenses/by/4.0/". Research use and release of derived items are both allowed with attribution. The preprint itself is CC BY 4.0 p
- size: 500 images, one class ("fire"), bounding-box annotations, 0 generated dataset versions, 5 stars, 4.1k views. Roboflow dateModified is 2023-01-05 and the page renders "Updated 4 yea
- labels: Bounding boxes for the single class "fire" on the 500 images, which give an image-level positive label by presence. The paper's binary labels were not taken from the dataset: it says correctness was "objectively validated by a human researc
- contamination: The Roboflow record's only date is dateModified 2023-01-05, so the data has been publicly indexed for roughly three years and eight months. No capture dates, no satellite, no sensor and no acquisition
- build: Rebuild from TS-SatFire, which we fetched and verified: https://pmc.ncbi.nlm.nih.gov/articles/PMC12630597/ , data at https://www.kaggle.com/datasets/z789456sx/ts-satfire , CC BY 4.0, 71 GB, 179 wildfire events, 3,552 VIIRS surface-reflectan
- effort: About 4 to 6 person-days. One day to pull and index TS-SatFire, one to two days to write the scene-sampling and balancing script against the

**[needs-work] Wildfire smoke classification and early localization (image-level smoke presence, tile-based and grid-based lo** (Detection and Perception, wildfire)
- data: SmokeBench. It is not a new image collection: it is a task construction over FIgLib. The paper says "Our benchmark is built on the FIgLib dataset [11] ... We use a subset of 5,046 images with ground-truth bounding box an
- page: https://arxiv.org/abs/2512.11215
- access: None found for SmokeBench itself. No release. The arXiv page lists no code, dataset or project link, and the full PDF contains no URL of any kind, no data-availability statement and no "code will be released" sentence. The underlying FIgLib images are a public
- license: The arXiv preprint is CC BY 4.0. The benchmark has no license because it has no release. FIgLib, the underlying imagery, has no formal license: the HPWREN page says "This data is provided as-is, no guarantees for anything is included" and "While we are making 
- size: 6,046 images: 5,046 FIgLib images with smoke bounding boxes as positives, plus 1,000 non-smoke images as negatives. No train, validation or test split is described. Tile task uses 
- labels: Yes, four label forms, all mechanically derived from the FIgLib boxes: binary smoke/no-smoke per image; binary per tile, where "Ground-truth labels are assigned to tiles that overlap annotated bounding boxes, while all others are marked neg
- contamination: FIgLib sequence folders are named YYYYMMDD_firename_camera, so every item can be dated exactly. The currently indexed range is 2016-06-04 to 2026-09-09, with 79 sequences from 2025 and 2026. SmokeBenc
- build: Two halves, and only one is blocked. (a) Classification is rebuildable now with no permission needed: FIgLib file names encode the offset in seconds from visible plume appearance (for example 1720453551_-02400.jpg is 2,400 seconds before, 1
- effort: Classification and time-to-detection: about 3 to 4 person-days, mostly a crawl of the 2025 and 2026 sequences and a file-name parser. Locali

**[ready] Temperature-grounded questions on aerial imagery (multiple-choice questions over paired RGB and radiometric th** (Detection and Perception, wildfire)
- data: WildFireVQA, a question set layered on the FLAME 3 UAV imagery. The paper: "WildFireVQA is built on FLAME 3 dataset, which provides synchronized aerial visible spectrum (RGB) imagery and radiometric thermal imagery colle
- page: https://huggingface.co/datasets/mobiiin/WildFire_VQA
- access: Public download, in two pieces. The questions and answers are on Hugging Face, not gated ("gated": false, "private": false), 71 downloads, mirrored on Kaggle at https://www.kaggle.com/datasets/caseypiere/wildfire-vqa , with evaluation code at https://github.co
- license: Questions and answers: the Hugging Face card field is "apache-2.0" and the repository README also names CC-BY-4.0 for the data, an internal inconsistency worth resolving with the authors before citing a single license. The paper carries CC BY 4.0. Images: the 
- size: 6,097 RGB-thermal samples, 34 questions each, 207,298 multiple-choice questions total. Distributed as 9 JSON files, about 217 MB, one per FLAME 3 burn unit (Shoetank fire and no-fi
- labels: Yes, in a form that scores automatically. We inspected the records: each carries image_id, rgb_path, thermal_path, temp_summary (min, max, mean, std, top3_mean, pct_over_200, pct_over_400), category, question_id, question, options, answer, 
- contamination: FLAME 3 burns are dated by site: Sycan Marsh Oregon October 2022, Shoetank Arizona November 2022, Willamette Valley Oregon September to October 2022, Hanna Hammock Florida February 2023, Hundred Arizo
- build: Not needed. The one integration cost is that rgb_path, thermal_path and the tiff_path inside the gt blocks are the author's absolute local paths (for example /home/mhabibp/Downloads/flame3333/Flame 3 Computer Vision Sets/Wilamette/Office Un
- effort: About 3 to 5 person-days. One day to register an IEEE account and pull the 38.82 GB, one day for path remapping and a loader, one day to sub

**[not-feasible-now] Contextual risk reasoning over satellite detections (turn detector outputs into a risk assessment and response** (Detection and Perception, wildfire)
- data: "Wildfire-Smoke-RS", announced in the WildfireVLM paper and README: 3,771 labeled images, classes Fire Zone and Smoke Plume, 416x416 multispectral-aligned patches from Landsat-8/9 (15 to 30 m) and GOES-16 (2 km, five-min
- page: https://github.com/Ayanzadeh93/_WildfireVLM_
- access: None found. The paper states the code and dataset are publicly available at this repository, and the README carries a green "Dataset" badge, but the badge links back to the same repository and the repository contains only README.md (4,897 bytes) and an assets/
- license: Not stated anywhere. The repository has no LICENSE file and the GitHub license field is null. The arXiv preprint is under the arXiv nonexclusive-distrib 1.0 license, not a Creative Commons license, which grants arXiv distribution rights only and does not permi
- size: 3,771 images claimed, 416x416 patches, two classes. Unverifiable, since nothing is distributed.
- labels: No, and this is the deeper problem. The detection half has class labels in principle, but the task the outline motivates is the reasoning half, and that has no reference answers at all. The paper scores open-ended risk reports with an LLM-a
- contamination: Not datable. The paper does not state the acquisition window for the Landsat-8/9 and GOES-16 scenes, and no per-item dates exist because no items exist. The preprint went up on 2026-02-09 with v2 on 2
- build: Build both halves ourselves. Detection items: use TS-SatFire, which we fetched and verified at https://pmc.ncbi.nlm.nih.gov/articles/PMC12630597/ with data at https://www.kaggle.com/datasets/z789456sx/ts-satfire , CC BY 4.0, 71 GB, 179 wild
- effort: About 2 to 3 weeks. Roughly a week to build the detection-plus-progression item generator over TS-SatFire, a week to design and pilot the ch

**[needs-work] UAV imagery tasks including wildfire propagation (multiple-choice reasoning over low-altitude UAV disaster ima** (Detection and Perception, both (the benchmark')
- data: DisasterBench, public test split. The repository README: "At present, we provide the test set for evaluation purposes. The train and val directories are initialized but kept empty, as the training and validation sets are
- page: https://github.com/TanmouTT/DisasterBench
- access: Public download of the test split, no registration. We pulled data/test/test_3000.json directly (7,019,609 bytes, HTTP 200) and confirmed 1,275 image files under data/test/images/ via the GitHub tree API. Repository created 2026-06-01, last pushed 2026-09-09, 
- license: Apache-2.0, declared in a LICENSE file (11,357 bytes) and confirmed by the GitHub API license field. Apache-2.0 permits research use, modification and redistribution of derived items with attribution and a notice file. The caveat is upstream: the paper says th
- size: Full benchmark as described: 5,330 real-world low-altitude UAV images, 29,300 reasoning samples, 14 scene types, 9 tasks. Public test split as measured: 3,000 items and 1,275 image
- labels: Yes, and they are in the public file. Each record carries image, question, options, reason and answer. The paper: "All samples in DisasterBench follow a unified multiple-choice VQA format ... performance is measured using exact-match accura
- contamination: Individual items cannot be dated. Image names are sequential (fire_249.png), no capture dates or source URLs travel with the data, and the paper gives no acquisition window for the scraped footage. Th
- build: Not needed for the fire slice. The 402 fire items are usable as shipped. If we want a wildfire-only subset separate from structure fire, the scene taxonomy does not provide it (Fire is a single category with no wildfire sub-label), so a man
- effort: About 3 to 5 person-days. Half a day to clone the 1.71 GB repository and load the test JSON, half a day to split the 402 fire items into wil

**[ready] Source check: FIgLib as a basis for wildfire smoke perception items (binary smoke presence, time to detection,** (Detection and Perception, candidate data, wildfire)
- data: FIgLib, the HPWREN Fire Ignition images Library: sequences of wildland fire images from fixed field-of-view cameras at HPWREN mountaintop sites in Southern California, each sequence spanning roughly 40 minutes before and
- page: https://www.hpwren.ucsd.edu/HPWREN-FIgLib/
- access: Public download, no registration, no request. Direct HTTP from https://cdn.hpwren.ucsd.edu/HPWREN-FIgLib-Data/index.html , which lists per-sequence directories of JPEGs plus an MP4 time-lapse, with whole sequences also available as tgz archives under a Tar/ di
- license: No formal license. Two sentences govern use: "This data is provided as-is, no guarantees for anything is included" and "While we are making the data publicly available, its use requires a credit reference to https://www.hpwren.ucsd.edu/ in derivative work." Re
- size: Roughly 30 GB as of July 2024 per the page, growing. We counted 453 distinct dated sequences in the current index (the page says "more than 400"); the 2022 paper's snapshot was 315
- labels: Yes, and free. Image file names are origin_timestamp_offset-in-seconds-from-visible-plume-appearance.jpg (for example 1720453551_-02400.jpg, 1720455951_+00000.jpg, 1720458232_+02281.jpg), so the sign of the offset is a per-image binary smok
- contamination: Every item dates itself. Directory names are YYYYMMDD_firename_camera, for example 20170708_Whittier_syp-n-mobo-m. Current range runs 2016-06-04 to 2026-09-09, with per-year counts of 14 (2016), 41 (2
- build: Not needed; this is itself the build route for item 2 above and a natural home for the outline's post-cutoff-items bullet in 7.8. Concretely: crawl the 2025 and 2026 sequence directories, parse the offset from each file name, sample a balan
- effort: About 3 to 4 person-days for a dated binary and time-to-detection task, most of it a polite crawl and a file-name parser. Add 2 to 3 weeks i

**[not-feasible-now] Source check: DetectiumFire as a basis for fire-scene perception and risk-severity items** (Detection and Perception, candidate data, both (indoor and str)
- data: DetectiumFire, a multimodal fire-understanding dataset from Tulane University with Aalto University, NeurIPS 2025 Datasets and Benchmarks track.
- page: https://www.kaggle.com/datasets/38b79c344bdfc55d1eed3d22fbaa9c31fad45e27edbbe9e3c529d6e5c4f93890
- access: Public download on Kaggle behind a free account. The Kaggle distribution record marks the zip download "requiresSubscription": true, which in Kaggle's schema means a signed-in user. Code and metadata at https://github.com/ZixuanLiu4869/DetectiumFire (created 2
- license: This is the blocker, and the two authoritative sources disagree. The arXiv paper states "License: CC BY-NC-SA 4.0". The Kaggle page, which is where the data is actually distributed, records "Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-N
- size: 22.5k fire-related images (14.5k real-world plus 8k synthetic) and 2.5k videos. The curated annotated core is 7k fire images with 7k deliberately hard non-fire images, and 1.7k fir
- labels: Partly, and none of it is question-shaped. What exists: YOLO-format bounding boxes made on Roboflow by human annotators over several months, with inter-annotator agreement reported in Appendix B.2; captions capped at 75 tokens covering what
- contamination: Items cannot be dated. The images were scraped from Google, Twitter, YouTube and TikTok plus some IoT capture during controlled demonstrations, and no capture dates, source URLs or acquisition window 
- build: If we want the severity-reasoning idea without the license problem, build it elsewhere. For structure fire, Fire360 (row below) has expert-verified labels and a research-use license. For wildfire severity, FIgLib gives dated imagery and fre
- effort: Low if only used internally: about 2 days to download, subset to the 1,045 forest and wildfire plus 1,159 residential images, and turn the f

**[needs-work] Source check: Fire360 as a basis for structure-fire perception and safety-reasoning items** (Detection and Perception, candidate data, structure fire (fire)
- data: Fire360, a benchmark of degraded 360-degree firefighting training video, NeurIPS 2025 Datasets and Benchmarks track.
- page: https://arxiv.org/abs/2506.02167
- access: Public download, no login. The dataset is hosted at https://uofi.box.com/v/fire360dataset , a University of Illinois Box shared link that resolves to uofi.app.box.com and returns HTTP 200 with public shared-item metadata. The paper notes the hosting choice and
- license: "Fire360 is released under a research-only MIT license with added restrictions that prohibit use in surveillance, enforcement, behavioral profiling, or nonconsensual monitoring contexts." That is a custom restricted license rather than MIT, despite the name. A
- size: 228 professionally recorded 360-degree videos, 180,000 seconds (50 hours), captured on a Ricoh Theta V at 3840x1920 and 60 fps, stitched to equirectangular panoramas. Splits are 13
- labels: Yes, expert-verified, with human baselines, which is rare in this group. Annotations: 348 temporal action instances across 8 categories; spatial bounding boxes averaging 5.7 objects per video; environmental tags including a 1 to 5 smoke gra
- contamination: Items cannot be dated precisely. The paper says recordings span summer and winter sessions at one North American firefighter training institute and gives no calendar dates. Public since the arXiv post
- build: Not needed for access; needed for scale. The limiting number is 100 VQA questions, which is too few to separate models with confidence, and 50 of those are free text. The extension route is documented and cheap: the authors ship the browser
- effort: About 1 to 2 weeks to use the 100 existing VQA items plus the safety checklists as shipped, mostly pulling 50 hours of 360-degree video and 

**[not-feasible-now] Source check: RSCC as a basis for post-fire change and burn-scar perception items** (Detection and Perception, candidate data, wildfire (post-event)
- data: RSCC, Remote Sensing Change Caption dataset for disaster events, NeurIPS 2025 Datasets and Benchmarks track. Built by cropping and captioning xBD and EBD imagery, both from the MAXAR Open Data Program.
- page: https://huggingface.co/datasets/BiliSakura/RSCC
- access: Public download, not gated. Hugging Face record: gated false, private false, created 2025-04-22, last modified 2026-02-14, 347 downloads, size category 10K to 100K. Code and a Google Drive subset at https://github.com/Bili-Sakura/RSCC , project page at https:/
- license: Declared CC-BY-4.0, on the Hugging Face card and in the paper's own data-license section: "The dataset is released under the [CC-BY-4.0], which permits unrestricted use, distribution, and reproduction in any medium, provided the original work is properly cited
- size: 62,351 bi-temporal pre/post image pairs at 512x512 (xBD 44,136 cropped without overlap from 1024x1024, EBD 18,215 native), across 31 global events. Splits: train 61,363 pairs over 
- labels: Yes in form, no in substance, and this is the reason for the verdict. The reference change captions were generated by a model, not written by people: "we call QvQ-Max (qvq-max-2025-03-25) API from Alibaba Cloud to automatically generate ann
- contamination: Events are dated precisely in the paper's Appendix A table. The 5 wildfire events are Portugal wildfires (17 to 24 June 2017), Santa Rosa wildfires (8 to 31 October 2017), Carr wildfire (23 July to 30
- build: Skip RSCC and go to the layer beneath it. xBD provides human building bounding boxes with Joint Damage Scale labels, described in the RSCC paper as "human annotations of building bounding boxes with damage assessment labels ... developed wi
- effort: About 1 week to build a damage-level item set from xBD's wildfire events once its license and download are verified, plus the license check 

