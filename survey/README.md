# Survey and data-check record

The survey half of the paper and the data availability check behind the benchmark, in the form they were produced. Nothing here calls a model.

| File | What it is |
|---|---|
| `prior-art-2026-09-13.json` | Pass 1 (2026-09-13): two panels, Claude and Gemini, each searching four angles for an existing survey-plus-benchmark of AI on fire tasks. |
| `prior-art-r2-2026-09-13.json` | Pass 2 (2026-09-13): 13 search units on Agy (Gemini 3.8 Flash High) over LLMs and agents on fire tasks, 138 kept works, each with the unit that found it, its kind, task-category tags, a keyword screen, and the models it evaluated. The 31 records that a scoping summary cited carry a `recheck` object with the fetched URLs, the outcome, and any correction. Unit tags and counts were never re-checked; the paper says so wherever it uses them. |
| `PRIOR-ART.md` | The scoping summary whose 31 cited records were re-checked claim by claim. |
| `data-check-2026-09-15.md` | The data availability check of 34 candidate tasks and sources (Appendix B): for each, the pages fetched, access, license, labels, contamination risk, and the verdict (3 ready, 20 need work, 11 not feasible). |
| `data-check-digest.md` | The short form of the same check. |
| `collection_counts.py` | Reproduces the collection profile table of Appendix A (kind, task tags, keyword screen 109 of 138, 90 with an evaluation, models-tested buckets) from `prior-art-r2-2026-09-13.json`. |

```
python survey/collection_counts.py
```

The search units, the unit contract, and the counting rules are described in Appendix A of the paper. The searches are agent-run and not saturated; several works surfaced in only one unit.
