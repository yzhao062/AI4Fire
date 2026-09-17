"""Reproduce the collection profile in Appendix A (Table: provisional search-unit counts) from the pass-2 record.

Usage: python survey/collection_counts.py

Every count is a search-unit tag that was never re-checked; the paper says so. The keyword screen and the
model-count buckets are the ones the paper describes: a work counts as LLM work with an evaluation when the
unit's llm_keyword_screen is true and its kind carries an evaluation.
"""
import collections
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
d = json.load((HERE / "prior-art-r2-2026-09-13.json").open(encoding="utf-8"))
works = d["works"]
print("kept works:", len(works), "| searched:", d["meta"]["searched"], "| window:", d["meta"]["window"])

kinds = collections.Counter(w["kind"] for w in works)
print("\nKind")
for k, n in kinds.most_common():
    print("  %-18s %3d" % (k, n))

tags = collections.Counter(t for w in works for t in w["task_categories"])
print("\nTask tag (one work can carry several)")
for k, n in tags.most_common():
    print("  %-28s %3d" % (k, n))

screen = [w for w in works if w.get("llm_keyword_screen")]
evals = [w for w in screen if w["kind"] in ("system-with-eval", "evaluation", "benchmark")]
print("\nkeyword screen finds an LLM, VLM, or agentic framework in %d works; %d of those carry an evaluation" % (len(screen), len(evals)))


def bucket(n):
    """n_models is an int, a numeric string, or 'unknown' in the unit records."""
    if n is None or n == "unknown":
        return "unknown"
    n = int(n)
    if n == 1:
        return "one"
    if n <= 3:
        return "2-3"
    if n <= 9:
        return "4-9"
    return "10 or more"


models = collections.Counter(bucket(w.get("n_models")) for w in evals)
print("\nModels tested, over the %d evaluated LLM works" % len(evals))
for k in ("unknown", "one", "2-3", "4-9", "10 or more"):
    print("  %-12s %3d" % (k, models.get(k, 0)))

rechecked = [w for w in works if w.get("recheck")]
print("\nworks with a re-check record:", len(rechecked))
print("  exists:", collections.Counter(w["recheck"].get("exists") for w in rechecked))
print("  re-check fields:", sorted({k for w in rechecked for k in w["recheck"]}))
