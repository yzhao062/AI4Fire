"""Two checks on the rule-v2 implementation inside run_allocation.py.

1. Regression: the pool rows gained fields for v2, and the v1 draw on the 300 items must still equal the
   analogue_ids stored in every v1 grounded response file.
2. Transfer: the runner's v2 draw on the 300 items must equal the frozen family-B rule (nn_pers_chg) applied
   through the development harness, which is the code the decision was made on.

    python retrieval-v2/check_v2_transfer.py
"""
import json
import pathlib
import random
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "design-B"))
import run_allocation as ra  # noqa: E402
import dev_eval  # noqa: E402
import rule as rule_b  # noqa: E402

items = ra.sample_items()
pool = ra.build_pool({i["incident_id"] for i in items})

# 1. v1 regression against the stored draws
stored = {}
for path in sorted((ROOT / "task-allocation").glob("responses-*-grounded.jsonl")):
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if len(rows) < 100:
        continue
    stored[path.name] = {r["item_id"]: r["analogue_ids"] for r in rows}
fresh_v1 = {it["item_id"]: [r["analogue_id"] for r in ra.analogues(pool, it)] for it in items}
for name, s in stored.items():
    diff = sum(1 for i in fresh_v1 if fresh_v1[i] != s.get(i))
    print("v1 regression against %-64s differing items: %d" % (name, diff))

# 2. transfer: runner v2 against the harness + frozen rule
v2 = ra.RuleV2(pool)
runner = {it["item_id"]: [r["analogue_id"] for r in v2.draw(it)] for it in items}
hpool = dev_eval.load_pool()
rule_b.prepare(hpool)
frozen = rule_b.RULES[rule_b.FROZEN]
harness = {}
for it in items:
    cands = dev_eval.candidates(hpool, it)
    harness[it["item_id"]] = [r["analogue_id"] for r in frozen(cands, it, random.Random(0))]
same = sum(1 for i in runner if runner[i] == harness[i])
print("v2 transfer: runner draw equals harness draw on %d of %d items (frozen rule %s)" % (same, len(items), rule_b.FROZEN))
for i in list(runner)[:300]:
    if runner[i] != harness[i]:
        print("  differs:", i, runner[i], harness[i])
        break
cov3 = sum(1 for i in runner if len(runner[i]) >= 3) / len(items)
print("v2 coverage-3 on the 300 items: %.3f; mean rows drawn %.2f" % (cov3, sum(len(v) for v in runner.values()) / len(items)))
raise SystemExit(0 if same == len(items) else 1)
