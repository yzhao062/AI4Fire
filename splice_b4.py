"""Move the re-queried grounded rows from task-allocation/b4-rerun/ into the response files (round-2 review, B4).

Each replaced row is appended to b4-rerun/replaced-rows.jsonl with the model name, so the pre-repair state stays
on disk. Run once; a second run finds nothing to replace.
"""
import json
import pathlib

TASK = pathlib.Path(__file__).resolve().parent / "task-allocation"
RERUN = TASK / "b4-rerun"
kept = RERUN / "replaced-rows.jsonl"
with kept.open("a", encoding="utf-8") as log:
    for new_file in sorted(RERUN.glob("responses-*-grounded.jsonl")):
        target = TASK / new_file.name
        new_row = json.loads(new_file.read_text(encoding="utf-8").strip())
        rows = [json.loads(l) for l in target.read_text(encoding="utf-8").splitlines() if l.strip()]
        idx = [i for i, r in enumerate(rows) if r["item_id"] == new_row["item_id"]]
        assert len(idx) == 1, (new_file.name, idx)
        old = rows[idx[0]]
        if old.get("analogue_ids") == new_row["analogue_ids"]:
            print(new_file.name, "already spliced"); continue
        log.write(json.dumps({"file": target.name, "replaced": old}) + "\n")
        rows[idx[0]] = new_row
        target.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        print("%-72s row %d: prediction %s -> %s" % (target.name, idx[0], old.get("prediction"), new_row["prediction"]))
