"""Probe a Bedrock model on one FIgLib item per condition and one Mesogeos item, printing tokens and the parsed answer.

Usage: python probe_bedrock.py bedrock:us.openai.gpt-6-astra
"""
import json
import pathlib
import sys
import time

import gw
import run_figlib
import run_mesogeos

model = sys.argv[1]
items = [json.loads(l) for l in (run_figlib.TASK / "items.jsonl").read_text(encoding="utf-8").splitlines()]
seq = items[0]["sequence"]
frames = sorted((i for i in items if i["sequence"] == seq), key=lambda i: i["offset_seconds"])
ref, judge = frames[0], frames[-1]
for cond, r in (("bare", None), ("grounded", ref["image"])):
    t0 = time.time()
    try:
        text, usage, served = gw.call(None, model, run_figlib.messages(judge, r), max_tokens=run_figlib.MAX_OUT)
        print("figlib %-8s %5.1fs usage=%s parsed=%s label=%s" % (cond, time.time() - t0, usage, run_figlib.parse(text), judge["label"]))
        print("   text:", repr(text[:160]))
    except Exception as exc:
        print("figlib %-8s FAILED: %s" % (cond, str(exc)[:300]))

mitems = [json.loads(l) for l in (run_mesogeos.TASK / "items.jsonl").read_text(encoding="utf-8").splitlines()]
it = next(i for i in mitems if i["split"] == "test")
clim = run_mesogeos.climatology()
for cond in ("bare", "grounded"):
    t0 = time.time()
    try:
        month = it["context"]["window_end"][5:7]
        msgs = run_mesogeos.render(it, clim.get(month) if cond == "grounded" else None)
        text, usage, served = gw.call(None, model, msgs, max_tokens=run_mesogeos.MAX_OUT)
        print("mesogeos %-8s %5.1fs usage=%s" % (cond, time.time() - t0, usage))
        print("   text:", repr(text[:200]))
    except Exception as exc:
        print("mesogeos %-8s FAILED: %s" % (cond, str(exc)[:300]))
        break
