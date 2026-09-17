"""Probe a gateway model on one allocation item, one Mesogeos item, and one FIgLib item, bare condition.

Prints the parsed answer, the token usage including reasoning tokens where the gateway reports them, so a
reasoning model's output cap can be checked before a full run spends the budget.

Usage: python probe_gateway.py gpt-6-astra
"""
import json
import sys

import gw
import run_allocation as ra
import run_mesogeos as rm
import run_figlib as rf

model = sys.argv[1]
key = gw.load_key()


def show(label, text, usage, served):
    d = usage or {}
    det = d.get("completion_tokens_details") or {}
    print("%-18s served=%s prompt=%s completion=%s reasoning=%s" % (
        label, served, d.get("prompt_tokens"), d.get("completion_tokens"), det.get("reasoning_tokens")))
    print("   raw:", (text or "")[:220].replace("\n", " "))


it = ra.sample_items()[0]
text, usage, served = gw.call(key, model, ra.render(it), max_tokens=ra.MAX_OUT)
show("allocation bare", text, usage, served)
print("   parsed:", ra.parse(text), "| target", it["target_personnel"], "| persistence", it["baseline_persistence"])

mit = next(json.loads(l) for l in (rm.TASK / "items.jsonl").read_text(encoding="utf-8").splitlines() if '"test"' in l)
text, usage, served = gw.call(key, model, rm.render(mit), max_tokens=rm.MAX_OUT)
show("mesogeos bare", text, usage, served)
print("   parsed:", rm.parse(text), "| label", mit.get("label"))

fit = next(json.loads(l) for l in (rf.TASK / "items.jsonl").read_text(encoding="utf-8").splitlines() if '"smoke"' in l and '"no smoke"' not in l)
text, usage, served = gw.call(key, model, rf.messages(fit), max_tokens=rf.MAX_OUT)
show("figlib bare", text, usage, served)
print("   parsed:", rf.parse(text), "| label", fit["label"])
