"""Run one model through the three version-1 tasks, one runner after another, logging each.

Usage: python run_all.py bedrock:qwen.qwen3-vl-235b-a22b [gpt-6-astra ...]

A model name with the bedrock: prefix goes through Bedrock; any other name goes through the gateway. Workers
can be set with --workers N before the model names (default 4). The log for each (model, task) pair lands in
logs/<path>-<safe model>-<task>.log, where path is bedrock or gateway.

Each runner rewrites its own response files at the end of each condition, so a task that dies leaves the
previous files intact.
"""
import pathlib
import re
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
LOGS = HERE / "logs"
LOGS.mkdir(exist_ok=True)
args = sys.argv[1:]
workers = "4"
if args[:1] == ["--workers"]:
    workers, args = args[1], args[2:]
RUNNERS = [("allocation", ["run_allocation.py", "--workers", workers]),
           ("mesogeos", ["run_mesogeos.py", "--workers", workers]),
           ("figlib", ["run_figlib.py", "--workers", workers])]

for model in args:
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", model)
    path = "bedrock" if model.startswith("bedrock:") else "gateway"
    for task, argv in RUNNERS:
        log = LOGS / ("%s-%s-%s.log" % (path, safe, task))
        t0 = time.time()
        with log.open("w", encoding="utf-8") as fh:
            rc = subprocess.call([sys.executable] + argv + ["--models", model], cwd=str(HERE), stdout=fh, stderr=subprocess.STDOUT)
        print("%s %s rc=%d %.0fs" % (model, task, rc, time.time() - t0), flush=True)
print("ALL DONE", flush=True)
