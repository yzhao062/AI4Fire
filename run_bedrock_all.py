"""Run one Bedrock model through the three version-1 tasks, one runner after another, logging each.

Usage: python run_bedrock_all.py bedrock:qwen.qwen3-vl-235b-a22b [bedrock:us.meta.llama4-maverick-17b-instruct-v1:0 ...]

Each runner rewrites its own response files at the end of each condition, so a task that dies leaves the
previous files intact. The log for each (model, task) pair lands in logs/bedrock-<safe model>-<task>.log.
"""
import pathlib
import re
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
LOGS = HERE / "logs"
LOGS.mkdir(exist_ok=True)
RUNNERS = [("allocation", ["run_allocation.py", "--workers", "4"]),
           ("mesogeos", ["run_mesogeos.py", "--workers", "4"]),
           ("figlib", ["run_figlib.py", "--workers", "4"])]

for model in sys.argv[1:]:
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", model)
    for task, argv in RUNNERS:
        log = LOGS / ("bedrock-%s-%s.log" % (safe, task))
        t0 = time.time()
        with log.open("w", encoding="utf-8") as fh:
            rc = subprocess.call([sys.executable] + argv + ["--models", model], cwd=str(HERE), stdout=fh, stderr=subprocess.STDOUT)
        print("%s %s rc=%d %.0fs" % (model, task, rc, time.time() - t0), flush=True)
print("ALL DONE", flush=True)
