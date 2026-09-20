"""Print the AI4Fire task inventory.

Running the benchmark itself needs the stored responses and the analysis scripts from the
repository; this command only answers what the tasks are and where to get them.
"""
import argparse
import json
import sys

from . import REPOSITORY, TASKS, __version__


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="ai4fire",
        description="Task metadata for the AI4Fire wildfire benchmark.",
        epilog="The benchmark itself lives at " + REPOSITORY,
    )
    parser.add_argument(
        "--json", action="store_true", help="emit the task inventory as JSON"
    )
    parser.add_argument("--version", action="version", version="ai4fire " + __version__)
    args = parser.parse_args(argv)

    if args.json:
        json.dump(list(TASKS), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    print("AI4Fire %s: five wildfire tasks, each run bare and grounded." % __version__)
    print("Each task is scored beside a non-LLM comparator, without a human in the loop.\n")
    for task in TASKS:
        print("%s  (%s)" % (task["name"], task["key"]))
        print("    source      %s [%s]" % (task["source"], task["license"]))
        print("    items       %d %s" % (task["items"], task["unit"]))
        print("    clustered   by %s" % task["cluster_unit"])
        print("    grounding   %s" % task["grounding"])
        print("    comparator  %s" % task["comparator"])
        print()
    print("Each grounding intervention appears on exactly one task, so the interventions are")
    print("not comparable to each other and the results are task-conditional.\n")
    print("Prompts, stored responses, scores, and the offline reproduction:")
    print("    %s" % REPOSITORY)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
