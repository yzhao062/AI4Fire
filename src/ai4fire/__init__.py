"""AI4Fire: task metadata for a five-task wildfire benchmark for language models and agents.

This package carries the benchmark's task inventory so it can be queried without cloning the
repository. It does not run the evaluation. Running it needs the stored responses and the
analysis scripts, which live at https://github.com/yzhao062/AI4Fire and are reproduced offline
from that checkout with `python reproduce_tables.py`.
"""

__version__ = "0.0.2"
__all__ = ["TASKS", "REPOSITORY", "__version__"]

REPOSITORY = "https://github.com/yzhao062/AI4Fire"

#: The five tasks, each scored without a human in the loop against a released answer key.
#: ``grounding`` is what the grounded arm adds over the bare arm; the interventions are
#: task-specific and not comparable to one another, so results are task-conditional.
#: ``comparator`` is the non-LLM reference the task is scored beside.
TASKS = (
    {
        "key": "allocation",
        "name": "Daily personnel allocation",
        "source": "ICS-209-PLUS (St. Denis et al., 2023)",
        "license": "CC BY 4.0",
        "items": 300,
        "unit": "fire-days",
        "cluster_unit": "incident",
        "grounding": "Retrieved analogues",
        "comparator": "Persistence; trained ratio regressor",
    },
    {
        "key": "figlib",
        "name": "Wildfire smoke detection",
        "source": "FIgLib (HPWREN, UC San Diego)",
        "license": "CC BY-NC-ND 4.0",
        "items": 224,
        "unit": "frames (196 paired)",
        "cluster_unit": "fire",
        "grounding": "Reference frame",
        "comparator": "Two constants; leave-one-fire-out frame-difference detector",
    },
    {
        "key": "mesogeos",
        "name": "Fire danger forecasting",
        "source": "Mesogeos Track A (Kondylatos et al., 2023)",
        "license": "CC BY 4.0",
        "items": 386,
        "unit": "cells",
        "cluster_unit": "spatial block by month",
        "grounding": "Monthly climatology",
        "comparator": "Calendar-month prior; temperature rule; trained classifier",
    },
    {
        "key": "wildfirevqa",
        "name": "Temperature-grounded aerial question answering",
        "source": "WildFireVQA (Habibpour et al., 2026) over FLAME 3",
        "license": "Apache-2.0 / CC BY 4.0 (upstream discrepancy, unresolved)",
        "items": 408,
        "unit": "items over 390 frames",
        "cluster_unit": "frame",
        "grounding": "Thermal summary block",
        "comparator": "Held-out per-question majority; closed-form thermal rule",
    },
    {
        "key": "tooluse",
        "name": "Fire data tool use",
        "source": "FPA-FOD 6th edition (Short, 2022)",
        "license": "US Government public domain",
        "items": 156,
        "unit": "items",
        "cluster_unit": "question family",
        "grounding": "Read-only SQL tool",
        "comparator": "Best constant per question family",
    },
)
