"""Model registry for AI4Fire benchmarks.

This module provides a single registry for all evaluated models, their labels, stems,
serving paths, weights groups, tiers, and display order.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Model:
    """Registry entry for an evaluated or tested model."""
    label: str
    stem: str
    model: str
    vendor: str
    weights: str  # "proprietary" or "open"
    path: str     # "gateway" or "bedrock"
    tier: str     # "core", "added", or "text"
    order: int
    max_out: int = 1536  # output cap the run used; 8192 for the reasoning models whose reasoning exhausted 1536
    tools: bool = True    # False for models whose tool probe failed; they run allocation and fire danger only

    @property
    def family(self) -> str:
        """Alias for figstyle backward compatibility ('open-weight' or 'proprietary')."""
        return "open-weight" if self.weights == "open" else "proprietary"

    def __getitem__(self, key: str) -> Any:
        if key == "family":
            return self.family
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except AttributeError:
            return default


# All models in display order: the sixteen full-capability models first, proprietary (orders 1..7) before open
# weight (10..18), then the nineteen text-only models of the 2026-09-18 sweep (20..38). Core models keep their
# six-model relative order (1, 2, 3, 4, 10, 11). Tiers: core (the six reported models), added (the ten Bedrock
# models that accept images and tools), text (text-only; tools=False for the two whose tool probe failed).
REGISTRY: List[Model] = [
    # --- Core proprietary models (4) ---
    Model(
        label="claude-opus-4.8",
        stem="claude-opus-4.8",
        model="claude-opus-4.8",
        vendor="Anthropic",
        weights="proprietary",
        path="gateway",
        tier="core",
        order=1,
    ),
    Model(
        label="claude-opus-5",
        stem="claude-opus-5",
        model="claude-opus-5",
        vendor="Anthropic",
        weights="proprietary",
        path="gateway",
        tier="core",
        order=2,
    ),
    Model(
        label="gemini-3.1-pro",
        stem="gemini-3.1-pro",
        model="gemini-3.1-pro",
        vendor="Google",
        weights="proprietary",
        path="gateway",
        tier="core",
        order=3,
    ),
    Model(
        label="gpt-6-astra",
        stem="gpt-6-astra",
        model="gpt-6-astra",
        vendor="OpenAI",
        weights="proprietary",
        path="gateway",
        tier="core",
        order=4,
    ),
    # --- Added proprietary models (3) ---
    Model(
        label="Nova Lite",
        stem="bedrock_amazon.nova-lite-v1_0",
        model="bedrock:amazon.nova-lite-v1:0",
        vendor="Amazon",
        weights="proprietary",
        path="bedrock",
        tier="added",
        order=5,
    ),
    Model(
        label="Nova Pro",
        stem="bedrock_amazon.nova-pro-v1_0",
        model="bedrock:amazon.nova-pro-v1:0",
        vendor="Amazon",
        weights="proprietary",
        path="bedrock",
        tier="added",
        order=6,
    ),
    Model(
        label="Nova 2 Lite",
        stem="bedrock_us.amazon.nova-2-lite-v1_0",
        model="bedrock:us.amazon.nova-2-lite-v1:0",
        vendor="Amazon",
        weights="proprietary",
        path="bedrock",
        tier="added",
        order=7,
    ),
    # --- Core open-weight models (2) ---
    Model(
        label="Qwen3-VL",
        stem="bedrock_qwen.qwen3-vl-235b-a22b",
        model="bedrock:qwen.qwen3-vl-235b-a22b",
        vendor="Alibaba",
        weights="open",
        path="bedrock",
        tier="core",
        order=10,
    ),
    Model(
        label="Llama 4 Maverick",
        stem="bedrock_us.meta.llama4-maverick-17b-instruct-v1_0",
        model="bedrock:us.meta.llama4-maverick-17b-instruct-v1:0",
        vendor="Meta",
        weights="open",
        path="bedrock",
        tier="core",
        order=11,
    ),
    # --- Added open-weight models (7) ---
    Model(
        label="Llama 4 Scout",
        stem="bedrock_us.meta.llama4-scout-17b-instruct-v1_0",
        model="bedrock:us.meta.llama4-scout-17b-instruct-v1:0",
        vendor="Meta",
        weights="open",
        path="bedrock",
        tier="added",
        order=12,
    ),
    Model(
        label="Mistral Large 3",
        stem="bedrock_mistral.mistral-large-3-675b-instruct",
        model="bedrock:mistral.mistral-large-3-675b-instruct",
        vendor="Mistral",
        weights="open",
        path="bedrock",
        tier="added",
        order=13,
    ),
    Model(
        label="Ministral 3 8B",
        stem="bedrock_mistral.ministral-3-8b-instruct",
        model="bedrock:mistral.ministral-3-8b-instruct",
        vendor="Mistral",
        weights="open",
        path="bedrock",
        tier="added",
        order=14,
    ),
    Model(
        label="Kimi K2.5",
        stem="bedrock_moonshotai.kimi-k2.5",
        model="bedrock:moonshotai.kimi-k2.5",
        vendor="Moonshot",
        weights="open",
        path="bedrock",
        tier="added",
        order=15,
    ),
    Model(
        label="Gemma 3 27B",
        stem="bedrock_google.gemma-3-27b-it",
        model="bedrock:google.gemma-3-27b-it",
        vendor="Google",
        weights="open",
        path="bedrock",
        tier="added",
        order=16,
    ),
    Model(
        label="Gemma 3 12B",
        stem="bedrock_google.gemma-3-12b-it",
        model="bedrock:google.gemma-3-12b-it",
        vendor="Google",
        weights="open",
        path="bedrock",
        tier="added",
        order=17,
    ),
    Model(
        label="Gemma 3 4B",
        stem="bedrock_google.gemma-3-4b-it",
        model="bedrock:google.gemma-3-4b-it",
        vendor="Google",
        weights="open",
        path="bedrock",
        tier="added",
        order=18,
    ),
    # --- Text-only models (19): allocation and fire danger, plus tool use where tools=True ---
    Model(
        label="Nova Micro",
        stem="bedrock_amazon.nova-micro-v1_0",
        model="bedrock:amazon.nova-micro-v1:0",
        vendor="Amazon",
        weights="proprietary",
        path="bedrock",
        tier="text",
        order=20,
    ),
    Model(
        label="Llama 3.3 70B",
        stem="bedrock_us.meta.llama3-3-70b-instruct-v1_0",
        model="bedrock:us.meta.llama3-3-70b-instruct-v1:0",
        vendor="Meta",
        weights="open",
        path="bedrock",
        tier="text",
        order=21,
    ),
    Model(
        label="Llama 3.1 70B",
        stem="bedrock_us.meta.llama3-1-70b-instruct-v1_0",
        model="bedrock:us.meta.llama3-1-70b-instruct-v1:0",
        vendor="Meta",
        weights="open",
        path="bedrock",
        tier="text",
        order=22,
    ),
    Model(
        label="Mistral Small 2402",
        stem="bedrock_mistral.mistral-small-2402-v1_0",
        model="bedrock:mistral.mistral-small-2402-v1:0",
        vendor="Mistral",
        weights="open",
        path="bedrock",
        tier="text",
        order=23,
    ),
    Model(
        label="Devstral 2 123B",
        stem="bedrock_mistral.devstral-2-123b",
        model="bedrock:mistral.devstral-2-123b",
        vendor="Mistral",
        weights="open",
        path="bedrock",
        tier="text",
        order=24,
    ),
    Model(
        label="Qwen3 32B",
        stem="bedrock_qwen.qwen3-32b-v1_0",
        model="bedrock:qwen.qwen3-32b-v1:0",
        vendor="Alibaba",
        weights="open",
        path="bedrock",
        tier="text",
        order=25,
    ),
    Model(
        label="Qwen3 Next 80B",
        stem="bedrock_qwen.qwen3-next-80b-a3b",
        model="bedrock:qwen.qwen3-next-80b-a3b",
        vendor="Alibaba",
        weights="open",
        path="bedrock",
        tier="text",
        order=26,
    ),
    Model(
        label="Qwen3 Coder 30B",
        stem="bedrock_qwen.qwen3-coder-30b-a3b-v1_0",
        model="bedrock:qwen.qwen3-coder-30b-a3b-v1:0",
        vendor="Alibaba",
        weights="open",
        path="bedrock",
        tier="text",
        order=27,
    ),
    Model(
        label="DeepSeek V3.2",
        stem="bedrock_deepseek.v3.2",
        model="bedrock:deepseek.v3.2",
        vendor="DeepSeek",
        weights="open",
        path="bedrock",
        tier="text",
        order=28,
    ),
    Model(
        label="GPT-OSS 120B",
        stem="bedrock_openai.gpt-oss-120b-1_0",
        model="bedrock:openai.gpt-oss-120b-1:0",
        vendor="OpenAI",
        weights="open",
        path="bedrock",
        tier="text",
        order=29,
    ),
    Model(
        label="GPT-OSS 20B",
        stem="bedrock_openai.gpt-oss-20b-1_0",
        model="bedrock:openai.gpt-oss-20b-1:0",
        vendor="OpenAI",
        weights="open",
        path="bedrock",
        tier="text",
        order=30,
    ),
    Model(
        label="GLM 5",
        stem="bedrock_zai.glm-5",
        model="bedrock:zai.glm-5",
        vendor="Z.AI",
        weights="open",
        path="bedrock",
        tier="text",
        order=31,
    ),
    Model(
        label="GLM 4.7",
        stem="bedrock_zai.glm-4.7",
        model="bedrock:zai.glm-4.7",
        vendor="Z.AI",
        weights="open",
        path="bedrock",
        tier="text",
        order=32,
    ),
    Model(
        label="GLM 4.7 Flash",
        stem="bedrock_zai.glm-4.7-flash",
        model="bedrock:zai.glm-4.7-flash",
        vendor="Z.AI",
        weights="open",
        path="bedrock",
        tier="text",
        order=33,
    ),
    Model(
        label="MiniMax M2.5",
        stem="bedrock_minimax.minimax-m2.5",
        model="bedrock:minimax.minimax-m2.5",
        vendor="MiniMax",
        weights="open",
        path="bedrock",
        tier="text",
        order=34,
        max_out=8192,
    ),
    Model(
        label="Kimi K2 Thinking",
        stem="bedrock_moonshot.kimi-k2-thinking",
        model="bedrock:moonshot.kimi-k2-thinking",
        vendor="Moonshot",
        weights="open",
        path="bedrock",
        tier="text",
        order=35,
        max_out=8192,
    ),
    Model(
        label="Nemotron Super 3 120B",
        stem="bedrock_nvidia.nemotron-super-3-120b",
        model="bedrock:nvidia.nemotron-super-3-120b",
        vendor="NVIDIA",
        weights="open",
        path="bedrock",
        tier="text",
        order=36,
    ),
    Model(
        label="Llama 3.1 8B",
        stem="bedrock_us.meta.llama3-1-8b-instruct-v1_0",
        model="bedrock:us.meta.llama3-1-8b-instruct-v1:0",
        vendor="Meta",
        weights="open",
        path="bedrock",
        tier="text",
        order=37,
        tools=False,
    ),
    Model(
        label="DeepSeek R1",
        stem="bedrock_us.deepseek.r1-v1_0",
        model="bedrock:us.deepseek.r1-v1:0",
        vendor="DeepSeek",
        weights="open",
        path="bedrock",
        tier="text",
        order=38,
        max_out=8192,
        tools=False,
    ),
]

TASK_CONDITIONS: Dict[str, Sequence[str]] = {
    "figlib": ("bare", "grounded"),
    "allocation": ("bare", "grounded"),
    "mesogeos": ("bare", "grounded"),
    "tooluse": ("bare", "tool"),
    "wildfirevqa": ("bare", "grounded"),
}


def by_stem() -> Dict[str, Model]:
    """Map stem to Model entry."""
    return {m.stem: m for m in REGISTRY}


def by_label() -> Dict[str, Model]:
    """Map label to Model entry."""
    return {m.label: m for m in REGISTRY}


def _has_response_files(m: Model, task: str, root: Optional[Path] = None) -> bool:
    """Check if response files exist for the model on the specified task."""
    base_root = root or ROOT
    task_clean = task.replace("task-", "")
    task_dir = base_root / f"task-{task_clean}"
    if not task_dir.is_dir():
        return False
    conditions = TASK_CONDITIONS.get(task_clean, ("bare", "grounded"))
    for cond in conditions:
        p = task_dir / f"responses-{m.stem}-{cond}.jsonl"
        # Also allow aerial -pilot suffix if regular does not exist
        if not p.exists():
            p_pilot = task_dir / f"responses-{m.stem}-{cond}-pilot.jsonl"
            if not p_pilot.exists():
                return False
    return True


def models(tier: Optional[str] = None, task: Optional[str] = None, root: Optional[Path] = None) -> List[Model]:
    """Return models filtered by tier and optional task presence, sorted by order.

    tier: "core" (the six reported models), "added" (the ten sweep models that ran all five tasks),
          "all" or its alias "full" (those sixteen), "text" (the nineteen text-only sweep models),
          "every" (all 35), or None (all tiers).
    task: task name ("figlib", "allocation", "mesogeos", "tooluse", "wildfirevqa")
          restricting to models whose response files exist for both conditions.
    """
    res = list(REGISTRY)
    if tier is not None:
        tier_lower = tier.lower()
        if tier_lower == "core":
            res = [m for m in res if m.tier == "core"]
        elif tier_lower == "added":
            res = [m for m in res if m.tier == "added"]
        elif tier_lower in ("all", "full"):
            res = [m for m in res if m.tier in ("core", "added")]
        elif tier_lower == "text":
            res = [m for m in res if m.tier == "text"]
        elif tier_lower == "every":
            pass
        else:
            raise ValueError(f"Unknown tier: {tier!r}")

    if task is not None:
        res = [m for m in res if _has_response_files(m, task, root)]

    return sorted(res, key=lambda m: m.order)


def stems_for(task: str, tier: Optional[str] = None, root: Optional[Path] = None) -> List[str]:
    """Return model stems whose response files exist for the task."""
    return [m.stem for m in models(tier=tier, task=task, root=root)]


def add_model_args(parser, default_tier: str = "core") -> None:
    """Add standardized --tier and --models arguments to an argparse parser."""
    parser.add_argument(
        "--tier",
        choices=["core", "added", "all", "full", "text", "every"],
        default=default_tier,
        help=("model tier to include: core = the six reported models, added = the ten sweep models that ran "
              "all five tasks, all (alias full) = those sixteen, text = the nineteen text-only sweep models, "
              f"every = all 35 (default: {default_tier})"),
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="comma-separated list of model stems or labels to include",
    )


def resolve_models(args, task: Optional[str] = None, default_tier: str = "core", root: Optional[Path] = None) -> List[Model]:
    """Resolve Model list from parsed CLI arguments, respecting --models or --tier."""
    models_arg = getattr(args, "models", None)
    if models_arg:
        selected = [s.strip() for s in models_arg.split(",") if s.strip()]
        b_stem = by_stem()
        b_lbl = by_label()
        resolved = []
        for s in selected:
            if s in b_stem:
                resolved.append(b_stem[s])
            elif s in b_lbl:
                resolved.append(b_lbl[s])
            else:
                raise ValueError(f"Unknown model stem or label: {s!r}")
        return sorted(resolved, key=lambda m: m.order)

    tier_val = getattr(args, "tier", default_tier)
    return models(tier=tier_val, task=task, root=root)
