#!/usr/bin/env python
"""Generate the grounding contrasts figure (figures/grounding_effects.pdf and .png).

This figure displays the paired grounded-minus-bare contrasts across the four run benchmark tasks:
  - Panel (a): Smoke detection (FIgLib, 196 paired items per model)
               Recall gain (grounded minus bare) with fire-clustered 95% bootstrap intervals,
               plus false-positive rate (FPR) change shown as a secondary mint marker.
  - Panel (b): Personnel allocation (ICS-209-PLUS, 300 items)
               Normalized-error difference (grounded minus bare) for retrieval rules v1 and v2
               on the same row (with vertical offset), with incident-clustered 95% bootstrap intervals.
  - Panel (c): Fire danger forecasting (Mesogeos Track A, 386 items)
               AUPRC difference (grounded minus bare) with 1-degree-by-calendar-month clustered
               95% bootstrap intervals.
  - Panel (d): Fire data tool use (FPA-FOD, 156 items)
               Accuracy difference (tool minus bare) with family-clustered 95% bootstrap
               intervals (12 clusters).

Model order (top to bottom):
  1. claude-opus-4.8
  2. claude-opus-5
  3. gemini-3.1-pro
  4. gpt-6-astra
  5. Qwen3-VL
  6. Llama 4 Maverick

Data sources in the AI4Fire repository:
  - analysis/cluster_uncertainty.py (--manifest manifest-v1.json --max-skew-min 100000)
    or precomputed analysis/cluster-six.stdout.txt:
    supplies paired contrasts and clustered 95% bootstrap intervals (20,000 resamples,
    base seed 20260915, clusters by fire for FIgLib, incident for allocation, and
    1.0 deg x calendar month for Mesogeos).
  - analysis/retrieval_v2.json:
    supplies allocation rule-v2 contrasts (v2 minus bare) with incident-clustered intervals.
  - analysis/figlib-paired.json:
    supplies per-model paired FIgLib metrics including bare and grounded FPR.
  - analysis/tooluse_paired.json (written by analysis/tooluse_paired.py):
    supplies the tool-minus-bare accuracy difference per model with its family-clustered
    interval (12 clusters, 20,000 resamples).

Styling conventions (CatchBench submission style):
  - Width: 6.5 in, height <= 2.6 in.
  - Typography: Sans-serif (DejaVu Sans / Arial), TrueType fonts (pdf.fonttype 42), >= 6 pt.
  - Spines: no top/right spines, gray (#999999) left/bottom spines, no tick marks.
  - Palette:
      Coral (#ED8D5A): focal marks (contrasts whose 95% interval excludes zero)
      Gray (#999999 / #C9C9C9): context (contrasts whose 95% interval includes zero)
      Mint (#BFDFD2, edge #8FB7A6): comparison layer (FPR change in panel a)
      Near-black (#1A1A1A): text
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# CatchBench Palette
COLOR_CORAL = "#ED8D5A"
COLOR_MINT = "#BFDFD2"
COLOR_MINT_EDGE = "#8FB7A6"
COLOR_GRAY = "#999999"
COLOR_LIGHT_GRAY = "#E6E6E6"
COLOR_TEXT = "#1A1A1A"
COLOR_SUBTITLE = "#666666"

MODELS = [
    {
        "id": "claude-opus-4.8",
        "label": "claude-opus-4.8",
        "cu_key": "claude-opus-4.8",
        "v2_key": "claude-opus-4.8",
        "fig_key": "claude-opus-4.8",
        "tool_key": "claude-opus-4.8",
    },
    {
        "id": "claude-opus-5",
        "label": "claude-opus-5",
        "cu_key": "claude-opus-5",
        "v2_key": "claude-opus-5",
        "fig_key": "claude-opus-5",
        "tool_key": "claude-opus-5",
    },
    {
        "id": "gemini-3.1-pro",
        "label": "gemini-3.1-pro",
        "cu_key": "gemini-3.1-pro",
        "v2_key": "gemini-3.1-pro",
        "fig_key": "gemini-3.1-pro",
        "tool_key": "gemini-3.1-pro",
    },
    {
        "id": "gpt-6-astra",
        "label": "gpt-6-astra",
        "cu_key": "gpt-6-astra",
        "v2_key": "gpt-6-astra",
        "fig_key": "gpt-6-astra",
        "tool_key": "gpt-6-astra",
    },
    {
        "id": "qwen3-vl",
        "label": "Qwen3-VL",
        "cu_key": "bedrock_qwen.qwen3-vl-235b-a22",
        "v2_key": "bedrock_qwen.qwen3-vl-235b-a22b",
        "fig_key": "bedrock_qwen.qwen3-vl-235b-a22b",
        "tool_key": "Qwen3-VL",
    },
    {
        "id": "llama-4-maverick",
        "label": "Llama 4 Maverick",
        "cu_key": "bedrock_us.meta.llama4-maveric",
        "v2_key": "bedrock_us.meta.llama4-maverick-17b-instruct-v1_0",
        "fig_key": "bedrock_us.meta.llama4-maverick-17b-instruct-v1_0",
        "tool_key": "Llama 4 Maverick",
    },
]


def load_cluster_uncertainty_output(repo_root: Path, force_rerun: bool = False) -> str:
    """Read cached cluster uncertainty stdout or run cluster_uncertainty.py."""
    cached_path = repo_root / "analysis" / "cluster-six.stdout.txt"
    if not force_rerun and cached_path.exists():
        return cached_path.read_text(encoding="utf-8")

    script_path = repo_root / "analysis" / "cluster_uncertainty.py"
    cmd = [
        sys.executable,
        str(script_path),
        "--manifest", "manifest-v1.json",
        "--max-skew-min", "100000",
    ]
    proc = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True, check=True)
    return proc.stdout


def parse_cluster_uncertainty(text: str) -> dict[str, dict[str, dict[str, float]]]:
    """Parse bare, grounded, diff, ci_lo, ci_hi from cluster_uncertainty stdout."""
    row_pattern = re.compile(
        r"^\s*(?P<arm>[a-zA-Z0-9_\.\-]+)\s+(?P<bare_n>\d+)\s+(?P<grd_n>\d+)\s+(?P<pair_n>\d+)\s+"
        r"(?P<n_clust>\d+)\s+(?P<bare>[\d\.\-]+)\s+(?P<grounded>[\d\.\-]+)\s+(?P<diff>[\d\.\-]+)\s+"
        r"\[(?P<ci_lo>[\d\.\-]+)\s*,\s*(?P<ci_hi>[\d\.\-]+)\]",
        re.M,
    )
    tasks = ["FIgLib", "allocation", "Mesogeos"]
    out: dict[str, dict[str, dict[str, float]]] = {}
    for i, t in enumerate(tasks):
        start = text.find(t)
        if start == -1:
            continue
        end = text.find(tasks[i + 1]) if i + 1 < len(tasks) else len(text)
        sub = text[start:end]
        out[t] = {}
        for m in row_pattern.finditer(sub):
            arm = m.group("arm")
            out[t][arm] = {
                "bare": float(m.group("bare")),
                "grounded": float(m.group("grounded")),
                "diff": float(m.group("diff")),
                "ci_lo": float(m.group("ci_lo")),
                "ci_hi": float(m.group("ci_hi")),
            }
    return out


def load_retrieval_v2(repo_root: Path) -> dict:
    path = repo_root / "analysis" / "retrieval_v2.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_figlib_paired(repo_root: Path) -> dict:
    path = repo_root / "analysis" / "figlib-paired.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    fpr: dict[str, dict[str, float]] = {}
    for r in data.get("rows", []):
        m = r["model"]
        c = r["condition"]
        fpr.setdefault(m, {})[c] = float(r["fpr"])
    return fpr


def load_tooluse_paired(repo_root: Path) -> dict:
    path = repo_root / "analysis" / "tooluse_paired.json"
    if not path.exists():
        raise SystemExit(f"{path} is missing; run analysis/tooluse_paired.py first")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["models"]


def gather_all_data(repo_root: Path, force_rerun: bool = False):
    cu_text = load_cluster_uncertainty_output(repo_root, force_rerun)
    cu_data = parse_cluster_uncertainty(cu_text)
    v2_data = load_retrieval_v2(repo_root)
    fig_data = load_figlib_paired(repo_root)
    tool_data = load_tooluse_paired(repo_root)

    records = []
    for m in MODELS:
        # 1. Panel (a) Smoke detection: recall gain (grounded minus bare)
        # In cu_data: diff = bare - grounded.
        # Thus recall gain = grounded - bare = -diff.
        # Clustered CI on gain = [-ci_hi, -ci_lo].
        cu_fig = cu_data["FIgLib"][m["cu_key"]]
        gain_pt = cu_fig["grounded"] - cu_fig["bare"]
        gain_lo = -cu_fig["ci_hi"]
        gain_hi = -cu_fig["ci_lo"]
        gain_excl_zero = (gain_lo > 0 and gain_hi > 0) or (gain_lo < 0 and gain_hi < 0)

        # FPR change
        fpr_bare = fig_data[m["fig_key"]]["bare"]
        fpr_grd = fig_data[m["fig_key"]]["grounded"]
        fpr_diff = fpr_grd - fpr_bare

        # 2. Panel (b) Allocation: normalized error diff (grounded minus bare)
        # For v1: in cu_data, diff = bare - grounded.
        # Thus diff (v1 - bare) = grounded - bare = -diff.
        # Clustered CI = [-ci_hi, -ci_lo].
        cu_alloc = cu_data["allocation"][m["cu_key"]]
        v1_pt = cu_alloc["grounded"] - cu_alloc["bare"]
        v1_lo = -cu_alloc["ci_hi"]
        v1_hi = -cu_alloc["ci_lo"]
        v1_excl_zero = (v1_lo > 0 and v1_hi > 0) or (v1_lo < 0 and v1_hi < 0)

        # For v2: from retrieval_v2.json
        v2_m = v2_data[m["v2_key"]]
        v2_pt = float(v2_m["d_v2_bare"])
        v2_lo = float(v2_m["d_v2_bare_ci"][0])
        v2_hi = float(v2_m["d_v2_bare_ci"][1])
        v2_excl_zero = (v2_lo > 0 and v2_hi > 0) or (v2_lo < 0 and v2_hi < 0)

        # 3. Panel (c) Fire danger: AUPRC diff (grounded minus bare)
        # In cu_data: diff = bare - grounded.
        # Thus AUPRC diff = grounded - bare = -diff.
        # Clustered CI = [-ci_hi, -ci_lo].
        cu_meso = cu_data["Mesogeos"][m["cu_key"]]
        meso_pt = cu_meso["grounded"] - cu_meso["bare"]
        meso_lo = -cu_meso["ci_hi"]
        meso_hi = -cu_meso["ci_lo"]
        meso_excl_zero = (meso_lo > 0 and meso_hi > 0) or (meso_lo < 0 and meso_hi < 0)

        # 4. Panel (d) Tool use: accuracy diff (tool minus bare), family-clustered interval
        tl = tool_data[m["tool_key"]]["tool_minus_bare"]
        tool_pt = float(tl["d"])
        tool_lo = float(tl["ci"][0])
        tool_hi = float(tl["ci"][1])
        tool_excl_zero = (tool_lo > 0 and tool_hi > 0) or (tool_lo < 0 and tool_hi < 0)

        records.append({
            "model": m["label"],
            "figlib_gain": (gain_pt, gain_lo, gain_hi, gain_excl_zero),
            "figlib_fpr": fpr_diff,
            "alloc_v1": (v1_pt, v1_lo, v1_hi, v1_excl_zero),
            "alloc_v2": (v2_pt, v2_lo, v2_hi, v2_excl_zero),
            "meso_auprc": (meso_pt, meso_lo, meso_hi, meso_excl_zero),
            "tool_acc": (tool_pt, tool_lo, tool_hi, tool_excl_zero),
        })

    return records


def print_verification_table(records: list[dict]):
    """Print cross-check table against the paper text and tables."""
    print("=" * 110)
    print("VERIFICATION OF DATA POINTS AND INTERVALS AGAINST PAPER TABLES & TEXT")
    print("=" * 110)
    print("Panel (a) Smoke detection (FIgLib): Recall gain [95% CI], FPR change")
    print("  Model             Recall gain [95% CI]           Paper check              FPR change")
    print("  " + "-" * 105)
    for r in records:
        pt, lo, hi, sig = r["figlib_gain"]
        paper_text = "excl 0 (Coral)" if sig else "incl 0 (Gray)"
        print(f"  {r['model']:<17} {pt:+.4f} [{lo:+.4f}, {hi:+.4f}]   {paper_text:<24} {r['figlib_fpr']:+.3f}")

    print("\nPanel (b) Personnel allocation: Normalized error diff [95% CI] (v1 and v2)")
    print("  Model             Rule v1 diff [95% CI]          Rule v2 diff [95% CI]")
    print("  " + "-" * 105)
    for r in records:
        v1_pt, v1_lo, v1_hi, v1_sig = r["alloc_v1"]
        v2_pt, v2_lo, v2_hi, v2_sig = r["alloc_v2"]
        v1_s = f"{v1_pt:+.4f} [{v1_lo:+.4f}, {v1_hi:+.4f}] ({'Coral' if v1_sig else 'Gray'})"
        v2_s = f"{v2_pt:+.4f} [{v2_lo:+.4f}, {v2_hi:+.4f}] ({'Coral' if v2_sig else 'Gray'})"
        print(f"  {r['model']:<17} {v1_s:<32} {v2_s}")

    print("\nPanel (c) Fire danger (Mesogeos): AUPRC diff [95% CI]")
    print("  Model             AUPRC diff [95% CI]            Status")
    print("  " + "-" * 105)
    for r in records:
        pt, lo, hi, sig = r["meso_auprc"]
        status = "excl 0 (Coral)" if sig else "incl 0 (Gray)"
        print(f"  {r['model']:<17} {pt:+.4f} [{lo:+.4f}, {hi:+.4f}]   {status}")

    print("\nPanel (d) Fire data tool use (FPA-FOD): accuracy diff, tool minus bare [95% CI], family clusters")
    print("  Model             Accuracy diff [95% CI]         Status")
    print("  " + "-" * 105)
    for r in records:
        pt, lo, hi, sig = r["tool_acc"]
        status = "excl 0 (Coral)" if sig else "incl 0 (Gray)"
        print(f"  {r['model']:<17} {pt:+.4f} [{lo:+.4f}, {hi:+.4f}]   {status}")
    print("=" * 110)


def plot_grounding_effects(records: list[dict], out_pdf: Path, out_png: Path):
    """Draw the 4-panel CatchBench forest plot at 6.5 x 2.45 in."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "text.color": COLOR_TEXT,
        "axes.labelcolor": COLOR_TEXT,
        "xtick.color": COLOR_TEXT,
        "ytick.color": COLOR_TEXT,
    })

    fig, axes = plt.subplots(
        1, 4,
        figsize=(6.5, 2.45),
        sharey=True,
        gridspec_kw={"wspace": 0.30, "left": 0.15, "right": 0.99, "top": 0.82, "bottom": 0.17}
    )

    n_models = len(records)
    # y-coordinates: row 5 is top model (claude-opus-4.8), row 0 is bottom (Llama 4 Maverick)
    y_pos = np.arange(n_models - 1, -1, -1)
    model_names = [r["model"] for r in records]

    # Style spines and ticks helper
    def format_ax(ax, x_limits, x_ticks, x_label, title_label):
        ax.set_xlim(x_limits)
        ax.set_xticks(x_ticks)
        ax.set_xlabel(x_label, fontsize=7.2, labelpad=3)
        ax.axvline(0, color=COLOR_GRAY, linestyle="--", linewidth=0.75, zorder=1)
        # Subtle horizontal guide lines for each row
        for y in y_pos:
            ax.axhline(y, color="#F4F4F4", linestyle="-", linewidth=0.6, zorder=0)

        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color(COLOR_GRAY)
            ax.spines[spine].set_linewidth(0.8)

        ax.tick_params(axis="both", length=0, labelsize=6.8)
        ax.set_title(title_label, loc="left", fontsize=8.0, fontweight="bold", pad=8)

    # -------------------------------------------------------------
    # Panel (a): Smoke detection (FIgLib)
    # -------------------------------------------------------------
    ax_a = axes[0]
    format_ax(
        ax_a,
        x_limits=(-0.04, 0.27),
        x_ticks=[0.0, 0.1, 0.2],
        x_label="Recall gain (grd. − bare)",
        title_label="(a) Smoke detection"
    )
    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels(model_names, fontsize=7.2)

    for i, r in enumerate(records):
        y = y_pos[i]
        pt, lo, hi, sig = r["figlib_gain"]
        col = COLOR_CORAL if sig else COLOR_GRAY
        # Horizontal CI line
        ax_a.plot([lo, hi], [y, y], color=col, linewidth=1.4, zorder=3, solid_capstyle="round")
        # Recall gain point estimate
        ax_a.plot(pt, y, marker="o", markersize=4.8, color=col, zorder=4)

        # FPR change: secondary mint marker
        fpr = r["figlib_fpr"]
        ax_a.plot(
            fpr, y,
            marker="^", markersize=4.0,
            markerfacecolor=COLOR_MINT, markeredgecolor=COLOR_MINT_EDGE, markeredgewidth=0.8,
            zorder=5
        )

    # Clean legend in upper-right open quadrant (x in [0.16, 0.27], y around rows 4-5)
    ax_a.plot([], [], marker="o", markersize=4.5, color=COLOR_CORAL, linestyle="-", linewidth=1.2, label="Recall gain")
    ax_a.plot([], [], marker="^", markersize=4.0, markerfacecolor=COLOR_MINT, markeredgecolor=COLOR_MINT_EDGE,
               markeredgewidth=0.8, linestyle="None", label="FPR change")
    ax_a.legend(
        loc="upper right", frameon=False, fontsize=6.2, handletextpad=0.3, handlelength=1.2,
        borderaxespad=0.4, labelcolor=COLOR_TEXT
    )

    # -------------------------------------------------------------
    # Panel (b): Personnel allocation (ICS-209-PLUS)
    # -------------------------------------------------------------
    ax_b = axes[1]
    format_ax(
        ax_b,
        x_limits=(-0.05, 0.17),
        x_ticks=[0.0, 0.08, 0.16],
        x_label="Norm. error diff. (grd. − bare)",
        title_label="(b) Allocation"
    )

    v_offset = 0.14
    for i, r in enumerate(records):
        y = y_pos[i]
        # v1 (top)
        v1_pt, v1_lo, v1_hi, v1_sig = r["alloc_v1"]
        col_v1 = COLOR_CORAL if v1_sig else COLOR_GRAY
        y_v1 = y + v_offset
        ax_b.plot([v1_lo, v1_hi], [y_v1, y_v1], color=col_v1, linewidth=1.3, zorder=3, solid_capstyle="round")
        ax_b.plot(v1_pt, y_v1, marker="o", markersize=4.3, color=col_v1, zorder=4)

        # v2 (bottom)
        v2_pt, v2_lo, v2_hi, v2_sig = r["alloc_v2"]
        col_v2 = COLOR_CORAL if v2_sig else COLOR_GRAY
        y_v2 = y - v_offset
        ax_b.plot([v2_lo, v2_hi], [y_v2, y_v2], color=col_v2, linewidth=1.3, zorder=3, solid_capstyle="round")
        ax_b.plot(v2_pt, y_v2, marker="s", markersize=3.9, color=col_v2, zorder=4)

    # Direct labels for v1 and v2 beside top model (claude-opus-4.8)
    top_y = y_pos[0]
    ax_b.text(
        0.046, top_y + v_offset, "v1",
        fontsize=6.2, verticalalignment="center", color=COLOR_SUBTITLE, fontweight="bold"
    )
    ax_b.text(
        0.046, top_y - v_offset, "v2",
        fontsize=6.2, verticalalignment="center", color=COLOR_SUBTITLE, fontweight="bold"
    )

    # -------------------------------------------------------------
    # Panel (c): Fire danger (Mesogeos)
    # -------------------------------------------------------------
    ax_c = axes[2]
    format_ax(
        ax_c,
        x_limits=(-0.11, 0.12),
        x_ticks=[-0.08, 0.0, 0.08],
        x_label="AUPRC diff. (grd. − bare)",
        title_label="(c) Fire danger"
    )

    for i, r in enumerate(records):
        y = y_pos[i]
        pt, lo, hi, sig = r["meso_auprc"]
        col = COLOR_CORAL if sig else COLOR_GRAY
        ax_c.plot([lo, hi], [y, y], color=col, linewidth=1.4, zorder=3, solid_capstyle="round")
        ax_c.plot(pt, y, marker="o", markersize=4.8, color=col, zorder=4)

    # -------------------------------------------------------------
    # Panel (d): Fire data tool use (FPA-FOD)
    # -------------------------------------------------------------
    ax_d = axes[3]
    format_ax(
        ax_d,
        x_limits=(-0.06, 1.08),
        x_ticks=[0.0, 0.5, 1.0],
        x_label="Accuracy diff. (tool − bare)",
        title_label="(d) Tool use"
    )

    for i, r in enumerate(records):
        y = y_pos[i]
        pt, lo, hi, sig = r["tool_acc"]
        col = COLOR_CORAL if sig else COLOR_GRAY
        ax_d.plot([lo, hi], [y, y], color=col, linewidth=1.4, zorder=3, solid_capstyle="round")
        ax_d.plot(pt, y, marker="o", markersize=4.8, color=col, zorder=4)

    # Adjust vertical limits
    ax_a.set_ylim(-0.55, n_models - 0.45)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    fig.savefig(out_png, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated {out_pdf} and {out_png}")


def main():
    parser = argparse.ArgumentParser(description="Generate grounding effects figure.")
    parser.add_argument("--repo", type=str, default=str(Path(__file__).resolve().parent.parent),
                        help="Path to AI4Fire repository root.")
    parser.add_argument("--out-pdf", type=str, default=None,
                        help="Path for output PDF file.")
    parser.add_argument("--out-png", type=str, default=None,
                        help="Path for output PNG file.")
    parser.add_argument("--rerun-cluster", action="store_true",
                        help="Force re-running cluster_uncertainty.py instead of using cached output.")
    args = parser.parse_args()

    repo_root = Path(args.repo).resolve()
    out_pdf = Path(args.out_pdf).resolve() if args.out_pdf else repo_root / "figures" / "grounding_effects.pdf"
    out_png = Path(args.out_png).resolve() if args.out_png else repo_root / "figures" / "grounding_effects.png"

    records = gather_all_data(repo_root, force_rerun=args.rerun_cluster)
    print_verification_table(records)
    plot_grounding_effects(records, out_pdf, out_png)


if __name__ == "__main__":
    main()
