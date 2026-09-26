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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import models
from models import add_model_args, resolve_models

# CatchBench Palette
COLOR_CORAL = "#ED8D5A"
COLOR_MINT = "#BFDFD2"
COLOR_MINT_EDGE = "#8FB7A6"
COLOR_GRAY = "#999999"
COLOR_LIGHT_GRAY = "#E6E6E6"
COLOR_TEXT = "#1A1A1A"
COLOR_SUBTITLE = "#666666"


def load_cluster_uncertainty_output(repo_root: Path, force_rerun: bool = False, tier: str = "core") -> str:
    """Read cached cluster uncertainty stdout or run cluster_uncertainty.py.

    The core tier runs under the manifest (the 36 reported files) and caches to cluster-six.stdout.txt; any
    other tier runs over every response file on disk, with unpaired variant arms allowed, and caches to
    cluster-<tier>.stdout.txt, so the six-model cache is never overwritten by a sweep render.
    """
    core = tier == "core"
    cached_path = repo_root / "analysis" / ("cluster-six.stdout.txt" if core else f"cluster-{tier}.stdout.txt")
    if not force_rerun and cached_path.exists():
        return cached_path.read_text(encoding="utf-8")

    script_path = repo_root / "analysis" / "cluster_uncertainty.py"
    cmd = [sys.executable, str(script_path), "--max-skew-min", "100000"]
    cmd += ["--manifest", "manifest-v1.json"] if core else ["--allow-unpaired"]
    proc = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True, check=True)
    if not core:
        cached_path.write_text(proc.stdout, encoding="utf-8")
    return proc.stdout


def parse_cluster_uncertainty(text: str) -> dict[str, dict[str, dict[str, float]]]:
    """Parse bare, grounded, diff, ci_lo, ci_hi from cluster_uncertainty stdout."""
    row_pattern = re.compile(
        r"^\s*(?P<arm>[a-zA-Z0-9_\.\-]+)\s+(?P<bare_n>\d+)\s+(?P<grd_n>\d+)\s+(?P<pair_n>\d+)\s+"
        r"(?P<n_clust>\d+)\s+(?P<bare>[\d\.\-]+)\s+(?P<grounded>[\d\.\-]+)\s+(?P<diff>[\d\.\-]+)\s+"
        r"\[\s*(?P<ci_lo>[\d\.\-]+)\s*,\s*(?P<ci_hi>[\d\.\-]+)\s*\]",  # a positive bound is padded with a space
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
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_figlib_paired(repo_root: Path) -> dict:
    path = repo_root / "analysis" / "figlib-paired.json"
    if not path.exists():
        return {}
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
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f).get("models", {})


def require_coverage(name: str, have, want, command: str) -> None:
    """Stop when a shared analysis JSON predates the selected tier.

    analysis/figlib-paired.json and analysis/tooluse_paired.json are written by scripts whose own default is the
    core tier, so rendering the sweep figure after a default analysis run would silently drop the added models'
    points while still labelling their rows. Name the command that refills the file instead.
    """
    missing = [m for m in want if m not in have]
    if missing:
        shown = ", ".join(missing[:6]) + (", ..." if len(missing) > 6 else "")
        raise SystemExit("%s is missing %d of %d selected models (%s).\nRegenerate it first: %s"
                         % (name, len(missing), len(want), shown, command))


def find_cu_entry(task_dict: dict, stem: str):
    """cluster_uncertainty.py prints arm names cut at 30 characters; take the exact name when it is there,
    otherwise the truncated name that shares the longest prefix with the stem (so bedrock_zai.glm-4.7 never
    stands in for bedrock_zai.glm-4.7-flash)."""
    if stem in task_dict:
        return task_dict[stem]
    best, best_len = None, 0
    for k, v in task_dict.items():
        if stem.startswith(k) or k.startswith(stem[:28]):
            n = len(os.path.commonprefix([k, stem]))
            if n > best_len:
                best, best_len = v, n
    return best


def gather_all_data(repo_root: Path, selected_models, force_rerun: bool = False, tier: str = "core"):
    cu_text = load_cluster_uncertainty_output(repo_root, force_rerun, tier)
    cu_data = parse_cluster_uncertainty(cu_text)
    v2_data = load_retrieval_v2(repo_root)
    fig_data = load_figlib_paired(repo_root)
    tool_data = load_tooluse_paired(repo_root)
    # A model is expected in a shared JSON only when its response files for that task exist; a tier whose models
    # never ran the task (the text-only sweep has no smoke files) is not a stale cache.
    def ran(m, task, cond):
        return (repo_root / f"task-{task}" / f"responses-{m.stem}-{cond}.jsonl").exists()

    require_coverage("analysis/figlib-paired.json", fig_data,
                     [m.stem for m in selected_models if ran(m, "figlib", "grounded")],
                     "python analysis/figlib_paired.py --tier %s" % tier)
    require_coverage("analysis/tooluse_paired.json", tool_data,
                     [m.label for m in selected_models if ran(m, "tooluse", "tool")],
                     "python analysis/tooluse_paired.py --tier %s" % tier)

    records = []
    for m in selected_models:
        stem = m.stem
        label = m.label

        # 1. Panel (a) Smoke detection: recall gain
        cu_fig = find_cu_entry(cu_data.get("FIgLib", {}), stem)
        if cu_fig is not None:
            gain_pt = cu_fig["grounded"] - cu_fig["bare"]
            gain_lo = -cu_fig["ci_hi"]
            gain_hi = -cu_fig["ci_lo"]
            gain_excl_zero = (gain_lo > 0 and gain_hi > 0) or (gain_lo < 0 and gain_hi < 0)
            figlib_gain = (gain_pt, gain_lo, gain_hi, gain_excl_zero)
        else:
            figlib_gain = None

        fig_entry = fig_data.get(stem, {})
        if "bare" in fig_entry and "grounded" in fig_entry:
            fpr_diff = fig_entry["grounded"] - fig_entry["bare"]
        else:
            fpr_diff = None

        # 2. Panel (b) Allocation: v1 and v2
        cu_alloc = find_cu_entry(cu_data.get("allocation", {}), stem)
        if cu_alloc is not None:
            v1_pt = cu_alloc["grounded"] - cu_alloc["bare"]
            v1_lo = -cu_alloc["ci_hi"]
            v1_hi = -cu_alloc["ci_lo"]
            v1_excl_zero = (v1_lo > 0 and v1_hi > 0) or (v1_lo < 0 and v1_hi < 0)
            alloc_v1 = (v1_pt, v1_lo, v1_hi, v1_excl_zero)
        else:
            alloc_v1 = None

        v2_m = v2_data.get(stem)
        if v2_m and "d_v2_bare" in v2_m:
            v2_pt = float(v2_m["d_v2_bare"])
            v2_lo = float(v2_m["d_v2_bare_ci"][0])
            v2_hi = float(v2_m["d_v2_bare_ci"][1])
            v2_excl_zero = (v2_lo > 0 and v2_hi > 0) or (v2_lo < 0 and v2_hi < 0)
            alloc_v2 = (v2_pt, v2_lo, v2_hi, v2_excl_zero)
        else:
            alloc_v2 = None

        # 3. Panel (c) Fire danger: AUPRC diff
        cu_meso = find_cu_entry(cu_data.get("Mesogeos", {}), stem)
        if cu_meso is not None:
            meso_pt = cu_meso["grounded"] - cu_meso["bare"]
            meso_lo = -cu_meso["ci_hi"]
            meso_hi = -cu_meso["ci_lo"]
            meso_excl_zero = (meso_lo > 0 and meso_hi > 0) or (meso_lo < 0 and meso_hi < 0)
            meso_auprc = (meso_pt, meso_lo, meso_hi, meso_excl_zero)
        else:
            meso_auprc = None

        # 4. Panel (d) Tool use: accuracy diff
        tl_m = tool_data.get(label) or tool_data.get(stem)
        if tl_m and "tool_minus_bare" in tl_m:
            tl = tl_m["tool_minus_bare"]
            tool_pt = float(tl["d"])
            tool_lo = float(tl["ci"][0])
            tool_hi = float(tl["ci"][1])
            tool_excl_zero = (tool_lo > 0 and tool_hi > 0) or (tool_lo < 0 and tool_hi < 0)
            tool_acc = (tool_pt, tool_lo, tool_hi, tool_excl_zero)
        else:
            tool_acc = None

        # Include model if it has data in at least one task
        if any(x is not None for x in (figlib_gain, alloc_v1, meso_auprc, tool_acc)):
            records.append({
                "model": label,
                "stem": stem,
                "figlib_gain": figlib_gain,
                "figlib_fpr": fpr_diff,
                "alloc_v1": alloc_v1,
                "alloc_v2": alloc_v2,
                "meso_auprc": meso_auprc,
                "tool_acc": tool_acc,
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
        if r["figlib_gain"] is not None:
            pt, lo, hi, sig = r["figlib_gain"]
            paper_text = "excl 0 (Coral)" if sig else "incl 0 (Gray)"
            fpr_str = f"{r['figlib_fpr']:+.3f}" if r["figlib_fpr"] is not None else "n/a"
            print(f"  {r['model']:<17} {pt:+.4f} [{lo:+.4f}, {hi:+.4f}]   {paper_text:<24} {fpr_str}")

    print("\nPanel (b) Personnel allocation: Normalized error diff [95% CI] (v1 and v2)")
    print("  Model             Rule v1 diff [95% CI]          Rule v2 diff [95% CI]")
    print("  " + "-" * 105)
    for r in records:
        v1_s = "n/a"
        if r["alloc_v1"] is not None:
            v1_pt, v1_lo, v1_hi, v1_sig = r["alloc_v1"]
            v1_s = f"{v1_pt:+.4f} [{v1_lo:+.4f}, {v1_hi:+.4f}] ({'Coral' if v1_sig else 'Gray'})"
        v2_s = "n/a"
        if r["alloc_v2"] is not None:
            v2_pt, v2_lo, v2_hi, v2_sig = r["alloc_v2"]
            v2_s = f"{v2_pt:+.4f} [{v2_lo:+.4f}, {v2_hi:+.4f}] ({'Coral' if v2_sig else 'Gray'})"
        print(f"  {r['model']:<17} {v1_s:<32} {v2_s}")

    print("\nPanel (c) Fire danger (Mesogeos): AUPRC diff [95% CI]")
    print("  Model             AUPRC diff [95% CI]            Status")
    print("  " + "-" * 105)
    for r in records:
        if r["meso_auprc"] is not None:
            pt, lo, hi, sig = r["meso_auprc"]
            status = "excl 0 (Coral)" if sig else "incl 0 (Gray)"
            print(f"  {r['model']:<17} {pt:+.4f} [{lo:+.4f}, {hi:+.4f}]   {status}")

    print("\nPanel (d) Fire data tool use (FPA-FOD): accuracy diff, tool minus bare [95% CI], family clusters")
    print("  Model             Accuracy diff [95% CI]         Status")
    print("  " + "-" * 105)
    for r in records:
        if r["tool_acc"] is not None:
            pt, lo, hi, sig = r["tool_acc"]
            status = "excl 0 (Coral)" if sig else "incl 0 (Gray)"
            print(f"  {r['model']:<17} {pt:+.4f} [{lo:+.4f}, {hi:+.4f}]   {status}")
    print("=" * 110)


def _interval(ax, y, pt, lo, hi, col, xlim, marker="o", ms=4.8, lw=1.4, label_dy=0.34):
    """One point with its interval. A point beyond the shared axis (Nova 2 Lite in the sweep) is clipped at the
    edge, marked with a triangle, and labelled with its value, so the row stays readable without rescaling the
    six-model axis."""
    x0, x1 = xlim
    if pt > x1 or pt < x0:
        edge = x1 if pt > x1 else x0
        ax.plot([max(lo, x0), min(hi, x1)], [y, y], color=col, linewidth=lw, zorder=3, solid_capstyle="butt")
        ax.plot(edge, y, marker=">" if pt > x1 else "<", markersize=ms, color=col, zorder=4, clip_on=False)
        ax.text(edge - 0.02 * (x1 - x0) if pt > x1 else edge + 0.02 * (x1 - x0), y + label_dy, f"{pt:+.2f}",
                fontsize=5.6, ha="right" if pt > x1 else "left", va="center", color=col)
    else:
        ax.plot([lo, hi], [y, y], color=col, linewidth=lw, zorder=3, solid_capstyle="round")
        ax.plot(pt, y, marker=marker, markersize=ms, color=col, zorder=4)


def plot_grounding_effects(records: list[dict], out_pdf: Path | None, out_png: Path):
    """Draw the 4-panel forest plot."""
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

    n_models = len(records)
    fig_height = 2.45 if n_models <= 6 else max(2.45, 0.35 * n_models + 0.6)
    left_margin = 0.15 if n_models <= 6 else 0.18

    fig, axes = plt.subplots(
        1, 4,
        figsize=(6.5, fig_height),
        sharey=True,
        gridspec_kw={"wspace": 0.30, "left": left_margin, "right": 0.99, "top": 0.82 if n_models <= 6 else 0.90, "bottom": 0.17 if n_models <= 6 else 0.10}
    )

    y_pos = np.arange(n_models - 1, -1, -1)
    model_names = [r["model"] for r in records]

    def format_ax(ax, x_limits, x_ticks, x_label, title_label):
        ax.set_xlim(x_limits)
        ax.set_xticks(x_ticks)
        ax.set_xlabel(x_label, fontsize=7.2, labelpad=3)
        ax.axvline(0, color=COLOR_GRAY, linestyle="--", linewidth=0.75, zorder=1)
        for y in y_pos:
            ax.axhline(y, color="#F4F4F4", linestyle="-", linewidth=0.6, zorder=0)

        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color(COLOR_GRAY)
            ax.spines[spine].set_linewidth(0.8)

        ax.tick_params(axis="both", length=0, labelsize=6.8)
        ax.set_title(title_label, loc="left", fontsize=8.0, fontweight="bold", pad=8)

    # Panel (a): Smoke detection
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
        if r["figlib_gain"] is not None:
            pt, lo, hi, sig = r["figlib_gain"]
            col = COLOR_CORAL if sig else COLOR_GRAY
            _interval(ax_a, y, pt, lo, hi, col, (-0.04, 0.27))

        if r["figlib_fpr"] is not None:
            fpr = r["figlib_fpr"]
            ax_a.plot(
                fpr, y,
                marker="^", markersize=4.0,
                markerfacecolor=COLOR_MINT, markeredgecolor=COLOR_MINT_EDGE, markeredgewidth=0.8,
                zorder=5
            )

    y_anchor = (y_pos[0] - 0.55) if n_models > 0 else 0.0
    ax_a.plot([], [], marker="o", markersize=4.5, color=COLOR_CORAL, linestyle="-", linewidth=1.2, label="Recall gain")
    ax_a.plot([], [], marker="^", markersize=4.0, markerfacecolor=COLOR_MINT, markeredgecolor=COLOR_MINT_EDGE,
               markeredgewidth=0.8, linestyle="None", label="FPR change")
    ax_a.legend(
        loc="upper right", bbox_to_anchor=(0.265, y_anchor), bbox_transform=ax_a.transData,
        frameon=False, fontsize=6.2, handletextpad=0.25, handlelength=1.0,
        borderaxespad=0.0, borderpad=0.0, labelcolor=COLOR_TEXT
    )

    # Panel (b): Personnel allocation
    ax_b = axes[1]
    format_ax(
        ax_b,
        x_limits=(-0.05, 0.17),
        x_ticks=[0.0, 0.08, 0.16],
        x_label="Norm. error diff. (grd. − bare)",
        title_label="(b) Allocation"
    )

    v_offset = 0.14
    x_max_b = 0.17
    for i, r in enumerate(records):
        y = y_pos[i]
        if r["alloc_v1"] is not None:
            v1_pt, v1_lo, v1_hi, v1_sig = r["alloc_v1"]
            col_v1 = COLOR_CORAL if v1_sig else COLOR_GRAY
            y_v1 = y + v_offset
            _interval(ax_b, y_v1, v1_pt, v1_lo, v1_hi, col_v1, (-0.05, x_max_b), ms=4.3, lw=1.3)

        if r["alloc_v2"] is not None:
            v2_pt, v2_lo, v2_hi, v2_sig = r["alloc_v2"]
            col_v2 = COLOR_CORAL if v2_sig else COLOR_GRAY
            y_v2 = y - v_offset
            ax_b.plot([v2_lo, v2_hi], [y_v2, y_v2], color=col_v2, linewidth=1.3, zorder=3, solid_capstyle="round")
            ax_b.plot(v2_pt, y_v2, marker="s", markersize=3.9, color=col_v2, zorder=4)

    ax_b.plot([], [], marker="o", markersize=4.3, color=COLOR_GRAY, linestyle="-", linewidth=1.3, label="v1")
    ax_b.plot([], [], marker="s", markersize=3.9, color=COLOR_GRAY, linestyle="-", linewidth=1.3, label="v2")
    ax_b.legend(
        loc="upper right", bbox_to_anchor=(0.165, y_anchor), bbox_transform=ax_b.transData,
        frameon=False, fontsize=6.2, handletextpad=0.25, handlelength=1.0,
        borderaxespad=0.0, borderpad=0.0, labelcolor=COLOR_TEXT
    )

    # Panel (c): Fire danger
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
        if r["meso_auprc"] is not None:
            pt, lo, hi, sig = r["meso_auprc"]
            col = COLOR_CORAL if sig else COLOR_GRAY
            ax_c.plot([lo, hi], [y, y], color=col, linewidth=1.4, zorder=3, solid_capstyle="round")
            ax_c.plot(pt, y, marker="o", markersize=4.8, color=col, zorder=4)

    # Panel (d): Tool use
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
        if r["tool_acc"] is not None:
            pt, lo, hi, sig = r["tool_acc"]
            col = COLOR_CORAL if sig else COLOR_GRAY
            ax_d.plot([lo, hi], [y, y], color=col, linewidth=1.4, zorder=3, solid_capstyle="round")
            ax_d.plot(pt, y, marker="o", markersize=4.8, color=col, zorder=4)

    ax_a.set_ylim(-0.55, n_models - 0.45)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    if out_pdf is not None:
        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
        print(f"Generated {out_pdf}")
    fig.savefig(out_png, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated {out_png}")


def main():
    parser = argparse.ArgumentParser(description="Generate grounding effects figure.")
    parser.add_argument("--repo", type=str, default=str(ROOT),
                        help="Path to repository root.")
    parser.add_argument("--out-pdf", type=Path, default=None,
                        help="Path for output PDF file.")
    parser.add_argument("--out-png", type=Path, default=None,
                        help="Path for output PNG file.")
    parser.add_argument("--rerun-cluster", action="store_true",
                        help="Force re-running cluster_uncertainty.py instead of using cached output.")
    add_model_args(parser, default_tier="core")
    args = parser.parse_args()

    repo_root = Path(args.repo).resolve()
    figs_dir = repo_root / "figures"
    # The default filename follows the tier, so a sweep render cannot overwrite the six-model figure the paper's
    # Figure 3 uses. --out-pdf and --out-png still override it.
    stem = {"core": "grounding_effects", "all": "grounding_effects_sweep"}.get(args.tier, f"grounding_effects_{args.tier}")
    if args.out_pdf:
        out_pdf = args.out_pdf
    elif args.out_png:
        out_pdf = None
    else:
        out_pdf = figs_dir / f"{stem}.pdf"
    out_png = args.out_png or (figs_dir / f"{stem}.png")

    selected_models = resolve_models(args, default_tier="core")

    records = gather_all_data(repo_root, selected_models, force_rerun=args.rerun_cluster, tier=args.tier)
    print_verification_table(records)
    plot_grounding_effects(records, out_pdf, out_png)


if __name__ == "__main__":
    main()
