"""One row per model across the five tasks, for the model sweep of 2026-09-18.

Reads the registry (models.py) and the response files, and scores every model with the same definitions the
paper's main tables use: allocation as table_rows_allocation.py (normalized error over parsed rows), fire danger
with run_mesogeos.score, smoke detection with analysis/figlib_paired.score on the items both arms answered, tool use
and aerial question answering as the share of correct answers over all items (an unparsed answer counts wrong, as
in the paired scorers). Writes analysis/model_sweep.json, prints the LaTeX rows of the two sweep tables (the
full-capability models on five tasks; the text-only models on three), and prints the tallies the paper quotes.

    python analysis/model_sweep.py            # every model in the registry that has files
    python analysis/model_sweep.py --tier all # the sixteen full-capability models only
"""
import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
from models import add_model_args, resolve_models  # noqa: E402
import run_mesogeos as rm  # noqa: E402
from figlib_paired import score as figlib_score  # noqa: E402

OUT = ROOT / "analysis" / "model_sweep.json"
MAJORITY = 0.627  # WildFireVQA majority baseline, 256 of 408 items (analysis/wildfirevqa_paired.py)


def rows_of(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def allocation(rows):
    ok = [r for r in rows if r.get("prediction") is not None]
    y = np.array([r["target"] for r in ok]); p = np.array([r["prediction"] for r in ok])
    b = np.array([r["persistence"] for r in ok]); s = np.array([r["fire_mean"] for r in ok])
    err = np.abs(y - p) / s
    berr = np.abs(y - b) / s
    return {"n": len(rows), "parsed": len(ok), "truncated": truncated(rows),
            "mae": float(np.mean(np.abs(y - p))), "nmae": float(np.mean(err)),
            "persistence_nmae": float(np.mean(berr)), "beats_persistence": float(np.mean(err < berr)),
            "within_25pct": float(np.mean(np.abs(p - y) <= 0.25 * y)),
            "persistence_copies": int(np.sum(p == b)),  # items whose prediction equals the previous day's count
            "nmae_without_worst": float(np.mean(np.sort(err)[:-1])) if len(err) > 1 else None}


def mesogeos(rows, label):
    s = rm.score(rows, label)
    return {"n": s["items"], "parsed": s["parsed"], "truncated": truncated(rows), "auprc": s["auprc"], "f1_fire": s["f1_fire"],
            "call_rate": s["positive_rate_called"],
            "omitted": sum(1 for r in rows if r.get("call") is None and r.get("probability") is not None)}


def figlib(bare_rows, grounded_rows):
    bare = {r["item_id"]: r for r in bare_rows}
    grounded = {r["item_id"]: r for r in grounded_rows}
    ids = sorted(i for i in bare if i in grounded and bare[i].get("prediction") is not None
                 and grounded[i].get("prediction") is not None)
    sb, sg = figlib_score(bare, ids), figlib_score(grounded, ids)
    return {"paired": len(ids), "truncated": truncated(bare_rows) + truncated(grounded_rows), "bare": {"recall": sb["recall_smoke"], "fpr": sb["fpr"], "accuracy": sb["accuracy"]},
            "grounded": {"recall": sg["recall_smoke"], "fpr": sg["fpr"], "accuracy": sg["accuracy"]},
            "found_delta": sg["counts"]["smoke_true_positive"] - sb["counts"]["smoke_true_positive"],
            "smoke_frames": sb["counts"]["smoke_frames"]}


def share_correct(rows):
    return {"n": len(rows), "accuracy": float(np.mean([bool(r.get("correct")) for r in rows])),
            "errors": sum(1 for r in rows if r.get("error")),
            "unparsed": sum(1 for r in rows if r.get("prediction") is None and not r.get("error")),
            "truncated": truncated(rows)}


def truncated(rows):
    """Rows the provider cut at the output cap. A truncated row usually carries no answer and scores wrong."""
    return sum(1 for r in rows if (r.get("usage") or {}).get("stop_reason") in ("max_tokens", "length"))


EXPECTED = {"allocation": (300, 300), "mesogeos": (386, 386), "figlib": (224, 196), "tooluse": (156, 156), "wildfirevqa": (408, 408)}
ARMS = {"allocation": ("bare", "grounded"), "mesogeos": ("bare", "grounded"), "figlib": ("bare", "grounded"),
        "tooluse": ("bare", "tool"), "wildfirevqa": ("bare", "grounded")}


def complete(m, task):
    """Both arm files exist with the full row count; a pilot or a run still in progress is skipped with a note."""
    rows = []
    for arm, n_exp in zip(ARMS[task], EXPECTED[task]):
        path = ROOT / f"task-{task}" / f"responses-{m.stem}-{arm}.jsonl"
        if not path.exists():
            return None
        r = rows_of(path)
        if len(r) != n_exp:
            print(f"% skipping {m.label} {task}/{arm}: {len(r)} of {n_exp} rows", file=sys.stderr)
            return None
        rows.append(r)
    return rows


def score_model(m):
    d = {"label": m.label, "stem": m.stem, "vendor": m.vendor, "weights": m.weights, "tier": m.tier,
         "max_out": m.max_out, "tools": m.tools, "tasks": {}}
    rows = complete(m, "allocation")
    if rows:
        d["tasks"]["allocation"] = {"bare": allocation(rows[0]), "grounded": allocation(rows[1])}
    rows = complete(m, "mesogeos")
    if rows:
        d["tasks"]["mesogeos"] = {"bare": mesogeos(rows[0], f"{m.stem}-bare"), "grounded": mesogeos(rows[1], f"{m.stem}-grounded")}
    rows = complete(m, "figlib")
    if rows:
        d["tasks"]["figlib"] = figlib(rows[0], rows[1])
    rows = complete(m, "tooluse")
    if rows:
        d["tasks"]["tooluse"] = {"bare": share_correct(rows[0]), "tool": share_correct(rows[1])}
        d["tasks"]["tooluse"]["tool"]["mean_calls"] = float(np.mean([r.get("n_tool_calls", 0) for r in rows[1]]))
        d["tasks"]["tooluse"]["tool"]["no_call_items"] = sum(1 for r in rows[1] if not r.get("n_tool_calls"))
    rows = complete(m, "wildfirevqa")
    if rows:
        d["tasks"]["wildfirevqa"] = {"bare": share_correct(rows[0]), "grounded": share_correct(rows[1])}
    return d


def f3(x):
    return "--" if x is None else f"{x:.3f}"


def tex_name(label):
    return label.replace("_", r"\_")


def print_tables(results):
    full = [r for r in results if r["tier"] in ("core", "added")]
    text = [r for r in results if r["tier"] == "text"]
    print("% --- full-capability models, five tasks, the body table (bare / grounded; tool use bare / tool) ---")
    print("% Model & alloc nMAE b & g & smoke recall b & g & AUPRC b & g & tool b & tool & aerial b & g")
    for r in full:
        T = r["tasks"]
        a, m, f, u, w = T.get("allocation"), T.get("mesogeos"), T.get("figlib"), T.get("tooluse"), T.get("wildfirevqa")
        cells = [tex_name(r["label"])]
        cells += [f3(a["bare"]["nmae"]), f3(a["grounded"]["nmae"])] if a else ["--", "--"]
        cells += [f3(f["bare"]["recall"]), f3(f["grounded"]["recall"])] if f else ["--", "--"]
        cells += [f3(m["bare"]["auprc"]), f3(m["grounded"]["auprc"])] if m else ["--", "--"]
        cells += [f3(u["bare"]["accuracy"]), f3(u["tool"]["accuracy"])] if u else ["--", "--"]
        cells += [f3(w["bare"]["accuracy"]), f3(w["grounded"]["accuracy"])] if w else ["--", "--"]
        print(" & ".join(cells) + r" \\")
    print("% --- full-capability models, five tasks, the appendix table with false-positive and fire-call rates ---")
    print("% Model & alloc nMAE b & g & smoke recall b & g & smoke FPR b & g & AUPRC b & g & call rate b & g & tool b & tool & aerial b & g")
    for r in full:
        T = r["tasks"]
        a, m, f, u, w = T.get("allocation"), T.get("mesogeos"), T.get("figlib"), T.get("tooluse"), T.get("wildfirevqa")
        cells = [tex_name(r["label"])]
        cells += [f3(a["bare"]["nmae"]), f3(a["grounded"]["nmae"])] if a else ["--", "--"]
        cells += [f3(f["bare"]["recall"]), f3(f["grounded"]["recall"]), f3(f["bare"]["fpr"]), f3(f["grounded"]["fpr"])] if f else ["--"] * 4
        cells += [f3(m["bare"]["auprc"]), f3(m["grounded"]["auprc"]), f3(m["bare"]["call_rate"]), f3(m["grounded"]["call_rate"])] if m else ["--"] * 4
        cells += [f3(u["bare"]["accuracy"]), f3(u["tool"]["accuracy"])] if u else ["--", "--"]
        cells += [f3(w["bare"]["accuracy"]), f3(w["grounded"]["accuracy"])] if w else ["--", "--"]
        print(" & ".join(cells) + r" \\")
    print("% --- text-only models, three tasks ---")
    print("% Model & cap & alloc nMAE b & g & AUPRC b & g & call rate b & g & tool b & tool")
    for r in text:
        T = r["tasks"]
        a, m, u = T.get("allocation"), T.get("mesogeos"), T.get("tooluse")
        cells = [tex_name(r["label"]), f"{r['max_out']:,}"]
        cells += [f3(a["bare"]["nmae"]), f3(a["grounded"]["nmae"])] if a else ["--", "--"]
        cells += [f3(m["bare"]["auprc"]), f3(m["grounded"]["auprc"]), f3(m["bare"]["call_rate"]), f3(m["grounded"]["call_rate"])] if m else ["--"] * 4
        cells += [f3(u["bare"]["accuracy"]), f3(u["tool"]["accuracy"])] if u else ["--", "--"]
        print(" & ".join(cells) + r" \\")


def tallies(results):
    out = {}
    have = lambda task: [r for r in results if task in r["tasks"]]  # noqa: E731
    a = have("allocation")
    out["allocation"] = {
        "models": len(a),
        "grounded_raises_nmae": sum(1 for r in a if r["tasks"]["allocation"]["grounded"]["nmae"] > r["tasks"]["allocation"]["bare"]["nmae"]),
        "bare_beats_persistence": sum(1 for r in a if r["tasks"]["allocation"]["bare"]["nmae"] < r["tasks"]["allocation"]["bare"]["persistence_nmae"]),
        "grounded_beats_persistence": sum(1 for r in a if r["tasks"]["allocation"]["grounded"]["nmae"] < r["tasks"]["allocation"]["grounded"]["persistence_nmae"]),
        "bare_nmae_range": [min(r["tasks"]["allocation"]["bare"]["nmae"] for r in a), max(r["tasks"]["allocation"]["bare"]["nmae"] for r in a)],
        "persistence_nmae": a[0]["tasks"]["allocation"]["bare"]["persistence_nmae"] if a else None,
    }
    f = have("figlib")
    # a tier can leave a task with no model (--tier text has no image task), and an empty range has no min
    out["figlib"] = {"models": 0} if not f else {
        "models": len(f),
        "grounded_raises_recall": sum(1 for r in f if r["tasks"]["figlib"]["grounded"]["recall"] > r["tasks"]["figlib"]["bare"]["recall"]),
        "grounded_lowers_recall": sum(1 for r in f if r["tasks"]["figlib"]["grounded"]["recall"] < r["tasks"]["figlib"]["bare"]["recall"]),
        "grounded_raises_fpr": sum(1 for r in f if r["tasks"]["figlib"]["grounded"]["fpr"] > r["tasks"]["figlib"]["bare"]["fpr"]),
        "found_delta_range": [min(r["tasks"]["figlib"]["found_delta"] for r in f), max(r["tasks"]["figlib"]["found_delta"] for r in f)],
        "bare_recall_range": [min(r["tasks"]["figlib"]["bare"]["recall"] for r in f), max(r["tasks"]["figlib"]["bare"]["recall"] for r in f)],
    }
    m = have("mesogeos")
    # a tier can leave a task with no model (--tier text has no image task), and an empty range has no min
    out["mesogeos"] = {"models": 0} if not m else {
        "models": len(m),
        "bare_auprc_range": [min(r["tasks"]["mesogeos"]["bare"]["auprc"] for r in m), max(r["tasks"]["mesogeos"]["bare"]["auprc"] for r in m)],
        "bare_above_temperature_rule_0.654": sum(1 for r in m if r["tasks"]["mesogeos"]["bare"]["auprc"] > 0.654),
        "bare_call_rate_below_0.10": sum(1 for r in m if r["tasks"]["mesogeos"]["bare"]["call_rate"] < 0.10),
        "bare_call_rate_range": [min(r["tasks"]["mesogeos"]["bare"]["call_rate"] for r in m), max(r["tasks"]["mesogeos"]["bare"]["call_rate"] for r in m)],
        "grounded_raises_auprc": sum(1 for r in m if r["tasks"]["mesogeos"]["grounded"]["auprc"] > r["tasks"]["mesogeos"]["bare"]["auprc"]),
    }
    u = have("tooluse")
    # a tier can leave a task with no model (--tier text has no image task), and an empty range has no min
    out["tooluse"] = {"models": 0} if not u else {
        "models": len(u),
        "tool_raises_accuracy": sum(1 for r in u if r["tasks"]["tooluse"]["tool"]["accuracy"] > r["tasks"]["tooluse"]["bare"]["accuracy"]),
        "bare_accuracy_range": [min(r["tasks"]["tooluse"]["bare"]["accuracy"] for r in u), max(r["tasks"]["tooluse"]["bare"]["accuracy"] for r in u)],
        "tool_accuracy_range": [min(r["tasks"]["tooluse"]["tool"]["accuracy"] for r in u), max(r["tasks"]["tooluse"]["tool"]["accuracy"] for r in u)],
        "tool_at_or_above_0.88": sum(1 for r in u if r["tasks"]["tooluse"]["tool"]["accuracy"] >= 0.88),
        "tool_below_bare": [r["label"] for r in u if r["tasks"]["tooluse"]["tool"]["accuracy"] <= r["tasks"]["tooluse"]["bare"]["accuracy"]],
        "tool_below_0.5": [r["label"] for r in u if r["tasks"]["tooluse"]["tool"]["accuracy"] < 0.5],
    }
    w = have("wildfirevqa")
    # a tier can leave a task with no model (--tier text has no image task), and an empty range has no min
    out["wildfirevqa"] = {"models": 0} if not w else {
        "models": len(w),
        "bare_accuracy_range": [min(r["tasks"]["wildfirevqa"]["bare"]["accuracy"] for r in w), max(r["tasks"]["wildfirevqa"]["bare"]["accuracy"] for r in w)],
        "grounded_accuracy_range": [min(r["tasks"]["wildfirevqa"]["grounded"]["accuracy"] for r in w), max(r["tasks"]["wildfirevqa"]["grounded"]["accuracy"] for r in w)],
        "bare_above_majority": sum(1 for r in w if r["tasks"]["wildfirevqa"]["bare"]["accuracy"] > MAJORITY),
        "grounded_above_majority": sum(1 for r in w if r["tasks"]["wildfirevqa"]["grounded"]["accuracy"] > MAJORITY),
        "grounded_raises_accuracy": sum(1 for r in w if r["tasks"]["wildfirevqa"]["grounded"]["accuracy"] > r["tasks"]["wildfirevqa"]["bare"]["accuracy"]),
    }
    out["truncated_rows"] = {}
    for r in results:
        n = 0
        for task, d in r["tasks"].items():
            if task == "figlib":
                n += d["truncated"]
            else:
                arms = ("bare", "tool") if task == "tooluse" else ("bare", "grounded")
                n += d[arms[0]]["truncated"] + d[arms[1]]["truncated"]
        if n:
            out["truncated_rows"][r["label"]] = n
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(OUT))
    add_model_args(ap, default_tier="every")
    args = ap.parse_args()
    selected = resolve_models(args, default_tier="every")
    results = [score_model(m) for m in selected]
    results = [r for r in results if r["tasks"]]
    print_tables(results)
    t = tallies(results)
    for task, d in t.items():
        print("%%", task, json.dumps(d))
    pathlib.Path(args.out).write_text(json.dumps({"models": results, "tallies": t}, indent=2) + "\n", encoding="utf-8")
    print("wrote", args.out, "for", len(results), "models")


if __name__ == "__main__":
    main()
