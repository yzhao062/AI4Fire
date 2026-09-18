"""Run the WildFireVQA aerial question-answering task: one question per call, bare or grounded with the thermal summary.

    python run_wildfirevqa.py --dry-run
    python run_wildfirevqa.py --models claude-opus-5 --conditions bare grounded --limit 3

Every call carries two images of the same FLAME 3 frame, the RGB aerial view (Corrected FOV) and the thermal view
rendered from the radiometric Celsius TIFF with the inferno colormap scaled from the frame's minimum to its maximum
(match_flame3.py writes both paths into task-wildfirevqa/image-map.json). The bare arm asks the question over the two
images alone. The grounded arm adds WildFireVQA's temp_summary block, the seven radiometric statistics of that TIFF
(min, max, mean, std, top-3 mean, share of pixels over 200 and over 400 degrees C), which is the retrieved thermal
statistic WildFireVQA's own finding turns on; including or withholding it runs the comparison on the same items.
WildFireVQA put all 34 questions of a frame into one prompt; this runner asks one question per call so that each
item is scored on its own and no answer can lean on a neighbour's. The answer contract follows the source template:
one option, verbatim, plus an applicability score in [0, 1] with the source rubric.

The output cap is the paper-wide 1,536 tokens. Temperature is zero where the endpoint allows it (gpt-6-astra runs at
its enforced default of 1), which gw.py handles.
"""
import argparse
import base64
import collections
import json
import pathlib
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import gw

S = pathlib.Path(__file__).parent
MAX_OUT = 1536
TASK = S / "task-wildfirevqa"
FLAME3 = S / "data" / "flame3" / "FLAME 3 CV Dataset (Sycan Marsh)"

SYSTEM_BARE = (
    "You are an expert wildfire visual question answering model.\n\n"
    "You are given:\n"
    "- one RGB aerial image of a prescribed burn, taken from a UAV\n"
    "- one thermal image of the same frame, rendered from the radiometric TIFF with the inferno colormap, scaled from "
    "the frame's minimum temperature (black) to its maximum (bright yellow)\n\n"
    "A hotspot is defined as any cluster of pixels that is significantly warmer than the surrounding area.\n\n"
    "Your task:\n"
    "1. Answer the question by choosing exactly one of the provided answer choices, copied verbatim.\n"
    "2. Rate how applicable (answerable) the question is for this specific frame using ONLY the provided inputs.\n\n"
    "The applicability score must be a floating-point value from 0.0 to 1.0 defined as:\n"
    "- 1.0: The information required to answer the question is clearly visible or measurable in the provided inputs.\n"
    "- 0.7 to 0.9: The question is partially answerable; evidence exists but is ambiguous, low resolution, occluded, or incomplete.\n"
    "- 0.4 to 0.6: The question is mostly guesswork; only weak or indirect evidence is present.\n"
    "- 0.1 to 0.3: Very little evidence is available; the question is barely answerable.\n"
    "- 0.0: The question cannot be answered from this frame.\n\n"
    "Output ONLY one JSON object and nothing else, in the form "
    '{"answer": "<one option, verbatim>", "applicability_score": <float from 0.0 to 1.0>}'
)
SYSTEM_GROUNDED = SYSTEM_BARE.replace(
    "the frame's minimum temperature (black) to its maximum (bright yellow)\n\n",
    "the frame's minimum temperature (black) to its maximum (bright yellow)\n"
    "- a temperature summary describing the radiometric TIFF, in degrees Celsius\n\n"
    "You must infer relative and approximate absolute temperatures by:\n"
    "- Using the provided min and max temperatures as the endpoints of the colormap scale\n"
    "- Assuming the colormap is applied monotonically\n"
    "- Mapping observed thermal colors to their approximate position within that range\n\n",
)
SUMMARY_KEYS = ("min", "max", "mean", "std", "top3_mean", "pct_over_200", "pct_over_400")


def b64(path):
    return base64.b64encode(pathlib.Path(path).read_bytes()).decode()


def image_parts(entry):
    rgb = FLAME3 / entry["rgb_corrected_fov"]
    thermal = S / entry["thermal_rendered_inferno"]
    return [{"type": "text", "text": "RGB aerial image:"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64(rgb)}},
            {"type": "text", "text": "Thermal image (inferno colormap of the radiometric TIFF):"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64(thermal)}}]


def question_text(item, grounded):
    lines = []
    if grounded:
        ts = {k: item["temp_summary"][k] for k in SUMMARY_KEYS}
        lines.append("Temperature summary of the radiometric TIFF (degrees Celsius; pct fields are shares of pixels): "
                     + json.dumps(ts))
        lines.append("")
    lines.append("Question: " + item["question"])
    lines.append("Answer choices: " + " | ".join(item["options"]))
    lines.append('Reply with one JSON object: {"answer": "<one of the choices, verbatim>", "applicability_score": <0.0 to 1.0>}')
    return "\n".join(lines)


def messages(item, entry, condition):
    grounded = condition == "grounded"
    content = image_parts(entry) + [{"type": "text", "text": question_text(item, grounded)}]
    return [{"role": "system", "content": SYSTEM_GROUNDED if grounded else SYSTEM_BARE},
            {"role": "user", "content": content}]


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9%<>.\- ]", " ", str(s).lower())).strip()


def parse(text, options):
    """Return (chosen option or None, applicability score or None, failure label or None)."""
    if not text:
        return None, None, "empty"
    answer, score = None, None
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            answer = obj.get("answer")
            score = obj.get("applicability_score")
        except (json.JSONDecodeError, AttributeError):
            pass
    if answer is None:
        m2 = re.search(r'"answer"\s*:\s*"([^"]*)"', text)
        answer = m2.group(1) if m2 else None
    if score is None:
        m3 = re.search(r'"applicability_score"\s*:\s*([0-9.]+)', text)
        score = float(m3.group(1)) if m3 else None
    try:
        score = float(score) if score is not None else None
        if score is not None and not (0.0 <= score <= 1.0):
            score = min(max(score, 0.0), 1.0)
    except (TypeError, ValueError):
        score = None
    chosen = None
    if answer is not None:
        na = norm(answer)
        exact = [o for o in options if norm(o) == na]
        if exact:
            chosen = exact[0]
        else:
            # An answer that quotes an option with extra words: take the longest option contained in it, if unique.
            inside = [o for o in options if norm(o) and norm(o) in na]
            if len(inside) >= 1:
                inside.sort(key=lambda o: -len(o))
                if len(inside) == 1 or len(inside[0]) > len(inside[1]):
                    chosen = inside[0]
    if chosen is None and answer is None:
        # No JSON at all: accept the one option the free text names, if it names exactly one.
        low = norm(text)
        named = [o for o in options if norm(o) and re.search(r"(?<![a-z0-9])" + re.escape(norm(o)) + r"(?![a-z0-9])", low)]
        if len(named) == 1:
            chosen = named[0]
    if chosen is None:
        return None, score, "no_option" if answer is not None else "no_json"
    return chosen, score, None


def score_rows(rows, label):
    ok = [r for r in rows if r["prediction"] is not None]
    out = {"run": label, "items": len(rows), "parsed": len(ok), "errors": sum(1 for r in rows if r.get("error"))}
    if not ok:
        return out
    correct = np.array([r["correct"] for r in rows], dtype=float)  # unparsed counts as wrong
    out["accuracy"] = float(correct.mean())
    out["accuracy_parsed_only"] = float(np.mean([r["correct"] for r in ok]))
    out["majority_baseline_accuracy"] = float(np.mean([r["baseline_majority_correct"] for r in rows]))
    by = collections.defaultdict(list)
    for r in rows:
        by[r["category"]].append(r["correct"])
    out["accuracy_by_category"] = {k: round(float(np.mean(v)), 3) for k, v in sorted(by.items())}
    byq = collections.defaultdict(list)
    for r in rows:
        byq[r["question_id"]].append(r["correct"])
    out["accuracy_by_question"] = {k: round(float(np.mean(v)), 3) for k, v in sorted(byq.items())}
    cf = [r for r in rows if r["no_image_answerable"] == "closed_form"]
    rest = [r for r in rows if r["no_image_answerable"] != "closed_form"]
    out["accuracy_closed_form_items"] = float(np.mean([r["correct"] for r in cf])) if cf else None
    out["accuracy_other_items"] = float(np.mean([r["correct"] for r in rest])) if rest else None
    scores = [r["applicability"] for r in rows if r.get("applicability") is not None]
    out["applicability_mean"] = float(np.mean(scores)) if scores else None
    hi = [r for r in rows if (r.get("applicability") or 0.0) >= 0.5]
    lo = [r for r in rows if r.get("applicability") is not None and r["applicability"] < 0.5]
    out["accuracy_applicability_ge_0.5"] = float(np.mean([r["correct"] for r in hi])) if hi else None
    out["accuracy_applicability_lt_0.5"] = float(np.mean([r["correct"] for r in lo])) if lo else None
    out["items_applicability_lt_0.5"] = len(lo)
    out["failures"] = dict(collections.Counter(r["failure"] for r in rows if r.get("failure")))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["claude-opus-5"])
    ap.add_argument("--conditions", nargs="*", default=["bare", "grounded"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--suffix", default="", help="appended to the response file name, for pilots")
    ap.add_argument("--retry-errors", action="store_true",
                    help="re-run only the rows of an existing response file whose call raised (transport errors), keep the rest")
    args = ap.parse_args()

    items = [json.loads(l) for l in (TASK / "items.jsonl").read_text(encoding="utf-8").splitlines()]
    image_map = json.loads((TASK / "image-map.json").read_text(encoding="utf-8"))
    missing = [it["item_id"] for it in items if it["image"]["image_uid"] not in image_map]
    if missing:
        raise SystemExit("%d items have no image on disk; run match_flame3.py first (e.g. %s)" % (len(missing), missing[0]))
    if args.limit:
        items = items[:args.limit]
    print("items: %d | images %d | categories %s" % (len(items), len({i["image"]["image_uid"] for i in items}),
                                                    dict(collections.Counter(i["category"] for i in items))))

    if args.dry_run:
        it = items[0]
        for cond in ("bare", "grounded"):
            m = messages(it, image_map[it["image"]["image_uid"]], cond)
            print("\n[%s] system chars: %d | user parts: %s | image base64 chars: %d" % (
                cond, len(m[0]["content"]), [p["type"] for p in m[1]["content"]],
                sum(len(p["image_url"]["url"]) for p in m[1]["content"] if p["type"] == "image_url")))
            print(m[1]["content"][-1]["text"])
        print("\nfirst item:", it["item_id"], "| answer", it["answer"], "| options", it["options"])
        return

    key = gw.load_key()
    summaries = []
    for model in args.models:
        for cond in args.conditions:
            def one(it):
                entry = image_map[it["image"]["image_uid"]]
                base = {"item_id": it["item_id"], "question_id": it["question_id"], "category": it["category"],
                        "image_uid": it["image"]["image_uid"], "answer": it["answer"], "options": it["options"],
                        "no_image_answerable": it["no_image_answerable"],
                        "baseline_majority_correct": it["baseline_majority_correct"]}
                try:
                    text, usage, served = gw.call(key, model, messages(it, entry, cond), max_tokens=MAX_OUT)
                except Exception as exc:
                    return dict(base, error=str(exc)[:300], raw=None, prediction=None, applicability=None,
                                correct=False, failure="error")
                pred, app, fail = parse(text, it["options"])
                return dict(base, raw=text, usage=usage, served_model=served, prediction=pred, applicability=app,
                            correct=(pred == it["answer"]), failure=fail, error=None)

            safe = re.sub(r"[^A-Za-z0-9._-]", "_", model)
            out_dir = TASK / "pilot" if args.suffix else TASK
            out_dir.mkdir(exist_ok=True)
            path = out_dir / ("responses-%s-%s%s.jsonl" % (safe, cond, args.suffix))
            if args.retry_errors:
                # Keep every row that returned an answer; re-run only the calls that raised.
                kept = {r["item_id"]: r for r in (json.loads(l) for l in path.read_text(encoding="utf-8").splitlines())}
                todo = [it for it in items if kept.get(it["item_id"], {}).get("error")]
                print("%s/%s: retrying %d errored rows of %d" % (model, cond, len(todo), len(kept)))
                with ThreadPoolExecutor(max_workers=args.workers) as ex:
                    for r in ex.map(one, todo):
                        kept[r["item_id"]] = r
                rows = [kept[it["item_id"]] for it in items]
            else:
                with ThreadPoolExecutor(max_workers=args.workers) as ex:
                    rows = list(ex.map(one, items))
            path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
            s = score_rows(rows, "%s/%s%s" % (model, cond, args.suffix))
            s["tokens_in"] = sum((r.get("usage") or {}).get("prompt_tokens", 0) for r in rows)
            s["tokens_out"] = sum((r.get("usage") or {}).get("completion_tokens", 0) for r in rows)
            summaries.append(s)
            print(json.dumps(s))

    if not args.suffix:
        path = TASK / "scores.json"
        old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        fresh = {s["run"] for s in summaries}
        path.write_text(json.dumps([s for s in old if s["run"] not in fresh] + summaries, indent=1), encoding="utf-8")
        print("\nwrote", path)


if __name__ == "__main__":
    main()
