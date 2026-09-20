"""Rebuild manifest-v1.json from the working tree.

The manifest lists the response files behind the paper: the 36 reported runs (three tasks, six models, bare
and grounded), the supporting repeat and repair files, the historic archives kept as evidence of earlier
defects, the files present in the task directories but excluded from the reported set, the trained non-LLM
baselines, the prompt-sensitivity and retrieval-rule variant runs, the tool-use runs, the aerial
question-answering runs, and the two model-sweep sections of 2026-09-18 (the added full-capability models on
allocation, fire danger, and smoke detection; the text-only models on allocation and fire danger), whose
membership comes from the tiers in models.py.  Membership of the other sections, reasons, and the skew
overrides are read from the existing manifest; row counts, SHA-256 checksums, the distinct served_model
values, and the modification times are recomputed from the files on disk, so the manifest can be
regenerated after any file is rewritten.

    python build_manifest.py            # rewrite manifest-v1.json in place
    python build_manifest.py --check    # exit 1 if any recorded checksum differs from the file on disk
"""
import argparse
import datetime
import hashlib
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent
MANIFEST = ROOT / "manifest-v1.json"
sys.path.insert(0, str(ROOT))
from models import models as registry_models  # noqa: E402

ADDED_STEMS = [m.stem for m in registry_models(tier="added")]
TEXT_STEMS = [m.stem for m in registry_models(tier="text")]

VARIANT_SECTIONS = {
    "baselines": ("Trained non-LLM baselines on the same items and the same displayed information; "
                  "written by baselines/mesogeos_trained.py and baselines/allocation_trained.py.",
                  ["task-mesogeos/responses-baseline-*.jsonl", "task-allocation/responses-baseline-*.jsonl"]),
    "prompt_variants": ("Prompt-sensitivity runs of run_mesogeos.py --variant p1 and p2, bare condition; "
                        "scored by analysis/prompt_sensitivity.py.",
                        ["task-mesogeos/responses-*-bare-p1.jsonl", "task-mesogeos/responses-*-bare-p2.jsonl"]),
    "retrieval_v2": ("Grounded allocation runs under the movement-conditioned analogue rule (run_allocation.py --rule v2); "
                     "scored by analysis/retrieval_v2.py.",
                     ["task-allocation/responses-*-grounded-v2.jsonl"]),
    "tooluse": ("Fire data tool-use runs on FPA-FOD (run_tooluse.py), bare and tool arms, 156 items; "
                "scored by analysis/tooluse_paired.py.",
                ["task-tooluse/responses-*.jsonl"]),
    "wildfirevqa": ("Temperature-grounded aerial question answering on WildFireVQA over FLAME 3 imagery "
                    "(run_wildfirevqa.py), bare and grounded arms, 408 items over 390 frames; "
                    "scored by analysis/wildfirevqa_paired.py; task-wildfirevqa/image-map.json maps each item to its frame.",
                    ["task-wildfirevqa/responses-*.jsonl", "task-wildfirevqa/image-map.json"]),
    "added_models": ("Model sweep of 2026-09-18: the added full-capability Bedrock models (tier 'added' in models.py) on "
                     "personnel allocation, fire danger, and smoke detection, bare and grounded, same prompts and "
                     "cap as the reported runs; their tool-use and aerial files sit in the tooluse and wildfirevqa sections.",
                     ["task-%s/responses-%s-*.jsonl" % (task, stem)
                      for task in ("allocation", "mesogeos", "figlib") for stem in ADDED_STEMS]),
    "text_models": ("Model sweep of 2026-09-18: the text-only Bedrock models (tier 'text' in models.py) on personnel "
                    "allocation and fire danger, bare and grounded; the three reasoning models ran at an 8,192-token "
                    "cap recorded in usage.max_out; tool-use files of the tool-capable ones sit in the tooluse section.",
                    ["task-%s/responses-%s-*.jsonl" % (task, stem)
                     for task in ("allocation", "mesogeos") for stem in TEXT_STEMS]),
}


def _open_path(p):
    path = pathlib.Path(p)
    if os.name == "nt":
        resolved = str(path.resolve())
        if not resolved.startswith("\\\\?\\"):
            return pathlib.Path("\\\\?\\" + resolved)
    return path


def sha256(path):
    h = hashlib.sha256()
    with open(_open_path(path), "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rows_of(path):
    """Rows of a .jsonl file; a .json or .txt supporting file has no rows and returns an empty list."""
    path = pathlib.Path(path)
    if path.suffix != ".jsonl":
        return []
    with open(_open_path(path), encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def describe(rel):
    path = ROOT / rel
    rows = rows_of(path)
    served = sorted({r["served_model"] for r in rows if r.get("served_model")})
    stamp = datetime.datetime.fromtimestamp(_open_path(path).stat().st_mtime).replace(microsecond=0).isoformat()
    return {"row_count": len(rows) if path.suffix == ".jsonl" else None, "sha256": sha256(path), "served_model": served,
            "written": {"earliest": stamp, "latest": stamp, "source": "file mtime"}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    m = json.loads(MANIFEST.read_text(encoding="utf-8"))
    drift = []
    for entry in m["reported"]:
        d = describe(entry["path"])
        if entry.get("sha256") and entry["sha256"] != d["sha256"]:
            drift.append(entry["path"])
        entry.update(d)
    for entry in m["supporting"] + m["excluded_from_reported"]:
        d = describe(entry["path"])
        if entry.get("sha256") and entry["sha256"] != d["sha256"]:
            drift.append(entry["path"])
        entry["row_count"], entry["sha256"] = d["row_count"], d["sha256"]
    for archive in m["historic"]:
        for entry in archive["files"]:
            digest = sha256(ROOT / entry["path"])
            if entry.get("sha256") and entry["sha256"] != digest:
                drift.append(entry["path"])
            entry["sha256"] = digest
    for section, (reason, patterns) in VARIANT_SECTIONS.items():
        files = sorted({p.relative_to(ROOT).as_posix() for pat in patterns for p in ROOT.glob(pat)})
        old = {e["path"]: e.get("sha256") for e in m.get(section, {}).get("files", [])}
        entries = []
        for rel in files:
            d = describe(rel)
            if old.get(rel) and old[rel] != d["sha256"]:
                drift.append(rel)
            entries.append({"path": rel, **d})
        m[section] = {"reason": reason, "files": entries}
    if args.check:
        print("checked %d reported, %d supporting, %d excluded, %d historic, %d variant files; %d changed"
              % (len(m["reported"]), len(m["supporting"]), len(m["excluded_from_reported"]),
                 sum(len(a["files"]) for a in m["historic"]),
                 sum(len(m[s]["files"]) for s in VARIANT_SECTIONS), len(drift)))
        for rel in drift:
            print("  changed since the manifest was written: " + rel)
        raise SystemExit(1 if drift else 0)
    MANIFEST.write_text(json.dumps(m, indent=2) + "\n", encoding="utf-8")
    print("wrote %s: %d reported, %d supporting, %d excluded, %d historic, %s"
          % (MANIFEST.name, len(m["reported"]), len(m["supporting"]), len(m["excluded_from_reported"]),
             sum(len(a["files"]) for a in m["historic"]),
             ", ".join("%d %s" % (len(m[s]["files"]), s) for s in VARIANT_SECTIONS)))
    if drift:
        print("checksums updated for %d files: %s" % (len(drift), ", ".join(drift)))


if __name__ == "__main__":
    main()
