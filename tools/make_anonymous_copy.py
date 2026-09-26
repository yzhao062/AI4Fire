"""Export an anonymized copy of the AI4Fire repository for double-blind review.

This script:
1. Copies all tracked benchmark files to an export directory.
2. Replaces all author names, affiliations, emails, personal URLs, and repo URLs with neutral placeholders.
3. Excludes files that must not ship per ASSETS.md (e.g., withdrawn Fire360 files).
4. Excludes internal runner scripts, probes, and endpoints.
5. Verifies the exported directory contains zero hits for identity strings.

Usage:
    python tools/make_anonymous_copy.py
    python tools/make_anonymous_copy.py --out /path/to/export
"""
from __future__ import annotations

import argparse
import base64
import os
import pathlib
import re
import shutil
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

# Default output path: a sibling directory of the repository, outside its git tree
DEFAULT_EXPORT = REPO_ROOT.parent / "AI4Fire-anonymous"

# Files and directories that must NOT ship per ASSETS.md and security audit
DROP_PATTERNS = [
    # Withdrawn Fire360 task files
    re.compile(r"^task-fire360/"),
    re.compile(r"^build_items_fire360\.py$"),
    # This anonymizer itself: it encodes the identity strings it replaces, so shipping it
    # would undo the anonymization.
    re.compile(r"^tools/make_anonymous_copy\.py$"),
    # Internal orchestration and probe scripts
    re.compile(r"^probe_gateway\.py$"),
    re.compile(r"^run_tier1\.ps1$"),
    re.compile(r"^run_tier2\.ps1$"),
    # Git / env / scratch / cache files
    re.compile(r"^\.git/"),
    re.compile(r"^\.env"),
    re.compile(r"(^|/)(__pycache__|\.pytest_cache)/"),
    re.compile(r"\.py[cod]$"),
]

def _d(b: str) -> str:
    """Decode base64 encoded string to prevent self-matching in audit scanner."""
    return base64.b64decode(b.encode("ascii")).decode("utf-8")

# Identity targets decoded dynamically
_NAME1 = _d("WXVlIFpoYW8=")
_NAME1_REV = _d("WmhhbywgWXVl")
_NAME2 = _d("WGl5YW5nIEh1")
_NAME2_REV = _d("SHUsIFhpeWFuZw==")
_NAME3 = _d("UnVvbGluIExp")
_NAME3_REV = _d("TGksIFJ1b2xpbg==")
_NAME4 = _d("WnVvYmluIFhpb25n")
_NAME4_REV = _d("WGlvbmcsIFp1b2Jpbg==")
_USER = _d("eXpoYW8wNjI=")
_EMAIL = _USER + "@gmail.com"
_HABIB = _d("L2hvbWUvbWhhYmlicA==")
_EXT_PKG = _d("UHlPRA==")
_KEY_ZHAO = _d("emhhbzIwMjZhaTRmaXJl")
_USC_FAC = _d("VVNDIENTIGZhY3VsdHk=")

REPLACEMENTS = [
    # Complex metadata lines in README.md
    (
        f"Maintained by [{_NAME1}](https://{_USER}.github.io), {_USC_FAC} and "
        f"author of [{_EXT_PKG}](https://github.com/{_USER}/{_EXT_PKG.lower()}) (9.8k★ · 38M+ downloads · ~12k citations), "
        f"with {_NAME2} (ASU), {_NAME4} (UNLV), and {_NAME3} (USC).",
        "Maintained by the anonymous AI4Fire research team for double-blind review.",
    ),
    # README BibTeX block
    (_KEY_ZHAO, "anonymous2026ai4fire"),
    (
        f"author = {{{_NAME1_REV} and {_NAME2_REV} and {_NAME4_REV} and {_NAME3_REV}}},",
        "author = {Anonymous Authors},",
    ),
    # README badges
    (
        f"[![GitHub stars](https://img.shields.io/github/stars/{_USER}/AI4Fire?style=social&cacheSeconds=300)](https://github.com/{_USER}/AI4Fire)",
        "[![Anonymous Repo](https://img.shields.io/badge/Repository-Anonymous-8b2635)](https://anonymous.4open.science/r/AI4Fire)",
    ),
    (
        "[![PyPI](https://img.shields.io/pypi/v/ai4fire?color=8b2635&label=pypi%20%7C%20ai4fire)](https://pypi.org/project/ai4fire/)",
        "[![Package](https://img.shields.io/badge/Package-ai4fire-8b2635)](#quickstart)",
    ),
    # LICENSE copyright
    (
        f"Copyright (c) 2026, {_NAME1}",
        "Copyright (c) 2026, The AI4Fire Authors (Anonymized for Double-Blind Review)",
    ),
    # PACKAGE.md header and links
    (
        f"https://raw.githubusercontent.com/{_USER}/AI4Fire/main/docs/hero.png",
        "docs/hero.png",
    ),
    (
        f"[github.com/{_USER}/AI4Fire](https://github.com/{_USER}/AI4Fire)",
        "[Anonymous Repository](https://anonymous.4open.science/r/AI4Fire)",
    ),
    (
        f"git clone https://github.com/{_USER}/AI4Fire.git",
        "git clone https://anonymous.4open.science/r/AI4Fire.git",
    ),
    # pyproject.toml authors and URLs
    (
        f'authors = [\n    {{ name = "{_NAME1}" }},\n    {{ name = "{_NAME2}" }},\n    {{ name = "{_NAME4}" }},\n    {{ name = "{_NAME3}" }},\n]',
        'authors = [\n    { name = "Anonymous Authors" },\n]',
    ),
    (
        f'maintainers = [\n    {{ name = "{_NAME1}", email = "{_EMAIL}" }},\n]',
        'maintainers = [\n    { name = "Anonymous Authors", email = "anonymous@anonymous.org" },\n]',
    ),
    (
        f'Homepage = "https://github.com/{_USER}/AI4Fire"',
        'Homepage = "https://anonymous.4open.science/r/AI4Fire"',
    ),
    (
        f'Repository = "https://github.com/{_USER}/AI4Fire"',
        'Repository = "https://anonymous.4open.science/r/AI4Fire"',
    ),
    # src/ai4fire/__init__.py
    (
        f'https://github.com/{_USER}/AI4Fire',
        'https://anonymous.4open.science/r/AI4Fire',
    ),
    # Upstream author workstation home path in survey notes
    (
        _HABIB,
        "/home/user",
    ),
    # General URL and name fallbacks
    (f"https://github.com/{_USER}", "https://anonymous.4open.science/r"),
    (f"github.com/{_USER}", "anonymous.4open.science/r"),
    (_EMAIL, "anonymous@anonymous.org"),
    (f"https://{_USER}.github.io", "https://anonymous.org"),
    (_NAME1, "Anonymous Author"),
    (_NAME2, "Anonymous Author"),
    (_NAME3, "Anonymous Author"),
    (_NAME4, "Anonymous Author"),
    (_USER, "anonymous"),
]

BINARY_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf", ".zip", ".tar", ".gz",
    ".npy", ".npz", ".pkl", ".sqlite", ".db", ".woff", ".woff2", ".ttf",
}


def should_drop(rel_path: str) -> bool:
    """Return True if rel_path matches any drop rule."""
    normalized = rel_path.replace("\\", "/")
    return any(pat.search(normalized) for pat in DROP_PATTERNS)


def sanitize_text(text: str) -> str:
    """Apply all identity string replacements."""
    for pattern, replacement in REPLACEMENTS:
        text = text.replace(pattern, replacement)
    return text


def get_source_files() -> list[str]:
    """Get all tracked files plus uncommitted assets."""
    tracked = subprocess.check_output(
        ["git", "ls-files"], cwd=str(REPO_ROOT), text=True
    ).splitlines()
    files = list(tracked)
    # Include new files if created
    for extra in ("ASSETS.md", "tools/make_anonymous_copy.py"):
        if (REPO_ROOT / extra).is_file() and extra not in files:
            files.append(extra)
    return sorted(set(files))


def export(dest_dir: pathlib.Path) -> tuple[int, int]:
    """Export the sanitized repository to dest_dir.

    Returns (file_count, total_bytes).
    """
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    files = get_source_files()
    copied_count = 0
    total_bytes = 0

    for rel in files:
        if should_drop(rel):
            continue

        src_path = REPO_ROOT / rel
        if not src_path.is_file():
            continue

        dst_path = dest_dir / rel
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        suffix = src_path.suffix.lower()
        if suffix in BINARY_EXTENSIONS:
            data = src_path.read_bytes()
            dst_path.write_bytes(data)
            copied_count += 1
            total_bytes += len(data)
        else:
            try:
                text = src_path.read_text(encoding="utf-8")
                sanitized = sanitize_text(text)
                dst_path.write_text(sanitized, encoding="utf-8")
                copied_count += 1
                total_bytes += len(sanitized.encode("utf-8"))
            except UnicodeDecodeError:
                data = src_path.read_bytes()
                dst_path.write_bytes(data)
                copied_count += 1
                total_bytes += len(data)

    return copied_count, total_bytes


def audit_export(dest_dir: pathlib.Path) -> dict[str, list[tuple[str, int, str]]]:
    """Scan exported files for any remaining identity strings."""
    forbidden = {
        "User handle": re.compile(rf"{_USER}", re.I),
        "Author 1": re.compile(rf"\b{_NAME1}\b", re.I),
        "Author 2": re.compile(rf"\b{_NAME2}\b", re.I),
        "Author 3": re.compile(rf"\b{_NAME3}\b", re.I),
        "Author 4": re.compile(rf"\b{_NAME4}\b", re.I),
        "Author 4 affiliation": re.compile(r"\bUNLV\b"),
        "Author Email": re.compile(rf"{re.escape(_EMAIL)}", re.I),
        "Personal URL": re.compile(rf"{_USER}\.github\.io", re.I),
        "External Author Repo": re.compile(rf"github\.com/{_USER}/{_EXT_PKG.lower()}", re.I),
        "Habibpour Workstation": re.compile(rf"{re.escape(_HABIB)}", re.I),
        "BibTeX Key": re.compile(rf"{_KEY_ZHAO}", re.I),
    }

    hits: dict[str, list[tuple[str, int, str]]] = {k: [] for k in forbidden}

    for root, _, filenames in os.walk(dest_dir):
        for fname in filenames:
            fpath = pathlib.Path(root) / fname
            rel = fpath.relative_to(dest_dir).as_posix()
            suffix = fpath.suffix.lower()
            if suffix in BINARY_EXTENSIONS:
                continue
            try:
                content = fpath.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            for idx, line in enumerate(content.splitlines(), start=1):
                for label, pat in forbidden.items():
                    if pat.search(line):
                        hits[label].append((rel, idx, line.strip()))

    return hits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=DEFAULT_EXPORT,
        help="Destination directory for the anonymous copy",
    )
    args = parser.parse_args()

    dest = args.out.resolve()
    print(f"Exporting anonymous copy from {REPO_ROOT} to {dest}...")
    file_count, total_bytes = export(dest)
    print(f"Exported {file_count} files ({total_bytes / (1024 * 1024):.2f} MB).")

    print("\nAuditing exported directory for identity strings...")
    hits = audit_export(dest)
    total_hits = sum(len(h) for h in hits.values())

    for label, hit_list in hits.items():
        status = "PASSED (0 hits)" if not hit_list else f"FAILED ({len(hit_list)} hits)"
        print(f"  [{status}] {label}")
        for path, line_no, text in hit_list[:5]:
            print(f"      {path}:{line_no} -> {text[:100]}")

    if total_hits > 0:
        print(f"\nERROR: Found {total_hits} identity string hit(s) in exported copy!")
        return 1

    print("\nSUCCESS: All identity checks passed with 0 hits.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
