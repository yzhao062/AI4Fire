"""Regenerate every figure the paper includes and compare it with the copy in figures/.

Each script is called the way the paper's figures are written, with both --out-pdf and --out-png, because a
PNG-only invocation takes a different save path in some scripts (dpi 300 with a tight bounding box, for
review) and produces a different image by design.

    python figures/check_reproduction.py            # all figures
    python figures/check_reproduction.py calibration prompt_sensitivity

Exit status is 1 if any figure differs, so this can gate a change to a figure script.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

# stem -> (generator script, extra arguments)
FIGURES = {
    "allocation_analogues": ("make_allocation_analogues.py", []),
    "calibration": ("make_calibration.py", []),
    "figlib_timeline": ("make_figlib_timeline.py", []),
    "grounding_effects": ("make_grounding_effects.py", ["--tier", "core"]),
    "grounding_effects_sweep": ("make_grounding_effects.py", ["--tier", "all"]),
    "prompt_sensitivity": ("make_prompt_sensitivity.py", []),
    "survey_landscape": ("make_survey_landscape.py", []),
    "wildfirevqa": ("make_wildfirevqa.py", []),
}


def png_difference(reference: Path, candidate: Path) -> str:
    """Empty string when the two PNGs are pixel-identical, otherwise what differs."""
    a = np.asarray(Image.open(reference).convert("RGB"), dtype=int)
    b = np.asarray(Image.open(candidate).convert("RGB"), dtype=int)
    if a.shape != b.shape:
        return f"size {a.shape[1]}x{a.shape[0]} against {b.shape[1]}x{b.shape[0]}"
    diff = np.abs(a - b)
    n = int((diff.sum(axis=2) > 0).sum())
    return "" if n == 0 else f"{n} pixels differ, largest channel difference {diff.max()}"


def compare(stem: str, out_dir: Path, paper_figures: Path | None) -> tuple[str, bool]:
    """Regenerate one figure and compare it with the copy in figures/, and with the paper's copy when given.

    The comparison is on PNG pixels. The PDF is required to exist, because that is the file the paper
    includes, but two PDF files that render the same are not byte-identical, so it is not compared here.
    """
    script, extra = FIGURES[stem]
    png, pdf = out_dir / f"{stem}.png", out_dir / f"{stem}.pdf"
    cmd = [sys.executable, str(HERE / script), "--out-png", str(png), "--out-pdf", str(pdf)] + extra
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO))
    if proc.returncode != 0 or not png.exists():
        tail = (proc.stderr or "").strip().splitlines()
        return (f"generator failed (rc={proc.returncode}) {tail[-1] if tail else ''}", False)
    if not pdf.exists():
        return ("generator wrote no PDF, the file the paper includes", False)

    committed = HERE / f"{stem}.png"
    if not committed.exists():
        return ("no committed PNG to compare with", False)
    message = png_difference(committed, png)
    if message:
        return (message, False)

    if paper_figures is not None:
        paper_png = paper_figures / f"{stem}.png"
        if not paper_png.exists():
            return ("identical here, absent from the paper's figures directory", False)
        message = png_difference(paper_png, png)
        if message:
            return (f"identical here, but the paper's copy {message}", False)
        return ("identical here and in the paper", True)
    return ("identical", True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stems", nargs="*", choices=sorted(FIGURES), metavar="STEM",
                    help="figure stems to check (default: all)")
    ap.add_argument("--paper-figures", type=Path, default=None,
                    help="also compare each PNG with the copy in this directory, such as the paper's figures/")
    args = ap.parse_args()
    stems = args.stems or sorted(FIGURES)

    ok = True
    with tempfile.TemporaryDirectory(prefix="figcheck-") as tmp:
        out_dir = Path(tmp)
        for stem in stems:
            message, passed = compare(stem, out_dir, args.paper_figures)
            ok &= passed
            print(f"{stem:26s} {message}")
    print("every figure reproduces on PNG pixels" if ok else "at least one figure differs")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
