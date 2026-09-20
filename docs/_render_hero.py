"""One-shot headless render of docs/hero.html to docs/hero.png.

Usage:
    python docs/_render_hero.py
    python docs/_render_hero.py --scale 1
    python docs/_render_hero.py --scale 2

Reads docs/hero.html (the self-contained source) and writes docs/hero.png.
Default scale 1 produces an exact 1400px width card conforming to the
1200-1600px width and <=600px height constraint. Scale 2 provides 2x retina
resolution for high-DPI displays.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import struct
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SRC = HERE / "hero.html"
OUT = HERE / "hero.png"


def get_png_dimensions(path: Path) -> tuple[int, int]:
    """Read width and height directly from PNG IHDR chunk without dependencies."""
    with open(path, "rb") as f:
        header = f.read(24)
        if len(header) >= 24 and header[:8] == b"\x89PNG\r\n\x1a\n":
            width, height = struct.unpack(">II", header[16:24])
            return width, height
    # Fallback to PIL if available
    try:
        from PIL import Image
        with Image.open(path) as img:
            return img.size
    except Exception:
        return 0, 0


def render(scale: int = 1) -> Path:
    if not SRC.exists():
        raise FileNotFoundError(f"Source file not found: {SRC}")

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("Playwright is not installed in the active environment.")
        print("To render docs/hero.png, install Playwright:")
        print("    pip install playwright && playwright install chromium")
        print(f"Alternatively, open {SRC} in a browser and export a screenshot to {OUT}.")
        sys.exit(1)

    url = SRC.as_uri()
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        # Viewport slightly wider than 1400px sheet to avoid scrollbars
        ctx = browser.new_context(
            viewport={"width": 1440, "height": 700},
            device_scale_factor=scale,
        )
        page = ctx.new_page()
        page.goto(url, wait_until="networkidle")
        page.evaluate("document.fonts.ready")
        page.wait_for_timeout(200)

        sheet = page.locator(".sheet")
        sheet.screenshot(path=str(OUT), omit_background=False)
        browser.close()

    width, height = get_png_dimensions(OUT)
    size_bytes = OUT.stat().st_size
    print(f"Output path: {OUT}")
    print(f"PNG dimensions: {width}x{height} (scale factor: {scale})")
    print(f"File size: {size_bytes} bytes")
    return OUT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scale",
        type=int,
        default=1,
        choices=[1, 2],
        help="Device scale factor (1 = natural 1400px width conforming to constraints; 2 = retina).",
    )
    args = parser.parse_args()
    render(scale=args.scale)


if __name__ == "__main__":
    main()
