"""Step 4's manual check: cover the camera lens and the pixel values must drop.

    .venv/bin/python tests/check_cover_lens.py

Needs the DevKit-E8 running the live loop (flashed firmware), SW4 on UART4 and the VS Code
Serial Monitor closed. Saves both crops to tests/out/.
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fs_tools as fs  # noqa: E402

OUT = fs.REPO / "tests" / "out"


def crop_luma(uart: fs.UartCapture, name: str) -> float:
    w, h, frame, rgb = fs.capture_crop(uart)
    OUT.mkdir(exist_ok=True)
    fs.write_png(OUT / f"{name}.png", w, h, rgb, scale=3)
    mean = statistics.mean((rgb[i] * 77 + rgb[i + 1] * 150 + rgb[i + 2] * 29) >> 8 for i in range(0, len(rgb), 3))
    print(f"  {name}: frame {frame}, mean brightness {mean:.0f}  (tests/out/{name}.png)")
    return mean


def main() -> int:
    with fs.UartCapture() as uart:
        input("Point the camera at the room, lens uncovered, and press Enter... ")
        open_mean = crop_luma(uart, "lens_open")
        input("Now cover the lens completely with your hand or a cap, and press Enter... ")
        covered_mean = crop_luma(uart, "lens_covered")
    ok = covered_mean < 40 and covered_mean < 0.5 * open_mean
    print(f"{'PASS' if ok else 'FAIL'}: brightness {open_mean:.0f} -> {covered_mean:.0f} when covered")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
