"""Step 6: sign all 24 letters at the board and score the live predictions.

    .venv/bin/python tests/letter_check.py                 # all 24 letters, 3 s each
    .venv/bin/python tests/letter_check.py --letters ABC --seconds 4 --save-crops

For each letter: hold the sign in the middle of the camera's view (the centre 256x256 of
the 320x320 frame) in front of a plain background, press Enter, keep holding while it
reads the live loop for --seconds. The letter counts as read when the most frequent
top-1 over that window is the right one. Writes tests/out/letters-<time>.json and, with
--save-crops, one PNG per letter of what the model saw (for retraining).

Needs the DevKit-E8 running the live loop, SW4 on UART4, the Serial Monitor closed.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fs_tools as fs  # noqa: E402

LABELS = "ABCDEFGHIKLMNOPQRSTUVWXY"
LIVE = re.compile(r"^([A-Y]) (\d\.\d\d) [A-Y] \d\.\d\d [A-Y] \d\.\d\d \| .* \| ([\d.]+) fps$", re.M)
OUT = fs.REPO / "tests" / "out"


def read_window(uart: fs.UartCapture, seconds: float) -> list[tuple[str, float]]:
    uart.data.clear()
    uart.read_for(seconds)
    return [(m.group(1), float(m.group(2))) for m in LIVE.finditer(uart.text)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--letters", default=LABELS, help="letters to test (default: all 24)")
    ap.add_argument("--seconds", type=float, default=3.0, help="seconds of live output per letter")
    ap.add_argument("--save-crops", action="store_true", help="save one camera crop per letter as PNG")
    args = ap.parse_args()
    letters = [c for c in args.letters.upper() if c in LABELS]

    OUT.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    results = []
    with fs.UartCapture() as uart:
        if not read_window(uart, 2.0):
            print("No live lines on the UART: is the board running the live loop (flashed, SW4 on UART4)?")
            return 2
        for letter in letters:
            input(f"\nSign  {letter}  in front of the camera and press Enter (keep holding)... ")
            preds = read_window(uart, args.seconds)
            votes = collections.Counter(p for p, _ in preds)
            top, n = votes.most_common(1)[0] if votes else ("?", 0)
            conf = [c for p, c in preds if p == letter]
            hit = top == letter
            crop_png = None
            if args.save_crops:
                w, h, _, rgb = fs.capture_crop(uart)
                crop_png = OUT / f"letter-{stamp}-{letter}.png"
                fs.write_png(crop_png, w, h, rgb, scale=2)
            results.append({"letter": letter, "read_as": top, "hit": hit, "frames": len(preds),
                            "votes": dict(votes), "share": n / max(len(preds), 1),
                            "mean_conf_when_right": sum(conf) / len(conf) if conf else 0.0,
                            "crop": str(crop_png) if crop_png else None})
            others = ", ".join(f"{k} {v}" for k, v in votes.most_common(4))
            print(f"  {'OK  ' if hit else 'MISS'} read as {top} in {n}/{len(preds)} frames  ({others})")

    hits = sum(r["hit"] for r in results)
    misses = [f"{r['letter']}->{r['read_as']}" for r in results if not r["hit"]]
    print(f"\nScore: {hits}/{len(results)} letters" + (f"; misses: {', '.join(misses)}" if misses else ""))
    report = OUT / f"letters-{stamp}.json"
    report.write_text(json.dumps({"score": hits, "total": len(results), "seconds": args.seconds,
                                  "results": results}, indent=2))
    print(f"Report: {report.relative_to(fs.REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
