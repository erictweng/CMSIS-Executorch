"""Read the live loop's telemetry over J-Link (no UART needed): letter, probability, presence,
fps, stage times, box, pending motion letter and its path, committed text length, handedness.

    .venv/bin/python tests/board_telemetry.py [--count 5] [--interval 1.0]
    .venv/bin/python tests/board_telemetry.py --count 60 --interval 0.5 --handedness

--handedness ends with a summary of the window: left/right flips between consecutive frames
of a hand (each flip mirrors every x feature the letter MLP sees), frames with an uncertain
handedness score (0.3-0.7) and the share of frames read as a left hand.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fs_tools as fs  # noqa: E402


N = 17


def read_tele() -> list[int]:
    base = fs.symbol_address("g_fs_tele")
    out = fs.jlink(["h", f"mem32 0x{base:08X} {N}", "g"])
    words: list[int] = []
    for line in out.splitlines():  # "2008A7A0 = 00000000 00000000 00000000 000044F4", four words per line
        m = re.match(r"^([0-9A-F]{8}) = ((?:[0-9A-F]{8} ?)+)$", line.strip())
        if m and base <= int(m.group(1), 16) < base + N * 4:
            words += [int(w, 16) for w in m.group(2).split()]
    if len(words) < N:
        raise RuntimeError(out[-600:])
    return words[:N]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--count", type=int, default=5)
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--handedness", action="store_true", help="summarise left/right flips over the window")
    args = ap.parse_args()
    first = last = None
    for i in range(args.count):
        t = read_tele()
        first = first or t
        last = t
        letter = chr(t[0]) if t[0] else "-"
        pending = chr(t[9]) if t[9] else "-"
        print(f"{letter} {t[1] / 1000:.2f} p{t[2] / 1000:.2f} | box {t[6]},{t[7]} {t[8]}px | det {t[4] / 1000:.1f}ms "
              f"lm {t[5] / 1000:.1f}ms | {t[3] / 1000:.1f} fps | pending {pending} path {t[10] / 1000:.2f} | text len {t[11]} "
              f"| hand {t[12] / 1000:.2f} {'R' if t[12] >= 500 else 'L'} flips {t[14]}")
        if i + 1 < args.count:
            time.sleep(args.interval)
    if args.handedness and first and last:
        frames, flips, unsure, left = (last[k] - first[k] for k in (13, 14, 15, 16))
        print(f"handedness over the window: {frames} hand frames, {flips} left/right flips, "
              f"{unsure} uncertain (0.3-0.7), {left} read as left"
              + (f" | {100 * flips / frames:.1f} % of frames flip, {100 * left / frames:.0f} % left" if frames else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
