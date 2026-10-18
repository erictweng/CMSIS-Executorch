#!/usr/bin/env python3
"""Watch the voice stage on a running board over J-Link (no UART needed): the mic level, the
model's top word and score every 0.5 s, and every accepted word as it is reported.

    .venv/bin/python tests/voice_check.py [--seconds 30] [--interval 0.5]

Say "yes", "no", "stop", "go" (or up, down, left, right, on, off) at the board from about
30 cm; the accepted words are listed at the end with their scores. Reads g_fs_voice_tele
(src/app_main.cpp) with the core halted for a moment per sample, like tests/board_telemetry.py.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml"))
import fs_tools as fs  # noqa: E402
import kws  # noqa: E402

N = 11


def read_tele() -> list[int]:
    base = fs.symbol_address("g_fs_voice_tele")
    out = fs.jlink(["h", f"mem32 0x{base:08X} {N}", "g"])
    words: list[int] = []
    for line in out.splitlines():
        m = re.match(r"^([0-9A-F]{8}) = ((?:[0-9A-F]{8} ?)+)$", line.strip())
        if m and base <= int(m.group(1), 16) < base + N * 4:
            words += [int(w, 16) for w in m.group(2).split()]
    if len(words) < N:
        raise RuntimeError(out[-600:])
    return words[:N]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=30)
    ap.add_argument("--interval", type=float, default=0.5)
    args = ap.parse_args()
    end = time.time() + args.seconds
    last_reported = None
    accepted: list[tuple[str, float]] = []
    first = None
    while time.time() < end:
        t = read_tele()
        first = first or t
        word = kws.LABELS[t[6]].strip("_") if 0 < t[6] < len(kws.LABELS) else "-"
        if last_reported is not None and t[10] != last_reported:
            accepted.append((word, t[7] / 1000))
            print(f"  >>> accepted: {word} {t[7] / 1000:.2f}")
        last_reported = t[10]
        print(f"running {t[0]} blocks {t[1]} overruns {t[2]} | rms {t[3]} | frames {t[4]} inferences {t[5]} "
              f"| top score {t[7] / 1000:.2f} | npu {t[8] / 1000:.1f}ms mfcc {t[9] / 1000:.2f}ms/frame | words {t[10]}")
        time.sleep(args.interval)
    if first:
        print(f"\n{args.seconds:.0f} s: {t[1] - first[1]} blocks, {t[5] - first[5]} inferences, "
              f"{t[2] - first[2]} overruns, {len(accepted)} words accepted: "
              + ", ".join(f"{w} {s:.2f}" for w, s in accepted))
    return 0


if __name__ == "__main__":
    sys.exit(main())
