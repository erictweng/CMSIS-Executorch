"""Calibrate the camera's image axes against the DevKit-E8 accelerometer (src/fs_orient_cal.h).

    .venv/bin/python tests/imu_calibrate.py

Two poses, each averaged over --samples J-Link reads of the live loop's gravity telemetry:
  1. the board as it is mounted for the demo, camera level (the image's top edge really up),
  2. the whole board rolled 90 degrees clockwise about the camera's optical axis, as seen from
     the camera's point of view (the image's right edge now at the bottom).
Pose 1 gives the image's "down" direction in sensor axes, pose 2 the image's "right". The two
are orthogonalised and written to src/fs_orient_cal.h; rebuild and flash afterwards. The
residual |down . right| should be under 0.1 (poses 90 degrees apart); otherwise redo it.

--check reads the current roll instead: hold the board level, expect about 0.

Needs the DevKit-E8 running the live loop (with the IMU up: boot prints "imu init OK") and
the VS Code Serial Monitor closed.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import board_telemetry as bt  # noqa: E402
import fs_tools as fs  # noqa: E402

CAL = fs.REPO / "src" / "fs_orient_cal.h"


def gravity(samples: int, interval: float) -> np.ndarray:
    gs = []
    for i in range(samples):
        t = bt.read_tele()
        gs.append([bt.signed(t[17 + k]) / 1000.0 for k in range(3)])
        print(f"  {i + 1}/{samples}: g = ({gs[-1][0]:+.3f} {gs[-1][1]:+.3f} {gs[-1][2]:+.3f})", flush=True)
        time.sleep(interval)
    g = np.mean(gs, axis=0)
    spread = np.max(np.std(gs, axis=0))
    if np.linalg.norm(g) < 0.5:
        raise SystemExit(f"|g| = {np.linalg.norm(g):.2f}: the IMU is not reporting (boot print says?)")
    if spread > 0.05:
        print(f"  warning: readings moved (std {spread:.3f} g); hold the board still", file=sys.stderr)
    return g


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", type=int, default=15)
    ap.add_argument("--interval", type=float, default=0.3)
    ap.add_argument("--check", action="store_true", help="print the live roll instead of calibrating")
    args = ap.parse_args()
    if args.check:
        for _ in range(args.samples):
            t = bt.read_tele()
            print(f"roll {bt.signed(t[20]) / 1000:+.1f} deg  g ({bt.signed(t[17]) / 1000:+.2f} {bt.signed(t[18]) / 1000:+.2f} {bt.signed(t[19]) / 1000:+.2f})")
            time.sleep(args.interval)
        return 0
    input("Pose 1: board mounted as for the demo, camera level. Hold still and press Enter... ")
    down = gravity(args.samples, args.interval)
    input("Pose 2: roll the whole board 90 degrees clockwise about the camera axis (image's right edge down). Press Enter... ")
    right = gravity(args.samples, args.interval)
    down /= np.linalg.norm(down)
    residual = abs(float(np.dot(right / np.linalg.norm(right), down)))
    right = right - np.dot(right, down) * down
    right /= np.linalg.norm(right)
    print(f"down  = ({down[0]:+.4f} {down[1]:+.4f} {down[2]:+.4f})")
    print(f"right = ({right[0]:+.4f} {right[1]:+.4f} {right[2]:+.4f})  residual |down.right| before orthogonalising {residual:.3f}")
    if residual > 0.1:
        print("residual over 0.1: the two poses were not 90 degrees apart; redo the calibration", file=sys.stderr)
        return 1
    fmt = lambda v: "{" + ", ".join(f"{x:.5f}f" for x in v) + "}"
    CAL.write_text(
        "// fs_orient_cal.h: the camera's image axes in the DevKit-E8 accelerometer's axes, written by\n"
        "// tests/imu_calibrate.py (two poses: board as mounted and level, then rolled 90 degrees\n"
        "// clockwise about the camera axis). Zero vectors = not calibrated: the roll stays 0 and the\n"
        "// landmarks are not levelled.\n"
        f"// Calibrated {time.strftime('%Y-%m-%d %H:%M')}, residual {residual:.3f}.\n"
        "#pragma once\n"
        "#define FS_ORIENT_CALIBRATED 1\n"
        f"#define FS_ORIENT_CAL_DOWN {fmt(down)}\n"
        f"#define FS_ORIENT_CAL_RIGHT {fmt(right)}\n")
    print(f"wrote {CAL}; rebuild and flash, then `tests/imu_calibrate.py --check` with the board level should read about 0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
