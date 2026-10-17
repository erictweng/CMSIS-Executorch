"""Grab what the model sees from the board: the 128x128 camera crop, saved as PNG.

    .venv/bin/python tests/grab_crop.py                    # one picture
    .venv/bin/python tests/grab_crop.py --count 5 --interval 2
    .venv/bin/python tests/grab_crop.py --name my_hand_B

Each picture is the centre 256x256 of the 320x320 camera frame, averaged down to 128x128
(the model input), scaled 3x. The live loop sends it as hex over the UART at 115200 baud,
so one picture takes about 8 s. Also prints the board's prediction for that moment.

Needs the DevKit-E8 running the live loop, SW4 on UART4 and the Serial Monitor closed.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fs_tools as fs  # noqa: E402

OUT = fs.REPO / "tests" / "out"
LIVE = re.compile(r"^([A-Y]) (\d\.\d\d) ([A-Y]) (\d\.\d\d) ([A-Y]) (\d\.\d\d) \|", re.M)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--count", type=int, default=1)
    ap.add_argument("--interval", type=float, default=0.0, help="seconds between pictures")
    ap.add_argument("--name", default="crop", help="file name prefix in tests/out/")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    with fs.UartCapture() as uart:
        for i in range(args.count):
            w, h, frame, rgb = fs.capture_crop(uart)
            before = LIVE.findall(uart.text.split("CROPDUMP BEGIN")[0])
            path = OUT / f"{args.name}-{time.strftime('%H%M%S')}-{i + 1}.png"
            fs.write_png(path, w, h, rgb, scale=3)
            guess = f"  board said: {' '.join(before[-1])}" if before else ""
            print(f"saved {path.relative_to(fs.REPO)} (frame {frame}){guess}")
            if i + 1 < args.count:
                time.sleep(args.interval)
    return 0


if __name__ == "__main__":
    sys.exit(main())
