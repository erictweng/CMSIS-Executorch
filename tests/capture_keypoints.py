"""Record the board's own hand landmarks for one letter, as MLP training data (J-Link, no UART).

    .venv/bin/python tests/capture_keypoints.py --letter R --seconds 10
    .venv/bin/python tests/capture_keypoints.py --letters RUKHNMSTOQDIABLY --seconds 10   # one after another

Sign the letter at the board (hand box >= 75 px on the panel: closer is better) and keep it
up for --seconds. With --letters each letter gets a countdown (--countdown seconds, time to
change the shape), then its recording; the terminal says which letter is next. The script polls `g_fs_lm_dump` (src/app_main.cpp) with the core halted
for each read, so every sample is one consistent frame: the 21 frame-pixel landmarks the
MLP's features are built from (levelled by the camera's roll: gravity straight down, see
src/fs_orient.h), the presence and handedness scores and the roll applied (saved as `roll`). Frames are kept when the
board's hand-frame counter advanced (no duplicates) and presence >= --min-presence.

The output has the layout of ml/keypoint_dataset.py (landmarks normalised by the 320 px
frame, size (320, 320), label = index into LABELS, source "board-live"), so
ml/keypoint_train.py takes it next to the dataset files; an existing --out is appended to.
Letters J and Z are motion letters and are not accepted.

Needs the DevKit-E8 running the live loop and the VS Code Serial Monitor closed.
"""
from __future__ import annotations

import argparse
import re
import struct
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fs_tools as fs  # noqa: E402

LABELS = "ABCDEFGHIKLMNOPQRSTUVWXY"
FRAME = 320
WORDS = 4 + 63  # counter, presence x1000, handedness x1000, roll millidegrees, 21 x (x, y, z) float bits


def signed(w: int) -> int:
    return w - (1 << 32) if w >= (1 << 31) else w


def read_dump(base: int) -> tuple[int, float, float, float, np.ndarray]:
    out = fs.jlink(["h", f"mem32 0x{base:08X} {WORDS}", "g"])
    words: list[int] = []
    for line in out.splitlines():  # "2008A7A0 = 00000000 00000000 00000000 000044F4"
        m = re.match(r"^([0-9A-F]{8}) = ((?:[0-9A-F]{8} ?)+)$", line.strip())
        if m and base <= int(m.group(1), 16) < base + WORDS * 4:
            words += [int(w, 16) for w in m.group(2).split()]
    if len(words) < WORDS:
        raise RuntimeError("J-Link read failed:\n" + out[-600:])
    xyz = np.array([struct.unpack("<f", struct.pack("<I", w))[0] for w in words[4:WORDS]], np.float32).reshape(21, 3)
    return words[0], words[1] / 1000.0, words[2] / 1000.0, signed(words[3]) / 1000.0, xyz


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--letter", help="the letter being signed (A-Y, not J or Z)")
    ap.add_argument("--letters", help="several letters in a row, e.g. RUKHNMSTOQDIABLY")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--out", default="model/keypoints_board_live.npz")
    ap.add_argument("--source", default="board-live")
    ap.add_argument("--min-presence", type=float, default=0.5)
    ap.add_argument("--countdown", type=float, default=3.0, help="seconds to get the hand up before recording")
    args = ap.parse_args()
    letters = (args.letters or "").upper().replace(" ", "") + (args.letter or "").upper()
    if not letters:
        ap.error("give --letter X or --letters XYZ")
    bad = [c for c in letters if c not in LABELS]
    if bad:
        print(f"{''.join(bad)!r}: not in {LABELS} (J and Z move; they are not static letters)", file=sys.stderr)
        return 2
    base = fs.symbol_address("g_fs_lm_dump")
    total = 0
    for n, letter in enumerate(letters):
        kept = record(base, letter, args, f"{n + 1}/{len(letters)}")
        total += kept
        if kept and n + 1 < len(letters):
            print("", flush=True)
    return 0 if total else 1


def record(base: int, letter: str, args, position: str) -> int:
    """Countdown, record `letter` for args.seconds, append to args.out; the frames kept."""
    # Baseline: the counter is already 1 after boot (the self-test crop goes through the same
    # path), and whatever frame is there predates the recording.
    last_frame = read_dump(base)[0]
    for i in range(int(args.countdown), 0, -1):
        print(f"[{position}] sign {letter} ... {i}", flush=True)
        time.sleep(1)
    print(f"[{position}] recording {letter} for {args.seconds:.0f} s", flush=True)
    lm, hd, pres, rolls = [], [], [], []
    polls, dupes, low, errors = 0, 0, 0, 0
    t_end = time.time() + args.seconds
    while time.time() < t_end:
        polls += 1
        try:
            frame, presence, right, roll_deg, xyz = read_dump(base)
        except RuntimeError as e:
            errors += 1
            if errors > 5:
                raise
            print(f"  read error ({e.args[0].splitlines()[0]}), retrying", file=sys.stderr)
            continue
        if frame == last_frame:
            dupes += 1
            continue
        last_frame = frame
        if presence < args.min_presence:
            low += 1
            continue
        lm.append(xyz / FRAME)
        hd.append(right)
        pres.append(presence)
        rolls.append(roll_deg)
        if len(lm) % 10 == 0:
            print(f"  {len(lm)} frames (presence {presence:.2f}, hand {'R' if right >= 0.5 else 'L'} {right:.2f})", flush=True)
    print(f"{len(lm)} frames kept of {polls} polls: {dupes} repeats, {low} below presence {args.min_presence}")
    if not lm:
        print(f"nothing recorded for {letter}: was the hand in view?", file=sys.stderr)
        return 0
    new = dict(landmarks=np.array(lm, np.float32), handedness=np.array(hd, np.float32),
               size=np.full((len(lm), 2), FRAME, np.int32), label=np.full(len(lm), LABELS.index(letter), np.int16),
               source=np.array([args.source] * len(lm)), roll=np.array(rolls, np.float32))
    out = Path(args.out)
    if out.exists():
        old = np.load(out, allow_pickle=False)
        new = {k: np.concatenate([old[k], new[k]]) if k in old.files else new[k] for k in new}
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **new)
    counts = np.bincount(new["label"].astype(int), minlength=len(LABELS))
    print(f"wrote {out}: {len(new['label'])} frames; per letter: "
          + " ".join(f"{LABELS[i]}={c}" for i, c in enumerate(counts) if c))
    return len(lm)


if __name__ == "__main__":
    sys.exit(main())
