"""Hand landmarks for a letter dataset through the BOARD's pipeline on the laptop: BlazePalm
(model/blazepalm_192.pt2) -> hand box + rotation -> rotated 224 crop -> hand_landmark_lite
(model/hand_landmark_224.pt2) -> landmarks, as src/fs_detect.c and src/fs_landmark.c do it.
Output has the same layout as ml/keypoint_dataset.py (MediaPipe's own pipeline), so both
can train the MLP; this one matches what the board will produce.

    .venv/bin/python -I ml/keypoint_dataset_board.py <images.parquet> <out.npz> --source marxulia-board [--limit N]

Images are square-cropped and pasted at ~55 % of a 320x320 frame on a grey background (the
detector expects a frame, not a hand filling it); the rest follows ml/blazepalm.py and
ml/handlandmark.py. Images where the detector or the presence gate says no hand are skipped.
"""
from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import blazepalm as bp  # noqa: E402
import handlandmark as hl  # noqa: E402

LABELS = hl.LABELS
FRAME = 320


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("parquet")
    ap.add_argument("out")
    ap.add_argument("--source", default="dataset-board")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--hand-px", type=int, default=176, help="pasted hand size in the 320 frame")
    args = ap.parse_args()
    here = Path(__file__).resolve().parents[1] / "model"
    detector = torch.export.load(here / "blazepalm_192.pt2").module(check_guards=False)
    landmarker = torch.export.load(here / "hand_landmark_224.pt2").module(check_guards=False)
    tbl = pq.read_table(args.parquet)
    images, labels = tbl.column("image").to_pylist(), tbl.column("label").to_pylist()
    n = len(images) if not args.limit else min(args.limit, len(images))
    lm_out, hd, size, lab = [], [], [], []
    no_palm = no_presence = 0
    t0 = time.time()
    for i in range(n):
        letter = chr(65 + labels[i])
        if letter not in LABELS:
            continue
        im = Image.open(io.BytesIO(images[i]["bytes"])).convert("RGB")
        w, h = im.size
        s = min(w, h)
        im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).resize((args.hand_px, args.hand_px), Image.BOX)
        frame = Image.new("RGB", (FRAME, FRAME), (96, 96, 96))
        off = (FRAME - args.hand_px) // 2
        frame.paste(im, (off, off))
        rgb = np.asarray(frame)
        x = torch.from_numpy(bp.resize_area(rgb, bp.INPUT)[None]).float().div(255)
        with torch.no_grad():
            reg, sc = detector(x)
        det = bp.best_detection(reg.numpy(), sc.numpy())
        if det is None:
            no_palm += 1
            continue
        box, kp, _score = det
        scale = FRAME / bp.INPUT
        centre, side, right, down = hl.crop_frame(np.asarray(box) * scale, kp * scale)
        crop = hl.crop_rotated(rgb, centre, side, right, down)
        with torch.no_grad():
            lm, presence, handed, _ = landmarker(torch.from_numpy(crop[None]).float().div(255))
        if float(presence) < hl.PRESENCE_THRESHOLD:
            no_presence += 1
            continue
        xyz = hl.decode(lm.numpy()[0], centre, side, right, down)
        lm_out.append(np.stack([xyz[:, 0] / FRAME, xyz[:, 1] / FRAME, xyz[:, 2] / FRAME], 1))  # normalised like MediaPipe's
        hd.append(float(handed))
        size.append((FRAME, FRAME))
        lab.append(LABELS.index(letter))
        if (i + 1) % 1000 == 0:
            print(f"{i + 1}/{n}: {len(lab)} hands, {no_palm} no palm, {no_presence} no presence, {time.time() - t0:.0f} s", file=sys.stderr)
    np.savez_compressed(args.out, landmarks=np.array(lm_out, np.float32), handedness=np.array(hd, np.float32),
                        size=np.array(size, np.int32), label=np.array(lab, np.int16), source=np.array([args.source] * len(lab)))
    counts = np.bincount(np.array(lab), minlength=len(LABELS)) if lab else np.zeros(1, int)
    print(f"{args.out}: {len(lab)} hands from {n} images ({no_palm} without a palm, {no_presence} failed presence); "
          f"per letter min {counts.min()} max {counts.max()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
