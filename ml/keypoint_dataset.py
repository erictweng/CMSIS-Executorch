"""Hand landmarks for every image of a letter dataset, extracted with MediaPipe's HandLandmarker
(Tasks API), as training data for the keypoint -> letter MLP (ml/keypoint_train.py).

    <tools-venv>/bin/python -I ml/keypoint_dataset.py <images.parquet> <hand_landmarker.task> \
        <out.npz> --source marxulia [--limit N]

The parquet has an `image` struct (bytes) and an integer `label` 0..25 = A..Z (HF image
datasets); J and Z are dropped (they move). Saved: `landmarks` (N, 21, 3) in normalised image
coordinates (x, y in [0, 1] of the image, z in the same units as x), `handedness` (N,) 1.0 =
right hand (MediaPipe's score for "Right"), `size` (N, 2) image width/height in pixels,
`label` (N,) index into FS_LABELS "ABCDEFGHIKLMNOPQRSTUVWXY", `source` (N,) str.
Images where MediaPipe finds no hand are skipped and counted.

Runs in the tools venv (mediapipe, pyarrow, Pillow), not the project venv.
"""
from __future__ import annotations

import argparse
import io
import sys
import time

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

LABELS = "ABCDEFGHIKLMNOPQRSTUVWXY"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("parquet")
    ap.add_argument("task", help="MediaPipe hand_landmarker.task")
    ap.add_argument("out")
    ap.add_argument("--source", default="dataset")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min-confidence", type=float, default=0.5)
    args = ap.parse_args()

    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    detector = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=args.task), running_mode=vision.RunningMode.IMAGE,
        num_hands=1, min_hand_detection_confidence=args.min_confidence))

    tbl = pq.read_table(args.parquet)
    images, labels = tbl.column("image").to_pylist(), tbl.column("label").to_pylist()
    n = len(images) if not args.limit else min(args.limit, len(images))
    lm, hd, size, lab = [], [], [], []
    skipped, t0 = 0, time.time()
    for i in range(n):
        letter = chr(65 + labels[i])
        if letter not in LABELS:
            continue
        rgb = np.asarray(Image.open(io.BytesIO(images[i]["bytes"])).convert("RGB"))
        res = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
        if not res.hand_landmarks:
            skipped += 1
            continue
        pts = res.hand_landmarks[0]
        lm.append([(p.x, p.y, p.z) for p in pts])
        cat = res.handedness[0][0]
        hd.append(cat.score if cat.category_name == "Right" else 1.0 - cat.score)
        size.append((rgb.shape[1], rgb.shape[0]))
        lab.append(LABELS.index(letter))
        if (i + 1) % 1000 == 0:
            print(f"{i + 1}/{n}: {len(lab)} hands, {skipped} without, {time.time() - t0:.0f} s", file=sys.stderr)
    np.savez_compressed(args.out, landmarks=np.array(lm, np.float32), handedness=np.array(hd, np.float32),
                        size=np.array(size, np.int32), label=np.array(lab, np.int16),
                        source=np.array([args.source] * len(lab)))
    counts = np.bincount(np.array(lab), minlength=len(LABELS))
    print(f"{args.out}: {len(lab)} hands from {n} images ({skipped} without a hand); per letter min {counts.min()} max {counts.max()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
