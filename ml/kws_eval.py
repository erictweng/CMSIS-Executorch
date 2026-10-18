#!/usr/bin/env python3
"""Score the ported DS-CNN (model/ds_cnn_kws.pt2) with our front end (ml/kws.py) on Speech
Commands v2 test clips: per-class and overall top-1, float and with the board's int8 input
quantization (the model itself runs in float here; the NPU's int8 weights are the a8w8 export).

    .venv/bin/python -I ml/kws_eval.py <speech_commands_test_dir> [--per-class 50] [--scale S --zp Z]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import kws  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("speech_commands")
    ap.add_argument("--per-class", type=int, default=50)
    ap.add_argument("--scale", type=float, default=None, help="int8 input scale (KWS_INPUT0_SCALE) to simulate")
    ap.add_argument("--zp", type=int, default=None)
    args = ap.parse_args()
    module = torch.export.load(HERE.parent / "model" / "ds_cnn_kws.pt2").module()
    sc = Path(args.speech_commands)
    rng = np.random.default_rng(0)
    total = correct = 0
    print(f"{'class':10s} {'n':>4s} {'top-1':>6s}  confusions")
    for k, label in enumerate(kws.LABELS):
        files = sorted(os.listdir(sc / label))
        pick = rng.choice(files, size=min(args.per_class, len(files)), replace=False)
        hits = 0
        conf: dict[str, int] = {}
        for f in pick:
            m = kws.mfcc(kws.read_wav(sc / label / f))
            if args.scale:
                m = (kws.quantize(m, args.scale, args.zp).astype(np.float32) - args.zp) * args.scale
            with torch.no_grad():
                logits = module(torch.from_numpy(m).reshape(1, 49, 10, 1))[0].numpy()
            top = int(np.argmax(logits))
            if top == k:
                hits += 1
            else:
                conf[kws.LABELS[top]] = conf.get(kws.LABELS[top], 0) + 1
        total += len(pick)
        correct += hits
        worst = ", ".join(f"{w} {n}" for w, n in sorted(conf.items(), key=lambda x: -x[1])[:3])
        print(f"{label:10s} {len(pick):4d} {100 * hits / len(pick):5.1f}%  {worst}")
    print(f"overall top-1 {100 * correct / total:.1f}% on {total} clips"
          + (f" (input quantized {args.scale} / {args.zp})" if args.scale else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
