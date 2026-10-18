#!/usr/bin/env python3
"""Build model/ds_cnn_kws.pt2 (the keyword-spotting DS-CNN for the "kws" method of model/model.py)
from the ML-Zoo weights that ml/kws_tflite_to_npz.py extracted, and model/calib_kws.npy (MFCCs of
Speech Commands clips for the quantizer's calibration).

    .venv/bin/python -I ml/kws_export.py <ds_cnn_weights.npz> <speech_commands_dir> [--calib 100]

The module reproduces the TFLite graph exactly: TF "SAME" padding on the 10x4 stride-2 stem
(4 rows above, 5 below, 1 column each side), four depthwise-separable 3x3 blocks, global average
pool, the 12-way linear layer. Softmax is left to the CPU (12 values; keeps the NPU output a
plain int8 logit vector). Input is NHWC 1x49x10x1 in MFCC units (no normalisation), the layout the
firmware writes; the convolutions run NCHW inside. The port is checked against the TFLite
interpreter's softmax on the reference clips stored in the npz (max abs diff printed, must be
< 1e-3). Trained weights: Arm ML-Zoo ds_cnn_small, Apache-2.0.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import kws  # noqa: E402

MODEL_DIR = HERE.parent / "model"


class DSCNN(nn.Module):
    def __init__(self, classes: int = 12, channels: int = 64) -> None:
        super().__init__()
        self.pad = nn.ZeroPad2d((1, 1, 4, 5))  # TF SAME for a 10x4 kernel, stride 2, on 49x10
        self.stem = nn.Conv2d(1, channels, (10, 4), stride=2)
        self.blocks = nn.ModuleList()
        for _ in range(4):
            self.blocks.append(nn.ModuleDict({
                "dw": nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
                "pw": nn.Conv2d(channels, channels, 1),
            }))
        self.pool = nn.AdaptiveAvgPool2d(1)  # what executorch's own Ethos-U DS-CNN test uses
        self.fc = nn.Linear(channels, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: 1 x 49 x 10 x 1 (NHWC)
        x = x.permute(0, 3, 1, 2)
        x = torch.relu(self.stem(self.pad(x)))
        for b in self.blocks:
            x = torch.relu(b["dw"](x))
            x = torch.relu(b["pw"](x))
        x = self.pool(x).flatten(1)
        return self.fc(x)


def load_weights(model: DSCNN, w: dict) -> None:
    def t(a):
        return torch.from_numpy(np.ascontiguousarray(a)).float()
    model.stem.weight.data = t(w["stem_w"]).permute(0, 3, 1, 2)  # OHWI -> OIHW
    model.stem.bias.data = t(w["stem_b"])
    for i, b in enumerate(model.blocks):
        b["dw"].weight.data = t(w[f"dw{i}_w"]).permute(3, 0, 1, 2)  # 1HWC -> C1HW
        b["dw"].bias.data = t(w[f"dw{i}_b"])
        b["pw"].weight.data = t(w[f"pw{i}_w"]).permute(0, 3, 1, 2)
        b["pw"].bias.data = t(w[f"pw{i}_b"])
    model.fc.weight.data = t(w["fc_w"])
    model.fc.bias.data = t(w["fc_b"])


def calibration_clips(sc_dir: Path, n: int, seed: int = 0) -> np.ndarray:
    """n clips spread over the 12 classes -> n x 49 x 10 MFCCs."""
    rng = np.random.default_rng(seed)
    per = -(-n // len(kws.LABELS))
    feats = []
    for label in kws.LABELS:
        files = sorted(os.listdir(sc_dir / label))
        for f in rng.choice(files, size=min(per, len(files)), replace=False):
            feats.append(kws.mfcc(kws.read_wav(sc_dir / label / f)))
    return np.stack(feats[:n]).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("weights")
    ap.add_argument("speech_commands")
    ap.add_argument("--calib", type=int, default=100)
    args = ap.parse_args()
    w = np.load(args.weights)
    model = DSCNN().eval()
    load_weights(model, w)

    ref = np.load(HERE.parent / "tests" / "fixtures" / "kws_ref.npz")
    with torch.no_grad():
        logits = model(torch.from_numpy(ref["mfcc"]).unsqueeze(-1)).numpy()
    probs = np.stack([kws.softmax(l) for l in logits])
    diff = np.abs(probs - w["ref_probs"]).max()
    agree = int((probs.argmax(1) == w["ref_probs"].argmax(1)).sum())
    print(f"port vs TFLite interpreter on {len(probs)} reference clips: max abs prob diff {diff:.2e}, "
          f"top-1 agreement {agree}/{len(probs)}")
    if diff > 1e-3:
        print("FAILED: the port does not reproduce the TFLite model")
        return 1

    example = (torch.zeros(1, kws.FRAMES, kws.COEFFS, 1),)
    ep = torch.export.export(model, example)
    MODEL_DIR.mkdir(exist_ok=True)
    torch.export.save(ep, MODEL_DIR / "ds_cnn_kws.pt2")
    calib = calibration_clips(Path(args.speech_commands), args.calib)
    np.save(MODEL_DIR / "calib_kws.npy", calib)
    print(f"wrote model/ds_cnn_kws.pt2 and model/calib_kws.npy {calib.shape} "
          f"(range {calib.min():.1f}..{calib.max():.1f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
