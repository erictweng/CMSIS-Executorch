# Fingerspell letter model — drop-in replacement for CMSIS-Executorch/model/model.py
# (Arm-Examples/CMSIS-Executorch@hackathon). Copy this file AND fingerspell_letters.pt2 +
# calib_128.npy into the starter's model/ folder, then set in cmsis-executorch.csolution.yml:
#     mlops: model: input-shape: 1x128x128x3   calibration-samples: 300
#     name: FingerspellLetters
# and run "Create AI layer".
#
# Input contract (C++ side): 1x128x128x3 NHWC, RGB, float = uint8 / 255. Output: 24 logits,
# label order in labels.txt (A..Y without J). The model normalizes internally.
"""Fingerspell 24-letter classifier, quantized to int8 for Ethos-U85 by create_ai_layer.py."""

from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
INPUT_SHAPE = (1, 128, 128, 3)


def get_model(input_shape: tuple[int, ...] = INPUT_SHAPE) -> torch.nn.Module:
    if tuple(input_shape) != INPUT_SHAPE:
        raise ValueError(f"this model was trained for {INPUT_SHAPE}, got {input_shape}")
    # A self-contained exported graph: no timm/torchvision needed in the starter's venv.
    return torch.export.load(HERE / "fingerspell_letters.pt2").module(check_guards=False)


def get_calibration_inputs(
    input_shape: tuple[int, ...] = INPUT_SHAPE, calibration_samples: int = 300
) -> list[torch.Tensor]:
    """Real preprocessed training crops (float [0,1], NHWC), saved by ml/export.py."""
    data = np.load(HERE / "calib_128.npy")  # N x 128 x 128 x 3 uint8
    n = min(calibration_samples, len(data))
    return [torch.from_numpy(data[i : i + 1]).float().div(255) for i in range(n)]
