# Fingerspell models for create_ai_layer.py: two methods in one ExecuTorch program.
#
#   detect    BlazePalm (MediaPipe palm_detection_lite, 192x192), model/blazepalm_192.pt2
#             (made by ml/blazepalm_export.py from Google's TFLite weights).
#             Input 1x192x192x3 NHWC float in [0, 1] (x / 255: MediaPipe's graph for the
#             192 models uses output_tensor_float_range 0..1; [-1, 1] was the 2019 256
#             model and puts the boxes off the hands). On the board the method takes the
#             int8 tensor (quantized_io; parameters in model_params.h). Outputs the raw
#             SSD head: regressors [1, 2016, 18] and score logits [1, 2016, 1]; sigmoid,
#             decode and NMS are CPU work (src/fs_detect.c, reference ml/blazepalm.py).
#   landmark  MediaPipe hand_landmark_lite (224x224), model/hand_landmark_224.pt2 (made by
#             ml/handlandmark_export.py). Input 1x224x224x3 NHWC float in [0, 1]; on the
#             board int8 (quantized_io). Outputs: landmarks (63 = 21 x (x, y, z) in input
#             pixels), presence (1, sigmoid), handedness (1, sigmoid, 1 = right), world
#             landmarks (63). The keypoint -> letter MLP runs on the CPU (src/fs_landmark.c)
#             with weights from get_params() (model/keypoint_mlp.npz).
#   kws       DS-CNN "small" keyword spotting (Arm ML-Zoo weights, Apache-2.0; ml/kws_export.py ->
#             model/ds_cnn_kws.pt2). Input 1x49x10x1 NHWC float: 49 frames of 10 MFCCs over 1 s
#             of 16 kHz audio (ml/kws.py, src/fs_voice.c); on the board int8 (quantized_io).
#             Output 12 logits: silence, unknown, yes, no, up, down, left, right, on, off, stop,
#             go; softmax on the CPU. documentation/voice-plan-status.md.
#   (classify, the MobileNetV2-0.5 pixel classifier of branch detect/blazepalm, is gone:
#    documentation/landmark-plan.md.)
#
# Calibration: detect on model/calib_detect_192.npy, landmark on model/calib_landmark_224.npy
# when present (camera crops, uint8 NHWC; collect them with tests/grab_crop.py), else on
# dataset crops from model/calib_128.npy (upscaled, with a note on stderr).
"""Fingerspell: palm detector + hand landmarks (+ keypoint MLP on the CPU), int8 for the Ethos-U85."""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
DETECT_SHAPE = (1, 192, 192, 3)
LANDMARK_SHAPE = (1, 224, 224, 3)
KWS_SHAPE = (1, 49, 10, 1)  # frames x MFCC coefficients, time-major
KWS_LABELS = "silence,unknown,yes,no,up,down,left,right,on,off,stop,go"
CLASSIFY_SHAPE = (1, 128, 128, 3)  # the retired pixel classifier (get_model(), for old tools)
DETECT_ANCHORS, DETECT_COORDS = 2016, 18
LANDMARKS = 21


@dataclass(frozen=True)
class Method:
    """One method of the program; the fields create_ai_layer.Method has."""

    name: str
    module: nn.Module
    example_inputs: tuple[torch.Tensor, ...]
    calibration: Callable[[], Iterable[tuple[torch.Tensor, ...]]] | None = None
    quantization: str = "a8w8"
    quantized_io: bool = False  # int8 in and out, parameters in model_params.h


def _load(name: str) -> nn.Module:
    # Self-contained exported graphs: no timm/onnx2torch needed in the starter's venv.
    return torch.export.load(HERE / name).module(check_guards=False)


def calib_detect(samples: int = 100) -> Iterable[tuple[torch.Tensor, ...]]:
    """Camera frames at 192x192 when collected, else dataset crops: float in [0, 1]."""
    path = HERE / "calib_detect_192.npy"
    if path.exists():
        data = np.load(path)  # N x 192 x 192 x 3 uint8
    else:
        print("[model] calib_detect_192.npy missing: calibrating detect on dataset crops", file=sys.stderr)
        crops = np.load(HERE / "calib_128.npy")  # N x 128 x 128 x 3 uint8
        data = np.stack([np.asarray(_resize(c, 192)) for c in crops[: samples]])
    n = min(samples, len(data))
    return [(torch.from_numpy(data[i : i + 1]).float().div(255),) for i in range(n)]


def calib_landmark(samples: int = 100) -> Iterable[tuple[torch.Tensor, ...]]:
    """Camera hand crops at 224x224 when collected, else dataset crops: float in [0, 1]."""
    path = HERE / "calib_landmark_224.npy"
    if path.exists():
        data = np.load(path)  # N x 224 x 224 x 3 uint8
    else:
        print("[model] calib_landmark_224.npy missing: calibrating landmark on dataset crops", file=sys.stderr)
        crops = np.load(HERE / "calib_128.npy")
        data = np.stack([np.asarray(_resize(c, 224)) for c in crops[: samples]])
    n = min(samples, len(data))
    return [(torch.from_numpy(data[i : i + 1]).float().div(255),) for i in range(n)]


def calib_kws(samples: int = 100) -> Iterable[tuple[torch.Tensor, ...]]:
    """MFCCs of Speech Commands clips (model/calib_kws.npy, N x 49 x 10 float32, ml/kws_export.py)."""
    data = np.load(HERE / "calib_kws.npy")
    n = min(samples, len(data))
    return [(torch.from_numpy(data[i : i + 1]).float().unsqueeze(-1),) for i in range(n)]


def calib_classify(samples: int = 300) -> Iterable[tuple[torch.Tensor, ...]]:
    """Real preprocessed training crops (float [0,1], NHWC), saved by ml/export.py."""
    data = np.load(HERE / "calib_128.npy")  # N x 128 x 128 x 3 uint8
    n = min(samples, len(data))
    return [(torch.from_numpy(data[i : i + 1]).float().div(255),) for i in range(n)]


def _resize(rgb: np.ndarray, size: int):
    from PIL import Image

    return Image.fromarray(rgb).resize((size, size), Image.BOX)


def get_methods(**params) -> list[Method]:
    calibration_samples = int(params.get("calibration_samples", 300))
    detect_x = torch.zeros(DETECT_SHAPE)
    return [
        # int8 activations: on 300 dataset images (150 direct, 150 pasted on backgrounds)
        # a8w8 agrees with the float model on hit/miss for 274, mean score difference 0.04,
        # box corners 2.5 px apart (median, 192 px input); a16w8: 293, 0.01, 0.7 px, but
        # its Vela scratch is 2.3 MB against a8w8's, which fits the SRAM0 temp pool.
        # quantized_io: the firmware writes the int8 input itself and decodes the int8 head
        # (DETECT_INPUT0_* / DETECT_OUTPUT<i>_* in model_params.h): the portable quantize and
        # dequantize operators took ~25 ms per frame on the Cortex-M55 for the 150k values.
        Method("detect", _load("blazepalm_192.pt2"), (detect_x,),
               lambda: calib_detect(min(calibration_samples, 100)), "a8w8", quantized_io=True),
        Method("landmark", _load("hand_landmark_224.pt2"), (torch.zeros(LANDMARK_SHAPE),),
               lambda: calib_landmark(min(calibration_samples, 100)), "a8w8", quantized_io=True),
        # Keyword spotting (voice reply): tiny, runs every 0.5 s from the same loop. The int8
        # input scale/zero point land in model_params.h as KWS_INPUT0_*; the firmware quantizes
        # its MFCCs with them (ML-Zoo's own int8 model used 1.084 / 100 for the same features).
        Method("kws", _load("ds_cnn_kws.pt2"), (torch.zeros(KWS_SHAPE),),
               lambda: calib_kws(min(calibration_samples, 100)), "a8w8", quantized_io=True),
    ]


def get_params() -> dict:
    """Constants the firmware needs; create_ai_layer.py writes them to model_params.h.
    The keypoint -> letter MLP (ml/keypoint_train.py -> model/keypoint_mlp.npz) comes along as
    float arrays: FS_MLP_W1[hidden][63], FS_MLP_B1[hidden], FS_MLP_W2[24][hidden], FS_MLP_B2[24]."""
    mlp = np.load(HERE / "keypoint_mlp.npz")
    hidden = int(mlp["hidden"])
    assert mlp["W1"].shape == (hidden, LANDMARKS * 3) and mlp["W2"].shape == (24, hidden), "keypoint_mlp.npz shapes"
    assert str(mlp["labels"]) == "ABCDEFGHIKLMNOPQRSTUVWXY", "MLP label order must be FS_LABELS"
    return {
        "FS_DETECT_SIZE": DETECT_SHAPE[1],
        "FS_DETECT_ANCHORS": DETECT_ANCHORS,
        "FS_DETECT_COORDS": DETECT_COORDS,
        "FS_LANDMARK_SIZE": LANDMARK_SHAPE[1],
        "FS_LANDMARKS": LANDMARKS,
        "FS_CLASSES": 24,
        "FS_MLP_HIDDEN": hidden,
        "FS_MLP_W1": mlp["W1"],
        "FS_MLP_B1": mlp["b1"],
        "FS_MLP_W2": mlp["W2"],
        "FS_MLP_B2": mlp["b2"],
        "FS_VOICE_FRAMES": KWS_SHAPE[1],
        "FS_VOICE_COEFFS": KWS_SHAPE[2],
        "FS_VOICE_CLASSES": len(KWS_LABELS.split(",")),
        "FS_VOICE_LABELS": KWS_LABELS,
    }


# The single-model contract, kept for tools that still call it.
def get_model(input_shape: tuple[int, ...] = CLASSIFY_SHAPE) -> nn.Module:
    if tuple(input_shape) != CLASSIFY_SHAPE:
        raise ValueError(f"the classifier was trained for {CLASSIFY_SHAPE}, got {input_shape}")
    return _load("fingerspell_letters.pt2")


def get_calibration_inputs(input_shape=CLASSIFY_SHAPE, calibration_samples: int = 300) -> list[torch.Tensor]:
    return [x for (x,) in calib_classify(calibration_samples)]
