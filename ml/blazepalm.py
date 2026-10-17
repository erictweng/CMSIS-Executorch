"""BlazePalm (MediaPipe palm_detection_lite, 192x192) reference pieces, numpy only.

The anchor table, the SSD decode, the palm box -> hand box rule and the no-rotation
crop the board implements in src/fs_detect.c. Constants are MediaPipe's
palm_detection_lite graph (ssd_anchors_calculator, tensors_to_detections,
detections_to_rects, rect_transformation):

    anchors   num_layers 4, strides 8/16/16/16, min_scale 0.1484375, max_scale 0.75,
              anchor offset 0.5, aspect ratio 1.0, fixed_anchor_size -> 2016 anchors
    decode    scores: sigmoid (clipped at +-100), boxes: centre + offset / 192 * anchor,
              18 values per anchor: cx cy w h, then 7 keypoints (x y)
    hand box  shift the palm box 0.5 x its height toward the fingers, scale 2.6x,
              make it square (MediaPipe also rotates by the wrist->middle-finger angle;
              the board skips the rotation)

The 192 model's two output tensors: regressors [1, 2016, 18] and scores [1, 2016, 1].
"""

from __future__ import annotations

import numpy as np

INPUT = 192
NUM_ANCHORS = 2016
NUM_COORDS = 18
NUM_KEYPOINTS = 7
STRIDES = (8, 16, 16, 16)
MIN_SCALE, MAX_SCALE = 0.1484375, 0.75
SCORE_THRESHOLD = 0.5
KP_WRIST, KP_MIDDLE_MCP = 0, 2
BOX_SHIFT_Y = -0.5   # toward the fingers (up, in palm-rect coordinates)
BOX_SCALE = 2.6


def calculate_scale(min_scale: float, max_scale: float, stride_index: int, num_strides: int) -> float:
    if num_strides == 1:
        return (min_scale + max_scale) * 0.5
    return min_scale + (max_scale - min_scale) * stride_index / (num_strides - 1.0)


def anchors() -> np.ndarray:
    """(2016, 4) rows of x_center, y_center, w, h in [0, 1] (w = h = 1 with fixed_anchor_size)."""
    out = []
    layer = 0
    while layer < len(STRIDES):
        scales = []
        last = layer
        while last < len(STRIDES) and STRIDES[last] == STRIDES[layer]:
            scale = calculate_scale(MIN_SCALE, MAX_SCALE, last, len(STRIDES))
            scales.append(scale)  # aspect ratio 1.0
            # interpolated_scale_aspect_ratio 1.0: one more anchor between this scale and the next
            if last == len(STRIDES) - 1:
                scales.append(np.sqrt(scale * 1.0))
            else:
                scales.append(np.sqrt(scale * calculate_scale(MIN_SCALE, MAX_SCALE, last + 1, len(STRIDES))))
            last += 1
        stride = STRIDES[layer]
        feature = int(np.ceil(INPUT / stride))
        for y in range(feature):
            for x in range(feature):
                for _ in scales:
                    out.append(((x + 0.5) / feature, (y + 0.5) / feature, 1.0, 1.0))
        layer = last
    a = np.array(out, dtype=np.float32)
    assert a.shape == (NUM_ANCHORS, 4), a.shape
    return a


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -100.0, 100.0)))


def decode(regressors: np.ndarray, scores: np.ndarray, anchor_table: np.ndarray | None = None):
    """Boxes (N, 4) as x0 y0 x1 y1 in input pixels, keypoints (N, 7, 2) in pixels, scores (N,)."""
    a = anchors() if anchor_table is None else anchor_table
    r = regressors.reshape(NUM_ANCHORS, NUM_COORDS).astype(np.float32)
    s = sigmoid(scores.reshape(NUM_ANCHORS).astype(np.float32))
    cx = r[:, 0] / INPUT * a[:, 2] + a[:, 0]
    cy = r[:, 1] / INPUT * a[:, 3] + a[:, 1]
    w = r[:, 2] / INPUT * a[:, 2]
    h = r[:, 3] / INPUT * a[:, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1) * INPUT
    kp = r[:, 4:].reshape(NUM_ANCHORS, NUM_KEYPOINTS, 2) / INPUT
    kp[:, :, 0] = kp[:, :, 0] * a[:, 2:3] + a[:, 0:1]
    kp[:, :, 1] = kp[:, :, 1] * a[:, 3:4] + a[:, 1:2]
    return boxes, kp * INPUT, s


def best_detection(regressors, scores, threshold: float = SCORE_THRESHOLD):
    """The highest-scoring palm (one hand for fingerspelling) or None: (box, keypoints, score)."""
    boxes, kp, s = decode(regressors, scores)
    i = int(np.argmax(s))
    if s[i] < threshold:
        return None
    return boxes[i], kp[i], float(s[i])


def hand_box(palm_box: np.ndarray, frame: int, shift_y: float = BOX_SHIFT_Y, scale: float = BOX_SCALE):
    """Square hand box (x0, y0, x1, y1) in frame pixels from a palm box in the same pixels:
    shift by shift_y x height (negative = up, toward the fingers), scale by `scale`, square
    on the longer side; may run off the frame (the crop pads with black)."""
    x0, y0, x1, y1 = (float(v) for v in palm_box)
    w, h = x1 - x0, y1 - y0
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2 + shift_y * h
    side = max(w, h) * scale
    return cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2


def crop_square(rgb: np.ndarray, box, size: int) -> np.ndarray:
    """Area-averaged crop of `box` (float pixels) from an HxWx3 uint8 frame, resized to
    size x size, black where the box runs off the frame. Integer box edges and a
    nearest-neighbour fallback keep it simple: this is the reference, not the board code."""
    from PIL import Image

    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    side = max(x1 - x0, y1 - y0)
    canvas = np.zeros((side, side, 3), np.uint8)
    sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x0 + side, w), min(y0 + side, h)
    if sx1 > sx0 and sy1 > sy0:
        canvas[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = rgb[sy0:sy1, sx0:sx1]
    return np.asarray(Image.fromarray(canvas).resize((size, size), Image.BOX))


def resize_area(rgb: np.ndarray, size: int) -> np.ndarray:
    """Box-filter resize of a square HxWx3 uint8 frame to size x size (the detector input)."""
    from PIL import Image

    return np.asarray(Image.fromarray(rgb).resize((size, size), Image.BOX))
