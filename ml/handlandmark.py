"""Hand landmarks (MediaPipe hand_landmark_lite, 224x224) reference pieces, numpy only.

What the board implements in src/fs_landmark.c, in the same order:

    crop        the palm detection (box + wrist / middle-finger keypoints, ml/blazepalm.py)
                -> a square crop of CROP_SCALE x the palm size, centred CROP_SHIFT x the
                palm size from the palm centre toward the fingers, rotated so the fingers
                point up (MediaPipe's rect_transformation + the rotation we skipped for the
                pixel classifier), resampled to 224x224
    decode      the model's 63 outputs -> 21 x (x, y, z) in crop pixels -> frame pixels
                (rotated back); presence and handedness are sigmoids already
    features    wrist-centred, scaled by the wrist -> middle-MCP distance, left hands
                mirrored to right; NOT rotation-normalised (P/K, Q/G differ by orientation)
    letters     MLP 63 -> HIDDEN -> 24 (ReLU), softmax; weights from model/keypoint_mlp.npz

Landmark indices: 0 wrist, 1-4 thumb, 5-8 index, 9-12 middle, 13-16 ring, 17-20 pinky
(MCP, PIP, DIP, tip; thumb CMC, MCP, IP, tip).
"""

from __future__ import annotations

import numpy as np

INPUT = 224
LANDMARKS = 21
WRIST, MIDDLE_MCP = 0, 9
CROP_SHIFT = 0.5    # x palm size, from the palm centre toward the fingers
CROP_SCALE = 2.6    # crop side = CROP_SCALE x palm size
PRESENCE_THRESHOLD = 0.5
LABELS = "ABCDEFGHIKLMNOPQRSTUVWXY"


def crop_frame(palm_box, palm_keypoints, kp_wrist: int = 0, kp_middle_mcp: int = 2):
    """The rotated crop's geometry from a palm detection (frame pixels): centre (2,), side
    (float), and the unit vectors `right` and `down` of the crop in frame coordinates.
    palm_keypoints are the detector's 7 (wrist = 0, middle-finger MCP = 2)."""
    x0, y0, x1, y1 = (float(v) for v in palm_box)
    size = max(x1 - x0, y1 - y0)
    kp = np.asarray(palm_keypoints, float)
    up = kp[kp_middle_mcp] - kp[kp_wrist]
    norm = np.linalg.norm(up)
    up = up / norm if norm > 1e-3 else np.array([0.0, -1.0])
    centre = np.array([(x0 + x1) / 2, (y0 + y1) / 2]) + CROP_SHIFT * size * up
    right = np.array([-up[1], up[0]])
    return centre, size * CROP_SCALE, right, -up


def crop_rotated(rgb: np.ndarray, centre, side: float, right, down, size: int = INPUT) -> np.ndarray:
    """Bilinear resample of the rotated square crop from an HxWx3 uint8 frame; black off-frame."""
    h, w = rgb.shape[:2]
    s = side / size
    u = (np.arange(size) + 0.5 - size / 2) * s
    fx = centre[0] + u[None, :] * right[0] + u[:, None] * down[0]
    fy = centre[1] + u[None, :] * right[1] + u[:, None] * down[1]
    x0 = np.floor(fx - 0.5).astype(int)
    y0 = np.floor(fy - 0.5).astype(int)
    ax = (fx - 0.5 - x0)[..., None]
    ay = (fy - 0.5 - y0)[..., None]

    def at(xi, yi):
        inside = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        out = np.zeros(xi.shape + (3,), float)
        out[inside] = rgb[yi[inside], xi[inside]]
        return out

    v = (at(x0, y0) * (1 - ax) * (1 - ay) + at(x0 + 1, y0) * ax * (1 - ay)
         + at(x0, y0 + 1) * (1 - ax) * ay + at(x0 + 1, y0 + 1) * ax * ay)
    return np.clip(np.round(v), 0, 255).astype(np.uint8)


def decode(landmarks63: np.ndarray, centre, side: float, right, down, size: int = INPUT) -> np.ndarray:
    """The model's 63 values (crop pixels) -> (21, 3) in frame pixels: x, y rotated back, z scaled."""
    s = side / size
    lm = np.asarray(landmarks63, float).reshape(LANDMARKS, 3)
    u, v = lm[:, 0] - size / 2, lm[:, 1] - size / 2
    out = np.empty_like(lm)
    out[:, 0] = centre[0] + (u * right[0] + v * down[0]) * s
    out[:, 1] = centre[1] + (u * right[1] + v * down[1]) * s
    out[:, 2] = lm[:, 2] * s
    return out


def features(landmarks: np.ndarray, right_hand: bool) -> np.ndarray:
    """(21, 3) landmarks in any consistent unit -> 63 MLP inputs: wrist at the origin, the
    wrist -> middle-MCP distance = 1, left hands mirrored (x -> -x)."""
    lm = np.asarray(landmarks, float).reshape(LANDMARKS, 3)
    p = lm - lm[WRIST]
    unit = np.linalg.norm(p[MIDDLE_MCP, :2])
    p = p / (unit if unit > 1e-6 else 1.0)
    if not right_hand:
        p[:, 0] = -p[:, 0]
    return p.reshape(-1).astype(np.float32)


def mlp(features63: np.ndarray, weights) -> np.ndarray:
    """Letter probabilities (24,) from the MLP weights (dict or npz with W1 b1 W2 b2)."""
    h = np.maximum(features63 @ weights["W1"].T + weights["b1"], 0.0)
    z = h @ weights["W2"].T + weights["b2"]
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()
