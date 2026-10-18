"""Accuracy of a keypoint MLP (model/keypoint_mlp.npz) on landmark sets from ml/keypoint_dataset*.py.

    .venv/bin/python -I ml/keypoint_eval.py model/keypoint_mlp.npz <set.npz> [more.npz ...]

Prints top-1 accuracy and the mean top-1 probability per set, and the most confused pairs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import handlandmark as hl  # noqa: E402
from keypoint_train import load  # noqa: E402


def main() -> int:
    w = np.load(sys.argv[1])
    weights = {k: w[k] for k in ("W1", "b1", "W2", "b2")}
    for path in sys.argv[2:]:
        X, y, src = load([path])
        probs = np.array([hl.mlp(x, weights) for x in X])
        pred = probs.argmax(1)
        conf = np.zeros((24, 24), int)
        for t, p in zip(y, pred):
            conf[t, p] += 1
        np.fill_diagonal(conf, 0)
        worst = np.dstack(np.unravel_index(np.argsort(-conf.ravel())[:5], conf.shape))[0]
        print(f"{Path(path).name}: {len(y)} samples, top-1 {np.mean(pred == y):.4f}, mean top-1 prob {probs.max(1).mean():.3f}, "
              f"correct >= 0.6: {np.mean((pred == y) & (probs.max(1) >= 0.6)):.3f}; confused: "
              + ", ".join(f"{hl.LABELS[t]}->{hl.LABELS[p]} x{conf[t, p]}" for t, p in worst if conf[t, p]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
