"""Train the keypoint -> letter MLP on landmarks from ml/keypoint_dataset.py.

    .venv/bin/python -I ml/keypoint_train.py <keypoints.npz> [more.npz ...] --out model/keypoint_mlp.npz

Features: ml/handlandmark.features() (wrist-centred, size-normalised, left mirrored to right),
from MediaPipe's normalised coordinates scaled by the image size. Split: 90/10 stratified per
source (seed 0). Augmentation: rotation +-15 deg about the wrist, scale 0.9-1.1, Gaussian noise
0.02 (in hand-size units), to cover the board's crop jitter. Reports val accuracy per source
and the most confused pairs, writes W1 b1 W2 b2 (float32) + the feature config.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import handlandmark as hl  # noqa: E402

HIDDEN = 128  # default (95.8 % val on Marxulia with --epochs 150 --rotation 10); --hidden overrides


def load(paths):
    X, y, src = [], [], []
    for p in paths:
        d = np.load(p, allow_pickle=False)
        lm = d["landmarks"].astype(np.float64)  # (N, 21, 3) normalised image coordinates
        size = d["size"].astype(np.float64)
        lm[:, :, 0] *= size[:, None, 0]
        lm[:, :, 1] *= size[:, None, 1]
        lm[:, :, 2] *= size[:, None, 0]
        for i in range(len(lm)):
            X.append(hl.features(lm[i], bool(d["handedness"][i] >= 0.5)))
        y.extend(d["label"].tolist())
        src.extend(d["source"].tolist())
    return np.array(X, np.float32), np.array(y, np.int64), np.array(src)


def augment(x: torch.Tensor, rng: torch.Generator, rotation: float = 15.0) -> torch.Tensor:
    n = x.shape[0]
    p = x.view(n, hl.LANDMARKS, 3).clone()
    a = (torch.rand(n, generator=rng) * 2 - 1) * (rotation * np.pi / 180)
    c, s = torch.cos(a)[:, None], torch.sin(a)[:, None]
    px, py = p[:, :, 0].clone(), p[:, :, 1].clone()
    p[:, :, 0], p[:, :, 1] = c * px - s * py, s * px + c * py
    p *= (0.9 + 0.2 * torch.rand(n, 1, 1, generator=rng))
    p += 0.02 * torch.randn(p.shape, generator=rng)
    return p.view(n, -1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz", nargs="+")
    ap.add_argument("--out", default="model/keypoint_mlp.npz")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--hidden", type=int, default=HIDDEN)
    ap.add_argument("--rotation", type=float, default=10.0, help="augmentation: +- degrees about the wrist")
    args = ap.parse_args()
    X, y, src = load(args.npz)
    rng = np.random.default_rng(0)
    val = np.zeros(len(y), bool)
    for s in np.unique(src):
        for c in np.unique(y):
            idx = np.where((src == s) & (y == c))[0]
            val[rng.choice(idx, max(1, len(idx) // 10), replace=False)] = True
    Xt, yt = torch.from_numpy(X[~val]), torch.from_numpy(y[~val])
    Xv, yv = torch.from_numpy(X[val]), torch.from_numpy(y[val])
    print(f"{len(y)} samples from {sorted(set(src.tolist()))}: train {len(yt)}, val {len(yv)}")
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    hidden = args.hidden
    net = nn.Sequential(nn.Linear(63, hidden), nn.ReLU(), nn.Linear(hidden, len(hl.LABELS)))
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    for epoch in range(args.epochs):
        net.train()
        perm = torch.randperm(len(yt), generator=g)
        for i in range(0, len(yt), 256):
            b = perm[i:i + 256]
            loss = nn.functional.cross_entropy(net(augment(Xt[b], g, args.rotation)), yt[b], label_smoothing=0.05)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if (epoch + 1) % 20 == 0 or epoch == args.epochs - 1:
            net.eval()
            with torch.no_grad():
                acc = (net(Xv).argmax(1) == yv).float().mean().item()
            print(f"epoch {epoch + 1}: loss {loss.item():.3f}, val acc {acc:.4f}")
    net.eval()
    with torch.no_grad():
        pred = net(Xv).argmax(1).numpy()
    for s in np.unique(src):
        m = src[val] == s
        print(f"val accuracy {s}: {np.mean(pred[m] == yv.numpy()[m]):.4f} ({m.sum()} samples)")
    conf = np.zeros((24, 24), int)
    for t, p in zip(yv.numpy(), pred): conf[t, p] += 1
    np.fill_diagonal(conf, 0)
    worst = np.dstack(np.unravel_index(np.argsort(-conf.ravel())[:6], conf.shape))[0]
    print("most confused (true -> predicted):", ", ".join(f"{hl.LABELS[t]}->{hl.LABELS[p]} x{conf[t, p]}" for t, p in worst if conf[t, p]))
    W1, b1 = net[0].weight.detach().numpy(), net[0].bias.detach().numpy()
    W2, b2 = net[2].weight.detach().numpy(), net[2].bias.detach().numpy()
    np.savez(args.out, W1=W1, b1=b1, W2=W2, b2=b2, hidden=hidden, labels=hl.LABELS)
    # The numpy reference must agree with torch on the val set.
    w = dict(W1=W1, b1=b1, W2=W2, b2=b2)
    ref = np.array([hl.mlp(x, w).argmax() for x in X[val][:200]])
    assert np.array_equal(ref, pred[:200]), "ml/handlandmark.mlp disagrees with torch"
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
