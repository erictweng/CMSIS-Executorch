"""Hand landmarks (src/fs_landmark.c + ml/handlandmark.py): host checks against the numpy
reference and the TFLite fixtures, the exported graph, the generated parameters.

    FS_SKIP_BUILD=1 .venv/bin/python -m unittest discover -s tests -p test_landmark.py -v

Board behaviour (letters, no-hand, fps, warm reset) is covered by tests/test_detector.py's
OnBoard class, whose live-line regexes include the landmark fields.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml"))
import handlandmark as hl  # noqa: E402
import fs_tools as fs  # noqa: E402

REPO = fs.REPO
SRC = REPO / "src"
FIXTURES = REPO / "tests" / "fixtures"
SOURCES = [SRC / "fs_detect.c", SRC / "fs_landmark.c", REPO / "tests" / "host" / "test_fs_landmark.c"]
CFLAGS = ["-std=c99", "-Wall", "-Wextra", "-Werror", "-pedantic", "-O1", "-I", str(SRC)]
FRAME, INPUT = 320, 224
QIN = (1 / 255, -128)
QLM = (1.18, -62)       # like the generated LANDMARK_OUTPUT0_* (1.18 px per step)
QSIG = (1 / 255, -128)  # presence and handedness


def host_cc() -> str | None:
    for cc in (os.environ.get("FS_CC"), "clang", "cc", "gcc"):
        if cc and shutil.which(cc):
            return cc
    return None


def scalar(a) -> float:
    """A fixture value stored as a 1-element array."""
    return float(np.asarray(a, dtype=np.float32).ravel()[0])


def quantize(v, q):
    return np.clip(np.round(np.asarray(v, float) / q[0]) + q[1], -128, 127).astype(np.int8)


def dequantize(q, p):
    return (q.astype(np.float32) - p[1]) * p[0]


def rgb565(rgb: np.ndarray) -> np.ndarray:
    r, g, b = (rgb[..., i].astype(np.uint16) for i in range(3))
    return ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3)


def rgb565_to_888(v: np.ndarray) -> np.ndarray:
    r5, g6, b5 = (v >> 11) & 0x1F, (v >> 5) & 0x3F, v & 0x1F
    return np.stack([(r5 << 3) | (r5 >> 2), (g6 << 2) | (g6 >> 4), (b5 << 3) | (b5 >> 2)], -1).astype(np.uint8)


@unittest.skipUnless(host_cc(), "no host C compiler (clang/cc/gcc) on PATH")
class LandmarkHost(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.exe = Path(cls.tmp.name) / "test_fs_landmark"
        cls.cc_result = subprocess.run([host_cc(), *CFLAGS, *map(str, SOURCES), "-lm", "-o", str(cls.exe)],
                                       capture_output=True, text=True)
        cls.ref = np.load(FIXTURES / "landmark_ref.npz")
        cls.names = [str(n) for n in cls.ref["names"]]
        cls.mlp = np.load(REPO / "model" / "keypoint_mlp.npz")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_mode(self, *args) -> str:
        self.assertEqual(self.cc_result.returncode, 0, self.cc_result.stderr)
        r = subprocess.run([str(self.exe), *map(str, args)], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        return r.stdout

    def test_compiles_as_strict_c99_without_warnings(self):
        self.assertEqual(self.cc_result.returncode, 0, self.cc_result.stderr)
        self.assertEqual(self.cc_result.stderr.strip(), "")

    def frame(self, seed: int):
        rng = np.random.default_rng(seed)
        rgb = rng.integers(0, 256, (FRAME, FRAME, 3), dtype=np.uint8)
        y, x = np.mgrid[0:FRAME, 0:FRAME]
        rgb[(x // 32 + y // 32) % 2 == 0] = (210, 120, 60)
        v = rgb565(rgb)
        path = Path(self.tmp.name) / f"frame{seed}.bin"
        path.write_bytes(v.astype("<u2").tobytes())
        return path, rgb565_to_888(v)

    def test_rotated_crop_matches_the_reference(self):
        path, img = self.frame(0)
        for centre, side, up in (((160, 160), 200.0, (0, -1)), ((120, 200), 150.0, (0.6, -0.8)), ((40, 300), 300.0, (1, 0))):
            with self.subTest(centre=centre, up=up):
                up = np.array(up, float) / np.linalg.norm(up)
                right, down = np.array([-up[1], up[0]]), -up
                out = Path(self.tmp.name) / "crop.bin"
                self.run_mode("crop", path, *centre, side, *up, *QIN, out)
                got = np.frombuffer(out.read_bytes(), np.int8).reshape(INPUT, INPUT, 3).astype(int)
                want = quantize(hl.crop_rotated(img, np.array(centre, float), side, right, down) / 255.0, QIN).astype(int)
                self.assertLessEqual(np.abs(got - want).max(), 2)
                self.assertGreater(np.mean(np.abs(got - want) <= 1), 0.99)

    def test_decode_and_features_match_the_reference(self):
        i = self.names.index("hand")
        lm = self.ref["landmarks"][i].astype(np.float32)
        presence, hand = scalar(self.ref["presence"][i]), scalar(self.ref["handedness"][i])
        lm_path = Path(self.tmp.name) / "lm.txt"
        lm_path.write_text(" ".join(f"{v:.5f}" for v in lm))
        centre, side, up = np.array([150.0, 170.0]), 180.0, np.array([0.6, -0.8])
        out = self.run_mode("decode", lm_path, presence, hand, *QLM, *QSIG, *QSIG, *centre, side, *up).splitlines()
        got_xyz = np.array(out[0].split(), float).reshape(21, 3)
        got_p, got_h = (float(v) for v in out[1].split())
        got_f = np.array(out[2].split(), float)
        qlm = dequantize(quantize(lm, QLM), QLM)
        right, down = np.array([-up[1], up[0]]), -up
        want_xyz = hl.decode(qlm, centre, side, right, down)
        np.testing.assert_allclose(got_xyz, want_xyz, atol=1e-2)
        self.assertAlmostEqual(got_p, float(dequantize(quantize(presence, QSIG), QSIG)), places=3)
        self.assertAlmostEqual(got_h, float(dequantize(quantize(hand, QSIG), QSIG)), places=3)
        np.testing.assert_allclose(got_f, hl.features(want_xyz, got_h >= 0.5), atol=1e-4)
        # The int8 landmarks stay within a quantization step of the float ones (crop px).
        self.assertLess(np.abs(qlm - lm).max(), QLM[0])

    def test_mlp_matches_the_reference_and_recognises_the_fixture(self):
        w = {k: self.mlp[k] for k in ("W1", "b1", "W2", "b2")}
        hidden = int(self.mlp["hidden"])
        wpath = Path(self.tmp.name) / "w.txt"
        wpath.write_text(f"{hidden}\n" + "\n".join(" ".join(f"{v:.7g}" for v in w[k].ravel()) for k in ("W1", "b1", "W2", "b2")))
        for name in ("hand", "composite"):
            with self.subTest(image=name):
                i = self.names.index(name)
                lm = self.ref["landmarks"][i].astype(np.float32).reshape(21, 3)
                f = hl.features(lm, scalar(self.ref["handedness"][i]) >= 0.5)
                fpath = Path(self.tmp.name) / "f.txt"
                fpath.write_text(" ".join(f"{v:.6f}" for v in f))
                got = np.array(self.run_mode("mlp", fpath, wpath).split(), float)
                want = hl.mlp(f, w)
                np.testing.assert_allclose(got, want, atol=1e-4)
                self.assertAlmostEqual(got.sum(), 1.0, places=4)
        # The "hand" fixture is a G from the dataset. The MLP must say so on MediaPipe's landmarks
        # for that image (mp_hand_*: the full pipeline with its rotated palm crop). The lite
        # model's landmarks on the tight unrotated 224 square (landmarks[hand]) are off enough
        # to read L: the model wants the rotated 2.6x palm crop the board gives it.
        f = hl.features(self.ref["mp_hand_landmarks"].astype(np.float32), scalar(self.ref["mp_hand_right"]) >= 0.5)
        self.assertEqual(hl.LABELS[int(np.argmax(hl.mlp(f, w)))], "G")

    def test_crop_smoothing_matches_the_reference(self):
        """fs_landmark_crop_smooth: converges to a constant input, keeps the up vector unit
        length, matches ml/handlandmark.smooth_crop, and a reset restarts from the raw crop."""
        rng = np.random.default_rng(3)
        n = 40
        crops = []
        for i in range(n):
            if i < 20:  # jitter around one hand
                centre, side = np.array([160.0, 150.0]) + rng.normal(0, 4, 2), 200.0 + rng.normal(0, 8)
                up = np.array([0.3, -1.0]) + rng.normal(0, 0.1, 2)
            else:  # a constant input: the average must converge to it
                centre, side, up = np.array([100.0, 220.0]), 150.0, np.array([-0.6, -0.8])
            up = up / np.linalg.norm(up)
            crops.append((centre, side, up))
        lines = [f"{c[0]} {c[1]} {s} {u[0]} {u[1]}" for c, s, u in crops]
        lines.insert(30, "reset")  # mid-way through the constant stretch
        path = Path(self.tmp.name) / "crops.txt"
        path.write_text("\n".join(lines) + "\n")
        alpha = hl.CROP_SMOOTH
        got = np.array([[float(v) for v in line.split()] for line in self.run_mode("smooth", alpha, path).splitlines()])
        self.assertEqual(got.shape, (n, 7))
        avg, want = None, []
        for i, (centre, side, up) in enumerate(crops):
            if i == 30:
                avg = None
            avg = hl.smooth_crop(avg, (centre, side, np.array([-up[1], up[0]]), -up), alpha)
            want.append([*avg[0], avg[1], *avg[2], *avg[3]])
        np.testing.assert_allclose(got, np.array(want), atol=1e-3)
        np.testing.assert_allclose(np.hypot(got[:, 5], got[:, 6]), 1.0, atol=1e-5)  # unit down vector
        np.testing.assert_allclose(np.hypot(got[:, 3], got[:, 4]), 1.0, atol=1e-5)  # unit right vector
        np.testing.assert_allclose(got[0], want[0], atol=1e-4)  # the first frame copies raw
        # Converged within 1 % of the constant input before the reset (10 frames at 0.5).
        c, s_, u = crops[29]
        self.assertLess(np.abs(got[29, :2] - c).max(), 0.01 * s_)
        self.assertLess(abs(got[29, 2] - s_), 0.01 * s_)
        self.assertLess(np.abs(got[29, 5:7] + u).max(), 0.01)
        # The reset restarts from the raw crop: identical to it on that frame.
        c, s_, u = crops[30]
        np.testing.assert_allclose(got[30], [*c, s_, -u[1], u[0], -u[0], -u[1]], atol=1e-4)

    def test_fixed_handedness_ignores_the_score(self):
        """FS_LM_HANDEDNESS_RIGHT: a frame the model scores as left (0.1) gives the same features
        as one it scores as right (0.9); auto mirrors x between the two; left mirrors both."""
        i = self.names.index("hand")
        lm = self.ref["landmarks"][i].astype(np.float32)
        lm_path = Path(self.tmp.name) / "lm.txt"
        lm_path.write_text(" ".join(f"{v:.5f}" for v in lm))
        centre, side, up = np.array([150.0, 170.0]), 180.0, np.array([0.0, -1.0])

        def feats(hand_score, mode):
            out = self.run_mode("decode", lm_path, 0.95, hand_score, *QLM, *QSIG, *QSIG, *centre, side, *up, mode).splitlines()
            return np.array(out[2].split(), float)

        right_r, right_l = feats(0.9, "right"), feats(0.1, "right")
        np.testing.assert_allclose(right_r, right_l, atol=1e-6)
        auto_r, auto_l = feats(0.9, "auto"), feats(0.1, "auto")
        np.testing.assert_allclose(auto_r, right_r, atol=1e-6)
        mirrored = auto_l.reshape(21, 3).copy()
        mirrored[:, 0] = -mirrored[:, 0]
        np.testing.assert_allclose(mirrored.ravel(), auto_r, atol=1e-6)
        np.testing.assert_allclose(feats(0.9, "left"), auto_l, atol=1e-6)
        # The build's default is RIGHT (fs_landmark.h) and the Python reference agrees.
        header = (SRC / "fs_landmark.h").read_text()
        self.assertIn("#define FS_LM_HANDEDNESS FS_LM_HANDEDNESS_RIGHT", header)
        self.assertEqual(hl.HANDEDNESS, "right")
        self.assertTrue(hl.right_hand(0.1) and hl.right_hand(0.9))
        self.assertFalse(hl.right_hand(0.1, "auto"))

    def test_presence_hold_matches_the_reference(self):
        """fs_landmark_presence_gate: a tracked hand survives up to HOLD_FRAMES frames of presence
        in [PRESENCE_HOLD, PRESENCE_THRESHOLD); a fresh hand needs the full threshold; a frame
        below PRESENCE_HOLD, or a hold that runs out, loses the hand."""
        seqs = {
            "short dip stays one hand": ([0.9, 0.9, 0.3, 0.3, 0.9], [1, 1, 1, 1, 1]),
            "six low frames drop it": ([0.9] + [0.3] * 6, [1, 1, 1, 1, 1, 1, 0]),
            "a fresh hand needs 0.5": ([0.3, 0.4, 0.45, 0.5], [0, 0, 0, 1]),
            "below the hold floor drops at once": ([0.9, 0.2, 0.3], [1, 0, 0]),
            "a good frame resets the hold": ([0.9] + [0.3] * 5 + [0.9] + [0.3] * 5 + [0.3], [1] * 12 + [0]),
            "reset": ([0.9, "reset", 0.3], [1, 0]),
        }
        for name, (values, want) in seqs.items():
            with self.subTest(name):
                path = Path(self.tmp.name) / "presence.txt"
                path.write_text("\n".join(str(v) for v in values) + "\n")
                got = [int(v) for v in self.run_mode("presence", path).split()]
                self.assertEqual(got, want)
                gate, ref = hl.PresenceGate(), []
                for v in values:
                    if v == "reset":
                        gate = hl.PresenceGate()
                    else:
                        ref.append(int(gate.step(v)))
                self.assertEqual(ref, want)
        self.assertEqual((hl.PRESENCE_HOLD, hl.HOLD_FRAMES), (0.25, 5))
        header = (SRC / "fs_landmark.h").read_text()
        self.assertIn("#define FS_LM_PRESENCE_HOLD 0.25f", header)
        self.assertIn("#define FS_LM_HOLD_FRAMES 5", header)

    def test_presence_is_low_on_black(self):
        self.assertLess(scalar(self.ref["presence"][self.names.index("black")]), 0.5)


class LandmarkModel(unittest.TestCase):
    def test_exported_graph_matches_tflite_on_the_fixtures(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not in this interpreter")
        from PIL import Image

        ref = np.load(FIXTURES / "landmark_ref.npz")
        net = torch.export.load(REPO / "model" / "hand_landmark_224.pt2").module(check_guards=False)
        for i, name in enumerate(str(n) for n in ref["names"]):
            x = np.asarray(Image.open(FIXTURES / f"landmark_{name}.png").convert("RGB"))
            with torch.no_grad():
                lm, presence, hand, _world = net(torch.from_numpy(x[None]).float().div(255))
            self.assertLess(np.abs(lm.numpy()[0] - ref["landmarks"][i].astype(np.float32)).max(), 0.2, name)  # fp16 fixture
            self.assertLess(abs(float(presence) - scalar(ref["presence"][i])), 1e-2, name)
            self.assertLess(abs(float(hand) - scalar(ref["handedness"][i])), 1e-2, name)

    def test_params_header_carries_the_mlp(self):
        params = (REPO / "ai_layer" / "model_params.h").read_text()
        mlp = np.load(REPO / "model" / "keypoint_mlp.npz")
        hidden = int(mlp["hidden"])
        self.assertIn(f"#define FS_MLP_HIDDEN {hidden}", params)
        self.assertIn(f"static const float FS_MLP_W1[{hidden}][63]", params)
        self.assertIn(f"static const float FS_MLP_W2[24][{hidden}]", params)
        self.assertNotIn('"classify"', (REPO / "model" / "model.py").read_text().split("# The single-model contract")[0])


if __name__ == "__main__":
    unittest.main()
