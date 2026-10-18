"""Voice (keyword spotting) on the DevKit-E8: host checks of src/fs_voice.c against ml/kws.py and
TensorFlow's own MFCCs (tests/fixtures/kws_ref.npz), of the exported DS-CNN against the TFLite
interpreter, of the generated parameters; and, with FS_HW=1, the board: the microphone level
rises when you talk, the model runs, a known word is recognised.

    FS_SKIP_BUILD=1 .venv/bin/python -m unittest discover -s tests -p test_voice.py -v
    FS_HW=1 .venv/bin/python -m unittest discover -s tests -p test_voice.py -v

Board tests read the loop's voice telemetry (g_fs_voice_tele) over J-Link, so they work with the
UART dead; the UART lines are checked when the port is available. The person-in-the-loop checks
(say "yes", "no", "stop", "go" at the board) are tests/voice_check.py.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml"))
import fs_tools as fs  # noqa: E402
import kws  # noqa: E402

REPO = fs.REPO
SRC = REPO / "src"
FIXTURES = REPO / "tests" / "fixtures"
SOURCES = [SRC / "fs_voice.c", REPO / "tests" / "host" / "test_fs_voice.c"]
CFLAGS = ["-std=c99", "-Wall", "-Wextra", "-Werror", "-pedantic", "-O1", "-I", str(SRC)]
Q = (1.084, 100)  # like ML-Zoo's int8 model; the real values come from model_params.h


def host_cc() -> str | None:
    for cc in (os.environ.get("FS_CC"), "clang", "cc", "gcc"):
        if cc and shutil.which(cc):
            return cc
    return None


def params() -> dict[str, str]:
    text = (REPO / "ai_layer" / "model_params.h").read_text()
    out = {}
    for line in text.splitlines():
        if line.startswith("#define "):
            parts = line.split(None, 2)
            if len(parts) == 3:
                out[parts[1]] = parts[2]
    return out


@unittest.skipUnless(host_cc(), "no host C compiler")
class VoiceHost(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.exe = Path(cls.tmp.name) / "test_fs_voice"
        cls.cc_result = subprocess.run([host_cc(), *CFLAGS, *map(str, SOURCES), "-lm", "-o", str(cls.exe)],
                                       capture_output=True, text=True)
        cls.ref = np.load(FIXTURES / "kws_ref.npz")
        cls.clips = Path(cls.tmp.name) / "clips.bin"
        cls.clips.write_bytes(cls.ref["wavs"].astype("<i2").tobytes())
        cls.n = len(cls.ref["wavs"])

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

    def test_numpy_front_end_matches_tensorflow(self):
        mine = np.stack([kws.mfcc(w) for w in self.ref["wavs"]])
        self.assertLess(np.abs(mine - self.ref["mfcc"]).max(), 1e-3)

    def test_c_front_end_matches_tensorflow(self):
        out = self.run_mode("mfcc", self.clips, self.n)
        c = np.array([[float(v) for v in l.split()] for l in out.strip().splitlines()]).reshape(self.n, 49, 10)
        self.assertLess(np.abs(c - self.ref["mfcc"]).max(), 0.05)  # float32 FFT vs TF's; log domain, values to -250

    def test_stream_quantizes_the_same_frames_in_any_chunk_size(self):
        want = np.stack([kws.quantize(kws.mfcc(w), *Q) for w in self.ref["wavs"]]).astype(int)
        for chunk in (320, 333, 1600, 16000):
            with self.subTest(chunk=chunk):
                lines = self.run_mode("stream", self.clips, self.n, chunk, *Q).strip().splitlines()
                self.assertTrue(all(lines[2 * i] == "frames 49 full 1" for i in range(self.n)), lines[0])
                got = np.array([[int(v) for v in lines[2 * i + 1].split()] for i in range(self.n)]).reshape(self.n, 49, 10)
                self.assertLessEqual(np.abs(got - want).max(), 1)
                self.assertGreater(np.mean(got == want), 0.98)

    def test_rms_is_the_root_mean_square(self):
        out = [float(v) for v in self.run_mode("rms", self.clips, self.n).split()]
        want = [float(np.sqrt(np.mean(w.astype(float) ** 2))) for w in self.ref["wavs"]]
        np.testing.assert_allclose(out, want, rtol=1e-3)

    def test_softmax_matches_numpy(self):
        logits = np.array([-20, 5, 60, -128, 0, 12, 7, 7, -3, 40, 127, 90], dtype=np.int8)
        path = Path(self.tmp.name) / "logits.txt"
        path.write_text(" ".join(map(str, logits)))
        out = np.array([float(v) for v in self.run_mode("softmax", path, 0.125, 3).split()])
        np.testing.assert_allclose(out, kws.softmax((logits.astype(float) - 3) * 0.125), atol=1e-5)

    def test_decision_rule_threshold_refractory_and_no_silence(self):
        def row(ms, k, p):
            v = np.full(12, (1 - p) / 11)
            v[k] = p
            return f"{ms} " + " ".join(f"{x:.4f}" for x in v)
        steps = [row(0, 2, 0.95), row(500, 2, 0.95), row(1000, 2, 0.95), row(1500, 0, 0.99), row(2000, 1, 0.99),
                 row(2500, 10, 0.69), row(3000, 10, 0.71), row(3500, 11, 0.9), row(4100, 11, 0.9)]
        path = Path(self.tmp.name) / "probs.txt"
        path.write_text("\n".join(steps) + "\n")
        out = [l.split()[1] for l in self.run_mode("decide", 0.7, 1000, path).strip().splitlines()]
        self.assertEqual(out, ["yes", "-", "yes", "-", "-", "-", "stop", "-", "go"])
        # the reference decider agrees
        d = kws.Decider(0.7, 1000)
        ref = [d.step(np.array([float(x) for x in s.split()[1:]]), int(s.split()[0])) or "-" for s in steps]
        self.assertEqual(out, ref)


class VoiceModel(unittest.TestCase):
    def test_exported_graph_matches_the_tflite_interpreter(self):
        import torch
        ref = np.load(FIXTURES / "kws_ref.npz")
        ep = torch.export.load(REPO / "model" / "ds_cnn_kws.pt2")
        module = ep.module()  # exported for batch 1: one clip per call
        with torch.no_grad():
            logits = np.stack([module(torch.from_numpy(m).reshape(1, 49, 10, 1))[0].numpy() for m in ref["mfcc"]])
        probs = np.stack([kws.softmax(l) for l in logits])
        self.assertLess(np.abs(probs - ref["ref_probs"]).max(), 1e-3)
        self.assertEqual(list(probs.argmax(1)), list(ref["ref_probs"].argmax(1)))

    def test_params_header_carries_the_kws_method(self):
        p = params()
        self.assertEqual(p.get("FS_VOICE_FRAMES"), "49")
        self.assertEqual(p.get("FS_VOICE_COEFFS"), "10")
        self.assertEqual(p.get("FS_VOICE_CLASSES"), "12")
        self.assertEqual(p.get("FS_VOICE_LABELS"), '"' + ",".join(kws.LABELS).replace("_", "") + '"')
        self.assertEqual(p.get("KWS_INPUT0_SIZE"), "490")
        self.assertEqual(p.get("KWS_OUTPUT0_SIZE"), "12")
        scale = float(p["KWS_INPUT0_SCALE"].rstrip("f"))
        self.assertTrue(0.5 < scale < 2.0, scale)  # MFCC units: ML-Zoo's int8 model used 1.084

    def test_c_labels_match_the_model(self):
        c = (SRC / "fs_voice.c").read_text()
        for word in kws.LABELS:
            self.assertIn(f'"{word.strip("_")}"', c)


@unittest.skipUnless(os.environ.get("FS_HW") == "1", "set FS_HW=1 to flash and run the DevKit-E8")
class OnBoard(unittest.TestCase):
    """Voice telemetry over J-Link: g_fs_voice_tele words (src/app_main.cpp):
    0 capture running, 1 blocks received, 2 overruns, 3 rms x1 of the last 0.5 s, 4 frames pushed,
    5 inferences, 6 last word class (0 = none yet), 7 last score x1000, 8 npu ms x1000,
    9 mfcc ms per frame x1000, 10 words reported."""
    N = 11

    @classmethod
    def setUpClass(cls):
        if os.environ.get("FS_SKIP_BUILD") != "1":
            r = fs.cbuild()
            assert r.returncode == 0, (r.stdout + r.stderr)[-2000:]
        cls.uart = None
        try:
            cls.uart = fs.UartCapture()
        except (fs.UartBusy, FileNotFoundError):
            pass
        cls.flash_log = fs.flash_and_start()
        if cls.uart:
            cls.console = cls.uart.read_for(fs.BOOT_TIMEOUT, until=b"live:") + cls.uart.read_for(8)
            cls.uart.close()
        else:
            time.sleep(min(fs.BOOT_TIMEOUT, 40))
            cls.console = ""
        cls.t1 = cls.tele()
        time.sleep(3)
        cls.t2 = cls.tele()
        cls.state = fs.core_state()

    @classmethod
    def tele(cls) -> list[int]:
        base = fs.symbol_address("g_fs_voice_tele")
        return [fs.read_word(base + 4 * i) for i in range(cls.N)]

    def test_flash_is_programmed(self):
        self.assertIn("O.K.", self.flash_log, self.flash_log[-800:])

    def test_capture_runs_and_blocks_keep_arriving(self):
        self.assertEqual(self.t2[0], 1, self.t2)
        blocks = self.t2[1] - self.t1[1]
        self.assertGreaterEqual(blocks, 100, (self.t1, self.t2))  # 50 blocks of 20 ms per second, 3 s

    def test_no_overruns_while_the_loop_runs(self):
        self.assertEqual(self.t2[2], self.t1[2], (self.t1, self.t2))

    def test_frames_and_inferences_advance(self):
        self.assertGreaterEqual(self.t2[4] - self.t1[4], 100)  # 150 frames in 3 s
        self.assertGreaterEqual(self.t2[5] - self.t1[5], 4)    # one inference per 0.5 s

    def test_front_end_and_model_are_cheap(self):
        self.assertLess(self.t2[9] / 1000, 2.0, "ms per MFCC frame")
        self.assertLess(self.t2[8] / 1000, 20.0, "NPU ms per inference")

    def test_level_is_not_stuck(self):
        self.assertGreater(self.t2[3], 0, "RMS of the last window is zero: no audio reaches the ring")

    def test_console_reports_voice_when_available(self):
        if not self.console:
            self.skipTest("no UART")
        self.assertIn("voice capture init OK", self.console)
        self.assertRegex(self.console, r"voice: rms \d+")

    def test_app_is_healthy(self):
        self.assertIsNone(fs.app_problem(self.state), self.state)


if __name__ == "__main__":
    unittest.main()
