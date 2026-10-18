"""Palm detector (BlazePalm, src/fs_detect.c + ml/blazepalm.py): host checks, and the live
loop with the detector on the DevKit-E8.

    .venv/bin/python -m unittest tests.test_detector -v            # host: C vs reference
    FS_HW=1 .venv/bin/python -m unittest tests.test_detector -v    # + flash and run the board

Host: src/fs_detect.c compiles as strict C99 with tests/host/test_fs_detect.c; its anchors,
decode, box rule, 320->192 resize and hand-box crop match ml/blazepalm.py and numpy on the
fixtures in tests/fixtures (three 192x192 inputs as PNG and the TFLite model's raw outputs
in fp16). With torch in the venv the exported model/blazepalm_192.pt2 is checked against the
same outputs. Board (FS_HW=1, UART): box lines, no-hand handling, fps, warm reset.
"""

from __future__ import annotations

import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml"))
import blazepalm as bp  # noqa: E402
import fs_tools as fs  # noqa: E402

REPO = fs.REPO
SRC = REPO / "src"
FIXTURES = REPO / "tests" / "fixtures"
SOURCES = [SRC / "fs_detect.c", REPO / "tests" / "host" / "test_fs_detect.c"]
CFLAGS = ["-std=c99", "-Wall", "-Wextra", "-Werror", "-pedantic", "-O1", "-I", str(SRC)]
FRAME, INPUT, CROP = 320, 192, 128
# int8 parameters like the generated model_params.h (DETECT_INPUT0_* / DETECT_OUTPUT<i>_*):
# input [0, 1] -> pixel - 128; regressors and logits as calibrated on dataset crops.
QIN = (1 / 255, -128)
QREG = (0.754, -56)
QLOG = (0.1095, 97)
# "X 0.93 U 0.04 M 0.01 | box 112,84 96x96 s0.87 | det 3.1ms cam 2.0ms pre 0.8ms npu 4.2ms ui 1.6ms | 21.3 fps"
BOX_LINE = re.compile(r"^([A-Y]) (\d\.\d\d) [A-Y] \d\.\d\d [A-Y] \d\.\d\d \| box (-?\d+),(-?\d+) (\d+)x(\d+) s(\d\.\d\d) "
                      r"\| det ([\d.]+)ms cam ([\d.]+)ms pre ([\d.]+)ms npu ([\d.]+)ms(?: ui ([\d.]+)ms)? \| ([\d.]+) fps$", re.M)
NO_HAND_LINE = re.compile(r"^no hand \| det ([\d.]+)ms(?: ui ([\d.]+)ms)? \| ([\d.]+) fps$", re.M)


def host_cc() -> str | None:
    for cc in (os.environ.get("FS_CC"), "clang", "cc", "gcc"):
        if cc and shutil.which(cc):
            return cc
    return None


def png(name: str) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(FIXTURES / f"detect_{name}.png").convert("RGB"))


def rgb565(rgb: np.ndarray) -> np.ndarray:
    r, g, b = (rgb[..., i].astype(np.uint16) for i in range(3))
    return ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3)


def rgb565_to_888(v: np.ndarray) -> np.ndarray:
    r5, g6, b5 = (v >> 11) & 0x1F, (v >> 5) & 0x3F, v & 0x1F
    return np.stack([(r5 << 3) | (r5 >> 2), (g6 << 2) | (g6 >> 4), (b5 << 3) | (b5 >> 2)], -1).astype(np.float64)


def quantize(v: np.ndarray, q: tuple[float, int]) -> np.ndarray:
    return np.clip(np.round(v / q[0]) + q[1], -128, 127).astype(np.int8)


def dequantize(q: np.ndarray, p: tuple[float, int]) -> np.ndarray:
    return (q.astype(np.float32) - p[1]) * p[0]


def area_resize(img: np.ndarray, size: int, box=None) -> np.ndarray:
    """Exact box-filter resize of HxWx3 floats: `box` (x0, y0, x1, y1) float pixels, black off-image."""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = box if box is not None else (0.0, 0.0, float(w), float(h))
    side = max(x1 - x0, y1 - y0)
    scale = side / size

    def taps(origin):
        m = np.zeros((size, max(h, w)))
        for k in range(size):
            a, b = origin + k * scale, origin + (k + 1) * scale
            for p in range(int(np.floor(a)), int(np.ceil(b))):
                if 0 <= p < m.shape[1]:
                    m[k, p] = max(0.0, min(b, p + 1) - max(a, p)) / scale
        return m

    ty, tx = taps(y0)[:, :h], taps(x0)[:, :w]
    return np.einsum("iy,jx,yxc->ijc", ty, tx, img)


@unittest.skipUnless(host_cc(), "no host C compiler (clang/cc/gcc) on PATH")
class DetectorHost(unittest.TestCase):
    """src/fs_detect.c on the laptop against ml/blazepalm.py and numpy."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.exe = Path(cls.tmp.name) / "test_fs_detect"
        cls.cc_result = subprocess.run([host_cc(), *CFLAGS, *map(str, SOURCES), "-lm", "-o", str(cls.exe)],
                                       capture_output=True, text=True)
        cls.ref = np.load(FIXTURES / "detect_ref.npz")
        cls.names = [str(n) for n in cls.ref["names"]]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_mode(self, *args: str) -> str:
        self.assertEqual(self.cc_result.returncode, 0, self.cc_result.stderr)
        r = subprocess.run([str(self.exe), *args], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        return r.stdout

    def test_compiles_as_strict_c99_without_warnings(self):
        self.assertEqual(self.cc_result.returncode, 0, self.cc_result.stderr)
        self.assertEqual(self.cc_result.stderr.strip(), "")

    def test_anchor_table_matches_mediapipe(self):
        rows = np.array([[float(v) for v in line.split()] for line in self.run_mode("anchors").splitlines()])
        want = bp.anchors()
        self.assertEqual(rows.shape, (bp.NUM_ANCHORS, 3))
        self.assertTrue(np.array_equal(rows[:, 0], np.arange(bp.NUM_ANCHORS)))
        np.testing.assert_allclose(rows[:, 1:], want[:, :2], atol=1e-6)
        # Spot values of MediaPipe's table: stride 8 first/last, stride 16 first/last.
        np.testing.assert_allclose(want[0, :2], [0.5 / 24, 0.5 / 24])
        np.testing.assert_allclose(want[1151, :2], [23.5 / 24, 23.5 / 24])
        np.testing.assert_allclose(want[1152, :2], [0.5 / 12, 0.5 / 12])
        np.testing.assert_allclose(want[2015, :2], [11.5 / 12, 11.5 / 12])

    def head(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        return self.ref["regressors"][i].astype(np.float32), self.ref["scores"][i].astype(np.float32)

    def decode_c(self, i: int) -> list[float]:
        reg, sc = self.head(i)
        path = Path(self.tmp.name) / f"head{i}.bin"
        path.write_bytes(reg.tobytes() + sc.reshape(-1).tobytes())
        args = (str(QREG[0]), str(QREG[1]), str(QLOG[0]), str(QLOG[1]))
        return [float(v) for v in self.run_mode("decode", str(path), *args).split()]

    def test_decode_matches_the_reference_on_the_hand_images(self):
        for name in ("hand", "composite"):
            with self.subTest(image=name):
                i = self.names.index(name)
                out = self.decode_c(i)
                reg, sc = self.head(i)
                # The C decode works on the int8 head: exact against the same quantization in
                # numpy, and within the quantization step of the float reference.
                qreg, qsc = dequantize(quantize(reg, QREG), QREG), dequantize(quantize(sc, QLOG), QLOG)
                boxes, kp, s = bp.decode(qreg, qsc)
                best = int(np.argmax(s))
                self.assertEqual(out[0], 1.0, "the C decode did not see a hand")
                self.assertGreaterEqual(out[1], 0.9)
                self.assertAlmostEqual(out[1], float(s[best]), places=3)
                self.assertEqual(int(out[2]), best)
                np.testing.assert_allclose(out[3:7], boxes[best], atol=1e-2)
                np.testing.assert_allclose(out[7:11], bp.hand_box(boxes[best] * FRAME / INPUT, FRAME), atol=2e-2)
                np.testing.assert_allclose(np.array(out[11:]).reshape(7, 2), kp[best], atol=1e-2)
                fboxes, _, fs_ = bp.decode(reg, sc)
                self.assertEqual(int(np.argmax(fs_)), best, "int8 picks another anchor than float")
                self.assertLess(np.abs(np.array(out[3:7]) - fboxes[best]).max(), QREG[0])

    def test_decode_reports_no_hand_on_black(self):
        out = self.decode_c(self.names.index("black"))
        self.assertEqual(out[0], 0.0)
        self.assertLess(out[1], 0.5)

    def test_hand_box_rule(self):
        for palm in ([80, 90, 112, 122], [10, 10, 20, 40], [150, 100, 190, 180]):
            with self.subTest(palm=palm):
                x0, y0, x1, y1 = bp.hand_box(np.array(palm, float), INPUT)
                self.assertAlmostEqual(x1 - x0, y1 - y0)                                   # square
                self.assertAlmostEqual(x1 - x0, max(palm[2] - palm[0], palm[3] - palm[1]) * 2.6)
                self.assertAlmostEqual((x0 + x1) / 2, (palm[0] + palm[2]) / 2)             # centred in x
                self.assertAlmostEqual((y0 + y1) / 2, (palm[1] + palm[3]) / 2 - 0.5 * (palm[3] - palm[1]))

    def frame_file(self, seed: int) -> tuple[Path, np.ndarray]:
        rng = np.random.default_rng(seed)
        frame = rng.integers(0, 65536, (FRAME, FRAME), dtype=np.uint16)
        y, x = np.mgrid[0:FRAME, 0:FRAME]
        frame[(x // 40 + y // 40) % 2 == 0] = rgb565(np.array([[200, 90, 30]]))[0]  # a checkerboard half
        path = Path(self.tmp.name) / f"frame{seed}.bin"
        path.write_bytes(frame.astype("<u2").tobytes())
        return path, frame

    def test_resize_320_to_192_is_the_exact_box_filter(self):
        path, frame = self.frame_file(0)
        out = Path(self.tmp.name) / "resized.bin"
        self.run_mode("resize", str(path), str(QIN[0]), str(QIN[1]), str(out))
        got = np.frombuffer(out.read_bytes(), np.int8).reshape(INPUT, INPUT, 3).astype(int)
        want = quantize(area_resize(rgb565_to_888(frame), INPUT) / 255.0, QIN).astype(int)
        self.assertLessEqual(np.abs(got - want).max(), 1, "int8 input differs from the box filter by more than 1")
        self.assertGreater(np.mean(got == want), 0.99)

    def test_crop_matches_the_box_filter_and_pads_with_black(self):
        path, frame = self.frame_file(1)
        img = rgb565_to_888(frame)
        for box in ((40.0, 60.0, 200.0, 220.0), (-50.0, 100.0, 170.0, 320.0), (200.5, 180.25, 420.5, 400.25),
                    (100.0, 100.0, 164.0, 164.0)):  # inside, off left, off right/bottom, smaller than 128
            with self.subTest(box=box):
                out = Path(self.tmp.name) / "crop.bin"
                self.run_mode("crop", str(path), *map(str, box), str(out))
                got = np.frombuffer(out.read_bytes(), np.uint8).reshape(CROP, CROP, 3).astype(float)
                want = area_resize(img, CROP, box)
                self.assertLessEqual(np.abs(got - np.round(want)).max(), 1.0)
                if box[0] < 0:
                    self.assertEqual(got[:, : int(-box[0] / (box[2] - box[0]) * CROP) - 1].max(), 0.0)

    def test_tracker_holds_and_releases(self):
        self.assertIn("0 failed", self.run_mode("track"))


class DetectorModel(unittest.TestCase):
    """The exported graph and the generated AI layer."""

    def test_exported_graph_matches_tflite_on_the_fixtures(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not in this interpreter")
        ref = np.load(FIXTURES / "detect_ref.npz")
        net = torch.export.load(REPO / "model" / "blazepalm_192.pt2").module(check_guards=False)
        for i, name in enumerate(str(n) for n in ref["names"]):
            with torch.no_grad():
                reg, sc = net(torch.from_numpy(png(name)[None]).float().div(255))
            self.assertLess(np.abs(sc.numpy()[0] - ref["scores"][i].astype(np.float32)).max(), 1e-2, name)  # fp16 fixture
            self.assertLess(np.abs(reg.numpy()[0] - ref["regressors"][i].astype(np.float32)).max(), 0.1, name)

    def test_model_params_and_layer_carry_both_methods(self):
        params = (REPO / "ai_layer" / "model_params.h").read_text()
        for define, value in (("FS_DETECT_SIZE", 192), ("FS_DETECT_ANCHORS", 2016), ("FS_DETECT_COORDS", 18),
                              ("FS_CLASSIFY_SIZE", 128), ("FS_CLASSES", 24), ("DETECT_INPUT0_SIZE", 192 * 192 * 3),
                              ("DETECT_INPUT0_ZERO_POINT", -128)):
            self.assertRegex(params, rf"#define {define}\s+{value}\b")
        values = dict(re.findall(r"#define (DETECT_\w+)\s+(\S+)", params))
        self.assertAlmostEqual(float(values["DETECT_INPUT0_SCALE"].rstrip("f")), 1 / 255, places=4)  # int8 = pixel - 128
        sizes = {int(values["DETECT_OUTPUT0_SIZE"]), int(values["DETECT_OUTPUT1_SIZE"])}
        self.assertEqual(sizes, {2016 * 18, 2016})
        for i in (0, 1):
            self.assertGreater(float(values[f"DETECT_OUTPUT{i}_SCALE"].rstrip("f")), 0.0)
        layer = (REPO / "ai_layer" / "ai_layer.clayer.yml").read_text()
        self.assertIn("Backend EthosU", layer)
        for op in ("Quantized quantize", "Quantized dequantize"):  # the only CPU operators
            self.assertIn(op, layer)

    def test_model_py_feeds_the_detector_0_to_1(self):
        model = (REPO / "model" / "model.py").read_text()
        self.assertNotIn("127.5", model, "BlazePalm 192 takes [0, 1] inputs, not [-1, 1]")
        self.assertIn('"a8w8"', model)
        self.assertIn("quantized_io=True", model, "the firmware feeds int8 and decodes the int8 head")


class Source(unittest.TestCase):
    def test_live_loop_runs_detect_then_classify_and_prints_the_box(self):
        app = (SRC / "app_main.cpp").read_text()
        self.assertIn('execute("detect"', app)
        self.assertIn('execute("classify"', app)
        self.assertNotIn('execute("forward")', app)
        self.assertIn("| box %d,%d %dx%d s%.2f | det %.1fms cam %.1fms pre %.1fms npu %.1fms", app)
        self.assertIn("no hand | det %.1fms", app)

    def test_detector_source_is_built_for_every_target(self):
        import yaml

        files = yaml.safe_load((REPO / "cmsis-executorch.cproject.yml").read_text())["project"]["groups"][0]["files"]
        detect = next(f for f in files if f["file"] == "./src/fs_detect.c")
        self.assertNotIn("for-context", detect)

    def test_live_regexes_accept_the_new_lines(self):
        line = "X 0.93 U 0.04 M 0.01 | box 112,84 96x96 s0.87 | det 3.1ms cam 2.0ms pre 0.8ms npu 4.2ms ui 1.6ms | 21.3 fps"
        self.assertTrue(BOX_LINE.match(line))
        self.assertTrue(NO_HAND_LINE.match("no hand | det 3.1ms ui 1.2ms | 21.9 fps"))
        self.assertTrue(NO_HAND_LINE.match("no hand | det 3.1ms | 21.9 fps"))


@unittest.skipUnless(os.environ.get("FS_HW") == "1", "set FS_HW=1 to flash and run the DevKit-E8")
class OnBoard(unittest.TestCase):
    """Flash, read the boot and ~8 s of live lines, then a warm reset. Whether a hand is in
    view is not known here: box lines and no-hand lines are both accepted, each checked for
    its own invariants; FS_EXPECT_HAND=1 / FS_EXPECT_NO_HAND=1 insist on one kind."""

    MIN_FPS = 15.0
    MAX_DET_PLUS_NPU_MS = 15.0

    @classmethod
    def setUpClass(cls):
        cls.uart_error = None
        try:
            uart = fs.UartCapture()
        except fs.UartBusy as e:
            uart, cls.uart_error = None, str(e)
        cls.flash_log = fs.flash_and_start()
        cls.console = ""
        if uart:
            cls.console = uart.read_for(fs.BOOT_TIMEOUT, until=b"live:") + uart.read_for(8)
            uart.data.clear()
        cls.state = fs.core_state()
        fs.reset_and_start()
        cls.console2 = ""
        if uart:
            cls.console2 = uart.read_for(fs.BOOT_TIMEOUT, until=b"live:") + uart.read_for(4)
            uart.close()
        cls.frames_after_reset = fs.read_word(fs.symbol_address("g_fs_loop_frames"))

    def need_uart(self):
        if self.uart_error:
            self.skipTest(self.uart_error)

    def test_flash_is_programmed(self):
        self.assertIn("O.K.", self.flash_log, self.flash_log[-1500:])

    def test_boot_self_test_still_passes_through_the_classifier(self):
        self.need_uart()
        self.assertIn("Test_result: PASS", self.console)
        self.assertIn("camera init OK 320x320 RGB565", self.console)

    def test_live_lines_are_boxes_or_no_hand_with_the_frame_rate(self):
        self.need_uart()
        boxes = list(BOX_LINE.finditer(self.console))
        none = list(NO_HAND_LINE.finditer(self.console))
        self.assertGreaterEqual(len(boxes) + len(none), 30, self.console[-1500:])
        if os.environ.get("FS_EXPECT_HAND") == "1":
            self.assertGreaterEqual(len(boxes), 30, "expected a hand in view")
        if os.environ.get("FS_EXPECT_NO_HAND") == "1":
            self.assertGreaterEqual(len(none), 30, "expected no hand in view")
        fps = [float(m.group(13)) for m in boxes[5:]] + [float(m.group(3)) for m in none[5:]]
        self.assertGreaterEqual(statistics.median(fps), self.MIN_FPS)
        for m in boxes:
            self.assertGreaterEqual(float(m.group(7)), 0.5, m.group(0))
            self.assertTrue(48 <= int(m.group(5)) <= 640 and int(m.group(5)) == int(m.group(6)), m.group(0))
        self.assertNotIn("failed", self.console)
        self.assertNotIn("live: no frame", self.console)

    def test_detector_and_classifier_meet_the_brief_budget(self):
        """The brief: det + npu < 15 ms and >= 20 fps. Measured 2026-10-17 with the int8 boundary:
        det 22.9 ms (resize 5.6, NPU method 17.3) and npu 8.1 ms; see the handoff, section 4."""
        self.need_uart()
        det = [float(m.group(8)) for m in BOX_LINE.finditer(self.console)] + \
              [float(m.group(1)) for m in NO_HAND_LINE.finditer(self.console)]
        npu = [float(m.group(11)) for m in BOX_LINE.finditer(self.console)]
        self.assertTrue(det, "no live lines")
        self.assertLess(statistics.median(det) + (statistics.median(npu) if npu else 0.0), self.MAX_DET_PLUS_NPU_MS,
                        f"det {statistics.median(det):.1f} ms + npu {statistics.median(npu) if npu else 0:.1f} ms")
        fps = [float(m.group(13)) for m in BOX_LINE.finditer(self.console)][5:] + \
              [float(m.group(3)) for m in NO_HAND_LINE.finditer(self.console)][5:]
        self.assertGreaterEqual(statistics.median(fps), 20.0)

    def test_app_is_healthy(self):
        self.assertIsNone(fs.app_problem(self.state), self.state)

    def test_everything_comes_back_after_a_warm_reset(self):
        self.assertGreaterEqual(self.frames_after_reset, 1)
        if not self.uart_error:
            self.assertIn("camera init OK 320x320 RGB565", self.console2)
            self.assertTrue(BOX_LINE.search(self.console2) or NO_HAND_LINE.search(self.console2), self.console2[-1000:])


if __name__ == "__main__":
    unittest.main()
