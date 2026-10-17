"""Camera steps 4 and 5 on the DevKit-E8.

Step 4: frames arrive continuously into a 3-block ring the CPI rotates through itself
(streaming mode); a 3 s probe prints the frame counter and 8 raw pixels; the crop
(centre 256x256 -> 128x128 RGB888) prints as text art and can be dumped as a PNG.
Step 5: the live loop (crop -> preprocess -> model -> top-3) runs forever and prints
`X 0.30 U 0.21 M 0.06 | cam 25.0ms pre 1.0ms npu 7.8ms | 19.8 fps` per frame.

    .venv/bin/python -m unittest discover -s tests -p "test_step4_5*.py" -v
    FS_HW=1 .venv/bin/python -m unittest discover -s tests -p "test_step4_5*.py" -v

The cover-the-lens check needs a person: tests/check_cover_lens.py.
"""

from __future__ import annotations

import os
import re
import statistics
import subprocess
import time
import unittest

import fs_tools as fs

SRC = fs.REPO / "src"
LIVE = re.compile(
    r"^([A-Y]) (\d\.\d\d) ([A-Y]) (\d\.\d\d) ([A-Y]) (\d\.\d\d) \| cam ([\d.]+)ms pre ([\d.]+)ms "
    r"npu ([\d.]+)ms(?: ui [\d.]+ms)? \| ([\d.]+) fps$", re.M)  # ui: with the LCD build
PROBE = re.compile(r"^probe frame (\d+) \| px((?: [0-9a-f]{4}){8}) \| mean (\d+)$", re.M)
PROBE_DONE = re.compile(r"^probe done: (\d+) frames in ([\d.]+) s = ([\d.]+) fps "
                        r"\(overflows (\d+) errors (\d+) restarts (\d+)\)$", re.M)
LABELS = "ABCDEFGHIKLMNOPQRSTUVWXY"
FPS_RANGE = (10.0, 35.0)  # MT9M114 320x320 runs at ~25 fps; the loop at ~19 with UART output
OUT = fs.REPO / "tests" / "out"


def counters() -> dict[str, int]:
    names = ["frames", "errors", "overflows", "restarts", "overrun", "streaming", "stage"]
    g = {n: fs.read_word(fs.symbol_address(f"g_fs_camera_{n}")) for n in names}
    g["loop"] = fs.read_word(fs.symbol_address("g_fs_loop_frames"))
    return g


class Source(unittest.TestCase):
    def test_cpi_runs_in_streaming_mode_with_three_buffers(self):
        self.assertEqual(fs.rte_define("RTE_CPI_STREAMING_ENABLE"), "1")
        self.assertEqual(fs.rte_define("RTE_CPI_NUM_ACTIVE_FRAMEBUFFERS"), "3")
        c = (SRC / "fs_camera.c").read_text()
        self.assertIn("#error", c)  # fs_camera refuses to build against the legacy single-buffer mode
        self.assertIn("cpi->CaptureVideo(s_frame_ptrs)", c)
        self.assertNotRegex(c, r"CaptureVideo\(s_frames\[", "no per-VSYNC re-arm (fails BUSY)")

    def test_frame_ring_has_an_overrun_canary(self):
        c = (SRC / "fs_camera.c").read_text()
        self.assertRegex(c, r"uint8_t frames\[FS_CAM_FRAMES\]\[FS_CAM_FRAME_BYTES\];\s*uint32_t canary\[")
        self.assertIn("canary_intact()", c)

    def test_board_stops_a_leftover_capture_before_the_secure_enclave(self):
        m = (fs.BOARD / "main.c").read_text()
        body = m[m.index("int main(void)"):]
        self.assertLess(body.index("stop_leftover_camera_capture();"), body.index("se_services_port_init();"))
        self.assertIn("PERIPH_CLK_ENA_CPI_CKEN", m, "only touch the CPI when its clock is on")

    def test_crop_box_is_the_centre_256_of_320(self):
        h = (SRC / "fs_camera.h").read_text()
        self.assertRegex(h, r"#define FS_CAM_BOX \(2 \* FS_SIZE\)")

    def test_live_loop_never_returns(self):
        app = (SRC / "app_main.cpp").read_text()
        self.assertIn("[[noreturn]] void live_loop(", app)
        self.assertRegex(app, r"for \(;;\) \{")


@unittest.skipUnless(os.environ.get("FS_HW") == "1", "set FS_HW=1 to flash and run the DevKit-E8")
class OnBoard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get("FS_SKIP_BUILD") != "1":
            r = fs.cbuild()
            assert r.returncode == 0, (r.stdout + r.stderr)[-2000:]
        cls.uart = fs.UartCapture()  # these tests need the console: fail if it is busy
        cls.flash_log = fs.flash_and_start()
        cls.uart.read_for(fs.BOOT_TIMEOUT, until=b"live:")
        cls.uart.read_for(6)
        cls.console = cls.uart.text
        cls.c1 = counters()
        time.sleep(5)
        cls.c2 = counters()
        cls.state = fs.core_state()
        w, h, frame, rgb = fs.capture_crop(cls.uart)
        cls.crop = (w, h, frame, rgb)
        OUT.mkdir(exist_ok=True)
        fs.write_png(OUT / "crop.png", w, h, rgb, scale=3)
        # Warm reset in the middle of the live loop: the board's main() must stop the
        # running capture before the Secure Enclave handshake, and the camera come back.
        cls.uart.data.clear()
        fs.reset_and_start()
        cls.uart.read_for(fs.BOOT_TIMEOUT, until=b"live:")
        cls.uart.read_for(3)
        cls.console_after_reset = cls.uart.text
        cls.uart.close()

    # ---- step 4: frames ----
    def test_probe_counts_frames_at_a_plausible_rate(self):
        m = PROBE_DONE.search(self.console)
        self.assertTrue(m, self.console[-3000:])
        frames, fps = int(m.group(1)), float(m.group(3))
        self.assertGreaterEqual(frames, 30)
        self.assertTrue(FPS_RANGE[0] <= fps <= FPS_RANGE[1], fps)
        self.assertEqual(int(m.group(6)), 0, "no capture restarts")

    def test_probe_frame_numbers_climb(self):
        seqs = [int(m.group(1)) for m in PROBE.finditer(self.console)]
        self.assertGreaterEqual(len(seqs), 8)
        self.assertEqual(seqs, sorted(set(seqs)), "frame numbers must strictly increase")

    def test_probe_pixels_are_live_not_stale_buffers(self):
        rows = [m.group(2) for m in PROBE.finditer(self.console)]
        # The broken legacy mode cycled 3 identical pixel sets; live frames differ (noise, exposure).
        self.assertGreater(len(set(rows)), 3, rows)
        self.assertTrue(any(px != " 0000" * 8 for px in rows), "all-black frames")

    def test_crop_text_art_is_printed(self):
        i = self.console.index("crop (centre 256x256 of 320x320, 2x2 average):")
        art = re.findall(r"^  \|(.{32})\|$", self.console[i:], re.M)[:16]
        self.assertEqual(len(art), 16)
        self.assertGreater(len(set("".join(art))), 2, "a flat image")

    def test_crop_dump_is_a_real_128x128_image(self):
        w, h, _, rgb = self.crop
        self.assertEqual((w, h, len(rgb)), (128, 128, 128 * 128 * 3))
        luma = [(rgb[i] * 77 + rgb[i + 1] * 150 + rgb[i + 2] * 29) >> 8 for i in range(0, len(rgb), 3)]
        self.assertGreater(statistics.pstdev(luma), 5, "no structure in the crop")
        self.assertTrue(10 < statistics.mean(luma) < 245, "crop is black or white")

    # ---- step 5: live loop ----
    def live(self):
        return [m.groups() for m in LIVE.finditer(self.console)]

    def test_live_lines_have_the_planned_format(self):
        live = self.live()
        self.assertGreaterEqual(len(live), 30, self.console[-2000:])
        for g in live:
            self.assertTrue(all(c in LABELS for c in (g[0], g[2], g[4])), g)
            p = [float(g[1]), float(g[3]), float(g[5])]
            self.assertEqual(p, sorted(p, reverse=True), g)
            self.assertLessEqual(sum(p), 1.0001, g)

    def test_live_loop_rate_and_stage_times(self):
        live = self.live()[5:]  # after the fps smoothing settles
        fps = statistics.median(float(g[9]) for g in live)
        npu = statistics.median(float(g[8]) for g in live)
        pre = statistics.median(float(g[7]) for g in live)
        self.assertTrue(FPS_RANGE[0] <= fps <= FPS_RANGE[1], fps)
        self.assertLess(npu, 20.0)
        self.assertLess(pre, 5.0)
        self.assertTrue(all(float(g[9]) < 100 for g in live), "fps spikes")

    def test_loop_keeps_classifying_every_frame(self):
        frames = self.c2["frames"] - self.c1["frames"]
        loops = self.c2["loop"] - self.c1["loop"]
        self.assertGreaterEqual(frames, 5 * FPS_RANGE[0])
        self.assertGreaterEqual(loops, 0.8 * frames, (frames, loops))

    def test_loop_never_returns_and_nothing_faults(self):
        self.assertEqual(self.state["exception"], "NoException", self.state)
        self.assertEqual(self.c2["streaming"], 1)

    def test_no_capture_errors_or_overrun_while_streaming(self):
        self.assertEqual(self.c2["errors"], self.c1["errors"], (self.c1, self.c2))
        self.assertEqual(self.c2["restarts"], 0)
        self.assertEqual(self.c2["overrun"], 0)
        self.assertNotIn("camera STOPPED", self.console)

    def test_no_model_errors_in_the_loop(self):
        self.assertNotRegex(self.console, r"(forward|set_input) failed")

    def test_camera_and_loop_come_back_after_a_warm_reset_mid_loop(self):
        t = self.console_after_reset
        self.assertIn("Test_result: PASS", t)
        self.assertIn("camera init OK 320x320 RGB565", t)
        self.assertGreaterEqual(len(LIVE.findall(t)), 10, t[-1500:])


if __name__ == "__main__":
    unittest.main()
