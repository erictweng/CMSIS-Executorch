"""Camera step 3: fs_camera_init() brings the MT9M114 up, starts streaming and prints
`camera init OK 320x320 RGB565` once a whole frame has arrived; the model test still passes.

    .venv/bin/python -m unittest discover -s tests -p "test_step3*.py" -v
    FS_HW=1 .venv/bin/python -m unittest discover -s tests -p "test_step3*.py" -v

FS_SKIP_BUILD=1 reuses the last DevKit-E8 build (the other targets are always built).
"""

from __future__ import annotations

import os
import re
import subprocess
import unittest

import yaml

import fs_tools as fs

SRC = fs.REPO / "src"
FRAME_BYTES = 320 * 320 * 2
SRAM_BULK = (0x02000000, 0x02000000 + 8 * 1024 * 1024)  # SRAM0+SRAM1, where .bss.ai_pool goes
FIRST_FRAME_TIMEOUT_MS = 2000
OTHER_TARGETS = ["SSE-320-U85", "AppKit-E8"]


def nm_sized() -> dict[str, tuple[int, int, str]]:
    out = subprocess.run([fs.gcc_tool("nm"), "-S", str(fs.AXF)], capture_output=True, text=True, check=True).stdout
    syms = {}
    for line in out.splitlines():
        p = line.split()
        if len(p) == 4:
            syms[p[3]] = (int(p[0], 16), int(p[1], 16), p[2])
    return syms


def camera_globals() -> dict[str, int]:
    names = ["stage", "status", "first_frame_ms", "frames", "errors", "streaming"]
    g = {n: fs.read_word(fs.symbol_address(f"g_fs_camera_{n}")) for n in names}
    base = fs.symbol_address("g_fs_camera_events")
    g["events"] = [fs.read_word(base + 4 * i) for i in range(7)]
    return g


class Source(unittest.TestCase):
    def test_fs_api_contract_declares_the_camera_functions(self):
        api = (SRC / "fs_api.h").read_text()
        self.assertIn("bool fs_camera_init(void);", api)
        self.assertIn("bool fs_camera_get_crop(uint8_t rgb[FS_SIZE * FS_SIZE * 3]);", api)
        self.assertRegex(api, r"#define FS_SIZE 128\b")

    def test_camera_header_matches_image_config_5(self):
        h = (SRC / "fs_camera.h").read_text()
        self.assertRegex(h, r"#define FS_CAM_WIDTH 320\b")
        self.assertRegex(h, r"#define FS_CAM_HEIGHT 320\b")
        self.assertRegex(h, r"#define FS_CAM_BYTES_PER_PIXEL 2\b")
        self.assertEqual(fs.rte_define("RTE_MT9M114_CAMERA_SENSOR_MIPI_IMAGE_CONFIG"), "5")

    def test_init_message_is_the_one_the_plan_checks_for(self):
        c = (SRC / "fs_camera.c").read_text()
        self.assertIn('printf("camera init OK %dx%d RGB565', c)
        self.assertIn("camera init FAILED at", c)

    def test_camera_source_is_built_only_for_the_devkit(self):
        files = yaml.safe_load((fs.REPO / "cmsis-executorch.cproject.yml").read_text())["project"]["groups"][0]["files"]
        cam = next(f for f in files if f["file"] == "./src/fs_camera.c")
        self.assertEqual(cam.get("for-context"), "+DevKit-E8")

    def test_app_initialises_and_stops_the_camera_only_with_the_camera_layer(self):
        app = (SRC / "app_main.cpp").read_text()
        guard = app.index("#ifdef RTE_VSTREAM_VIDEOIN")
        self.assertLess(guard, app.index('#include "fs_camera.h"'))
        init, test = app.index("fs_camera_init();"), app.index('printf("Test_result')
        self.assertLess(test, init, "the model self-test runs before the camera starts")
        self.assertIn("live_loop(module);", app)  # step 5: never returns while the camera runs


@unittest.skipIf(os.environ.get("FS_SKIP_BUILD") == "1" and not fs.AXF.exists(), "no build to reuse")
class Build(unittest.TestCase):
    build: subprocess.CompletedProcess | None = None

    @classmethod
    def setUpClass(cls):
        if os.environ.get("FS_SKIP_BUILD") != "1":
            fs.cbuild_setup()
            cls.build = fs.cbuild()
        cls.syms = nm_sized()

    def test_devkit_build_succeeds(self):
        if self.build is None:
            self.skipTest("FS_SKIP_BUILD=1")
        out = self.build.stdout + self.build.stderr
        self.assertEqual(self.build.returncode, 0, out[-3000:])
        self.assertNotRegex(out, r"fs_camera\.c:\d+:\d+: (warning|error)")

    def test_camera_code_is_linked(self):
        for s in ("fs_camera_init", "fs_camera_latest_frame", "fs_camera_get_crop", "on_cpi_event", "Driver_CPI"):
            self.assertIn(s, self.syms)

    def test_frame_ring_is_three_full_frames_and_a_canary_in_bulk_sram(self):
        addr, size, _ = self.syms["s_ring"]
        self.assertEqual(size, 3 * FRAME_BYTES + 4096)
        self.assertEqual(addr % 32, 0, "frames must be cache-line aligned")
        self.assertTrue(SRAM_BULK[0] <= addr and addr + size <= SRAM_BULK[1], hex(addr))

    def test_frame_ring_does_not_overlap_the_model_memory(self):
        a = self.syms["s_ring"]
        # The pools are in an anonymous namespace: match the mangled names.
        for other in ("g_method_pool", "g_temp_pool", "g_input"):
            b = next(v for k, v in self.syms.items() if k == other or k.endswith(f"{len(other)}{other}E"))
            self.assertTrue(a[0] + a[1] <= b[0] or b[0] + b[1] <= a[0], f"frame ring overlaps {other}")

    def test_image_still_fits_its_mram_load_region(self):
        m = re.search(r"Load Region LR_ROM \(Base: 0x([0-9a-f]+), Size: 0x([0-9a-f]+), Max: 0x([0-9a-f]+)",
                      fs.MAP.read_text())
        _, size, limit = (int(g, 16) for g in m.groups())
        self.assertLessEqual(size, limit)

    def test_targets_without_the_camera_still_build_without_it(self):
        for target in OTHER_TARGETS:
            with self.subTest(target=target):
                r = fs.run(["cbuild", str(fs.SOLUTION), "--context", f"cmsis-executorch.Debug+{target}", "--packs"])
                self.assertEqual(r.returncode, 0, (r.stdout + r.stderr)[-2000:])
                out = fs.REPO / "out" / "cmsis-executorch" / target / "Debug"
                cb = (out / f"cmsis-executorch.Debug+{target}.cbuild.yml").read_text()
                self.assertIn("src/app_main.cpp", cb)
                self.assertNotIn("src/fs_camera.c", cb)
                syms = subprocess.run([fs.gcc_tool("nm"), str(out / "cmsis-executorch.axf")],
                                      capture_output=True, text=True, check=True).stdout
                self.assertNotIn("fs_camera_init", syms)
                self.assertIn("app_main", syms)


@unittest.skipUnless(os.environ.get("FS_HW") == "1", "set FS_HW=1 to flash and run the DevKit-E8")
class OnBoard(unittest.TestCase):
    """Flash, read the UART, then reset once more: the camera must come up again
    after a warm reset (an earlier version left the Secure Enclave hung)."""

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
            cls.console = uart.read_for(fs.BOOT_TIMEOUT, until=b"camera init") + uart.read_for(1)
        cls.first = camera_globals()
        cls.first_state = fs.core_state()
        if uart:
            uart.data.clear()
        fs.reset_and_start()
        cls.console2 = ""
        if uart:
            cls.console2 = uart.read_for(fs.BOOT_TIMEOUT, until=b"camera init") + uart.read_for(1)
            uart.close()
        cls.second = camera_globals()

    def need_uart(self):
        if self.uart_error:
            self.skipTest(self.uart_error)

    def test_flash_is_programmed(self):
        self.assertIn("O.K.", self.flash_log, self.flash_log[-1500:])

    def test_uart_prints_camera_init_ok(self):
        self.need_uart()
        m = re.search(r"camera init OK 320x320 RGB565 \(first frame after (\d+) ms", self.console)
        self.assertTrue(m, self.console)
        self.assertLessEqual(int(m.group(1)), FIRST_FRAME_TIMEOUT_MS)

    def test_camera_init_prints_after_the_model_self_test(self):
        self.need_uart()
        self.assertLess(self.console.index("Test_result:"), self.console.index("camera init OK"))

    def test_model_test_still_passes(self):
        self.need_uart()
        self.assertRegex(self.console, r"Top-3: B [0-9.]+ .*\(expected B\)")
        self.assertIn("Test_result: PASS", self.console)
        self.assertNotIn("camera init FAILED", self.console)

    def test_init_succeeded_with_a_whole_frame(self):
        g = self.first
        self.assertEqual(g["stage"], 0, g)
        self.assertGreaterEqual(g["frames"], 1, g)
        self.assertGreaterEqual(g["events"][2], 2, "a frame needs two VSYNCs")
        self.assertTrue(0 < g["first_frame_ms"] <= FIRST_FRAME_TIMEOUT_MS, g)

    def test_camera_keeps_streaming_and_the_app_is_healthy(self):
        # Step 5: the live loop never returns, so the camera stays on.
        self.assertEqual(self.first["streaming"], 1, self.first)
        self.assertIsNone(fs.app_problem(self.first_state), self.first_state)

    def test_camera_comes_up_again_after_a_warm_reset(self):
        g = self.second
        self.assertEqual(g["stage"], 0, g)
        self.assertGreaterEqual(g["frames"], 1, g)
        if not self.uart_error:
            self.assertIn("camera init OK 320x320 RGB565", self.console2)
            self.assertIn("Test_result: PASS", self.console2)

    def test_at_most_one_capture_error_at_stream_start_after_a_warm_reset(self):
        # Starting the stream logs one CSI-2 event on most runs (and a few more on the first
        # run after power-up while the sensor settles); errors while streaming are checked
        # in test_step4_5_camera_live.
        self.assertLessEqual(self.second["errors"], 1, self.second)


if __name__ == "__main__":
    unittest.main()
