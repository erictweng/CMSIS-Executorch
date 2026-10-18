"""Display: the DevKit-E8 LCD shows the camera crop, the live letter with its confidence and
the fps, every frame, on top of the three board fixes from hackathon-pico-faces (SRAM0 above
the A32 boot stub, nothing the NPU uses in SRAM1, caches cleaned at boot).

    .venv/bin/python -m unittest tests.test_display -v            # source + build checks
    FS_HW=1 .venv/bin/python -m unittest tests.test_display -v    # + flash and run the board

FS_SKIP_BUILD=1 reuses the last DevKit-E8 build (the other targets are always built).
"""

from __future__ import annotations

import os
import re
import statistics
import subprocess
import unittest

import yaml

import fs_tools as fs

SRC = fs.REPO / "src"
BOARD = fs.REPO / "board" / "DevKit-E8"
MEM_REGIONS = BOARD / "RTE" / "Device" / "AE822FA0E5597LS0_M55_HP" / "app_mem_regions.h"
DISPLAY_COMPONENTS = ["Device:SOC Peripherals:CDC", "Device:SOC Peripherals:MIPI DSI",
                      "BSP:External peripherals:ILI9806E LCD panel"]
OTHER_TARGETS = ["SSE-320-U85", "AppKit-E8"]
SRAM0 = (0x02000000, 0x02400000)
A32_STUB_END = 0x02010000           # APP_SRAM_POOL_OFFSET: the ATOC loads the boot stub below
FRAME_BUFFER_BYTES = 480 * 800 * 2  # RGB565
# With a hand: "X 0.93 U 0.04 M 0.01 | box 112,84 96x96 s0.87 | det 3.1ms cam 2.0ms pre 0.8ms npu 4.2ms
# ui 1.6ms | 21.3 fps"; without: "no hand | det 3.1ms ui 1.2ms | 21.9 fps". Groups: ui, fps.
LIVE = re.compile(r"^(?:[A-Y] \d\.\d\d [A-Y] \d\.\d\d [A-Y] \d\.\d\d \| box -?\d+,-?\d+ \d+x\d+ s\d\.\d\d "
                  r"\| det [\d.]+ms cam [\d.]+ms pre [\d.]+ms npu [\d.]+ms|no hand \| det [\d.]+ms) "
                  r"ui ([\d.]+)ms \| ([\d.]+) fps$", re.M)
MIN_FPS = 15.0
MAX_UI_MS = 10.0  # fs_api.h: fs_ui_render must return in < 10 ms
# Frame-buffer rows (480 px, 2 bytes each) of the layout in src/fs_display.cpp.
ROW_BYTES = 480 * 2
CROP_ROWS, CROP_X = (44, 428), 48
LETTER_ROWS, LETTER_X = (448, 532), 48


def nm_sized() -> dict[str, tuple[int, int, str]]:
    out = subprocess.run([fs.gcc_tool("nm"), "-S", str(fs.AXF)], capture_output=True, text=True, check=True).stdout
    syms = {}
    for line in out.splitlines():
        p = line.split()
        if len(p) == 4:
            syms[p[3]] = (int(p[0], 16), int(p[1], 16), p[2])
    return syms


def symbol(syms: dict, name: str) -> tuple[int, int, str]:
    """A plain or anonymous-namespace (mangled) symbol."""
    return next(v for k, v in syms.items() if k == name or k.endswith(f"{len(name)}{name}E"))


class Source(unittest.TestCase):
    def test_layer_has_the_display_components(self):
        layer = yaml.safe_load(fs.LAYER.read_text())["layer"]
        components = [c["component"] for c in layer["components"]]
        for want in DISPLAY_COMPONENTS:
            self.assertIn(want, components)

    def test_display_define_and_source_are_devkit_only(self):
        layer = yaml.safe_load(fs.LAYER.read_text())["layer"]
        self.assertIn("FINGERSPELL_LCD", layer["define"])
        for other in ("Corstone-320", "AppKit-E8"):
            text = (fs.REPO / "board" / other / "Board-U85.clayer.yml").read_text()
            self.assertNotIn("FINGERSPELL_LCD", text, other)
        files = yaml.safe_load((fs.REPO / "cmsis-executorch.cproject.yml").read_text())["project"]["groups"][0]["files"]
        display = next(f for f in files if f["file"] == "./src/fs_display.cpp")
        self.assertEqual(display.get("for-context"), "+DevKit-E8")

    def test_panel_is_480x800_rgb565_on_two_dsi_lanes(self):
        self.assertEqual(fs.rte_define("RTE_CDC200"), "1")
        self.assertEqual(fs.rte_define("RTE_CDC200_PIXEL_FORMAT"), "2")
        self.assertEqual(fs.rte_define("RTE_PANEL_HACTIVE_TIME"), "480")
        self.assertEqual(fs.rte_define("RTE_PANEL_VACTIVE_LINE"), "800")
        self.assertEqual(fs.rte_define("RTE_MIPI_DSI"), "1")
        self.assertEqual(int(fs.rte_define("RTE_ILI9806E_PANEL_DSI_N_LANES"), 0), 2)

    def test_ui_contract_is_implemented(self):
        c = (SRC / "fs_display.cpp").read_text()
        self.assertIn("bool fs_ui_init(void)", c)
        self.assertIn("void fs_ui_render(const fs_decoder_t* d, const fs_metrics_t* m, fs_event_t last_event)", c)
        self.assertIn('section(".bss.lcd_frame_buf")', c)
        self.assertIn("SCB_CleanDCache_by_Addr", c)

    def test_display_never_reconfigures_the_se_memory_blocks(self):
        # SERVICES_set_run_cfg with memory blocks left MRAM/SRAM0 unusable once (pico-faces).
        self.assertNotIn("SERVICES_set_run_cfg(", (SRC / "fs_display.cpp").read_text())

    def test_bulk_sram_starts_above_the_a32_stub_and_excludes_sram1(self):
        h = MEM_REGIONS.read_text()
        self.assertRegex(h, r"#define APP_SRAM_POOL_OFFSET\s+0x00010000\b")
        self.assertRegex(h, r"#define SRAM0_SRAM1_COMBINED\s+0\b")
        for script in ("linker_ac6_mram.sct.src", "linker_gnu_mram.ld.src"):
            text = (BOARD / script).read_text()
            self.assertIn("APP_SRAM0_BASE + APP_SRAM_POOL_OFFSET", text, script)
            self.assertIn(".bss.lcd_frame_buf", text, script)
        ac6 = (BOARD / "linker_ac6_mram.sct.src").read_text()
        sram1 = ac6[ac6.index("RW_SRAM1 "):]
        self.assertNotIn("lcd_frame_buf", sram1[:sram1.index("}")], "the frame buffer must not go to SRAM1")

    def test_main_cleans_the_caches_before_anything_else(self):
        m = (BOARD / "main.c").read_text()
        self.assertLess(m.index("stop_leftover_camera_capture();\n"), m.index("SCB_CleanInvalidateDCache();"))
        self.assertLess(m.index("SCB_CleanInvalidateDCache();"), m.index("SCB_InvalidateICache();"))
        self.assertLess(m.index("SCB_InvalidateICache();"), m.index("board_pins_config();"))
        self.assertLess(m.index("vbat_init();"), m.index("stdio_init();"))

    def test_live_loop_prints_the_render_time(self):
        app = (SRC / "app_main.cpp").read_text()
        self.assertIn("npu %.1fms ui %.1fms | %.1f fps", app)
        self.assertIn("fs_display_set_top3(top3, r.prob);", app)


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
        self.assertNotRegex(out, r"fs_display\.cpp:\d+:\d+: (warning|error)")

    def test_display_components_resolve_from_the_alif_pack(self):
        cb = yaml.safe_load(fs.CBUILD_YML.read_text())["build"]
        resolved = {c["component"] for c in cb["components"]}
        for want in DISPLAY_COMPONENTS:
            vendor_name = [k for k in resolved if k.split("::", 1)[-1].split("@")[0] == want]
            self.assertTrue(vendor_name, f"{want} not resolved")
            self.assertTrue(vendor_name[0].startswith("AlifSemiconductor::"), vendor_name[0])

    def test_display_code_is_linked(self):
        for s in ("fs_ui_init", "fs_ui_render", "fs_display_set_top3", "Driver_CDC200"):
            # C++ names are mangled (_Z20fs_display_set_top3PKcPKf): match by substring.
            self.assertTrue(any(s in k for k in self.syms), s)

    def test_frame_buffer_is_480x800_rgb565_aligned_in_sram0(self):
        addr, size, _ = symbol(self.syms, "framebuffer")
        self.assertEqual(size, FRAME_BUFFER_BYTES)
        self.assertEqual(addr % 32, 0)
        self.assertTrue(A32_STUB_END <= addr and addr + size <= SRAM0[1], hex(addr))

    def test_every_bulk_sram_buffer_is_in_sram0_above_the_stub(self):
        buffers = [symbol(self.syms, n) for n in ("s_ring", "g_method_pool", "g_temp_pool", "framebuffer")]
        for addr, size, _ in buffers:
            self.assertGreaterEqual(addr, A32_STUB_END, hex(addr))
            self.assertLessEqual(addr + size, SRAM0[1], "nothing the NPU or the CDC200 uses goes to SRAM1")
        ordered = sorted(buffers)
        for (a, sa, _), (b, _, _) in zip(ordered, ordered[1:]):
            self.assertLessEqual(a + sa, b, "bulk SRAM buffers overlap")

    def test_sram1_is_empty(self):
        m = re.search(r"Execution Region RW_SRAM1 \(Base: 0x[0-9a-f]+, Size: 0x([0-9a-f]+)", fs.MAP.read_text())
        self.assertTrue(m, "RW_SRAM1 region missing: SRAM0_SRAM1_COMBINED must be 0")
        self.assertEqual(int(m.group(1), 16), 0)

    def test_targets_without_the_display_still_build_without_it(self):
        for target in OTHER_TARGETS:
            with self.subTest(target=target):
                # --active keeps each target's tmp/<target> tree; a --context build regenerates
                # the solution-level tmp/ and drops tmp/DevKit-E8, which later tests inspect.
                r = fs.run(["cbuild", str(fs.SOLUTION), "--active", target, "--packs"])
                self.assertEqual(r.returncode, 0, (r.stdout + r.stderr)[-2000:])
                out = fs.REPO / "out" / "cmsis-executorch" / target / "Debug"
                cb = (out / f"cmsis-executorch.Debug+{target}.cbuild.yml").read_text()
                self.assertNotIn("src/fs_display.cpp", cb)
                syms = subprocess.run([fs.gcc_tool("nm"), str(out / "cmsis-executorch.axf")],
                                      capture_output=True, text=True, check=True).stdout
                self.assertNotIn("fs_ui_render", syms)


@unittest.skipUnless(os.environ.get("FS_HW") == "1", "set FS_HW=1 to flash and run the DevKit-E8")
class OnBoard(unittest.TestCase):
    """Flash, read the boot and ~6 s of live lines, read the frame buffer twice with the
    debugger attached since the flash, then a warm reset: everything must come back."""

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
            cls.console = uart.read_for(fs.BOOT_TIMEOUT, until=b"live:") + uart.read_for(6)
        syms = nm_sized()
        cls.fb = symbol(syms, "framebuffer")[0]
        cls.words = {name: cls.fb_words(name) for name in ("crop", "letter")}
        cls.words2 = {name: cls.fb_words(name) for name in ("crop", "letter")}
        cls.display_errors = fs.read_word(symbol(syms, "display_errors")[0])
        cls.state = fs.core_state()
        if uart:
            uart.data.clear()
        fs.reset_and_start()
        cls.console2 = ""
        if uart:
            cls.console2 = uart.read_for(fs.BOOT_TIMEOUT, until=b"live:") + uart.read_for(4)
            uart.close()
        cls.frames_after_reset = fs.read_word(fs.symbol_address("g_fs_loop_frames"))

    @classmethod
    def fb_words(cls, area: str) -> list[int]:
        rows, x = (CROP_ROWS, CROP_X) if area == "crop" else (LETTER_ROWS, LETTER_X)
        words = []
        for row in range(rows[0] + 20, rows[1] - 20, (rows[1] - rows[0] - 40) // 4):
            for col in range(x + 4, x + 60, 16):
                words.append(fs.read_word(cls.fb + row * ROW_BYTES + col * 2))
        return words

    def need_uart(self):
        if self.uart_error:
            self.skipTest(self.uart_error)

    def live(self) -> list[re.Match]:
        return list(LIVE.finditer(self.console))

    def test_flash_is_programmed(self):
        self.assertIn("O.K.", self.flash_log, self.flash_log[-1500:])

    def test_panel_starts_without_dsi_errors(self):
        self.need_uart()
        self.assertIn("Display: 480x800 RGB565 panel started", self.console)
        self.assertNotIn("DSI error", self.console)
        self.assertNotIn("Display: CDC200", self.console)  # driver failures
        self.assertNotIn("Display: not started", self.console)
        self.assertEqual(self.display_errors, 0)

    def test_self_test_and_camera_still_pass(self):
        self.need_uart()
        self.assertIn("Test_result: PASS", self.console)
        self.assertIn("camera init OK 320x320 RGB565", self.console)
        self.assertLess(self.console.index("Display:"), self.console.index("camera init OK"))

    def test_live_lines_carry_the_render_time_and_keep_the_frame_rate(self):
        self.need_uart()
        live = self.live()
        self.assertGreaterEqual(len(live), 30, self.console[-1500:])
        settled = live[5:]
        self.assertGreaterEqual(statistics.median(float(m.group(2)) for m in settled), MIN_FPS)
        self.assertLess(statistics.median(float(m.group(1)) for m in settled), MAX_UI_MS)
        self.assertLess(max(float(m.group(1)) for m in settled), 2 * MAX_UI_MS, "ui time spikes")

    def test_crop_area_shows_the_live_camera(self):
        self.assertNotEqual(self.words["crop"], self.words2["crop"], "the crop area did not change between reads")

    def test_letter_area_is_drawn(self):
        self.assertTrue(any(w != 0 for w in self.words["letter"]), "the letter area is all black")

    def test_app_is_healthy_while_rendering(self):
        self.assertIsNone(fs.app_problem(self.state), self.state)

    def test_everything_comes_back_after_a_warm_reset(self):
        self.assertGreaterEqual(self.frames_after_reset, 1)
        if not self.uart_error:
            self.assertIn("Display: 480x800 RGB565 panel started", self.console2)
            self.assertIn("camera init OK 320x320 RGB565", self.console2)
            self.assertTrue(LIVE.search(self.console2), self.console2[-1000:])


if __name__ == "__main__":
    unittest.main()
