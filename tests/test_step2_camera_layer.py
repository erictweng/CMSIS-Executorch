"""Camera step 2: the DevKit-E8 board layer carries the MT9M114 camera stack, set to
320x320 RGB565 (image config 5), and the firmware still builds and still passes.

    .venv/bin/python -m unittest tests.test_step2_camera_layer -v        # config + build
    FS_HW=1 .venv/bin/python -m unittest tests.test_step2_camera_layer -v  # + flash and run

FS_SKIP_BUILD=1 reuses the last build. FS_BASE_REF (default `fingerspell`) is the
branch the app/model/RTE changes are compared against.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import unittest

import yaml

import fs_tools as fs

CAMERA_COMPONENTS = [
    "BSP:External peripherals:CAMERA Sensor MT9M114",
    "CMSIS Driver:I2C",
    "CMSIS Driver:vStream:VideoIn&DevKit-E8",
    "Device:SOC Peripherals:CPI",
    "Device:SOC Peripherals:DMA",
    "Device:SOC Peripherals:GPIO",
    "Device:SOC Peripherals:MIPI CSI2",
    "Device:SOC Peripherals:MIPI DSI CSI2 DPHY",
]
# The components the headless runner already had: must survive the change.
BASE_COMPONENTS = [
    "Device:Startup",
    "CMSIS Driver:USART",
    "Machine Learning:NPU Support:Ethos-U Driver&Generic U85",
    "Services:Secure Enclave:core&Source",
    "Services:Retarget IO:STDOUT",
]
CAMERA_OBJECTS = [
    "MT9M114_Camera_Sensor.o", "Camera_Sensor_i2c.o", "Driver_CPI.o", "cpi.o",
    "Driver_MIPI_CSI2.o", "DPHY_CSI2.o", "Driver_I2C.o", "Driver_DMA.o", "Driver_IO.o",
    "vstream_video_in.o",
]
BASE_REF = os.environ.get("FS_BASE_REF", "fingerspell")


def layer() -> dict:
    return yaml.safe_load(fs.LAYER.read_text())["layer"]


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=fs.REPO, capture_output=True, text=True)


class LayerConfig(unittest.TestCase):
    def test_layer_has_every_camera_component(self):
        listed = [c["component"] for c in layer()["components"]]
        for c in CAMERA_COMPONENTS:
            self.assertIn(c, listed)

    def test_layer_keeps_the_headless_runner_components(self):
        listed = [c["component"] for c in layer()["components"]]
        for c in BASE_COMPONENTS:
            self.assertIn(c, listed)

    def test_layer_lists_no_duplicate_components(self):
        listed = [c["component"] for c in layer()["components"]]
        self.assertEqual(len(listed), len(set(listed)))

    def test_layer_brings_the_cmsis_driver_pack_for_the_vstream_api(self):
        packs = [p["pack"] for p in layer()["packs"]]
        self.assertTrue(any(p.startswith("ARM::CMSIS-Driver@") for p in packs), packs)

    def test_layer_provides_vstream_video_in(self):
        provides = layer()["connections"][0]["provides"]
        self.assertIn("CMSIS_VSTREAM_VIDEO_IN", provides)

    def test_layer_keeps_the_ai_pool_placement(self):
        defines = {k: v for d in layer()["define"] for k, v in (d.items() if isinstance(d, dict) else [(d, None)])}
        self.assertEqual(defines["APP_POOL_SECTION"], '\\".bss.ai_pool\\"')
        self.assertEqual(defines["APP_METHOD_POOL_SIZE"], 0x100000)
        self.assertEqual(defines["APP_TEMP_POOL_SIZE"], 0x100000)  # 1 MB since the display: all of SRAM0 is shared


class RteConfig(unittest.TestCase):
    def test_image_config_is_5_320x320_rgb565(self):
        self.assertEqual(fs.rte_define("RTE_MT9M114_CAMERA_SENSOR_MIPI_IMAGE_CONFIG"), "5")
        self.assertRegex(fs.RTE_DEVICE.read_text(), r"<5=>\s+320x320_RGB565")

    def test_mipi_sensor_is_enabled_on_i2c1(self):
        self.assertEqual(fs.rte_define("RTE_MT9M114_CAMERA_SENSOR_MIPI_ENABLE"), "1")
        self.assertEqual(fs.rte_define("RTE_MT9M114_CAMERA_SENSOR_MIPI_I2C_INSTANCE"), "1")
        self.assertEqual(fs.rte_define("RTE_I2C1"), "1")

    def test_csi_and_cpi_are_enabled(self):
        self.assertEqual(fs.rte_define("RTE_CPI"), "1")
        self.assertEqual(fs.rte_define("RTE_MIPI_CSI2"), "1")

    def test_cpi_drives_the_mt9m114_over_mipi_not_the_parallel_port(self):
        self.assertEqual(fs.rte_define("RTE_MT9M114_CAMERA_SENSOR_CPI_ENABLE"), "0")

    def test_every_i2c_instance_has_stuck_low_timeouts(self):
        # Pack 2.2.1's Driver_I2C.c reads these; the 2.2.0-era RTE_Device.h lacked them.
        for n in range(4):
            for line in ("SCL", "SDA"):
                self.assertEqual(fs.rte_define(f"RTE_I2C{n}_{line}_STUCK_LOW_TIMEOUT"), "1000000")

    def test_rte_device_changes_are_only_the_intended_ones(self):
        base = git("show", f"{BASE_REF}:{fs.RTE_DEVICE.relative_to(fs.REPO)}")
        if base.returncode:
            self.skipTest(f"base ref {BASE_REF} not available")
        old = dict(re.findall(r"^#define\s+(\w+)\s+(\S+)", base.stdout, re.M))
        new = dict(re.findall(r"^#define\s+(\w+)\s+(\S+)", fs.RTE_DEVICE.read_text(), re.M))
        changed = {k for k in old.keys() & new.keys() if old[k] != new[k]}
        # Step 2: image config 5. Step 4: CPI streaming mode with 3 frame buffers.
        # Display: RGB565 frame buffer (test_display.py). Whatever the base ref already
        # has is not a change; nothing else may differ.
        self.assertLessEqual(changed, {"RTE_MT9M114_CAMERA_SENSOR_MIPI_IMAGE_CONFIG",
                                       "RTE_CPI_STREAMING_ENABLE", "RTE_CPI_NUM_ACTIVE_FRAMEBUFFERS",
                                       "RTE_CDC200_PIXEL_FORMAT"})
        self.assertEqual(set(old) - set(new), set())
        self.assertLessEqual(set(new) - set(old),
                             {f"RTE_I2C{n}_{l}_STUCK_LOW_TIMEOUT" for n in range(4) for l in ("SCL", "SDA")})


@unittest.skipIf(os.environ.get("FS_SKIP_BUILD") == "1" and not fs.AXF.exists(), "no build to reuse")
class Build(unittest.TestCase):
    setup: subprocess.CompletedProcess | None = None
    build: subprocess.CompletedProcess | None = None

    @classmethod
    def setUpClass(cls):
        if os.environ.get("FS_SKIP_BUILD") != "1":
            cls.setup = fs.cbuild_setup()
            cls.build = fs.cbuild()

    def test_setup_resolves_every_dependency(self):
        if self.setup is None:
            self.skipTest("FS_SKIP_BUILD=1")
        out = self.setup.stdout + self.setup.stderr
        self.assertEqual(self.setup.returncode, 0, out[-2000:])
        self.assertNotIn("dependency validation", out)
        self.assertNotRegex(out, r"(?m)^error|\berror csolution")

    def test_build_succeeds(self):
        if self.build is None:
            self.skipTest("FS_SKIP_BUILD=1")
        out = self.build.stdout + self.build.stderr
        self.assertEqual(self.build.returncode, 0, out[-3000:])
        self.assertIn("1 succeeded, 0 failed", out)
        self.assertNotRegex(out, r": error: ")

    def test_build_outputs_exist_and_are_fresh(self):
        for f in (fs.AXF, fs.HEX, fs.MAP):
            self.assertTrue(f.exists(), f)
            self.assertGreaterEqual(f.stat().st_mtime, fs.LAYER.stat().st_mtime, f"{f} older than the layer")

    def test_every_camera_component_resolves_from_the_alif_pack(self):
        cb = yaml.safe_load(fs.CBUILD_YML.read_text())["build"]
        resolved = {c["component"]: c for c in cb["components"]}
        for want in CAMERA_COMPONENTS:
            vendor_name = [k for k in resolved if k.split("::", 1)[-1].split("@")[0] == want]
            self.assertTrue(vendor_name, f"{want} not resolved")
            self.assertTrue(vendor_name[0].startswith("AlifSemiconductor::"), vendor_name[0])

    def test_every_camera_driver_is_compiled(self):
        objects = {p.name for p in (fs.REPO / "tmp" / fs.TARGET).rglob("*.o")}
        for o in CAMERA_OBJECTS:
            self.assertIn(o, objects)

    def test_mt9m114_driver_is_compiled_for_320x320(self):
        """Preprocess the driver with its own compile command: config 5 must reach it."""
        entries = json.loads(fs.COMPILE_COMMANDS.read_text())
        entry = next(e for e in entries if e["file"].endswith("MT9M114_Camera_Sensor.c"))
        args = shlex.split(entry["command"]) if "command" in entry else list(entry["arguments"])
        cleaned, skip = [], False
        for a in args:
            if skip:
                skip = False
                continue
            if a in ("-o", "-MF", "-MT", "-MQ"):
                skip = True
                continue
            if a in ("-c", "-MD", "-MMD"):
                continue
            cleaned.append(a)
        r = subprocess.run(cleaned + ["-E", "-dM"], cwd=entry["directory"], capture_output=True,
                           text=True, env=fs.tool_env())
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        macros = dict(re.findall(r"^#define (\w+) (.+)$", r.stdout, re.M))
        self.assertEqual(macros["MT9M114_CAMERA_SENSOR_MIPI_FRAME_WIDTH"].strip(), "320")
        self.assertEqual(macros["MT9M114_CAMERA_SENSOR_MIPI_FRAME_HEIGHT"].strip(), "320")
        self.assertEqual(macros["RTE_MT9M114_CAMERA_SENSOR_MIPI_IMAGE_CONFIG"].strip(), "5")

    def test_camera_interrupt_handler_is_in_the_image(self):
        self.assertRegex(fs.MAP.read_text(), r"\.text\.CAM_IRQHandler\s+Driver_CPI\.o")

    def test_image_fits_its_mram_load_region(self):
        m = re.search(r"Load Region LR_ROM \(Base: 0x([0-9a-f]+), Size: 0x([0-9a-f]+), Max: 0x([0-9a-f]+)",
                      fs.MAP.read_text())
        base, size, limit = (int(g, 16) for g in m.groups())
        self.assertEqual(base, 0x80200000, "the debug stub's boot address")
        self.assertLessEqual(size, limit, f"image {size:#x} overflows LR_ROM {limit:#x}")

    def test_model_and_ai_layer_are_unchanged(self):
        # (src/ is unchanged in step 2 itself; later steps add the camera code there.) The
        # letter classifier and its calibration data: the detector branch adds a second method
        # to model/model.py and the program, so the generated files are not compared.
        if git("rev-parse", "--verify", BASE_REF).returncode:
            self.skipTest(f"base ref {BASE_REF} not available")
        r = git("diff", "--stat", BASE_REF, "--", "model/fingerspell_letters.pt2", "model/calib_128.npy",
                "ai_layer/model_pte.h")
        self.assertEqual(r.stdout.strip(), "", r.stdout)


@unittest.skipUnless(os.environ.get("FS_HW") == "1", "set FS_HW=1 to flash and run the DevKit-E8")
class OnBoard(unittest.TestCase):
    """The camera stack is linked in but not started yet: the model test must still PASS."""

    @classmethod
    def setUpClass(cls):
        cls.uart = None
        cls.uart_error = None
        try:
            cls.uart = fs.UartCapture()
        except fs.UartBusy as e:
            cls.uart_error = str(e)
        cls.flash_log = fs.flash_and_start()
        cls.console = cls.uart.read_for(fs.BOOT_TIMEOUT, until=b"Test_result:") if cls.uart else ""
        if cls.uart:
            cls.console = cls.uart.read_for(1)  # the rest of the line
            cls.uart.close()

    def test_flash_is_programmed_and_verified(self):
        self.assertIn("O.K.", self.flash_log, self.flash_log[-1500:])
        self.assertNotRegex(self.flash_log, r"(?i)error|failed")

    def test_reset_vector_points_at_this_build(self):
        self.assertEqual(fs.read_word(0x80200004) & ~1, fs.symbol_address("Reset_Handler") & ~1)

    def test_app_is_healthy(self):
        """Running (the live camera loop never returns) or ended normally; no fault."""
        st = fs.core_state()
        self.assertIsNone(fs.app_problem(st), st)

    def test_uart_reports_pass(self):
        if self.uart_error:
            self.skipTest(self.uart_error)
        self.assertIn("Fingerspell letters:", self.console, self.console)
        self.assertRegex(self.console, r"Top-3: B [0-9.]+ .*\(expected B\)")
        self.assertIn("Test_result: PASS", self.console)


if __name__ == "__main__":
    unittest.main()
