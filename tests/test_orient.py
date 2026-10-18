"""Camera roll from gravity and landmark levelling (src/fs_orient.c vs ml/handlandmark.py).

    python3 -m unittest tests.test_orient -v

No board or venv needed: a host C compiler and numpy.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml"))
import handlandmark as hl  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
SOURCES = [SRC / "fs_orient.c", REPO / "tests" / "host" / "test_fs_orient.c"]
CFLAGS = ["-std=c99", "-Wall", "-Wextra", "-Werror", "-pedantic", "-O1", "-I", str(SRC)]
# A made-up mounting: image down = -Y of the sensor, image right = +X (any orthonormal pair works).
DOWN, RIGHT = np.array([0.0, -1.0, 0.0]), np.array([1.0, 0.0, 0.0])


def host_cc() -> str | None:
    for cc in (os.environ.get("FS_CC"), "clang", "cc", "gcc"):
        if cc and shutil.which(cc):
            return cc
    return None


def gravity_for_roll(deg: float) -> np.ndarray:
    """The sensor's gravity reading when the camera is rolled `deg` (gravity's image direction
    is `deg` from image-down toward image-right)."""
    r = np.radians(deg)
    return np.cos(r) * DOWN + np.sin(r) * RIGHT


@unittest.skipUnless(host_cc(), "no host C compiler (clang/cc/gcc) on PATH")
class OrientHost(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.exe = Path(cls.tmp.name) / "test_fs_orient"
        cls.cc = subprocess.run([host_cc(), *CFLAGS, *map(str, SOURCES), "-lm", "-o", str(cls.exe)],
                                capture_output=True, text=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_mode(self, *args) -> str:
        self.assertEqual(self.cc.returncode, 0, self.cc.stderr)
        r = subprocess.run([str(self.exe), *map(str, args)], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        return r.stdout

    def c_roll(self, g, down=DOWN, right=RIGHT) -> float:
        return float(self.run_mode("roll", *down, *right, *g))

    def test_compiles_as_strict_c99_without_warnings(self):
        self.assertEqual(self.cc.returncode, 0, self.cc.stderr)
        self.assertEqual(self.cc.stderr.strip(), "")

    def test_roll_follows_the_camera(self):
        for deg in (0, 15, -15, 45, 90, -90, 135, 179):
            with self.subTest(deg=deg):
                g = gravity_for_roll(deg)
                self.assertAlmostEqual(self.c_roll(g), deg, places=3)
                self.assertAlmostEqual(np.degrees(hl.roll(DOWN, RIGHT, g)), deg, places=5)
        # The calibration poses themselves: level gives 0, rolled 90 clockwise gives 90.
        self.assertAlmostEqual(self.c_roll(DOWN), 0.0, places=5)
        self.assertAlmostEqual(self.c_roll(RIGHT), 90.0, places=4)
        # Pitch (gravity tilting along the optical axis) does not roll the image.
        axis = np.cross(DOWN, RIGHT)
        self.assertAlmostEqual(self.c_roll(0.8 * DOWN + 0.6 * axis), 0.0, places=4)

    def test_roll_is_zero_without_a_calibration_or_a_reading(self):
        self.assertEqual(self.c_roll(gravity_for_roll(40), down=np.zeros(3), right=np.zeros(3)), 0.0)
        self.assertEqual(self.c_roll(np.zeros(3)), 0.0)
        self.assertEqual(self.c_roll(0.2 * gravity_for_roll(40)), 0.0)  # below ORIENT_MIN_G
        self.assertEqual(hl.roll(np.zeros(3), np.zeros(3), gravity_for_roll(40)), 0.0)
        self.assertEqual(hl.roll(DOWN, RIGHT, 0.2 * gravity_for_roll(40)), 0.0)
        # The camera looking along gravity (board flat, lens up or down): no image "down", roll 0,
        # even with a little in-plane leakage from the mounting.
        axis = np.cross(DOWN, RIGHT)
        for g in (axis, -axis, 0.98 * axis + 0.2 * gravity_for_roll(64)):
            self.assertEqual(self.c_roll(g), 0.0)
            self.assertEqual(hl.roll(DOWN, RIGHT, g), 0.0)
        # Tilted well past that (0.5 g in the plane) the roll is defined again.
        g = 0.85 * axis + 0.5 * gravity_for_roll(64)
        self.assertAlmostEqual(self.c_roll(g), 64.0, places=3)
        self.assertIn("#define FS_ORIENT_MIN_INPLANE 0.3f", (SRC / "fs_orient.h").read_text())

    def test_levelling_puts_gravity_straight_down(self):
        """A hand whose wrist->MCP vector points at `deg` from image-up in a camera rolled by
        `deg` is upright once levelled; the C and numpy versions agree; roll 0 is the identity."""
        rng = np.random.default_rng(1)
        pts = rng.uniform(50, 270, (21, 3))
        about = pts[0, :2].copy()
        path = Path(self.tmp.name) / "pts.txt"
        path.write_text("\n".join(f"{x} {y} {z}" for x, y, z in pts) + "\n")
        for deg in (0, 30, -60, 90, 170):
            with self.subTest(deg=deg):
                got = np.array([[float(v) for v in line.split()] for line in self.run_mode("level", deg, *about, path).splitlines()])
                want = hl.level(pts, np.radians(deg), about)
                np.testing.assert_allclose(got, want, atol=1e-3)
                np.testing.assert_allclose(got[:, 2], pts[:, 2])  # z untouched
                np.testing.assert_allclose(got[0, :2], about, atol=1e-4)  # the pivot stays
        np.testing.assert_allclose(hl.level(pts, 0.0, about), pts)
        # Gravity's image direction for a roll of `deg` is (sin, cos) in (x, y); levelling by
        # that roll maps it to straight down (0, 1).
        for deg in (25, -70, 120):
            r = np.radians(deg)
            v = hl.level(np.array([[0.0, 0.0, 0.0], [np.sin(r), np.cos(r), 0.0]]), r, (0.0, 0.0))
            np.testing.assert_allclose(v[1, :2], [0.0, 1.0], atol=1e-6)
        # A hand upright in the world points against gravity. In a camera rolled by `deg`
        # gravity's image direction is (sin, cos), so the hand's wrist->MCP vector appears as
        # (-sin, -cos); after levelling it points image-up (0, -1) again.
        for deg in (35, -80):
            r = np.radians(deg)
            mcp = np.array([-np.sin(r), -np.cos(r), 0.0])
            lv = hl.level(np.array([[0.0, 0.0, 0.0], mcp]), r, (0.0, 0.0))
            np.testing.assert_allclose(lv[1, :2], [0.0, -1.0], atol=1e-6)

    def test_gravity_average_converges_and_resets(self):
        readings = [gravity_for_roll(20)] * 40 + ["reset"] + [gravity_for_roll(-50)] * 3
        path = Path(self.tmp.name) / "g.txt"
        path.write_text("\n".join("reset" if isinstance(r, str) else " ".join(map(str, r)) for r in readings) + "\n")
        got = np.array([[float(v) for v in line.split()] for line in self.run_mode("gravity", hl.ORIENT_ALPHA, path).splitlines()])
        self.assertEqual(got.shape, (43, 3))
        np.testing.assert_allclose(got[0], gravity_for_roll(20), atol=1e-5)  # first frame copies
        np.testing.assert_allclose(got[39], gravity_for_roll(20), atol=1e-4)  # converged
        np.testing.assert_allclose(got[40], gravity_for_roll(-50), atol=1e-5)  # reset restarts from raw
        # A step from 20 to -50 deg settles within ~1 deg after 40 frames at alpha 0.1 (the EMA
        # of the vector, then the roll of the average).
        path.write_text("\n".join(" ".join(map(str, gravity_for_roll(20))) for _ in range(5))
                        + "\n" + "\n".join(" ".join(map(str, gravity_for_roll(-50))) for _ in range(40)) + "\n")
        got = np.array([[float(v) for v in line.split()] for line in self.run_mode("gravity", hl.ORIENT_ALPHA, path).splitlines()])
        self.assertLess(abs(np.degrees(hl.roll(DOWN, RIGHT, got[-1])) + 50), 1.0)
        self.assertIn("#define FS_ORIENT_ALPHA 0.1f", (SRC / "fs_orient.h").read_text())


if __name__ == "__main__":
    unittest.main()
