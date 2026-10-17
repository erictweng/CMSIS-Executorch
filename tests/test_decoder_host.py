"""Decoder: src/fs_decoder.c (the fs_api.h Decoder contract) compiles as strict C99 on the
laptop and passes tests/host/test_fs_decoder.c, which feeds it synthetic probability streams
("HELLO" with the double L, "HI MOM" with the space, backspace/clear, buffer wrap, clock wrap).

    python3 -m unittest tests.test_decoder_host -v

No board, venv or toolchain needed: only a host C compiler (FS_CC, default clang, then cc).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SOURCES = [REPO / "src" / "fs_decoder.c", REPO / "tests" / "host" / "test_fs_decoder.c"]
CFLAGS = ["-std=c99", "-Wall", "-Wextra", "-Werror", "-pedantic", "-O1", "-I", str(REPO / "src")]


def host_cc() -> str | None:
    for cc in (os.environ.get("FS_CC"), "clang", "cc", "gcc"):
        if cc and shutil.which(cc):
            return cc
    return None


@unittest.skipUnless(host_cc(), "no host C compiler (clang/cc/gcc) on PATH")
class DecoderHost(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        exe = Path(cls.tmp.name) / "test_fs_decoder"
        cls.cc_result = subprocess.run([host_cc(), *CFLAGS, *map(str, SOURCES), "-o", str(exe)],
                                       capture_output=True, text=True)
        cls.run_result = (subprocess.run([str(exe)], capture_output=True, text=True)
                          if cls.cc_result.returncode == 0 else None)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_compiles_as_strict_c99_without_warnings(self):
        self.assertEqual(self.cc_result.returncode, 0, self.cc_result.stderr)
        self.assertEqual(self.cc_result.stderr.strip(), "")

    def test_host_suite_passes(self):
        self.assertIsNotNone(self.run_result, "did not compile")
        out = self.run_result.stdout + self.run_result.stderr
        self.assertEqual(self.run_result.returncode, 0, out)
        self.assertNotIn("FAIL", out)
        self.assertRegex(out, r"fs_decoder: \d+ checks, 0 failed")


if __name__ == "__main__":
    unittest.main()
