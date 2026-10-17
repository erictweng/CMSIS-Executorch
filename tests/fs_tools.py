"""Helpers for the Fingerspell board tests: find the toolchain, build, flash over
J-Link, read the UART, and inspect the core.

Stdlib only (plus PyYAML from the project venv), so the tests run with
`.venv/bin/python -m unittest discover tests`.

Environment overrides:
  FS_TOOLS_ROOT  vcpkg artifact root holding cmsis-toolbox, cmake, ninja, armclang
                 (default: the first ~/.vcpkg/artifacts/* that has cmsis-toolbox 2.15.0)
  FS_UART        serial port of the DevKit-E8 PRG USB (default: the first /dev/cu.usbmodem*)
  FS_HW=1        run the tests that flash and run the board
"""

from __future__ import annotations

import glob
import os
import re
import select
import subprocess
import tempfile
import termios
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SOLUTION = REPO / "cmsis-executorch.csolution.yml"
TARGET = "DevKit-E8"
OUT = REPO / "out" / "cmsis-executorch" / TARGET / "Debug"
AXF = OUT / "cmsis-executorch.axf"
HEX = OUT / "cmsis-executorch.hex"
MAP = OUT / "cmsis-executorch.axf.map"
CBUILD_YML = OUT / f"cmsis-executorch.Debug+{TARGET}.cbuild.yml"
COMPILE_COMMANDS = OUT / "compile_commands.json"
BOARD = REPO / "board" / TARGET
LAYER = BOARD / "Board-U85.clayer.yml"
RTE_DEVICE = BOARD / "RTE" / "Device" / "AE822FA0E5597LS0_M55_HP" / "RTE_Device.h"
JLINK_DEVICE = "AE822FA0E5597LS0_M55_HP"


def tools_root() -> Path:
    if os.environ.get("FS_TOOLS_ROOT"):
        return Path(os.environ["FS_TOOLS_ROOT"]).expanduser()
    for root in sorted(glob.glob(os.path.expanduser("~/.vcpkg/artifacts/*"))):
        if (Path(root) / "tools.open.cmsis.pack.cmsis.toolbox" / "2.15.0" / "bin" / "cbuild").exists():
            return Path(root)
    raise FileNotFoundError("CMSIS-Toolbox 2.15.0 not found under ~/.vcpkg/artifacts; set FS_TOOLS_ROOT")


def tool_env() -> dict[str, str]:
    """PATH and toolchain variables for CMSIS-Toolbox 2.15.0 (not VS Code's bundled 2.13)."""
    t = tools_root()
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join([
        str(t / "tools.open.cmsis.pack.cmsis.toolbox" / "2.15.0" / "bin"),
        str(t / "tools.kitware.cmake" / "4.2.1" / "bin"),
        str(t / "tools.ninja.build.ninja" / "1.13.2"),
        str(t / "compilers.arm.armclang" / "6.24.0" / "bin"),
        env.get("PATH", ""),
    ])
    env["AC6_TOOLCHAIN_6_24_0"] = str(t / "compilers.arm.armclang" / "6.24.0" / "bin")
    env.setdefault("CMSIS_PACK_ROOT", os.path.expanduser("~/.cache/arm/packs"))
    return env


def run(cmd: list[str], timeout: float = 900, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=REPO, env=tool_env(), capture_output=True, text=True,
                          timeout=timeout, **kw)


def cbuild_setup() -> subprocess.CompletedProcess:
    return run(["cbuild", "setup", str(SOLUTION), "--active", TARGET, "--packs"])


def cbuild() -> subprocess.CompletedProcess:
    return run(["cbuild", str(SOLUTION), "--active", TARGET])


def gcc_tool(name: str) -> str:
    return str(tools_root() / "compilers.arm.arm.none.eabi.gcc" / "14.3.1" / "bin" / f"arm-none-eabi-{name}")


def symbol_address(name: str) -> int:
    out = subprocess.run([gcc_tool("nm"), str(AXF)], capture_output=True, text=True, check=True).stdout
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[2] == name:
            return int(parts[0], 16)
    raise KeyError(name)


def rte_define(name: str, path: Path = RTE_DEVICE) -> str | None:
    m = re.search(rf"^#define\s+{re.escape(name)}\s+(\S+)", path.read_text(), re.M)
    return m.group(1) if m else None


# ---- board: J-Link ---------------------------------------------------------------

def jlink(commands: list[str], timeout: float = 120) -> str:
    """Run JLinkExe with a command file over SWD; returns its combined output."""
    with tempfile.NamedTemporaryFile("w", suffix=".jlink", delete=False) as f:
        f.write("\n".join(["connect", *commands, "qc"]) + "\n")
        script = f.name
    try:
        r = subprocess.run(["JLinkExe", "-device", JLINK_DEVICE, "-if", "SWD", "-speed", "4000",
                            "-autoconnect", "1", "-NoGui", "1", "-CommandFile", script],
                           capture_output=True, text=True, timeout=timeout)
        return r.stdout + r.stderr
    finally:
        os.unlink(script)


def flash_and_start(hex_path: Path = HEX) -> str:
    return jlink(["h", f'loadfile "{hex_path}"', "r", "g"], timeout=300)


def read_word(addr: int) -> int:
    out = jlink(["h", f"mem32 0x{addr:08X} 1", "g"])
    m = re.search(rf"{addr:08X} = ([0-9A-F]{{8}})", out)
    if not m:
        raise RuntimeError(f"could not read 0x{addr:08X}:\n{out[-800:]}")
    return int(m.group(1), 16)


def core_state() -> dict[str, int | str]:
    """Halt briefly, read PC, IPSR, the fault status registers and the stacked PC, resume."""
    out = jlink(["h", "mem32 0xE000ED28 1", "mem32 0xE000ED2C 1", "g"])
    pc = int(re.search(r"^PC = ([0-9A-F]+)", out, re.M).group(1), 16)
    ipsr = re.search(r"IPSR = ([0-9A-F]+) \(([^)]*)\)", out)
    sp = int(re.search(r"^SP\(R13\)= ([0-9A-F]+)", out, re.M).group(1), 16)
    cfsr = int(re.search(r"E000ED28 = ([0-9A-F]{8})", out).group(1), 16)
    hfsr = int(re.search(r"E000ED2C = ([0-9A-F]{8})", out).group(1), 16)
    state: dict[str, int | str] = {"pc": pc, "ipsr": int(ipsr.group(1), 16), "exception": ipsr.group(2),
                                   "sp": sp, "cfsr": cfsr, "hfsr": hfsr}
    if state["ipsr"] == 3:  # HardFault: the exception frame's PC is at SP + 24
        state["stacked_pc"] = read_word(sp + 24)
    return state


# ---- board: UART -------------------------------------------------------------------

def uart_port() -> str:
    if os.environ.get("FS_UART"):
        return os.environ["FS_UART"]
    ports = sorted(glob.glob("/dev/cu.usbmodem*"))
    if not ports:
        raise FileNotFoundError("no /dev/cu.usbmodem* port: is the DevKit-E8 PRG USB connected?")
    return ports[0]


class UartBusy(RuntimeError):
    pass


class UartCapture:
    """Open the UART at 115200 8N1 before the board starts, collect bytes in the background
    of the caller's own work, and return them as text."""

    def __init__(self, port: str | None = None):
        self.port = port or uart_port()
        try:
            self.fd = os.open(self.port, os.O_RDONLY | os.O_NOCTTY | os.O_NONBLOCK)
        except OSError as e:
            if e.errno == 16:
                raise UartBusy(f"{self.port} is busy: close the VS Code Serial Monitor") from e
            raise
        a = termios.tcgetattr(self.fd)
        a[0] = 0
        a[1] = 0
        a[2] = termios.CS8 | termios.CREAD | termios.CLOCAL
        a[3] = 0
        a[4] = a[5] = termios.B115200
        termios.tcsetattr(self.fd, termios.TCSANOW, a)
        termios.tcflush(self.fd, termios.TCIFLUSH)
        self.data = bytearray()

    def read_for(self, seconds: float, until: bytes | None = None) -> str:
        end = time.time() + seconds
        while time.time() < end:
            r, _, _ = select.select([self.fd], [], [], 0.1)
            if r:
                try:
                    self.data += os.read(self.fd, 4096)
                except BlockingIOError:
                    pass
                if until is not None and until in self.data:
                    break
        return self.text

    @property
    def text(self) -> str:
        return self.data.decode("utf-8", "replace")

    def close(self) -> None:
        os.close(self.fd)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
