"""Helpers for the Fingerspell board tests: find the toolchain, build, flash over
J-Link, read the UART, and inspect the core.

Stdlib only (plus PyYAML from the project venv), so the tests run with
`.venv/bin/python -m unittest discover tests`.

Environment overrides:
  FS_TOOLS_ROOT  vcpkg artifact root holding cmsis-toolbox, cmake, ninja, armclang
                 (default: the first ~/.vcpkg/artifacts/* that has cmsis-toolbox 2.15.0)
  FS_UART        serial port of the DevKit-E8 PRG USB (default: the first /dev/cu.usbmodem*)
  FS_HW=1        run the tests that flash and run the board
  FS_BOOT_TIMEOUT seconds to wait for UART output after a reset (default 90)
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
# Seconds to wait for the app's UART output after a reset. A boot normally prints within a
# second, but when the Secure Enclave answers slowly main()'s SE calls take ~25 s.
BOOT_TIMEOUT = float(os.environ.get("FS_BOOT_TIMEOUT", "90"))


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


# Do not touch the CPI registers (0x49030000) from the debugger: with the CPI clock
# off, a J-Link access there locked up the M55_HP debug port until a power cycle.
# The firmware stops the camera itself (fs_camera_stop) before main() returns.


def flash_and_start(hex_path: Path = HEX) -> str:
    return jlink(["h", f'loadfile "{hex_path}"', "r", "g"], timeout=300)


def reset_and_start() -> str:
    return jlink(["h", "r", "g"])


def se_stuck() -> bool:
    """True when the core sits in the SE services request loop (the Secure Enclave stopped
    answering; only a power cycle of the board clears it)."""
    out = jlink(["h", "g"])
    pc = int(re.search(r"^PC = ([0-9A-F]+)", out, re.M).group(1), 16)
    try:
        lo = symbol_address("SERVICES_send_request")
    except KeyError:
        return False
    return lo <= pc < lo + 0x100


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


# ---- camera crop dump -> PNG -------------------------------------------------------

def jlink_write_word(addr: int, value: int) -> str:
    return jlink(["h", f"w4 0x{addr:08X} 0x{value:08X}", "g"])


def parse_crop_dump(text: str) -> tuple[int, int, int, bytes]:
    """Parse the last 'CROPDUMP BEGIN w h frame n' ... 'CROPDUMP END' block: (w, h, frame, RGB bytes)."""
    m = list(re.finditer(r"CROPDUMP BEGIN (\d+) (\d+) frame (\d+)\r?\n(.*?)CROPDUMP END", text, re.S))
    if not m:
        raise ValueError("no complete CROPDUMP block in the UART text")
    w, h, frame, body = int(m[-1].group(1)), int(m[-1].group(2)), int(m[-1].group(3)), m[-1].group(4)
    rows = [r.strip() for r in body.splitlines() if r.strip()]
    if len(rows) != h or any(len(r) != w * 6 for r in rows):
        raise ValueError(f"malformed dump: {len(rows)} rows")
    return w, h, frame, bytes.fromhex("".join(rows))


def write_png(path: Path, w: int, h: int, rgb: bytes, scale: int = 1) -> None:
    import struct
    import zlib
    rows = []
    for y in range(h):
        row = rgb[y * w * 3:(y + 1) * w * 3]
        if scale > 1:
            row = b"".join(row[x * 3:x * 3 + 3] * scale for x in range(w))
        rows += [b"\x00" + row] * scale

    def chunk(t: bytes, d: bytes) -> bytes:
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w * scale, h * scale, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(b"".join(rows), 9)) + chunk(b"IEND", b"")
    Path(path).write_bytes(png)


def capture_crop(uart: "UartCapture", timeout: float = 30) -> tuple[int, int, int, bytes]:
    """Ask the running live loop for one crop dump and return it (needs the UART open)."""
    uart.data.clear()
    jlink_write_word(symbol_address("g_fs_dump_request"), 1)
    uart.read_for(timeout, until=b"CROPDUMP END")
    return parse_crop_dump(uart.text)


def app_problem(state: dict) -> str | None:
    """None when the app is healthy: either still running (DevKit-E8: the live camera loop
    never returns) or ended normally (main() returned: _sys_exit's semihosting BKPT faults
    with DEBUGEVT and no memory, bus or usage fault). Otherwise a description."""
    if state["exception"] == "NoException" or state["ipsr"] == 0:
        return None
    if state["exception"] == "HardFault" and state["hfsr"] == 0x80000000 and state["cfsr"] == 0:
        sys_exit = symbol_address("_sys_exit")
        if sys_exit <= state.get("stacked_pc", 0) < sys_exit + 16:
            return None
    return f"fault: {state['exception']} HFSR={state['hfsr']:#x} CFSR={state['cfsr']:#x} PC={state['pc']:#x}"
