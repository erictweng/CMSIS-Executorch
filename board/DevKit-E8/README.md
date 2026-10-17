# Board layer: Alif Ensemble E8 DevKit (M55_HP + Ethos-U85)

Device: `Alif Semiconductor::AE822FA0E5597LS0:M55_HP` (Cortex-M55 high-performance
core, 400 MHz) with the Ethos-U85 (256 MACs) of the Ensemble E8.

Derived from the Ensemble pack's `Boards/DevKit-e8/Layers/M55_HP/Board_HP-U85.clayer.yml`
with MT9M114 camera input and ILI9806E LCD output. Ethernet, USB and VIO
drivers are left out. The target-set in the
csolution debugs it through the on-board J-Link (`J-Link Server`, SWD at
4 MHz, `start-pname: M55_HP`).

| File | Purpose |
|------|---------|
| `Board-U85.clayer.yml` | Layer: startup, SE services, UART4 stdio, Ethos-U85 driver, memory placement |
| `main.c` | Pin/GPIO config, SE services, clocks, stdio, NPU init, then `app_main()` |
| `retarget_stdio.c` | `stdio_init()` for UART4 through the CMSIS USART driver (115200 8N1) |
| `ethos_setup.c` | Ethos-U85 driver init at `NPU_HG_BASE`, IRQ 366, prints the NPU banner |
| `ethosu_cb_dcache.c` | D-cache clean/invalidate hooks for NPU buffers outside the TCMs |
| `../../src/fs_display.cpp` | LCD power/driver setup and RGB565 photo/text drawing through the UI API |
| `linker_ac6_mram.sct.src`, `linker_gnu_mram.ld.src` | Pack linker scripts plus a 32 kB stack and the `.bss.ai_pool` section in bulk SRAM |
| `RTE/` | Configuration files carried with the layer (see below) |

## LCD output

The DevKit-E8 layer enables `FINGERSPELL_LCD` and the ILI9806E/CDC200/MIPI DSI
drivers. `src/fs_display.cpp` implements `fs_ui_init()` and `fs_ui_render()`
from `src/fs_api.h`. The LCD uses 480x800 RGB565 (`RTE_CDC200_PIXEL_FORMAT 2`).

At boot it shows the saved test crop, predicted letter, confidence and
PASS/FAIL. After camera startup, `live_loop()` supplies its existing 128x128
RGB888 crop, the upstream decoder state and stage timings to the display.
The live screen shows the crop, current letter/confidence, the last 52 text
characters, NPU time and loop FPS. Camera capture and inference still run
continuously; the decoder's existing three-frame commit rule is used.

The LCD module borrows the crop through `fs_display_set_frame()`; keep that
buffer valid until `fs_ui_render()` returns. It uses the shared decoder types,
without another decoder implementation. If LCD startup fails, the existing
top-3 UART stream continues and `fs_ui_render()` prints the text/state fallback.

The GNU script puts the 768000-byte framebuffer in SRAM1 and includes it in
the startup zero table. SRAM0 keeps the 3 MiB AI pools and 618496-byte camera
ring/canary. Together those buffers would exceed SRAM0's 4 MiB capacity.
The AC6 script uses the combined 8 MiB region, or SRAM1 for LCD scanout when
the regions are configured separately. Display power setup preserves the
current SE profile and requests both SRAM banks and the MIPI power domains.

This integration needs a board check with both the MT9M114 and LCD attached:
confirm the boot image/PASS screen, live crop/letters/text, camera error counters,
render time (the UI contract targets under 10 ms) and reset while streaming.
Builds and host tests do not verify camera/LCD operation together.

Local validation: GCC 14.3.1 builds passed for DevKit-E8 and the headless
SSE-320-U85 target. The decoder host suite and five camera source checks
passed; hardware tests were skipped. The DevKit-E8 ELF places the LCD buffer
at `0x02400000` in SRAM1, with no overlap with the AI/camera buffers and a
matching startup zero-table entry. AC6 validation was blocked before source
compilation because this Mac has no Arm Compiler license configured.

## Memory layout

| Region | Address | Used for |
|--------|---------|----------|
| MRAM (HP application region) | `0x80200000`, 2 MB | Constants, the embedded `.pte` model; with AC6 also the code, which executes in place. With GCC and Clang the startup code executes from MRAM and copies the rest of the code to ITCM (`linker_gnu_mram.ld.src`) |
| DTCM (SRAM3) | `0x20000000` (core alias; `0x50800000` global), 1 MB | `.data`/`.bss`, 96 kB heap, 32 kB stack |
| SRAM0/SRAM1 (bulk) | `0x02000000`, 8 MB combined (`SRAM0_SRAM1_COMBINED` in `app_mem_regions.h`; the GNU script uses SRAM0 alone, 4 MB) | `.bss.ai_pool`: the runner's 1 MB method pool and 2 MB temp pool (NPU scratch) |

The pool sizes and section come from the `define:` node of the layer
(`APP_METHOD_POOL_SIZE`, `APP_TEMP_POOL_SIZE`, `APP_POOL_SECTION`) and are
consumed by `src/app_main.cpp`.

## RTE configuration

`RTE/` is committed for this layer, as for the Corstone-320 one; here it
matters: Alif's stdio retarget refuses to build unless UART4 is in
polling mode (`RTE_UART4_BLOCKING_MODE_ENABLE 1` in `RTE_Device.h`), and the
Conductor-generated `pins.h`/`board_defs.h` differ from the pack defaults. The
files are the pack's own DevKit-E8 layer configuration.

## STDIO

UART4 on the PRG USB connector, 115200 baud. Set **SW4 to UART4** for the
console; SW4 in position **SEUART** (the default) routes the same USB serial
port to the Secure Enclave for SETOOLS. The on-board J-Link sits behind the
same connector.

## Before the first debug session

Program the debug stubs into the device's ATOC with Alif SETOOLS once:
**Terminal > Run Task > Alif: Install M55_HP debug stubs (DevKit-E8, single core configuration)**. The task needs
the VS Code setting `alif.setools.root` and SW4 in position SEUART. See
[the getting-started README](../../README.md).
