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

The layer defines `FINGERSPELL_LCD` and adds the CDC200, MIPI DSI and ILI9806E
components (480x800 portrait panel on J21, two DSI lanes, RGB565 via
`RTE_CDC200_PIXEL_FORMAT 2`). `src/fs_display.cpp` implements `fs_ui_init()` and
`fs_ui_render()` from `src/fs_api.h`:

- top: `FINGERSPELL`; the 128x128 crop the model sees, scaled 3x (384x384) in a
  1 px frame (the sensor's orientation already gives a mirror image; `kMirrorX` and
  `kFlipY` in `fs_display.cpp` flip the preview only, never the model input);
- the model's top letter in big type with its probability as a percentage and a
  bar, green once the decoder holds it above its commit threshold, grey below;
- the 2nd/3rd guesses, then `CAM .. NPU .. UI .. MS  .. FPS`;
- bottom: the text the decoder has committed.

At boot the same screen shows the baked-in test crop and PASS/FAIL until the
camera starts. `live_loop()` hands the crop (`fs_display_set_frame`), the top-3
(`fs_display_set_top3`), the decoder and the timings to the display every frame
and prints the render time as `ui x.xms` in the live UART line. The static parts
are drawn once; per frame only the crop is rewritten (whole rows) and a text
field when its string changes, and only the touched rows are cleaned out of the
data cache for the CDC200: about 1.8 ms per frame, the loop stays at ~19 fps.

If the panel does not start, the live UART line continues and `fs_ui_render()`
prints the contract's `TEXT|live|conf|fps|npu_ms` fallback line per frame.
Display power is the VBAT D-PHY bits only (`main.c`'s `vbat_init()`); the layer
never calls `SERVICES_set_run_cfg` with memory blocks, which once left MRAM/SRAM0
unusable until a SETOOLS recovery.

## Memory layout

| Region | Address | Used for |
|--------|---------|----------|
| MRAM (HP application region) | `0x80200000`, 2 MB | Constants, the embedded `.pte` model; with AC6 also the code, which executes in place. With GCC and Clang the startup code executes from MRAM and copies the rest of the code to ITCM (`linker_gnu_mram.ld.src`) |
| DTCM (SRAM3) | `0x20000000` (core alias; `0x50800000` global), 1 MB | `.data`/`.bss`, 96 kB heap, 32 kB stack |
| SRAM0 (bulk) | `0x02010000`, 4 MB less the 64 kB A32 boot stub the ATOC loads at `0x02000000` (`APP_SRAM_POOL_OFFSET`, `SRAM0_SRAM1_COMBINED 0` in `app_mem_regions.h`) | `.bss.ai_pool`: the camera ring (0.6 MB), the runner's 1 MB method pool and 1 MB temp pool (NPU scratch); `.bss.lcd_frame_buf`: the 0.75 MB RGB565 frame buffer. 3.3 MB in all |
| SRAM1 | `0x02400000`, 4 MB | Empty: the Ethos-U85 cannot reach it |

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
