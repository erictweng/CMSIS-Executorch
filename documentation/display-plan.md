# Fingerspell on the DevKit-E8 LCD: plan and context

This is a handoff for an agent (or a teammate) adding the display to the Fingerspell
firmware. It is meant to be read top to bottom before touching code. The display is the
**P4 / S3 "display" stream** of the team plan (fingerspell repo `docs/S1_CAMERA_PLAN.md`,
`docs/SPRINTS.md`: "display bring-up: text line + live guess + confidence bar"; the
fallback if it slips is the UART output).

**Goal:** the 4.3" LCD shows what the camera sees (the 128x128 hand-box crop the model
gets), the live letter in big type, its confidence as a bar, and fps, updated every frame.

## 1. Where things stand (2026-10-17)

- Repo: `erictweng/CMSIS-Executorch` (fork of Arm-Examples), remote `fork`. Work on a
  branch from **`fingerspell`** and open the PR against `fingerspell` on the fork.
- `fingerspell` has: the Fingerspell model (MobileNetV2-0.5, 128x128, int8, 55/55 ops on
  the Ethos-U85); the MT9M114 camera at 320x320 RGB565 (`src/fs_camera.c/.h`); the live
  loop in `src/app_main.cpp` (`live_loop()`: crop -> preprocess -> model -> top-3, one UART
  line per frame, ~19 fps); the team contract `src/fs_api.h`; board tests in `tests/`.
- Docs to read first: `documentation/fingerspell-bringup.md` (build, flash, gotchas),
  `board/DevKit-E8/README.md`, and on branch **`hackathon-pico-faces`**:
  `board/DevKit-E8/README.md` and `board/DevKit-E8/display.c` (a working LCD driver for
  this board).
- Live UART line today: `X 0.30 U 0.21 M 0.06 | cam 25.0ms pre 1.0ms npu 7.8ms | 19.8 fps`.

## 2. Do these first: three board fixes from `hackathon-pico-faces`

pico-faces found these on the same board. `fingerspell` does not have them yet, and they
are likely behind the spontaneous reboots and "boot hangs in the first NPU inference"
seen on `fingerspell`. Land them as their own commit, before the display.

1. **The A32 boot stub lives at `0x02000000`.** The debug-stub ATOC
   (`.alif/M55_HP_mram_cfg.json`, entry `A32_STUB`, `loadAddress 0x02000000`, flags
   `load, boot`) makes the Secure Enclave load and start an A32 stub at the start of SRAM0.
   On `fingerspell` the camera ring `s_ring` starts exactly there (`nm`: `02000000 s_ring`),
   so the camera DMA overwrites the running A32 code 25 times a second. Fix as in
   pico-faces: in `RTE/Device/AE822FA0E5597LS0_M55_HP/app_mem_regions.h` add
   `#define APP_SRAM_POOL_OFFSET 0x00010000` and start the bulk-SRAM execution regions in
   `linker_ac6_mram.sct.src` / `linker_gnu_mram.ld.src` at `BASE + APP_SRAM_POOL_OFFSET`
   (copy the pico-faces diff: `git diff fingerspell hackathon-pico-faces -- board/DevKit-E8/linker_ac6_mram.sct.src`).
2. **The Ethos-U85 cannot reach SRAM1.** pico-faces sets `SRAM0_SRAM1_COMBINED 0` and
   keeps everything the NPU touches in SRAM0 (4 MB at `0x02000000`, minus the 64 kB stub).
   `fingerspell` still has `SRAM0_SRAM1_COMBINED 1` (8 MB). Today's SRAM0 use is
   0x02000000-0x02397000 (ring 0.6 MB + method pool 1 MB + temp pool 2 MB). The LCD frame
   buffer (1.15 MB) does not fit on top: shrink `APP_TEMP_POOL_SIZE` (Vela needs 321 KiB
   for this model; 0x80000 = 512 KB is plenty, verify with the tests) so ring + pools + LCD
   stay in SRAM0, or put the LCD frame buffer in SRAM1 only after checking the CDC200 can
   read it there. Check placement with `arm-none-eabi-nm -S -n` on the `.axf`.
3. **Stale cache lines after flash + warm reset.** pico-faces' `main.c` runs
   `SCB_CleanInvalidateDCache(); SCB_InvalidateICache();` first thing in `main()`: "after a
   flash update and a warm reset from the debugger the data cache can still hold lines of
   the previous image, and the runtime then reads a corrupted program". Add it to
   `board/DevKit-E8/main.c` **after** `stop_leftover_camera_capture()` and before
   `board_pins_config()`.

Keep everything else in `fingerspell`'s `main.c` (especially `stop_leftover_camera_capture()`).

## 3. The display hardware and the code to reuse

- Panel: ILI9806E, 480 x 800 portrait, connector J21, 2-lane MIPI DSI, driven by the CDC200
  display controller. `RTE_Device.h` on `fingerspell` already has it configured
  (`RTE_CDC200 1`, `RTE_CDC200_PIXEL_FORMAT 1` = RGB888, `RTE_PANEL_HACTIVE_TIME 480`,
  `RTE_PANEL_VACTIVE_LINE 800`, `RTE_MIPI_DSI 1`, `RTE_ILI9806E_PANEL_DSI_N_LANES 2`);
  `board_defs.h` has the LCD backlight (GPIO6_1) and reset (GPIO15) pins.
- Components to add to `board/DevKit-E8/Board-U85.clayer.yml` (none of them is in the
  `fingerspell` build yet): `Device:SOC Peripherals:CDC`, `Device:SOC Peripherals:MIPI DSI`,
  `BSP:External peripherals:ILI9806E LCD panel`. `MIPI DSI CSI2 DPHY` and `GPIO` are already
  there (the camera uses them). Run `cbuild setup ... --packs` after editing and fix any
  "dependency validation" message it prints.
- MIPI D-PHY power: the display needs the TX D-PHY and the PLL powered (`VBAT->PWR_CTRL`).
  `fs_camera.c::mipi_dphy_power_on()` already clears the TX, RX and PLL power/isolation
  bits, but only when the camera starts; if the display comes up first, power it in
  `main.c` (pico-faces' `vbat_init()` does exactly this) and keep the camera's call harmless.
- Driver: copy `board/DevKit-E8/display.c` from `hackathon-pico-faces` unchanged as a
  start. `board_display_image(rgb, w, h, ch)` brings up CDC200 + DSI + panel on first use
  (`Initialize`, `PowerControl(FULL)`, `Control(CDC200_CONFIGURE_DISPLAY, framebuffer)`,
  `Start`) and blits an RGB image scaled by the largest integer factor, centred, into the
  RGB888 frame buffer `lcd_frame[800][480][3]` in section `.bss.lcd_frame_buf` (the
  `fingerspell` linker scripts already route that section to bulk SRAM). Pixel byte order
  in memory is **B, G, R**. Each blit ends with `SCB_CleanDCache_by_Addr` on the rows it
  wrote. pico-faces showed 128x128 images scaled 3x to 384x384.

## 4. What to build

Implement the UI half of the team contract in `src/fs_api.h` in a new `src/fs_ui.c`
(`fs_ui_init()`, `fs_ui_render(const fs_decoder_t*, const fs_metrics_t*, fs_event_t)`), on
top of a small board API in `display.c`:

- `board_display_image()` (exists) plus two helpers: fill a rectangle with a colour, and
  draw text with an embedded bitmap font (an 8x8 or 5x7 ASCII font, scaled by an integer
  factor; keep it in flash as `const`). No other dependencies.
- Layout on the 480x800 portrait panel (suggested):
  - top 40 px: title `FINGERSPELL`, small;
  - the camera crop, 128x128 scaled 3x = 384x384, centred horizontally, y 48..432, with a
    1 px frame so the hand box is obvious;
  - below: the live letter, very large (font scaled ~12x, ~96 px tall), and its confidence
    as a percentage plus a horizontal bar (green at >= 0.6, the decoder's commit threshold;
    grey below);
  - a line with the 2nd/3rd guesses, and `cam/npu ms, fps`;
  - bottom: the committed text from the decoder (`fs_decoder_t::text`), empty until the S3
    decoder lands.
- Wire it into `live_loop()` in `src/app_main.cpp` behind `#ifdef APP_DISPLAY` (a define in
  the DevKit-E8 layer, as pico-faces does), after the UART line: show `g_crop` and the
  top-3. The FVP and AppKit-E8 targets must still build without it (`fs_camera.c` uses
  `for-context: +DevKit-E8` in `cmsis-executorch.cproject.yml`; do the same for
  `fs_ui.c`, or guard it).
- Budget: `fs_api.h` says `fs_ui_render` must return in < 10 ms. The 384x384 blit is
  ~442k pixel writes; redraw only what changed (the crop every frame, text only when the
  letter or value changes) and clean only the dirty rows. Do not let the loop drop far
  below today's ~19 fps; print the render time in the UART line (`ui x.xms`).
- Mirror check: confirm with a hand whether the crop needs a horizontal flip so the user
  sees a mirror image; if so, flip only the displayed image, never the model input.

## 5. Rules learned the hard way on this board

- **Never re-arm the CPI per frame and never use the legacy single-buffer mode**: the CPI
  then writes frames to ever higher addresses (~5 MB/s) through SRAM and beyond. The camera
  runs in CPI streaming mode (`RTE_CPI_STREAMING_ENABLE 1`, 3 buffers) and `fs_camera.c`
  has a canary that stops it on overrun. Keep both.
- **Do not access CPI registers (`0x49030000`) from J-Link**: with the CPI clock off it locks
  the debug port until a power cycle.
- **Do not call `SERVICES_set_run_cfg` with memory blocks** (pico-faces README): it left
  MRAM/SRAM0 unusable until a SETOOLS maintenance recovery.
- **J-Link and a board that booted on its own**: connecting J-Link to a board running
  standalone with the camera streaming preceded one lock-up. Prefer: flash with J-Link,
  then test over the UART only; read memory over J-Link only in a session that has had the
  debugger attached since the flash.
- **VS Code holds the serial port** (`/dev/tty.usbmodem0012195786141`) whenever its Serial
  Monitor or another serial extension is connected, even after "Stop Monitoring"; the tools
  then fail with "Resource busy". `lsof /dev/tty.usbmodem*` shows the holder; quit VS Code.
- **VS Code's CMSIS Solution 1.66 bundles CMSIS-Toolbox 2.13**, which rejects the
  csolution's `mlops:` node; build from a terminal with 2.15.0 (section 6).
- The DWT cycle counter used for the timings may not run without a debugger on some boots;
  if the UART shows `0.0ms`/`0.0 fps`, switch the timing to SysTick.

### Recovering the board
- No UART output, J-Link "Could not connect" / "Failed to power up DAP": unplug the PRG USB
  for 10 s, replug (check macOS did not ask "Allow accessory to connect?").
- Still unreachable: reinstall the debug stub. SW4 to **SEUART**, port free, then
  `cp .alif/M55_HP_mram_cfg.json ~/Documents/app-release-exec-macos/build/config/ &&
  cp .alif/M55_HP_mram_stub.bin ~/Documents/app-release-exec-macos/build/images/ &&
  cd ~/Documents/app-release-exec-macos && ./app-gen-toc -f build/config/M55_HP_mram_cfg.json && ./app-write-mram -p`
  (if "Target did not respond", run it in a retry loop and power-cycle the board). SW4
  back to **UART4**, then flash.

## 6. Build, flash, test

```bash
# toolchain (CMSIS-Toolbox 2.15.0, Arm Compiler 6.24) for a plain shell
T=~/.vcpkg/artifacts/2139c4c6
export PATH=$T/tools.open.cmsis.pack.cmsis.toolbox/2.15.0/bin:$T/tools.kitware.cmake/4.2.1/bin:$T/tools.ninja.build.ninja/1.13.2:$PATH
export AC6_TOOLCHAIN_6_24_0=$T/compilers.arm.armclang/6.24.0/bin

cbuild setup cmsis-executorch.csolution.yml --active DevKit-E8 --packs   # after layer edits
cbuild cmsis-executorch.csolution.yml --active DevKit-E8

# flash + start (or: .venv/bin/python -c "import sys; sys.path.insert(0,'tests'); import fs_tools as fs; print(fs.flash_and_start()[-300:])")
printf 'connect\nh\nloadfile "out/cmsis-executorch/DevKit-E8/Debug/cmsis-executorch.hex"\nr\ng\nqc\n' > /tmp/load.jlink
JLinkExe -device AE822FA0E5597LS0_M55_HP -if SWD -speed 4000 -autoconnect 1 -NoGui 1 -CommandFile /tmp/load.jlink

# tests (stdlib + PyYAML in .venv); FS_HW=1 flashes and runs the board
FS_HW=1 .venv/bin/python -m unittest discover -s tests -v
```
- `tests/fs_tools.py`: toolchain env, `cbuild`, `flash_and_start()`, `reset_and_start()`,
  `UartCapture` (115200 8N1), `core_state()`, `read_word()`, `symbol_address()`,
  `capture_crop()` + `write_png()` (camera crop as PNG), `app_problem()`.
- Existing suites: `test_step2_camera_layer.py`, `test_step3_camera_init.py`,
  `test_step4_5_camera_live.py` (61 tests, all passing on the board before this work).
  Run them all after every change; they must stay green.
- Tools for people: `tests/grab_crop.py` (save what the model sees), `tests/letter_check.py`
  (24-letter score), `tests/check_cover_lens.py`.

### Tests to add (`tests/test_display.py`)
- Source/build: the three components resolve from `AlifSemiconductor::Ensemble`;
  `APP_DISPLAY` is defined only for DevKit-E8; FVP and AppKit-E8 still build without
  `fs_ui.o`/`display.o`; `lcd_frame` is 480x800x3 bytes, 32-byte aligned, in SRAM0 above
  `0x02010000`, not overlapping `s_ring` or the pools; nothing the NPU uses lies in SRAM1;
  `s_ring` starts at or above `0x02010000` (A32 stub).
- On the board (`FS_HW=1`): UART shows `Display: 480x800 RGB888 panel started`, no
  `DSI error`, `Test_result: PASS`, `camera init OK`, live lines at >= 15 fps with the `ui`
  time < 10 ms; with the debugger attached since the flash, read a few `lcd_frame` words in
  the crop area and check they change between two reads (live camera) and that the
  letter area is not all black; the board survives a warm reset mid-loop (camera, display
  and live lines come back).
- People check: a photo of the LCD with a hand sign, posted in the PR.

## 7. Definition of done
- The LCD shows the live camera crop, the letter, its confidence bar and fps, every frame.
- Section 2's three fixes are in, with tests; the full suite passes on the board.
- The loop still runs at >= 15 fps; UART output unchanged apart from the added `ui` time.
- PR against `fingerspell` on `erictweng/CMSIS-Executorch`, with a photo of the panel and
  the test results; `documentation/fingerspell-bringup.md` gets a short "Display" section.
