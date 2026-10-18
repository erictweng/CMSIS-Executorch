# Voice on the DevKit-E8: findings, decision and plan (status 2026-10-18 15:00 PDT)

Deliverable of `documentation/voice-plan.md` section 5. Context gathering started at
**14:48 PDT on Sunday 2026-10-18**; the submission closes at **17:00 PDT the same day**
(`docs/SPRINTS.md` in the fingerspell repo, "Hard deadline: Sun Oct 18, 5:00pm PDT"; the code
freeze there is 15:30). Nothing in this file is implemented. Every number below was read from
the files or URLs named next to it; anything that could not be verified is marked **assumption**.

## 1. Decision

**No-go for today's submission; go as the first post-hackathon feature.** The microphone is
there (two I2S MEMS mics on I2S3 of the HP core, pins already muxed in this project) and
trained DS-CNN weights exist (Arm ML-Zoo `ds_cnn_small`, Apache-2.0, TensorFlow/TFLite, a
1-2 h port into the PyTorch definition), but there are 2 hours to the deadline, the plan
below is about 9 working hours, and its first on-board check needs a UART that has been dead
since last night (`landmark-plan.md` status, 2026-10-17 23:20). A microphone added in that
window would be exactly the "badly integrated" input the judging text scores at zero, and
the cut list drops mic before anything else. Keep the camera build frozen; mention voice as
the next step in the submission README (section 6).

## 2. Findings: the repo as it is now

Sources: `git log` of `landmark/pipeline` (head `a2c334a5`), `documentation/landmark-plan.md`,
`hand-detection-handoff.md`, `temporal-average-plan.md`, `display-plan.md`,
`fingerspell-bringup.md`, `src/app_main.cpp`, `src/fs_api.h`, `src/fs_display.cpp`,
`src/fs_camera.c`, `board/DevKit-E8/Board-U85.clayer.yml`, the DevKit-E8 `RTE_Device.h` and
`app_mem_regions.h`, and `out/cmsis-executorch/DevKit-E8/Debug/cmsis-executorch.axf.map`
(built 2026-10-18 13:59 from `landmark/pipeline`).

### 2.1 Branches

- `fork/fingerspell` is at `c0d9cef5` (merge of PR #10, the voice brief). `landmark/pipeline`
  is two commits ahead of it apart from the brief (`23a3bd00` decoder/orientation,
  `a2c334a5` joystick), both pushed. The brief's "repo as it is now" is `landmark/pipeline`.
- The working tree on `landmark/pipeline` has an **uncommitted rewrite of `README.md`**
  (170 insertions: pitch, pipeline table, hardware, build/flash, "Changing the models"). The
  committed `README.md` on both branches is still the upstream hackathon guide, as the brief
  said. The rewrite has no workload-placement section and no voice mention (section 6).
- This plan lives on `voice/kws`, branched from `fork/fingerspell` at `c0d9cef5`.

### 2.2 The live loop (`src/app_main.cpp`, `live_loop()`)

One `for (;;)`, no RTOS, no SysTick use. Per iteration: wait for a camera frame
(`fs_camera_latest_frame`, a 100 us busy-poll loop on the CPI VSYNC counter), read the IMU,
resize to 192 and run `detect`, rotated 224 crop and run `landmark`, presence gate, MLP,
decoder, joystick, LCD render, one UART line, telemetry words. Timing comes from the 100 MHz
REFCLK counter (`cycles()`), not the DWT.

| Measured (board, Debug build) | Value | Source |
|---|---|---|
| Loop rate with a hand | 15-16 fps (period 63-67 ms) | `landmark-plan.md` status 23:20 |
| Loop rate without a hand | 18.9 fps | `landmark-plan.md` status (night) |
| `det` (resize 5.6 + NPU 17.3) | 22.9 ms | `fingerspell-bringup.md` Detector |
| `lm` (crop + NPU + decode + MLP) | 6.0 ms | `landmark-plan.md` status 23:20 |
| `ui` | 1.6-1.9 ms | README rewrite / bringup Display |
| `cam` (frame wait + crop) | ~2 ms | README rewrite line |
| IMU read | printed in `det split:` every 64 frames | `app_main.cpp` |

So ~33 ms of work per frame and the rest waiting for the camera (the MT9M114 streams at
~19-25 fps). An extra ~1-3 ms per frame of audio work and one tiny NPU call every 0.5 s do
not change the fps class; this must still be measured (phase 4).

Telemetry without the UART: `g_fs_tele[21]` and `g_fs_lm_dump[]` are `volatile uint32_t`
arrays read with J-Link `mem32` by `tests/board_telemetry.py` and `tests/capture_keypoints.py`
(`fs_tools.jlink`, symbols via `arm-none-eabi-nm`). A voice block would be a third array.

### 2.3 Memory (linker map of the current image)

| Region | Used | Capacity | Free | Where from |
|---|---|---|---|---|
| MRAM code+RO (`LR_ROM`, base 0x80200000) | 2,589,584 B | 3,604,480 B (0x370000) | **~991 KB** | map line 9; `app_mem_regions.h` `APP_MRAM_HP_SIZE` |
| of which the program (`.rodata.model`) | 2,265,152 B | | | map line 2170 |
| SRAM0 bulk (`RW_SRAM0`, 0x02010000) | 4,008,960 B | 4,128,768 B | **117 KB** | map line 16 |
| : camera ring (`.bss.ai_pool` fs_camera) | 618,496 B | | | map |
| : method pool 1 MB + temp pool 1.5 MB | 2,621,440 B | | | `Board-U85.clayer.yml` defines |
| : LCD frame buffer | 768,000 B | | | map |
| DTCM `.data/.bss` (`RW_RAM`, 0x20000000) | 568,840 B | 884,736 B (1 MB - 96 KB heap - 32 KB app heap - 32 KB stack) | **~308 KB** | map line 37; `linker_ac6_mram.sct.src` `__RW_SIZE` |
| ITCM | 0 | 256 KB | 256 KB | map line 11 |
| SRAM1 | 0 | 4 MB | 4 MB, but the NPU and CDC cannot read it (`SRAM0_SRAM1_COMBINED 0`) | `app_mem_regions.h` |

Correction to the brief: DTCM is not "nearly full"; ~308 KB of its `.bss` window is free
(the bulk buffers moved to SRAM0 earlier). The audio ring (48 KB) and the MFCC state (~6 KB)
fit in DTCM, which the I2S FIFO interrupt path reaches without DMA (a DMA path would need
the DTCM global alias `0x50800000`, `board/DevKit-E8/README.md` line 57; avoid it).

Vela (last export log, 2026-10-17 23:33, `Ethos_U85_SRAM_MRAM` / `Shared_Sram`,
`.cmsis/ensemble_vela.ini`): `detect` 290 M MACs, SRAM 885.7 KiB, 0 CPU operators;
`landmark` 146 M MACs, SRAM 1184.8 KiB, 0 CPU operators. The temp pool is 1.5 MB for the
largest method; a DS-CNN scratch will be far below that (expected < 50 KB; **assumption**
until exported). `create_ai_layer.py` prints per-method delegates and CPU operators and fails
with `AI_LAYER_STRICT=1` when a method is not a single delegate.

### 2.4 Peripherals already initialised (DevKit-E8 layer)

Camera (CPI + MIPI CSI-2 + MT9M114 over I2C1, streaming mode, 3 buffers), LCD (CDC200 +
MIPI DSI + ILI9806E), IMU (ICM-42670-P over I3C), joystick (LPGPIO P15_0/P15_2 through
`Driver_GPIO15`), UART4 console, Ethos-U85 with the NPU IRQ, DMA component present, SE
services. **No audio component is in the build** (`Board-U85.clayer.yml` components list).
`RTE_Device.h` has `RTE_I2S3 1` (IRQ mode, `RTE_I2S3_DMA_ENABLE 0`, RX trigger level 7,
WSS 32 cycles) and `RTE_PDM 1`; `RTE_LPI2S` and `RTE_LPPDM` are inside `#if defined(RTSS_HE)`
(lines 3217-3222, 10740-10780) and are **not** enabled for this HP build, contrary to the
brief's reading.

### 2.5 The AI-layer flow

`model/model.py` `get_methods()` returns `Method(name, module, example_inputs, calibration,
quantization, quantized_io)` for `detect` and `landmark`; `get_params()` returns constants
and float arrays that `create_ai_layer.py` writes to `ai_layer/model_params.h` (the MLP
weights go this way). A `kws` method is one more `Method(...)` in that list with
`quantized_io=True`, so the firmware gets `KWS_INPUT0_SCALE / _ZERO_POINT / _SIZE` and
`KWS_OUTPUT0_*` and writes int8 features directly (the lesson from the detector: the
portable quantize/dequantize operators at `-O1` cost ~70 cycles per element; for 490 values
that is negligible either way, but the convention is set). Labels and the MFCC constants go
into `get_params()`. Export takes ~2-3 min (`AI_LAYER_STRICT=1`).

### 2.6 Tests and tools

- `tests/fs_tools.py`: toolchain env (CMSIS-Toolbox 2.15.0 under `~/.vcpkg/artifacts/2139c4c6`),
  `cbuild()`, `flash_and_start()` (J-Link, one attempt, `BOOT_TIMEOUT` 90 s), `UartCapture`
  (115200 8N1, raises `UartBusy` when VS Code holds the port), `jlink()`, `read_word()`,
  `symbol_address()`, `core_state()`, `app_problem()`.
- `tests/board_telemetry.py`: `g_fs_tele` over J-Link with the core halted for a moment
  (`h`, `mem32`, `g`); the pattern for a voice readout.
- Board suites (`FS_HW=1`): `test_step2_camera_layer`, `test_step3_camera_init`,
  `test_step4_5_camera_live`, `test_display`, `test_detector`, each with a `Source`, a `Build`
  and an `OnBoard` class; `OnBoard.setUpClass` builds unless `FS_SKIP_BUILD=1`, flashes once,
  reads the UART until `live:` plus 6-8 s, does a warm reset and reads again. Host suites:
  `test_decoder_host` (352 decoder checks), `test_landmark`, `test_orient`, `test_input_host`.
- Duration: recorded runs in the session logs: `test_detector.OnBoard` alone 22-38 s with
  `FS_SKIP_BUILD=1`; the whole suite without hardware 61-70 s for 106 tests. A full
  `FS_HW=1` run has five flash-boot-reset cycles plus one build: **estimate 6-8 minutes**,
  not measured today (the board was not flashed during the freeze, and the UART is dead).

### 2.7 Board rules (unchanged plus the newer ones)

From `display-plan.md` section 5, `hand-detection-handoff.md` section 7,
`fingerspell-bringup.md`: never re-arm the CPI per frame or use its legacy single-buffer
mode; never touch CPI registers (`0x49030000`) from J-Link; never call `SERVICES_set_run_cfg`
with memory blocks; flash with J-Link, then test over the UART; read memory over J-Link only
in a session that has had the debugger attached since the flash; quit VS Code (it holds
`/dev/tty.usbmodem*`) before the UART tools; build with CMSIS-Toolbox 2.15.0 from a terminal;
one `cbuild` at a time (concurrent runs wipe `tmp/`); timings from REFCLK, not the DWT;
`main()` cleans the caches first; the A32 stub lives at `0x02000000` (bulk SRAM starts at
`0x02010000`). Newer: the IMU driver's dynamic-address wait has no timeout (`g_fs_imu_stage`
says where it stuck); the two IMU pack drivers (ICM42670P, BMI323) define the same global
callbacks and cannot both be linked; the board restarted once mid-run with no reset pressed
(open); a flash sometimes boots straight into the semihosting exit and a J-Link reset fixes
it; the I2S/PDM mics share nothing with these, but an audio IRQ adds a third interrupt source
next to the CPI and the NPU (keep its handler short, priorities to be set).

## 3. Findings: the microphone

Pack `~/.cache/arm/packs/AlifSemiconductor/Ensemble/2.2.1` (`P`), checked against the project's
copies of `board_defs.h`/`pins.h`/`RTE_Device.h`.

- **The DevKit-E8 has four MEMS microphones on board, no add-on needed.** The pack's board
  entry lists "I2S Digital MICs" (2) and "PDM Digital MICs" (2) (`P/AlifSemiconductor.Ensemble.pdsc`
  lines 1604-1605); the HE layer README names the I2S pair U14, U16 (`P/Boards/DevKit-e8/Layers/M55_HE/README.md`
  line 34); `board_defs.h` lines 174-176 set `BOARD_PDM_ENABLED 1`, `BOARD_PDM_PDM_MIC_PRESENT 1`.
  The Zephyr board page for `ensemble_e8_dk` lists "I2S/PDM digital microphones" too.
  Part numbers: the pack's generic I2S demo calls the I2S3 receiver an SPH0645LM4H-1 24-bit
  MEMS mic (`P/Boards/Templates/Baremetal/demo_i2s.c` line 20); **assumption** that U14/U16
  on the E8 DevKit are that part. The PDM pair's part number is not in the pack.
- **From the M55-HP, the "MIC Input" is I2S3** (`board_defs.h` lines 107-113:
  `BOARD_MIC_INPUT_I2S_INSTANCE 3` outside `RTSS_HE`). Its pins are already muxed in this
  project: P8_6 `I2S3_SCLK_B`, P8_7 `I2S3_WS_B`, P9_0 `I2S3_SDI_B`, "Selected ... for MIC Input"
  (`board/DevKit-E8/RTE/BSP/AE822FA0E5597LS0_M55_HP/pins.h` lines 537-556). The PDM mics are
  on PDM channels 4 and 5 (`board_defs.h` `BOARD_PDM_MIC_PDM_INPUT 2`, D2/C2; pins P5_4, P6_7
  also muxed). LPI2S/LPPDM are HE-only (no IRQ numbers in the HP `soc.h`).
- **The vStream AudioIn wrapper is thin and usable but defaults to the HE instance**
  (`P/Boards/DevKit-e8/Drivers/vstream_audio_in.c` lines 26-40: `AUDIO_CFG_SAI_INSTANCE LP`,
  16-bit, 16000 Hz). It configures the SAI as master, I2S protocol, 16-bit, frame 32, without
  `ARM_SAI_MONO_MODE`, so blocks are **stereo-interleaved** (32,000 int16 per second). It
  re-arms `Receive()` per block from the SAI callback; unlike the CPI case this is safe
  because `Driver_I2S.c` clears `rx_busy` before calling back (lines 1318-1324, 1389-1394).
  The pack's HP layers do not include it at all; it needs `-DAUDIO_CFG_SAI_INSTANCE=3`.
- **Decision for the plan: drive `Driver_SAI3` directly** in IRQ mode (`RTE_I2S3_DMA_ENABLE 0`
  as today) with `ARM_SAI_MONO_MODE` and 16-bit data (the top 16 bits of the mic's 24-bit
  MSB-aligned word), 16 kHz exactly (SCLK = 2 x 32 x 16 kHz = 1.024 MHz = 76.8 MHz / 75,
  `P/drivers/include/i2s.h` lines 577-583, `sys_ctrl_i2s.h` divider range 2..1023), one
  `Receive()` of 320 samples (20 ms) re-armed from the `RECEIVE_COMPLETE` callback into a
  ring in DTCM. The same reasons as for the camera: the driver is small, the wrapper adds a
  layer that hides errors, and mono halves the buffer. The PDM path (channels 4/5,
  `ARM_PDM_MODE_AUDIOFREQ_16K_DECM_*`, 16-bit only, needs the 76.8 MHz HFOSCx2 enabled through
  SE services and the FIR/IIR tables from `demo_pdm.c`) is the fallback if the I2S mics are
  silent. **Risk (assumption):** the SPH0645 is known for a one-bit-clock data timing quirk on
  some I2S receivers; phase 1's RMS check will show it as a DC offset or a sign flip.
- Components to add to `board/DevKit-E8/Board-U85.clayer.yml`: `CMSIS Driver:SAI` (pack
  component `AlifSemiconductor::CMSIS Driver:SAI`, requires the DMA component, already there).
  No RTE change needed for I2S3. No pin change.

## 4. Findings: the DS-CNN weights and front end

- **No PyTorch checkpoint of the 49x10 DS-CNN exists.** `executorch/examples/models/mlperf_tiny/ds_cnn.py`
  (also installed in `.venv`, ExecuTorch 1.5.1) is an architecture with random weights, no
  download; `alifsemi/sdk-alif` `samples/modules/executorch/models/kws/model.py` is the same
  (its README and `main.cpp` say "using untrained model"); Hugging Face has no `Arm/kws` or
  `Arm/ds_cnn` (the only PyTorch KWS checkpoint found, `priyadeepjaiswal9c/tiny-kws`, is a
  different 119 k-parameter 64x101 log-mel model). Checked 2026-10-18 by the research agent;
  URLs in `documentation/voice-plan.md` section 2 plus
  https://github.com/ARM-software/ML-zoo/tree/master/models/keyword_spotting/ds_cnn_small/model_package_tf
  and https://github.com/mlcommons/tiny/tree/master/benchmark/training/keyword_spotting.
- **Trained weights, Apache-2.0, TensorFlow:** Arm ML-Zoo `ds_cnn_small/model_package_tf`
  (repo archived July 2025, still fetchable): `model_archive/TFLite/tflite_fp32/ds_cnn_s.tflite`
  (input `(1,490)` float, 93.89 %), `tflite_int8/ds_cnn_s_quantized.tflite` (47,616 B, 93.11 %),
  a TF2 checkpoint and SavedModel, and golden `testing_input/input/0.npy` /
  `testing_output/Identity/0.npy`. Labels: silence, unknown, yes, no, up, down, left, right,
  on, off, stop, go. MLCommons tiny ships `kws_ref_model_float32.tflite` (~92 %) with a
  **different** front end (30 ms window, `tf.signal` mel, alphabetical labels); not chosen.
- **Front end of the ML-Zoo weights** (from `recreate_model.sh`, `data_preprocessing.py`,
  `models.py` and the TF `audio_ops` kernels): 16 kHz mono float in [-1, 1], 1000 ms clip
  (zero-padded), periodic Hann window of **640 samples (40 ms)**, stride **320 (20 ms)**,
  FFT length 1024 (513 bins), power spectrum, **40 mel filters, 20-4000 Hz**, each energy
  floored at 1e-12, natural log, DCT-II with `sqrt(2/40)` scaling, coefficients 0..9 (c0
  kept), **49 frames** -> `[49, 10]` time-major, flattened to 490, no mean/variance
  normalisation. **Assumption:** the mel scale is HTK (`1127 ln(1 + f/700)`); Arm's
  evaluation-kit `DsCnnMfcc` (40 bins, 20-4000 Hz, `UseHtkMethod = true`) was written to match
  these weights, and its `Mfcc.hpp` is the reference C++ implementation. Phase 2 checks the
  C front end against the TFLite golden input, which settles it.
- **The port** (phase 0): read the fp32 TFLite tensors with the TFLite interpreter in the
  tools venv (TensorFlow is there, not in `.venv`), permute into a PyTorch module that matches
  TF `SAME` padding exactly (`ZeroPad2d((1, 1, 4, 5))` + `Conv2d(1, 64, (10, 4), stride 2)`,
  BN folded, biases present; four depthwise-separable blocks; `AdaptiveAvgPool2d(1)`
  instead of `AvgPool2d((24, 5))`, as ExecuTorch's own `backends/arm/test/models/test_ds_cnn.py`
  does for the Ethos-U stride constraint; `Linear(64, 12)`), save as `model/ds_cnn_kws.pt2`,
  check the softmax against the golden output and the TFLite interpreter. **The installed
  `ds_cnn.py` uses symmetric `padding=(5, 1)` and `bias=False`, which is off by one row from
  TF `SAME`; write our own module in `ml/kws.py` rather than loading into it.** Calibration:
  the golden vector plus MFCCs of ~50 Speech Commands clips (download a few hundred WAVs,
  not the 2.3 GB set) through the numpy reference; input and output int8 scale/zero point come
  out of `create_ai_layer.py` as `KWS_INPUT0_*`.
- **Training from scratch** is not in the plan: both reference recipes are TensorFlow, need
  the 2.4 GB Speech Commands v2 download and 30 k steps / 36 epochs with no published
  wall-clock figure (**assumption** 1-3 h on an M-series Mac with cached features).

## 5. Open questions from the brief, answered

1. **Usable microphone without extra hardware?** Yes: I2S MEMS mics on I2S3 (section 3).
2. **Trained weights?** Yes, as TensorFlow/TFLite (ML-Zoo `ds_cnn_small`); a weight port,
   not training. The MFCC parameters are listed in section 4 and must be matched exactly;
   the golden input/output pair is the check.
3. **MFCC on the M55:** compute **one 10-coefficient frame per 20 ms stride as the audio
   arrives**, never a 49-frame burst: one frame is one 1024-point real FFT (plain C99
   radix-2 on floats, ~50 k flops) plus 40 mel sums, 40 logs and a 40x10 DCT, so a few
   hundred microseconds per frame at 400 MHz (**estimate**, to be measured in phase 2);
   three frames arrive per camera frame, so under 1 ms per loop iteration. CMSIS-DSP
   (`arm_mfcc_f32`, Helium) is **not installed** (`~/.cache/arm/packs/ARM` has no CMSIS-DSP;
   the index lists CMSIS-DSP 1.18.0, so `cpackget add ARM::CMSIS-DSP@1.18.0` would fetch it)
   and is the optimisation if the plain version costs more than ~1 ms per frame. The host
   reference is numpy in `ml/kws.py`, tested like `ml/handlandmark.py`.
4. **Threading:** the I2S FIFO interrupt (every 7-8 samples) copies into a 1.5 s ring in
   DTCM (`volatile` write index); the loop calls `fs_voice_poll()` once per camera frame
   right after `fs_camera_latest_frame()`: it consumes complete 320-sample strides from the
   ring, appends MFCC frames to a rolling 49x10 feature window, and every 25 new frames
   (0.5 s) runs the `kws` method and the decision rule. No second thread, no SysTick, the
   NPU is only ever called from the loop, so it cannot collide with `detect`/`landmark`.
   The ring holds 1.5 s, so a loop iteration can be up to ~1 s late without losing audio.
5. **NPU sharing:** a third method in the same `.pte`, run sequentially from the loop; the
   temp pool (1.5 MB) is sized for `landmark` and the DS-CNN scratch is far smaller
   (**assumption**: tens of KB; the export prints it). DS-CNN is conv, depthwise conv, ReLU,
   average pool, FC, all supported by the Arm backend; the export must show
   "1 Ethos-U delegate(s); CPU operators: none" under `AI_LAYER_STRICT=1`.
6. **Memory:** audio ring 48 KB (int16, 1.5 s mono) + feature window 2 KB + FFT work 8 KB
   in DTCM (308 KB free); the program grows by ~40-50 KB in MRAM (991 KB free); SRAM0 is not
   touched (117 KB free, nothing needed there since no DMA). No pool size changes expected.
7. **UX (to validate with the team):** a recognised word shows as the hearing person's
   reply on the LCD: the bottom row of the words card (`kWordsY` 616, 4 rows x 14 chars,
   `fs_display.cpp` lines 90-91) becomes `> YES` in a second colour, and the signer keeps
   three rows; the UART prints `voice: yes 0.91 npu 1.2ms`. Score threshold **0.7**, refractory
   **1 s** after a word, "silence"/"unknown" never shown. Optional decoder hook: "stop" pauses
   letter commits, "go" resumes (one flag in `fs_decoder_t`, host-tested); off unless the
   team wants it.
8. **If it is late:** everything behind `APP_VOICE` (a define in the DevKit-E8 layer, like
   `FINGERSPELL_LCD`), `fs_voice.c` built `for-context: +DevKit-E8`, the `kws` method added to
   `model.py` behind `FS_VOICE=1` in the environment so the shipped `ai_layer/` is regenerated
   only when the feature is on; the camera build stays shippable at every commit.

## 6. The phased plan (post-hackathon; ~9 working hours)

Each phase ends in something a person can see. Times are estimates for one person with the
board and the UART working; the cut lines assume a start at T0.

| Phase | What | Check a person can see | Estimate | Cut line |
|---|---|---|---|---|
| 0 Export spike | `ml/kws_export.py`: TFLite -> `model/ds_cnn_kws.pt2`, golden check; `kws` method in `model/model.py`; `AI_LAYER_STRICT=1` export | Export log: `kws: 1 Ethos-U delegate(s); CPU operators: none`, Vela SRAM printed, program size; softmax matches the golden output to 1e-3 | 1.5 h | not a single delegate or scratch > 1 MB at T0+2h: stop, write it up |
| 1 Mic capture | `CMSIS Driver:SAI` in the layer, `src/fs_voice.c` (I2S3 mono 16 kHz, 1.5 s ring, RMS), `g_fs_voice[]` telemetry | UART `voice: rms 112` every 0.5 s rising when you talk; `tests/test_voice.py::Mic` reads 5 s of levels (UART, or J-Link words while the UART is dead) and asserts quiet vs talking differ by 3x | 2 h | no level change by T0+4.5h: try PDM for 45 min, then stop |
| 2 Features | MFCC in `fs_voice.c` (C99, host-tested) vs `ml/kws.py` (numpy) on the ML-Zoo golden WAV and a recorded clip | `voice: mfcc 0.3ms/frame`; host test max abs diff < 0.05 per coefficient against numpy, and the golden 490 vector reproduced from its WAV | 2 h | not matching by T0+7h: stop, leave `APP_VOICE` out of the demo build (a bare level meter is not worth shipping) |
| 3 Model on board | run `kws` every 0.5 s, decision rule, UART line | `voice: yes 0.91 npu 1.2ms` for a person saying the four words, and for a known WAV played from the laptop; false triggers in 60 s of room noise counted | 1.5 h | |
| 4 Integration | reply row on the LCD, optional decoder hook, `APP_VOICE`, `FS_HW=1` suites green, fps before/after | LCD photo with `> YES` under the signed text; suite log; fps numbers with and without voice | 2 h | |
| 5 Docs | "Voice" section in `fingerspell-bringup.md` with ms per inference, fps before/after, memory; README workload placement (mic -> M55 MFCC -> U85 DS-CNN -> M55 decode/LCD) | PR with the numbers and the photo | 1 h | |

Files to touch: `ml/kws.py`, `ml/kws_export.py`, `model/ds_cnn_kws.pt2`, `model/model.py`,
`model/calib_kws.npy`, `src/fs_voice.c/.h`, `src/app_main.cpp` (poll call, UART line,
telemetry), `src/fs_display.cpp` (reply row), `src/fs_api.h` (decoder pause flag, if wanted),
`src/fs_decoder.c` (same), `board/DevKit-E8/Board-U85.clayer.yml` (SAI component, `APP_VOICE`),
`cmsis-executorch.cproject.yml` (`fs_voice.c` for DevKit-E8, `optimize: speed`),
`tests/test_voice.py`, `tests/host/test_fs_voice.c`, `tests/fixtures/kws_*`,
`documentation/fingerspell-bringup.md`, `README.md`. Branch: `voice/kws` off `fingerspell`.

## 7. For the submission README (today)

The uncommitted README rewrite on `landmark/pipeline` has no workload-placement section and
no voice mention. Whoever finalises it: the workload table is camera -> M55 resize -> U85
BlazePalm -> M55 crop -> U85 hand_landmark_lite -> M55 MLP/decoder/LCD; add one "What's next"
line: "a microphone reply path (I2S MEMS mic -> MFCC on the M55 -> DS-CNN keyword spotting on
the Ethos-U85 -> reply on the LCD) is planned; see `documentation/voice-plan-status.md`".
Do not claim it as a feature.

## 8. What I need from a person before any code

1. The go-ahead for the post-hackathon work on `voice/kws`, and when (after the freeze).
2. Confirmation that the I2S mics are the ones to use (or a preference for the PDM pair).
3. Whether "stop"/"go" should control the decoder, or words are display-only.
4. The UART: a working serial port (or an agreed J-Link-only test protocol) before phase 1.

## 9. Status log

- 2026-10-18 15:30: go-ahead (PR #11 merged). Durable worktree `/Users/ericweng/CMSIS-Executorch-voice`
  on `voice/kws` (rebased onto `fingerspell` at `3f123170`).
- 2026-10-18 16:00: **phases 0-2 done on the host, phases 1 and 3 built, nothing flashed.**
  Weights: Arm ML-Zoo `ds_cnn_small` fp32 TFLite (LFS files, fetched through
  `github.com/.../raw/...`), extracted with `ml/kws_tflite_to_npz.py` (tools venv) and ported by
  `ml/kws_export.py`: matches the TFLite interpreter to 3.5e-6 (36 clips, 36/36 top-1).
  Front end: `ml/kws.py` matches TensorFlow's `audio_spectrogram` + `mfcc` to 2.5e-5 (the
  filterbank takes the magnitude, not the power, and the mel scale is HTK: both confirmed
  numerically); `src/fs_voice.c` matches TF to 1.1e-4 and the stream quantizes bit-exactly
  against the reference for chunk sizes 320, 333, 1600 and 16000. Gate 0 (strict export):
  `kws` is one Ethos-U delegate, no CPU operators, Vela SRAM 23.6 KiB, weights 29.6 KiB, input
  scale 1.094 / zero point 98, output scale 0.234 / zero point 46; program 2,299,024 bytes.
  Mic capture written against `Driver_SAI3` (IRQ, mono, 16-bit, HFOSCx2 enabled through the
  SE services), ring 48 KB in DTCM. Integration: `voice_poll()` once per loop iteration after
  the stage timers, inference every 25 new frames, `voice: rms ..` every 0.5 s,
  `voice: <word> <score> npu <ms>` on acceptance, reply row on the LCD, `g_fs_voice_tele[11]`,
  boot `Voice_test: PASS` on a TF-computed "yes" clip. DevKit image builds: code 194,016 B,
  RO 2,429,132 B, ZI 4,804,964 B (MRAM free 973 KB, DTCM free 244 KB, SRAM0 118 KB).
  Host accuracy of the port (1200 Speech Commands test clips): 89.1 % float, 88.8 % with the
  int8 input; silence 48 %, so a noisy room may trigger words (5 % of noise clips pass 0.7).
  Host tests: `tests/test_voice.py` 10 host tests green. **Pending the board** (another agent
  holds it): phase 1 level check, mfcc ms per frame, NPU ms per inference, fps before/after,
  the four words spoken, false triggers in 60 s of room noise, `FS_HW=1` suites.
