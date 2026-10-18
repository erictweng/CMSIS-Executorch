# Speech-to-text on the DevKit-E8 (Tier B): context-gathering and planning brief

A brief for the agent adding free-speech transcription to Fingerspell: the hearing person
talks, the board turns it into text on the LCD next to the signer's spelled words. **This
brief does not authorise implementation.** Its deliverable is a plan
(`documentation/stt-plan-status.md`, section 5) that a person approves before any code is
written. It is the successor of `documentation/voice-plan.md` (Tier A, keyword spotting) and
reuses everything that brief and its status file established about the microphone, the
loop, the memory map and the board rules. Read those first; do not redo them.

Why a separate brief: Tier A recognises twelve words. Tier B is a 10 M-parameter Conformer
whose weights do not fit on-chip memory, whose scratch does not fit one SRAM bank, and
whose reference port is a Zephyr application on the other core. It is a multi-day job with
three go/no-go gates before any audio is involved. Written 2026-10-18 after the hackathon
submission; nothing here is for the demo build.

## 1. The goal

Free English speech, a few seconds at a time, transcribed on the board and shown as the
reader's reply line under the signed text. No phone, no cloud. Accept: a fixed-length
window (several seconds), a transcription that appears after the window ends rather than
word by word, and a word error rate of several percent on clean speech. Keep the camera
pipeline (`landmark/pipeline`) shippable at every commit, as with Tier A.

Two shapes are acceptable; the plan picks one with the numbers from section 3:

- **Time-shared**: one core (M55-HP), one program with the vision methods and the STT
  method; a "listen" gesture (joystick press, or "stop" from Tier A if it exists) pauses the
  camera loop, records N seconds, runs STT, shows the text, resumes.
- **Dual-core**: STT on the M55-HE as Alif's sample does, the camera pipeline on the HP
  unchanged, text passed over the MHU. Only possible if the NPU question in section 4 is
  answered favourably.

## 2. What exists (verified 2026-10-18 15:00-15:20 PDT; URLs are the sources)

**The model: `Arm/stt_en_conformer_executorch_small`** (Hugging Face,
https://huggingface.co/Arm/stt_en_conformer_executorch_small, last modified 2025-10-23).

- Card: Conformer, **10 M parameters** (the voice brief said 8.7 M: wrong), trained on
  LibriSpeech 960 h, **7 % WER on test-clean** for the model as shipped; the float
  checkpoint file is named `conformer_fp32_cln_wer_6_47.pt` (6.47 % is the float number).
- Files: `Conformer_ArmQuantizer_quant_exported_program.pt2` (the quantized exported
  program for the Ethos-U backend), `conformer_fp32_cln_wer_6_47.pt`, `tokenizer.model`
  (SentencePiece), `README.md`, `LICENSE`, `BigScience_OpenRAIL-M.pdf`; repository total
  92.9 MB. **Licence: BigScience OpenRAIL-M v1.1**, a use-restricted licence (not Apache).
  Read its Attachment A before planning a public demo or redistribution; the plan must
  say whether our use is allowed and what the README must state.
- Front end on the card: 16 kHz, 25 ms / 400-sample window, 10 ms / 160-sample hop, FFT 512,
  **80 mel bands**, MelSpectrogram; normalisation not stated. The Arm evaluation kit's
  Conformer use case feeds a **700 x 80** input (7 s at 10 ms frames), right-pads short
  clips with **-20.0** (so the input is log-mel, floor -20), takes the argmax per frame,
  collapses repeats and looks the pieces up in `librispeech_sp.pieces` (greedy CTC; the
  tokenizer script is `build_sp_128_librispeech.py`, so 128 pieces + blank = 129 outputs:
  **assumption**, confirm from the `.pt2` output shape). Source:
  https://gitlab.arm.com/artificial-intelligence/ethos-u/ml-embedded-evaluation-kit/-/raw/main/docs/use_cases/asr.md
  (it also reports **206,206,056 NPU cycles per inference** on the FVP and a 7 s / 15 s
  inconsistency between its pre- and post-processing sections: resolve it from the code).
- Memory on the card (Ethos-U85-256, `Shared_Sram`, system config
  `Ethos_U85_SYS_Flash_High`, Vela extras `--output-format=raw --debug-force-regor`):
  **5.7 MB peak intermediate-tensor SRAM, 10.8 MB read-only weights in external memory.**
- Toolchain on the card: **ExecuTorch 1.0**; the `.pt2` was produced with it. This repo
  pins ExecuTorch 1.5.1, torch 2.14.0, Vela 5.1.0 (`requirements.txt`). Whether a 1.0
  exported program loads and re-lowers under 1.5.1 is gate 0.
- Training and PTQ recipe: https://github.com/Arm-Examples/ML-examples/tree/main/pytorch-conformer-train-quantize
  (`training/`, `post_training_quantization/ptq_evaluate_conformer_10M.py`; the PTQ README
  says ExecuTorch `release/1.0`, Python 3.10, the `sooftware/conformer` package, LibriSpeech
  via torchaudio, ~8 % WER after quantization). This is the only place the mel front end,
  the padding and the calibration are written as code; read the script.

**The reference port: `alifsemi/sdk-alif` `samples/modules/executorch/alif_asr`**
(https://github.com/alifsemi/sdk-alif/tree/main/samples/modules/executorch/alif_asr).

- Zephyr, built for `alif_e8_dk/ae822fa0e5597xx0/rtss_he` (**the HE core**) with
  `-S ethos-u85-enable -DETHOSU_TARGET_NPU_CONFIG=ethos-u85-256 -DML_FWK_TMP_MEM_SIZE=0x002C0000`
  (2.75 MB temp pool). A port of the Alif ML Embedded Evaluation Kit; the model comes from
  the `alif-mlek` module's `set_up_default_resources.py --ml-frameworks executorch` as
  `resources_downloaded/asr/conformer_fp32_cln_wer_6_47_arm_delegate_ethos-u85-256.pte`;
  the labels from `resources/asr/labels/librispeech_sp.pieces`. The `alif-mlek` repository
  URL is not in `west.yml` (it is pulled through the Zephyr manifest's optional groups):
  find it, and find where the script downloads the `.pte` from.
- `Kconfig`: `MODEL_IN_EXT_FLASH` ("Initialize OSPI controller to XIP mode. Initialize the
  onboard flash device", links the model to external flash, generates `ospi0/1.bin`),
  `ACTIVATION_BUF_SZ` default `0x00108000`, `AUDIO_SAMPLES 16000`, `AUDIO_STRIDE 8000`,
  `AUDIO_CHANNELS 2`, `SAMPLE_CNT 400`, `NUM_BUFFERS 4`, `I2S_GAIN 20`, `ETHOS_U_AXI0_ONLY`,
  `ENABLE_DISPLAY` (LVGL). `prj.conf`: CMSIS-DSP (transform, fast math, statistics,
  complex math), audio input over I2S, the model in OSPI flash with XIP on an IS25WX device,
  "ASR model is large, so we need to store it in external flash".
- `sram1_temp_buf.ld`: the ExecuTorch temp pool (`sTmpAllocationPool`,
  `.bss.NoInit.temp_buf_sram`) is placed in **SRAM1** because the HE DTCM (256 KB) would
  overflow. **This contradicts our board layer's rule that the Ethos-U85 cannot reach SRAM1**
  (`board/DevKit-E8/README.md` line 59, from `hackathon-pico-faces`). One of them is wrong
  for our configuration; section 3.3 resolves it on the board.
- Sources: `src/use_case/alif_asr/src/MainLoop.cc` (4 KB), `UseCaseHandlerEt.cc` (18 KB),
  `../../alif_common/audio/AudioBackend.cpp`; the mel front end and the decoder are in the
  `mlek::asr_impl` library of `alif-mlek` (the evaluation kit's `source/lib/mlek`).
- Not in the README: how the OSPI flash is programmed, the measured inference time, the
  microphone used. The voice brief's "J-Flash, XIP at 0xC0000000" came from an older
  reading; re-verify.

**On this board and in this repo.**

- The DevKit-E8 has a 32 MB ISSI **IS25WX256** octal flash on OSPI1 at XIP address
  `0xC0000000` (pack `Boards/DevKit-e8/board_defs.h` `BOARD_ISSI_FLASH_OSPI_INSTANCE 1`,
  `BOARD_OSPI1_MEM_SIZE 33554432`; pdsc flash algorithm `Flash/algorithms/Ensemble_IS25WX256.FLM`
  start `0xC0000000` size `0x02000000`). The pack has a bare-metal XIP path: component
  `BSP:OSPI FLASH XIP:core` (`ospi_xip/source/issi_flash/issi_flash.c`, `ospi_drv.c`,
  `setup_flash_xip()` from `ospi_xip/include/setup_flash_xip.h`, config
  `ospi_xip/config/ospi_xip_user.h`: instance 1, image address `0xC0000000`, 100 MHz,
  wait cycles 16), a `Driver_Flash` for writing the flash from the application
  (`Boards/Templates/Baremetal/demo_flash.c`), and `ospi_xip/app/app_issi_flash.c` (sets
  up XIP and jumps). Our `app_mem_regions.h` already defines `APP_OSPI_FLASH_HP_BASE
  0xC0200000` / size `0x00200000` with `APP_BOOT_OSPI_FLASH 0`; these are the pack
  defaults, not sized for 10.8 MB. SEGGER J-Flash is installed
  (`/Applications/SEGGER/JLink*/JFlash.app`, `JFlashSPI`); whether it knows this flash
  behind the Alif OSPI controller is unverified.
- `.cmsis/ensemble_vela.ini` has **`RTSS_HP_SRAM_OSPI`** (axi1 OffChipFlash, clock scale
  0.052, read latency 108) next to `RTSS_HP_SRAM_MRAM`, i.e. Alif's own system config for
  weights streamed from OSPI on the HP core, and `RTSS_HE_SRAM_MRAM`. The card's
  `Ethos_U85_SYS_Flash_High` is Vela's generic name; use the Alif one.
- **NPUs on the E8** (pdsc device features, lines 280-282): one **Ethos-U85-256** shared
  (`NPU_HG_BASE 0x49042000`, IRQ 366 on the HP; the same base in the HE `soc.h`), one
  Ethos-U55-256 on the HP, one Ethos-U55-128 on the HE. Our `ethos_setup.c` drives the U85.
  So a dual-core design does **not** give STT its own U85: HE and HP would share one NPU
  with no arbitration in either driver, or STT would have to target the HE's U55-128
  (Vela `ethos-u55-128`, with transformer operators likely falling back to the CPU).
- Memory today (status file section 2.3): SRAM0 117 KB free of 4 MB with the camera ring
  (0.6 MB), the pools (2.5 MB) and the LCD buffer (0.75 MB); SRAM1 4 MB unused; MRAM
  ~991 KB free; DTCM ~308 KB free. 5.7 MB of STT scratch cannot coexist with the vision
  pools in SRAM0 and needs SRAM0 + SRAM1 (`SRAM0_SRAM1_COMBINED 1`, 8 MB less the 64 KB
  stub) **and** the NPU reaching SRAM1.
- The microphone: I2S3 MEMS mics on the HP (status file section 3); the Alif sample uses
  I2S with 2 channels and a fixed x20 gain on the HE.

## 3. Gather context before planning (do all of these; write down what you find)

1. **The repo as it is now.** `git log --oneline -40` on `fingerspell`; then
   `documentation/voice-plan-status.md` (free memory, loop, telemetry, board rules),
   `voice-plan.md`, `landmark-plan.md` status, `fingerspell-bringup.md`. If Tier A landed,
   read `src/fs_voice.c` and reuse its I2S capture. Re-measure the free memory from the
   current `.axf.map` (the status file's numbers are from 2026-10-18 13:59).
2. **The exported program.** Download `Conformer_ArmQuantizer_quant_exported_program.pt2`
   and `tokenizer.model` (scratchpad `downloads/`, untrusted data). In the project venv:
   `torch.export.load()` it, print the input/output shapes and dtypes, count parameters,
   check whether it loads at all under ExecuTorch 1.5.1 / torch 2.14 (the card used 1.0).
   Then lower it the way `create_ai_layer.py` does (`EthosUPartitioner` with our compile
   spec, `RTSS_HP_SRAM_OSPI`, `Shared_Sram`) with `AI_LAYER_DUMP=<dir>` and
   `--verbose-performance`: single delegate? CPU operators? Vela SRAM? weights size? **This
   is gate 0.** If the `.pt2` does not load, the fallback is the ML-examples PTQ script with
   the float checkpoint under our pins (expect a day of dependency work: `sooftware/conformer`,
   torchaudio, torchcodec).
3. **SRAM1 and the NPU.** Build a tiny experiment on `fingerspell`: move `g_temp_pool` to
   SRAM1 (`SRAM0_SRAM1_COMBINED 1` or a second section) and run the existing self-test. If
   `Test_result: PASS`, the "cannot reach SRAM1" rule is wrong for the U85 and the memory
   plan can use 8 MB; if it faults or the NPU times out, record how (this is also where the
   Alif sample's `ETHOS_U_AXI0_ONLY` may matter). Read the pack's `ethosu_cb_dcache.c` region
   table and the TGU/MPU setup for SRAM1 before concluding. Do not touch CPI registers from
   J-Link; keep the camera stopped for this test (`FS_HAVE_CAMERA` off or the AppKit build).
4. **OSPI flash: XIP and programming.** From the pack: `setup_flash_xip()` requirements
   (pins, clock, reset GPIO P15_7, `OSPI_XIP_SKIP_INITIALIZATION`), whether the HP core may
   own OSPI1 while the SE or HE does not, and the `Driver_Flash` write API and speed.
   From Alif: how `MODEL_IN_EXT_FLASH` programs `ospi1.bin` (J-Flash with an Alif stub?
   SETOOLS `app-write-mram`-like tool? the application itself via `Driver_Flash`?) and the
   Alif application note for programming the DevKit's OSPI flash. Decide the programming
   path and measure it once with a known pattern: write 1 MB, read it back through XIP at
   `0xC0000000`, compare. **Gate 1.**
5. **Vela with weights in OSPI.** Compile the delegate with `RTSS_HP_SRAM_OSPI` and the
   weights placed in the flash region (the `.pte` keeps the command stream and weights
   together; find how the Alif sample splits the model into `ospi1.bin` and what the
   runtime loads from where: `extract_delegate_segments=False` on the card means the
   delegate blob is inside the `.pte`; our `model_pte.c` puts the whole `.pte` in MRAM, which
   is 991 KB short of 10.8 MB). Options to evaluate: the whole `.pte` in OSPI (XIP, the
   runtime reads it in place), or a split loader. Record Vela's estimate and the real
   inference time on the board for one 7 s window. **Gate 2**: a transcription of a baked-in
   LibriSpeech clip printed on the UART, with the time.
6. **The front end.** Read the evaluation kit's mel implementation (`source/lib/mlek`) and
   the PTQ script's torchaudio parameters; write the numpy reference (`ml/stt.py`) and
   check it against the `.pt2` on the host with a WAV (WER on 20 test-clean clips, host
   only). Decide whether the M55 computes 80-mel log spectrograms for 7 s (700 frames of a
   512-point FFT, ~10x Tier A's work) fast enough; CMSIS-DSP is not installed
   (`cpackget add ARM::CMSIS-DSP@1.18.0`).
7. **Board state and rules.** All of the voice status file's section 2.7, plus: the HE
   core is not initialised by our image (the SE boots only the HP from the ATOC); a
   dual-core design needs a second ATOC entry and a second build.
8. **The schedule.** There is no deadline now. Note the time you start and the hours spent
   per gate; the plan is judged on whether the gates were honestly measured.

## 4. Open questions the plan must answer

1. Does the `.pt2` load and lower under our ExecuTorch 1.5.1 pins, and what does Vela say
   (delegates, CPU operators, SRAM, weight bytes)? If not, is re-exporting with the
   ML-examples script under our pins feasible, and in how long?
2. Can the U85 use SRAM1 from the HP (section 3.3)? If not, STT scratch (5.7 MB) does not
   fit and Tier B is **no-go on this board**; say so and stop.
3. How are 10.8 MB of weights put in the OSPI flash and read by the NPU at speed: XIP
   throughput measured, programming tool and time, and what happens to the boot (the
   flash is not erased by a J-Link MRAM flash, so once programmed it stays).
4. Real inference time for 7 s of audio on the HP U85 with OSPI weights. The vision models
   run ~5x Vela's estimate on this board; 206 M NPU cycles plus 10.8 MB of weight traffic
   over OSPI (100 MHz octal) suggests seconds, not hundreds of milliseconds. The UX depends
   on it: a 7 s window plus a 3 s wait is still a usable reply; 20 s is not.
5. Time-shared or dual-core? With one U85 and no arbitration, time-shared on the HP is the
   default; dual-core only if STT fits the HE's U55-128 with acceptable CPU fallback (unlikely
   for a Conformer) or if a clean NPU hand-over protocol over the MHU is cheap.
6. Memory plan with the camera: if time-shared, can the vision pools, the camera ring and
   the LCD buffer stay allocated while STT runs (needs ~5.7 + 3.9 MB), or must STT mode
   release the camera ring and the vision temp pool (then re-init the camera afterwards:
   `fs_camera_init` takes ~1 s and the CPI rules apply)?
7. Front-end cost on the M55 and where the 7 s of audio live (224 KB int16 mono; DTCM
   has ~308 KB; the mel buffer 700 x 80 floats = 224 KB more, or int8 if the input is
   quantized at the boundary).
8. Licence: is OpenRAIL-M v1.1 compatible with a public demo and this repository's
   Apache-2.0 files? What must the README say?
9. UX: how listening starts and ends (joystick UP = record 7 s? voice activity detection
   from Tier A's RMS? the Silero VAD example in `executorch/examples/arm`?), how the text
   is shown (the reply row, scrolling), and what the signer sees while the camera is paused.
10. What it costs the demo: everything behind `APP_STT`, built only for DevKit-E8, off by
    default; the OSPI flash contents are a separate artefact with its own checksum printed
    at boot.

## 5. Deliverable: the plan, then stop

Write `documentation/stt-plan-status.md` with:

- Findings from section 3 with sources; every assumption marked; the three gate results
  (`.pt2` loads and lowers; SRAM1 reachable; flash programmed and read back) as measured.
- Decision: go / no-go for Tier B on this board, one line, with the reason.
- If go, the phases, each ending in a check a person can see, with time estimates and a
  stop rule:
  - Phase 1, weights in flash: the OSPI flash programmed with the delegate's weights, a
    checksum printed at boot that matches the host's.
  - Phase 2, model on board: a baked-in 7 s LibriSpeech clip transcribed on the UART with
    the inference time, Vela SRAM and the measured XIP throughput written down; WER on 10
    clips through the board vs the host reference.
  - Phase 3, audio: 7 s from the I2S mic through the M55 front end; the same clip played
    from a laptop speaker transcribed on the board; ms per window for the front end.
  - Phase 4, integration: listen gesture, camera pause/resume, the reply row, `APP_STT`,
    all suites green with `FS_HW=1`, fps unaffected outside STT mode.
  - Phase 5, docs: "Speech-to-text" section in `fingerspell-bringup.md` with the numbers,
    the OSPI programming procedure, the licence note; README workload placement
    (mic -> M55 log-mel -> U85 Conformer with weights in OSPI -> M55 CTC decode -> LCD).
- Files to touch and the branch name (`voice/stt` off `fingerspell`).
- What you need from a person: the go-ahead, the time-shared vs dual-core choice, the
  licence decision, and whether the OSPI flash may be reprogrammed (it is shared with
  nothing today, but it is a one-way change until someone erases it).

Post the status file in a PR against `fingerspell` on `erictweng/CMSIS-Executorch` (docs
only) and wait for approval in that PR before writing code.

## 6. Build, flash, test (unchanged)

```
T=~/.vcpkg/artifacts/2139c4c6
export PATH=$T/tools.open.cmsis.pack.cmsis.toolbox/2.15.0/bin:$T/tools.kitware.cmake/4.2.1/bin:$T/tools.ninja.build.ninja/1.13.2:$PATH
export AC6_TOOLCHAIN_6_24_0=$T/compilers.arm.armclang/6.24.0/bin
cbuild setup cmsis-executorch.csolution.yml --active DevKit-E8 --packs   # after layer edits
cbuild cmsis-executorch.csolution.yml --active DevKit-E8
.venv/bin/python -c "import sys; sys.path.insert(0,'tests'); import fs_tools as fs; print(fs.flash_and_start()[-300:])"
FS_HW=1 .venv/bin/python -m unittest discover -s tests -v
```

Rules that matter more here than before: never touch CPI registers from J-Link; the OSPI
flash is programmed with a tool that must not disturb the MRAM boot table (if a step needs
SETOOLS, SW4 to SEUART and back); read memory over J-Link only in a session attached since
the flash; one `cbuild` at a time.
