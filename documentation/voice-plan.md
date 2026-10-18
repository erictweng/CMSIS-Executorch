# Voice on the DevKit-E8: context-gathering and planning brief

A brief for the agent adding a microphone path to Fingerspell. **This brief does not
authorise implementation.** Its deliverable is a plan (`documentation/voice-plan-status.md`,
section 5) that a person approves before any code is written. The reason: this repository has
moved fast (landmark pipeline, decoder, LCD, telemetry), and anything written about it a day
ago may be stale. Gather the current state first, then plan, then ask.

## 1. The goal, as the team stated it

- `SPEC.md` (fingerspell repo): v1 is camera -> letters -> words on screen. Stretch, only
  after v1 is frozen: "mic keyword model for hearing-person replies".
- `docs/SPRINTS.md` S4 stretch #1: "Mic DS-CNN replies (yes/no/stop/go) on the NPU. Lifts
  Multimodal 15%."
- Cut list, first to go: "autocomplete -> mic -> display UI". Voice is the first stretch item
  cut if anything slips. Never drop: camera -> NPU -> letter, and the README performance numbers.
- Judging (Devpost): "Multimodal and Sensor Integration - 15 points: meaningful use of
  cameras, microphones and/or sensors. Reward quality of integration and sensor fusion rather
  than simply the number of inputs used." A badly integrated microphone scores nothing.

So the voice feature is: the hearing person speaks a short reply ("yes", "no", "stop", "go",
...) into the board's microphone, the board recognises it on the NPU, and the reply appears
on the LCD next to the signer's spelled text. A two-way exchange on one device.

## 2. What exists outside this repo (verified 2026-10-18)

Two tiers, both PyTorch + ExecuTorch on the Ethos-U85, both from Arm/Alif:

**Tier A: keyword spotting (fits in MRAM, realistic in hours).**
- Model: DS-CNN "small", MLPerf Tiny keyword spotting. 12 classes: silence, unknown, yes,
  no, up, down, left, right, on, off, stop, go. **23,756 parameters**, ~2.7 MMACs per 1 s
  clip (counted). Int8 PTE is ~35 KB.
- Reference PyTorch definitions: `examples/models/mlperf_tiny/ds_cnn.py` in
  `pytorch/executorch` (also a `streaming_wakeword.py`, 47 K params, one 40-dim feature
  frame per call), and `samples/modules/executorch/models/kws/model.py` in `alifsemi/sdk-alif`.
  Note: these are architectures. Check which repo ships **trained weights** before
  planning on them; the Alif sample's README says it loads pre-compiled `.pte` files and the
  model dir has `model.py` only. If no PyTorch weights exist, training on Google Speech
  Commands v2 (~1 hour on a laptop; Arm's ML-Zoo reports 94-95% for DS-CNN) or converting
  Arm's TFLite `ds_cnn_clustered_int8` weights are the options.
- Reference pipeline (Alif ML Embedded Evaluation Kit, `docs/use_cases/kws.md`): 16 kHz
  mono, 640-sample (40 ms) window, 320-sample (20 ms) stride, 10 MFCC coefficients, 49 frames
  = one 1-second input of shape 49x10; a new inference every 0.5 s on a sliding window;
  score threshold 0.7. Alif's Zephyr sample reports **~107 ms per inference** on an Ensemble
  kit (that number is from the README; measure, do not quote it as ours).
- Audio on this board: the Alif pack has `Boards/DevKit-e8/Drivers/vstream_audio_in.c`
  (vStream AudioIn over `Driver_SAI`, defaults 16 kHz / 16-bit) and `board_defs.h` names a
  "MIC Input" on I2S instance 3 / LPI2S, with `BOARD_LPPDM_ENABLED 1`; our `RTE_Device.h` has
  `RTE_I2S3`, `RTE_LPI2S`, `RTE_PDM`, `RTE_LPPDM` enabled. **Which microphone is physically on
  the DevKit-E8 (PDM MEMS mic vs I2S codec vs none without an add-on board) is not
  verified.** Finding out is step 3.4.

**Tier B: speech-to-text (needs external flash; not a same-day task).**
- Alif's `alif_asr` sample (`alifsemi/sdk-alif`, Zephyr) runs a Conformer from
  `Arm/stt_en_conformer_executorch_small` (Hugging Face, Oct 2025): 8.7 M params, WER 6.47%
  LibriSpeech test-clean, ~0.35 GMACs per second of audio, 7-second clip (700x80 mel) per
  inference, greedy CTC over a 129-piece SentencePiece vocabulary, **weights in the 32 MB
  OSPI flash** (`CONFIG_MODEL_IN_EXT_FLASH`, XIP at 0xC0000000, programmed with J-Flash).
  Card figures: 10.8 MB read-only data in external memory, 5.7 MB SRAM peak.
- Why not now: it is a Zephyr application on the HE core, needs the OSPI flash programmed
  with Segger J-Flash (a separate stub install and tool), runs on 7 s clips (not streaming),
  and the U85 memory configuration (`Ethos_U85_SYS_Flash_High`, scratch in SRAM1) differs
  from ours. Port it after the hackathon if the demo wants free speech.
- Also upstream in `pytorch/executorch/examples/arm`: `silero_vad_example_ethos_u` (voice
  activity detection, 32 ms frames), `smollm2_example_ethos_u` and
  `tinystories42m_example_ethos_u` (LLMs, w8a16 KV cache, ~103 MiB PTE in OSPI).

## 3. Gather context before planning (do all of these; write down what you find)

Read, in this order, and note anything that contradicts this brief:

1. **The repo as it is now.** `git log --oneline -40` on `fingerspell`, then
   `documentation/landmark-plan.md`, `documentation/hand-detection-handoff.md`,
   `documentation/temporal-average-plan.md`, `documentation/display-plan.md`,
   `documentation/fingerspell-bringup.md`. Then `src/app_main.cpp` (the live loop: what runs
   per frame, how the loop is timed, how `g_fs_*` telemetry is exposed),
   `src/fs_api.h` (the team contract: the decoder and UI interfaces you must plug into),
   `src/fs_decoder.c`, `src/fs_display.cpp`, `src/fs_landmark.c`, `src/fs_detect.c`,
   `src/fs_camera.c`. Answer: what is the current per-frame budget (fps, ms per stage), how
   much MRAM and SRAM0 are free (`arm-none-eabi-nm -S -n` on the `.axf`, Vela reports in
   `ai_layer/`), and which board peripherals are already initialised.
2. **The AI-layer flow.** `create_ai_layer.py` is the multi-method version ported from
   `hackathon-pico-faces`: read `model/model.py` to see how `detect`, `landmark`/`classify`
   methods are declared, calibrated and quantised, and how `model_params.h` is produced. A
   `kws` method must be added the same way, not as a separate export script.
3. **Tests and tools.** `tests/fs_tools.py` (flash, UART capture, J-Link reads,
   `app_problem`), `tests/board_telemetry.py` (reading the loop over J-Link when the UART is
   busy), `tests/test_step4_5_camera_live.py` and `tests/test_landmark.py` (how on-board
   tests are structured). Note how long the full suite takes with `FS_HW=1`.
4. **The microphone.** Verify what the DevKit-E8 has: the Alif DevKit-E8 user guide
   (schematic or board description), the pack's `Boards/DevKit-e8/board_defs.h` (MIC lines
   ~105-160, LPPDM/LPI2S flags), `pins.h` (PDM/I2S pin mux), and
   `Boards/DevKit-e8/Drivers/vstream_audio_in.c` (which SAI instance, bit depth, DMA, block
   size). Decide PDM vs I2S, which core/instance, and whether the vStream wrapper is usable
   (remember the camera: the vStream VideoIn wrapper hid CPI errors and its per-frame re-arm
   failed BUSY, so prefer reading the driver and using it directly if the wrapper is thin).
5. **The KWS model weights.** Check `pytorch/executorch` (`examples/models/mlperf_tiny/`,
   any `*.pt`/`*.pth` or download URLs), `alifsemi/sdk-alif` (`samples/modules/executorch/
   models/kws/`, README "Building the Model" section), Arm ML-Zoo (`models/keyword_spotting/
   ds_cnn_small/`), and Hugging Face (`search=ds_cnn`, `keyword spotting speech commands`)
   for trained DS-CNN weights in PyTorch or TFLite. Record exactly what exists, its licence,
   and the MFCC parameters it was trained with.
6. **Board state and rules.** `documentation/fingerspell-bringup.md` gotchas, plus: never
   re-arm the CPI per frame; never touch CPI registers from J-Link; never call
   `SERVICES_set_run_cfg` with memory blocks; flash with J-Link then test over the UART only;
   quit VS Code before using the serial port; build with CMSIS-Toolbox 2.15.0 from the
   terminal. Add any new rules you find in the newer docs.
7. **The team's schedule.** Submission closes Sunday 5:00 pm PDT (`docs/SPRINTS.md`,
   fingerspell repo; confirm with the organisers). Note the time you start.

## 4. Open questions the plan must answer

- Is there a usable microphone on the DevKit-E8 without extra hardware? If not, voice is
  out for the hackathon; say so and stop.
- Do trained DS-CNN weights exist in a usable form, or does the plan include training? If
  training: data download (~2.3 GB Speech Commands v2), time, and the MFCC front end must
  match the training features exactly (window, stride, number of mels, log scale,
  normalisation).
- Where does the MFCC front end run and how long does it take on the M55 at 400 MHz
  (CMSIS-DSP has `arm_mfcc_f32`; the MLEK has a reference implementation in
  `source/lib/mlek/common/Mfcc.hpp`)? It must not steal more than a few ms per 20 ms stride
  from the camera loop.
- Threading: the live loop is a single `for(;;)`. Audio arrives continuously by DMA; the
  KWS inference (1 s window every 0.5 s) must be interleaved with camera frames without
  dropping either. Options: run KWS every N camera frames from the same loop; or a
  SysTick/DMA-interrupt-driven ring buffer with the loop polling "window ready". No RTOS is
  in this build.
- NPU sharing: the camera pipeline runs 1-3 NPU methods per frame; a KWS method in the same
  `.pte` adds one more every 0.5 s. Check the scratch/temp pool sizing for the largest method
  and that Vela keeps DS-CNN fully on the NPU (it should: conv, depthwise, pool, FC).
- Memory: KWS is tiny, but the audio ring (1.5 s at 16 kHz 16-bit = 48 KB) and MFCC buffers
  need a home; DTCM is nearly full (`display-plan.md`, `landmark-plan.md`).
- UX: what does a recognised word do? Proposal to validate with the team: show it as the
  hearing person's reply line on the LCD ("> YES") and in the UART log, with a 0.7 threshold
  and a 1 s refractory period; "stop"/"go" could also control the decoder (pause/resume).
- What it costs the demo if it is late: the plan must keep the camera build shippable at
  every step (feature behind `APP_VOICE`, built only for DevKit-E8, off by default until the
  on-board test passes).

## 5. Deliverable of this brief: the plan, then stop

Write `documentation/voice-plan-status.md` with:
1. **Findings** from section 3, each with the file or URL it came from. Mark every
   assumption you could not verify.
2. **Decision**: go / no-go for the hackathon, in one line, with the reason (microphone
   availability, weights availability, time left).
3. **If go, the phased plan**, each phase ending in a check a person can see:
   - Phase 1, mic capture: a 1-second ring buffer fills; a UART line prints the RMS level
     every 0.5 s and it rises when you talk. Test: `tests/test_voice.py::Mic` reads 5 s of
     levels and asserts quiet vs. talking differ.
   - Phase 2, features: MFCC 49x10 on the M55; print the ms per window; compare one window
     against the Python reference (same WAV, max abs diff).
   - Phase 3, model: `kws` method in the `.pte`, 100% on the NPU, Vela SRAM noted; board
     prints the word, score and NPU ms; test with a known WAV played from the host and with a
     person saying the four words.
   - Phase 4, integration: reply line on the LCD, decoder hook, `APP_VOICE` define, all
     existing suites still green with `FS_HW=1`, fps impact measured and written down.
   - Phase 5, docs: a "Voice" section in `fingerspell-bringup.md` with the measured numbers
     (ms per inference, fps before/after, memory), and the submission README's
     workload-placement section updated (mic -> M55 MFCC -> U85 DS-CNN -> M55 decode/LCD).
     As of 2026-10-18 `README.md` on `fingerspell` is still the upstream hackathon guide with
     no such section; if that is still true, note it for whoever writes the submission README
     rather than inventing one here.
   Give each phase a time estimate and a cut line ("if Phase 2 is not done by HH:MM, stop
   and remove the branch from the demo build").
4. **Files you intend to touch** and the branch name (`voice/kws` off `fingerspell`).
5. **What you need from a person** before starting: the go-ahead, the mic/add-on board
   confirmation, and whether the team wants the words to control the decoder.

Post the status file in a PR against `fingerspell` on `erictweng/CMSIS-Executorch` (docs
only) and wait for approval in that PR before writing code.

## 6. Build, flash, test (unchanged from the other briefs)

```bash
T=~/.vcpkg/artifacts/2139c4c6
export PATH=$T/tools.open.cmsis.pack.cmsis.toolbox/2.15.0/bin:$T/tools.kitware.cmake/4.2.1/bin:$T/tools.ninja.build.ninja/1.13.2:$PATH
export AC6_TOOLCHAIN_6_24_0=$T/compilers.arm.armclang/6.24.0/bin
cbuild setup cmsis-executorch.csolution.yml --active DevKit-E8 --packs   # after layer edits
cbuild cmsis-executorch.csolution.yml --active DevKit-E8
.venv/bin/python -c "import sys; sys.path.insert(0,'tests'); import fs_tools as fs; print(fs.flash_and_start()[-300:])"
FS_HW=1 .venv/bin/python -m unittest discover -s tests -v
```
