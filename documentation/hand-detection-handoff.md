# Hand detection on the DevKit-E8: handoff (state on 2026-10-17, evening)

You are continuing the work described in `documentation/hand-detection-plan.md` (the brief:
BlazePalm palm detector + letter classifier on the Ethos-U85, box drawn on the LCD). Read the
brief first, then this file, then the files named in section 2. Sections 4 and 5 record what
was measured and decided (do not redo); section 6 is the remaining work.

**Where things stand.** The detector pipeline runs end to end on the board: `detect` (int8 in,
int8 out, no CPU operators) then `classify` per frame, box and letter on the UART and the
LCD, `no hand` without one. The section 5 bug (boxes off the hands) was the input range:
BlazePalm 192 takes `[0, 1]`, not `[-1, 1]`. The first board build spent 51 ms per frame in
the detector stage; see section 4 "Board timing" for the breakdown and what was done about it.

> **2026-10-17 late:** the classifier was replaced on branch `landmark/pipeline` (MediaPipe hand
> landmarks + keypoint MLP, motion letters Z/J in the decoder): read
> `documentation/landmark-plan.md` for that branch. This file describes `detect/blazepalm`
> (merged into `fingerspell`), which the landmark branch builds on.

## 1. Repo and branch

- Repo `erictweng/CMSIS-Executorch` (remote `fork`), working copy `/Users/ericweng/CMSIS-Executorch`.
- Work on branch **`detect/blazepalm`**. It was branched from `display/lcd-live` (the LCD work
  and the three board fixes, both verified on the board) and has the brief merged in.
  Commits on it, newest first:
  - `e260b0d1` Palm detector on the board: detect then classify, int8 boundary, box on the
    LCD (everything in sections 4-6 of this file; the commit message has the summary).
  - `6cccd3c4` Port of the multi-method AI-layer generator from `hackathon-pico-faces`
    (`create_ai_layer.py`: `get_methods()` / `get_params()` / `AI_LAYER_STRICT`), plus
    `ml/blazepalm.py` (anchors, decode, box rule, reference crops) and `ml/classifier_eval.py`.
  - `818e03a9` merge of `docs/hand-detection-plan` (the brief) into the branch.
  - `20844721` Display: live crop, letter, confidence bar, fps on the LCD; `ui x.xms` in the
    UART line; timings moved from the DWT to the 100 MHz REFCLK counter.
  - `482af306` DevKit-E8 board fixes: bulk SRAM above the A32 stub (`APP_SRAM_POOL_OFFSET`),
    `SRAM0_SRAM1_COMBINED 0`, caches cleaned in `main()`, no `SERVICES_set_run_cfg`.
- Untracked and **not ours**: `model/pico_faces/` (60 MB of pico-faces checkpoints,
  `dit_qat.pt`, `vae_final.pt`, that came along with the generator port). Delete or move it;
  do not commit it.
- `ai_layer/model_pte_data.inc` (the program's bytes) is git-ignored: run `create_ai_layer.py`
  after a fresh checkout before building (~3 min).
- Do not touch `display/lcd-live` or `fingerspell`; PRs go against `fingerspell`.

## 2. Read these

`documentation/hand-detection-plan.md` (the brief), `documentation/display-plan.md` (board
rules, recovery), `documentation/fingerspell-bringup.md` (build/flash, the "Display"
section), `board/DevKit-E8/README.md`, `src/fs_api.h`, `src/fs_camera.c/.h`,
`src/app_main.cpp`, `src/fs_display.cpp`, `create_ai_layer.py` (its docstring),
`model/model.py`, `ml/blazepalm.py`, `ml/blazepalm_export.py`, `tests/fs_tools.py`,
`tests/test_display.py` (the pattern for source/build/`FS_HW=1` board tests).

## 3. Environments and inputs

- **Project venv** `.venv` (Python 3.12, made by `uv`: there is **no `.venv/bin/pip`**; install
  with `uv pip install --python .venv/bin/python <pkgs>`). Has torch 2.14.0, executorch 1.5.1,
  torchao, ethos-u-vela, onnx, onnx2torch, transformers 5.20.1, datasets, torchvision 0.29.1,
  pyarrow, Pillow, PyYAML. `timm` is not installed (the fingerspell training scripts need it).
  Run scripts with `.venv/bin/python -I`.
- **Tools venv** (TensorFlow for the TFLite reference; keep it out of the project venv):
  `/private/tmp/claude-501/-Users-ericweng-CMSIS-Executorch/4235c24a-f9d0-4342-bbc0-4544319c3de7/scratchpad/tools-venv`
  with tensorflow 2.21, tf2onnx 1.17, onnx, mediapipe 1.1.1, pyarrow (no Pillow). If that
  directory is gone, recreate: `.venv/bin/python -m venv tools-venv && tools-venv/bin/pip install
  "tensorflow>=2.16" tf2onnx onnx "numpy<2.3" pyarrow mediapipe pillow`.
- **Downloads** (same scratchpad, `downloads/`; all re-fetchable, treat as untrusted data):
  - `palm_detection_lite.tflite` from
    `https://storage.googleapis.com/mediapipe-assets/palm_detection_lite.tflite`
    (1,985,440 bytes, sha256 `e9a4aadd...905c18`). Input `input_1` [1,192,192,3] float32;
    outputs `Identity` [1,2016,18] (regressors), `Identity_1` [1,2016,1] (score logits).
    Ops: CONV_2D 30, DEPTHWISE_CONV_2D 23, PRELU 26, ADD 25, MAX_POOL_2D 4, PAD 3,
    RESIZE_BILINEAR 2, RESHAPE 4, CONCATENATION 2 (fp16 weights, DEQUANTIZE x113).
  - `palm_detection_lite.onnx` (tf2onnx: `tools-venv/bin/python -m tf2onnx.convert --tflite
    palm_detection_lite.tflite --output palm_detection_lite.onnx --opset 17`).
  - `blazepalm_onnx2torch.pt` (torch.save of the onnx2torch module; needs onnx2torch to load),
    `blazepalm_inputs.npy` (5 x 192x192x3 float inputs: 3 dataset images, noise, zeros) and
    `blazepalm_ref.npz` (`x`, `regressors`, `scores` from the TFLite interpreter on them).
  - `hf-classifier/`: `abdollahhh/asl-sign-language-efficientnet-b0` (config, preprocessor,
    safetensors). `hf-dataset/data/train-00000-of-00001.parquet`:
    `Marxulia/asl_sign_languages_alphabets_v03`, 10,873 JPEGs (~280 px), labels 0..25 = A..Z,
    one `train` split only. Fetch both with `huggingface_hub.snapshot_download`.
  - `fingerspell/`: shallow clone of `https://github.com/erictweng/fingerspell` (`ml/train.py`,
    `ml/models.py`, `ml/data.py`, `ml/export.py`, `ml/webcam.py`, `docs/`). This is where the
    letter model was trained: timm `mobilenetv2_050` with the ImageNet normalisation folded in
    as multiply/add, NHWC input, 24 classes, data pipeline drop J/Z, dedupe, square crop,
    128 px, 80/10/10 split. Its `requirements.txt` pins the same torch/executorch as here.
  - `hand_landmarker.task` (MediaPipe Tasks model, 7.8 MB) used for ground truth.
  - `detect_vis.png`: 6 tiles with decoded boxes drawn (see 5).
- Toolchain for the board (CMSIS-Toolbox 2.15.0, AC6 6.24), for a plain shell:
  ```
  T=~/.vcpkg/artifacts/2139c4c6
  export PATH=$T/tools.open.cmsis.pack.cmsis.toolbox/2.15.0/bin:$T/tools.kitware.cmake/4.2.1/bin:$T/tools.ninja.build.ninja/1.13.2:$PATH
  export AC6_TOOLCHAIN_6_24_0=$T/compilers.arm.armclang/6.24.0/bin
  AI_LAYER_STRICT=1 .venv/bin/python create_ai_layer.py cmsis-executorch.cbuild-mlops.yml   # ~2 min
  cbuild setup cmsis-executorch.csolution.yml --active DevKit-E8 --packs
  cbuild cmsis-executorch.csolution.yml --active DevKit-E8
  FS_SKIP_BUILD=1 .venv/bin/python -m unittest discover -s tests -v        # 86 tests, no board
  FS_HW=1 FS_SKIP_BUILD=1 .venv/bin/python -m unittest discover -s tests -p "test_display.py" -v
  ```
  Build the other targets with `--active AppKit-E8` / `--active SSE-320-U85`; a `--context`
  build wipes `tmp/DevKit-E8` and later tests then find no objects.

## 4. What has been established (do not redo)

**Detector conversion.** `ml/blazepalm_export.py` makes `model/blazepalm_192.pt2`. It replaces
onnx2torch's Reshape (4), Resize (2), PReLU (26) and dynamic Pad (3) modules with fixed
equivalents (torch.export cannot guard on their tensor-valued branches), clones every weight
to its own contiguous storage (the pt2 weight packager overflowed otherwise) and strips
`nn_module_stack` metadata (onnx2torch names contain `/ ; ,`, which breaks deserialisation).
The result matches the TFLite interpreter on the 5 reference inputs: worst score-logit
difference 2.3e-4, worst relative regressor difference 2.4e-6. The exported graph concatenates
the stride-8 head first (24x24x2 = 1152 anchors, reshaped (y, x, anchor, coord)) then the
stride-16 head (12x12x6 = 864), the same order as `ml/blazepalm.anchors()` (2016 rows).

**Vela.** `detect` is a single Ethos-U delegate (the Arm backend has a PReLU decomposition
pass; the bilinear 2x resize is accepted), 290 M MACs. With `quantized_io=True` (section 4
"Board timing") it has **no CPU operators at all**; `classify` keeps its quantize/dequantize
(49k values, ~1 ms). Two-method program (both a8w8): **2,059,552 bytes**. Vela SRAM scratch:
**detect a8w8 885.7 KiB** (a16w8 would be 2305.9 KiB), classifier 321 KiB; both fit the 1 MB
temp pool. The DevKit and AppKit builds link with the HP MRAM region extended to `0x370000`;
the FVP build puts `model_pte.o` in DDR (`ER_ROM2` of the SSE-320 scatter file) because the
2 MB FPGA SRAM holds the runtime only.

**Quantization** (with `[0, 1]` inputs, calibration on 100 dataset crops, against the float
model on 300 images = 150 direct + 150 composites, hit at score >= 0.5):

| | hit/miss agree | quant hits / float hits | mean score diff | box corner distance (192 px) |
|---|---|---|---|---|
| a8w8 | 274/300 | 230 / 238 | 0.040 | median 2.5 px, p90 5.6 px |
| a16w8 | 293/300 | 237 / 238 | 0.012 | median 0.7 px, p90 1.8 px |

Decision: **a8w8** (fits the SRAM; the EMA smooths the 2.5 px jitter). Flip the method's
`quantization` in `model/model.py` if SRAM is ever found for a16w8.

**Board timing** (first build, Debug = `-O1` everywhere): `det 51.5ms` per frame, of which the
CPU resize to a float 192x192 input took 19.7 ms and the method call 31.5 ms, with the NPU a
small part of that: ExecuTorch's portable `quantize_per_tensor` (110k floats) and
`dequantize_per_tensor` (38k) at `-O1` run at ~70 cycles per element. Fixes: (1) `Method(...,
quantized_io=True)` in `create_ai_layer.py`/`model/model.py` applies ExecuTorch's
`quantize_io_pass` so the method takes int8 and returns the int8 head; the scale/zero point
of each tensor land in `ai_layer/model_params.h` (`DETECT_INPUT0_SCALE` ..., input is pixel -
128) and the firmware converts (`fs_detect_preprocess` writes int8, `fs_detect_decode` reads
int8); (2) the resize writes int8 straight from the 5:3 box filter with no per-pixel
branches; (3) `src/fs_detect.c` is built with `optimize: speed` (file-level in the cproject).
Result on the board: `det 22.9ms` (`det split: resize 5.6ms detector 17.3ms`), `no hand`
lines at 18-20 fps; the classifier's `npu 8.1ms` is unchanged. **Still over the brief's budget
(det + npu < 15 ms, >= 20 fps)**: `test_detector_and_classifier_meet_the_brief_budget` fails
and says so. The remaining 17.3 ms is the Ethos-U method itself with no CPU operators left:
290 M MACs would be ~3 ms at 256 MACs/cycle and 400 MHz, and the classifier shows the same
gap (Vela estimated 2.26 ms, the board runs 7.8-8.1 ms, noted in `fingerspell-bringup.md`
before this work). Unverified suspects, in order: the NPU clock actually programmed on the
board (`SystemCoreClock` is printed; the NPU_HP clock is not), weights streamed from MRAM at
every inference (`Shared_Sram`: 1.1 MB per detector run; Vela's own estimate is not printed by
`create_ai_layer.py`, run Vela on the dumped TOSA with `AI_LAYER_DUMP=<dir>` and
`--verbose-performance` to get it), and the whole-cache clean/invalidate callbacks in
`board/DevKit-E8/ethosu_cb_dcache.c`. The CPU resize (5.6 ms) could drop further with Helium
or a 4x subsample, but it is not the bottleneck.

**Classifier.** `ml/classifier_eval.py` on 1,000 held-out images (random split, seed 0, of
the same dataset the HF model was trained on, so optimistic): EfficientNet-B0 **99.5 % at 224,
55.6 % at 128**. It has 4.04 M parameters, i.e. ~4.0 MB of int8 weights: with the 1.1 MB
detector and 0.25 MB of code it does **not fit** the 3.44 MB MRAM code region (total MRAM is
5.5 MB and the lower 2 MB is the HE core's region, used by the ATOC/boot config; do not take
it without checking `.alif/M55_HP_mram_cfg.json`). Its `id2label` is **not alphabetical**
(`A,B,K,L,M,...,C,U,...,D,E,...`) and its preprocessor std is ~0.478 per channel, not
ImageNet's 0.229; use `config.json`/`preprocessor_config.json`, not the brief's values.
Decision needed from Eric (not taken): keep MobileNetV2-0.5 and fine-tune it with the
fingerspell `ml/` scripts on detector-geometry crops (fits easily, recommended), or a
~2 MB model (mobilenetv3_small_100 / mobilenetv2_100 are in `ml/models.py`'s candidates).

**Camera.** Since ~15:35 the sensor delivers all-zero frames (probe pixels `0x0000`, mean 0)
at 19 fps with no errors; the stream is healthy, the scene is black (lens covered or the
module facing the desk after the LCD was mounted?). Eric was asked and has not answered.
Until it sees the room there are no camera frames for `model/calib_detect_192.npy` and no
board test can pass its crop checks. `tests/grab_crop.py` saves the 128 crop as PNG; a
`grab_frames.py` for full 320x320 frames (dump via the UART like `dump_crop()`) still has to
be written.

## 5. Settled: the input range (was "boxes are not on the hands")

Confirmed and fixed: MediaPipe's 192x192 palm models take `[0, 1]` inputs
(`output_tensor_float_range 0..1` in `palm_detection_cpu.pbtxt`); `[-1, 1]` is the 2019
256x256 model. Float detector on 150 dataset images / 150 composites (dataset hand at 160 px
on a blurred background in a 320 frame): `[-1, 1]` hit 47/150 and 53/150 with the palm centre
inside the pasted hand 23 %; `[0, 1]` hit 114/150 and 124/150, centre inside the hand 98 %,
hand-box IoU with the pasted hand 0.59 median (the hand box is 2.6x the palm, so 1.0 is not
expected). Changed: `calib_detect()` (`/255`), the brief's section 2.3, the TFLite reference
(regenerated with `[0, 1]` inputs; the pt2 matches it to 1e-4 on the logits), and the
quantization measurement (section 4). Fixtures: `tests/fixtures/detect_{hand,composite,black}.png`
+ `detect_ref.npz` (fp16 TFLite outputs). Scratch scripts of the measurement are not in the
repo (`range_test.py`, `quant_agree.py`, `make_ref.py` in the session scratchpad).

## 6. Done on the board side, and what remains

Done (brief sections 2-4 and 6): `src/fs_detect.c/.h` (resize, anchors from the index,
int8 decode, hand box, EMA/hold, hand-box crop; host harness `tests/host/test_fs_detect.c`
and `tests/test_detector.py`), `src/app_main.cpp` (`detect` then `classify`, the UART lines
of the brief, `-DFS_FIXED_CENTRE_CROP` for A/B, `det split:` breakdown every 64 frames),
`src/fs_display.cpp` (80x80 inset of the whole frame with the box in the crop view's corner;
the layout and `tests/test_display.py`'s frame-buffer probes are unchanged), the regexes in
`tests/test_display.py` and `tests/test_step4_5_camera_live.py` (box lines or `no hand` lines),
the `fs_camera_get_crop` symbol in `tests/test_step3_camera_init.py` (the compiler inlines
`fs_camera_crop` now that only the A/B path calls it).

Also fixed on the board with Eric at the camera (2026-10-17 evening): the hand-box shift
follows the wrist -> middle-finger keypoints (a fixed "up" put the crop on the forearm with
this camera's mounting), and the sensor's auto-exposure is now enabled in `fs_camera_init()`
(frames were almost black; see `fingerspell-bringup.md`, "Camera exposure"). With both, a
real F in front of a busy room reads F on every frame but only at 0.3-0.4: the classifier
itself (dataset crops, plain backgrounds) is now the limit, which is item 3 below.

Remaining:
1. **Hand in view.** The board tests only saw "no hand" plus a few boxes on the scene the
   camera happens to look at; nobody has signed at the detector yet. Run `FS_HW=1
   FS_EXPECT_HAND=1 ... test_detector.py` with a hand in view, `tests/letter_check.py` standing
   back, and take the LCD photo for the PR.
2. **Calibration frames.** `model/calib_detect_192.npy` from ~100 real camera frames: write
   `tests/grab_frames.py` (dump the 320x320 RGB565 frame over the UART like `dump_crop()`),
   then re-run `create_ai_layer.py`; until then `detect` is calibrated on dataset crops (the
   generator prints a note).
3. **Classifier fine-tune** on detector-geometry crops (Eric's decision from section 4 is
   still open; keeping MobileNetV2-0.5 is recommended). Needs `timm` in a venv and the
   fingerspell `ml/` scripts (scratchpad clone, re-fetchable).
4. **Tests on the board** for the suites touched here (`test_display.py`, `test_step4_5_camera_live.py`,
   `test_step3_camera_init.py` with `FS_HW=1`): run after any change to the live line format.
5. Docs: `documentation/fingerspell-bringup.md` has a "Detector" section; add the board
   numbers from the final run and the PR against `fingerspell` with the LCD photo.

## 7. Board rules learned here (in addition to the brief's section 5)

- The M55 DWT cycle counter only runs while a debugger asserts NIDEN: a board that booted on
  its own printed `0.0ms` and a runaway fps. Timings use `REFCLK_CNTRead->CNTCVL` (100 MHz,
  `SystemREFClock`, enabled by `sys_busy_loop_init_ns()`); keep it that way.
- J-Link flashing a board that booted standalone with the camera streaming fails about half
  the time ("CPU could not be halted", "Could not find core"); a plain retry has always worked
  so far. `tests/fs_tools.flash_and_start()` does one attempt.
- VS Code holds `/dev/tty.usbmodem0012195786141` while its Serial Monitor is connected
  (`lsof /dev/tty.usbmodem*` shows `Code Helper`); the UART tools then fail with "Resource
  busy". Ask Eric to close the monitor; quitting VS Code kills an agent running inside it.
- `fs_display.cpp` must never call `SERVICES_set_run_cfg` with memory blocks (test enforces).
- Reading board state without the UART: halt, `mem32` the symbols, go (J-Link); a script
  that reads loop/camera counters, `display_started`, and frame-buffer words twice is at
  `<scratchpad>/board_state.py` (symbols via `fromelf --text -s`; anonymous-namespace names
  are mangled, e.g. `_ZN12_GLOBAL__N_111framebufferE`).
- Reverting regenerated `RTE/_Debug_AppKit-E8/*` and `RTE/_Debug_SSE-320-U85/*` headers after
  building those targets keeps the diff clean (only version stamps change).

## 8. Definition of done

Unchanged from the brief's section 7. Commit messages end with
`Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` (or your own attribution line),
PR descriptions with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.
