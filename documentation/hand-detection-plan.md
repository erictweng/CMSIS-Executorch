# Hand detector + classifier pipeline on the DevKit-E8: task brief

A brief for the agent implementing the two-model pipeline: a palm detector finds the hand
in the camera frame, the crop goes to a letter classifier. Read it top to bottom before
touching code. Everything here applies to `erictweng/CMSIS-Executorch` (remote `fork`).

**Goal:** the board classifies the signer's hand wherever it is in the 320x320 camera frame
(full body, face in frame, distance), prints the letter per frame as today, and reports
"no hand" when there is none. Loop rate stays above 20 fps.

## 1. The pair

**Detector: MediaPipe BlazePalm (palm detection lite), 192x192.**
- Single-stage SSD on a BlazeNet backbone, ~1.0 M params, ~0.2 GMACs at 192², standard
  convs only (no swish/SE), so Vela keeps it on the NPU.
- Detects the palm, not the fingers, so the box is stable across every letter shape.
- Trained in the wild: full bodies, faces in frame, distance, lighting.
- Weights: Google's `palm_detection_lite.tflite` (MediaPipe Hands). Convert to PyTorch
  through one of the open BlazePalm ports (the MediaPipe-to-PyTorch "blazepalm" ports
  reproduce the graph and load the TFLite weights), verify the PyTorch port against the
  TFLite output on a few images (max abs diff < 1e-3 on the raw head outputs), then export
  through this repo's TOSA/Vela flow (section 4).

**Classifier: `abdollahhh/asl-sign-language-efficientnet-b0`, run at 128x128, int8.**
- Hugging Face `transformers` EfficientNet-B0, 4.08 M params, 99.0% reported on
  `Marxulia/asl_sign_languages_alphabets_v03` (the same dataset as the current model).
- ~125 MMACs at 128² (385 at 224²). SE blocks and swish: sigmoid and mul are native in Vela.
- 26 outputs (A-Z). J and Z are masked out in software; the 24-letter decoder stays.

Budget: ~0.33 GMACs per frame, both models on the NPU, roughly 5-8 ms NPU time. The camera's
40 ms frame period (25 fps) remains the limit.

## 2. Camera stream -> detector

Input: one 320x320 RGB565 frame from the CPI ring (`fs_camera_latest_frame()`, as today).

1. Convert RGB565 -> RGB888 (`rgb565_to_888` in `src/fs_camera.c`).
2. Resize 320x320 -> 192x192 with area averaging (box filter). The frame is square, so no
   letterboxing. Keep the scale factor 320/192 to map boxes back.
3. Normalise to float in [-1, 1]: `x / 127.5 - 1` (BlazePalm's contract). For the int8 build
   this folds into the quantizer's input scale/zero-point; on the board it is one quantize op.
4. Layout: NHWC 1x192x192x3 (the TFLite/TOSA graph's layout; this repo's export flow is NHWC).
5. Run the detector. Output: 2016 anchors x (score, cx, cy, w, h, 7 keypoints).
6. Decode on the CPU (~2016 iterations):
   - sigmoid on the scores, keep anchors with score > 0.5;
   - box = anchor centre + offset x anchor scale (standard SSD decode; generate the anchor
     table once at init exactly like MediaPipe's `ssd_anchors_calculator` for the 192² lite
     model: strides 8/16/16/16, 2 + 6 anchors per cell, 2016 total);
   - NMS: keep the highest-score box (one hand for fingerspelling), or weighted NMS over
     overlapping boxes for stability.
7. Hand box from palm box. The palm box covers the palm only; MediaPipe's rule is: rotate by
   the wrist->middle-finger keypoint angle, shift the centre 0.5 x box height toward the
   fingers, scale by 2.6x. Skip the rotation (the classifier was trained on upright hands and
   rotating costs CPU): do shift + scale, make the box square, clip to the frame.
8. Temporal smoothing: exponential moving average on box centre and size (alpha ~0.5), and
   hold the last box for ~5 frames when the detector misses one.
9. No hand (no score > 0.5 for N frames, N ~ 5): skip classification, emit "no hand" to the
   decoder, which treats it as idle (same path as the 1 s -> space rule).

## 3. Hand box -> classifier

1. Crop the box from the original 320x320 RGB888 frame (not the 192² detector input).
2. Resize to 128x128 with area averaging. Pad with black where the box runs off the frame
   rather than squashing it (the training crops are square).
3. Normalise the EfficientNet way: the model's `preprocessor_config.json` uses ImageNet
   mean/std, `(x/255 - mean)/std`, mean (0.485, 0.456, 0.406), std (0.229, 0.224, 0.225).
   Bake this into the export as a leading multiply/add on the graph so the board still feeds
   `uint8/255` (same trick as `ml/models.py` in the fingerspell repo: normalise with a
   multiply, not a divide, or the Arm quantizer's decomposition passes fail).
4. Layout: NHWC 1x128x128x3; the NCHW transpose is inside the exported graph.
5. Run, softmax over 26, zero J and Z, renormalise, top-3.
6. Decoder as planned (`src/fs_api.h`): commit at >= 0.6 for 3 consecutive frames, release
   gap before the same letter repeats, 1 s idle -> space.

## 4. Where this lands in the repo

Base the work on **`display/lcd-live`** (or `fingerspell` once that PR is merged): it carries
the LCD and the three board fixes (A32 stub offset, NPU buffers out of SRAM1, caches clean at
boot) that a 4 MB program and a second model need. Read first:
`documentation/fingerspell-bringup.md`, `documentation/display-plan.md`,
`board/DevKit-E8/README.md`, `src/fs_camera.c/.h`, `src/app_main.cpp`, `src/fs_api.h`.

**Two models in one program.** `fingerspell`'s `create_ai_layer.py` only knows the single
`get_model()` / `get_calibration_inputs()` contract. Branch `hackathon-pico-faces` has the
multi-method version (`get_methods()` returning `Method(name, module, example_inputs,
calibration, quantization)`, plus `get_params()` -> `model_params.h`); it exported two
methods (`dit_step`, `decode`) into one `.pte`. Port that `create_ai_layer.py` onto the
branch, then `model/model.py` exposes two methods:
- `detect`: BlazePalm, input 1x192x192x3, outputs the raw score and box/keypoint tensors
  (do not put sigmoid/decode/NMS in the graph; they are CPU work);
- `classify`: EfficientNet-B0 with the baked normalisation, input 1x128x128x3, 26 logits.
Calibrate `detect` on ~100 camera frames (grab them with `tests/grab_crop.py`-style dumps of
the full frame, or render the dataset images onto room backgrounds) and `classify` on ~300
dataset crops. Run `create_ai_layer.py` with `AI_LAYER_STRICT=1`: each method must be a single
Ethos-U delegate; print and fix any operator that stays on the CPU.

**Fine-tune before trusting the 99%.** EfficientNet-B0 was trained at 224². Fine-tune at 128²
on crops produced by *this detector* (run the PyTorch BlazePalm offline over the dataset, apply
the same box rule from section 2.7, crop, resize) so training crops and board crops match.
Report held-out accuracy at 128² before and after. Script it in the fingerspell repo's
`ml/` (train/export already exist there) and commit the resulting `.pt2` under `model/`.

**Board code.** New `src/fs_detect.c/.h` (resize to 192², anchors, decode, NMS, box rule,
EMA, no-hand logic), `fs_camera_get_crop()` now takes the box instead of the fixed centre
(keep the fixed-centre path behind a define for A/B tests), `live_loop()` runs `detect` then
`classify` and prints per frame:
`X 0.93 U 0.04 M 0.01 | box 112,84 96x96 s0.87 | det 3.1ms cam 2.0ms pre 0.8ms npu 4.2ms | 21.3 fps`
and `no hand | det 3.1ms | 21.9 fps` when there is none. The LCD shows the detected box on
the camera view.

**Memory.** Two `.pte`s in one program ~5 MB: needs the extended HP MRAM region
(`APP_MRAM_HP_SIZE` 0x370000, from the display branch). Method and temp pools: size from
Vela's report for both methods (expect < 1 MB total scratch); keep everything the NPU touches
in SRAM0 above `0x02010000`. Verify with `arm-none-eabi-nm -S -n` on the `.axf`.

## 5. Rules on this board (do not relearn these)

- Never re-arm the CPI per frame; never use the legacy single-buffer mode (runaway DMA).
  The camera runs in streaming mode with a 3-block ring and a canary; keep both.
- Do not access CPI registers (`0x49030000`) from J-Link.
- Do not call `SERVICES_set_run_cfg` with memory blocks.
- Flash with J-Link, then test over the UART only; read memory over J-Link only in a session
  that has had the debugger attached since the flash.
- VS Code holds the serial port while its Serial Monitor is connected; quit VS Code before
  running the tools. `lsof /dev/tty.usbmodem*` shows the holder.
- Build with CMSIS-Toolbox 2.15.0 from the terminal (commands in
  `documentation/fingerspell-bringup.md`), not VS Code's bundled 2.13.
- Board recovery (no UART, J-Link cannot connect): power cycle; if still dead, reinstall the
  debug stub with SETOOLS over SEUART (steps in `display-plan.md`, section 5).

## 6. Tests to add (`tests/test_detector.py`)

Host (no board):
- The PyTorch BlazePalm port matches the TFLite model on 5 images (raw outputs, max abs diff
  < 1e-3); the anchor table has 2016 rows and matches MediaPipe's for the first/last 5.
- The C decode (compiled on the host, or re-implemented in Python from the same constants)
  gives the same box as the Python reference on 3 saved detector outputs.
- The box rule (shift + 2.6x scale + square + clip) on 3 known palm boxes.
- `create_ai_layer.py` output: two methods, each a single delegate, 0 CPU ops besides
  quantize/dequantize; `.pte` size < 5.5 MB; Vela SRAM per method recorded.
- Classifier at 128² on the held-out split: accuracy >= 95% int8 (after fine-tuning), int8 vs
  float top-1 agreement >= 97%.

On the board (`FS_HW=1`, UART only):
- Boot self-test still prints `Test_result: PASS` (the baked-in B crop, through the
  classifier only).
- With a hand in view: >= 30 live lines with a box, scores > 0.5, box size between 48 and
  256 px, box centre moving with the hand (two captures with the hand left vs right).
- With no hand (lens covered or empty scene): `no hand` lines, no letter committed.
- Median fps >= 20; det + npu < 15 ms; no capture errors, no overrun, no model errors; warm
  reset mid-loop comes back.
- `tests/letter_check.py` run by a person: score posted in the PR, with the signer standing
  back so the hand is < 1/4 of the frame.

## 7. Definition of done
- Hand found anywhere in the frame; letter printed per frame; `no hand` when none.
- Both models on the NPU, >= 20 fps, all existing suites plus `test_detector.py` green on
  the board.
- Fine-tuned classifier `.pt2` and the BlazePalm `.pt2` committed under `model/` with the
  scripts that made them; `documentation/fingerspell-bringup.md` gets a "Detector" section
  with the measured numbers (fps, ms per stage, Vela SRAM, `.pte` size, held-out accuracy).
- PR against `fingerspell` on `erictweng/CMSIS-Executorch` with a photo of the LCD showing
  the box and the letter.
