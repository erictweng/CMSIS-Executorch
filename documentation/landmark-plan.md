# Fingerspell with hand landmarks: master plan (branch `landmark/pipeline`)

Written 2026-10-17 for the team. Read it top to bottom before touching code; it builds on
`documentation/hand-detection-plan.md` (the detector) and `hand-detection-handoff.md`
(what is on the board today, the board rules, the numbers).

## 1. Why

Today's pipeline is BlazePalm (palm detector, NPU) -> hand box -> MobileNetV2-0.5 letter
classifier (NPU). With the camera exposure and the crop fixed, a real hand in a busy room
reads the right letter but at 0.3-0.4 confidence: the classifier was trained on clean
studio-style photos and sees pixels, so background, lighting, blur and skin tone all reach
it. The landmark route makes those irrelevant by construction: MediaPipe's hand landmark
model turns the crop into 21 keypoints (trained by Google on diverse real-world hands), and
a tiny classifier maps keypoints to letters. The letter decision never sees a pixel.

Trade-off accepted: a second NPU model about the size of the detector (loop rate drops from
18-20 fps toward ~12 fps at today's Ethos-U speed) and a conversion job like BlazePalm's,
in exchange for accuracy that holds across people, rooms and cameras.

## 2. Architecture

```
camera 320x320 RGB565
  -> fs_detect_preprocess        320 -> 192 int8                       CPU  5.6 ms
  -> method "detect"             BlazePalm, raw head int8               NPU  ~17 ms
  -> fs_detect_decode + track    palm box, 7 keypoints, score, EMA      CPU  <0.1 ms
  -> fs_landmark_crop            rotated square hand crop -> 224 int8   CPU  ~4 ms (new)
  -> method "landmark"           hand landmark lite, int8 in/out        NPU  ~15-20 ms (new)
  -> fs_landmark_decode          21 x (x, y, z) in frame px, presence   CPU  <0.1 ms (new)
  -> fs_landmark_letters         normalise, MLP 63 -> 64 -> 24, softmax CPU  <0.1 ms (new)
  -> fs_decoder (unchanged)      commit / space / release rules
  -> UART line + LCD             box, skeleton, letter, timings
```

Models in the program (one `.pte`, two methods, both `quantized_io`):

| method | model | input | outputs | int8 weights |
|---|---|---|---|---|
| `detect` | BlazePalm palm_detection_lite 192 (today) | 1x192x192x3 int8 | regressors 2016x18, logits 2016 | 1.1 MB |
| `landmark` | MediaPipe hand_landmark_lite 224 (new) | 1x224x224x3 int8 | landmarks 63, presence 1, handedness 1, world landmarks 63 | ~1.0 MB |

The keypoint-to-letter MLP (~6 k parameters) runs on the CPU in C; its weights are emitted
into `ai_layer/model_params.h` by `create_ai_layer.py` from `model.get_params()` (the
generator already writes float arrays). MobileNetV2-0.5 and its `classify` method are
removed from the program; `model/fingerspell_letters.pt2` stays in the repo for the
fallback branch `detect/blazepalm`.

### Decisions

1. **Replace, do not stack.** The pixel classifier goes; one detector + one landmark model
   + an MLP. MRAM stays ~2.4 of 3.44 MB; SRAM scratch is shared by the methods in turn.
2. **int8 first (a8w8, quantized IO), measured against float** on 300 dataset images the way
   the detector was (landmark error in px, letter agreement). a16w8 only if a8w8 fails and
   Vela's scratch for it fits the temp pool (the detector's a16w8 wanted 2.3 MB; it will not).
3. **Rotate the crop on the CPU.** MediaPipe feeds the landmark model an upright hand: the
   crop is rotated by the wrist -> middle-finger angle from the palm keypoints we already
   decode, with the same 2.6x / -0.5 shift rule. A rotated 224x224 bilinear crop of the
   RGB565 frame is ~4 ms. The landmarks come back in crop coordinates and are rotated back
   into frame coordinates, so orientation information is kept.
4. **Letter MLP on frame-oriented, wrist-centred, size-normalised 2D+z landmarks** (63
   inputs). Do **not** rotation-normalise: P/K and Q/G differ only by orientation.
   Handedness is folded in by mirroring left hands to right (x -> -x) before the MLP, so
   one classifier serves both hands.
5. **Presence gate.** The landmark model's presence score (< 0.5) means "box but no hand":
   the tracker treats it as a miss (same hold / no-hand rules as the detector's).
6. **Training data for the MLP** is landmarks, extracted on the laptop with MediaPipe
   itself (the `hand_landmarker.task` already downloaded) from: Marxulia (10.9 k), the
   Kaggle ASL Alphabet set (subsampled ~1 k per class, for lighting/blur/angle variety),
   and real frames from this camera as they become available. Because the inputs are
   keypoints, the camera domain matters far less than for pixels.
7. **Self-test contract kept.** Boot still prints `Test_result: PASS`: the baked-in test
   crop goes through `landmark` + MLP and must give its letter.
8. **Fixed-centre fallback stays** behind `FS_FIXED_CENTRE_CROP` for A/B, as today.

## 3. Files

New:
- `ml/handlandmark_export.py` TFLite -> ONNX -> onnx2torch -> fixed-shape graph -> `model/hand_landmark_224.pt2`, checked against the TFLite interpreter (same recipe as `ml/blazepalm_export.py`).
- `ml/handlandmark.py` numpy reference: rotated crop rule, landmark decode/normalisation, MLP forward, letter probabilities.
- `ml/keypoint_dataset.py` MediaPipe landmarks for every training image -> `model/keypoints_{train,val}.npz` (labels, 63 features, handedness, source).
- `ml/keypoint_train.py` trains the MLP (PyTorch, minutes), reports val accuracy per source, exports `model/keypoint_mlp.npz` (W1, b1, W2, b2, normalisation constants).
- `model/hand_landmark_224.pt2`, `model/keypoint_mlp.npz`, `model/calib_landmark_224.npy` (crops for calibration; dataset crops until camera frames exist).
- `src/fs_landmark.c/.h` rotated 224 crop to int8, landmark decode (int8 -> px, presence), mirroring, normalisation, MLP, softmax. Pure C99, host-tested.
- `tests/host/test_fs_landmark.c`, `tests/test_landmark.py` (host: crop vs numpy, decode vs fixtures, MLP vs numpy, generator output; board: letter lines, no-hand, fps, warm reset), `tests/fixtures/landmark_*` (2 PNG crops + fp16 TFLite outputs).

Changed:
- `model/model.py` methods `detect` + `landmark` (both `quantized_io=True`), `classify` removed; `get_params()` adds the MLP weights and the landmark constants (`FS_LANDMARK_SIZE 224`, `FS_LANDMARKS 21`).
- `src/app_main.cpp` pipeline above; UART line `A 0.93 S 0.04 T 0.01 | box 112,84 96x96 s0.87 p0.98 | det 22.9ms cam 2.0ms lm 18.0ms ui 1.6ms | 12.5 fps` (`p` = presence, `lm` = crop + landmark + MLP) and `no hand | det .. | fps`; self-test through the landmark path.
- `src/fs_display.cpp` the 21-point skeleton drawn on the crop view (and the box inset as today).
- `src/fs_detect.h` exposes the palm keypoints' angle helper (already decoded).
- `cmsis-executorch.cproject.yml` adds `fs_landmark.c/.h` (`optimize: speed`).
- `board/*/app_mem_regions.h` only if Vela's scratch asks for a larger temp pool (SRAM0 budget: ring 0.6 + method 1 + temp 1 + LCD 0.75 of 3.94 MB; the method pool can shrink).
- `tests/test_display.py`, `tests/test_step4_5_camera_live.py`, `tests/test_step3_camera_init.py`, `tests/letter_check.py`, `tests/grab_crop.py` live-line regexes and the crop size.
- `create_ai_layer.py` none expected (`quantized_io` and `get_params()` arrays exist); a method with four outputs is new to `quantize_io()` and is covered by the loop.
- `documentation/fingerspell-bringup.md` "Landmarks" section with the numbers; `hand-detection-handoff.md` pointer; this file's status section.

Untouched: `src/fs_camera.c/.h`, `src/fs_decoder.c`, `src/fs_api.h` (the decoder contract stays: 24 probabilities in), the detector code and its tests.

## 4. Phases and gates

0. **Export spike (go / no-go, ~1 h).** Download `hand_landmark_lite.tflite`, convert,
   run `create_ai_layer.py` with `AI_LAYER_STRICT=1` and read: single delegate? CPU
   operators? Vela SRAM scratch? `.pte` size? Go if it delegates cleanly and the scratch
   fits a pool SRAM0 can give (< 1.6 MB). No-go -> back to the data route on
   `detect/blazepalm`.
1. **Reference + fixtures (half day).** `ml/handlandmark.py`, pt2 vs TFLite check,
   a8w8 vs float landmark error on 300 crops, fixtures, host tests for crop/decode.
2. **Keypoint classifier (half day).** Extract landmarks from the datasets with MediaPipe,
   train the MLP, report val accuracy per source (gate: >= 95 % on Marxulia val, >= 90 % on
   the Kaggle set), export weights through `get_params()`.
3. **Firmware (one day).** `fs_landmark.c`, `app_main.cpp`, display, tests; build all three
   targets; flash; measure `lm`, fps, SRAM; self-test PASS; letters on a real hand.
4. **Tune (half day).** Presence threshold, EMA on landmarks or probabilities, crop scale;
   real frames from the demo room into the MLP training set; `letter_check.py` score.
5. **Docs and PR** against `fingerspell` with the LCD photo and the numbers.

Gate after phase 3: median confidence on cleanly signed letters >= 0.7 in the demo room and
loop >= 12 fps; otherwise ship `detect/blazepalm` + the data route for the demo.

## 5. Risks

- **int8 accuracy of landmark regression.** Mitigation: measure in phase 1; per-channel
  weights are already on; if needed, a16w8 only for the landmark head is not available in
  the Arm quantizer, so the fallback is to accept a8w8 error if the MLP trained on
  int8-decoded landmarks still clears the gates.
- **NPU time.** Both models run ~5x slower than Vela's estimate on this board (unexplained,
  see the handoff section 4). At 12 fps the demo still works; fixing the gap helps both
  routes and is worth a parallel look (NPU clock, MRAM weight streaming, cache callbacks).
- **Conversion surprises.** Same classes of fix-up as BlazePalm (reshape, PReLU, pad);
  budgeted in phase 0/1.
- **Rotation on the CPU** costs ~4 ms; a nearest-neighbour crop halves it if needed.
- **Letters that rely on motion (J, Z)** stay excluded, as today.

## 6. Status

- 2026-10-17 20:50: phase 0 **go**. `hand_landmark_lite` exported (`ml/handlandmark_export.py`,
  one extra fix-up: dynamic squeeze) and matches TFLite to 2e-4 px. Strict export: `landmark`
  is one Ethos-U delegate with no CPU operators, 146 M MACs, int8 weights 1061 KiB, Vela
  scratch **1185 KiB** (temp pool raised to 1.5 MB on the DevKit), program 2,265,152 bytes
  with the detector. Output quantization: landmarks 1.18 px per step, presence/handedness
  1/255.
- 2026-10-17 21:30: phases 1-3 built. `ml/handlandmark.py` reference, `src/fs_landmark.c/.h`,
  host tests (`tests/test_landmark.py`), keypoint MLP (hidden 128, 95.8 % val on 7,763
  Marxulia landmarks from MediaPipe; confusions E/O, Q/G, U/R, P/Q), `app_main.cpp` pipeline,
  skeleton on the LCD. int8 landmarks within 0.8 px median of float (p90 4.2 px), presence
  agreement 96 %. DevKit image builds: MRAM 2.57 of 3.44 MB, SRAM0 3.82 of 3.94 MB. Not yet
  run on the board (USB dropped). Open: the lite model on a tight unrotated crop gives
  different landmarks than MediaPipe's pipeline; `ml/keypoint_dataset_board.py` extracts
  training landmarks through the board's own pipeline so the MLP sees what the board sees.
