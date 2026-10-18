# How the code works

Fingerspell turns a camera's view of a signing hand into text on the DevKit-E8. One frame
goes through two neural networks on the Ethos-U85 NPU and a small amount of C on the
Cortex-M55, about 15-18 times a second. This page follows one frame through the code; the
build, flash and test commands are in the [README](../README.md).

## The pipeline, one frame

```
camera 320x320 RGB565        src/fs_camera.c        CPI streaming into a 3-frame ring in SRAM0
      |
      v
palm detector (NPU)          src/fs_detect.c        BlazePalm 192x192 -> 2016 anchors;
      |                                              decode + best box on the CPU, ~23 ms
      v
hand box, smoothed           src/fs_detect.c        2.6x the palm, held 5 frames on a miss
      |
      v
rotated 224x224 crop         src/fs_landmark.c      wrist -> middle finger points up
      |
      v
hand landmarks (NPU)         src/fs_landmark.c      hand_landmark_lite -> 21 (x, y, z),
      |                                              presence, handedness, ~6 ms
      v
levelling by gravity         src/fs_orient.c        accelerometer (src/fs_imu.c) gives the
      |                                              camera's roll; landmarks are rotated
      v                                              so "up" is the real vertical
63 features -> letter MLP    src/fs_landmark.c      wrist-centred, size-normalised, mirrored
      |                                              to a right hand; 63-128-24 MLP on the CPU
      v
decoder                      src/fs_decoder.c       averages, commits, repeats, Z/J by motion
      |
      v
panel + joystick             src/fs_display.cpp     camera view, hand box, skeleton, letter,
                             src/fs_commit_hold.c   confidence, text; buttons edit the text
```

`src/app_main.cpp` owns the loop (`live_loop()`): it takes the newest camera frame, calls
`detect()`, `landmark()` and `letters()` in turn, steps the decoder, reads the joystick, draws
the panel and prints one line per frame. The model program (`ai_layer/model.pte`) holds two
methods, `detect` and `landmark`, both int8 with int8 inputs and outputs so the CPU never
converts floats at the model boundary (`model/model.py`, exported by `create_ai_layer.py`).

## Stage by stage

**Camera.** The MT9M114 streams 320x320 RGB565 frames over MIPI CSI-2 into a ring of three
buffers; the loop always reads the newest complete frame (`fs_camera_latest_frame`). The
camera is never re-armed per frame: the streaming mode with the ring is what keeps the board
stable, and a canary after the ring catches a runaway DMA.

**Palm detection.** `fs_detect.c` resizes the frame to 192x192, quantizes it, and runs
MediaPipe's BlazePalm. The NPU returns raw SSD outputs; the C code builds the 2016-anchor
table once, applies the sigmoid, takes the best anchor above the score threshold, and turns
the palm box into a hand box (shifted toward the fingers, scaled 2.6x, squared). The box is
smoothed over frames and kept for a few frames when the detector misses, so the picture does
not flicker. The Python reference for all of this is `ml/blazepalm.py`.

**Landmarks.** `fs_landmark.c` cuts a 224x224 crop out of the frame rotated so the hand
points up (the angle from the wrist to the middle-finger knuckle), because the landmark
model expects an upright hand. hand_landmark_lite returns 21 keypoints in crop pixels plus a
presence score (is there a hand?) and a handedness score. The keypoints are mapped back to
frame pixels for the panel and the decoder's motion rule. Presence below the gate counts as
"no hand", with a short hold so one bad frame does not reset everything.

**Orientation.** The board's accelerometer (`fs_imu.c`) gives the direction of gravity; a
two-pose calibration (`tests/imu_calibrate.py` -> `src/fs_orient_cal.h`) maps it into image
axes. `fs_orient.c` rotates the landmarks about the wrist by the camera's roll, so tilting
the board does not change what the letter MLP sees.

**Letters.** The 21 keypoints become 63 features: wrist at the origin, scaled by the hand
size, x mirrored so left and right hands look the same. A 63-128-24 MLP (weights in
`ai_layer/model_params.h`, trained by `ml/keypoint_train.py` on MediaPipe landmarks of the
dataset photos and on landmarks captured through this very pipeline) gives probabilities
for the 24 static letters (A-Y without J; Z and J come from motion).

**Decoder.** `fs_decoder.c` is pure C99 with the same code on the laptop and the board
(`tests/host/test_fs_decoder.c`, 361 checks). It averages the probabilities over frames,
commits a letter once it has stayed above the commit threshold for two frames, keeps it as
the live letter down to a lower hold threshold, and lets the same letter commit again only
after the hand was out of view briefly (a bounce for "LL"). D and I are held as "pending":
if the fingertip then travels about a hand's width the letter becomes Z or J. Spaces come
from the joystick.

**Panel.** `fs_display.cpp` draws into a 480x800 RGB565 frame buffer that the display
controller scans out of SRAM0. Static parts are drawn once; each frame rewrites the camera
view (the whole frame at 6/5 with the hand box and skeleton), the 100 px inset of the crop
the landmark model saw, and only the text fields whose content changed. A newly committed
letter sits dimmed for 0.7 s before it joins the confirmed line, and a letter committed in
that time replaces it (`fs_commit_hold.c`), so a wrong first read that is corrected at once
never lands. Render time is 2-3 ms of a 10 ms budget.

**Joystick.** `fs_joystick.c` reads four switches on GPIO15, `fs_input.c` turns presses into
one event each: RIGHT space, LEFT backspace, DOWN clear, UP swap the panel's large and small
views.

## Memory and timing

- MRAM: the program with both models is about 2.5 MB in a 3.4 MB code region.
- SRAM0 (4 MB, the only bulk SRAM the NPU reaches), above the 64 kB boot stub: the camera
  ring (0.6 MB), the model's method pool (1 MB) and temp pool (1.5 MB, the NPU scratch of
  both methods), the LCD frame buffer (0.75 MB). SRAM1 is unused.
- Per frame: detector ~23 ms, landmarks ~6 ms, decoder and panel ~3 ms; 15-18 fps with a
  hand in view. The timings use the SoC's 100 MHz REFCLK counter (the M55's cycle counter
  only runs with a debugger attached).

## Where to look

| Question | File |
|---|---|
| The contracts between stages | `src/fs_api.h` |
| The loop, the model calls, the per-frame line | `src/app_main.cpp` |
| Models, calibration, exported methods | `model/model.py`, `create_ai_layer.py`, `ml/` |
| Board bring-up: camera, display, memory map, gotchas | `board/DevKit-E8/README.md`, `documentation/fingerspell-bringup.md` |
| Measuring letters at the board | `tests/letter_metrics.py`, `tests/board_telemetry.py` |
| Host tests of the pure-C stages | `tests/host/`, `python3 -m unittest tests.test_decoder_host tests.test_input_host tests.test_commit_hold_host` |
| Design notes and status logs | `documentation/landmark-plan.md`, `documentation/temporal-average-plan.md` |
