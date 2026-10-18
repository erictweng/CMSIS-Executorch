# Master prompt: fix hand-sign -> letter accuracy on the landmark pipeline

Branch `landmark/pipeline`, DevKit-E8. Written 2026-10-18 after a signing session at the board.
Paste this whole file as the task prompt. Read the files it names before changing anything.

## Goal

A right-handed signer in the demo room holds each static ASL letter and the board types it
once, with median confidence >= 0.7, within 0.5 s. Today the pipeline (BlazePalm -> rotated
crop -> hand_landmark_lite on the NPU -> 63 keypoint features -> MLP 63-128-24 -> decoder)
gets these wrong on a real hand:

| Signed | Board says | Note |
|---|---|---|
| R | X / K / T / D at 0.25-0.45, or V / U when the hand reads as left | worst letter |
| U | K | |
| H | K | |
| N | S / T | fingers hide the thumb |
| M, O, Q | low confidence | |
| D | wrong / unstable | also feeds the D-vs-Z motion rule |
| I then J motion | the panel appears to freeze while I is pending | pending rule in `src/fs_decoder.c` |

## What was measured (do not re-derive; build on it)

Two 60 s runs of `tests/board_telemetry.py --count 120 --interval 0.5 --handedness` (reads
`g_fs_tele[17]` over J-Link; the UART on this board is dead).

1. **Handedness is unreliable on closed/crossed shapes.** Open shapes: 1.2 % of frames flip
   left/right, 2 % read as left (signer is right-handed). Holding R: a 10 s stretch read as
   LEFT at 0.01-0.08 and the mirrored features scored V 0.88; flips rose to ~7 % of frames; H
   hovered at 0.15-0.48. `fs_landmark_features()` in `src/fs_landmark.c` mirrors x by the
   per-frame `lm->right >= 0.5`, so every flip negates all 21 x features the MLP sees.
2. **Distance causes dropouts.** Hand box ~55 px: ~45 % of samples had presence 0.2-0.4 and
   the board reported "no hand" (gate `FS_LM_PRESENCE 0.5` in `src/fs_landmark.h`). Each
   dropout resets the feature/probability averages and re-arms the repeat rule (text grew 6
   characters while holding R). Box >= 75 px: presence 0.94-0.99, no dropouts.
3. **R is not recognised even with correct handedness.** Confidence spread over R/X/K/T/D at
   0.25-0.45 for 60+ frames. That is a training-data / keypoint-precision problem, not noise;
   averaging and decoder changes cannot fix it. The MLP was trained on dataset photos
   (`model/keypoints_marxulia.npz` + `model/keypoints_marxulia_board.npz`, 8,115 hands
   through the board's own pipeline on the laptop), never on this signer or camera.
4. **Temporal averaging is already in and flashed** (crop geometry 0.5, features 0.5,
   probabilities 0.4, hysteresis 0.45, repeat rule tied to hand loss; spec
   `documentation/temporal-average-plan.md`). A laptop simulation showed a longer window
   (7-10 frames) drops letters during normal-speed spelling; do not lengthen it.
5. **The board restarted once mid-run** (all counters went to zero, no reset pressed).
   Unexplained; watch for it.
6. The int8 landmark model is 0.8 px median / 4.2 px p90 off float; R's crossed fingertips
   are a few px apart, so precision is a secondary suspect after training data.

## Status (2026-10-18, later the same day)

Tasks 1, 2 and 4 are implemented, host-tested (308 decoder checks, 12 Python tests) and
flashed; task 3's firmware (`g_fs_lm_dump`) and tooling (`tests/capture_keypoints.py`,
`ml/keypoint_train.py --upweight`) are in and the capture was dry-run against the board.
Not done: the capture itself (needs the signer: `.venv/bin/python tests/capture_keypoints.py
--letters RUKHNMSTOQDIABLY --seconds 10`, twice at different hand positions), the retrain,
and task 5's table. Nothing is committed. Resume at task 3's "Capture" bullet.

## Tasks, in this order

### 1. Fix the signing hand instead of trusting the model's handedness
- `src/fs_landmark.h`: add `FS_LM_HANDEDNESS` with values AUTO / RIGHT / LEFT; default RIGHT.
  In RIGHT/LEFT the mirror in `fs_landmark_features()` is a constant; AUTO keeps today's
  behaviour. Mirror the same choice in `ml/handlandmark.py` (`features()`), since
  `ml/keypoint_dataset_board.py` and the host tests use it.
- Keep the telemetry word 12 (handedness score) so the model's guess is still visible.
- Host test in `tests/host/test_fs_landmark.c`: a left-scored frame with RIGHT fixed gives
  the same features as a right-scored one.

### 2. Hold through short presence dropouts
- Today: presence < 0.5 on one frame = no hand. Change to: once a hand is tracked, keep it
  for up to `FS_LM_HOLD_FRAMES` (start at 5, matching `FS_DET_HOLD_FRAMES`) frames of
  presence >= 0.25; the feature/crop averages and the decoder's hand state are not reset
  during the hold. A fresh hand still needs >= 0.5. Where: the `hand` decision in `infer()`
  in `src/app_main.cpp` and/or `fs_landmark_decode`; keep the rule pure C so it is
  host-testable (a small `fs_lm_presence_gate` state struct is fine).
- Host test: a presence stream 0.9, 0.9, 0.3, 0.3, 0.9 stays one hand; 0.3 x 6 drops it.
- Board check: at ~55 px hand box the "no hand" samples should fall from ~45 % to near 0.

### 3. Record this signer's keypoints and retrain the MLP
- Firmware: copy the 63 smoothed features the MLP actually sees (`fs` in `letters()`,
  `src/app_main.cpp`) into a `volatile float g_fs_features[63]` each frame, plus a frame
  counter, so J-Link can read them (`tests/fs_tools.py`: `symbol_address`, `jlink`).
- New `tests/capture_keypoints.py --letter R --seconds 10 --out model/keypoints_board_live.npz`:
  polls as fast as J-Link allows (~5-10 Hz is enough), keeps frames where the letter counter
  advanced and presence >= 0.5, labels them with `--letter`, appends to the npz in the same
  layout as `ml/keypoint_dataset.py` (landmarks 21x3, handedness, size, label, source
  "board-live"). Note the features are already wrist-centred/normalised/mirrored; either
  store raw landmarks (preferred: expose `lm.xyz` instead and run `features()` offline) or
  add a `features` array and teach `ml/keypoint_train.py` to accept it. Prefer raw.
- Capture, with the signer at the closer distance (box >= 75 px), 10 s each, two or three
  hand positions: R, U, K, H, N, M, S, T, O, Q, D, I, and a handful of easy letters (A, B,
  L, Y) so the new source is not only hard cases. Target >= 150 frames per letter.
- Retrain: `.venv/bin/python -I ml/keypoint_train.py model/keypoints_marxulia.npz
  model/keypoints_marxulia_board.npz model/keypoints_board_live.npz`. Gates: held-out
  accuracy per source printed by the script; marxulia-board must stay >= 94 %, board-live
  >= 90 %. If the live set is small, upweight it (sample weights) rather than epochs.
- Consider adding engineered features before the MLP if R/U/K/H still confuse: index-tip to
  middle-tip distance, sign of (index tip x - middle tip x) in the hand frame, thumb-tip to
  index-MCP distance. Keep the input count in `get_params()`/`model_params.h` in sync.
- `.venv/bin/python create_ai_layer.py cmsis-executorch.cbuild-mlops.yml` regenerates
  `ai_layer/model_params.h` (MLP weights come from `model.get_params()`).

### 4. D / I / J / Z motion rule
- `accumulate()` in `src/fs_decoder.c` sums raw per-frame fingertip steps above
  `motion_deadband` 0.06 hand sizes; keypoint jitter can push a held D past
  `motion_threshold` 1.0. Change to net displacement from the pending start (or a smoothed
  tip), and commit Z/J as soon as the threshold is crossed instead of waiting for the
  signer to move on. Show the pending state on the panel ("I?" or a dot) so the wait does
  not look like a freeze. Keep `tests/host/test_fs_decoder.c` green (287 checks) and add:
  held D with +-0.08 hand-size jitter for 1.5 s commits D, not Z.

### 5. Measure, then tune
- Per letter, 24 letters, signer at the closer distance: top-1 over a 5 s hold, median
  confidence, commits per 10 s hold (target 1). Write the table into
  `documentation/landmark-plan.md` status log.
- Only after the table: lower `commit_threshold` to 0.50 / `hold_threshold` 0.40 if letters
  sit at 0.5-0.6; try a16w8 or `hand_landmark_full` only if precision is the remaining
  blocker (MRAM free ~0.87 MB; Vela scratch must fit the 1.5 MB temp pool).

## Rules of the board (from the handoff docs and memory; they cost hours to learn)
- Build: CMSIS-Toolbox 2.15.0 from `~/.vcpkg/artifacts/2139c4c6` (commands in
  `documentation/fingerspell-bringup.md`), `cbuild ... --active DevKit-E8`; never two
  targets at once, never `--context` (wipes `tmp/`).
- Flash: J-Link command file (`fs_tools.flash_and_start`); "CPU could not be halted" means
  retry, usually once.
- UART is dead; read state with `tests/board_telemetry.py` / `fs_tools.jlink`. Close the
  VS Code Serial Monitor first.
- The Debug build is -O1; hot C files have `optimize: speed` in the cproject.
- Both NPU models use `quantized_io`; inputs int8, decode int8 in C.
- Host tests need only clang: `python3 -m unittest tests.test_decoder_host tests.test_landmark -v`.
- The loop must stay >= 12 fps (today 15-16 with a hand).

## Do not
- Lengthen the temporal averages or add a majority vote (simulated: drops letters).
- Change the detector, the landmark model export, or the input ranges.
- Commit or push without being asked; today's uncommitted work (averaging, hysteresis,
  handedness telemetry) is in the tree and should be committed first as its own change.
- Ask the signer which letters are weak; measure them (task 5) and let the table say.

## Done when
Task 5's table shows every static letter at top-1 correct with median confidence >= 0.7 and
one commit per hold, D holds as D, J and Z resolve on motion, and the host tests are green.
Report the table, the per-source MLP accuracies and the fps in the final message.
