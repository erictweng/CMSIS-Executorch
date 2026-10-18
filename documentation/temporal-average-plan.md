# Temporal averaging: implementation spec and plan

Branch `landmark/pipeline`. Status: **sections 2, 3 and 6 implemented and host-tested
(2026-10-18); steps 1, 4 and 5 of the plan need the board.** Deviations from the spec as
written: the hand-loss state is `last_hand_ms` (not `hand_seen`); a frame counts as no-hand
when `motion->valid` is false or the probabilities sum to under 0.5, whichever applies; a
held (hysteresis) frame keeps the candidate but does not count toward `commit_frames`;
`prob_alpha` or `hold_threshold` left at 0 in a custom cfg means off, so older configs keep
their behaviour; `g_fs_tele[1]` still carries the raw MLP probability (the decoder's
`live_conf` is the averaged one). Later the same day: the space rule changed from a pause to
the hand leaving the frame for `space_ms` (600 ms), and the section 5 defaults went in
(`commit_threshold` 0.50, `hold_threshold` 0.40, `commit_frames` 2).
Companion to `documentation/landmark-plan.md`; the jitter analysis it follows up is in the
status log there.

## 1. Problem

Two symptoms from the board, same root cause: every frame is decided on its own.

1. **Spam.** The text fills with `b b b b` or `a a a a` while one letter is held. The decoder
   commits once per confident streak, but the confidence dips under 0.60 for a few frames,
   300 ms pass with no confident letter, the `released` flag flips, and the same letter
   commits again. After a 1 s dip a space goes in too, hence the spaces between the b's.
2. **Slow first commit.** When the hand comes up it takes a long time before a letter prints
   (the user sees it as "waiting for 70 %"; the actual threshold is 0.60 for 3 consecutive
   frames). The confidence hovers and flickers between two letters, so the 3-frame streak
   keeps resetting.

Both come from per-frame noise. Where the noise enters the pipeline, in order of size:

| Stage | Source | Today |
|---|---|---|
| Crop geometry | `fs_landmark_crop_frame` uses the raw detection every frame: centre, side and rotation jump, so the landmark model sees a different picture each frame | not smoothed (the display box is, via `fs_track_update`) |
| Landmarks | int8 output, 1.18 px per step; 0.8 px median, 4 px p90 error | feature EMA 0.5 (`fs_landmark_smooth`), built in commit 50560f36, **not yet flashed** |
| MLP output | one frame of features in, a sharp softmax out; close pairs (B/A, D/O, M/N, R/U) flip | none |
| Decoder | hard 0.60 threshold, no hysteresis; same letter re-commits after a 300 ms dip | none |

Averaging makes the stream steady; it does not make the MLP more confident on a shape it
half-recognises. If a letter sits at 0.55 it will still not print. That is handled separately
(section 5) by lowering the thresholds, which averaging makes safe.

## 2. Design

Three averagers and one decoder change, each independently testable on the laptop.

### 2.1 Crop geometry (new)

Average the rotated crop the landmark model sees, the same way the display box is averaged.

- State `fs_lm_crop_t` average plus `valid` flag, next to `g_smooth` in `app_main.cpp`.
- Per frame with a fresh detection: `raw = fs_landmark_crop_frame(det)`, then
  `cx, cy, side += a * (raw - avg)`; the up vector (`-dx, -dy`) is averaged the same way and
  renormalised, `rx, ry` rebuilt from it. First frame copies raw.
- `a` = `FS_LM_CROP_SMOOTH` 0.5 (matches `FS_DET_EMA`). Reset when the hand is lost, in the
  same places `letters_reset()` is called.
- The held path (detector miss, `hold` true) already reuses the last crop; it now reuses the
  averaged one.

Why first: it is the largest avoidable source, two lines of math, and it also steadies the
landmark stage input so the feature EMA has less to do.

### 2.2 Features (exists, tune)

`fs_landmark_smooth` stays. Alpha 0.5 for now; revisit after 2.3 is in, since two averagers
in series add latency (section 4). Likely end state: features 0.6, probabilities 0.4.

### 2.3 Probabilities (new, the "temporal average")

An exponential average of the 24 MLP probabilities, inside the decoder so the host tests
cover it and the display shows the averaged confidence.

- `fs_decoder_cfg_t` gains `float prob_alpha` (default 0.4; 1.0 = off, old behaviour).
- `fs_decoder_t` gains `float avg[FS_CLASSES]; bool avg_valid;`.
- At the top of `fs_decoder_step_motion`: if the frame has no hand (`motion->valid` false, or
  the probabilities sum to under 0.5 when `motion` is NULL) set `avg_valid = false` and use
  the raw zeros, so the space timer and no-hand behaviour are unchanged. Otherwise
  `avg = raw` on the first frame, `avg += alpha * (raw - avg)` after, and the rest of the
  step reads `avg`. Renormalising is unnecessary: an EMA of distributions is a distribution.
- Initialising from the first frame (not from zero) is what keeps the first commit fast: a
  hand that comes up steady at 0.8 commits after the usual 3 frames with no warm-up.
  Averaging only costs time when the letter changes, which is the debounce we want.

Why EMA and not a sliding window: one 24-float state, no ring buffer, same result for this
purpose, and the host tests already drive the decoder with 30 fps synthetic streams.

### 2.4 Decoder hysteresis and repeat rule (new)

Averaging alone still leaves a hard edge at 0.60. Two rules fix the spam directly:

- **Hysteresis.** `hold_threshold` (default 0.45). A letter needs `commit_threshold` to
  start a streak and to commit; once it is the live candidate it stays confident while its
  averaged probability is at or above `hold_threshold`. The `released` flag and the space
  timer are driven by this lower edge, so a dip to 0.5 is no longer a gap.
- **Repeat rule.** The same letter commits again only after the hand was lost (`motion->valid`
  false for `release_ms`) or a different letter committed in between. A confidence dip alone
  never re-arms it. Double letters ("LL") are signed by bouncing the hand, which the
  detector sees as a short loss, so the behaviour the user expects survives.
- `commit_frames` stays 3 but now counts frames of the averaged stream.

The pending D/I rule is unchanged; it reads the same averaged `top` and `confident`.

## 3. Files

| File | Change |
|---|---|
| `src/fs_landmark.h/.c` | `fs_landmark_crop_smooth(fs_lm_crop_t* avg, bool* valid, const fs_lm_crop_t* raw, float alpha)`, `FS_LM_CROP_SMOOTH` |
| `src/app_main.cpp` | crop average state in `infer()`, reset with `letters_reset()`; decoder built with the new cfg; `g_fs_tele[1]` already carries the live confidence (now averaged) |
| `src/fs_api.h` | cfg fields `prob_alpha`, `hold_threshold`; state `avg[]`, `avg_valid`, `hand_seen`; comment block for the new rules |
| `src/fs_decoder.c` | averaging at the top of the step, hysteresis, repeat rule tied to hand loss |
| `tests/host/test_fs_decoder.c` | new tests, see section 6; existing `test_hello_double_letter` adapted to lose the hand between the L's |
| `tests/host/test_fs_landmark.c` | crop smoothing: converges to a constant input, up vector stays unit length, reset restarts from raw |
| `ml/handlandmark.py` | `smooth_crop()` reference so the board extraction (`ml/keypoint_dataset_board.py`) can mirror it if the MLP is retrained later |
| `tests/test_decoder_host.py` | already builds and runs the host test; no change expected |
| `documentation/landmark-plan.md` | status log entry; decision 9 (temporal averaging) |
| `documentation/fingerspell-bringup.md` | Decoder section: the new parameters and what to tune |

Not touched: the models, `create_ai_layer.py`, `model/model.py`, the display, the camera.
No rebuild of the `.pte`; this is a CPU-side change only.

## 4. Latency budget

At the measured 15-16 fps a frame is about 65 ms.

| Step | Frames | Time |
|---|---|---|
| Hand appears, tracker locks | 1 | 65 ms |
| Crop average settles (alpha 0.5, within 12 %) | 3 | 200 ms |
| Feature + probability EMA, first frame copies raw | 0 extra | 0 |
| Commit streak | 3 | 200 ms |
| **First letter on screen** | **~7** | **~450 ms** |
| Change to a new letter (two EMAs in series must cross 0.60, then 3 frames) | 5-7 | 350-450 ms |

A fluent signer produces 2-3 letters per second, so 350-450 ms per letter is at the edge.
If it feels slow after measuring: drop `commit_frames` to 2 (the averaged stream makes
that safe), or raise `prob_alpha` to 0.5. The crop average does not add to letter-change
latency, since the hand box barely moves between letters.

## 5. Confidence (the second note in the paste)

Averaging will not lift a letter that the MLP scores at 0.5. What will:

- The flashed image predates the retrained MLP (94.5 % on board-pipeline landmarks) and
  the feature EMA. Flashing commit 50560f36 is step zero and may remove much of this.
- With averaging in place, lower `commit_threshold` to 0.50 and `hold_threshold` to 0.40.
  A single noisy frame can no longer trigger it, so the lower edge is safe.
- If specific letters still sit low on the real hand, that is MLP training data, not
  smoothing: capture the board's landmarks for those letters and add them to
  `model/keypoints_marxulia_board.npz` (the tooling exists, `ml/keypoint_dataset_board.py`).

## 6. Tests

Host (`tests/host/test_fs_decoder.c`, run through `tests/test_decoder_host.py`):

- A letter held at 0.8 with every 4th frame dipping to 0.3 for 10 s commits **once**.
- A/B flicker (0.55 / 0.45 alternating) commits nothing.
- A steady 0.8 letter commits within 3 frames of its first frame (no warm-up).
- Letter A then B, both steady: B commits within 7 frames of the change.
- A dip to 0.5 for 20 frames does not re-arm the letter; losing the hand for `release_ms`
  does.
- Space still appears 1 s after the hand is lost; no space during a confidence dip.
- `prob_alpha` 1.0 reproduces today's results on the existing tests (regression guard).
- Motion tests (Z, J) pass unchanged with averaging on.

Host (`tests/host/test_fs_landmark.c`): crop smoothing as in section 3.

Board (`tests/test_detector.py` OnBoard, `tests/board_telemetry.py`): the fps gate stays
at 12; averaging adds under 0.1 ms of CPU per frame.

## 7. Plan

1. **Flash what is built** (commit 50560f36: retrained MLP + feature EMA) when the board is
   back on USB. Hold B for 10 s, read `decoder:` lines or telemetry, count commits. This is
   the baseline every later step is measured against.
2. **Decoder: averaging + hysteresis + repeat rule** with host tests. About an hour. Pure C,
   no board needed to prove it.
3. **Crop geometry smoothing** with host tests. About an hour.
4. **Flash and measure** with a signer: commits per 10 s hold (target 1), time to first
   letter (target under 0.5 s), "HELLO" and "ABC" sequences (every letter once, in order),
   Z and J still resolve.
5. **Tune** `prob_alpha`, `commit_frames`, the thresholds from the measurements. If letter
   changes feel laggy, consider a speed-adaptive alpha (One Euro filter: smooth more when
   still, less when moving) as a follow-up, not now.
6. Docs and status log; push to `fingerspell` when the user asks.

Gate between 3 and 4: all host tests green, DevKit image builds, SRAM/MRAM unchanged.

## 8. Decisions and alternatives rejected

- **Average inside the decoder, not in `app_main.cpp`.** Keeps the behaviour under the host
  tests and the displayed confidence consistent with what commits.
- **EMA over sliding window.** Same effect, no buffer, tunable with one number.
- **No majority vote over the last N letters.** It is a cruder average of the same signal and
  throws away the confidence; hysteresis on the averaged probability does the job.
- **No temporal classifier (GRU over landmark sequences).** Would need recorded sequences we
  do not have; the averaging plus hysteresis is expected to remove most of the spam. Keep as
  the fallback if measurements in step 4 say otherwise.
- **Repeat rule tied to hand loss, not to a confidence gap.** The confidence gap is exactly
  the signal the noise fakes.
