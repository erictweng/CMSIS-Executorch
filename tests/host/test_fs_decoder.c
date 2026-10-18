// Host tests for src/fs_decoder.c against the fs_api.h decoder contract. Feeds synthetic
// probability streams at 30 fps; recorded webcam.py streams can replace them later.
//
// The first block of tests predates the temporal averaging and runs with its parameters
// (prob_alpha 1.0 = averaging off, commit 0.60 for 3 frames, hold 0.45, space 1000 ms): the
// regression guard for the per-frame rules. The "avg_" tests run the defaults. The hand-loss
// repeat rule and the hand-out-of-view space apply to both.
//
//   clang -std=c99 -Wall -Wextra -Werror -pedantic -I src src/fs_decoder.c \
//         tests/host/test_fs_decoder.c -o /tmp/test_fs_decoder && /tmp/test_fs_decoder

#include <stdio.h>
#include <string.h>

#include "fs_api.h"

#define FRAME_MS 33u

static int g_checks, g_failures;

#define CHECK(cond)                                                      \
  do {                                                                   \
    g_checks++;                                                          \
    if (!(cond)) {                                                       \
      g_failures++;                                                      \
      printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);             \
    }                                                                    \
  } while (0)

#define CHECK_STR(d, want)                                                              \
  do {                                                                                  \
    g_checks++;                                                                         \
    if (strcmp((d)->text, (want)) != 0 || (d)->len != (int)strlen(want)) {              \
      g_failures++;                                                                     \
      printf("FAIL %s:%d: text \"%s\" (len %d), want \"%s\"\n", __FILE__, __LINE__,     \
             (d)->text, (d)->len, (want));                                              \
    }                                                                                   \
  } while (0)

typedef struct {
  int letters, spaces, none;
  fs_event_t last;
} events_t;

// The defaults, or the pre-averaging parameters when g_legacy is set.
static bool g_legacy = true;
static void init(fs_decoder_t* d) {
  fs_decoder_init(d, NULL);
  if (!g_legacy) return;
  fs_decoder_cfg_t cfg = d->cfg;
  cfg.prob_alpha = 1.0f;
  cfg.commit_threshold = 0.60f;
  cfg.hold_threshold = 0.45f;
  cfg.commit_frames = 3;
  cfg.space_ms = 1000;
  fs_decoder_init(d, &cfg);
}

// Probabilities with `conf` on `letter` and the rest spread evenly over the other classes.
static void probs_for(char letter, float conf, float probs[FS_CLASSES]) {
  const char* p = strchr(FS_LABELS, letter);
  const int idx = letter ? (int)(p - FS_LABELS) : 0;
  const float rest = (1.0f - conf) / (FS_CLASSES - 1);
  for (int i = 0; i < FS_CLASSES; ++i) probs[i] = (i == idx) ? conf : rest;
}

// Feed `frames` frames of `letter` at `conf`, advancing *t by FRAME_MS each.
static events_t feed(fs_decoder_t* d, char letter, float conf, int frames, uint32_t* t) {
  events_t ev = {0, 0, 0, FS_EV_NONE};
  float probs[FS_CLASSES];
  probs_for(letter, conf, probs);
  for (int i = 0; i < frames; ++i) {
    ev.last = fs_decoder_step(d, probs, *t);
    *t += FRAME_MS;
    if (ev.last == FS_EV_LETTER) ev.letters++;
    else if (ev.last == FS_EV_SPACE) ev.spaces++;
    else ev.none++;
  }
  return ev;
}

// Low-confidence frames (the hand moving, still in view) covering at least `ms`.
static events_t gap(fs_decoder_t* d, uint32_t ms, uint32_t* t) {
  return feed(d, 'A', 0.20f, (int)((ms + FRAME_MS - 1) / FRAME_MS), t);
}

// Frames without a hand (all-zero probabilities, as the board feeds them) covering `ms`.
static events_t lost(fs_decoder_t* d, uint32_t ms, uint32_t* t) {
  events_t ev = {0, 0, 0, FS_EV_NONE};
  const float zeros[FS_CLASSES] = {0};
  for (int i = 0; i < (int)((ms + FRAME_MS - 1) / FRAME_MS); ++i) {
    ev.last = fs_decoder_step(d, zeros, *t);
    *t += FRAME_MS;
    if (ev.last == FS_EV_LETTER) ev.letters++;
    else if (ev.last == FS_EV_SPACE) ev.spaces++;
    else ev.none++;
  }
  return ev;
}

// Commit one letter the normal way: hold it for exactly commit_frames.
static events_t sign(fs_decoder_t* d, char letter, uint32_t* t) {
  return feed(d, letter, 0.90f, d->cfg.commit_frames, t);
}

static void test_defaults(void) {
  fs_decoder_t d;
  fs_decoder_init(&d, NULL);
  CHECK(d.cfg.commit_threshold == 0.50f);
  CHECK(d.cfg.commit_frames == 2);
  CHECK(d.cfg.space_ms == 600);
  CHECK(d.cfg.release_ms == 300);
  CHECK(d.cfg.hold_threshold == 0.40f);
  CHECK(d.cfg.prob_alpha == 0.4f);
  CHECK(d.len == 0 && d.text[0] == '\0' && d.live == 0);
}

static void test_commit_rule_and_hold(void) {
  fs_decoder_t d;
  init(&d);
  uint32_t t = 0;
  events_t ev = feed(&d, 'A', 0.90f, 2, &t);
  CHECK(ev.letters == 0);
  CHECK(d.live == 'A' && d.live_conf == 0.90f);
  ev = feed(&d, 'A', 0.90f, 1, &t);
  CHECK(ev.last == FS_EV_LETTER);
  CHECK_STR(&d, "A");
  ev = feed(&d, 'A', 0.90f, 60, &t);  // held for two seconds: commits once only
  CHECK(ev.letters == 0 && ev.spaces == 0);
  CHECK_STR(&d, "A");
}

static void test_below_threshold_never_commits(void) {
  fs_decoder_t d;
  init(&d);
  uint32_t t = 0;
  events_t ev = feed(&d, 'B', 0.59f, 30, &t);
  CHECK(ev.letters == 0 && ev.spaces == 0);
  CHECK(d.live == 0 && d.live_conf == 0.59f);
  CHECK_STR(&d, "");
  ev = feed(&d, 'B', 0.60f, 3, &t);  // the threshold itself counts
  CHECK(ev.last == FS_EV_LETTER);
  CHECK_STR(&d, "B");
}

static void test_frames_must_be_consecutive(void) {
  fs_decoder_t d;
  init(&d);
  uint32_t t = 0;
  feed(&d, 'C', 0.90f, 2, &t);
  feed(&d, 'C', 0.30f, 1, &t);  // one dip resets the streak
  events_t ev = feed(&d, 'C', 0.90f, 2, &t);
  CHECK(ev.letters == 0);
  CHECK_STR(&d, "");
  ev = feed(&d, 'C', 0.90f, 1, &t);
  CHECK(ev.last == FS_EV_LETTER);
  CHECK_STR(&d, "C");
}

static void test_candidate_change_resets_streak(void) {
  fs_decoder_t d;
  init(&d);
  uint32_t t = 0;
  feed(&d, 'A', 0.90f, 2, &t);
  events_t ev = feed(&d, 'B', 0.90f, 2, &t);
  CHECK(ev.letters == 0);
  ev = feed(&d, 'B', 0.90f, 1, &t);
  CHECK(ev.last == FS_EV_LETTER);
  CHECK_STR(&d, "B");
}

static void test_hello_double_letter(void) {
  fs_decoder_t d;
  init(&d);
  uint32_t t = 0;
  CHECK(sign(&d, 'H', &t).last == FS_EV_LETTER);
  gap(&d, 200, &t);  // moving between letters: no space, no effect on the next letter
  CHECK(sign(&d, 'E', &t).last == FS_EV_LETTER);
  gap(&d, 200, &t);
  CHECK(sign(&d, 'L', &t).last == FS_EV_LETTER);
  CHECK_STR(&d, "HEL");

  feed(&d, 'L', 0.90f, 10, &t);       // still held: no repeat
  feed(&d, 'L', 0.30f, 1, &t);        // a one-frame flicker is not a release
  events_t ev = feed(&d, 'L', 0.90f, 6, &t);
  CHECK(ev.letters == 0);
  CHECK_STR(&d, "HEL");
  gap(&d, 350, &t);  // a confidence gap, hand in view: still no release
  CHECK(sign(&d, 'L', &t).letters == 0);
  CHECK_STR(&d, "HEL");

  lost(&d, 350, &t);  // the hand drops out for >= release_ms: the second L may commit
  CHECK(sign(&d, 'L', &t).last == FS_EV_LETTER);
  gap(&d, 200, &t);
  CHECK(sign(&d, 'O', &t).last == FS_EV_LETTER);
  CHECK_STR(&d, "HELLO");
}

static void test_hi_mom_space(void) {
  fs_decoder_t d;
  init(&d);
  uint32_t t = 0;
  sign(&d, 'H', &t);
  gap(&d, 200, &t);
  sign(&d, 'I', &t);
  CHECK_STR(&d, "HI");

  events_t ev = gap(&d, 3000, &t);  // a pause with the hand in view is never a space
  CHECK(ev.spaces == 0);
  CHECK_STR(&d, "HI");
  ev = lost(&d, 990, &t);  // just under space_ms without a hand
  CHECK(ev.spaces == 0);
  ev = lost(&d, 100, &t);
  CHECK(ev.spaces == 1);
  CHECK_STR(&d, "HI ");
  ev = lost(&d, 3000, &t);  // one space per absence, however long
  CHECK(ev.spaces == 0);
  CHECK_STR(&d, "HI ");

  sign(&d, 'M', &t);
  gap(&d, 200, &t);
  sign(&d, 'O', &t);
  gap(&d, 200, &t);
  sign(&d, 'M', &t);
  CHECK_STR(&d, "HI MOM");
}

static void test_no_space_before_any_text(void) {
  fs_decoder_t d;
  init(&d);
  uint32_t t = 5000;  // the clock need not start at zero
  events_t ev = lost(&d, 4000, &t);
  CHECK(ev.spaces == 0 && ev.letters == 0);
  CHECK_STR(&d, "");
}

static void test_backspace_and_clear(void) {
  fs_decoder_t d;
  init(&d);
  uint32_t t = 0;
  sign(&d, 'H', &t);
  gap(&d, 200, &t);
  sign(&d, 'I', &t);
  lost(&d, 1100, &t);
  CHECK_STR(&d, "HI ");
  CHECK(fs_decoder_backspace(&d) == FS_EV_BACKSPACE);
  CHECK_STR(&d, "HI");
  CHECK(fs_decoder_backspace(&d) == FS_EV_BACKSPACE);
  CHECK(fs_decoder_backspace(&d) == FS_EV_BACKSPACE);
  CHECK_STR(&d, "");
  CHECK(fs_decoder_backspace(&d) == FS_EV_NONE);

  sign(&d, 'A', &t);
  CHECK(fs_decoder_clear(&d) == FS_EV_CLEAR);
  CHECK_STR(&d, "");
  gap(&d, 400, &t);
  CHECK(sign(&d, 'B', &t).last == FS_EV_LETTER);  // decoding continues after a clear
  CHECK_STR(&d, "B");
}

static void test_backspace_rearms_the_held_letter(void) {
  fs_decoder_t d;
  init(&d);
  uint32_t t = 0;
  CHECK(sign(&d, 'A', &t).last == FS_EV_LETTER);
  CHECK(feed(&d, 'A', 0.90f, 10, &t).letters == 0);  // still held
  CHECK(fs_decoder_backspace(&d) == FS_EV_BACKSPACE);
  CHECK_STR(&d, "");
  events_t ev = feed(&d, 'A', 0.90f, d.cfg.commit_frames, &t);  // no hand bounce needed
  CHECK(ev.letters == 1);
  CHECK_STR(&d, "A");
  CHECK(feed(&d, 'A', 0.90f, 30, &t).letters == 0);  // and once only
}

static void test_backspace_of_a_trailing_space_is_not_undone_until_the_hand_returns(void) {
  fs_decoder_t d;
  init(&d);
  uint32_t t = 0;
  sign(&d, 'H', &t);
  lost(&d, 1100, &t);
  CHECK_STR(&d, "H ");
  CHECK(fs_decoder_backspace(&d) == FS_EV_BACKSPACE);
  CHECK_STR(&d, "H");
  CHECK(lost(&d, 3000, &t).spaces == 0);  // the hand stays away: the space does not come back
  CHECK_STR(&d, "H");
  sign(&d, 'I', &t);
  CHECK_STR(&d, "HI");
  CHECK(lost(&d, 1100, &t).spaces == 1);  // a later absence spaces again
  CHECK_STR(&d, "HI ");
}

static void test_backspace_of_a_letter_blocks_the_space_too(void) {
  fs_decoder_t d;
  init(&d);
  uint32_t t = 0;
  sign(&d, 'H', &t);
  gap(&d, 200, &t);
  sign(&d, 'I', &t);
  lost(&d, 200, &t);  // not yet a space
  CHECK(fs_decoder_backspace(&d) == FS_EV_BACKSPACE);
  CHECK_STR(&d, "H");
  CHECK(lost(&d, 2000, &t).spaces == 0);
  CHECK_STR(&d, "H");
  sign(&d, 'I', &t);
  CHECK(lost(&d, 1100, &t).spaces == 1);
  CHECK_STR(&d, "HI ");
}

static void test_backspace_on_empty_leaves_the_state_alone(void) {
  fs_decoder_t d;
  init(&d);
  uint32_t t = 0;
  sign(&d, 'A', &t);
  lost(&d, 350, &t);
  sign(&d, 'A', &t);
  CHECK_STR(&d, "AA");
  CHECK(fs_decoder_backspace(&d) == FS_EV_BACKSPACE);
  CHECK(fs_decoder_backspace(&d) == FS_EV_BACKSPACE);
  CHECK(fs_decoder_backspace(&d) == FS_EV_NONE);
  CHECK_STR(&d, "");
  CHECK(feed(&d, 'A', 0.90f, 10, &t).letters == 1);  // re-armed by the real backspaces
  CHECK_STR(&d, "A");
}

static void test_full_buffer_drops_oldest(void) {
  fs_decoder_t d;
  init(&d);
  uint32_t t = 0;
  const int n = FS_TEXT_MAX + 20;
  char want[FS_TEXT_MAX + 32];
  for (int i = 0; i < n; ++i) {
    const char letter = (i % 2) ? 'B' : 'A';  // alternating: no release needed
    want[i] = letter;
    CHECK(sign(&d, letter, &t).last == FS_EV_LETTER);
    gap(&d, 100, &t);
  }
  want[n] = '\0';
  CHECK(d.len == FS_TEXT_MAX - 1);
  CHECK(d.text[d.len] == '\0' && (int)strlen(d.text) == d.len);
  CHECK(strcmp(d.text, want + (n - (FS_TEXT_MAX - 1))) == 0);
}

static void test_custom_config(void) {
  fs_decoder_cfg_t cfg = {.commit_threshold = 0.80f, .commit_frames = 1, .space_ms = 500, .release_ms = 100};
  fs_decoder_t d;
  fs_decoder_init(&d, &cfg);
  uint32_t t = 0;
  CHECK(feed(&d, 'K', 0.79f, 1, &t).last == FS_EV_NONE);
  CHECK(feed(&d, 'K', 0.81f, 1, &t).last == FS_EV_LETTER);
  CHECK(lost(&d, 500, &t).spaces == 1);
  CHECK_STR(&d, "K ");
  CHECK(d.cfg.prob_alpha == 1.0f);            // unset: no averaging
  CHECK(d.cfg.hold_threshold == 0.80f);       // unset: no hysteresis

  cfg.commit_frames = 0;  // clamped to 1
  fs_decoder_init(&d, &cfg);
  CHECK(d.cfg.commit_frames == 1);
  cfg.hold_threshold = 0.90f;  // above the commit threshold: no hysteresis
  cfg.prob_alpha = 1.5f;       // out of range: no averaging
  fs_decoder_init(&d, &cfg);
  CHECK(d.cfg.hold_threshold == 0.80f && d.cfg.prob_alpha == 1.0f);
}

static void test_clock_wraparound(void) {
  fs_decoder_t d;
  init(&d);
  uint32_t t = 0xFFFFFFFFu - 400u;
  sign(&d, 'A', &t);
  CHECK(lost(&d, 1100, &t).spaces == 1);  // the gap spans the 32-bit wrap
  CHECK_STR(&d, "A ");
  CHECK(sign(&d, 'A', &t).last == FS_EV_LETTER);  // and so does the release
  CHECK_STR(&d, "A A");
}

static void test_argmax_tie_takes_lowest_index(void) {
  fs_decoder_t d;
  init(&d);
  float probs[FS_CLASSES] = {0};
  probs[0] = 0.7f;  // A
  probs[1] = 0.7f;  // B
  fs_decoder_step(&d, probs, 0);
  CHECK(d.live == 'A');
}


// ---- motion letters: D / Z and I / J through fs_decoder_step_motion ---------------------

// Feed `frames` frames of `letter` with the fingertip moving `step` hand sizes per frame
// (a straight path), advancing *t; hand size 100 px, hand present.
static events_t feed_motion(fs_decoder_t* d, char letter, float conf, int frames, float step, uint32_t* t,
                            float* tip) {
  events_t ev = {0, 0, 0, FS_EV_NONE};
  float probs[FS_CLASSES];
  probs_for(letter, conf, probs);
  for (int i = 0; i < frames; ++i) {
    tip[0] += step * 100.0f;
    fs_motion_t m = {true, {tip[0], tip[1]}, {tip[0], tip[1]}, 100.0f};
    ev.last = fs_decoder_step_motion(d, probs, &m, *t);
    *t += FRAME_MS;
    if (ev.last == FS_EV_LETTER) ev.letters++;
    else if (ev.last == FS_EV_SPACE) ev.spaces++;
    else ev.none++;
  }
  return ev;
}

// Feed `frames` frames of `letter` with the fingertip jittering: +-`amp` hand sizes about
// `tip`, alternating direction each frame (what keypoint noise on a held hand looks like).
static events_t feed_jitter(fs_decoder_t* d, char letter, float conf, int frames, float amp, uint32_t* t,
                            const float* tip) {
  events_t ev = {0, 0, 0, FS_EV_NONE};
  float probs[FS_CLASSES];
  probs_for(letter, conf, probs);
  for (int i = 0; i < frames; ++i) {
    const float sgn = (i & 1) ? -1.0f : 1.0f;
    const float x = tip[0] + sgn * amp * 100.0f, y = tip[1] - sgn * amp * 100.0f * 0.5f;
    fs_motion_t m = {true, {x, y}, {x, y}, 100.0f};
    ev.last = fs_decoder_step_motion(d, probs, &m, *t);
    *t += FRAME_MS;
    if (ev.last == FS_EV_LETTER) ev.letters++;
    else if (ev.last == FS_EV_SPACE) ev.spaces++;
    else ev.none++;
  }
  return ev;
}

static void test_motion_d_held_still_is_d(void) {
  fs_decoder_t d;
  init(&d);
  uint32_t t = 0;
  float tip[2] = {160, 160};
  events_t ev = feed_motion(&d, 'D', 0.90f, 3, 0.0f, &t, tip);
  CHECK(ev.last == FS_EV_PENDING);  // not committed yet
  CHECK(d.pending == 'D' && d.live == 'D');
  CHECK_STR(&d, "");
  ev = feed_jitter(&d, 'D', 0.90f, 10, 0.02f, &t, tip);  // jitter below the dead band
  CHECK(ev.letters == 0 && d.pending_path == 0.0f);
  ev = feed_motion(&d, 'A', 0.20f, 3, 0.0f, &t, tip);  // hand relaxes: resolves on the 3rd frame
  CHECK(ev.letters == 1);
  CHECK_STR(&d, "D");
  CHECK(d.pending == 0);
}

static void test_motion_jittery_d_held_is_still_d(void) {
  // Keypoint jitter of +-0.08 hand sizes (above the dead band) for 1.5 s: a path sum would
  // read 0.08 x 50 = 4 hand sizes of "stroke"; the displacement stays near 0.09 and the
  // hold times out as D.
  fs_decoder_t d;
  init(&d);
  uint32_t t = 0;
  float tip[2] = {160, 160};
  feed_motion(&d, 'D', 0.90f, 3, 0.0f, &t, tip);
  events_t ev = feed_jitter(&d, 'D', 0.90f, 50, 0.08f, &t, tip);  // past the 1.5 s timeout at 30 fps
  CHECK(d.pending_path < 0.2f);
  CHECK(ev.letters == 1);
  CHECK_STR(&d, "D");
  CHECK(d.pending == 0);
}

static void test_motion_d_with_a_stroke_is_z(void) {
  fs_decoder_t d;
  init(&d);
  uint32_t t = 0;
  float tip[2] = {160, 160};
  feed_motion(&d, 'D', 0.90f, 3, 0.0f, &t, tip);
  events_t ev = feed_motion(&d, 'D', 0.90f, 6, 0.2f, &t, tip);  // 1.2 hand sizes of travel
  CHECK(ev.letters == 1);  // Z commits as soon as the displacement reaches the threshold
  CHECK(d.pending == 0 && d.live == 'D');
  CHECK_STR(&d, "Z");
  ev = sign(&d, 'A', &t);  // moves on to an A: commits normally
  CHECK(ev.last == FS_EV_LETTER);
  CHECK_STR(&d, "ZA");
}

static void test_motion_still_shape_after_the_stroke_does_not_commit_again(void) {
  // After the Z the hand keeps the D shape for the rest of the stroke and a while after:
  // no second pending D, no "ZD". Losing the hand re-arms it.
  fs_decoder_t d;
  init(&d);
  uint32_t t = 0;
  float tip[2] = {160, 160};
  feed_motion(&d, 'D', 0.90f, 3, 0.0f, &t, tip);
  feed_motion(&d, 'D', 0.90f, 6, 0.2f, &t, tip);
  CHECK_STR(&d, "Z");
  events_t ev = feed_motion(&d, 'D', 0.90f, 30, 0.1f, &t, tip);  // the rest of the Z, 1 s
  CHECK(ev.letters == 0 && d.pending == 0);
  ev = feed_motion(&d, 'D', 0.90f, 30, 0.0f, &t, tip);  // then held still
  CHECK(ev.letters == 0 && d.pending == 0);
  CHECK_STR(&d, "Z");
  float probs[FS_CLASSES];
  probs_for('A', 0.20f, probs);
  fs_motion_t gone = {false, {0, 0}, {0, 0}, 0};
  for (int i = 0; i < 12; ++i) {  // the hand leaves for 400 ms
    fs_decoder_step_motion(&d, probs, &gone, t);
    t += FRAME_MS;
  }
  ev = feed_motion(&d, 'D', 0.90f, 3, 0.0f, &t, tip);  // D again: pends, then commits as D
  CHECK(ev.last == FS_EV_PENDING && d.pending == 'D');
  feed_motion(&d, 'A', 0.20f, 3, 0.0f, &t, tip);
  CHECK_STR(&d, "ZD");
}

static void test_motion_i_with_a_hook_is_j(void) {
  fs_decoder_t d;
  init(&d);
  uint32_t t = 0;
  float tip[2] = {100, 100};
  feed_motion(&d, 'I', 0.90f, 3, 0.0f, &t, tip);
  events_t ev = feed_motion(&d, 'I', 0.90f, 8, 0.15f, &t, tip);  // J commits mid-stroke
  CHECK(ev.letters == 1);
  CHECK_STR(&d, "J");
  // The hand disappears: nothing more to resolve.
  float probs[FS_CLASSES];
  probs_for('A', 0.20f, probs);
  fs_motion_t gone = {false, {0, 0}, {0, 0}, 0};
  CHECK(fs_decoder_step_motion(&d, probs, &gone, t) == FS_EV_NONE);
  CHECK_STR(&d, "J");
}

static void test_motion_pending_times_out_as_the_still_letter(void) {
  fs_decoder_t d;
  init(&d);
  uint32_t t = 0;
  float tip[2] = {160, 160};
  feed_motion(&d, 'D', 0.90f, 3, 0.0f, &t, tip);
  events_t ev = feed_motion(&d, 'D', 0.90f, 60, 0.0f, &t, tip);  // held 2 s
  CHECK(ev.letters == 1);
  CHECK_STR(&d, "D");
  CHECK(d.pending == 0);
}

static void test_motion_brief_dropout_does_not_end_the_hold(void) {
  fs_decoder_t d;
  init(&d);
  uint32_t t = 0;
  float tip[2] = {160, 160};
  feed_motion(&d, 'D', 0.90f, 3, 0.0f, &t, tip);
  feed_motion(&d, 'D', 0.90f, 3, 0.2f, &t, tip);  // 0.6 hand sizes so far
  events_t ev = feed_motion(&d, 'U', 0.70f, 2, 0.2f, &t, tip);  // two frames of another shape mid-stroke
  CHECK(ev.letters == 0 && d.pending == 'D');
  ev = feed_motion(&d, 'D', 0.90f, 3, 0.2f, &t, tip);  // the stroke continues: Z
  CHECK(ev.letters == 1);
  CHECK_STR(&d, "Z");
  feed_motion(&d, 'A', 0.20f, 3, 0.0f, &t, tip);
  CHECK_STR(&d, "Z");
}

static void test_clear_drops_a_pending_d(void) {
  fs_decoder_t d;
  init(&d);
  uint32_t t = 0;
  float tip[2] = {160, 160};
  feed_motion(&d, 'D', 0.90f, 3, 0.0f, &t, tip);
  CHECK(d.pending == 'D');
  CHECK(fs_decoder_clear(&d) == FS_EV_CLEAR);
  CHECK(d.pending == 0 && d.pending_path == 0.0f);
  events_t ev = feed_motion(&d, 'A', 0.20f, 5, 0.0f, &t, tip);  // nothing left to resolve
  CHECK(ev.letters == 0);
  CHECK_STR(&d, "");
}

static void test_clear_then_the_held_letter_commits_again(void) {
  fs_decoder_t d;
  init(&d);
  uint32_t t = 0;
  sign(&d, 'B', &t);
  feed(&d, 'B', 0.90f, 5, &t);
  CHECK(fs_decoder_clear(&d) == FS_EV_CLEAR);
  CHECK_STR(&d, "");
  CHECK(feed(&d, 'B', 0.90f, d.cfg.commit_frames, &t).letters == 1);
  CHECK_STR(&d, "B");
  // A D that already resolved pends again after a clear.
  init(&d);
  t = 0;
  float tip[2] = {160, 160};
  feed_motion(&d, 'D', 0.90f, 3, 0.0f, &t, tip);
  feed_motion(&d, 'A', 0.20f, 3, 0.0f, &t, tip);  // the hand relaxes: D
  CHECK_STR(&d, "D");
  CHECK(fs_decoder_clear(&d) == FS_EV_CLEAR);
  events_t ev = feed_motion(&d, 'D', 0.90f, 3, 0.0f, &t, tip);
  CHECK(ev.last == FS_EV_PENDING && d.pending == 'D');
  feed_motion(&d, 'A', 0.20f, 3, 0.0f, &t, tip);
  CHECK_STR(&d, "D");
}

static void test_motion_other_letters_commit_as_before(void) {
  fs_decoder_t d;
  init(&d);
  uint32_t t = 0;
  float tip[2] = {160, 160};
  events_t ev = feed_motion(&d, 'B', 0.90f, 3, 0.3f, &t, tip);  // a moving B is still a B
  CHECK(ev.last == FS_EV_LETTER);
  CHECK_STR(&d, "B");
  init(&d);
  t = 0;
  ev = sign(&d, 'D', &t);  // no motion input: D commits like any letter
  CHECK(ev.last == FS_EV_LETTER);
  CHECK_STR(&d, "D");
}



// ---- temporal averaging (prob_alpha 0.4), hysteresis, the hand-loss repeat rule -----------

static void test_avg_held_letter_with_dips_commits_once(void) {
  fs_decoder_t d;
  init(&d);
  uint32_t t = 0;
  events_t total = {0, 0, 0, FS_EV_NONE};
  for (int i = 0; i < 300; ++i) {  // 10 s: every 4th frame the raw confidence dips to 0.3
    events_t ev = feed(&d, 'B', (i % 4 == 3) ? 0.30f : 0.80f, 1, &t);
    total.letters += ev.letters;
    total.spaces += ev.spaces;
  }
  CHECK(total.letters == 1 && total.spaces == 0);
  CHECK_STR(&d, "B");
  CHECK(d.live == 'B' && d.live_conf > 0.40f);
}

static void test_avg_flicker_commits_nothing(void) {
  fs_decoder_t d;
  init(&d);
  uint32_t t = 0;
  float ab[FS_CLASSES] = {0}, ba[FS_CLASSES] = {0};
  ab[0] = 0.48f; ab[1] = 0.40f;  // A over B, both under the commit threshold
  ba[0] = 0.40f; ba[1] = 0.48f;  // B over A
  int letters = 0;
  for (int i = 0; i < 90; ++i) {
    if (fs_decoder_step(&d, (i % 2) ? ba : ab, t) == FS_EV_LETTER) letters++;
    t += FRAME_MS;
  }
  CHECK(letters == 0 && d.live == 0);
  CHECK_STR(&d, "");
}

static void test_avg_steady_letter_commits_in_two_frames(void) {
  fs_decoder_t d;
  init(&d);
  uint32_t t = 0;
  events_t ev = feed(&d, 'C', 0.80f, 1, &t);
  CHECK(ev.letters == 0 && d.live == 'C' && d.live_conf == 0.80f);  // first frame copies raw
  ev = feed(&d, 'C', 0.80f, 1, &t);
  CHECK(ev.last == FS_EV_LETTER);
  CHECK_STR(&d, "C");
}

static void test_avg_live_conf_is_the_average(void) {
  fs_decoder_t d;
  init(&d);
  uint32_t t = 0;
  feed(&d, 'A', 0.90f, 3, &t);
  feed(&d, 'A', 0.50f, 1, &t);  // 0.9 + 0.4 x (0.5 - 0.9)
  CHECK(d.live == 'A' && d.live_conf > 0.73f && d.live_conf < 0.75f);
}

static void test_avg_letter_change_within_seven_frames(void) {
  fs_decoder_t d;
  init(&d);
  uint32_t t = 0;
  feed(&d, 'A', 0.90f, 10, &t);
  CHECK_STR(&d, "A");
  events_t ev = feed(&d, 'B', 0.90f, 7, &t);
  CHECK(ev.letters == 1);
  CHECK_STR(&d, "AB");
  ev = feed(&d, 'B', 0.90f, 30, &t);
  CHECK(ev.letters == 0);
}

static void test_avg_hysteresis_holds_the_candidate(void) {
  fs_decoder_t d;
  init(&d);
  uint32_t t = 0;
  feed(&d, 'C', 0.90f, 3, &t);
  feed(&d, 'C', 0.45f, 20, &t);  // settles at 0.45: above hold, below commit
  CHECK(d.live == 'C' && d.live_conf < 0.50f && d.live_conf >= 0.40f);
  feed(&d, 'C', 0.35f, 20, &t);  // settles below hold
  CHECK(d.live == 0);
  // A letter that only touches the threshold once does not commit on held frames alone.
  init(&d);
  t = 0;
  feed(&d, 'K', 0.52f, 1, &t);
  events_t ev = feed(&d, 'K', 0.40f, 30, &t);
  CHECK(ev.letters == 0 && d.live == 'K');
  CHECK_STR(&d, "");
}

static void test_avg_dip_does_not_rearm_but_hand_loss_does(void) {
  fs_decoder_t d;
  init(&d);
  uint32_t t = 0;
  CHECK(sign(&d, 'B', &t).last == FS_EV_LETTER);
  events_t ev = feed(&d, 'B', 0.50f, 20, &t);  // a dip to 0.5 (660 ms)
  CHECK(ev.letters == 0 && ev.spaces == 0 && d.live == 'B');
  ev = feed(&d, 'B', 0.90f, 30, &t);
  CHECK(ev.letters == 0);
  ev = feed(&d, 'B', 0.20f, 20, &t);  // a deeper dip, hand still in view: no release either
  ev = feed(&d, 'B', 0.90f, 30, &t);
  CHECK(ev.letters == 0);
  CHECK_STR(&d, "B");
  lost(&d, 350, &t);  // the hand bounces out for release_ms
  ev = feed(&d, 'B', 0.90f, 3, &t);
  CHECK(ev.letters == 1);
  CHECK_STR(&d, "BB");
  lost(&d, 200, &t);  // too short a bounce
  ev = feed(&d, 'B', 0.90f, 30, &t);
  CHECK(ev.letters == 0);
  CHECK_STR(&d, "BB");
}

static void test_avg_space_after_hand_lost_not_during_dip(void) {
  fs_decoder_t d;
  init(&d);
  uint32_t t = 0;
  sign(&d, 'H', &t);
  events_t ev = feed(&d, 'H', 0.45f, 40, &t);  // 1.3 s dip, held by hysteresis
  CHECK(ev.spaces == 0);
  ev = gap(&d, 3000, &t);  // 3 s with the hand in view and no letter: still no space
  CHECK(ev.spaces == 0);
  ev = lost(&d, 500, &t);  // out of view, under space_ms (long enough to repeat a letter)
  CHECK(ev.spaces == 0);
  ev = lost(&d, 200, &t);
  CHECK(ev.spaces == 1);
  CHECK_STR(&d, "H ");
  CHECK(!d.avg_valid);
  sign(&d, 'I', &t);  // the average restarts from the first frame with a hand
  CHECK_STR(&d, "H I");
}

static void test_avg_hello(void) {
  fs_decoder_t d;
  init(&d);
  uint32_t t = 0;
  const char* word = "HELLO";
  for (const char* c = word; *c; ++c) {
    if (c > word) lost(&d, 350, &t);  // the hand moves between letters
    events_t ev = feed(&d, *c, 0.85f, 8, &t);
    CHECK(ev.letters == 1);
  }
  CHECK_STR(&d, "HELLO");
}

static void test_avg_motion_letters(void) {
  fs_decoder_t d;
  init(&d);
  uint32_t t = 0;
  float tip[2] = {160, 160};
  feed_motion(&d, 'D', 0.90f, 3, 0.0f, &t, tip);
  CHECK(d.pending == 'D');
  events_t ev = feed_motion(&d, 'D', 0.90f, 6, 0.2f, &t, tip);  // the Z stroke: commits on the way
  CHECK(ev.letters == 1);
  CHECK_STR(&d, "Z");
  ev = feed(&d, 'A', 0.90f, 7, &t);  // moves on to an A
  CHECK(ev.letters == 1);
  CHECK_STR(&d, "ZA");

  init(&d);
  t = 0;
  tip[0] = tip[1] = 100;
  feed_motion(&d, 'I', 0.90f, 3, 0.0f, &t, tip);
  ev = feed_motion(&d, 'I', 0.90f, 8, 0.15f, &t, tip);
  CHECK(ev.letters == 1);
  CHECK_STR(&d, "J");
  float probs[FS_CLASSES];
  probs_for('A', 0.20f, probs);
  fs_motion_t gone = {false, {0, 0}, {0, 0}, 0};
  CHECK(fs_decoder_step_motion(&d, probs, &gone, t) == FS_EV_NONE);  // the hand disappears
  CHECK_STR(&d, "J");

  init(&d);  // jitter on a held D with averaging on: still D
  t = 0;
  tip[0] = tip[1] = 160;
  feed_motion(&d, 'D', 0.90f, 3, 0.0f, &t, tip);
  ev = feed_jitter(&d, 'D', 0.90f, 50, 0.08f, &t, tip);
  CHECK(ev.letters == 1);
  CHECK_STR(&d, "D");

  init(&d);
  t = 0;
  tip[0] = tip[1] = 160;
  feed_motion(&d, 'D', 0.90f, 3, 0.0f, &t, tip);
  ev = feed_motion(&d, 'D', 0.90f, 60, 0.0f, &t, tip);  // held still 2 s: D
  CHECK(ev.letters == 1);
  CHECK_STR(&d, "D");
}

int main(void) {
  // The pre-averaging parameters: the per-frame rules as before.
  g_legacy = true;
  test_defaults();
  test_commit_rule_and_hold();
  test_below_threshold_never_commits();
  test_frames_must_be_consecutive();
  test_candidate_change_resets_streak();
  test_hello_double_letter();
  test_hi_mom_space();
  test_no_space_before_any_text();
  test_backspace_and_clear();
  test_backspace_rearms_the_held_letter();
  test_backspace_of_a_trailing_space_is_not_undone_until_the_hand_returns();
  test_backspace_of_a_letter_blocks_the_space_too();
  test_backspace_on_empty_leaves_the_state_alone();
  test_full_buffer_drops_oldest();
  test_custom_config();
  test_clock_wraparound();
  test_argmax_tie_takes_lowest_index();
  test_motion_d_held_still_is_d();
  test_motion_jittery_d_held_is_still_d();
  test_motion_d_with_a_stroke_is_z();
  test_motion_still_shape_after_the_stroke_does_not_commit_again();
  test_motion_i_with_a_hook_is_j();
  test_motion_pending_times_out_as_the_still_letter();
  test_motion_brief_dropout_does_not_end_the_hold();
  test_clear_drops_a_pending_d();
  test_clear_then_the_held_letter_commits_again();
  test_motion_other_letters_commit_as_before();
  // The defaults: averaging on.
  g_legacy = false;
  test_avg_held_letter_with_dips_commits_once();
  test_avg_flicker_commits_nothing();
  test_avg_steady_letter_commits_in_two_frames();
  test_avg_live_conf_is_the_average();
  test_avg_letter_change_within_seven_frames();
  test_avg_hysteresis_holds_the_candidate();
  test_avg_dip_does_not_rearm_but_hand_loss_does();
  test_avg_space_after_hand_lost_not_during_dip();
  test_avg_hello();
  test_avg_motion_letters();
  printf("fs_decoder: %d checks, %d failed\n", g_checks, g_failures);
  return g_failures ? 1 : 0;
}
