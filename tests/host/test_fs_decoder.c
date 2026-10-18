// Host tests for src/fs_decoder.c against the fs_api.h decoder contract. Feeds synthetic
// probability streams at 30 fps; recorded webcam.py streams can replace them later.
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

// Low-confidence frames (the hand moving or absent) covering at least `ms`.
static events_t gap(fs_decoder_t* d, uint32_t ms, uint32_t* t) {
  return feed(d, 'A', 0.20f, (int)((ms + FRAME_MS - 1) / FRAME_MS), t);
}

// Commit one letter the normal way: hold it for exactly commit_frames.
static events_t sign(fs_decoder_t* d, char letter, uint32_t* t) {
  return feed(d, letter, 0.90f, d->cfg.commit_frames, t);
}

static void test_defaults(void) {
  fs_decoder_t d;
  fs_decoder_init(&d, NULL);
  CHECK(d.cfg.commit_threshold == 0.60f);
  CHECK(d.cfg.commit_frames == 3);
  CHECK(d.cfg.space_ms == 1000);
  CHECK(d.cfg.release_ms == 300);
  CHECK(d.len == 0 && d.text[0] == '\0' && d.live == 0);
}

static void test_commit_rule_and_hold(void) {
  fs_decoder_t d;
  fs_decoder_init(&d, NULL);
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
  fs_decoder_init(&d, NULL);
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
  fs_decoder_init(&d, NULL);
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
  fs_decoder_init(&d, NULL);
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
  fs_decoder_init(&d, NULL);
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

  gap(&d, 350, &t);  // the hand drops for >= release_ms: the second L may commit
  CHECK(sign(&d, 'L', &t).last == FS_EV_LETTER);
  gap(&d, 200, &t);
  CHECK(sign(&d, 'O', &t).last == FS_EV_LETTER);
  CHECK_STR(&d, "HELLO");
}

static void test_hi_mom_space(void) {
  fs_decoder_t d;
  fs_decoder_init(&d, NULL);
  uint32_t t = 0;
  sign(&d, 'H', &t);
  gap(&d, 200, &t);
  sign(&d, 'I', &t);
  CHECK_STR(&d, "HI");

  events_t ev = gap(&d, 990, &t);  // just under space_ms since the last confident frame
  CHECK(ev.spaces == 0);
  ev = gap(&d, 100, &t);
  CHECK(ev.spaces == 1);
  CHECK_STR(&d, "HI ");
  ev = gap(&d, 3000, &t);  // one space per pause, however long
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
  fs_decoder_init(&d, NULL);
  uint32_t t = 5000;  // the clock need not start at zero
  events_t ev = gap(&d, 4000, &t);
  CHECK(ev.spaces == 0 && ev.letters == 0);
  CHECK_STR(&d, "");
}

static void test_backspace_and_clear(void) {
  fs_decoder_t d;
  fs_decoder_init(&d, NULL);
  uint32_t t = 0;
  sign(&d, 'H', &t);
  gap(&d, 200, &t);
  sign(&d, 'I', &t);
  gap(&d, 1100, &t);
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

static void test_full_buffer_drops_oldest(void) {
  fs_decoder_t d;
  fs_decoder_init(&d, NULL);
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
  CHECK(gap(&d, 500, &t).spaces == 1);
  CHECK_STR(&d, "K ");

  cfg.commit_frames = 0;  // clamped to 1
  fs_decoder_init(&d, &cfg);
  CHECK(d.cfg.commit_frames == 1);
}

static void test_clock_wraparound(void) {
  fs_decoder_t d;
  fs_decoder_init(&d, NULL);
  uint32_t t = 0xFFFFFFFFu - 400u;
  sign(&d, 'A', &t);
  CHECK(gap(&d, 1100, &t).spaces == 1);  // the gap spans the 32-bit wrap
  CHECK_STR(&d, "A ");
  CHECK(sign(&d, 'A', &t).last == FS_EV_LETTER);  // and so does the release
  CHECK_STR(&d, "A A");
}

static void test_argmax_tie_takes_lowest_index(void) {
  fs_decoder_t d;
  fs_decoder_init(&d, NULL);
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

static void test_motion_d_held_still_is_d(void) {
  fs_decoder_t d;
  fs_decoder_init(&d, NULL);
  uint32_t t = 0;
  float tip[2] = {160, 160};
  events_t ev = feed_motion(&d, 'D', 0.90f, 3, 0.0f, &t, tip);
  CHECK(ev.last == FS_EV_PENDING);  // not committed yet
  CHECK(d.pending == 'D' && d.live == 'D');
  CHECK_STR(&d, "");
  ev = feed_motion(&d, 'D', 0.90f, 10, 0.02f, &t, tip);  // jitter below the dead band
  CHECK(ev.letters == 0 && d.pending_path == 0.0f);
  ev = feed_motion(&d, 'A', 0.20f, 3, 0.0f, &t, tip);  // hand relaxes: resolves on the 3rd frame
  CHECK(ev.letters == 1);
  CHECK_STR(&d, "D");
  CHECK(d.pending == 0);
}

static void test_motion_d_with_a_stroke_is_z(void) {
  fs_decoder_t d;
  fs_decoder_init(&d, NULL);
  uint32_t t = 0;
  float tip[2] = {160, 160};
  feed_motion(&d, 'D', 0.90f, 3, 0.0f, &t, tip);
  events_t ev = feed_motion(&d, 'D', 0.90f, 6, 0.2f, &t, tip);  // 1.2 hand sizes of travel
  CHECK(ev.letters == 0 && ev.last == FS_EV_PENDING);
  CHECK(d.pending_path > 1.0f);
  ev = sign(&d, 'A', &t);  // moves on to an A: Z resolves at the switch, A commits normally
  CHECK_STR(&d, "ZA");
}

static void test_motion_i_with_a_hook_is_j(void) {
  fs_decoder_t d;
  fs_decoder_init(&d, NULL);
  uint32_t t = 0;
  float tip[2] = {100, 100};
  feed_motion(&d, 'I', 0.90f, 3, 0.0f, &t, tip);
  feed_motion(&d, 'I', 0.90f, 8, 0.15f, &t, tip);
  // The hand disappears: resolves immediately.
  float probs[FS_CLASSES];
  probs_for('A', 0.20f, probs);
  fs_motion_t gone = {false, {0, 0}, {0, 0}, 0};
  CHECK(fs_decoder_step_motion(&d, probs, &gone, t) == FS_EV_LETTER);
  CHECK_STR(&d, "J");
}

static void test_motion_pending_times_out_as_the_still_letter(void) {
  fs_decoder_t d;
  fs_decoder_init(&d, NULL);
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
  fs_decoder_init(&d, NULL);
  uint32_t t = 0;
  float tip[2] = {160, 160};
  feed_motion(&d, 'D', 0.90f, 3, 0.0f, &t, tip);
  feed_motion(&d, 'D', 0.90f, 3, 0.2f, &t, tip);
  feed_motion(&d, 'U', 0.70f, 2, 0.2f, &t, tip);  // two frames of another shape mid-stroke
  events_t ev = feed_motion(&d, 'D', 0.90f, 3, 0.2f, &t, tip);
  CHECK(ev.letters == 0 && d.pending == 'D');
  feed_motion(&d, 'A', 0.20f, 3, 0.0f, &t, tip);
  CHECK_STR(&d, "Z");
}

static void test_motion_other_letters_commit_as_before(void) {
  fs_decoder_t d;
  fs_decoder_init(&d, NULL);
  uint32_t t = 0;
  float tip[2] = {160, 160};
  events_t ev = feed_motion(&d, 'B', 0.90f, 3, 0.3f, &t, tip);  // a moving B is still a B
  CHECK(ev.last == FS_EV_LETTER);
  CHECK_STR(&d, "B");
  fs_decoder_init(&d, NULL);
  t = 0;
  ev = sign(&d, 'D', &t);  // no motion input: D commits like any letter
  CHECK(ev.last == FS_EV_LETTER);
  CHECK_STR(&d, "D");
}

int main(void) {
  test_defaults();
  test_commit_rule_and_hold();
  test_below_threshold_never_commits();
  test_frames_must_be_consecutive();
  test_candidate_change_resets_streak();
  test_hello_double_letter();
  test_hi_mom_space();
  test_no_space_before_any_text();
  test_backspace_and_clear();
  test_full_buffer_drops_oldest();
  test_custom_config();
  test_clock_wraparound();
  test_argmax_tie_takes_lowest_index();
  test_motion_d_held_still_is_d();
  test_motion_d_with_a_stroke_is_z();
  test_motion_i_with_a_hook_is_j();
  test_motion_pending_times_out_as_the_still_letter();
  test_motion_brief_dropout_does_not_end_the_hold();
  test_motion_other_letters_commit_as_before();
  printf("fs_decoder: %d checks, %d failed\n", g_checks, g_failures);
  return g_failures ? 1 : 0;
}
