// Host tests for src/fs_commit_hold.c: the panel's hold on newly committed letters.
//
//   clang -std=c99 -Wall -Wextra -Werror -pedantic -I src src/fs_commit_hold.c \
//         tests/host/test_fs_commit_hold.c -o /tmp/test_fs_commit_hold && /tmp/test_fs_commit_hold

#include <stdio.h>
#include <string.h>

#include "fs_commit_hold.h"

static int g_checks, g_failures;

#define CHECK(cond)                                                      \
  do {                                                                   \
    g_checks++;                                                          \
    if (!(cond)) {                                                       \
      g_failures++;                                                      \
      printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);             \
    }                                                                    \
  } while (0)

#define CHECK_SHOWN(h, want)                                                                   \
  do {                                                                                         \
    g_checks++;                                                                                \
    if (strcmp((h)->shown, (want)) != 0 || (h)->len != (int)strlen(want)) {                    \
      g_failures++;                                                                            \
      printf("FAIL %s:%d: shown \"%s\" (len %d), wanted \"%s\"\n", __FILE__, __LINE__, (h)->shown, \
             (h)->len, (want));                                                                \
    }                                                                                          \
  } while (0)

#define HOLD_MS 700u
#define FRAME_MS 66u  // ~15 fps

// Idle frames from t0 (exclusive) up to and including t1; counts the confirmations.
static int idle_until(fs_commit_hold_t* h, uint32_t t0, uint32_t t1) {
  int confirmed = 0;
  for (uint32_t t = t0 + FRAME_MS; t <= t1; t += FRAME_MS)
    if (fs_commit_hold_update(h, FS_EV_NONE, 0, t)) confirmed++;
  return confirmed;
}

static void test_single_letter_confirms_after_the_hold(void) {
  fs_commit_hold_t h;
  fs_commit_hold_init(&h, HOLD_MS);
  CHECK(!fs_commit_hold_update(&h, FS_EV_LETTER, 'A', 1000));
  CHECK(h.held == 'A');
  CHECK_SHOWN(&h, "");
  CHECK(!fs_commit_hold_update(&h, FS_EV_NONE, 0, 1000 + HOLD_MS - 1));
  CHECK_SHOWN(&h, "");
  CHECK(fs_commit_hold_update(&h, FS_EV_NONE, 0, 1000 + HOLD_MS));
  CHECK(h.held == 0);
  CHECK_SHOWN(&h, "A");
  CHECK(!fs_commit_hold_update(&h, FS_EV_NONE, 0, 1000 + HOLD_MS + FRAME_MS));  // confirmed once only
}

static void test_a_newer_letter_replaces_the_held_one(void) {
  fs_commit_hold_t h;
  fs_commit_hold_init(&h, HOLD_MS);
  fs_commit_hold_update(&h, FS_EV_LETTER, 'H', 0);
  idle_until(&h, 0, 1000);
  CHECK_SHOWN(&h, "H");
  fs_commit_hold_update(&h, FS_EV_LETTER, 'M', 2000);  // E read as M first
  CHECK(idle_until(&h, 2000, 2264) == 0);
  CHECK(!fs_commit_hold_update(&h, FS_EV_LETTER, 'E', 2300));  // the correction, 300 ms later
  CHECK(h.held == 'E');
  CHECK_SHOWN(&h, "H");  // M never appeared
  CHECK(!fs_commit_hold_update(&h, FS_EV_NONE, 0, 2300 + HOLD_MS - 1));
  CHECK(fs_commit_hold_update(&h, FS_EV_NONE, 0, 2300 + HOLD_MS));
  CHECK_SHOWN(&h, "HE");
}

static void test_letter_on_the_frame_the_hold_expires(void) {
  fs_commit_hold_t h;
  fs_commit_hold_init(&h, HOLD_MS);
  fs_commit_hold_update(&h, FS_EV_LETTER, 'A', 0);
  // The held letter has earned its hold: it is confirmed, the new one starts its own.
  CHECK(fs_commit_hold_update(&h, FS_EV_LETTER, 'B', HOLD_MS));
  CHECK_SHOWN(&h, "A");
  CHECK(h.held == 'B');
}

static void test_space_takes_the_held_letter_then_the_space(void) {
  fs_commit_hold_t h;
  fs_commit_hold_init(&h, HOLD_MS);
  fs_commit_hold_update(&h, FS_EV_LETTER, 'H', 0);
  idle_until(&h, 0, 1000);
  fs_commit_hold_update(&h, FS_EV_LETTER, 'I', 1000);
  CHECK(fs_commit_hold_update(&h, FS_EV_SPACE, ' ', 1200));  // hand away before the hold ran out
  CHECK_SHOWN(&h, "HI ");
  CHECK(h.held == 0);
  fs_commit_hold_update(&h, FS_EV_SPACE, ' ', 1300);  // no double space
  CHECK_SHOWN(&h, "HI ");
}

static void test_space_never_leads(void) {
  fs_commit_hold_t h;
  fs_commit_hold_init(&h, HOLD_MS);
  CHECK(!fs_commit_hold_update(&h, FS_EV_SPACE, ' ', 100));
  CHECK_SHOWN(&h, "");
}

static void test_backspace_during_and_after_the_hold(void) {
  fs_commit_hold_t h;
  fs_commit_hold_init(&h, HOLD_MS);
  fs_commit_hold_update(&h, FS_EV_LETTER, 'A', 0);
  idle_until(&h, 0, 1000);
  fs_commit_hold_update(&h, FS_EV_LETTER, 'B', 1000);
  CHECK(!fs_commit_hold_update(&h, FS_EV_BACKSPACE, 0, 1100));  // drops the held B
  CHECK(h.held == 0);
  CHECK_SHOWN(&h, "A");
  CHECK(!fs_commit_hold_update(&h, FS_EV_BACKSPACE, 0, 1200));  // removes the confirmed A
  CHECK_SHOWN(&h, "");
  CHECK(!fs_commit_hold_update(&h, FS_EV_BACKSPACE, 0, 1300));  // nothing left: no-op
  CHECK_SHOWN(&h, "");
}

static void test_clear_drops_everything(void) {
  fs_commit_hold_t h;
  fs_commit_hold_init(&h, HOLD_MS);
  fs_commit_hold_update(&h, FS_EV_LETTER, 'A', 0);
  idle_until(&h, 0, 1000);
  fs_commit_hold_update(&h, FS_EV_LETTER, 'B', 1000);
  CHECK(!fs_commit_hold_update(&h, FS_EV_CLEAR, 0, 1100));
  CHECK_SHOWN(&h, "");
  CHECK(h.held == 0);
}

static void test_pending_and_none_do_not_disturb_a_hold(void) {
  fs_commit_hold_t h;
  fs_commit_hold_init(&h, HOLD_MS);
  fs_commit_hold_update(&h, FS_EV_LETTER, 'D', 0);
  CHECK(!fs_commit_hold_update(&h, FS_EV_PENDING, 'D', 100));
  CHECK(!fs_commit_hold_update(&h, FS_EV_NONE, 0, 200));
  CHECK(h.held == 'D');
  CHECK_SHOWN(&h, "");
  CHECK(fs_commit_hold_update(&h, FS_EV_NONE, 0, HOLD_MS));
  CHECK_SHOWN(&h, "D");
}

static void test_oldest_character_drops_when_full(void) {
  fs_commit_hold_t h;
  fs_commit_hold_init(&h, HOLD_MS);
  uint32_t t = 0;
  for (int i = 0; i < 130; ++i) {
    fs_commit_hold_update(&h, FS_EV_LETTER, (char)('A' + i % 24), t);
    t += HOLD_MS;
    fs_commit_hold_update(&h, FS_EV_NONE, 0, t);
  }
  CHECK(h.len == FS_TEXT_MAX - 1);
  CHECK(h.shown[0] == (char)('A' + 3));  // the first three dropped
  CHECK(h.shown[h.len - 1] == (char)('A' + 129 % 24));
  CHECK(h.shown[h.len] == '\0');
}

static void test_time_wrap(void) {
  fs_commit_hold_t h;
  fs_commit_hold_init(&h, HOLD_MS);
  const uint32_t start = 0xFFFFFF00u;
  fs_commit_hold_update(&h, FS_EV_LETTER, 'W', start);
  CHECK(!fs_commit_hold_update(&h, FS_EV_NONE, 0, start + HOLD_MS - 1));  // wrapped past 0
  CHECK(fs_commit_hold_update(&h, FS_EV_NONE, 0, start + HOLD_MS));
  CHECK_SHOWN(&h, "W");
}

int main(void) {
  test_single_letter_confirms_after_the_hold();
  test_a_newer_letter_replaces_the_held_one();
  test_letter_on_the_frame_the_hold_expires();
  test_space_takes_the_held_letter_then_the_space();
  test_space_never_leads();
  test_backspace_during_and_after_the_hold();
  test_clear_drops_everything();
  test_pending_and_none_do_not_disturb_a_hold();
  test_oldest_character_drops_when_full();
  test_time_wrap();
  printf("fs_commit_hold: %d checks, %d failed\n", g_checks, g_failures);
  return g_failures ? 1 : 0;
}
