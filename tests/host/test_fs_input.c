// Host tests for src/fs_input.c: joystick samples -> one event per press.
//
//   clang -std=c99 -Wall -Wextra -Werror -pedantic -I src src/fs_input.c \
//         tests/host/test_fs_input.c -o /tmp/test_fs_input && /tmp/test_fs_input

#include <stdio.h>

#include "fs_input.h"

static int g_checks, g_failures;

#define CHECK(cond)                                                      \
  do {                                                                   \
    g_checks++;                                                          \
    if (!(cond)) {                                                       \
      g_failures++;                                                      \
      printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);             \
    }                                                                    \
  } while (0)

// `n` samples of the same state; the number of non-NONE events.
static int hold(fs_input_t* s, bool down, bool left, int n) {
  int events = 0;
  for (int i = 0; i < n; ++i)
    if (fs_input_update(s, down, left, false, false) != FS_INPUT_NONE) events++;
  return events;
}

static void test_press_gives_one_event(void) {
  fs_input_t s;
  fs_input_init(&s);
  CHECK(fs_input_update(&s, false, false, false, false) == FS_INPUT_NONE);
  CHECK(fs_input_update(&s, false, true, false, false) == FS_INPUT_BACKSPACE);
  CHECK(fs_input_update(&s, false, false, false, false) == FS_INPUT_NONE);
  CHECK(fs_input_update(&s, true, false, false, false) == FS_INPUT_CLEAR);
  CHECK(fs_input_update(&s, false, false, false, false) == FS_INPUT_NONE);
}

static void test_hold_does_not_repeat(void) {
  fs_input_t s;
  fs_input_init(&s);
  fs_input_update(&s, false, false, false, false);
  CHECK(fs_input_update(&s, false, true, false, false) == FS_INPUT_BACKSPACE);
  CHECK(hold(&s, false, true, 30) == 0);
  fs_input_update(&s, false, false, false, false);
  CHECK(fs_input_update(&s, true, false, false, false) == FS_INPUT_CLEAR);
  CHECK(hold(&s, true, false, 30) == 0);
}

static void test_release_and_press_again(void) {
  fs_input_t s;
  fs_input_init(&s);
  fs_input_update(&s, false, false, false, false);
  for (int i = 0; i < 3; ++i) {
    CHECK(fs_input_update(&s, false, true, false, false) == FS_INPUT_BACKSPACE);
    CHECK(fs_input_update(&s, false, false, false, false) == FS_INPUT_NONE);
  }
  for (int i = 0; i < 3; ++i) {
    CHECK(fs_input_update(&s, true, false, false, false) == FS_INPUT_CLEAR);
    CHECK(fs_input_update(&s, false, false, false, false) == FS_INPUT_NONE);
  }
}

static void test_both_in_one_sample_is_clear(void) {
  fs_input_t s;
  fs_input_init(&s);
  fs_input_update(&s, false, false, false, false);
  CHECK(fs_input_update(&s, true, true, false, false) == FS_INPUT_CLEAR);
  CHECK(hold(&s, true, true, 5) == 0);
  CHECK(fs_input_update(&s, true, false, false, false) == FS_INPUT_NONE);   // LEFT released, DOWN still held
  CHECK(fs_input_update(&s, true, true, false, false) == FS_INPUT_BACKSPACE);  // a new LEFT press under a held DOWN
}

static void test_first_sample_never_fires(void) {
  fs_input_t s;
  fs_input_init(&s);
  CHECK(fs_input_update(&s, true, true, false, false) == FS_INPUT_NONE);
  CHECK(fs_input_update(&s, true, true, false, false) == FS_INPUT_NONE);
  CHECK(fs_input_update(&s, false, false, false, false) == FS_INPUT_NONE);
  CHECK(fs_input_update(&s, true, false, false, false) == FS_INPUT_CLEAR);
  fs_input_init(&s);
  CHECK(fs_input_update(&s, false, true, false, false) == FS_INPUT_NONE);  // init again: pressed first sample
  CHECK(fs_input_update(&s, false, false, false, false) == FS_INPUT_NONE);
  CHECK(fs_input_update(&s, false, true, false, false) == FS_INPUT_BACKSPACE);
}

static void test_right_adds_a_space_and_loses_to_the_others(void) {
  fs_input_t s;
  fs_input_init(&s);
  CHECK(fs_input_update(&s, false, false, true, false) == FS_INPUT_NONE);  // held at boot: no event
  fs_input_update(&s, false, false, false, false);
  CHECK(fs_input_update(&s, false, false, true, false) == FS_INPUT_SPACE);
  CHECK(fs_input_update(&s, false, false, true, false) == FS_INPUT_NONE);  // no auto-repeat
  fs_input_update(&s, false, false, false, false);
  CHECK(fs_input_update(&s, true, false, true, false) == FS_INPUT_CLEAR);     // clear wins
  fs_input_update(&s, false, false, false, false);
  CHECK(fs_input_update(&s, false, true, true, false) == FS_INPUT_BACKSPACE); // then backspace
  fs_input_update(&s, false, false, false, false);
  CHECK(fs_input_update(&s, false, false, true, false) == FS_INPUT_SPACE);
}

static void test_up_swaps_the_views_and_loses_to_the_others(void) {
  fs_input_t s;
  fs_input_init(&s);
  CHECK(fs_input_update(&s, false, false, false, true) == FS_INPUT_NONE);  // held at boot: no event
  fs_input_update(&s, false, false, false, false);
  CHECK(fs_input_update(&s, false, false, false, true) == FS_INPUT_SWAP_VIEW);
  CHECK(fs_input_update(&s, false, false, false, true) == FS_INPUT_NONE);  // no auto-repeat
  fs_input_update(&s, false, false, false, false);
  CHECK(fs_input_update(&s, false, false, true, true) == FS_INPUT_SPACE);  // space wins over the swap
  fs_input_update(&s, false, false, false, false);
  CHECK(fs_input_update(&s, false, false, false, true) == FS_INPUT_SWAP_VIEW);
}

int main(void) {
  test_press_gives_one_event();
  test_up_swaps_the_views_and_loses_to_the_others();
  test_right_adds_a_space_and_loses_to_the_others();
  test_hold_does_not_repeat();
  test_release_and_press_again();
  test_both_in_one_sample_is_clear();
  test_first_sample_never_fires();
  printf("fs_input: %d checks, %d failed\n", g_checks, g_failures);
  return g_failures ? 1 : 0;
}
