// fs_commit_hold.h — the panel's copy of the committed text, with a hold on each new letter.
//
// UI only: the decoder (fs_decoder.c) commits letters as it does today; the panel keeps its own
// text and shows a newly committed letter dimmed for hold_ms before it joins the confirmed
// line. A letter the decoder commits during that hold replaces the held one, so a wrong letter
// followed by the right one within hold_ms (M then E while signing "hello") shows only the
// right one. A space (word gap) confirms the held letter at once, then adds the space; a
// backspace during the hold drops the held letter (the decoder removed exactly that one);
// clear drops everything.
//
// One call per frame with that frame's decoder event. Pure C99, host tests in
// tests/host/test_fs_commit_hold.c. Notes: the decoder may append two letters in one frame
// (a pending D/I resolving plus the next letter) and report one FS_EV_LETTER; the caller
// passes the last character, which is the right one under the replace rule. A backspace that
// replaces a commit from the same frame (joystick pressed within one frame of a commit) drops
// the held letter the decoder kept; rare and accepted.
#pragma once
#include <stdbool.h>
#include <stdint.h>

#include "fs_api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  char shown[FS_TEXT_MAX];  // the panel's confirmed text; the oldest character drops when full
  int len;
  char held;                // the letter waiting to join shown (0 = none)
  uint32_t held_since_ms;
  uint32_t hold_ms;
} fs_commit_hold_t;

void fs_commit_hold_init(fs_commit_hold_t* h, uint32_t hold_ms);
// One call per frame. `committed` is the decoder's newest character (d->text[d->len - 1]), read
// only for FS_EV_LETTER. The held letter is confirmed first (hold_ms elapsed, unsigned so a
// wrapping now_ms is fine), then the event applies. Returns true on a frame a letter joins shown.
bool fs_commit_hold_update(fs_commit_hold_t* h, fs_event_t event, char committed, uint32_t now_ms);

#ifdef __cplusplus
}
#endif
