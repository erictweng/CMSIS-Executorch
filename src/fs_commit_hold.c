// fs_commit_hold.c — see fs_commit_hold.h.
#include "fs_commit_hold.h"

#include <string.h>

// The same rule as fs_decoder.c's append: at FS_TEXT_MAX - 1 characters the oldest drops.
static void append(fs_commit_hold_t* h, char c) {
  if (h->len >= FS_TEXT_MAX - 1) {
    memmove(h->shown, h->shown + 1, (size_t)(h->len - 1));
    h->len--;
  }
  h->shown[h->len++] = c;
  h->shown[h->len] = '\0';
}

void fs_commit_hold_init(fs_commit_hold_t* h, uint32_t hold_ms) {
  memset(h, 0, sizeof(*h));
  h->hold_ms = hold_ms;
}

bool fs_commit_hold_update(fs_commit_hold_t* h, fs_event_t event, char committed, uint32_t now_ms) {
  bool confirmed = false;
  if (h->held && now_ms - h->held_since_ms >= h->hold_ms) {
    append(h, h->held);
    h->held = 0;
    confirmed = true;
  }
  switch (event) {
    case FS_EV_LETTER:
      if (committed) {  // replaces a letter still held: the earlier one is never shown
        h->held = committed;
        h->held_since_ms = now_ms;
      }
      break;
    case FS_EV_SPACE:
      if (h->held) {  // the word is over: take the last letter now
        append(h, h->held);
        h->held = 0;
        confirmed = true;
      }
      if (h->len > 0 && h->shown[h->len - 1] != ' ') append(h, ' ');  // as the decoder: no leading or double space
      break;
    case FS_EV_BACKSPACE:
      if (h->held) {
        h->held = 0;  // the decoder removed exactly the held letter
      } else if (h->len > 0) {
        h->shown[--h->len] = '\0';
      }
      break;
    case FS_EV_CLEAR:
      h->len = 0;
      h->shown[0] = '\0';
      h->held = 0;
      break;
    case FS_EV_NONE:
    case FS_EV_PENDING:
    default:
      break;
  }
  return confirmed;
}
