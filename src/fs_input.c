// fs_input.c: see fs_input.h.
#include "fs_input.h"

void fs_input_init(fs_input_t* s) {
  s->down = true;
  s->left = true;
}

fs_input_event_t fs_input_update(fs_input_t* s, bool down_pressed, bool left_pressed) {
  const bool down_edge = down_pressed && !s->down, left_edge = left_pressed && !s->left;
  s->down = down_pressed;
  s->left = left_pressed;
  return down_edge ? FS_INPUT_CLEAR : left_edge ? FS_INPUT_BACKSPACE : FS_INPUT_NONE;
}
