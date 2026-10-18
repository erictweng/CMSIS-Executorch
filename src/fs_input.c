// fs_input.c: see fs_input.h.
#include "fs_input.h"

void fs_input_init(fs_input_t* s) {
  s->down = true;
  s->left = true;
  s->right = true;
  s->up = true;
}

fs_input_event_t fs_input_update(fs_input_t* s, bool down_pressed, bool left_pressed, bool right_pressed,
                                 bool up_pressed) {
  const bool down_edge = down_pressed && !s->down, left_edge = left_pressed && !s->left,
             right_edge = right_pressed && !s->right, up_edge = up_pressed && !s->up;
  s->down = down_pressed;
  s->left = left_pressed;
  s->right = right_pressed;
  s->up = up_pressed;
  return down_edge    ? FS_INPUT_CLEAR
         : left_edge  ? FS_INPUT_BACKSPACE
         : right_edge ? FS_INPUT_SPACE
         : up_edge    ? FS_INPUT_SWAP_VIEW
                      : FS_INPUT_NONE;
}
