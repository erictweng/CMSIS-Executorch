// fs_input.h: joystick presses to text-editing events. Pure C99 (host tests in
// tests/host/test_fs_input.c); the board's GPIO read is in fs_joystick.c.
#pragma once
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum { FS_INPUT_NONE = 0, FS_INPUT_BACKSPACE, FS_INPUT_CLEAR, FS_INPUT_SPACE, FS_INPUT_SWAP_VIEW } fs_input_event_t;

// The previous sample of each button (true = pressed). fs_input_init() sets all to
// pressed, so the first sample can only release, never fire: a stick held at boot cannot
// wipe the screen.
typedef struct {
  bool down, left, right, up;
} fs_input_t;

void fs_input_init(fs_input_t* s);
// One sample per frame. An event on the press (released -> pressed) only: holding gives
// nothing more, no auto-repeat. DOWN pressed -> FS_INPUT_CLEAR, LEFT pressed ->
// FS_INPUT_BACKSPACE, RIGHT pressed -> FS_INPUT_SPACE, UP pressed -> FS_INPUT_SWAP_VIEW (the
// panel swaps its camera view and hand inset); several in one sample -> clear wins, then
// backspace, then space.
fs_input_event_t fs_input_update(fs_input_t* s, bool down_pressed, bool left_pressed, bool right_pressed,
                                 bool up_pressed);

#ifdef __cplusplus
}
#endif
