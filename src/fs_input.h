// fs_input.h: joystick presses to text-editing events. Pure C99 (host tests in
// tests/host/test_fs_input.c); the board's GPIO read is in fs_joystick.c.
#pragma once
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum { FS_INPUT_NONE = 0, FS_INPUT_BACKSPACE, FS_INPUT_CLEAR } fs_input_event_t;

// The previous sample of each button (true = pressed). fs_input_init() sets both to
// pressed, so the first sample can only release, never fire: a stick held at boot cannot
// wipe the screen.
typedef struct {
  bool down, left;
} fs_input_t;

void fs_input_init(fs_input_t* s);
// One sample per frame. An event on the press (released -> pressed) only: holding gives
// nothing more, no auto-repeat. DOWN pressed -> FS_INPUT_CLEAR, LEFT pressed ->
// FS_INPUT_BACKSPACE; both in one sample -> FS_INPUT_CLEAR.
fs_input_event_t fs_input_update(fs_input_t* s, bool down_pressed, bool left_pressed);

#ifdef __cplusplus
}
#endif
