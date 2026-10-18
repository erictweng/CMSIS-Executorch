// fs_joystick.h: the DevKit-E8's 5-way joystick on LPGPIO (port 15), active low. DOWN
// (JOY_SW_C, P15_2), LEFT (JOY_SW_A, P15_0), RIGHT (JOY_SW_D, P15_3) and UP (JOY_SW_B, P15_1)
// are read. Built with the DevKit-E8 board layer only; the press -> event logic is in
// fs_input.c (host-tested).
#pragma once
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

extern volatile int32_t g_fs_joy_status;   // the failing driver call's return code (0 = OK)
extern volatile uint32_t g_fs_joy_reads;   // successful polls (J-Link telemetry)

// Initialize + PowerControl(FULL) + SetDirection(INPUT) on the four pins; false (and a UART
// line) if the driver refuses. Boot already muxed the pins with pull-ups and debounce
// (gpios.h), so no pinmux is touched, and the LCD reset on P15_5 is untouched.
bool fs_joystick_init(void);
// Pressed = pin reads low. Returns false (all outputs false) when not up or a read fails.
bool fs_joystick_read(bool* down_pressed, bool* left_pressed, bool* right_pressed, bool* up_pressed);

#ifdef __cplusplus
}
#endif
