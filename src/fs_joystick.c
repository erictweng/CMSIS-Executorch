// fs_joystick.c: see fs_joystick.h. Mirrors what the pack's VIO driver does for the stick
// (per-pin SetDirection + GetValue, no pin Setup: "Reconfiguring pin causes issue").
#include "fs_joystick.h"

#include <stdio.h>

#include "RTE_Components.h"
#ifdef RTE_Drivers_IO
#include "Driver_IO.h"
#include "board_defs.h"  // BOARD_JOY_SW_A_* (LEFT), BOARD_JOY_SW_C_* (DOWN)

#if BOARD_JOY_SW_A_GPIO_PORT != BOARD_JOY_SW_C_GPIO_PORT
#error "the joystick switches are expected on one GPIO port"
#endif
extern ARM_DRIVER_GPIO ARM_Driver_GPIO_(BOARD_JOY_SW_C_GPIO_PORT);
static ARM_DRIVER_GPIO* const gpio = &ARM_Driver_GPIO_(BOARD_JOY_SW_C_GPIO_PORT);
#define FS_JOY_DOWN_PIN BOARD_JOY_SW_C_GPIO_PIN
#define FS_JOY_LEFT_PIN BOARD_JOY_SW_A_GPIO_PIN
#endif

volatile int32_t g_fs_joy_status = 0;
volatile uint32_t g_fs_joy_reads = 0;

static bool up = false;

#ifdef RTE_Drivers_IO
// The driver refuses GetValue on an unpowered pin and PowerControl before Initialize.
static bool pin_up(uint8_t pin) {
  int32_t status = gpio->Initialize(pin, NULL);
  const char* call = "Initialize";
  if (status == ARM_DRIVER_OK) {
    status = gpio->PowerControl(pin, ARM_POWER_FULL);
    call = "PowerControl";
  }
  if (status == ARM_DRIVER_OK) {
    status = gpio->SetDirection(pin, GPIO_PIN_DIRECTION_INPUT);
    call = "SetDirection";
  }
  if (status != ARM_DRIVER_OK) {
    g_fs_joy_status = status;
    printf("joystick init FAILED at %s pin %d (status %ld)\n", call, (int)pin, (long)status);
    return false;
  }
  return true;
}
#endif

bool fs_joystick_init(void) {
#ifdef RTE_Drivers_IO
  if (!pin_up(FS_JOY_DOWN_PIN) || !pin_up(FS_JOY_LEFT_PIN)) return false;
  up = true;
  bool down = false, left = false;
  if (!fs_joystick_read(&down, &left)) return false;
  printf("joystick init OK (down %d left %d): DOWN clears the text, LEFT deletes one character\n", (int)down,
         (int)left);
  return true;
#else
  printf("joystick: no GPIO driver in this build\n");
  return false;
#endif
}

bool fs_joystick_read(bool* down_pressed, bool* left_pressed) {
  *down_pressed = false;
  *left_pressed = false;
  if (!up) return false;
#ifdef RTE_Drivers_IO
  uint32_t down = 1, left = 1;
  int32_t status = gpio->GetValue(FS_JOY_DOWN_PIN, &down);
  if (status == ARM_DRIVER_OK) status = gpio->GetValue(FS_JOY_LEFT_PIN, &left);
  if (status != ARM_DRIVER_OK) {
    g_fs_joy_status = status;
    return false;
  }
  *down_pressed = down == 0u;  // active low
  *left_pressed = left == 0u;
  g_fs_joy_reads++;
  return true;
#else
  return false;
#endif
}
