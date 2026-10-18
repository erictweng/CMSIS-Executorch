// fs_orient.h: the camera's roll relative to gravity, from the board's accelerometer, and the
// levelling of hand landmarks by it, so the letter MLP measures the hand's orientation from
// the true vertical rather than the camera's.
//
// Image axes: x right, y down. The calibration gives those two directions as unit vectors in
// the accelerometer's axes (tests/imu_calibrate.py, two poses -> src/fs_orient_cal.h). The
// roll is the angle of the gravity direction in the image, measured from image-down toward
// image-right: 0 when the camera is level. Levelling rotates the landmarks about the wrist by
// that angle, so gravity points straight down afterwards. Uncalibrated (zero vectors) or no
// sensor (|g| ~ 0): roll 0, nothing changes.
//
// Pure C99 without hardware (tests/host/test_fs_orient.c; reference ml/handlandmark.py).
#pragma once
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FS_ORIENT_ALPHA 0.1f   // gravity EMA weight on the new reading (the board moves slowly)
#define FS_ORIENT_MIN_G 0.5f   // |g| below this: no usable sensor reading
// Gravity's component in the image plane below this (g): the camera looks along gravity (board
// flat on a table, lens up or down), the image has no "down" and the roll is undefined -> 0.
#define FS_ORIENT_MIN_INPLANE 0.3f

typedef struct {
  float down[3];   // image +y (down) in accelerometer axes, unit length
  float right[3];  // image +x (right) in accelerometer axes, unit length
} fs_orient_cal_t;

typedef struct {
  float g[3];  // averaged gravity, in g
  bool valid;
} fs_orient_t;

void fs_orient_reset(fs_orient_t* s);
// Feed a raw accelerometer reading (g); the average restarts from the first one.
void fs_orient_gravity(fs_orient_t* s, const float g[3], float alpha);
// The image's roll relative to gravity, radians (0 when level, uncalibrated, or no reading).
float fs_orient_roll(const fs_orient_cal_t* cal, const float g[3]);
// Rotate the landmarks' x, y about `about` by `roll` so gravity points straight down.
void fs_orient_level(float xyz[][3], int count, float roll, const float about[2]);

#ifdef __cplusplus
}
#endif
