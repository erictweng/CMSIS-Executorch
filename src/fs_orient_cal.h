// fs_orient_cal.h: the camera's image axes in the DevKit-E8 accelerometer's axes, written by
// tests/imu_calibrate.py (two poses: board as mounted and level, then rolled 90 degrees
// clockwise about the camera axis). Zero vectors = not calibrated: the roll stays 0 and the
// landmarks are not levelled.
// Calibrated 2026-10-18 12:51, residual 0.041.
#pragma once
#define FS_ORIENT_CALIBRATED 1
#define FS_ORIENT_CAL_DOWN {0.03263f, -0.99825f, 0.04931f}
#define FS_ORIENT_CAL_RIGHT {0.99661f, 0.03622f, 0.07384f}
