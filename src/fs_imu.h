// fs_imu.h: the DevKit-E8's ICM-42670-P accelerometer over I3C (Alif pack driver), read once
// per frame for the camera's roll (fs_orient.h). Only built with the DevKit-E8 board layer.
#pragma once
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Which fs_imu_init() stage is running / failed; readable over J-Link when the UART is dead.
// The pack driver's bus setup can spin without a timeout if the I3C interrupt never fires, so
// a boot that stops shows the stage it is stuck in.
typedef enum {
  FS_IMU_NOT_RUN = 0,
  FS_IMU_INITIALIZING,   // driver Initialize: I3C + GPIO init
  FS_IMU_POWERING,       // PowerControl FULL: I3C bus setup, dynamic address, WHO_AM_I, config
  FS_IMU_OK,
  FS_IMU_ERR_INIT,
  FS_IMU_ERR_POWER,
} fs_imu_stage_t;
extern volatile uint32_t g_fs_imu_stage;
extern volatile int32_t g_fs_imu_status;   // the failing driver call's return code
extern volatile uint32_t g_fs_imu_reads;   // successful reads
extern volatile uint32_t g_fs_imu_bad;     // all-zero readings (bus error: the driver still says OK)

bool fs_imu_init(void);
// Accelerometer in g; false when the sensor is not up or the reading is all zero.
bool fs_imu_read(float g[3]);

#ifdef __cplusplus
}
#endif
