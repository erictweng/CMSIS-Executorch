// fs_imu.c: see fs_imu.h. The driver (components/Source/ICM42670P_IMU.c in the Alif pack) is
// polled: no data-ready interrupt, one blocking register write + 6-byte read per call.
#include "fs_imu.h"

#include <stdio.h>

#include "RTE_Components.h"
#ifdef RTE_Drivers_ICM42670P
#include "Driver_IMU.h"

extern ARM_DRIVER_IMU ICM42670P;
static ARM_DRIVER_IMU* const imu = &ICM42670P;
#endif

volatile uint32_t g_fs_imu_stage = FS_IMU_NOT_RUN;
volatile int32_t g_fs_imu_status = 0;
volatile uint32_t g_fs_imu_reads = 0;
volatile uint32_t g_fs_imu_bad = 0;

static bool up = false;

bool fs_imu_init(void) {
#ifdef RTE_Drivers_ICM42670P
  g_fs_imu_stage = FS_IMU_INITIALIZING;
  printf("imu: init (I3C + GPIO)\n");
  int32_t status = imu->Initialize();
  if (status != ARM_DRIVER_OK) {
    g_fs_imu_stage = FS_IMU_ERR_INIT;
    g_fs_imu_status = status;
    printf("imu init FAILED at Initialize (status %ld)\n", (long)status);
    return false;
  }
  g_fs_imu_stage = FS_IMU_POWERING;
  printf("imu: power (I3C bus, dynamic address, WHO_AM_I 0x67, config)\n");
  status = imu->PowerControl(ARM_POWER_FULL);
  if (status != ARM_DRIVER_OK) {
    g_fs_imu_stage = FS_IMU_ERR_POWER;
    g_fs_imu_status = status;
    printf("imu init FAILED at PowerControl (status %ld)\n", (long)status);
    return false;
  }
  up = true;
  g_fs_imu_stage = FS_IMU_OK;
  float g[3] = {0.f, 0.f, 0.f};
  const bool ok = fs_imu_read(g);
  printf("imu init OK: g = (%.3f, %.3f, %.3f)%s\n", (double)g[0], (double)g[1], (double)g[2], ok ? "" : " (zero reading)");
  return true;
#else
  printf("imu: no ICM42670P driver in this build\n");
  return false;
#endif
}

bool fs_imu_read(float g[3]) {
#ifdef RTE_Drivers_ICM42670P
  if (!up) return false;
  ARM_IMU_COORDINATES xyz = {{0, 0}, {0, 0}, {0, 0}};
  if (imu->Control(IMU_GET_ACCELEROMETER_DATA, (uint32_t)&xyz) != ARM_DRIVER_OK) return false;
  g[0] = (float)xyz.x.int_val + (float)xyz.x.fract_val * 1e-6f;
  g[1] = (float)xyz.y.int_val + (float)xyz.y.fract_val * 1e-6f;
  g[2] = (float)xyz.z.int_val + (float)xyz.z.fract_val * 1e-6f;
  if (g[0] == 0.0f && g[1] == 0.0f && g[2] == 0.0f) {  // a failed bus read leaves zeros
    g_fs_imu_bad++;
    return false;
  }
  g_fs_imu_reads++;
  return true;
#else
  (void)g;
  return false;
#endif
}
