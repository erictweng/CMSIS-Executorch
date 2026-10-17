// fs_camera.c: the DevKit-E8 MT9M114 camera behind the fs_api.h camera contract.
//
// The sensor sits on MIPI CSI-2; Alif's CPI driver configures it over I2C1 and
// captures its frames into SRAM by DMA. The image format is
// RTE_MT9M114_CAMERA_SENSOR_MIPI_IMAGE_CONFIG = 5 in RTE_Device.h: 320x320 RGB565.
//
// This drives the CPI driver directly rather than through the board's vStream
// VideoIn wrapper: the wrapper folds every CPI error into one end-of-stream event,
// and bring-up needs to know which error it is (FIFO overrun, bus error, CSI-2).
// Frames go into a two-block ring, re-armed on every VSYNC like the wrapper does.
//
// Step 3 of the S1 camera plan: bring the camera up, start streaming, and only
// report OK once a whole frame has arrived. fs_camera_get_crop() follows in step 4.

#include "fs_camera.h"

#include <stdio.h>

#include "RTE_Components.h"
#include CMSIS_device_header

#include "Driver_CPI.h"
#include "sys_utils.h"

extern ARM_DRIVER_CPI Driver_CPI;
static ARM_DRIVER_CPI* const cpi = &Driver_CPI;

// The frame ring lives in the bulk SRAM next to the model pools: the board layer's
// linker script routes .bss.ai_pool there (DTCM is too small for 2 x 200 KB) and
// the CPI's DMA can reach it. 32-byte aligned for D-cache maintenance.
#ifdef APP_POOL_SECTION
#define FS_CAM_BUF_ATTRIBUTES __attribute__((section(APP_POOL_SECTION), aligned(32)))
#else
#define FS_CAM_BUF_ATTRIBUTES __attribute__((aligned(32)))
#endif
static uint8_t s_frames[FS_CAM_FRAMES][FS_CAM_FRAME_BYTES] FS_CAM_BUF_ATTRIBUTES;

volatile uint32_t g_fs_camera_frames = 0;
volatile uint32_t g_fs_camera_overflows = 0;
volatile uint32_t g_fs_camera_errors = 0;
volatile uint32_t g_fs_camera_stage = FS_CAM_NOT_RUN;
volatile int32_t g_fs_camera_status = 0;
volatile uint32_t g_fs_camera_first_frame_ms = 0;
volatile uint32_t g_fs_camera_events[FS_CAM_EVENT_BITS] = {0};
volatile uint32_t g_fs_camera_streaming = 0;

static volatile uint32_t s_capture_idx = 0;  // ring block the CPI is writing
static volatile bool s_synced = false;      // first VSYNC seen (frame boundary known)

// From the pack's DevKit-E8 M55_HP main.c (vbat_init): the trimmed board layer does
// not power the MIPI D-PHY, which the CSI-2 receiver needs.
#define VBAT_PWR_CTRL_TX_DPHY_PWR_MASK (1U << 0)
#define VBAT_PWR_CTRL_TX_DPHY_ISO (1U << 1)
#define VBAT_PWR_CTRL_RX_DPHY_PWR_MASK (1U << 4)
#define VBAT_PWR_CTRL_RX_DPHY_ISO (1U << 5)
#define VBAT_PWR_CTRL_DPHY_PLL_PWR_MASK (1U << 8)
#define VBAT_PWR_CTRL_DPHY_PLL_ISO (1U << 9)
#define VBAT_PWR_CTRL_DPHY_VPH_1P8_PWR_BYP_EN (1U << 12)

static void mipi_dphy_power_on(void) {
  VBAT->PWR_CTRL &= ~(VBAT_PWR_CTRL_TX_DPHY_PWR_MASK | VBAT_PWR_CTRL_RX_DPHY_PWR_MASK |
                      VBAT_PWR_CTRL_DPHY_PLL_PWR_MASK | VBAT_PWR_CTRL_DPHY_VPH_1P8_PWR_BYP_EN);
  VBAT->PWR_CTRL &= ~(VBAT_PWR_CTRL_TX_DPHY_ISO | VBAT_PWR_CTRL_RX_DPHY_ISO | VBAT_PWR_CTRL_DPHY_PLL_ISO);
}

#define FS_CAM_ERROR_EVENTS                                                                       \
  (ARM_CPI_EVENT_ERR_CAMERA_INPUT_FIFO_OVERRUN | ARM_CPI_EVENT_ERR_CAMERA_OUTPUT_FIFO_OVERRUN |   \
   ARM_CPI_EVENT_ERR_HARDWARE | ARM_CPI_EVENT_MIPI_CSI2_ERROR)

// CPI interrupt context.
static void on_cpi_event(uint32_t event) {
  for (uint32_t b = 0; b < FS_CAM_EVENT_BITS; b++)
    if (event & (1U << b)) g_fs_camera_events[b]++;
  if (event & FS_CAM_ERROR_EVENTS) g_fs_camera_errors++;

  if (event & ARM_CPI_EVENT_CAMERA_FRAME_VSYNC_DETECTED) {
    if (!s_synced) {
      s_synced = true;  // start of the first frame: nothing complete yet
    } else {
      // Start of the next frame: the block just written is complete. Point the
      // CPI at the other block.
      g_fs_camera_frames++;
      s_capture_idx ^= 1U;
      cpi->CaptureVideo(s_frames[s_capture_idx]);
    }
  }
}

static bool fail(fs_camera_stage_t stage, const char* what, int32_t status) {
  g_fs_camera_stage = stage;
  g_fs_camera_status = status;
  printf("camera init FAILED at %s (stage %d, status %ld; events vsync %lu infifo %lu outfifo %lu "
         "bus %lu csi2 %lu)\n",
         what, (int)stage, (long)status, (unsigned long)g_fs_camera_events[2],
         (unsigned long)g_fs_camera_events[3], (unsigned long)g_fs_camera_events[4],
         (unsigned long)g_fs_camera_events[5], (unsigned long)g_fs_camera_events[6]);
  return false;
}

bool fs_camera_init(void) {
  g_fs_camera_frames = 0;
  g_fs_camera_overflows = 0;
  g_fs_camera_errors = 0;
  for (uint32_t b = 0; b < FS_CAM_EVENT_BITS; b++) g_fs_camera_events[b] = 0;
  s_capture_idx = 0;
  s_synced = false;

  mipi_dphy_power_on();

  // CPI, MIPI CSI-2 and D-PHY setup, then the MT9M114: power/reset GPIOs, chip ID
  // (0x2481) and its register table over I2C1.
  int32_t status = cpi->Initialize(on_cpi_event);
  if (status != ARM_DRIVER_OK) return fail(FS_CAM_ERR_INIT, "CPI Initialize", status);
  status = cpi->PowerControl(ARM_POWER_FULL);
  if (status != ARM_DRIVER_OK) return fail(FS_CAM_ERR_INIT, "CPI PowerControl", status);
  (void)cpi->Stop();  // a capture left running by a previous core reset
  status = cpi->Control(CPI_CAMERA_SENSOR_CONFIGURE, 0U);
  if (status != ARM_DRIVER_OK) return fail(FS_CAM_ERR_SENSOR, "MT9M114 configure over I2C1", status);
  status = cpi->Control(CPI_CONFIGURE, 0U);
  if (status != ARM_DRIVER_OK) return fail(FS_CAM_ERR_INIT, "CPI configure", status);
  status = cpi->Control(CPI_EVENTS_CONFIGURE, ARM_CPI_EVENT_CAMERA_CAPTURE_STOPPED |
                                                  ARM_CPI_EVENT_CAMERA_FRAME_VSYNC_DETECTED |
                                                  FS_CAM_ERROR_EVENTS);
  if (status != ARM_DRIVER_OK) return fail(FS_CAM_ERR_INIT, "CPI events configure", status);

  status = cpi->CaptureVideo(s_frames[0]);
  if (status != ARM_DRIVER_OK) return fail(FS_CAM_ERR_START, "CPI CaptureVideo", status);
  g_fs_camera_streaming = 1;

  // Init only counts as OK once a whole frame has arrived: the sensor answering on
  // I2C does not prove the CSI-2 link and the CPI DMA work.
  uint32_t waited_ms = 0;
  while (g_fs_camera_frames == 0 && waited_ms < FS_CAM_FIRST_FRAME_TIMEOUT_MS) {
    sys_busy_loop_us(1000);
    waited_ms++;
  }
  if (g_fs_camera_frames == 0) {
    fs_camera_stop();
    return fail(FS_CAM_ERR_NO_FRAME, "first frame (timeout)", (int32_t)waited_ms);
  }
  g_fs_camera_first_frame_ms = waited_ms;

  g_fs_camera_stage = FS_CAM_OK;
  g_fs_camera_status = 0;
  printf("camera init OK %dx%d RGB565 (first frame after %lu ms, %lu error events)\n", FS_CAM_WIDTH,
         FS_CAM_HEIGHT, (unsigned long)waited_ms, (unsigned long)g_fs_camera_errors);
  return true;
}

void fs_camera_stop(void) {
  if (!g_fs_camera_streaming) return;
  (void)cpi->Stop();
  (void)cpi->PowerControl(ARM_POWER_OFF);
  (void)cpi->Uninitialize();
  g_fs_camera_streaming = 0;
}
