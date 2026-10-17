// fs_camera.h: the DevKit-E8 camera beyond the fs_api.h contract (fs_camera_init,
// fs_camera_get_crop): frame access, the frame geometry and the counters the UART
// log and the board tests read.
#pragma once
#include <stdint.h>

#include "fs_api.h"

#ifdef __cplusplus
extern "C" {
#endif

// MT9M114 image config 5 (RTE_Device.h): 320x320 RGB565, 2 bytes per pixel.
#define FS_CAM_WIDTH 320
#define FS_CAM_HEIGHT 320
#define FS_CAM_BYTES_PER_PIXEL 2
#define FS_CAM_FRAME_BYTES (FS_CAM_WIDTH * FS_CAM_HEIGHT * FS_CAM_BYTES_PER_PIXEL)
// Ring of frame blocks: the CPI writes one while the newest complete one is read;
// with three, a block handed out is not overwritten for two frame periods (~66 ms).
#define FS_CAM_FRAMES 3

// fs_camera_get_crop()'s box: the centre FS_CAM_BOX x FS_CAM_BOX of the frame,
// averaged 2x2 down to FS_SIZE x FS_SIZE (the model input).
#define FS_CAM_BOX (2 * FS_SIZE)

// fs_camera_init() waits this long for the first complete frame before it reports OK.
#define FS_CAM_FIRST_FRAME_TIMEOUT_MS 2000
// fs_camera_latest_frame() restarts the capture once if no frame arrives in this time.
#define FS_CAM_STALL_MS 250

// Which fs_camera_init() stage failed (0 = none); fs_camera_init() prints it too.
typedef enum {
  FS_CAM_OK = 0,
  FS_CAM_ERR_INIT,         // CPI / MIPI CSI-2 / D-PHY driver setup
  FS_CAM_ERR_SENSOR,       // MT9M114 over I2C1: power, reset, chip ID 0x2481, register table
  FS_CAM_ERR_START,        // CPI CaptureVideo
  FS_CAM_ERR_NO_FRAME,     // streaming started but no frame within FS_CAM_FIRST_FRAME_TIMEOUT_MS
  FS_CAM_NOT_RUN = 0xFF,   // fs_camera_init() has not been called
} fs_camera_stage_t;

// Wait for a frame newer than the last one returned and return its RGB565 pixels
// (row-major, FS_CAM_WIDTH x FS_CAM_HEIGHT, D-cache invalidated), with its frame
// number in *seq if seq is not NULL. Restarts a stalled capture once; NULL if
// still no frame within timeout_ms.
const uint16_t* fs_camera_latest_frame(uint32_t timeout_ms, uint32_t* seq);

// RGB565 -> RGB888 crop of one frame (the conversion fs_camera_get_crop uses).
void fs_camera_crop(const uint16_t* frame, uint8_t rgb[FS_SIZE * FS_SIZE * 3]);

// Stop capturing and power the CPI down.
void fs_camera_stop(void);

// Counters updated from the CPI callback (interrupt context).
#define FS_CAM_EVENT_BITS 7  // ARM_CPI_EVENT_* bits 0..6
extern volatile uint32_t g_fs_camera_frames;    // completed frames (VSYNC after the first)
extern volatile uint32_t g_fs_camera_overflows; // complete frames replaced before anyone read them
extern volatile uint32_t g_fs_camera_errors;    // FIFO overrun, bus or CSI-2 error events
extern volatile uint32_t g_fs_camera_restarts;  // captures restarted after a stall or a stop
extern volatile uint32_t g_fs_camera_overrun;   // 1: the CPI wrote past the ring and was stopped
// Per ARM_CPI_EVENT_* bit: [0] capture stopped, [2] VSYNC, [3] input FIFO overrun,
// [4] output FIFO overrun, [5] bus (AXI) error, [6] MIPI CSI-2 error.
extern volatile uint32_t g_fs_camera_events[FS_CAM_EVENT_BITS];
extern volatile uint32_t g_fs_camera_streaming; // 1 between a successful start and fs_camera_stop()
extern volatile uint32_t g_fs_camera_stage;     // fs_camera_stage_t of the last fs_camera_init()
extern volatile int32_t g_fs_camera_status;     // driver status code of that stage
extern volatile uint32_t g_fs_camera_first_frame_ms; // fs_camera_init() start to first frame

#ifdef __cplusplus
}
#endif
