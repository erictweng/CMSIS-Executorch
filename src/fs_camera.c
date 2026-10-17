// fs_camera.c: the DevKit-E8 MT9M114 camera behind the fs_api.h camera contract.
//
// The sensor sits on MIPI CSI-2; Alif's CPI driver configures it over I2C1 and
// captures its frames into SRAM by DMA. The image format is
// RTE_MT9M114_CAMERA_SENSOR_MIPI_IMAGE_CONFIG = 5 in RTE_Device.h: 320x320 RGB565.
//
// This drives the CPI driver directly rather than through the board's vStream
// VideoIn wrapper: the wrapper folds every CPI error into one end-of-stream event,
// and bring-up needs to know which error it is (FIFO overrun, bus error, CSI-2).
//
// Frames go into a ring of FS_CAM_FRAMES blocks that the CPI rotates through by
// itself (streaming mode: RTE_CPI_STREAMING_ENABLE 1, RTE_CPI_NUM_ACTIVE_FRAMEBUFFERS
// = FS_CAM_FRAMES); each VSYNC event carries the index of the block it just
// finished, which becomes the newest complete frame. (Re-arming CaptureVideo() on
// every VSYNC, as the vStream wrapper does, fails with BUSY while the CPI captures:
// in the legacy single-buffer mode the CPI then kept writing past the ring.)

#include "fs_camera.h"

#include <stdio.h>

#include "RTE_Components.h"
#include CMSIS_device_header

#include "Driver_CPI.h"
#include "RTE_Device.h"
#include "sys_utils.h"

extern ARM_DRIVER_CPI Driver_CPI;
static ARM_DRIVER_CPI* const cpi = &Driver_CPI;

// The frame ring lives in the bulk SRAM next to the model pools: the board layer's
// linker script routes .bss.ai_pool there (DTCM is too small for 3 x 200 KB) and
// the CPI's DMA can reach it. 32-byte aligned for D-cache maintenance.
#ifdef APP_POOL_SECTION
#define FS_CAM_BUF_ATTRIBUTES __attribute__((section(APP_POOL_SECTION), aligned(32)))
#else
#define FS_CAM_BUF_ATTRIBUTES __attribute__((aligned(32)))
#endif
// The ring and, right behind it in the same object, a canary the CPI must never write:
// a CPI misconfigured to write past its buffers (the legacy single-buffer mode did, at
// 5 MB/s) would run through SRAM into the rest of the address space within seconds and
// take the board down. fs_camera_latest_frame() checks the canary and stops the camera.
#define FS_CAM_CANARY_WORDS 1024
#define FS_CAM_CANARY 0xC0FFEE11u
static struct {
  uint8_t frames[FS_CAM_FRAMES][FS_CAM_FRAME_BYTES];
  uint32_t canary[FS_CAM_CANARY_WORDS];
} s_ring FS_CAM_BUF_ATTRIBUTES;
#define s_frames (s_ring.frames)
// CaptureVideo() in streaming mode takes an array of the block addresses.
static void* s_frame_ptrs[FS_CAM_FRAMES] = {s_ring.frames[0], s_ring.frames[1], s_ring.frames[2]};
volatile uint32_t g_fs_camera_overrun = 0;  // 1: the CPI wrote past the ring; camera stopped

#if (RTE_CPI_STREAMING_ENABLE != 1) || (RTE_CPI_NUM_ACTIVE_FRAMEBUFFERS != FS_CAM_FRAMES)
#error "fs_camera needs CPI streaming mode with FS_CAM_FRAMES buffers (RTE_Device.h)"
#endif

volatile uint32_t g_fs_camera_frames = 0;
volatile uint32_t g_fs_camera_overflows = 0;
volatile uint32_t g_fs_camera_errors = 0;
volatile uint32_t g_fs_camera_restarts = 0;
volatile uint32_t g_fs_camera_stage = FS_CAM_NOT_RUN;
volatile int32_t g_fs_camera_status = 0;
volatile uint32_t g_fs_camera_first_frame_ms = 0;
volatile uint32_t g_fs_camera_events[FS_CAM_EVENT_BITS] = {0};
volatile uint32_t g_fs_camera_streaming = 0;

static volatile uint32_t s_complete_idx = 0;  // newest complete block
static volatile uint32_t s_complete_seq = 0;  // its frame number (g_fs_camera_frames)
static volatile bool s_synced = false;        // first VSYNC seen (frame boundary known)
static volatile bool s_stopped = false;       // the CPI stopped on its own (error)
static uint32_t s_read_seq = 0;               // newest frame number handed to the app

static bool canary_intact(void) {
  SCB_InvalidateDCache_by_Addr(s_ring.canary, sizeof(s_ring.canary));
  for (uint32_t i = 0; i < FS_CAM_CANARY_WORDS; i += 64)
    if (s_ring.canary[i] != FS_CAM_CANARY) return false;
  return s_ring.canary[FS_CAM_CANARY_WORDS - 1] == FS_CAM_CANARY;
}

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
  if (event & ARM_CPI_EVENT_CAMERA_CAPTURE_STOPPED) s_stopped = true;

  if (event & ARM_CPI_EVENT_CAMERA_FRAME_VSYNC_DETECTED) {
    const uint32_t done = (event >> ARM_CPI_VSYNC_BUF_IDX_Pos) & 0x3U;  // block just finished
    if (!s_synced) {
      s_synced = true;  // start of the first frame: nothing complete yet
    } else if (done < FS_CAM_FRAMES) {
      if (s_complete_seq != s_read_seq) g_fs_camera_overflows++;  // previous one never read
      s_complete_idx = done;
      s_complete_seq = ++g_fs_camera_frames;
    }
  }
}

static int32_t start_capture(void) {
  s_synced = false;
  s_stopped = false;
  return cpi->CaptureVideo(s_frame_ptrs);
}

static void restart_capture(void) {
  (void)cpi->Stop();
  g_fs_camera_restarts++;
  (void)start_capture();
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
  g_fs_camera_restarts = 0;
  for (uint32_t b = 0; b < FS_CAM_EVENT_BITS; b++) g_fs_camera_events[b] = 0;
  s_complete_seq = 0;
  s_read_seq = 0;

  for (uint32_t i = 0; i < FS_CAM_CANARY_WORDS; i++) s_ring.canary[i] = FS_CAM_CANARY;
  SCB_CleanDCache_by_Addr(s_ring.canary, sizeof(s_ring.canary));
  g_fs_camera_overrun = 0;

  mipi_dphy_power_on();

  // CPI, MIPI CSI-2 and D-PHY setup, then the MT9M114: power/reset GPIOs, chip ID
  // (0x2481) and its register table over I2C1.
  int32_t status = cpi->Initialize(on_cpi_event);
  if (status != ARM_DRIVER_OK) return fail(FS_CAM_ERR_INIT, "CPI Initialize", status);
  status = cpi->PowerControl(ARM_POWER_FULL);
  if (status != ARM_DRIVER_OK) return fail(FS_CAM_ERR_INIT, "CPI PowerControl", status);
  (void)cpi->Stop();
  status = cpi->Control(CPI_CAMERA_SENSOR_CONFIGURE, 0U);
  if (status != ARM_DRIVER_OK) return fail(FS_CAM_ERR_SENSOR, "MT9M114 configure over I2C1", status);
  status = cpi->Control(CPI_CONFIGURE, 0U);
  if (status != ARM_DRIVER_OK) return fail(FS_CAM_ERR_INIT, "CPI configure", status);
  status = cpi->Control(CPI_EVENTS_CONFIGURE, ARM_CPI_EVENT_CAMERA_CAPTURE_STOPPED |
                                                  ARM_CPI_EVENT_CAMERA_FRAME_VSYNC_DETECTED |
                                                  FS_CAM_ERROR_EVENTS);
  if (status != ARM_DRIVER_OK) return fail(FS_CAM_ERR_INIT, "CPI events configure", status);

  status = start_capture();
  if (status != ARM_DRIVER_OK) return fail(FS_CAM_ERR_START, "CPI CaptureVideo", status);
  g_fs_camera_streaming = 1;

  // Init only counts as OK once a whole frame has arrived: the sensor answering on
  // I2C does not prove the CSI-2 link and the CPI DMA work.
  uint32_t waited_ms = 0;
  while (g_fs_camera_frames == 0 && waited_ms < FS_CAM_FIRST_FRAME_TIMEOUT_MS) {
    if (s_stopped && waited_ms > 0 && (waited_ms % FS_CAM_STALL_MS) == 0) restart_capture();
    sys_busy_loop_us(1000);
    waited_ms++;
  }
  if (g_fs_camera_frames == 0) {
    fs_camera_stop();
    return fail(FS_CAM_ERR_NO_FRAME, "first frame (timeout)", (int32_t)waited_ms);
  }
  g_fs_camera_first_frame_ms = waited_ms;
  if (!canary_intact()) {
    fs_camera_stop();
    g_fs_camera_overrun = 1;
    return fail(FS_CAM_ERR_NO_FRAME, "frame ring overrun (CPI wrote past its buffers)", 0);
  }

  g_fs_camera_stage = FS_CAM_OK;
  g_fs_camera_status = 0;
  printf("camera init OK %dx%d RGB565 (first frame after %lu ms, %lu error events)\n", FS_CAM_WIDTH,
         FS_CAM_HEIGHT, (unsigned long)waited_ms, (unsigned long)g_fs_camera_errors);
  return true;
}

const uint16_t* fs_camera_latest_frame(uint32_t timeout_ms, uint32_t* seq) {
  if (!g_fs_camera_streaming) return NULL;
  if (!canary_intact()) {
    fs_camera_stop();
    g_fs_camera_overrun = 1;
    printf("camera STOPPED: the CPI wrote past its frame ring\n");
    return NULL;
  }
  uint32_t ticks = 0, stall_ticks = 0;  // 100 us each
  while (s_complete_seq == s_read_seq) {
    if (ticks >= timeout_ms * 10U) return NULL;
    // A CPI error can stop the capture; restart it rather than wait forever.
    if (s_stopped || stall_ticks >= FS_CAM_STALL_MS * 10U) {
      restart_capture();
      stall_ticks = 0;
    }
    sys_busy_loop_us(100);
    ticks++;
    stall_ticks++;
  }
  __disable_irq();
  const uint32_t idx = s_complete_idx;
  s_read_seq = s_complete_seq;
  __enable_irq();
  uint8_t* block = s_frames[idx];
  SCB_InvalidateDCache_by_Addr(block, FS_CAM_FRAME_BYTES);
  if (seq != NULL) *seq = s_read_seq;
  return (const uint16_t*)block;
}

// RGB565 (5-6-5, R in the top bits) to 8 bits per channel, replicating the top bits
// into the bottom so full scale maps to 255.
static inline void rgb565_to_888(uint16_t v, unsigned* r, unsigned* g, unsigned* b) {
  const unsigned r5 = (v >> 11) & 0x1Fu, g6 = (v >> 5) & 0x3Fu, b5 = v & 0x1Fu;
  *r = (r5 << 3) | (r5 >> 2);
  *g = (g6 << 2) | (g6 >> 4);
  *b = (b5 << 3) | (b5 >> 2);
}

void fs_camera_crop(const uint16_t* frame, uint8_t rgb[FS_SIZE * FS_SIZE * 3]) {
  const int x0 = (FS_CAM_WIDTH - FS_CAM_BOX) / 2, y0 = (FS_CAM_HEIGHT - FS_CAM_BOX) / 2;
  for (int y = 0; y < FS_SIZE; ++y) {
    const uint16_t* row0 = frame + (y0 + 2 * y) * FS_CAM_WIDTH + x0;
    const uint16_t* row1 = row0 + FS_CAM_WIDTH;
    uint8_t* out = rgb + y * FS_SIZE * 3;
    for (int x = 0; x < FS_SIZE; ++x) {
      unsigned r = 0, g = 0, b = 0, cr, cg, cb;
      const uint16_t px[4] = {row0[2 * x], row0[2 * x + 1], row1[2 * x], row1[2 * x + 1]};
      for (int k = 0; k < 4; ++k) {
        rgb565_to_888(px[k], &cr, &cg, &cb);
        r += cr;
        g += cg;
        b += cb;
      }
      out[3 * x + 0] = (uint8_t)((r + 2) / 4);
      out[3 * x + 1] = (uint8_t)((g + 2) / 4);
      out[3 * x + 2] = (uint8_t)((b + 2) / 4);
    }
  }
}

bool fs_camera_get_crop(uint8_t rgb[FS_SIZE * FS_SIZE * 3]) {
  const uint16_t* frame = fs_camera_latest_frame(1000, NULL);
  if (frame == NULL) return false;
  fs_camera_crop(frame, rgb);
  return true;
}

void fs_camera_stop(void) {
  if (!g_fs_camera_streaming) return;
  (void)cpi->Stop();
  (void)cpi->PowerControl(ARM_POWER_OFF);
  (void)cpi->Uninitialize();
  g_fs_camera_streaming = 0;
}
