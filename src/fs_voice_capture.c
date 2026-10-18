// fs_voice_capture.c: the DevKit-E8 I2S microphones behind fs_voice_capture.h.
//
// The two I2S MEMS mics (SPH0645LM4H-class, 24-bit MSB-aligned words in 32-bit slots)
// hang on I2S3; from the M55-HP that is "MIC Input" (board_defs.h) and the pack's pins.h
// already muxes P8_6 SCLK, P8_7 WS, P9_0 SDI. This drives Driver_SAI3 directly rather
// than through the vStream AudioIn wrapper: the wrapper defaults to the HE-only LPI2S,
// receives stereo-interleaved, and hides which driver call failed.
//
// Mode: master (the I2S3 block generates SCLK/WS from the 76.8 MHz HFOSCx2 through the
// CLKCTL divider: 76.8 MHz / 75 = 1.024 MHz = 2 x 32 x 16 kHz, exact), I2S protocol,
// 16-bit data (the top 16 bits of the mic's 24-bit word: the controller takes the first
// 16 SCLKs after WS, which is what we want as PCM), mono (the driver reads both FIFO
// halves and keeps the left one, so Receive(buf, n) counts n mono samples), IRQ mode
// (RTE_I2S3_DMA_ENABLE 0: the FIFO interrupt at trigger level 7 copies 8 frames at a
// time, every 0.5 ms).
//
// The ring is plain .bss, which the board layer's linker script keeps in DTCM. DTCM is
// not cached and the driver copies from the FIFO with the CPU, so no cache maintenance
// is needed here. If the ring ever moves to SRAM0 (e.g. for a DMA path), invalidate each
// block before reading it the way fs_camera.c does.

#include "fs_voice_capture.h"

#include <stdio.h>
#include <string.h>

#include "RTE_Components.h"
#include CMSIS_device_header

volatile uint32_t g_fs_voice_stage = FS_VOICE_NOT_RUN;
volatile int32_t g_fs_voice_status = 0;
volatile uint32_t g_fs_voice_blocks = 0;
volatile uint32_t g_fs_voice_overruns = 0;
volatile uint32_t g_fs_voice_errors = 0;
volatile uint32_t g_fs_voice_restarts = 0;
volatile uint32_t g_fs_voice_running = 0;

#ifdef RTE_Drivers_SAI

#include "Driver_SAI.h"
#include "RTE_Device.h"
#include "se_services_port.h"
#include "soc_features.h"

#if (RTE_I2S3 != 1) || (RTE_I2S3_DMA_ENABLE != 0) || (RTE_I2S3_WSS_CLOCK_CYCLES != 2)
#error "fs_voice_capture needs Driver_SAI3 in IRQ mode with 32-cycle word select (RTE_Device.h)"
#endif

extern ARM_DRIVER_SAI Driver_SAI3;
static ARM_DRIVER_SAI* const sai = &Driver_SAI3;

// The driver wants 4-byte aligned buffers for a 32-cycle word select; 32 keeps the
// option of cache maintenance should the ring move out of DTCM.
static int16_t s_mic_ring[FS_VOICE_RING_BLOCKS][FS_VOICE_BLOCK] __attribute__((aligned(32)));

// Block sequence numbers (not indices): g_fs_voice_blocks is the number of completed
// blocks, so the driver fills block g_fs_voice_blocks % RING. The reader has consumed
// s_rd_seq whole blocks plus s_rd_off samples of block s_rd_seq. Both reader fields are
// written by the reader under disabled IRQs and by the callback when it drops the oldest
// unread block; the reader re-checks s_rd_seq after each copy to catch that.
static volatile uint32_t s_rd_seq = 0;
static volatile uint32_t s_rd_off = 0;
static bool s_started = false;  // between a successful init and stop (re-arm allowed)

// Arm the next Receive(). IRQ or main-loop context; the caller holds IRQs off in the
// latter case. The block to fill is g_fs_voice_blocks % RING; if the reader still owns
// it (RING unread blocks), drop the oldest unread block instead of waiting.
static void arm_next(void) {
  const uint32_t wr = g_fs_voice_blocks;
  if (wr - s_rd_seq >= FS_VOICE_RING_BLOCKS) {
    g_fs_voice_overruns++;
    s_rd_seq = wr - (FS_VOICE_RING_BLOCKS - 1);
    s_rd_off = 0;
  }
  const int32_t status = sai->Receive(s_mic_ring[wr % FS_VOICE_RING_BLOCKS], FS_VOICE_BLOCK);
  if (status == ARM_DRIVER_OK) {
    g_fs_voice_running = 1;
  } else {
    g_fs_voice_status = status;
    g_fs_voice_errors++;
    g_fs_voice_running = 0;  // fs_voice_capture_read() retries from the main loop
  }
}

// I2S3 interrupt context. Driver_I2S.c clears rx_busy before raising RECEIVE_COMPLETE,
// so a Receive() from here is accepted. No printf here.
static void on_sai_event(uint32_t event) {
  if (event & ARM_SAI_EVENT_RECEIVE_COMPLETE) {
    g_fs_voice_blocks++;
    arm_next();
  }
  if (event & (ARM_SAI_EVENT_RX_OVERFLOW | ARM_SAI_EVENT_FRAME_ERROR)) {
    g_fs_voice_errors++;
    // A FIFO overflow loses samples but the transfer in flight continues (the driver
    // keeps rx_busy and only masks the overflow interrupt until the next Receive()).
    // Only if the driver did give up re-arm, else a second Receive() would return BUSY.
    if (!sai->GetStatus().rx_busy) arm_next();
  }
}

static bool fail(fs_voice_stage_t stage, const char* what, int32_t status) {
  g_fs_voice_stage = stage;
  g_fs_voice_status = status;
  printf("voice capture init FAILED at %s (stage %d, status %ld)\n", what, (int)stage, (long)status);
  (void)sai->PowerControl(ARM_POWER_OFF);
  (void)sai->Uninitialize();
  return false;
}

bool fs_voice_capture_init(void) {
  g_fs_voice_blocks = 0;
  g_fs_voice_overruns = 0;
  g_fs_voice_errors = 0;
  g_fs_voice_restarts = 0;
  g_fs_voice_running = 0;
  s_rd_seq = 0;
  s_rd_off = 0;
  s_started = false;

#if SOC_FEAT_CLK76P8M_CLK_ENABLE
  // The I2S SCLK divider runs off the 76.8 MHz HFOSCx2, which the Secure Enclave gates;
  // board main.c only asks for HFOSC and the 100 MHz clock. Same call as the pack's
  // demo_i2s.c; needs se_services_port_init() to have run (board main.c does that).
  uint32_t se_error = 0;
  const uint32_t se_status =
      SERVICES_clocks_enable_clock(se_services_s_handle, CLKEN_HFOSCx2, true, &se_error);
  if (se_status != SERVICES_REQ_SUCCESS) {
    g_fs_voice_stage = FS_VOICE_ERR_CLOCK;
    g_fs_voice_status = (int32_t)se_status;
    printf("voice capture init FAILED at SE HFOSCx2 clock enable (stage %d, status %ld, se %lu)\n",
           (int)FS_VOICE_ERR_CLOCK, (long)se_status, (unsigned long)se_error);
    return false;
  }
#endif

  int32_t status = sai->Initialize(on_sai_event);
  if (status != ARM_DRIVER_OK) return fail(FS_VOICE_ERR_INIT, "SAI3 Initialize", status);
  status = sai->PowerControl(ARM_POWER_FULL);
  if (status != ARM_DRIVER_OK) return fail(FS_VOICE_ERR_POWER, "SAI3 PowerControl", status);

  // arg1 is the frame length in bits and must equal 2 x the data size (Driver_I2S.c
  // compares the raw value, so 32, not ARM_SAI_FRAME_LENGTH(32)); arg2 the sample rate.
  // MCLK pin inactive (the default) selects the internal 76.8 MHz source and sets the
  // divider for arg2. No ARM_SAI_FRAME_LENGTH, companding or polarity bits: unsupported.
  status = sai->Control(ARM_SAI_CONFIGURE_RX | ARM_SAI_MODE_MASTER | ARM_SAI_ASYNCHRONOUS |
                            ARM_SAI_PROTOCOL_I2S | ARM_SAI_DATA_SIZE(16) | ARM_SAI_MONO_MODE,
                        2 * 16, FS_VOICE_RATE);
  if (status != ARM_DRIVER_OK) return fail(FS_VOICE_ERR_CONFIG, "SAI3 configure RX", status);

  // Arm the first block before enabling the receiver (the vStream AudioIn order), so
  // the FIFO never fills with nobody draining it.
  status = sai->Receive(s_mic_ring[0], FS_VOICE_BLOCK);
  if (status != ARM_DRIVER_OK) return fail(FS_VOICE_ERR_RECEIVE, "SAI3 Receive", status);
  g_fs_voice_running = 1;
  status = sai->Control(ARM_SAI_CONTROL_RX, 1, 0);
  if (status != ARM_DRIVER_OK) {
    g_fs_voice_running = 0;
    (void)sai->Control(ARM_SAI_ABORT_RECEIVE, 0, 0);
    return fail(FS_VOICE_ERR_START, "SAI3 control RX enable", status);
  }
  s_started = true;

  g_fs_voice_stage = FS_VOICE_OK;
  g_fs_voice_status = 0;
  printf("voice capture init OK (I2S3 16 kHz mono 16-bit)\n");
  return true;
}

uint32_t fs_voice_capture_read(int16_t* dst, uint32_t max_samples) {
  if (!s_started) return 0;
  // A Receive() that failed inside the callback leaves the stream stopped; retry here,
  // where a stuck transfer can be aborted first.
  if (!g_fs_voice_running) {
    __disable_irq();
    (void)sai->Control(ARM_SAI_ABORT_RECEIVE, 0, 0);
    g_fs_voice_restarts++;
    arm_next();
    __enable_irq();
  }

  uint32_t n = 0;
  while (n < max_samples) {
    __disable_irq();
    const uint32_t seq = s_rd_seq, off = s_rd_off, wr = g_fs_voice_blocks;
    __enable_irq();
    if (seq == wr) break;  // block seq is still being filled

    uint32_t take = FS_VOICE_BLOCK - off;
    if (take > max_samples - n) take = max_samples - n;
    memcpy(dst + n, &s_mic_ring[seq % FS_VOICE_RING_BLOCKS][off], take * sizeof(int16_t));

    // The callback only re-uses block seq after moving s_rd_seq past it, so an unchanged
    // s_rd_seq proves the copy was not raced; otherwise discard it and resync.
    __disable_irq();
    const bool intact = (s_rd_seq == seq);
    if (intact) {
      if (off + take == FS_VOICE_BLOCK) {
        s_rd_seq = seq + 1;
        s_rd_off = 0;
      } else {
        s_rd_off = off + take;
      }
    }
    __enable_irq();
    if (intact) n += take;
  }
  return n;
}

void fs_voice_capture_stop(void) {
  if (!s_started) return;
  s_started = false;
  (void)sai->Control(ARM_SAI_CONTROL_RX, 0, 0);  // receiver, SCLK and RX IRQ off, rx_busy cleared
  (void)sai->Control(ARM_SAI_ABORT_RECEIVE, 0, 0);
  g_fs_voice_running = 0;
  (void)sai->PowerControl(ARM_POWER_OFF);
  (void)sai->Uninitialize();
  // The HFOSCx2 clock stays on: cheap, and other blocks may be using it.
}

#else  // no CMSIS Driver:SAI component in this build

bool fs_voice_capture_init(void) {
  printf("voice capture: no SAI driver in this build\n");
  return false;
}

uint32_t fs_voice_capture_read(int16_t* dst, uint32_t max_samples) {
  (void)dst;
  (void)max_samples;
  return 0;
}

void fs_voice_capture_stop(void) {}

#endif
