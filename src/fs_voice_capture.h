// fs_voice_capture.h: the DevKit-E8 I2S microphones (I2S3, Driver_SAI3) as a stream of
// 16 kHz mono 16-bit samples in a ring the main loop drains. The MFCC/KWS half
// (fs_voice.c) consumes it through fs_voice_capture_read(); the counters are read over
// J-Link by the board tests, so they are plain globals.
#pragma once
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FS_VOICE_RATE 16000  // Hz; SCLK = 2 x 32 x 16 kHz = 1.024 MHz = 76.8 MHz / 75 exactly
#define FS_VOICE_BLOCK 320   // samples per Receive(): 20 ms, one MFCC stride
// Ring of blocks the driver rotates through: 75 x 320 = 24,000 samples = 1.5 s, 48 KB of
// DTCM. The main loop may be up to ~1.48 s late (74 unread blocks) before samples drop.
#define FS_VOICE_RING_BLOCKS 75

// Which fs_voice_capture_init() stage failed (0 = none); the init prints it too.
typedef enum {
  FS_VOICE_OK = 0,
  FS_VOICE_ERR_CLOCK,     // SE services: enabling the 76.8 MHz HFOSCx2 the I2S SCLK is divided from
  FS_VOICE_ERR_INIT,      // Driver_SAI3 Initialize
  FS_VOICE_ERR_POWER,     // Driver_SAI3 PowerControl(ARM_POWER_FULL): module clock, IRQ
  FS_VOICE_ERR_CONFIG,    // Driver_SAI3 Control(ARM_SAI_CONFIGURE_RX ...): mode, size, rate
  FS_VOICE_ERR_RECEIVE,   // first Receive() of a block
  FS_VOICE_ERR_START,     // Control(ARM_SAI_CONTROL_RX, 1): receiver and SCLK on
  FS_VOICE_NOT_RUN = 0xFF,  // fs_voice_capture_init() has not been called
} fs_voice_stage_t;

// Bring up Driver_SAI3 (master, I2S protocol, mono, 16-bit, FS_VOICE_RATE) and start
// continuous reception into the ring. Prints one line either way; false means the app
// runs on without voice.
bool fs_voice_capture_init(void);

// Copy up to max_samples not-yet-read samples, oldest first, into dst and return how
// many were copied (0 when no complete block is pending). Non-blocking; main-loop
// context only, safe against the IRQ that fills the ring.
uint32_t fs_voice_capture_read(int16_t* dst, uint32_t max_samples);

// Stop reception and power the I2S3 block down.
void fs_voice_capture_stop(void);

// Telemetry (plain globals for J-Link mem32 reads).
extern volatile uint32_t g_fs_voice_stage;     // fs_voice_stage_t of the last init
extern volatile int32_t g_fs_voice_status;     // driver status code of that stage (or of a failed re-arm)
extern volatile uint32_t g_fs_voice_blocks;    // blocks received since init (50 per second)
extern volatile uint32_t g_fs_voice_overruns;  // blocks dropped because the reader fell FS_VOICE_RING_BLOCKS behind
extern volatile uint32_t g_fs_voice_errors;    // RX FIFO overflow / frame error events, failed re-arms
extern volatile uint32_t g_fs_voice_restarts;  // receptions re-armed from the main loop after a failed re-arm
extern volatile uint32_t g_fs_voice_running;   // 1 while a Receive() is armed

#ifdef __cplusplus
}
#endif
