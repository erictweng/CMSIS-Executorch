// fs_api.h — the contracts between Fingerspell's four work streams.
// Code against these; each stream ships its own fs_<name>.c and can be tested alone with stubs.
// Pure C99 so the decoder/UI logic also compiles and unit-tests on a laptop.
#pragma once
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FS_SIZE 128     // model input is FS_SIZE x FS_SIZE x 3
#define FS_CLASSES 24   // letters, in FS_LABELS order
#define FS_LABELS "ABCDEFGHIKLMNOPQRSTUVWXY"  // A..Y without J (Z also excluded)
#define FS_TEXT_MAX 128

// ---- Camera (Franco) ------------------------------------------------------
// Init once. Returns false (and prints why over UART) if the sensor isn't found.
bool fs_camera_init(void);
// Fill rgb with the latest frame's centered square crop, resized to FS_SIZE x FS_SIZE,
// RGB888, row-major HWC (rgb[(y*FS_SIZE + x)*3 + c]). Blocks until a new frame. false = error.
bool fs_camera_get_crop(uint8_t rgb[FS_SIZE * FS_SIZE * 3]);

// ---- Model (Eric) ---------------------------------------------------------
bool fs_model_init(void);
// Runs the NPU model on one crop; writes softmax probabilities. Returns false on error.
bool fs_classify(const uint8_t rgb[FS_SIZE * FS_SIZE * 3], float probs[FS_CLASSES]);

// ---- Decoder (Khangai) ----------------------------------------------------
typedef struct {
  float commit_threshold;   // default 0.60
  int commit_frames;        // default 3 consecutive frames
  uint32_t space_ms;        // default 1000: no confident letter this long -> space
  uint32_t release_ms;      // default 300: gap needed before the same letter commits again
} fs_decoder_cfg_t;

typedef enum { FS_EV_NONE = 0, FS_EV_LETTER, FS_EV_SPACE, FS_EV_BACKSPACE, FS_EV_CLEAR } fs_event_t;

typedef struct {
  fs_decoder_cfg_t cfg;
  char text[FS_TEXT_MAX];   // committed text; oldest characters drop off when full
  int len;
  char live;                // current top letter (or 0 if below threshold)
  float live_conf;          // its probability
  // private state (implementation detail; zero-initialised by fs_decoder_init)
  char cand, last_committed;
  int streak;
  bool released;
  uint32_t last_conf_ms;
} fs_decoder_t;

void fs_decoder_init(fs_decoder_t* d, const fs_decoder_cfg_t* cfg);  // cfg NULL -> defaults
// Feed one frame's probabilities. now_ms = monotonic milliseconds. Returns what happened.
fs_event_t fs_decoder_step(fs_decoder_t* d, const float probs[FS_CLASSES], uint32_t now_ms);
// Button / console input.
fs_event_t fs_decoder_backspace(fs_decoder_t* d);
fs_event_t fs_decoder_clear(fs_decoder_t* d);

// ---- Output / UI (P4) -----------------------------------------------------
typedef struct {
  float fps;
  float camera_ms, preprocess_ms, npu_ms, decode_ms, e2e_ms;
} fs_metrics_t;

bool fs_ui_init(void);
// Draw current state. Called once per frame; must return in < 10 ms.
// UART fallback prints one line: "TEXT|live|conf|fps|npu_ms" for tools/console_view.py.
void fs_ui_render(const fs_decoder_t* d, const fs_metrics_t* m, fs_event_t last_event);

#ifdef __cplusplus
}
#endif
