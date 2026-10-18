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
// Temporal averaging and hysteresis (fs_decoder.c): the 24 probabilities are averaged over
// frames (exponential, weight prob_alpha on the new frame, started from the first frame so a
// steady hand commits without warm-up, reset when the hand is lost) and every rule below
// reads the averaged stream. A letter needs commit_threshold to start a streak and on each
// frame the streak counts; once it is the candidate it stays the live letter down to
// hold_threshold, so a confidence dip is not a gap. The same letter commits again only after
// the hand was lost for release_ms or another letter committed in between. A space comes
// from the hand leaving the frame for space_ms; a pause with the hand in view never makes
// one, so slow or hesitant signing does not break words.
typedef struct {
  float commit_threshold;   // default 0.50
  float hold_threshold;     // default 0.40: the live candidate stays while >= this (0: = commit_threshold)
  float prob_alpha;         // default 0.4: weight of the new frame in the probability average (0 / 1: off)
  int commit_frames;        // default 2 consecutive frames at or above commit_threshold
  uint32_t space_ms;        // default 600: no hand this long -> space
  uint32_t release_ms;      // default 300: no hand this long before the same letter commits again
                            // (a bounce of the hand out of view between release_ms and space_ms
                            // repeats a letter; longer makes a space)
  // Motion letters (fs_decoder_step_motion): D and I are held as "pending"; the fingertip's
  // displacement from where the hold began decides D vs Z and I vs J. Z / J commit the
  // moment the displacement reaches motion_threshold; D / I commit when the signer moves on.
  float motion_threshold;   // default 1.0: displacement in hand sizes that makes it Z / J
  float motion_deadband;    // default 0.06: displacement below this (hand sizes) reads as 0
  uint32_t pending_timeout_ms;  // default 1500: a pending letter held this long resolves as is
  int pending_drop_frames;  // default 3: frames of another/no letter that end the pending hold
} fs_decoder_cfg_t;

typedef enum { FS_EV_NONE = 0, FS_EV_LETTER, FS_EV_SPACE, FS_EV_BACKSPACE, FS_EV_CLEAR, FS_EV_PENDING } fs_event_t;

// Per-frame motion input for fs_decoder_step_motion(): fingertips and the hand size in the
// same units (frame pixels; hand_size = wrist -> middle-finger-MCP distance); valid = false
// when there is no hand this frame.
typedef struct {
  bool valid;
  float index_tip[2];
  float pinky_tip[2];
  float hand_size;
} fs_motion_t;

typedef struct {
  fs_decoder_cfg_t cfg;
  char text[FS_TEXT_MAX];   // committed text; oldest characters drop off when full
  int len;
  char live;                // current top letter (or 0 if below threshold)
  float live_conf;          // its averaged probability
  char pending;             // 'D' or 'I' held until the hand moves on (0 = none)
  float pending_path;       // the fingertip's largest displacement from the hold's start, in hand sizes
  // private state (implementation detail; zero-initialised by fs_decoder_init)
  float avg[FS_CLASSES];    // the averaged probabilities (what the rules read)
  bool avg_valid;
  char cand, last_committed;
  int streak;
  bool released;
  uint32_t last_hand_ms;
  uint32_t pending_since_ms;
  int pending_drop;
  float start_tip[2];       // where the fingertip was when the hold began
  bool start_tip_valid;
} fs_decoder_t;

void fs_decoder_init(fs_decoder_t* d, const fs_decoder_cfg_t* cfg);  // cfg NULL -> defaults
// Feed one frame's probabilities. now_ms = monotonic milliseconds. Returns what happened.
fs_event_t fs_decoder_step(fs_decoder_t* d, const float probs[FS_CLASSES], uint32_t now_ms);
// The same with the hand's motion: D and I become pending (FS_EV_PENDING) instead of
// committing, and commit as D / Z or I / J when the signer moves on, times out, or the hand
// is lost. motion NULL behaves like fs_decoder_step().
fs_event_t fs_decoder_step_motion(fs_decoder_t* d, const float probs[FS_CLASSES], const fs_motion_t* motion,
                                  uint32_t now_ms);
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
