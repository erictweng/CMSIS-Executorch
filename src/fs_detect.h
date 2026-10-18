// fs_detect.h: the palm detector's CPU side (BlazePalm, MediaPipe palm_detection_lite,
// 192x192) for the DevKit-E8: the 320x320 RGB565 camera frame to the detector's float
// input, the anchor table, the SSD decode, the palm box -> hand box rule, temporal
// smoothing, and the hand-box crop that feeds the letter classifier.
//
// Pure C99 without hardware so it compiles and unit-tests on a laptop
// (tests/host/test_fs_detect.c); the Python reference is ml/blazepalm.py.
#pragma once
#include <stdbool.h>
#include <stdint.h>

#include "fs_api.h"

#ifdef __cplusplus
extern "C" {
#endif

#define FS_DET_SIZE 192        // detector input, FS_DET_SIZE x FS_DET_SIZE x 3 int8 (quantized [0, 1])
#define FS_DET_ANCHORS 2016    // stride 8: 24x24x2 (1152), then stride 16: 12x12x6 (864)
#define FS_DET_COORDS 18       // cx cy w h, then 7 keypoints (x y), in input pixels
#define FS_DET_KEYPOINTS 7
#define FS_DET_THRESHOLD 0.5f  // sigmoid(score logit) >= 0.5, i.e. logit >= 0
#define FS_DET_BOX_SHIFT_Y (-0.5f)  // palm box -> hand box: shift by -0.5 x height (toward the fingers)
#define FS_DET_BOX_SCALE 2.6f       // and grow 2.6x, square (no rotation on the board)
#define FS_DET_MAX_SIDE 640.0f      // hand boxes larger than this are clamped (2x the frame)
#define FS_DET_EMA 0.5f             // box smoothing: new = EMA x detection + (1 - EMA) x previous
#define FS_DET_HOLD_FRAMES 5        // keep the last box this many missed frames, then "no hand"

// Camera frame geometry (fs_camera.h's FS_CAM_WIDTH/HEIGHT; repeated so the host build
// needs no camera header).
#define FS_DET_FRAME 320

typedef struct {
  float x0, y0, x1, y1;  // pixels; a hand box may run off the frame (the crop pads with black)
} fs_box_t;

// int8 quantization of a method input or output: real = (q - zero_point) * scale. The
// values come from model_params.h (DETECT_INPUT0_SCALE, ...), written by create_ai_layer.py.
typedef struct {
  float scale;
  int zero_point;
} fs_quant_t;

typedef struct {
  fs_box_t palm;                        // palm box in detector-input pixels (192)
  float keypoints[FS_DET_KEYPOINTS][2];  // wrist, index/middle/ring/pinky MCP, ... (192 px)
  float score;                          // sigmoid of the best anchor's logit
  int anchor;                           // index of that anchor
} fs_detection_t;

typedef struct {
  bool hand;       // a box is available (detected this frame or held from a previous one)
  bool fresh;      // the detector fired this frame
  int misses;      // consecutive frames without a detection
  float score;     // last detection's score
  fs_box_t box;    // smoothed square hand box in frame pixels
} fs_track_t;

// Anchor i's centre in [0, 1] (MediaPipe ssd_anchors_calculator for the 192 lite model:
// strides 8/16/16/16, offset 0.5, fixed anchor size 1x1); i in [0, FS_DET_ANCHORS).
void fs_detect_anchor(int i, float* cx, float* cy);

// 320x320 RGB565 frame -> 192x192x3 NHWC int8: exact 5:3 box filter (each output pixel
// averages a 5/3 x 5/3 input area), the [0, 1] value quantized with `input`.
void fs_detect_preprocess(const uint16_t* frame, int8_t* out, fs_quant_t input);

// The raw int8 head (regressors FS_DET_ANCHORS x FS_DET_COORDS with `rq`, score logits
// FS_DET_ANCHORS with `lq`) -> the best palm. Returns false when no anchor reaches the
// threshold (*det still holds the best anchor).
bool fs_detect_decode(const int8_t* regressors, fs_quant_t rq, const int8_t* logits, fs_quant_t lq,
                      fs_detection_t* det);

// Palm box (192 px) -> square hand box in frame pixels (shift, scale, square, clamp side).
fs_box_t fs_detect_hand_box(const fs_box_t* palm, float input_to_frame);

// Temporal smoothing and hold: feed every frame; `hit` with its box when the detector fired.
void fs_track_reset(fs_track_t* t);
void fs_track_update(fs_track_t* t, bool hit, const fs_box_t* box, float score);

// Crop `box` (frame pixels, square) from the RGB565 frame, box-filter it to
// FS_SIZE x FS_SIZE RGB888 (HWC), black where the box runs off the frame.
void fs_detect_crop(const uint16_t* frame, const fs_box_t* box, uint8_t rgb[FS_SIZE * FS_SIZE * 3]);

#ifdef __cplusplus
}
#endif
