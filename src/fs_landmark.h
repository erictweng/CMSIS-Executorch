// fs_landmark.h: the hand landmark model's CPU side (MediaPipe hand_landmark_lite, 224x224)
// and the keypoint -> letter MLP: the rotated hand crop from the palm detection, the int8
// landmark decode back into frame pixels, the MLP features and the letter probabilities.
//
// Pure C99 without hardware so it compiles and unit-tests on a laptop
// (tests/host/test_fs_landmark.c); the Python reference is ml/handlandmark.py.
#pragma once
#include <stdbool.h>
#include <stdint.h>

#include "fs_api.h"
#include "fs_detect.h"

#ifdef __cplusplus
extern "C" {
#endif

#define FS_LM_SIZE 224          // landmark model input, FS_LM_SIZE x FS_LM_SIZE x 3 int8
#define FS_LM_COUNT 21          // landmarks, 3 values each (x, y in input pixels, z alike)
#define FS_LM_FEATURES (FS_LM_COUNT * 3)
#define FS_LM_WRIST 0
#define FS_LM_MIDDLE_MCP 9
#define FS_LM_CROP_SHIFT 0.5f   // crop centre: palm centre + 0.5 x palm size toward the fingers
#define FS_LM_CROP_SCALE 2.6f   // crop side = 2.6 x palm size
#define FS_LM_PRESENCE 0.5f     // presence below this: no hand in the crop
#define FS_LM_MAX_SIDE 640.0f

// The rotated square crop in frame pixels: centre, side, and the crop's unit vectors
// (right = crop +x, down = crop +y) so that the fingers point up in the crop.
typedef struct {
  float cx, cy, side;
  float rx, ry;  // right
  float dx, dy;  // down
} fs_lm_crop_t;

typedef struct {
  float xyz[FS_LM_COUNT][3];  // frame pixels (z in the same units, relative)
  float presence;             // 0..1
  float right;                // 0..1, handedness (1 = right hand as the model sees it)
} fs_landmarks_t;

// Crop geometry from the palm detection (192 px) scaled by input_to_frame (320 / 192).
fs_lm_crop_t fs_landmark_crop_frame(const fs_detection_t* det, float input_to_frame);

// Resample the rotated crop from the 320x320 RGB565 frame (bilinear, black off-frame) to
// FS_LM_SIZE x FS_LM_SIZE x 3 NHWC int8, [0, 1] quantized with `input`.
void fs_landmark_crop(const uint16_t* frame, const fs_lm_crop_t* crop, int8_t* out, fs_quant_t input);

// The model's int8 outputs -> landmarks in frame pixels, presence and handedness.
void fs_landmark_decode(const int8_t lm63[FS_LM_FEATURES], fs_quant_t lq, int8_t presence, fs_quant_t pq,
                        int8_t handedness, fs_quant_t hq, const fs_lm_crop_t* crop, fs_landmarks_t* out);

// MLP inputs: wrist at the origin, wrist -> middle-MCP distance = 1, left hands mirrored.
void fs_landmark_features(const fs_landmarks_t* lm, float f[FS_LM_FEATURES]);

// Letter probabilities from the MLP (W1 hidden x 63, b1 hidden, W2 FS_CLASSES x hidden, b2).
void fs_landmark_letters(const float f[FS_LM_FEATURES], const float* W1, const float* b1, const float* W2,
                         const float* b2, int hidden, float probs[FS_CLASSES]);

#ifdef __cplusplus
}
#endif
