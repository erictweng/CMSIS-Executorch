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
#define FS_LM_PRESENCE 0.5f     // presence a fresh hand needs (and a held one drops below FS_LM_PRESENCE_HOLD)
#define FS_LM_MAX_SIDE 640.0f

// Which hand the signer uses. The model's per-frame handedness score flips on crossed or
// closed shapes (R, H: 7 % of frames, with 10 s stretches of "left" on a right hand), and
// every flip mirrors the 21 x features the MLP sees. Fixed RIGHT / LEFT ignores the score;
// AUTO mirrors by it (score < 0.5 = left), which is what the datasets are built with.
#define FS_LM_HANDEDNESS_AUTO 0
#define FS_LM_HANDEDNESS_RIGHT 1
#define FS_LM_HANDEDNESS_LEFT 2
#ifndef FS_LM_HANDEDNESS
#define FS_LM_HANDEDNESS FS_LM_HANDEDNESS_RIGHT
#endif

// Presence gate with a hold: a fresh hand needs presence >= FS_LM_PRESENCE; a tracked hand
// survives up to FS_LM_HOLD_FRAMES consecutive frames of presence in
// [FS_LM_PRESENCE_HOLD, FS_LM_PRESENCE) (a far or blurred hand scores 0.2-0.4 for a frame
// or two), and is lost on a frame below FS_LM_PRESENCE_HOLD or when the hold runs out.
// The averages and the decoder's hand state stay alive through a hold.
#define FS_LM_PRESENCE_HOLD 0.25f
#define FS_LM_HOLD_FRAMES 5
typedef struct {
  bool hand;  // a hand is being tracked
  int held;   // consecutive held frames so far
} fs_lm_presence_t;
void fs_landmark_presence_reset(fs_lm_presence_t* g);
// Feed every frame's presence; true when there is a hand this frame (fresh or held).
bool fs_landmark_presence_gate(fs_lm_presence_t* g, float presence);

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

// Temporal average of the crop geometry (exponential, weight alpha on the new frame): the
// centre, side and up vector are averaged, the up vector renormalised and right / down
// rebuilt from it, so the landmark model sees a steady picture while the detection jitters.
// *valid false: the first frame copies raw. Reset (*valid = false) when the hand is lost.
#define FS_LM_CROP_SMOOTH 0.5f
void fs_landmark_crop_smooth(fs_lm_crop_t* avg, bool* valid, const fs_lm_crop_t* raw, float alpha);

// Resample the rotated crop from the 320x320 RGB565 frame (bilinear, black off-frame) to
// FS_LM_SIZE x FS_LM_SIZE x 3 NHWC int8, [0, 1] quantized with `input`.
void fs_landmark_crop(const uint16_t* frame, const fs_lm_crop_t* crop, int8_t* out, fs_quant_t input);

// The model's int8 outputs -> landmarks in frame pixels, presence and handedness.
void fs_landmark_decode(const int8_t lm63[FS_LM_FEATURES], fs_quant_t lq, int8_t presence, fs_quant_t pq,
                        int8_t handedness, fs_quant_t hq, const fs_lm_crop_t* crop, fs_landmarks_t* out);

// MLP inputs: wrist at the origin, wrist -> middle-MCP distance = 1, left hands mirrored to
// right. `handedness` is one of FS_LM_HANDEDNESS_*; fs_landmark_features() uses the build's
// FS_LM_HANDEDNESS.
void fs_landmark_features_as(const fs_landmarks_t* lm, int handedness, float f[FS_LM_FEATURES]);
void fs_landmark_features(const fs_landmarks_t* lm, float f[FS_LM_FEATURES]);

// Temporal average of the MLP features across frames (exponential, weight FS_LM_SMOOTH on the
// new frame): steadier letters and confidences. Reset it when the hand is lost.
#define FS_LM_SMOOTH 0.5f
typedef struct {
  float f[FS_LM_FEATURES];
  bool valid;
} fs_lm_smooth_t;
void fs_landmark_smooth_reset(fs_lm_smooth_t* s);
void fs_landmark_smooth(fs_lm_smooth_t* s, const float f[FS_LM_FEATURES], float alpha, float out[FS_LM_FEATURES]);

// Letter probabilities from the MLP (W1 hidden x 63, b1 hidden, W2 FS_CLASSES x hidden, b2).
void fs_landmark_letters(const float f[FS_LM_FEATURES], const float* W1, const float* b1, const float* W2,
                         const float* b2, int hidden, float probs[FS_CLASSES]);

#ifdef __cplusplus
}
#endif
