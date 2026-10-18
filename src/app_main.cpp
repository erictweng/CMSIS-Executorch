// Fingerspell — app_main.cpp for Arm-Examples/CMSIS-Executorch@hackathon (replaces src/app_main.cpp).
//
// Pipeline per frame (documentation/landmark-plan.md): palm detector ("detect", 192x192, NPU)
// -> hand box -> rotated hand crop -> hand landmark model ("landmark", 224x224, NPU) -> 21
// keypoints -> keypoint MLP (CPU) -> letter probabilities -> decoder. The letter decision
// never sees a pixel, so background, lighting and skin tone do not reach it.
//
// Boot: the baked-in 128x128 test crop (test_image.h) is pasted into a synthetic 320x320
// frame and run through the same pipeline; the top letter must be kTestExpected
// (Test_result: PASS/FAIL). On a board with the camera (DevKit-E8) it then probes the camera
// for 3 s (step 4), prints the first centre crop as text art, and runs the live loop forever.
// Without a camera (FVP, AppKit-E8) main() returns after the test as before.
//
// Input contracts (model/model.py, quantized IO, parameters in model_params.h): detect
// 1x192x192x3 int8, raw SSD head int8; landmark 1x224x224x3 int8, outputs int8 landmarks
// (63), presence (1), handedness (1), world landmarks (63, unused).

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>

#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/platform/runtime.h>

#include "arm_embedded_module.hpp"
#include "fs_detect.h"
#include "fs_landmark.h"
#include "model_params.h"  // FS_DETECT_*, FS_CLASSIFY_SIZE, FS_CLASSES from model/model.py
#include "model_pte.h"
#include "test_image.h"  // kTestImage[128*128*3] (uint8 RGB), kTestExpected (char)

#include "RTE_Components.h"
#ifdef FINGERSPELL_LCD
#include "fs_display.h"
#endif
#ifdef RTE_VSTREAM_VIDEOIN  // board layer with the MT9M114 camera (DevKit-E8)
#include CMSIS_device_header
#include "fs_camera.h"
#include "sys_clocks.h"  // SystemREFClock
#include "sys_utils.h"   // sys_busy_loop_init_ns
#define FS_HAVE_CAMERA 1
#endif
#ifdef RTE_Drivers_ICM42670P  // the DevKit-E8's accelerometer: the camera's roll relative to gravity
#include "fs_imu.h"
#define FS_HAVE_IMU 1
#endif
#if defined(FINGERSPELL_LCD) && defined(RTE_Drivers_IO)  // the DevKit-E8 joystick edits the text
#include "fs_input.h"
#include "fs_joystick.h"
#define FS_HAVE_JOYSTICK 1
#endif
#include "fs_orient.h"
#include "fs_orient_cal.h"

using arm::embedded::EmbeddedModule;
using executorch::aten::DimOrderType;
using executorch::aten::ScalarType;
using executorch::aten::SizesType;
using executorch::aten::Tensor;
using executorch::aten::TensorImpl;
using executorch::extension::BufferDataLoader;
using executorch::runtime::EValue;
using executorch::runtime::MemoryAllocator;

// Set from the debugger (tests/fs_tools.py) to make the live loop print the next classifier
// crop as hex between CROPDUMP BEGIN/END lines; the loop clears it.
extern "C" {
volatile uint32_t g_fs_dump_request = 0;
volatile uint32_t g_fs_loop_frames = 0;  // live-loop iterations (frames through the detector)
// Per-frame telemetry for the debugger (tests/fs_tools.py read_word): the UART is not always
// available. Letters as ASCII codes, probabilities and times x1000, path in 1/1000 hand sizes.
volatile uint32_t g_fs_tele[21] = {0};  // 0 letter 1 prob 2 presence 3 fps 4 det_ms 5 lm_ms 6 box_x 7 box_y 8 box_w
                                       // 9 pending letter 10 path 11 committed text length
                                       // 12 handedness x1000 (this frame; >= 500 = right, no mirror)
                                       // counters since boot: 13 hand frames 14 left/right flips between
                                       // consecutive hand frames 15 frames with handedness in [0.3, 0.7]
                                       // 16 left (mirrored) frames
                                       // 17 18 19 averaged gravity x y z in milli-g (int32 bits)
                                       // 20 camera roll relative to gravity, millidegrees (int32 bits)
// The landmarks the MLP's features are built from, for tests/capture_keypoints.py (J-Link):
// 0 hand-frame counter (advances only on frames with a hand), 1 presence x1000,
// 2 handedness x1000, 3 the roll applied (millidegrees, int32 bits), 4.. the 21 x (x, y, z)
// levelled frame-pixel landmarks as float bits.
volatile uint32_t g_fs_lm_dump[4 + FS_LM_FEATURES] = {0};
}

namespace {

#ifndef APP_METHOD_POOL_SIZE
#define APP_METHOD_POOL_SIZE (4 * 1024 * 1024)
#endif
#ifndef APP_TEMP_POOL_SIZE
#define APP_TEMP_POOL_SIZE (4 * 1024 * 1024)
#endif
#ifdef APP_POOL_SECTION
#define APP_POOL_ATTRIBUTES __attribute__((section(APP_POOL_SECTION)))
#else
#define APP_POOL_ATTRIBUTES
#endif

constexpr size_t kMethodPoolSize = APP_METHOD_POOL_SIZE;
constexpr size_t kTempPoolSize = APP_TEMP_POOL_SIZE;
alignas(16) uint8_t g_method_pool[kMethodPoolSize] APP_POOL_ATTRIBUTES;
alignas(16) uint8_t g_temp_pool[kTempPoolSize] APP_POOL_ATTRIBUTES;

constexpr int kSize = FS_SIZE;  // the 128x128 RGB888 preview crop (display, dumps, test image)
constexpr int kClasses = FS_CLASSES;
constexpr char kLabels[kClasses + 1] = FS_LABELS;
static_assert(FS_DETECT_SIZE == FS_DET_SIZE && FS_DETECT_ANCHORS == FS_DET_ANCHORS &&
                  FS_DETECT_COORDS == FS_DET_COORDS,
              "fs_detect.h and model/model.py disagree on the detector");
static_assert(FS_LANDMARK_SIZE == FS_LM_SIZE && FS_LANDMARKS == FS_LM_COUNT, "fs_landmark.h vs model/model.py");
static_assert(sizeof(FS_MLP_W1) == sizeof(float) * FS_MLP_HIDDEN * FS_LM_FEATURES, "MLP W1 shape");
static_assert(sizeof(FS_MLP_W2) == sizeof(float) * kClasses * FS_MLP_HIDDEN, "MLP W2 shape");

// ---- timing: the SoC's 100 MHz REFCLK counter ----------------------------------------
// Not the M55's DWT cycle counter: it only runs while a debugger asserts NIDEN, so a board
// that booted on its own printed 0.0 ms everywhere. The REFCLK counter is free-running
// (the pack's sys_busy_loop_init_ns enables it) and wraps every 43 s; differences are fine.
#ifdef FS_HAVE_CAMERA
void cycles_init() { sys_busy_loop_init_ns(); }
inline uint32_t cycles() { return REFCLK_CNTRead->CNTCVL; }
inline uint32_t cycles_per_ms() { return SystemREFClock / 1000; }
inline float cycles_ms(uint32_t c) { return c * 1000.0f / SystemREFClock; }
#else
inline uint32_t cycles() { return 0; }
#endif

constexpr int kFrame = FS_DET_FRAME;  // 320
constexpr float kInputToFrame = static_cast<float>(kFrame) / FS_DET_SIZE;

// The methods' int8 inputs (written by the CPU, read by the NPU through the runtime's copy).
alignas(16) int8_t g_detect_input[FS_DET_SIZE * FS_DET_SIZE * 3];
alignas(16) int8_t g_landmark_input[FS_LM_SIZE * FS_LM_SIZE * 3];
alignas(16) uint8_t g_crop[kSize * kSize * 3];  // RGB888 preview of the landmark crop

constexpr fs_quant_t kDetectInput = {DETECT_INPUT0_SCALE, DETECT_INPUT0_ZERO_POINT};
constexpr fs_quant_t kLandmarkInput = {LANDMARK_INPUT0_SCALE, LANDMARK_INPUT0_ZERO_POINT};
static_assert(DETECT_INPUT0_SIZE == FS_DET_SIZE * FS_DET_SIZE * 3, "detector input size");
static_assert(LANDMARK_INPUT0_SIZE == FS_LM_SIZE * FS_LM_SIZE * 3, "landmark input size");
// The detector's two int8 outputs, told apart by size: regressors 2016x18, score logits 2016.
static_assert((DETECT_OUTPUT0_SIZE == FS_DET_ANCHORS * FS_DET_COORDS && DETECT_OUTPUT1_SIZE == FS_DET_ANCHORS) ||
                  (DETECT_OUTPUT1_SIZE == FS_DET_ANCHORS * FS_DET_COORDS && DETECT_OUTPUT0_SIZE == FS_DET_ANCHORS),
              "detector outputs are the regressors and the score logits");
constexpr bool kRegressorsFirst = DETECT_OUTPUT0_SIZE == FS_DET_ANCHORS * FS_DET_COORDS;
constexpr fs_quant_t kRegressorsQ = kRegressorsFirst ? fs_quant_t{DETECT_OUTPUT0_SCALE, DETECT_OUTPUT0_ZERO_POINT}
                                                     : fs_quant_t{DETECT_OUTPUT1_SCALE, DETECT_OUTPUT1_ZERO_POINT};
constexpr fs_quant_t kLogitsQ = kRegressorsFirst ? fs_quant_t{DETECT_OUTPUT1_SCALE, DETECT_OUTPUT1_ZERO_POINT}
                                                 : fs_quant_t{DETECT_OUTPUT0_SCALE, DETECT_OUTPUT0_ZERO_POINT};
// The landmark model's outputs in TFLite order: landmarks 63, presence 1, handedness 1, world 63.
static_assert(LANDMARK_OUTPUT0_SIZE == FS_LM_FEATURES && LANDMARK_OUTPUT1_SIZE == 1 && LANDMARK_OUTPUT2_SIZE == 1,
              "landmark outputs: landmarks, presence, handedness, world landmarks");
constexpr fs_quant_t kLandmarksQ = {LANDMARK_OUTPUT0_SCALE, LANDMARK_OUTPUT0_ZERO_POINT};
constexpr fs_quant_t kPresenceQ = {LANDMARK_OUTPUT1_SCALE, LANDMARK_OUTPUT1_ZERO_POINT};
constexpr fs_quant_t kHandednessQ = {LANDMARK_OUTPUT2_SCALE, LANDMARK_OUTPUT2_ZERO_POINT};

// A method's single NHWC int8 input tensor over a caller-owned buffer. The runtime keeps a
// pointer to the TensorImpl, so these live as long as the module (static below).
struct MethodInput {
  std::array<SizesType, 4> sizes;
  std::array<DimOrderType, 4> dim_order;
  TensorImpl impl;
  MethodInput(int side, void* data)
      : sizes{1, side, side, 3}, dim_order{0, 1, 2, 3},
        impl(ScalarType::Char, sizes.size(), sizes.data(), data, dim_order.data()) {}
  bool set(EmbeddedModule& module, const char* method) {
    Tensor tensor(&impl);
    const auto err = module.set_input(method, EValue(tensor), 0);
    if (err != executorch::runtime::Error::Ok)
      printf("%s: set_input failed (err=%u)\n", method, static_cast<unsigned>(err));
    return err == executorch::runtime::Error::Ok;
  }
};

// The int8 output tensors of a method run, by position.
struct Outputs {
  const int8_t* data[4] = {nullptr, nullptr, nullptr, nullptr};
  int numel[4] = {0, 0, 0, 0};
  int count = 0;
};

bool run(EmbeddedModule& module, const char* method, MethodInput& input, Outputs* out) {
  if (!input.set(module, method)) return false;
  auto outputs = module.execute(method);
  if (!outputs.ok()) {
    printf("%s failed (err=%u)\n", method, static_cast<unsigned>(outputs.error()));
    return false;
  }
  *out = Outputs{};
  for (const EValue& v : *outputs) {
    if (!v.isTensor() || out->count == 4) continue;
    Tensor t = v.toTensor();
    if (t.scalar_type() != ScalarType::Char) continue;
    out->data[out->count] = t.const_data_ptr<int8_t>();
    out->numel[out->count] = static_cast<int>(t.numel());
    out->count++;
  }
  return true;
}

// Runs the detector on g_detect_input and decodes the best palm. false (and prints why) on
// a runtime error; *hit tells whether that palm reaches the score threshold.
bool detect(EmbeddedModule& module, fs_detection_t* det, bool* hit) {
  static MethodInput input(FS_DET_SIZE, g_detect_input);
  Outputs o;
  if (!run(module, "detect", input, &o)) return false;
  const int8_t* regressors = nullptr;
  const int8_t* logits = nullptr;
  for (int i = 0; i < o.count; ++i) {
    if (o.numel[i] == FS_DET_ANCHORS * FS_DET_COORDS) regressors = o.data[i];
    else if (o.numel[i] == FS_DET_ANCHORS) logits = o.data[i];
  }
  if (regressors == nullptr || logits == nullptr) {
    printf("detect: unexpected outputs (%d tensors)\n", o.count);
    return false;
  }
  *hit = fs_detect_decode(regressors, kRegressorsQ, logits, kLogitsQ, det);
  return true;
}

// Runs the landmark model on g_landmark_input (the crop `c`); landmarks in frame pixels,
// presence and handedness, plus the raw crop-space landmarks for the display.
bool landmark(EmbeddedModule& module, const fs_lm_crop_t* c, fs_landmarks_t* lm, float crop_xy[FS_LM_COUNT][2]) {
  static MethodInput input(FS_LM_SIZE, g_landmark_input);
  Outputs o;
  if (!run(module, "landmark", input, &o)) return false;
  if (o.count < 3 || o.numel[0] != FS_LM_FEATURES || o.numel[1] != 1 || o.numel[2] != 1) {
    printf("landmark: unexpected outputs (%d tensors)\n", o.count);
    return false;
  }
  fs_landmark_decode(o.data[0], kLandmarksQ, o.data[1][0], kPresenceQ, o.data[2][0], kHandednessQ, c, lm);
  for (int i = 0; i < FS_LM_COUNT; ++i) {
    crop_xy[i][0] = static_cast<float>(o.data[0][3 * i] - kLandmarksQ.zero_point) * kLandmarksQ.scale;
    crop_xy[i][1] = static_cast<float>(o.data[0][3 * i + 1] - kLandmarksQ.zero_point) * kLandmarksQ.scale;
  }
  return true;
}

struct TopK {
  int top[3];
  float prob[3];
  float probabilities[kClasses];  // every class, for the decoder
};

// Letter probabilities from the landmarks through the MLP; top-3. The crop geometry the
// landmark model sees (fs_landmark_crop_smooth) and the features (fs_landmark_smooth) are
// averaged over frames so the letter and its confidence do not jitter; both averages restart
// whenever the hand was lost (letters_reset). The decoder averages the probabilities too.
fs_lm_smooth_t g_smooth = {{0}, false};
fs_lm_crop_t g_crop_avg = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
bool g_crop_avg_valid = false;
fs_lm_presence_t g_presence = {false, 0};  // the presence gate with its hold (fs_landmark.h)
void letters_reset() {
  fs_landmark_smooth_reset(&g_smooth);
  fs_landmark_presence_reset(&g_presence);
  g_crop_avg_valid = false;
}

// The camera's roll relative to gravity (fs_orient.h): the accelerometer is read once per frame
// in the live loop, averaged, and turned into the angle the landmarks are rotated by before
// the MLP's features, so the hand's orientation is measured from the true vertical. 0 without
// a sensor or a calibration (src/fs_orient_cal.h).
const fs_orient_cal_t kOrientCal = {FS_ORIENT_CAL_DOWN, FS_ORIENT_CAL_RIGHT};
fs_orient_t g_orient = {{0.f, 0.f, 0.f}, false};
float g_roll = 0.f;  // radians
uint32_t int_bits(float v) { return static_cast<uint32_t>(static_cast<int32_t>(v)); }

// Publish this frame's (levelled) landmarks for the laptop (tests/capture_keypoints.py reads
// them over J-Link with the core halted, so the block is consistent).
void publish_landmarks(const fs_landmarks_t& lm) {
  for (int i = 0; i < FS_LM_COUNT; ++i)
    for (int c = 0; c < 3; ++c) {
      uint32_t bits;
      memcpy(&bits, &lm.xyz[i][c], sizeof bits);
      g_fs_lm_dump[4 + 3 * i + c] = bits;
    }
  g_fs_lm_dump[1] = static_cast<uint32_t>(lm.presence * 1000.f);
  g_fs_lm_dump[2] = static_cast<uint32_t>(lm.right * 1000.f);
  g_fs_lm_dump[3] = int_bits(g_roll * 180000.f / 3.14159265f);
  g_fs_lm_dump[0]++;
}
void letters(const fs_landmarks_t& lm, TopK* r) {
  float f[FS_LM_FEATURES], fs[FS_LM_FEATURES];
  fs_landmark_features(&lm, f);
  fs_landmark_smooth(&g_smooth, f, FS_LM_SMOOTH, fs);
  fs_landmark_letters(fs, &FS_MLP_W1[0][0], FS_MLP_B1, &FS_MLP_W2[0][0], FS_MLP_B2, FS_MLP_HIDDEN, r->probabilities);
  int top[3] = {-1, -1, -1};
  for (int k = 0; k < 3; ++k)
    for (int i = 0; i < kClasses; ++i)
      if ((k == 0 || i != top[0]) && (k < 2 || i != top[1]) && (top[k] < 0 || r->probabilities[i] > r->probabilities[top[k]]))
        top[k] = i;
  for (int k = 0; k < 3; ++k) {
    r->top[k] = top[k];
    r->prob[k] = r->probabilities[top[k]];
  }
}

// The 128x128 RGB888 preview of the 224x224 int8 landmark crop (nearest neighbour): the
// model's rotated input, used for the boot self-test picture.
void preview_crop(const int8_t* crop, uint8_t* rgb) {
  for (int y = 0; y < kSize; ++y) {
    const int sy = y * FS_LM_SIZE / kSize;
    for (int x = 0; x < kSize; ++x) {
      const int8_t* p = crop + (sy * FS_LM_SIZE + x * FS_LM_SIZE / kSize) * 3;
      uint8_t* o = rgb + (y * kSize + x) * 3;
      for (int c = 0; c < 3; ++c) o[c] = static_cast<uint8_t>(p[c] - kLandmarkInput.zero_point);
    }
  }
}

// One whole frame through the pipeline: detector, crop, landmarks, letters. Returns false
// on a runtime error. *hand: a hand with presence >= threshold was found (fresh detection,
// or the held crop geometry `*crop` when the detector missed and hold is true). A fresh
// detection updates the averaged crop geometry, which is what the landmark model sees.
bool infer(EmbeddedModule& module, const uint16_t* frame, bool hold, fs_detection_t* det, bool* detected,
           fs_lm_crop_t* crop, fs_landmarks_t* lm, float crop_xy[FS_LM_COUNT][2], bool* hand, TopK* r,
           uint32_t* t_detected, uint32_t* t_cropped) {
  fs_detect_preprocess(frame, g_detect_input, kDetectInput);
  if (!detect(module, det, detected)) return false;
  if (t_detected) *t_detected = cycles();
  *hand = false;
  if (!*detected && !hold) {
    letters_reset();
    return true;
  }
  if (*detected) {
    const fs_lm_crop_t raw = fs_landmark_crop_frame(det, kInputToFrame);
    fs_landmark_crop_smooth(&g_crop_avg, &g_crop_avg_valid, &raw, FS_LM_CROP_SMOOTH);
    *crop = g_crop_avg;
  }
  fs_landmark_crop(frame, crop, g_landmark_input, kLandmarkInput);
  if (t_cropped) *t_cropped = cycles();
  if (!landmark(module, crop, lm, crop_xy)) return false;
  if (!fs_landmark_presence_gate(&g_presence, lm->presence)) {  // no hand (the hold ran out or never started)
    letters_reset();
    return true;
  }
  *hand = true;
  // The MLP sees the landmarks levelled by the camera's roll (gravity straight down); the
  // display and the decoder keep the camera's view (`*lm`: the decoder uses magnitudes only).
  fs_landmarks_t level = *lm;
  fs_orient_level(level.xyz, FS_LM_COUNT, g_roll, lm->xyz[FS_LM_WRIST]);
  letters(level, r);
  publish_landmarks(level);
  return true;
}

// The baked-in test crop (128x128 RGB888) as a 320x320 RGB565 frame: scaled 2x in the
// centre, black around it.
void test_frame(const uint8_t* rgb, uint16_t* frame) {
  memset(frame, 0, kFrame * kFrame * sizeof(uint16_t));
  for (int y = 0; y < 2 * kSize; ++y)
    for (int x = 0; x < 2 * kSize; ++x) {
      const uint8_t* p = rgb + ((y / 2) * kSize + x / 2) * 3;
      frame[(y + (kFrame - 2 * kSize) / 2) * kFrame + x + (kFrame - 2 * kSize) / 2] =
          static_cast<uint16_t>(((p[0] >> 3) << 11) | ((p[1] >> 2) << 5) | (p[2] >> 3));
    }
}

#ifdef FS_HAVE_CAMERA

// ---- step 4: frames ----------------------------------------------------------------
// 8 fixed sample points (x, y) across the 320x320 frame: centre, ring, corners.
constexpr int kProbe[8][2] = {{160, 160}, {160, 40}, {160, 280}, {40, 160},
                              {280, 160}, {80, 80},  {240, 80},  {80, 240}};

unsigned frame_mean_luma(const uint16_t* f) {
  unsigned long acc = 0;
  unsigned n = 0;
  for (int y = 0; y < FS_CAM_HEIGHT; y += 4)
    for (int x = 0; x < FS_CAM_WIDTH; x += 4, ++n) {
      const uint16_t v = f[y * FS_CAM_WIDTH + x];
      const unsigned r = (v >> 11) << 3, g = ((v >> 5) & 0x3F) << 2, b = (v & 0x1F) << 3;
      acc += (r * 77u + g * 150u + b * 29u) >> 8;
    }
  return static_cast<unsigned>(acc / n);
}

// Grab frames for `ms` milliseconds; every ~250 ms print the frame number, 8 raw
// pixels and the mean brightness. Cover the lens and the numbers drop.
void camera_probe(uint32_t ms) {
  const uint32_t start = cycles(), period = 250 * cycles_per_ms();
  uint32_t next = start, first_seq = 0, last_seq = 0;
  while (cycles() - start < ms * cycles_per_ms()) {
    uint32_t seq = 0;
    const uint16_t* f = fs_camera_latest_frame(500, &seq);
    if (f == nullptr) {
      printf("probe: no frame\n");
      continue;
    }
    if (first_seq == 0) first_seq = seq;
    last_seq = seq;
    if (static_cast<int32_t>(cycles() - next) >= 0) {
      next += period;
      printf("probe frame %lu | px", static_cast<unsigned long>(seq));
      for (const auto& p : kProbe) printf(" %04x", f[p[1] * FS_CAM_WIDTH + p[0]]);
      printf(" | mean %u\n", frame_mean_luma(f));
    }
  }
  const float secs = cycles_ms(cycles() - start) / 1000.0f;
  printf("probe done: %lu frames in %.1f s = %.1f fps (overflows %lu errors %lu restarts %lu)\n",
         static_cast<unsigned long>(last_seq - first_seq + 1), secs, (last_seq - first_seq + 1) / secs,
         static_cast<unsigned long>(g_fs_camera_overflows), static_cast<unsigned long>(g_fs_camera_errors),
         static_cast<unsigned long>(g_fs_camera_restarts));
}

// ---- step 4: the crop as text art and as hex ---------------------------------------
// The team's print_input(): mean colour, then a 32x16 luma map contrast-stretched to this
// image's darkest..brightest cell so the hand outline shows.
void print_input(const uint8_t* rgb) {
  unsigned long sum[3] = {0, 0, 0};
  for (int i = 0; i < kSize * kSize; ++i)
    for (int c = 0; c < 3; ++c) sum[c] += rgb[i * 3 + c];
  printf("Input: %dx%d RGB, mean R=%lu G=%lu B=%lu\n", kSize, kSize, sum[0] / (kSize * kSize),
         sum[1] / (kSize * kSize), sum[2] / (kSize * kSize));
  static const char kRamp[] = " .:-=+*#%@";
  constexpr int kCols = 32, kRows = 16, kCw = kSize / kCols, kCh = kSize / kRows;
  unsigned cell[kRows][kCols];
  unsigned lo = 255, hi = 0;
  for (int r = 0; r < kRows; ++r)
    for (int c = 0; c < kCols; ++c) {
      unsigned long acc = 0;
      for (int y = r * kCh; y < (r + 1) * kCh; ++y)
        for (int x = c * kCw; x < (c + 1) * kCw; ++x) {
          const uint8_t* p = rgb + (y * kSize + x) * 3;
          acc += (p[0] * 77u + p[1] * 150u + p[2] * 29u) >> 8;
        }
      unsigned v = acc / (kCw * kCh);
      cell[r][c] = v;
      lo = v < lo ? v : lo;
      hi = v > hi ? v : hi;
    }
  unsigned span = hi > lo ? hi - lo : 1;
  for (int r = 0; r < kRows; ++r) {
    char line[kCols + 1];
    for (int c = 0; c < kCols; ++c) line[c] = kRamp[(cell[r][c] - lo) * 9 / span];
    line[kCols] = '\0';
    printf("  |%s|\n", line);
  }
}

// One crop as hex, one row per line, for tests/fs_tools.py to turn into a PNG.
void dump_crop(const uint8_t* rgb, uint32_t seq) {
  static const char kHex[] = "0123456789abcdef";
  printf("CROPDUMP BEGIN %d %d frame %lu\n", kSize, kSize, static_cast<unsigned long>(seq));
  char line[kSize * 3 * 2 + 1];
  for (int y = 0; y < kSize; ++y) {
    const uint8_t* row = rgb + y * kSize * 3;
    for (int i = 0; i < kSize * 3; ++i) {
      line[2 * i] = kHex[row[i] >> 4];
      line[2 * i + 1] = kHex[row[i] & 0xF];
    }
    line[kSize * 3 * 2] = '\0';
    printf("%s\n", line);
  }
  printf("CROPDUMP END\n");
}

// ---- step 5: the live loop -----------------------------------------------------------
// Never returns. Per frame: wait for a camera frame, resize to 192 and run the palm detector,
// decode; with a palm (or a held box): rotated 224 crop, landmark model, presence gate, MLP,
// decoder, and print
//   A 0.93 S 0.04 T 0.01 | box 112,84 96x96 s0.87 p0.98 | det 22.9ms cam 2.0ms lm 18.0ms | 12.5 fps
// (box: the axis-aligned hand box, s: detector score, p: landmark presence); without a hand:
//   no hand | det 22.9ms | 21.9 fps
// det = resize + detector + decode, cam = frame wait + rotated crop, lm = landmark model +
// decode + MLP; fps is the loop rate, smoothed. With the LCD (FINGERSPELL_LCD) the crop, the
// skeleton and the letter are drawn and "ui x.xms" precedes the fps.
// FS_FIXED_CENTRE_CROP: the centre 256x256 as the landmark crop (A/B tests); the detector
// still runs and prints its box.
[[noreturn]] void live_loop(EmbeddedModule& module) {
  printf("live: camera -> detect -> crop -> landmark -> letters (det = resize + detector + decode, "
         "cam = wait for frame + crop, lm = landmark model + MLP)\n");
  uint32_t last = cycles();
  float period_ms = 0.f;
  fs_track_t track;
  fs_track_reset(&track);
  fs_lm_crop_t crop = {kFrame / 2.f, kFrame / 2.f, 256.f, 1.f, 0.f, 0.f, 1.f};
  letters_reset();
#ifdef FINGERSPELL_LCD
  fs_decoder_t decoder;
  fs_decoder_init(&decoder, nullptr);
  fs_display_set_frame(g_crop);
#endif
#ifdef FS_HAVE_JOYSTICK
  fs_input_t joy;
  fs_input_init(&joy);
#endif
  for (;;) {
    const uint32_t t0 = cycles();
    uint32_t seq = 0;
    const uint16_t* f = fs_camera_latest_frame(1000, &seq);
    if (f == nullptr) {
      printf("live: no frame (errors %lu restarts %lu)\n", static_cast<unsigned long>(g_fs_camera_errors),
             static_cast<unsigned long>(g_fs_camera_restarts));
      continue;
    }
    const uint32_t t_frame = cycles();
#ifdef FS_HAVE_IMU
    {  // the camera's roll this frame, from the averaged gravity vector
      float g[3];
      if (fs_imu_read(g)) fs_orient_gravity(&g_orient, g, FS_ORIENT_ALPHA);
      g_roll = g_orient.valid ? fs_orient_roll(&kOrientCal, g_orient.g) : 0.f;
    }
#endif
    const uint32_t t_imu = cycles();
    fs_detection_t det;
    fs_landmarks_t lm;
    float crop_xy[FS_LM_COUNT][2];  // crop-space landmarks (unused by the display now)
    bool detected = false, hand = false;
    TopK r = {};
    uint32_t t_detected = t_frame, t_cropped = t_frame;
#ifdef FS_FIXED_CENTRE_CROP
    const bool hold = true;  // always run the landmark model on the centre crop
#else
    const bool hold = track.hand;  // the detector missed: keep using the last crop while held
#endif
    if (!infer(module, f, hold, &det, &detected, &crop, &lm, crop_xy, &hand, &r, &t_detected, &t_cropped)) continue;
    const uint32_t t_inferred = cycles();
    const fs_box_t box = fs_detect_hand_box(&det, kInputToFrame);
    fs_track_update(&track, detected && (hand || !hold), &box, det.score);
    // Where the time goes, every 64 frames.
    if ((g_fs_loop_frames & 63) == 1)
      printf("det split: imu %.2fms resize+detector %.1fms crop %.1fms landmark+mlp %.1fms | roll %.1f deg g (%.2f %.2f %.2f)\n",
             cycles_ms(t_imu - t_frame), cycles_ms(t_detected - t_imu), cycles_ms(t_cropped - t_detected),
             cycles_ms(t_inferred - t_cropped), static_cast<double>(g_roll * 57.29578f), static_cast<double>(g_orient.g[0]),
             static_cast<double>(g_orient.g[1]), static_cast<double>(g_orient.g[2]));

    if (g_fs_loop_frames > 0) {  // the first frame has no previous one to time against
      const float dt = cycles_ms(t0 - last);
      period_ms = period_ms == 0.f ? dt : 0.8f * period_ms + 0.2f * dt;
    }
    last = t0;
    g_fs_loop_frames++;
    const float fps = period_ms > 0.f ? 1000.f / period_ms : 0.f;
    const float det_ms = cycles_ms(t_detected - t_imu);
    const float cam_ms = cycles_ms(t_frame - t0) + (hand || hold ? cycles_ms(t_cropped - t_detected) : 0.f);
    const float lm_ms = hand || hold ? cycles_ms(t_inferred - t_cropped) : 0.f;
    // The panel shows the camera's view of the hand box (upright, as the signer sees it), not
    // the rotated model input; the skeleton is drawn from the frame-space landmarks.
    if (hand) fs_detect_crop(f, &track.box, g_crop);
    const uint32_t t5 = cycles();
#ifdef FINGERSPELL_LCD
    // The free-running 32 kHz timer keeps milliseconds monotonic across REFCLK wraps. Without
    // a hand the decoder sees all-zero probabilities: no letter, and a space once the hand has
    // been out of view for space_ms.
    const uint32_t now_ms = static_cast<uint32_t>(
        static_cast<uint64_t>(S32K_CNTRead->CNTCVL) * 1000u / 32768u);
    // Fingertips and hand size for the motion letters (Z from D, J from I), frame pixels.
    const fs_motion_t motion = {hand, {lm.xyz[8][0], lm.xyz[8][1]}, {lm.xyz[20][0], lm.xyz[20][1]},
                                hand ? hypotf(lm.xyz[9][0] - lm.xyz[0][0], lm.xyz[9][1] - lm.xyz[0][1]) : 0.f};
    fs_event_t event = fs_decoder_step_motion(&decoder, r.probabilities, &motion, now_ms);
#ifdef FS_HAVE_JOYSTICK
    {  // DOWN clears the text, LEFT deletes one character; one event per press, clear wins
      bool down = false, left = false;
      if (fs_joystick_read(&down, &left)) {
        const fs_input_event_t in = fs_input_update(&joy, down, left);
        const fs_event_t edit = in == FS_INPUT_CLEAR       ? fs_decoder_clear(&decoder)
                                : in == FS_INPUT_BACKSPACE ? fs_decoder_backspace(&decoder)
                                                           : FS_EV_NONE;
        if (edit != FS_EV_NONE) event = edit;
      }
    }
#endif
    const uint32_t t6 = cycles();
    const fs_metrics_t metrics = {fps, cam_ms, det_ms, lm_ms, cycles_ms(t6 - t5), cycles_ms(t6 - t0)};
    const char top3[3] = {hand ? kLabels[r.top[0]] : '\0', hand ? kLabels[r.top[1]] : '\0', hand ? kLabels[r.top[2]] : '\0'};
    fs_display_set_top3(top3, r.prob);
    fs_display_set_detection(f, hand, detected, track.box.x0, track.box.y0, track.box.x1, track.box.y1);
    fs_display_set_landmarks(hand ? &track.box : nullptr, hand ? lm.xyz : nullptr);
    fs_ui_render(&decoder, &metrics, event);
    const float ui_ms = cycles_ms(cycles() - t5);
    if (hand)
      printf("%c %.2f %c %.2f %c %.2f | box %d,%d %dx%d s%.2f p%.2f | det %.1fms cam %.1fms lm %.1fms ui %.1fms | %.1f fps\n",
             kLabels[r.top[0]], r.prob[0], kLabels[r.top[1]], r.prob[1], kLabels[r.top[2]], r.prob[2],
             static_cast<int>(track.box.x0), static_cast<int>(track.box.y0),
             static_cast<int>(track.box.x1 - track.box.x0), static_cast<int>(track.box.y1 - track.box.y0),
             track.score, lm.presence, det_ms, cam_ms, lm_ms, ui_ms, fps);
    else
      printf("no hand | det %.1fms ui %.1fms | %.1f fps\n", det_ms, ui_ms, fps);
    g_fs_tele[0] = hand ? static_cast<uint32_t>(kLabels[r.top[0]]) : 0u;
    g_fs_tele[1] = static_cast<uint32_t>(r.prob[0] * 1000.f);
    g_fs_tele[2] = static_cast<uint32_t>(lm.presence * 1000.f);
    g_fs_tele[3] = static_cast<uint32_t>(fps * 1000.f);
    g_fs_tele[4] = static_cast<uint32_t>(det_ms * 1000.f);
    g_fs_tele[5] = static_cast<uint32_t>(lm_ms * 1000.f);
    g_fs_tele[6] = static_cast<uint32_t>(static_cast<int>(track.box.x0));
    g_fs_tele[7] = static_cast<uint32_t>(static_cast<int>(track.box.y0));
    g_fs_tele[8] = static_cast<uint32_t>(static_cast<int>(track.box.x1 - track.box.x0));
    g_fs_tele[9] = static_cast<uint32_t>(decoder.pending);
    g_fs_tele[10] = static_cast<uint32_t>(decoder.pending_path * 1000.f);
    g_fs_tele[11] = static_cast<uint32_t>(decoder.len);
    for (int i = 0; i < 3; ++i) g_fs_tele[17 + i] = int_bits(g_orient.g[i] * 1000.f);
    g_fs_tele[20] = int_bits(g_roll * 180000.f / 3.14159265f);
    {  // handedness diagnostics: does the left-hand mirror flip between frames of one hand?
      static int last_side = -1;  // -1: no hand in the previous frame
      if (hand) {
        const int side = lm.right >= 0.5f ? 1 : 0;
        g_fs_tele[12] = static_cast<uint32_t>(lm.right * 1000.f);
        g_fs_tele[13]++;
        if (last_side >= 0 && side != last_side) g_fs_tele[14]++;
        if (lm.right >= 0.3f && lm.right <= 0.7f) g_fs_tele[15]++;
        if (!side) g_fs_tele[16]++;
        last_side = side;
      } else {
        last_side = -1;
      }
    }
    if (event == FS_EV_LETTER || (decoder.pending && (g_fs_loop_frames & 7) == 0))
      printf("decoder: %s%c path %.2f text \"%s\"\n", decoder.pending ? "pending " : "committed ",
             decoder.pending ? decoder.pending : decoder.text[decoder.len ? decoder.len - 1 : 0], decoder.pending_path,
             decoder.text);
    if (event == FS_EV_BACKSPACE || event == FS_EV_CLEAR)
      printf("decoder: %s text \"%s\"\n", event == FS_EV_CLEAR ? "clear" : "backspace", decoder.text);
#else
    if (hand)
      printf("%c %.2f %c %.2f %c %.2f | box %d,%d %dx%d s%.2f p%.2f | det %.1fms cam %.1fms lm %.1fms | %.1f fps\n",
             kLabels[r.top[0]], r.prob[0], kLabels[r.top[1]], r.prob[1], kLabels[r.top[2]], r.prob[2],
             static_cast<int>(track.box.x0), static_cast<int>(track.box.y0),
             static_cast<int>(track.box.x1 - track.box.x0), static_cast<int>(track.box.y1 - track.box.y0),
             track.score, lm.presence, det_ms, cam_ms, lm_ms, fps);
    else
      printf("no hand | det %.1fms | %.1f fps\n", det_ms, fps);
#endif
    if (g_fs_dump_request) {
      dump_crop(g_crop, seq);  // the last landmark crop's preview (stale while there is no hand)
      g_fs_dump_request = 0;
      last = cycles();  // the dump takes seconds: keep it out of the fps
    }
  }
}
#endif  // FS_HAVE_CAMERA

}  // namespace

extern "C" int app_main(void) {
  executorch::runtime::runtime_init();
  printf("Fingerspell detect + landmark: %lu byte program\n", model_pte_size);

  EmbeddedModule module(
      model_pte, model_pte_size,
      std::make_unique<BufferDataLoader>(model_pte, model_pte_size),
      std::make_unique<MemoryAllocator>(kMethodPoolSize, g_method_pool),
      std::make_unique<MemoryAllocator>(kTempPoolSize, g_temp_pool));

  // Boot self-test: the baked-in crop as a frame through the whole pipeline (the centre crop
  // when the detector does not fire on it), run twice: the live loop runs the models on every
  // frame, so a second run has to give the same answer.
  static uint16_t frame[kFrame * kFrame];
  test_frame(kTestImage, frame);
  fs_detection_t det;
  fs_landmarks_t lm;
  float crop_xy[FS_LM_COUNT][2];
  fs_lm_crop_t crop = {kFrame / 2.f, kFrame / 2.f, 2.f * kSize, 1.f, 0.f, 0.f, 1.f};
  bool detected = false, hand = false, hand2 = false;
  TopK r = {}, r2 = {};
  if (!infer(module, frame, true, &det, &detected, &crop, &lm, crop_xy, &hand, &r, nullptr, nullptr)) return 1;
  printf("Self-test: detector %s (s%.2f), landmark presence %.2f, %s hand\n", detected ? "hit" : "miss (centre crop)",
         det.score, lm.presence, lm.right >= 0.5f ? "right" : "left");
  if (hand)
    printf("Top-3: %c %.2f | %c %.2f | %c %.2f   (expected %c)\n", kLabels[r.top[0]], r.prob[0],
           kLabels[r.top[1]], r.prob[1], kLabels[r.top[2]], r.prob[2], kTestExpected);
  else
    printf("Top-3: none (no hand in the test crop)   (expected %c)\n", kTestExpected);
  letters_reset();
  if (!infer(module, frame, true, &det, &detected, &crop, &lm, crop_xy, &hand2, &r2, nullptr, nullptr)) return 1;
  const bool repeat_ok = hand2 == hand && (!hand || r2.top[0] == r.top[0]);
  printf("Repeat run: %s\n", repeat_ok ? "same top-1" : "FAILED");
  const bool pass = hand && kLabels[r.top[0]] == kTestExpected && repeat_ok;
  printf("Test_result: %s\n", pass ? "PASS" : "FAIL");
#ifdef FINGERSPELL_LCD
  if (hand) preview_crop(g_landmark_input, g_crop);
  if (!fs_display_result(hand ? g_crop : kTestImage, hand ? kLabels[r.top[0]] : '-', hand ? r.prob[0] : 0.f,
                         kTestExpected, pass))
    printf("LCD unavailable: UART output remains active\n");
#endif

#ifdef FS_HAVE_CAMERA
  // The camera starts after the self-test, so the test above never shares memory traffic
  // with the CPI's DMA. Prints "camera init OK 320x320 RGB565" or the stage that failed.
  cycles_init();
  const bool camera = fs_camera_init();
#ifdef FS_HAVE_IMU
  // After the camera, so a stuck IMU bring-up (the pack driver can spin without a timeout) is
  // visible as the last stage printed and does not cost the camera. Prints "imu init OK g = ..".
  fs_orient_reset(&g_orient);
  fs_imu_init();
  printf("orient: %s (src/fs_orient_cal.h)\n", FS_ORIENT_CALIBRATED ? "calibrated" : "not calibrated: roll stays 0");
#endif
#ifdef FS_HAVE_JOYSTICK
  fs_joystick_init();  // last: a refusal is the last line printed and costs nothing else
#endif
  if (camera) {
    printf("core clock %lu Hz\n", static_cast<unsigned long>(SystemCoreClock));
    camera_probe(3000);                         // step 4: frames
    if (fs_camera_get_crop(g_crop)) {           // step 4: the centre crop
      printf("crop (centre %dx%d of %dx%d, 2x2 average):\n", FS_CAM_BOX, FS_CAM_BOX, FS_CAM_WIDTH,
             FS_CAM_HEIGHT);
      print_input(g_crop);
    }
    live_loop(module);                          // step 5: never returns
  }
  printf("camera unavailable: no live loop\n");
#endif
  printf("\x04");
  fflush(stdout);
  return pass ? 0 : 1;
}
