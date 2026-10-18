// Fingerspell — app_main.cpp for Arm-Examples/CMSIS-Executorch@hackathon (replaces src/app_main.cpp).
//
// Boot: runs the letter classifier (method "classify") on a baked-in 128x128 RGB test crop
// (test_image.h, made by ml/make_test_image.py from the held-out split) and prints the top-3
// letters plus the expected one (Test_result: PASS/FAIL). On a board with the camera (DevKit-E8)
// it then:
//   step 4  probes the camera for 3 s: frame counter, 8 raw RGB565 pixels, mean brightness, fps;
//   step 4  prints the first centre crop as text art (fs_camera_get_crop, 128x128);
//   step 5  runs the live loop forever: frame -> palm detector ("detect", 192x192, decoded on the
//           CPU by fs_detect.c) -> hand box -> crop -> "classify" -> top-3, timed per stage.
// Without a camera (FVP, AppKit-E8) main() returns after the test as before.
//
// Input contracts (model/model.py): detect 1x192x192x3 NHWC int8 (uint8 / 255 quantized with
// DETECT_INPUT0_SCALE / _ZERO_POINT from model_params.h), outputs the raw SSD head as int8
// (regressors 2016x18, score logits 2016, DETECT_OUTPUT<i>_*); classify 1x128x128x3 NHWC
// float = uint8 / 255 (normalization is inside the model), 24 float logits in kLabels order.

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>

#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/platform/runtime.h>

#include "arm_embedded_module.hpp"
#include "fs_detect.h"
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

constexpr int kSize = FS_CLASSIFY_SIZE;
constexpr int kClasses = FS_CLASSES;
constexpr char kLabels[kClasses + 1] = FS_LABELS;
static_assert(kSize == FS_SIZE, "fs_api.h and model/model.py disagree on the classifier input");
static_assert(FS_DETECT_SIZE == FS_DET_SIZE && FS_DETECT_ANCHORS == FS_DET_ANCHORS &&
                  FS_DETECT_COORDS == FS_DET_COORDS,
              "fs_detect.h and model/model.py disagree on the detector");

alignas(16) float g_input[kSize * kSize * 3];

// uint8 RGB888 (HWC) -> float [0,1] NHWC.
void preprocess_rgb888(const uint8_t* rgb, float* out) {
  for (int i = 0; i < kSize * kSize * 3; ++i) out[i] = rgb[i] * (1.0f / 255.0f);
}

// A method's single NHWC input tensor over a caller-owned buffer. The runtime keeps a
// pointer to the TensorImpl, so these live as long as the module (static below).
struct MethodInput {
  std::array<SizesType, 4> sizes;
  std::array<DimOrderType, 4> dim_order;
  TensorImpl impl;
  MethodInput(int side, void* data, ScalarType type)
      : sizes{1, side, side, 3}, dim_order{0, 1, 2, 3},
        impl(type, sizes.size(), sizes.data(), data, dim_order.data()) {}
  bool set(EmbeddedModule& module, const char* method) {
    Tensor tensor(&impl);
    const auto err = module.set_input(method, EValue(tensor), 0);
    if (err != executorch::runtime::Error::Ok)
      printf("%s: set_input failed (err=%u)\n", method, static_cast<unsigned>(err));
    return err == executorch::runtime::Error::Ok;
  }
};

struct TopK {
  int top[3];
  float prob[3];
  float probabilities[kClasses];  // every class, for the decoder
};

// Runs the classifier on g_input; softmax + top-3. false (and prints why) on error.
bool classify(EmbeddedModule& module, TopK* r) {
  static MethodInput input(kSize, g_input, ScalarType::Float);
  if (!input.set(module, "classify")) return false;
  auto outputs = module.execute("classify");
  if (!outputs.ok() || outputs->empty() || !outputs->front().isTensor()) {
    printf("classify failed (err=%u)\n", outputs.ok() ? 0u : static_cast<unsigned>(outputs.error()));
    return false;
  }
  Tensor t = outputs->front().toTensor();
  if (t.numel() != kClasses || t.scalar_type() != ScalarType::Float) {
    printf("unexpected output: %u elements\n", static_cast<unsigned>(t.numel()));
    return false;
  }
  const float* logits = t.const_data_ptr<float>();

  float mx = logits[0];
  for (int i = 1; i < kClasses; ++i) mx = logits[i] > mx ? logits[i] : mx;
  float prob[kClasses], sum = 0.f;
  for (int i = 0; i < kClasses; ++i) sum += (prob[i] = std::exp(logits[i] - mx));
  for (int i = 0; i < kClasses; ++i) r->probabilities[i] = prob[i] / sum;
  int top[3] = {-1, -1, -1};
  for (int k = 0; k < 3; ++k)
    for (int i = 0; i < kClasses; ++i)
      if ((k == 0 || i != top[0]) && (k < 2 || i != top[1]) && (top[k] < 0 || prob[i] > prob[top[k]])) top[k] = i;
  for (int k = 0; k < 3; ++k) {
    r->top[k] = top[k];
    r->prob[k] = prob[top[k]] / sum;
  }
  return true;
}

#ifdef FS_HAVE_CAMERA
alignas(16) uint8_t g_crop[kSize * kSize * 3];
// The detector's int8 input (110 kB), written by fs_detect_preprocess; the method takes it
// as is (quantized_io in model/model.py), so no quantize operator runs on the CPU.
alignas(16) int8_t g_detect_input[FS_DET_SIZE * FS_DET_SIZE * 3];
constexpr fs_quant_t kDetectInput = {DETECT_INPUT0_SCALE, DETECT_INPUT0_ZERO_POINT};
// The two int8 outputs, told apart by size: regressors 2016x18, score logits 2016.
static_assert(DETECT_INPUT0_SIZE == FS_DET_SIZE * FS_DET_SIZE * 3, "detector input size");
static_assert((DETECT_OUTPUT0_SIZE == FS_DET_ANCHORS * FS_DET_COORDS && DETECT_OUTPUT1_SIZE == FS_DET_ANCHORS) ||
                  (DETECT_OUTPUT1_SIZE == FS_DET_ANCHORS * FS_DET_COORDS && DETECT_OUTPUT0_SIZE == FS_DET_ANCHORS),
              "detector outputs are the regressors and the score logits");
constexpr bool kRegressorsFirst = DETECT_OUTPUT0_SIZE == FS_DET_ANCHORS * FS_DET_COORDS;
constexpr fs_quant_t kRegressorsQ = kRegressorsFirst ? fs_quant_t{DETECT_OUTPUT0_SCALE, DETECT_OUTPUT0_ZERO_POINT}
                                                     : fs_quant_t{DETECT_OUTPUT1_SCALE, DETECT_OUTPUT1_ZERO_POINT};
constexpr fs_quant_t kLogitsQ = kRegressorsFirst ? fs_quant_t{DETECT_OUTPUT1_SCALE, DETECT_OUTPUT1_ZERO_POINT}
                                                 : fs_quant_t{DETECT_OUTPUT0_SCALE, DETECT_OUTPUT0_ZERO_POINT};

// Runs the detector on g_detect_input and decodes the best palm. false (and prints why) on
// a runtime error; *hit tells whether that palm reaches the score threshold.
bool detect(EmbeddedModule& module, fs_detection_t* det, bool* hit) {
  static MethodInput input(FS_DET_SIZE, g_detect_input, ScalarType::Char);
  if (!input.set(module, "detect")) return false;
  auto outputs = module.execute("detect");
  if (!outputs.ok()) {
    printf("detect failed (err=%u)\n", static_cast<unsigned>(outputs.error()));
    return false;
  }
  const int8_t* regressors = nullptr;
  const int8_t* logits = nullptr;
  for (const EValue& v : *outputs) {
    if (!v.isTensor()) continue;
    Tensor t = v.toTensor();
    if (t.scalar_type() != ScalarType::Char) continue;
    if (t.numel() == FS_DET_ANCHORS * FS_DET_COORDS) regressors = t.const_data_ptr<int8_t>();
    else if (t.numel() == FS_DET_ANCHORS) logits = t.const_data_ptr<int8_t>();
  }
  if (regressors == nullptr || logits == nullptr) {
    printf("detect: unexpected outputs (%u values)\n", static_cast<unsigned>(outputs->size()));
    return false;
  }
  *hit = fs_detect_decode(regressors, kRegressorsQ, logits, kLogitsQ, det);
  return true;
}

// ---- timing: the SoC's 100 MHz REFCLK counter ----------------------------------------
// Not the M55's DWT cycle counter: it only runs while a debugger asserts NIDEN, so a board
// that booted on its own printed 0.0 ms everywhere. The REFCLK counter is free-running
// (the pack's sys_busy_loop_init_ns enables it) and wraps every 43 s; differences are fine.
void cycles_init() { sys_busy_loop_init_ns(); }
inline uint32_t cycles() { return REFCLK_CNTRead->CNTCVL; }
inline uint32_t cycles_per_ms() { return SystemREFClock / 1000; }
inline float cycles_ms(uint32_t c) { return c * 1000.0f / SystemREFClock; }

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
// Never returns. Per frame: wait for a camera frame, resize it to 192 and run the palm
// detector, decode, track the hand box; with a hand: crop the box from the frame, preprocess,
// classify, and print
//   L 0.93 B 0.04 A 0.01 | box 112,84 96x96 s0.87 | det 3.1ms cam 2.0ms pre 0.8ms npu 4.2ms | 18.4 fps
// (box: top-left corner and size in frame pixels, s: detector score); without one:
//   no hand | det 3.1ms | 21.9 fps
// det = resize + detector + decode, cam = frame wait + crop, pre = to float, npu = classifier +
// softmax; fps is the loop rate, smoothed over the last frames. With the LCD (FINGERSPELL_LCD)
// the frame is also drawn on the panel and "ui x.xms" (decoder + render) precedes the fps.
// FS_FIXED_CENTRE_CROP: classify the centre 256x256 as before the detector (A/B tests); the
// detector still runs and prints.
[[noreturn]] void live_loop(EmbeddedModule& module) {
  printf("live: camera -> detect -> crop -> classify -> top-3 (det = resize + detector + decode, "
         "cam = wait for frame + crop, pre = to float, npu = classifier + softmax)\n");
  uint32_t last = cycles();
  float period_ms = 0.f;
  fs_track_t track;
  fs_track_reset(&track);
  constexpr float kInputToFrame = static_cast<float>(FS_CAM_WIDTH) / FS_DET_SIZE;
#ifdef FINGERSPELL_LCD
  fs_decoder_t decoder;
  fs_decoder_init(&decoder, nullptr);
  fs_display_set_frame(g_crop);
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
    fs_detect_preprocess(f, g_detect_input, kDetectInput);
    const uint32_t t_resized = cycles();
    fs_detection_t det;
    bool hit = false;
    if (!detect(module, &det, &hit)) continue;
    const fs_box_t box = fs_detect_hand_box(&det, kInputToFrame);
    fs_track_update(&track, hit, &box, det.score);
    const uint32_t t1 = cycles();
    const float det_ms = cycles_ms(t1 - t_frame);
    // Where the detector's time goes, every 64 frames: the CPU resize to int8, and the
    // method (the NPU) with the decode.
    if ((g_fs_loop_frames & 63) == 1)
      printf("det split: resize %.1fms detector %.1fms\n", cycles_ms(t_resized - t_frame), cycles_ms(t1 - t_resized));

    if (g_fs_loop_frames > 0) {  // the first frame has no previous one to time against
      const float dt = cycles_ms(t0 - last);
      period_ms = period_ms == 0.f ? dt : 0.8f * period_ms + 0.2f * dt;
    }
    last = t0;
    g_fs_loop_frames++;
    const float fps = period_ms > 0.f ? 1000.f / period_ms : 0.f;

    TopK r = {};
    float cam_ms = cycles_ms(t_frame - t0), pre_ms = 0.f, npu_ms = 0.f;
    if (track.hand) {
#ifdef FS_FIXED_CENTRE_CROP
      fs_camera_crop(f, g_crop);
#else
      fs_detect_crop(f, &track.box, g_crop);
#endif
      const uint32_t t2 = cycles();
      preprocess_rgb888(g_crop, g_input);
      const uint32_t t3 = cycles();
      if (!classify(module, &r)) continue;
      const uint32_t t4 = cycles();
      cam_ms += cycles_ms(t2 - t1);
      pre_ms = cycles_ms(t3 - t2);
      npu_ms = cycles_ms(t4 - t3);
    }
    const uint32_t t5 = cycles();
#ifdef FINGERSPELL_LCD
    // The free-running 32 kHz timer keeps milliseconds monotonic across REFCLK wraps. Without
    // a hand the decoder sees all-zero probabilities: no letter, and a space after its idle time.
    const uint32_t now_ms = static_cast<uint32_t>(
        static_cast<uint64_t>(S32K_CNTRead->CNTCVL) * 1000u / 32768u);
    const fs_event_t event = fs_decoder_step(&decoder, r.probabilities, now_ms);
    const uint32_t t6 = cycles();
    // The LCD: crop, the frame inset with the box, top-3 and the decoder's text. Its time is
    // printed as "ui" below.
    const fs_metrics_t metrics = {fps, cam_ms, pre_ms, npu_ms, cycles_ms(t6 - t5), cycles_ms(t6 - t0)};
    if (track.hand) {
      const char top3[3] = {kLabels[r.top[0]], kLabels[r.top[1]], kLabels[r.top[2]]};
      fs_display_set_top3(top3, r.prob);
    } else {
      const char none[3] = {0, 0, 0};
      fs_display_set_top3(none, r.prob);
    }
    fs_display_set_detection(f, track.hand, track.fresh, track.box.x0, track.box.y0, track.box.x1, track.box.y1);
    fs_ui_render(&decoder, &metrics, event);
    const float ui_ms = cycles_ms(cycles() - t5);
    if (track.hand)
      printf("%c %.2f %c %.2f %c %.2f | box %d,%d %dx%d s%.2f | det %.1fms cam %.1fms pre %.1fms npu %.1fms ui %.1fms | %.1f fps\n",
             kLabels[r.top[0]], r.prob[0], kLabels[r.top[1]], r.prob[1], kLabels[r.top[2]], r.prob[2],
             static_cast<int>(track.box.x0), static_cast<int>(track.box.y0),
             static_cast<int>(track.box.x1 - track.box.x0), static_cast<int>(track.box.y1 - track.box.y0),
             track.score, det_ms, cam_ms, pre_ms, npu_ms, ui_ms, fps);
    else
      printf("no hand | det %.1fms ui %.1fms | %.1f fps\n", det_ms, ui_ms, fps);
#else
    if (track.hand)
      printf("%c %.2f %c %.2f %c %.2f | box %d,%d %dx%d s%.2f | det %.1fms cam %.1fms pre %.1fms npu %.1fms | %.1f fps\n",
             kLabels[r.top[0]], r.prob[0], kLabels[r.top[1]], r.prob[1], kLabels[r.top[2]], r.prob[2],
             static_cast<int>(track.box.x0), static_cast<int>(track.box.y0),
             static_cast<int>(track.box.x1 - track.box.x0), static_cast<int>(track.box.y1 - track.box.y0),
             track.score, det_ms, cam_ms, pre_ms, npu_ms, fps);
    else
      printf("no hand | det %.1fms | %.1f fps\n", det_ms, fps);
#endif
    if (g_fs_dump_request) {
      dump_crop(g_crop, seq);  // the last classifier crop (stale while there is no hand)
      g_fs_dump_request = 0;
      last = cycles();  // the dump takes seconds: keep it out of the fps
    }
  }
}
#endif  // FS_HAVE_CAMERA

}  // namespace

extern "C" int app_main(void) {
  executorch::runtime::runtime_init();
  printf("Fingerspell detect + classify: %lu byte program\n", model_pte_size);

  EmbeddedModule module(
      model_pte, model_pte_size,
      std::make_unique<BufferDataLoader>(model_pte, model_pte_size),
      std::make_unique<MemoryAllocator>(kMethodPoolSize, g_method_pool),
      std::make_unique<MemoryAllocator>(kTempPoolSize, g_temp_pool));

  // Boot self-test on the baked-in crop through the classifier only, run twice: the live
  // loop runs the model on every frame, so a second run has to give the same answer.
  preprocess_rgb888(kTestImage, g_input);
  TopK r, r2;
  if (!classify(module, &r)) return 1;
  printf("Top-3: %c %.2f | %c %.2f | %c %.2f   (expected %c)\n", kLabels[r.top[0]], r.prob[0],
         kLabels[r.top[1]], r.prob[1], kLabels[r.top[2]], r.prob[2], kTestExpected);
  const bool repeat_ok = classify(module, &r2) && r2.top[0] == r.top[0];
  printf("Repeat run: %s\n", repeat_ok ? "same top-1" : "FAILED");
  const bool pass = kLabels[r.top[0]] == kTestExpected && repeat_ok;
  printf("Test_result: %s\n", pass ? "PASS" : "FAIL");
#ifdef FINGERSPELL_LCD
  if (!fs_display_result(kTestImage, kLabels[r.top[0]], r.prob[0], kTestExpected, pass))
    printf("LCD unavailable: UART output remains active\n");
#endif

#ifdef FS_HAVE_CAMERA
  // The camera starts after the self-test, so the test above never shares memory traffic
  // with the CPI's DMA. Prints "camera init OK 320x320 RGB565" or the stage that failed.
  cycles_init();
  const bool camera = fs_camera_init();
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
