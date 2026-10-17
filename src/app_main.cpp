// Fingerspell — app_main.cpp for Arm-Examples/CMSIS-Executorch@hackathon (replaces src/app_main.cpp).
//
// Boot: runs the 24-letter model on a baked-in 128x128 RGB test crop (test_image.h, made by
// ml/make_test_image.py from the held-out split) and prints the top-3 letters plus the expected one
// (Test_result: PASS/FAIL). On a board with the camera (DevKit-E8) it then:
//   step 4  probes the camera for 3 s: frame counter, 8 raw RGB565 pixels, mean brightness, fps;
//   step 4  prints the first camera crop as text art (fs_camera_get_crop: the hand box, 128x128);
//   step 5  runs the live loop forever: camera crop -> preprocess -> model -> top-3, timed per stage.
// Without a camera (FVP, AppKit-E8) main() returns after the test as before.
//
// Input contract: 1x128x128x3 NHWC float, value = uint8 / 255 (normalization is inside the model).
// Output: 24 float logits in kLabels order (A..Y without J).

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
#include "model_pte.h"
#include "test_image.h"  // kTestImage[128*128*3] (uint8 RGB), kTestExpected (char)

#include "RTE_Components.h"
#ifdef RTE_VSTREAM_VIDEOIN  // board layer with the MT9M114 camera (DevKit-E8)
#include CMSIS_device_header
#include "fs_camera.h"
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

// Set from the debugger (tests/fs_tools.py) to make the live loop print the next camera
// crop as hex between CROPDUMP BEGIN/END lines; the loop clears it.
extern "C" {
volatile uint32_t g_fs_dump_request = 0;
volatile uint32_t g_fs_loop_frames = 0;  // live-loop iterations (frames classified)
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

constexpr int kSize = 128;
constexpr int kClasses = 24;
constexpr char kLabels[kClasses + 1] = "ABCDEFGHIKLMNOPQRSTUVWXY";

alignas(16) float g_input[kSize * kSize * 3];

// uint8 RGB888 (HWC) -> float [0,1] NHWC.
void preprocess_rgb888(const uint8_t* rgb, float* out) {
  for (int i = 0; i < kSize * kSize * 3; ++i) out[i] = rgb[i] * (1.0f / 255.0f);
}

struct TopK {
  int top[3];
  float prob[3];
};

// Runs the model on g_input; softmax + top-3. false (and prints why) on error.
bool classify(EmbeddedModule& module, TopK* r) {
  std::array<SizesType, 4> sizes{1, kSize, kSize, 3};
  std::array<DimOrderType, 4> dim_order{0, 1, 2, 3};
  TensorImpl input_impl(ScalarType::Float, sizes.size(), sizes.data(), g_input, dim_order.data());
  Tensor input(&input_impl);

  const auto set = module.set_input(EValue(input), 0);
  if (set != executorch::runtime::Error::Ok) {
    printf("set_input failed (err=%u)\n", static_cast<unsigned>(set));
    return false;
  }
  auto outputs = module.execute("forward");
  if (!outputs.ok() || outputs->empty() || !outputs->front().isTensor()) {
    printf("forward failed (err=%u)\n", outputs.ok() ? 0u : static_cast<unsigned>(outputs.error()));
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

// ---- timing: the M55 cycle counter -----------------------------------------------
void cycles_init() {
  DCB->DEMCR |= DCB_DEMCR_TRCENA_Msk;
  DWT->CYCCNT = 0;
  DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}
inline uint32_t cycles() { return DWT->CYCCNT; }
inline float cycles_ms(uint32_t c) { return c * 1000.0f / SystemCoreClock; }

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
  const uint32_t start = cycles(), period = SystemCoreClock / 4;
  uint32_t next = start, first_seq = 0, last_seq = 0;
  while (cycles() - start < ms * (SystemCoreClock / 1000)) {
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
// Never returns. Per frame: camera crop (waits for a new frame), preprocess, model, and one line
//   L 0.93 B 0.04 A 0.01 | cam 12.0ms pre 3.1ms npu 2.3ms | 18.4 fps
// fps is the loop rate, smoothed over the last frames.
[[noreturn]] void live_loop(EmbeddedModule& module) {
  printf("live: camera -> model -> top-3 (cam = wait for frame + crop, pre = to float, "
         "npu = forward + softmax)\n");
  uint32_t last = cycles();
  float period_ms = 0.f;
  for (;;) {
    const uint32_t t0 = cycles();
    uint32_t seq = 0;
    const uint16_t* f = fs_camera_latest_frame(1000, &seq);
    if (f == nullptr) {
      printf("live: no frame (errors %lu restarts %lu)\n", static_cast<unsigned long>(g_fs_camera_errors),
             static_cast<unsigned long>(g_fs_camera_restarts));
      continue;
    }
    fs_camera_crop(f, g_crop);
    const uint32_t t1 = cycles();
    preprocess_rgb888(g_crop, g_input);
    const uint32_t t2 = cycles();
    TopK r;
    if (!classify(module, &r)) continue;
    const uint32_t t3 = cycles();

    if (g_fs_loop_frames > 0) {  // the first frame has no previous one to time against
      const float dt = cycles_ms(t0 - last);
      period_ms = period_ms == 0.f ? dt : 0.8f * period_ms + 0.2f * dt;
    }
    last = t0;
    g_fs_loop_frames++;
    printf("%c %.2f %c %.2f %c %.2f | cam %.1fms pre %.1fms npu %.1fms | %.1f fps\n", kLabels[r.top[0]],
           r.prob[0], kLabels[r.top[1]], r.prob[1], kLabels[r.top[2]], r.prob[2], cycles_ms(t1 - t0),
           cycles_ms(t2 - t1), cycles_ms(t3 - t2), period_ms > 0.f ? 1000.f / period_ms : 0.f);
    if (g_fs_dump_request) {
      dump_crop(g_crop, seq);
      g_fs_dump_request = 0;
      last = cycles();  // the dump takes seconds: keep it out of the fps
    }
  }
}
#endif  // FS_HAVE_CAMERA

}  // namespace

extern "C" int app_main(void) {
  executorch::runtime::runtime_init();
  printf("Fingerspell letters: %lu byte model\n", model_pte_size);

  EmbeddedModule module(
      model_pte, model_pte_size,
      std::make_unique<BufferDataLoader>(model_pte, model_pte_size),
      std::make_unique<MemoryAllocator>(kMethodPoolSize, g_method_pool),
      std::make_unique<MemoryAllocator>(kTempPoolSize, g_temp_pool));

  // Boot self-test on the baked-in crop, run twice: the live loop runs the model on every
  // frame, so a second run has to give the same answer as the first.
  preprocess_rgb888(kTestImage, g_input);
  TopK r, r2;
  if (!classify(module, &r)) return 1;
  printf("Top-3: %c %.2f | %c %.2f | %c %.2f   (expected %c)\n", kLabels[r.top[0]], r.prob[0],
         kLabels[r.top[1]], r.prob[1], kLabels[r.top[2]], r.prob[2], kTestExpected);
  const bool repeat_ok = classify(module, &r2) && r2.top[0] == r.top[0];
  printf("Repeat run: %s\n", repeat_ok ? "same top-1" : "FAILED");
  const bool pass = kLabels[r.top[0]] == kTestExpected && repeat_ok;
  printf("Test_result: %s\n", pass ? "PASS" : "FAIL");

#ifdef FS_HAVE_CAMERA
  // The camera starts after the self-test, so the test above never shares memory traffic
  // with the CPI's DMA. Prints "camera init OK 320x320 RGB565" or the stage that failed.
  cycles_init();
  const bool camera = fs_camera_init();
  if (camera) {
    printf("core clock %lu Hz\n", static_cast<unsigned long>(SystemCoreClock));
    camera_probe(3000);                         // step 4: frames
    if (fs_camera_get_crop(g_crop)) {           // step 4: the hand box
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
