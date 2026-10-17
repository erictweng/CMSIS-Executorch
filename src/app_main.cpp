// Fingerspell — app_main.cpp for Arm-Examples/CMSIS-Executorch@hackathon (replaces src/app_main.cpp).
//
// G1 board test: runs the 24-letter model on a baked-in 128x128 RGB test crop (test_image.h, made by
// ml/make_test_image.py from the held-out split) and prints the top-3 letters plus the expected one.
// Tomorrow the camera frame replaces kTestImage; preprocess_rgb888() is the contract for that.
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

using arm::embedded::EmbeddedModule;
using executorch::aten::DimOrderType;
using executorch::aten::ScalarType;
using executorch::aten::SizesType;
using executorch::aten::Tensor;
using executorch::aten::TensorImpl;
using executorch::extension::BufferDataLoader;
using executorch::runtime::EValue;
using executorch::runtime::MemoryAllocator;

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

// uint8 RGB888 (HWC) -> float [0,1] NHWC. The camera path will call this on the cropped box.
void preprocess_rgb888(const uint8_t* rgb, float* out) {
  for (int i = 0; i < kSize * kSize * 3; ++i) out[i] = rgb[i] * (1.0f / 255.0f);
}

}  // namespace

extern "C" int app_main(void) {
  executorch::runtime::runtime_init();
  printf("Fingerspell letters: %lu byte model\n", model_pte_size);

  EmbeddedModule module(
      model_pte, model_pte_size,
      std::make_unique<BufferDataLoader>(model_pte, model_pte_size),
      std::make_unique<MemoryAllocator>(kMethodPoolSize, g_method_pool),
      std::make_unique<MemoryAllocator>(kTempPoolSize, g_temp_pool));

  preprocess_rgb888(kTestImage, g_input);
  std::array<SizesType, 4> sizes{1, kSize, kSize, 3};
  std::array<DimOrderType, 4> dim_order{0, 1, 2, 3};
  TensorImpl input_impl(ScalarType::Float, sizes.size(), sizes.data(), g_input, dim_order.data());
  Tensor input(&input_impl);

  auto outputs = module.forward(input);
  if (!outputs.ok() || outputs->empty() || !outputs->front().isTensor()) {
    printf("forward failed (err=%u)\n", outputs.ok() ? 0u : static_cast<unsigned>(outputs.error()));
    return 1;
  }
  Tensor t = outputs->front().toTensor();
  if (t.numel() != kClasses || t.scalar_type() != ScalarType::Float) {
    printf("unexpected output: %u elements\n", static_cast<unsigned>(t.numel()));
    return 1;
  }
  const float* logits = t.const_data_ptr<float>();

  // softmax + top-3
  float mx = logits[0];
  for (int i = 1; i < kClasses; ++i) mx = logits[i] > mx ? logits[i] : mx;
  float prob[kClasses], sum = 0.f;
  for (int i = 0; i < kClasses; ++i) sum += (prob[i] = std::exp(logits[i] - mx));
  int top[3] = {-1, -1, -1};
  for (int k = 0; k < 3; ++k)
    for (int i = 0; i < kClasses; ++i)
      if ((k == 0 || i != top[0]) && (k < 2 || i != top[1]) && (top[k] < 0 || prob[i] > prob[top[k]])) top[k] = i;

  printf("Top-3: %c %.2f | %c %.2f | %c %.2f   (expected %c)\n", kLabels[top[0]], prob[top[0]] / sum,
         kLabels[top[1]], prob[top[1]] / sum, kLabels[top[2]], prob[top[2]] / sum, kTestExpected);
  bool pass = kLabels[top[0]] == kTestExpected;
  printf("Test_result: %s\n", pass ? "PASS" : "FAIL");
  printf("\x04");
  fflush(stdout);
  return pass ? 0 : 1;
}
