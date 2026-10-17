// SPDX-License-Identifier: Apache-2.0
#include "fs_display.h"

#if defined(FINGERSPELL_LCD)
#include <cstdio>

extern "C" {
#include "RTE_Components.h"
#include CMSIS_device_header
#include "RTE_Device.h"
#include "Driver_CDC200.h"
#include "se_services_port.h"
extern ARM_DRIVER_CDC200 Driver_CDC200;
}

namespace {
constexpr int kWidth = 480;
constexpr int kHeight = 800;
static_assert(RTE_PANEL_HACTIVE_TIME == kWidth && RTE_PANEL_VACTIVE_LINE == kHeight,
              "This layout requires the DK-E8 480x800 panel");
static_assert(RTE_CDC200_PIXEL_FORMAT == 2, "The framebuffer uses RGB565");

// RGB565 scanout uses 768000 bytes. The GNU linker puts it in SRAM1,
// separate from the AI pools and camera ring in SRAM0.
alignas(32) uint16_t framebuffer[kWidth * kHeight]
    __attribute__((section(".bss.lcd_frame_buf")));
volatile uint32_t display_errors;
bool display_started;
const uint8_t* preview;

constexpr uint16_t rgb565(unsigned r, unsigned g, unsigned b) {
  return ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3);
}
constexpr uint16_t kBackground = rgb565(12, 20, 34);
constexpr uint16_t kCard = rgb565(24, 37, 56);
constexpr uint16_t kWhite = rgb565(240, 247, 255);
constexpr uint16_t kMuted = rgb565(151, 175, 199);
constexpr uint16_t kGreen = rgb565(76, 230, 168);
constexpr uint16_t kRed = rgb565(255, 113, 113);

// Five columns per glyph, with seven rows encoded in the low bits.
constexpr uint8_t kLetters[26][5] = {
    {0x7e,0x11,0x11,0x11,0x7e}, {0x7f,0x49,0x49,0x49,0x36},
    {0x3e,0x41,0x41,0x41,0x22}, {0x7f,0x41,0x41,0x22,0x1c},
    {0x7f,0x49,0x49,0x49,0x41}, {0x7f,0x09,0x09,0x09,0x01},
    {0x3e,0x41,0x49,0x49,0x7a}, {0x7f,0x08,0x08,0x08,0x7f},
    {0x00,0x41,0x7f,0x41,0x00}, {0x20,0x40,0x41,0x3f,0x01},
    {0x7f,0x08,0x14,0x22,0x41}, {0x7f,0x40,0x40,0x40,0x40},
    {0x7f,0x02,0x0c,0x02,0x7f}, {0x7f,0x04,0x08,0x10,0x7f},
    {0x3e,0x41,0x41,0x41,0x3e}, {0x7f,0x09,0x09,0x09,0x06},
    {0x3e,0x41,0x51,0x21,0x5e}, {0x7f,0x09,0x19,0x29,0x46},
    {0x46,0x49,0x49,0x49,0x31}, {0x01,0x01,0x7f,0x01,0x01},
    {0x3f,0x40,0x40,0x40,0x3f}, {0x1f,0x20,0x40,0x20,0x1f},
    {0x3f,0x40,0x38,0x40,0x3f}, {0x63,0x14,0x08,0x14,0x63},
    {0x07,0x08,0x70,0x08,0x07}, {0x61,0x51,0x49,0x45,0x43}};
constexpr uint8_t kDigits[10][5] = {
    {0x3e,0x51,0x49,0x45,0x3e}, {0x00,0x42,0x7f,0x40,0x00},
    {0x42,0x61,0x51,0x49,0x46}, {0x21,0x41,0x45,0x4b,0x31},
    {0x18,0x14,0x12,0x7f,0x10}, {0x27,0x45,0x45,0x45,0x39},
    {0x3c,0x4a,0x49,0x49,0x30}, {0x01,0x71,0x09,0x05,0x03},
    {0x36,0x49,0x49,0x49,0x36}, {0x06,0x49,0x49,0x29,0x1e}};
constexpr uint8_t kPercent[5] = {0x63,0x13,0x08,0x64,0x63};

void rect(int x, int y, int w, int h, uint16_t color) {
  for (int row = y; row < y + h && row < kHeight; ++row)
    for (int col = x; col < x + w && col < kWidth; ++col)
      if (row >= 0 && col >= 0) framebuffer[row * kWidth + col] = color;
}

void text(int x, int y, const char* value, int scale, uint16_t color) {
  for (; *value; ++value, x += 6 * scale) {
    const uint8_t* glyph = nullptr;
    if (*value >= 'A' && *value <= 'Z') glyph = kLetters[*value - 'A'];
    if (*value >= '0' && *value <= '9') glyph = kDigits[*value - '0'];
    if (*value == '%') glyph = kPercent;
    if (!glyph) continue;
    for (int col = 0; col < 5; ++col)
      for (int row = 0; row < 7; ++row)
        if (glyph[col] & (1u << row))
          rect(x + col * scale, y + row * scale, scale, scale, color);
  }
}

void render(const uint8_t* rgb, char predicted, float confidence, char expected, bool pass) {
  const uint16_t accent = pass ? kGreen : kRed;
  rect(0, 0, kWidth, kHeight, kBackground);
  text(42, 28, "FINGERSPELL", 6, kWhite);
  text(42, 87, "ON DEVICE AI", 3, kGreen);
  rect(108, 137, 264, 264, kMuted);
  for (int y = 0; y < 128; ++y) {
    for (int x = 0; x < 128; ++x) {
      const uint8_t* p = rgb + (y * 128 + x) * 3;
      rect(112 + 2*x, 141 + 2*y, 2, 2, rgb565(p[0], p[1], p[2]));
    }
  }
  text(114, 418, "TEST IMAGE", 3, kMuted);
  rect(24, 463, 432, 222, kCard);
  const char letter[] = {predicted, '\0'};
  text(54, 488, letter, 24, kWhite);
  text(223, 490, "RESULT", 4, kMuted);
  char percentage[8];
  const int percent = static_cast<int>(confidence * 100.0f + 0.5f);
  std::snprintf(percentage, sizeof(percentage), "%d%%", percent);
  text(223, 537, percentage, 6, kWhite);
  text(223, 600, "CONFIDENCE", 2, kMuted);
  rect(223, 631, 186, 36, accent);
  text(280, 638, pass ? "PASS" : "FAIL", 3, kBackground);
  char expected_text[] = "EXPECTED  X";
  expected_text[10] = expected;
  text(110, 713, expected_text, 4, accent);
  text(70, 768, "SAVED IMAGE DEMO", 3, kMuted);
}

void display_event(uint32_t event) {
  if (event & ARM_CDC_DSI_ERROR_EVENT) display_errors = event;
}

bool driver_ok(const char* stage, int32_t result) {
  std::printf("LCD %s: %ld\n", stage, static_cast<long>(result));
  std::fflush(stdout);
  return result == ARM_DRIVER_OK;
}

bool enable_display_power() {
  uint32_t service_error = 0;
  run_profile_t profile = {};
  uint32_t status = SERVICES_get_run_cfg(se_services_s_handle, &profile, &service_error);
  if (status || service_error) {
    std::printf("LCD get power profile failed: %lu / %lu\n",
                static_cast<unsigned long>(status), static_cast<unsigned long>(service_error));
    return false;
  }
  // Preserve the NPU/CPU memory and clocks already enabled by board startup.
  profile.memory_blocks |= MRAM_MASK | SRAM0_MASK | SRAM1_MASK;
  profile.phy_pwr_gating |= MIPI_PLL_DPHY_MASK | MIPI_TX_DPHY_MASK |
                            MIPI_RX_DPHY_MASK | LDO_PHY_MASK;
  status = SERVICES_set_run_cfg(se_services_s_handle, &profile, &service_error);
  if (status || service_error) {
    std::printf("LCD set power profile failed: %lu / %lu\n",
                static_cast<unsigned long>(status), static_cast<unsigned long>(service_error));
    return false;
  }
  // MIPI-only portion of Alif's DevKit-E8 vbat_init() example.
  VBAT->PWR_CTRL &= ~((1u << 0) | (1u << 4) | (1u << 8) | (1u << 12));
  VBAT->PWR_CTRL &= ~((1u << 1) | (1u << 5) | (1u << 9));
  return true;
}
}  // namespace

bool fs_ui_init(void) {
  if (display_started) return true;
  if (!enable_display_power()) return false;
  rect(0, 0, kWidth, kHeight, kBackground);
  SCB_CleanDCache_by_Addr(framebuffer, sizeof(framebuffer));
  __DSB();
  if (!driver_ok("initialize", Driver_CDC200.Initialize(display_event))) return false;
  if (!driver_ok("power", Driver_CDC200.PowerControl(ARM_POWER_FULL))) return false;
  if (!driver_ok("configure", Driver_CDC200.Control(
          CDC200_CONFIGURE_DISPLAY, reinterpret_cast<uint32_t>(framebuffer)))) return false;
  if (!driver_ok("start", Driver_CDC200.Start())) return false;
  display_started = true;
  return true;
}

bool fs_display_result(const uint8_t* rgb, char predicted, float confidence,
                       char expected, bool pass) {
  if (!fs_ui_init()) return false;
  render(rgb, predicted, confidence, expected, pass);
  SCB_CleanDCache_by_Addr(framebuffer, sizeof(framebuffer));
  __DSB();
  std::printf("LCD ready: %c, expected %c, %s (saved image)\n",
              predicted, expected, pass ? "PASS" : "FAIL");
  std::fflush(stdout);
  return true;
}

void fs_display_set_frame(const uint8_t* rgb) { preview = rgb; }

void fs_ui_render(const fs_decoder_t* d, const fs_metrics_t* m, fs_event_t last_event) {
  const auto& decoder = *d;
  if (!display_started) {
    std::printf("%s|%c|%.2f|%.1f|%.1f\n", decoder.text,
                decoder.live ? decoder.live : '-', decoder.live_conf, m->fps, m->npu_ms);
    return;
  }
  const uint8_t* rgb = preview;
  if (display_errors) {
    std::printf("LCD DSI error event: 0x%lx\n", static_cast<unsigned long>(display_errors));
    display_errors = 0;
  }
  rect(0, 0, kWidth, kHeight, kBackground);
  text(42, 22, "FINGERSPELL", 6, kWhite);
  text(36, 78, "BOARD CAMERA  E8 NPU", 3, kGreen);
  rect(108, 111, 264, 264, kMuted);
  if (rgb) {
    for (int y = 0; y < 128; ++y)
      for (int x = 0; x < 128; ++x) {
        const auto* p = rgb + (y * 128 + x) * 3;
        rect(112 + 2*x, 115 + 2*y, 2, 2, rgb565(p[0], p[1], p[2]));
      }
  } else {
    rect(112, 115, 256, 256, kCard);
    text(143, 220, "NO IMAGE", 3, kMuted);
  }
  char state[40];
  if (decoder.live) {
    std::snprintf(state, sizeof(state), "LIVE %c  %d%%", decoder.live,
                  static_cast<int>(decoder.live_conf * 100 + 0.5f));
  } else {
    std::snprintf(state, sizeof(state), "SHOW A LETTER");
  }
  text(28, 400, state, 3, decoder.live ? kGreen : kMuted);
  text(30, 448, "YOUR WORDS", 3, kMuted);
  rect(24, 482, 432, 215, kCard);
  const char* visible = decoder.text + (decoder.len > 52 ? decoder.len - 52 : 0);
  if (!*visible) text(48, 520, "SPELL A WORD", 5, kMuted);
  for (int row = 0; row < 4 && *visible; ++row) {
    char part[14] = {};
    int i = 0;
    for (; i < 13 && *visible; ++i) part[i] = *visible++;
    text(44, 498 + row * 49, part, 5, last_event == FS_EV_LETTER ? kGreen : kWhite);
  }
  std::snprintf(state, sizeof(state), "NPU %d MS  FPS %d", static_cast<int>(m->npu_ms + 0.5f),
                static_cast<int>(m->fps + 0.5f));
  text(30, 720, state, 3, kMuted);
  text(66, 766, "HOLD A LETTER", 4, kMuted);
  // The controller reads SRAM directly, so publish the CPU's drawing first.
  SCB_CleanDCache_by_Addr(framebuffer, sizeof(framebuffer));
  __DSB();
}
#endif
