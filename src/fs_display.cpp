// SPDX-License-Identifier: Apache-2.0
// Fingerspell on the DevKit-E8 LCD (ILI9806E, 480x800 portrait, RGB565 via the CDC200).
//
// The UI half of fs_api.h: fs_ui_init() brings the panel up, fs_ui_render() draws one frame.
// Layout, top to bottom: title, the whole 320x320 camera frame scaled 6/5 (384x384) in a 1 px
// frame with the detected hand box and the hand skeleton (21 landmarks) over it, and a 100 px
// inset of the 128x128 crop the classifier sees in its top-right corner, the live letter in big type with its confidence as a percentage and a
// bar (green from the decoder's commit threshold up, grey below), the 2nd/3rd guesses, the
// stage timings and fps, and the text the decoder has committed. A newly committed letter is
// shown dimmed for kHoldMs before it joins that text, and a letter committed meanwhile
// replaces it (fs_commit_hold.h): a wrong letter corrected within the hold never shows.
//
// Budget (fs_api.h): fs_ui_render() must return in < 10 ms. The static parts are drawn once;
// per frame only the crop is rewritten (as whole rows, no per-pixel calls) and a text field
// only when its string changes. The CDC200 reads the frame buffer straight from SRAM0, so
// only the rows touched this frame are cleaned out of the data cache.
#include "fs_display.h"

#if defined(FINGERSPELL_LCD)
#include "fs_commit_hold.h"  // the panel's copy of the committed text, with the hold on new letters
#include "fs_detect.h"  // FS_DET_FRAME
#include <cstdio>
#include <cstring>

extern "C" {
#include "RTE_Components.h"
#include CMSIS_device_header
#include "RTE_Device.h"
#include "Driver_CDC200.h"
#include "sys_clocks.h"  // SystemREFClock
extern ARM_DRIVER_CDC200 Driver_CDC200;
}

namespace {
constexpr int kWidth = 480;
constexpr int kHeight = 800;
static_assert(RTE_PANEL_HACTIVE_TIME == kWidth && RTE_PANEL_VACTIVE_LINE == kHeight,
              "This layout requires the DK-E8 480x800 panel");
static_assert(RTE_CDC200_PIXEL_FORMAT == 2, "The framebuffer uses RGB565");

// The preview only; the model input is never flipped. kMirrorX swaps left and right
// (a mirror across the vertical axis), kFlipY top and bottom. Checked on the board with a
// hand: the sensor's own orientation already gives a mirror image, so neither is applied.
constexpr bool kMirrorX = false;
constexpr bool kFlipY = false;

// RGB565 scanout, 768000 bytes, in SRAM0 above the A32 stub with the AI pools and
// the camera ring (the linker scripts route .bss.lcd_frame_buf there).
alignas(32) uint16_t framebuffer[kWidth * kHeight]
    __attribute__((section(".bss.lcd_frame_buf")));
volatile uint32_t display_errors;
bool display_started;
bool fallback_announced;
const uint8_t* preview;
char top3_letters[3];
float top3_probs[3];
const uint16_t* detection_frame;
bool detection_hand, detection_fresh;
float detection_box[4];
bool landmarks_valid;
float landmarks_box[4];     // the crop view's hand box in frame pixels
float landmarks_frame[21][3];
// Finger chains for the skeleton: wrist -> MCP -> PIP -> DIP -> tip, plus the palm arc.
constexpr uint8_t kBones[][2] = {{0, 1}, {1, 2}, {2, 3}, {3, 4}, {0, 5}, {5, 6}, {6, 7}, {7, 8}, {5, 9}, {9, 10},
                                 {10, 11}, {11, 12}, {9, 13}, {13, 14}, {14, 15}, {15, 16}, {13, 17}, {0, 17},
                                 {17, 18}, {18, 19}, {19, 20}};

constexpr uint16_t rgb565(unsigned r, unsigned g, unsigned b) {
  return ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3);
}
constexpr uint16_t kBackground = rgb565(12, 20, 34);
constexpr uint16_t kCard = rgb565(24, 37, 56);
constexpr uint16_t kWhite = rgb565(240, 247, 255);
constexpr uint16_t kMuted = rgb565(151, 175, 199);
constexpr uint16_t kGrey = rgb565(80, 95, 115);
constexpr uint16_t kGreen = rgb565(76, 230, 168);
constexpr uint16_t kRed = rgb565(255, 113, 113);

// ---- layout (pixels) ---------------------------------------------------------------
constexpr int kTitleY = 8;
constexpr int kCropScale = 3;
constexpr int kCropSize = FS_SIZE * kCropScale;             // 384
constexpr int kCropX = (kWidth - kCropSize) / 2;            // 48
constexpr int kCropY = 44;
// The frame inset: the 320x320 camera frame subsampled 4x in the crop view's top-right.
// The main view (kCropX/Y, kCropSize: the 384 px square) shows the whole 320x320 camera frame
// scaled 6/5 with the hand box and skeleton; the 128x128 crop the classifier sees is the
// inset in its top-right corner (100 px, 1 px frame). The self-test screen still uses the
// square for the test crop at 3x (blit_crop).
constexpr int kViewNum = 6, kViewDen = 5;                        // 320 * 6 / 5 = 384
static_assert(FS_DET_FRAME * kViewNum / kViewDen == kCropSize, "the frame view must fill the square");
constexpr int kInsetSize = 100;
constexpr int kInsetX = kCropX + kCropSize - kInsetSize - 4, kInsetY = kCropY + 4;
constexpr int kLetterX = 48, kLetterY = 448, kLetterScale = 12;  // 60 x 84 px
constexpr int kPercentX = 132, kPercentY = 452, kPercentScale = 5;
constexpr int kBarX = 132, kBarY = 497, kBarW = 300, kBarH = 24;
constexpr int kGuessY = 548, kGuessScale = 3;
constexpr int kMetricsY = 580, kMetricsScale = 3;
constexpr int kWordsY = 616, kWordsH = 176, kWordsScale = 5;    // 4 rows of 14 chars
constexpr int kWordsCols = 14, kWordsRows = 4, kWordsRowH = 42;
constexpr int kWordsX = 36, kWordsTextY = kWordsY + 10;  // the first cell of the text card
// A committed letter is shown dimmed for this long before it joins the confirmed line; a letter
// committed meanwhile replaces it (fs_commit_hold.h). UI only: the decoder keeps both.
constexpr uint32_t kHoldMs = 700;

// Five columns per glyph, with seven rows encoded in the low bits (bit 0 = top row).
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
constexpr uint8_t kDot[5] = {0x00,0x60,0x60,0x00,0x00};
constexpr uint8_t kColon[5] = {0x00,0x36,0x36,0x00,0x00};
constexpr uint8_t kDash[5] = {0x08,0x08,0x08,0x08,0x08};
constexpr uint8_t kQuestion[5] = {0x02,0x01,0x51,0x09,0x06};

const uint8_t* glyph(char c) {
  if (c >= 'a' && c <= 'z') c = static_cast<char>(c - 'a' + 'A');
  if (c >= 'A' && c <= 'Z') return kLetters[c - 'A'];
  if (c >= '0' && c <= '9') return kDigits[c - '0'];
  switch (c) {
    case '%': return kPercent;
    case '.': return kDot;
    case ':': return kColon;
    case '-': return kDash;
    case '?': return kQuestion;
    default: return nullptr;  // space and anything unknown: advance only
  }
}

// Rows written since the last cache clean (inclusive), or dirty_lo > dirty_hi for none.
int dirty_lo = kHeight, dirty_hi = -1;
void mark_rows(int y0, int y1) {
  if (y0 < dirty_lo) dirty_lo = y0;
  if (y1 > dirty_hi) dirty_hi = y1;
}

void rect(int x, int y, int w, int h, uint16_t color) {
  if (x < 0) { w += x; x = 0; }
  if (y < 0) { h += y; y = 0; }
  if (x + w > kWidth) w = kWidth - x;
  if (y + h > kHeight) h = kHeight - y;
  if (w <= 0 || h <= 0) return;
  for (int row = y; row < y + h; ++row) {
    uint16_t* p = framebuffer + row * kWidth + x;
    for (int col = 0; col < w; ++col) p[col] = color;
  }
  mark_rows(y, y + h - 1);
}

// 5x7 glyphs scaled by `scale`, one scaled column of spacing between characters.
void text(int x, int y, const char* value, int scale, uint16_t color) {
  for (; *value; ++value, x += 6 * scale) {
    const uint8_t* g = glyph(*value);
    if (!g) continue;
    for (int col = 0; col < 5; ++col)
      for (int row = 0; row < 7; ++row)
        if (g[col] & (1u << row)) rect(x + col * scale, y + row * scale, scale, scale, color);
  }
}

// A text field redrawn only when its string or colour changes: the box is cleared to
// `fill` first, so a shorter string leaves no remains of the longer one.
struct Field {
  int x, y, w, h, scale;
  uint16_t fill;
  char last[48];
  uint16_t last_color;
  bool drawn;
  void set(const char* value, uint16_t color) {
    if (drawn && color == last_color && std::strncmp(value, last, sizeof(last)) == 0) return;
    rect(x, y, w, h, fill);
    text(x, y, value, scale, color);
    std::strncpy(last, value, sizeof(last) - 1);
    last[sizeof(last) - 1] = '\0';
    last_color = color;
    drawn = true;
  }
  void reset() { drawn = false; }
};

Field letter_field = {kLetterX, kLetterY, 6 * kLetterScale, 7 * kLetterScale, kLetterScale, kBackground, {}, 0, false};
Field percent_field = {kPercentX, kPercentY, 4 * 6 * kPercentScale, 7 * kPercentScale, kPercentScale, kBackground, {}, 0, false};
Field guess_field = {kCropX, kGuessY, kWidth - 2 * kCropX, 7 * kGuessScale, kGuessScale, kBackground, {}, 0, false};
Field metrics_field = {kCropX, kMetricsY, kWidth - 2 * kCropX, 7 * kMetricsScale, kMetricsScale, kBackground, {}, 0, false};
Field words_field[kWordsRows] = {
    {kWordsX, kWordsTextY + 0 * kWordsRowH, kWordsCols * 6 * kWordsScale, 7 * kWordsScale, kWordsScale, kCard, {}, 0, false},
    {kWordsX, kWordsTextY + 1 * kWordsRowH, kWordsCols * 6 * kWordsScale, 7 * kWordsScale, kWordsScale, kCard, {}, 0, false},
    {kWordsX, kWordsTextY + 2 * kWordsRowH, kWordsCols * 6 * kWordsScale, 7 * kWordsScale, kWordsScale, kCard, {}, 0, false},
    {kWordsX, kWordsTextY + 3 * kWordsRowH, kWordsCols * 6 * kWordsScale, 7 * kWordsScale, kWordsScale, kCard, {}, 0, false}};
int bar_fill = -1;
uint16_t bar_color;
bool crop_main = false;         // UP on the joystick: the crop large and the camera small instead
fs_commit_hold_t commit_hold;   // the panel's text and the letter on hold
int held_cell_drawn = -1;       // where the dimmed letter was drawn last frame (-1: none)
char held_char_drawn = 0;

// The parts that never change: title, the crop's 1 px frame, the bar's trough, the words card.
void draw_static() {
  rect(0, 0, kWidth, kHeight, kBackground);
  text((kWidth - 11 * 6 * 4) / 2, kTitleY, "FINGERSPELL", 4, kWhite);
  rect(kCropX - 1, kCropY - 1, kCropSize + 2, 1, kMuted);
  rect(kCropX - 1, kCropY + kCropSize, kCropSize + 2, 1, kMuted);
  rect(kCropX - 1, kCropY, 1, kCropSize, kMuted);
  rect(kCropX + kCropSize, kCropY, 1, kCropSize, kMuted);
  rect(kBarX, kBarY, kBarW, kBarH, kCard);
  rect(24, kWordsY, kWidth - 48, kWordsH, kCard);
  letter_field.reset();
  percent_field.reset();
  guess_field.reset();
  metrics_field.reset();
  for (auto& f : words_field) f.reset();
  bar_fill = -1;
}

// The crop pixel shown at preview position (px, py).
inline const uint8_t* source_pixel(const uint8_t* rgb, int px, int py) {
  const int x = kMirrorX ? FS_SIZE - 1 - px : px;
  const int y = kFlipY ? FS_SIZE - 1 - py : py;
  return rgb + (y * FS_SIZE + x) * 3;
}

// The 128x128 RGB888 crop scaled kCropScale times: each preview row is converted once into
// a line buffer, which is then copied to kCropScale frame-buffer rows.
void blit_crop(const uint8_t* rgb) {
  static uint16_t line[kCropSize];
  for (int sy = 0; sy < FS_SIZE; ++sy) {
    uint16_t* l = line;
    for (int sx = 0; sx < FS_SIZE; ++sx) {
      const uint8_t* px = source_pixel(rgb, sx, sy);
      const uint16_t v = rgb565(px[0], px[1], px[2]);
      for (int k = 0; k < kCropScale; ++k) *l++ = v;
    }
    for (int k = 0; k < kCropScale; ++k)
      std::memcpy(framebuffer + (kCropY + sy * kCropScale + k) * kWidth + kCropX, line, sizeof(line));
  }
  mark_rows(kCropY, kCropY + kCropSize - 1);
}

// Frame pixel -> main-view pixel (6/5), and a view coordinate clamped to the square.
inline int view_of(float frame_px) { return static_cast<int>(frame_px * kViewNum / kViewDen); }
inline int clamp_view(int v) { return v < 0 ? 0 : v > kCropSize - 1 ? kCropSize - 1 : v; }

// The whole camera frame (RGB565, copied straight into the frame buffer) scaled 6/5 into the
// main view, nearest neighbour through a column map, with the hand box on it (2 px, green
// while the detector sees the hand this frame, grey while the box is held). The view's 1 px
// frame turns red while there is no hand.
void blit_frame() {
  if (!detection_frame) return;
  static int xmap[kCropSize];
  static bool mapped = false;
  if (!mapped) {
    for (int x = 0; x < kCropSize; ++x) {
      const int sx = x * kViewDen / kViewNum;
      xmap[x] = kMirrorX ? FS_DET_FRAME - 1 - sx : sx;
    }
    mapped = true;
  }
  for (int y = 0; y < kCropSize; ++y) {
    int sy = y * kViewDen / kViewNum;
    if (kFlipY) sy = FS_DET_FRAME - 1 - sy;
    const uint16_t* src = detection_frame + sy * FS_DET_FRAME;
    uint16_t* dst = framebuffer + (kCropY + y) * kWidth + kCropX;
    for (int x = 0; x < kCropSize; ++x) dst[x] = src[xmap[x]];
  }
  mark_rows(kCropY, kCropY + kCropSize - 1);
  if (!detection_hand) return;
  int b[4];
  for (int i = 0; i < 4; ++i) b[i] = clamp_view(view_of(detection_box[i]));
  const uint16_t color = detection_fresh ? kGreen : kMuted;
  rect(kCropX + b[0], kCropY + b[1], b[2] - b[0] + 1, 2, color);
  rect(kCropX + b[0], kCropY + b[3] - 1, b[2] - b[0] + 1, 2, color);
  rect(kCropX + b[0], kCropY + b[1], 2, b[3] - b[1] + 1, color);
  rect(kCropX + b[2] - 1, kCropY + b[1], 2, b[3] - b[1] + 1, color);
}

// The square's 1 px frame: red while there is no hand.
void draw_view_edge() {
  const uint16_t edge = detection_hand ? kMuted : kRed;
  rect(kCropX - 1, kCropY - 1, kCropSize + 2, 1, edge);
  rect(kCropX - 1, kCropY + kCropSize, kCropSize + 2, 1, edge);
  rect(kCropX - 1, kCropY, 1, kCropSize, edge);
  rect(kCropX + kCropSize, kCropY, 1, kCropSize, edge);
}

// The swapped layout's inset: the whole camera frame in the 100 px inset (nearest neighbour)
// with the hand box on it; drawn over the crop view, inside its rows.
void blit_frame_inset() {
  if (!detection_frame) return;
  static int map[kInsetSize];
  static bool mapped = false;
  if (!mapped) {
    for (int i = 0; i < kInsetSize; ++i) map[i] = i * FS_DET_FRAME / kInsetSize;
    mapped = true;
  }
  const uint16_t edge = detection_hand ? kMuted : kRed;
  rect(kInsetX - 1, kInsetY - 1, kInsetSize + 2, kInsetSize + 2, edge);
  for (int y = 0; y < kInsetSize; ++y) {
    const uint16_t* src = detection_frame + map[y] * FS_DET_FRAME;
    uint16_t* dst = framebuffer + (kInsetY + y) * kWidth + kInsetX;
    for (int x = 0; x < kInsetSize; ++x) dst[x] = src[map[x]];
  }
  if (!detection_hand) return;
  int b[4];
  for (int i = 0; i < 4; ++i) {
    const int v = static_cast<int>(detection_box[i] * kInsetSize / FS_DET_FRAME);
    b[i] = v < 0 ? 0 : v > kInsetSize - 1 ? kInsetSize - 1 : v;
  }
  const uint16_t color = detection_fresh ? kGreen : kMuted;
  rect(kInsetX + b[0], kInsetY + b[1], b[2] - b[0] + 1, 1, color);
  rect(kInsetX + b[0], kInsetY + b[3], b[2] - b[0] + 1, 1, color);
  rect(kInsetX + b[0], kInsetY + b[1], 1, b[3] - b[1] + 1, color);
  rect(kInsetX + b[2], kInsetY + b[1], 1, b[3] - b[1] + 1, color);
}

// The 128x128 crop the classifier sees, in the inset (100 px, nearest neighbour) with its
// 1 px frame; drawn over the main view, inside its rows.
void blit_hand_inset(const uint8_t* rgb) {
  static int map[kInsetSize];
  static bool mapped = false;
  if (!mapped) {
    for (int i = 0; i < kInsetSize; ++i) map[i] = i * FS_SIZE / kInsetSize;
    mapped = true;
  }
  const uint16_t edge = detection_hand ? kMuted : kRed;
  rect(kInsetX - 1, kInsetY - 1, kInsetSize + 2, kInsetSize + 2, edge);
  for (int y = 0; y < kInsetSize; ++y) {
    uint16_t* dst = framebuffer + (kInsetY + y) * kWidth + kInsetX;
    for (int x = 0; x < kInsetSize; ++x) {
      const uint8_t* px = source_pixel(rgb, map[x], map[y]);
      dst[x] = rgb565(px[0], px[1], px[2]);
    }
  }
}

// A 1 px line between two frame-buffer points (clipped by rect()).
void line(int x0, int y0, int x1, int y1, uint16_t color) {
  const int dx = x1 > x0 ? x1 - x0 : x0 - x1, dy = y1 > y0 ? y1 - y0 : y0 - y1;
  const int n = dx > dy ? dx : dy;
  for (int i = 0; i <= n; ++i) {
    const int x = x0 + (n ? (x1 - x0) * i / n : 0), y = y0 + (n ? (y1 - y0) * i / n : 0);
    rect(x, y, 1, 1, color);
  }
}

// The hand skeleton on the main view and dots on the inset, drawn after both, inside their
// rows. Default layout: frame-pixel landmarks scaled 6/5 on the camera view, crop-space dots
// on the hand inset; swapped: crop-space skeleton on the crop view, frame-pixel dots on the
// camera inset.
void draw_landmarks() {
  if (!landmarks_valid) return;
  int px[21], py[21];
  const float side = landmarks_box[2] - landmarks_box[0] > 1.f ? landmarks_box[2] - landmarks_box[0] : 1.f;
  for (int i = 0; i < 21; ++i) {
    if (crop_main) {
      px[i] = kCropX + static_cast<int>((landmarks_frame[i][0] - landmarks_box[0]) * kCropSize / side);
      py[i] = kCropY + static_cast<int>((landmarks_frame[i][1] - landmarks_box[1]) * kCropSize / side);
    } else {
      px[i] = kCropX + view_of(landmarks_frame[i][0]);
      py[i] = kCropY + view_of(landmarks_frame[i][1]);
    }
  }
  auto inside = [](int x, int y) { return x >= kCropX && x < kCropX + kCropSize && y >= kCropY && y < kCropY + kCropSize; };
  for (const auto& b : kBones)
    if (inside(px[b[0]], py[b[0]]) && inside(px[b[1]], py[b[1]])) line(px[b[0]], py[b[0]], px[b[1]], py[b[1]], kGreen);
  const int dot = crop_main ? 5 : 3;
  for (int i = 0; i < 21; ++i)
    if (inside(px[i], py[i])) rect(px[i] - dot / 2, py[i] - dot / 2, dot, dot, i == 0 ? kRed : kWhite);
  for (int i = 0; i < 21; ++i) {
    int x, y;
    if (crop_main) {
      x = static_cast<int>(landmarks_frame[i][0] * kInsetSize / FS_DET_FRAME);
      y = static_cast<int>(landmarks_frame[i][1] * kInsetSize / FS_DET_FRAME);
    } else {
      x = static_cast<int>((landmarks_frame[i][0] - landmarks_box[0]) * kInsetSize / side);
      y = static_cast<int>((landmarks_frame[i][1] - landmarks_box[1]) * kInsetSize / side);
    }
    if (x < 0 || y < 0 || x >= kInsetSize || y >= kInsetSize) continue;
    rect(kInsetX + x, kInsetY + y, 1, 1, kWhite);
  }
}

void draw_confidence(char letter, float conf, bool committed_colour) {
  const char s[2] = {letter ? letter : '-', '\0'};
  letter_field.set(s, letter ? (committed_colour ? kGreen : kWhite) : kMuted);
  char pct[8];
  std::snprintf(pct, sizeof(pct), "%d%%", static_cast<int>(conf * 100.0f + 0.5f));
  percent_field.set(pct, kWhite);
  int fill = static_cast<int>(conf * kBarW + 0.5f);
  if (fill > kBarW) fill = kBarW;
  const uint16_t color = committed_colour ? kGreen : kGrey;
  if (fill != bar_fill || color != bar_color) {
    rect(kBarX, kBarY, kBarW, kBarH, kCard);
    rect(kBarX, kBarY, fill, kBarH, color);
    bar_fill = fill;
    bar_color = color;
  }
}

// The committed text, last kWordsCols * kWordsRows characters, wrapped into the card.
// `reserve` cells are kept free after the text (the dimmed letter on hold).
void draw_words(const char* words, int len, uint16_t color, int reserve = 0) {
  const int visible = kWordsCols * kWordsRows;
  const char* p = words + (len + reserve > visible ? len + reserve - visible : 0);
  for (int row = 0; row < kWordsRows; ++row) {
    char part[kWordsCols + 1] = {};
    int i = 0;
    for (; i < kWordsCols && *p; ++i) part[i] = *p++;
    words_field[row].set(part, color);
  }
}

// Push the rows written since the last call to memory for the CDC200.
void flush() {
  if (dirty_hi < dirty_lo) return;
  uint16_t* start = framebuffer + dirty_lo * kWidth;
  SCB_CleanDCache_by_Addr(start, (dirty_hi - dirty_lo + 1) * kWidth * static_cast<int>(sizeof(uint16_t)));
  __DSB();
  dirty_lo = kHeight;
  dirty_hi = -1;
}

void display_event(uint32_t event) {
  if (event & ARM_CDC_DSI_ERROR_EVENT) display_errors = event;
}

bool driver_ok(const char* stage, int32_t result) {
  if (result != ARM_DRIVER_OK) {
    std::printf("Display: CDC200 %s failed: %ld\n", stage, static_cast<long>(result));
    std::fflush(stdout);
  }
  return result == ARM_DRIVER_OK;
}

void enable_display_power() {
  // The D-PHYs are already powered by main.c (vbat_init) and by the camera; this
  // repeats it in case the display comes up first. Do not call SERVICES_set_run_cfg
  // with memory blocks here: on this board it once left MRAM/SRAM0 unusable until a
  // SETOOLS recovery (see board/DevKit-E8/README.md on hackathon-pico-faces).
  VBAT->PWR_CTRL &= ~((1u << 0) | (1u << 4) | (1u << 8) | (1u << 12));
  VBAT->PWR_CTRL &= ~((1u << 1) | (1u << 5) | (1u << 9));
}
}  // namespace

bool fs_ui_init(void) {
  if (display_started) return true;
  enable_display_power();
  fs_commit_hold_init(&commit_hold, kHoldMs);
  draw_static();
  flush();
  if (!driver_ok("Initialize", Driver_CDC200.Initialize(display_event))) return false;
  if (!driver_ok("PowerControl", Driver_CDC200.PowerControl(ARM_POWER_FULL))) return false;
  if (!driver_ok("Control", Driver_CDC200.Control(CDC200_CONFIGURE_DISPLAY,
                                                  reinterpret_cast<uint32_t>(framebuffer))))
    return false;
  if (!driver_ok("Start", Driver_CDC200.Start())) return false;
  display_started = true;
  std::printf("Display: %dx%d RGB565 panel started (frame buffer at 0x%08lx)\n", kWidth, kHeight,
              reinterpret_cast<unsigned long>(framebuffer));
  std::fflush(stdout);
  return true;
}

bool fs_display_result(const uint8_t* rgb, char predicted, float confidence,
                       char expected, bool pass) {
  if (!fs_ui_init()) return false;
  blit_crop(rgb);
  draw_confidence(predicted, confidence, pass);
  char line[48];
  std::snprintf(line, sizeof(line), "TEST IMAGE %s  EXPECTED %c", pass ? "PASS" : "FAIL", expected);
  guess_field.set(line, pass ? kGreen : kRed);
  metrics_field.set("WAITING FOR THE CAMERA", kMuted);
  draw_words("", 0, kWhite);
  flush();
  return true;
}

void fs_display_set_frame(const uint8_t* rgb) { preview = rgb; }

void fs_display_swap_views(void) { crop_main = !crop_main; }

void fs_display_set_detection(const uint16_t* frame565, bool hand, bool fresh, float x0, float y0, float x1,
                              float y1) {
  detection_frame = frame565;
  detection_hand = hand;
  detection_fresh = fresh;
  detection_box[0] = x0;
  detection_box[1] = y0;
  detection_box[2] = x1;
  detection_box[3] = y1;
}

void fs_display_set_landmarks(const fs_box_t* box, const float xyz[21][3]) {
  landmarks_valid = box != nullptr && xyz != nullptr;
  if (!landmarks_valid) return;
  landmarks_box[0] = box->x0;
  landmarks_box[1] = box->y0;
  landmarks_box[2] = box->x1;
  landmarks_box[3] = box->y1;
  for (int i = 0; i < 21; ++i)
    for (int c = 0; c < 3; ++c) landmarks_frame[i][c] = xyz[i][c];
}

void fs_display_set_top3(const char letters[3], const float probs[3]) {
  for (int i = 0; i < 3; ++i) {
    top3_letters[i] = letters[i];
    top3_probs[i] = probs[i];
  }
}

void fs_ui_render(const fs_decoder_t* d, const fs_metrics_t* m, fs_event_t last_event) {
  static float last_ui_ms;  // the previous frame's render time, shown in the metrics line
  // The 100 MHz REFCLK counter (app_main's cycles_init enables it): the M55's DWT cycle
  // counter does not run when the board boots without a debugger.
  const uint32_t t0 = REFCLK_CNTRead->CNTCVL;
  if (!display_started) {
    // UART fallback from the contract: "TEXT|live|conf|fps|npu_ms", one line per frame.
    if (!fallback_announced) {
      std::printf("Display: not started, UART fallback lines follow\n");
      fallback_announced = true;
    }
    std::printf("%s|%c|%.2f|%.1f|%.1f\n", d->text, d->live ? d->live : '-', d->live_conf, m->fps,
                m->npu_ms);
    return;
  }
  if (display_errors) {
    std::printf("Display: DSI error event 0x%lx\n", static_cast<unsigned long>(display_errors));
    display_errors = 0;
  }
  if (crop_main) {  // UP on the joystick: the classifier crop large, the camera small
    if (preview) blit_crop(preview);
    blit_frame_inset();
  } else {
    blit_frame();                           // the camera, with the hand box
    if (preview && detection_frame) blit_hand_inset(preview);  // what the classifier sees
  }
  draw_view_edge();
  draw_landmarks();

  // The model's top letter and its probability, green once the decoder holds it above
  // its commit threshold (d->live), white below.
  const char letter = top3_letters[0] ? top3_letters[0] : d->live;
  const float conf = top3_letters[0] ? top3_probs[0] : d->live_conf;
  draw_confidence(letter, conf, d->live != 0);

  char line[48];
  if (d->pending)  // a motion letter's still shape is held: say so, or the wait looks like a freeze
    std::snprintf(line, sizeof(line), "%c HELD: MOVE FOR %c  PATH %d%%", d->pending, d->pending == 'D' ? 'Z' : 'J',
                  static_cast<int>(d->pending_path * 100.0f + 0.5f));
  else if (top3_letters[1])
    std::snprintf(line, sizeof(line), "THEN %c %d%%  %c %d%%", top3_letters[1],
                  static_cast<int>(top3_probs[1] * 100.0f + 0.5f), top3_letters[2],
                  static_cast<int>(top3_probs[2] * 100.0f + 0.5f));
  else
    std::snprintf(line, sizeof(line), "SHOW A LETTER");
  guess_field.set(line, kMuted);

  std::snprintf(line, sizeof(line), "CAM %d LM %d UI %d MS  %d FPS",
                static_cast<int>(m->camera_ms + 0.5f), static_cast<int>(m->npu_ms + 0.5f),
                static_cast<int>(last_ui_ms + 0.5f), static_cast<int>(m->fps + 0.5f));
  metrics_field.set(line, kMuted);

  // The confirmed text is the panel's own copy: a committed letter is held (dimmed) for
  // kHoldMs and replaced by a letter committed meanwhile. The same 32 kHz clock as the
  // decoder's now_ms in app_main.cpp (the REFCLK-derived ms would jump every 43 s).
  const uint32_t now_ms = static_cast<uint32_t>(static_cast<uint64_t>(S32K_CNTRead->CNTCVL) * 1000u / 32768u);
  const char committed = d->len > 0 ? d->text[d->len - 1] : 0;
  const bool confirmed = fs_commit_hold_update(&commit_hold, last_event, committed, now_ms);
  const int reserve = commit_hold.held ? 1 : 0;
  const int visible = kWordsCols * kWordsRows;
  const int offset = commit_hold.len + reserve > visible ? commit_hold.len + reserve - visible : 0;
  const int cell = commit_hold.held ? commit_hold.len - offset : -1;
  if (cell != held_cell_drawn || commit_hold.held != held_char_drawn) {
    // The row strings do not change when a held letter appears, moves or goes: repaint the
    // rows it touched so no dimmed glyph lingers.
    if (held_cell_drawn >= 0) words_field[held_cell_drawn / kWordsCols].reset();
    if (cell >= 0) words_field[cell / kWordsCols].reset();
    held_cell_drawn = cell;
    held_char_drawn = commit_hold.held;
  }
  draw_words(commit_hold.shown, commit_hold.len, confirmed ? kGreen : kWhite, reserve);
  if (cell >= 0) {  // the dimmed letter, in the cell it will land in (same geometry as Field::set)
    const char held[2] = {commit_hold.held, '\0'};
    text(kWordsX + (cell % kWordsCols) * 6 * kWordsScale, kWordsTextY + (cell / kWordsCols) * kWordsRowH, held,
         kWordsScale, kMuted);
  }
  flush();
  last_ui_ms = (REFCLK_CNTRead->CNTCVL - t0) * 1000.0f / SystemREFClock;
}
#endif
