// fs_detect.c: the palm detector's CPU side (see fs_detect.h). Numbers and order follow
// ml/blazepalm.py, which follows MediaPipe's palm_detection_lite graph.
#include "fs_detect.h"

#include <math.h>
#include <string.h>

// ---- anchors ----------------------------------------------------------------------
// MediaPipe's ssd_anchors_calculator for the 192 lite model: layer strides 8, 16, 16, 16
// with aspect ratio 1 and one interpolated scale per layer. Stride 8 is one layer of
// 24x24 cells with 2 anchors each; the three stride-16 layers merge into 12x12 cells with
// 6 anchors each. fixed_anchor_size makes every anchor 1x1, so only the centre matters,
// and the head emits anchors in (y, x, anchor) order: the index decodes directly.
#define FS_DET_S8_CELLS 24
#define FS_DET_S8_PER_CELL 2
#define FS_DET_S8_ANCHORS (FS_DET_S8_CELLS * FS_DET_S8_CELLS * FS_DET_S8_PER_CELL)  // 1152
#define FS_DET_S16_CELLS 12
#define FS_DET_S16_PER_CELL 6

void fs_detect_anchor(int i, float* cx, float* cy) {
  int cells, cell;
  if (i < FS_DET_S8_ANCHORS) {
    cells = FS_DET_S8_CELLS;
    cell = i / FS_DET_S8_PER_CELL;
  } else {
    cells = FS_DET_S16_CELLS;
    cell = (i - FS_DET_S8_ANCHORS) / FS_DET_S16_PER_CELL;
  }
  *cx = ((float)(cell % cells) + 0.5f) / (float)cells;
  *cy = ((float)(cell / cells) + 0.5f) / (float)cells;
}

// ---- RGB565 -> RGB888 (as fs_camera.c: top bits replicated so full scale is 255) ----
static inline void rgb565_to_888(uint16_t v, unsigned* r, unsigned* g, unsigned* b) {
  const unsigned r5 = (v >> 11) & 0x1Fu, g6 = (v >> 5) & 0x3Fu, b5 = v & 0x1Fu;
  *r = (r5 << 3) | (r5 >> 2);
  *g = (g6 << 2) | (g6 >> 4);
  *b = (b5 << 3) | (b5 >> 2);
}

// ---- 320 -> 192: exact 5:3 box filter -------------------------------------------------
// Three consecutive outputs span five inputs: output 0 covers input 0 and 2/3 of input 1,
// output 1 covers 1/3 of input 1, input 2 and 1/3 of input 3, output 2 covers 2/3 of input
// 3 and input 4. With the weights x3 that is 3,2 | 1,3,1 | 2,3 (each sums to 5), applied
// horizontally and vertically: every output is a weighted sum of up to 9 pixels over 25.

// One input row, horizontally 320 -> 192, weights x3 (values up to 5 x 255 = 1275).
static void hrow_53(const uint16_t* row, uint16_t* out) {
  for (int m = 0; m < FS_DET_SIZE / 3; ++m, row += 5, out += 9) {
    unsigned r[5], g[5], b[5];
    for (int k = 0; k < 5; ++k) rgb565_to_888(row[k], &r[k], &g[k], &b[k]);
    out[0] = (uint16_t)(3 * r[0] + 2 * r[1]);
    out[1] = (uint16_t)(3 * g[0] + 2 * g[1]);
    out[2] = (uint16_t)(3 * b[0] + 2 * b[1]);
    out[3] = (uint16_t)(r[1] + 3 * r[2] + r[3]);
    out[4] = (uint16_t)(g[1] + 3 * g[2] + g[3]);
    out[5] = (uint16_t)(b[1] + 3 * b[2] + b[3]);
    out[6] = (uint16_t)(2 * r[3] + 3 * r[4]);
    out[7] = (uint16_t)(2 * g[3] + 3 * g[4]);
    out[8] = (uint16_t)(2 * b[3] + 3 * b[4]);
  }
}

// q = round(average / 255 / scale) + zero_point, average = acc / 25.
static inline int8_t quantize_acc(unsigned acc, float k, int zero_point) {
  int q = (int)((float)acc * k + 0.5f) + zero_point;
  return (int8_t)(q < -128 ? -128 : q > 127 ? 127 : q);
}

void fs_detect_preprocess(const uint16_t* frame, int8_t* out, fs_quant_t input) {
  static uint16_t h[5][FS_DET_SIZE * 3];  // five input rows, horizontally filtered
  const float k = 1.0f / (25.0f * 255.0f * input.scale);
  const int zp = input.zero_point;
  for (int n = 0; n < FS_DET_SIZE / 3; ++n) {
    for (int t = 0; t < 5; ++t) hrow_53(frame + (size_t)(5 * n + t) * FS_DET_FRAME, h[t]);
    int8_t* o0 = out + (size_t)(3 * n) * FS_DET_SIZE * 3;
    int8_t* o1 = o0 + FS_DET_SIZE * 3;
    int8_t* o2 = o1 + FS_DET_SIZE * 3;
    for (int i = 0; i < FS_DET_SIZE * 3; ++i) {
      o0[i] = quantize_acc(3u * h[0][i] + 2u * h[1][i], k, zp);
      o1[i] = quantize_acc(h[1][i] + 3u * h[2][i] + h[3][i], k, zp);
      o2[i] = quantize_acc(2u * h[3][i] + 3u * h[4][i], k, zp);
    }
  }
}

// ---- decode -----------------------------------------------------------------------
static float sigmoid(float x) {
  if (x > 100.0f) x = 100.0f;
  if (x < -100.0f) x = -100.0f;
  return 1.0f / (1.0f + expf(-x));
}

bool fs_detect_decode(const int8_t* regressors, fs_quant_t rq, const int8_t* logits, fs_quant_t lq,
                      fs_detection_t* det) {
  // argmax of the int8 logits is the argmax of the sigmoid (the scale is positive), and
  // sigmoid(x) >= 0.5 <=> x >= 0 <=> q >= zero_point.
  int best = 0;
  for (int i = 1; i < FS_DET_ANCHORS; ++i)
    if (logits[i] > logits[best]) best = i;
  float acx, acy;
  fs_detect_anchor(best, &acx, &acy);
  const int8_t* q = regressors + (size_t)best * FS_DET_COORDS;
  float r[FS_DET_COORDS];
  for (int i = 0; i < FS_DET_COORDS; ++i) r[i] = (float)(q[i] - rq.zero_point) * rq.scale;
  const float inv = 1.0f / (float)FS_DET_SIZE;
  const float cx = r[0] * inv + acx, cy = r[1] * inv + acy, w = r[2] * inv, h = r[3] * inv;
  det->palm.x0 = (cx - w / 2) * FS_DET_SIZE;
  det->palm.y0 = (cy - h / 2) * FS_DET_SIZE;
  det->palm.x1 = (cx + w / 2) * FS_DET_SIZE;
  det->palm.y1 = (cy + h / 2) * FS_DET_SIZE;
  for (int k = 0; k < FS_DET_KEYPOINTS; ++k) {
    det->keypoints[k][0] = (r[4 + 2 * k] * inv + acx) * FS_DET_SIZE;
    det->keypoints[k][1] = (r[5 + 2 * k] * inv + acy) * FS_DET_SIZE;
  }
  det->score = sigmoid((float)(logits[best] - lq.zero_point) * lq.scale);
  det->anchor = best;
  return logits[best] >= lq.zero_point;
}

// ---- palm box -> hand box ------------------------------------------------------------
fs_box_t fs_detect_hand_box(const fs_box_t* palm, float input_to_frame) {
  const float w = (palm->x1 - palm->x0) * input_to_frame, h = (palm->y1 - palm->y0) * input_to_frame;
  const float cx = (palm->x0 + palm->x1) / 2 * input_to_frame;
  const float cy = (palm->y0 + palm->y1) / 2 * input_to_frame + FS_DET_BOX_SHIFT_Y * h;
  float side = (w > h ? w : h) * FS_DET_BOX_SCALE;
  if (side > FS_DET_MAX_SIDE) side = FS_DET_MAX_SIDE;
  if (side < 1.0f) side = 1.0f;
  fs_box_t b = {cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2};
  return b;
}

// ---- temporal smoothing ---------------------------------------------------------------
void fs_track_reset(fs_track_t* t) { memset(t, 0, sizeof(*t)); }

void fs_track_update(fs_track_t* t, bool hit, const fs_box_t* box, float score) {
  if (!hit) {
    t->fresh = false;
    if (t->hand && ++t->misses >= FS_DET_HOLD_FRAMES) t->hand = false;
    return;
  }
  if (t->hand) {
    const float a = FS_DET_EMA, b = 1.0f - FS_DET_EMA;
    t->box.x0 = a * box->x0 + b * t->box.x0;
    t->box.y0 = a * box->y0 + b * t->box.y0;
    t->box.x1 = a * box->x1 + b * t->box.x1;
    t->box.y1 = a * box->y1 + b * t->box.y1;
  } else {
    t->box = *box;
  }
  t->hand = true;
  t->fresh = true;
  t->misses = 0;
  t->score = score;
}

// ---- hand-box crop: arbitrary square box -> 128x128, box filter, black off-frame ----
// Per output coordinate, the input pixels it covers and their coverage weights (sum 1).
// FS_DET_MAX_SIDE / FS_SIZE = 5 pixels per output at most, so 7 taps cover any alignment.
#define FS_DET_MAX_TAPS 7
typedef struct {
  int first;
  int count;
  float w[FS_DET_MAX_TAPS];
} fs_taps_t;

static void make_taps(float origin, float scale, fs_taps_t taps[FS_SIZE]) {
  for (int k = 0; k < FS_SIZE; ++k) {
    const float a = origin + k * scale, b = a + scale;
    int p0 = (int)floorf(a), p1 = (int)ceilf(b);
    if (p1 <= p0) p1 = p0 + 1;
    if (p1 - p0 > FS_DET_MAX_TAPS) p1 = p0 + FS_DET_MAX_TAPS;
    taps[k].first = p0;
    taps[k].count = p1 - p0;
    for (int p = p0; p < p1; ++p) {
      const float lo = a > (float)p ? a : (float)p, hi = b < (float)(p + 1) ? b : (float)(p + 1);
      taps[k].w[p - p0] = hi > lo ? (hi - lo) / scale : 0.0f;
    }
  }
}

// One frame row through the horizontal taps; pixels off the frame count as black.
static void hrow_box(const uint16_t* row, const fs_taps_t taps[FS_SIZE], float* out) {
  for (int k = 0; k < FS_SIZE; ++k) {
    float r = 0, g = 0, b = 0;
    for (int t = 0; t < taps[k].count; ++t) {
      const int x = taps[k].first + t;
      if (x < 0 || x >= FS_DET_FRAME) continue;
      unsigned pr, pg, pb;
      rgb565_to_888(row[x], &pr, &pg, &pb);
      r += taps[k].w[t] * pr;
      g += taps[k].w[t] * pg;
      b += taps[k].w[t] * pb;
    }
    out[3 * k] = r;
    out[3 * k + 1] = g;
    out[3 * k + 2] = b;
  }
}

void fs_detect_crop(const uint16_t* frame, const fs_box_t* box, uint8_t rgb[FS_SIZE * FS_SIZE * 3]) {
  static fs_taps_t xtaps[FS_SIZE], ytaps[FS_SIZE];
  static float hrows[2][FS_SIZE * 3];
  int hrow_index[2] = {-1, -1};
  const float side = box->x1 - box->x0 > box->y1 - box->y0 ? box->x1 - box->x0 : box->y1 - box->y0;
  const float scale = (side > 1.0f ? side : 1.0f) / FS_SIZE;
  make_taps(box->x0, scale, xtaps);
  make_taps(box->y0, scale, ytaps);
  for (int i = 0; i < FS_SIZE; ++i) {
    float acc[FS_SIZE * 3];
    memset(acc, 0, sizeof(acc));
    for (int t = 0; t < ytaps[i].count; ++t) {
      const int y = ytaps[i].first + t;
      if (y < 0 || y >= FS_DET_FRAME) continue;
      // Consecutive output rows share at most their boundary input rows: two cached rows.
      int slot = hrow_index[0] == y ? 0 : hrow_index[1] == y ? 1 : -1;
      if (slot < 0) {
        slot = hrow_index[0] < hrow_index[1] ? 0 : 1;  // replace the older row
        hrow_box(frame + (size_t)y * FS_DET_FRAME, xtaps, hrows[slot]);
        hrow_index[slot] = y;
      }
      const float w = ytaps[i].w[t];
      const float* h = hrows[slot];
      for (int k = 0; k < FS_SIZE * 3; ++k) acc[k] += w * h[k];
    }
    uint8_t* o = rgb + (size_t)i * FS_SIZE * 3;
    for (int k = 0; k < FS_SIZE * 3; ++k) {
      const float v = acc[k] + 0.5f;
      o[k] = (uint8_t)(v < 0.0f ? 0 : v > 255.0f ? 255 : (int)v);
    }
  }
}
