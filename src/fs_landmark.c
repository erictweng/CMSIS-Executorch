// fs_landmark.c: see fs_landmark.h; numbers and order follow ml/handlandmark.py.
#include "fs_landmark.h"

#include <math.h>
#include <string.h>

#define KP_WRIST 0
#define KP_MIDDLE_MCP 2  // the palm detector's keypoint order

static inline void rgb565_to_888f(uint16_t v, float* r, float* g, float* b) {
  const unsigned r5 = (v >> 11) & 0x1Fu, g6 = (v >> 5) & 0x3Fu, b5 = v & 0x1Fu;
  *r = (float)((r5 << 3) | (r5 >> 2));
  *g = (float)((g6 << 2) | (g6 >> 4));
  *b = (float)((b5 << 3) | (b5 >> 2));
}

fs_lm_crop_t fs_landmark_crop_frame(const fs_detection_t* det, float input_to_frame) {
  const fs_box_t* p = &det->palm;
  const float w = (p->x1 - p->x0) * input_to_frame, h = (p->y1 - p->y0) * input_to_frame;
  const float size = w > h ? w : h;
  float ux = det->keypoints[KP_MIDDLE_MCP][0] - det->keypoints[KP_WRIST][0];
  float uy = det->keypoints[KP_MIDDLE_MCP][1] - det->keypoints[KP_WRIST][1];
  const float len = sqrtf(ux * ux + uy * uy);
  if (len > 1e-3f) {
    ux /= len;
    uy /= len;
  } else {
    ux = 0.0f;
    uy = -1.0f;
  }
  fs_lm_crop_t c;
  c.cx = (p->x0 + p->x1) / 2 * input_to_frame + FS_LM_CROP_SHIFT * size * ux;
  c.cy = (p->y0 + p->y1) / 2 * input_to_frame + FS_LM_CROP_SHIFT * size * uy;
  c.side = size * FS_LM_CROP_SCALE;
  if (c.side > FS_LM_MAX_SIDE) c.side = FS_LM_MAX_SIDE;
  if (c.side < 1.0f) c.side = 1.0f;
  c.rx = -uy;  // right = up rotated by +90 degrees in image coordinates
  c.ry = ux;
  c.dx = -ux;  // down = -up
  c.dy = -uy;
  return c;
}

// Bilinear sample at frame position (fx, fy) in pixel-centre coordinates; black off-frame.
static inline void sample(const uint16_t* frame, float fx, float fy, float* r, float* g, float* b) {
  const float px = fx - 0.5f, py = fy - 0.5f;
  const int x0 = (int)floorf(px), y0 = (int)floorf(py);
  const float ax = px - (float)x0, ay = py - (float)y0;
  *r = *g = *b = 0.0f;
  for (int j = 0; j < 2; ++j) {
    const int y = y0 + j;
    if (y < 0 || y >= FS_DET_FRAME) continue;
    const float wy = j ? ay : 1.0f - ay;
    for (int i = 0; i < 2; ++i) {
      const int x = x0 + i;
      if (x < 0 || x >= FS_DET_FRAME) continue;
      const float w = wy * (i ? ax : 1.0f - ax);
      float cr, cg, cb;
      rgb565_to_888f(frame[y * FS_DET_FRAME + x], &cr, &cg, &cb);
      *r += w * cr;
      *g += w * cg;
      *b += w * cb;
    }
  }
}

static inline int8_t quantize(float v, float k, int zero_point) {
  int q = (int)(v * k + 0.5f) + zero_point;  // v >= 0
  return (int8_t)(q < -128 ? -128 : q > 127 ? 127 : q);
}

void fs_landmark_crop(const uint16_t* frame, const fs_lm_crop_t* c, int8_t* out, fs_quant_t input) {
  const float s = c->side / FS_LM_SIZE;
  const float k = 1.0f / (255.0f * input.scale);
  for (int v = 0; v < FS_LM_SIZE; ++v) {
    const float dv = ((float)v + 0.5f - FS_LM_SIZE / 2) * s;
    const float ox = c->cx + dv * c->dx, oy = c->cy + dv * c->dy;
    int8_t* o = out + (size_t)v * FS_LM_SIZE * 3;
    for (int u = 0; u < FS_LM_SIZE; ++u, o += 3) {
      const float du = ((float)u + 0.5f - FS_LM_SIZE / 2) * s;
      float r, g, b;
      sample(frame, ox + du * c->rx, oy + du * c->ry, &r, &g, &b);
      o[0] = quantize(r, k, input.zero_point);
      o[1] = quantize(g, k, input.zero_point);
      o[2] = quantize(b, k, input.zero_point);
    }
  }
}

void fs_landmark_decode(const int8_t lm63[FS_LM_FEATURES], fs_quant_t lq, int8_t presence, fs_quant_t pq,
                        int8_t handedness, fs_quant_t hq, const fs_lm_crop_t* c, fs_landmarks_t* out) {
  const float s = c->side / FS_LM_SIZE;
  for (int i = 0; i < FS_LM_COUNT; ++i) {
    const float x = (float)(lm63[3 * i] - lq.zero_point) * lq.scale - FS_LM_SIZE / 2;
    const float y = (float)(lm63[3 * i + 1] - lq.zero_point) * lq.scale - FS_LM_SIZE / 2;
    const float z = (float)(lm63[3 * i + 2] - lq.zero_point) * lq.scale;
    out->xyz[i][0] = c->cx + (x * c->rx + y * c->dx) * s;
    out->xyz[i][1] = c->cy + (x * c->ry + y * c->dy) * s;
    out->xyz[i][2] = z * s;
  }
  out->presence = (float)(presence - pq.zero_point) * pq.scale;
  out->right = (float)(handedness - hq.zero_point) * hq.scale;
}

void fs_landmark_features(const fs_landmarks_t* lm, float f[FS_LM_FEATURES]) {
  const float* w = lm->xyz[FS_LM_WRIST];
  const float mx = lm->xyz[FS_LM_MIDDLE_MCP][0] - w[0], my = lm->xyz[FS_LM_MIDDLE_MCP][1] - w[1];
  float unit = sqrtf(mx * mx + my * my);
  if (unit <= 1e-6f) unit = 1.0f;
  const float mirror = lm->right >= 0.5f ? 1.0f : -1.0f;
  for (int i = 0; i < FS_LM_COUNT; ++i) {
    f[3 * i] = (lm->xyz[i][0] - w[0]) / unit * mirror;
    f[3 * i + 1] = (lm->xyz[i][1] - w[1]) / unit;
    f[3 * i + 2] = (lm->xyz[i][2] - w[2]) / unit;
  }
}

void fs_landmark_smooth_reset(fs_lm_smooth_t* s) { s->valid = false; }

void fs_landmark_smooth(fs_lm_smooth_t* s, const float f[FS_LM_FEATURES], float alpha, float out[FS_LM_FEATURES]) {
  if (!s->valid) {
    memcpy(s->f, f, sizeof(s->f));
    s->valid = true;
  } else {
    for (int i = 0; i < FS_LM_FEATURES; ++i) s->f[i] += alpha * (f[i] - s->f[i]);
  }
  memcpy(out, s->f, sizeof(s->f));
}

void fs_landmark_letters(const float f[FS_LM_FEATURES], const float* W1, const float* b1, const float* W2,
                         const float* b2, int hidden, float probs[FS_CLASSES]) {
  float h[256];
  if (hidden > 256) hidden = 256;
  for (int j = 0; j < hidden; ++j) {
    float a = b1[j];
    const float* w = W1 + (size_t)j * FS_LM_FEATURES;
    for (int i = 0; i < FS_LM_FEATURES; ++i) a += w[i] * f[i];
    h[j] = a > 0.0f ? a : 0.0f;
  }
  float mx = -1e30f;
  for (int k = 0; k < FS_CLASSES; ++k) {
    float z = b2[k];
    const float* w = W2 + (size_t)k * hidden;
    for (int j = 0; j < hidden; ++j) z += w[j] * h[j];
    probs[k] = z;
    if (z > mx) mx = z;
  }
  float sum = 0.0f;
  for (int k = 0; k < FS_CLASSES; ++k) sum += (probs[k] = expf(probs[k] - mx));
  for (int k = 0; k < FS_CLASSES; ++k) probs[k] /= sum;
}
