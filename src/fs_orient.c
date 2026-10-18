// fs_orient.c: see fs_orient.h.
#include "fs_orient.h"

#include <math.h>

void fs_orient_reset(fs_orient_t* s) {
  s->g[0] = s->g[1] = s->g[2] = 0.0f;
  s->valid = false;
}

void fs_orient_gravity(fs_orient_t* s, const float g[3], float alpha) {
  if (!s->valid) {
    for (int i = 0; i < 3; ++i) s->g[i] = g[i];
    s->valid = true;
    return;
  }
  for (int i = 0; i < 3; ++i) s->g[i] += alpha * (g[i] - s->g[i]);
}

static float dot3(const float a[3], const float b[3]) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

float fs_orient_roll(const fs_orient_cal_t* cal, const float g[3]) {
  if (sqrtf(dot3(g, g)) < FS_ORIENT_MIN_G) return 0.0f;
  const float x = dot3(g, cal->right), y = dot3(g, cal->down);
  if (sqrtf(x * x + y * y) < FS_ORIENT_MIN_INPLANE) return 0.0f;  // uncalibrated, or looking along gravity
  return atan2f(x, y);
}

void fs_orient_level(float xyz[][3], int count, float roll, const float about[2]) {
  if (roll == 0.0f) return;
  const float c = cosf(roll), s = sinf(roll);
  for (int i = 0; i < count; ++i) {
    const float x = xyz[i][0] - about[0], y = xyz[i][1] - about[1];
    xyz[i][0] = about[0] + c * x - s * y;
    xyz[i][1] = about[1] + s * x + c * y;
  }
}
