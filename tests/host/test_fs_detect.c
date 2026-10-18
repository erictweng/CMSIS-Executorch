// Host harness for src/fs_detect.c, driven by tests/test_detector.py. Modes:
//   anchors                       print every anchor centre (index cx cy)
//   decode  <head.bin> rscale rzp lscale lzp
//                                 regressors (2016x18 f32) then logits (2016 f32), quantized here
//                                 to int8 with those parameters -> one line:
//                                 hit score anchor palm(x0 y0 x1 y1) hand(x0 y0 x1 y1 in frame px) kp...
//   resize  <frame.bin> scale zp <out.bin>  320x320 RGB565 (u16 LE) -> 192x192x3 int8
//   crop    <frame.bin> x0 y0 x1 y1 <out.bin>  -> 128x128x3 u8
//   track                         synthetic hit/miss sequence, self-checking
//
//   clang -std=c99 -Wall -Wextra -Werror -pedantic -I src src/fs_detect.c \
//         tests/host/test_fs_detect.c -lm -o /tmp/test_fs_detect
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "fs_detect.h"

static int g_checks, g_failures;
#define CHECK(cond)                                           \
  do {                                                        \
    g_checks++;                                               \
    if (!(cond)) {                                            \
      g_failures++;                                           \
      printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);  \
    }                                                         \
  } while (0)

static void* read_file(const char* path, size_t want) {
  FILE* f = fopen(path, "rb");
  if (!f) { perror(path); exit(2); }
  void* buf = malloc(want);
  if (fread(buf, 1, want, f) != want) { fprintf(stderr, "%s: short read (want %zu)\n", path, want); exit(2); }
  fclose(f);
  return buf;
}

static void write_file(const char* path, const void* data, size_t n) {
  FILE* f = fopen(path, "wb");
  if (!f) { perror(path); exit(2); }
  fwrite(data, 1, n, f);
  fclose(f);
}

// round-to-nearest int8 quantization, as the runtime's quantize operator would do it.
static int8_t quantize(float v, fs_quant_t q) {
  const float x = v / q.scale + (float)q.zero_point;
  const int r = (int)(x >= 0.0f ? x + 0.5f : x - 0.5f);
  return (int8_t)(r < -128 ? -128 : r > 127 ? 127 : r);
}

static int mode_track(void) {
  fs_track_t t;
  fs_track_reset(&t);
  const fs_box_t a = {100, 100, 200, 200}, b = {120, 100, 220, 200};
  fs_track_update(&t, false, NULL, 0);
  CHECK(!t.hand);
  fs_track_update(&t, true, &a, 0.9f);
  CHECK(t.hand && t.fresh && t.misses == 0 && t.box.x0 == 100 && t.box.x1 == 200);
  fs_track_update(&t, true, &b, 0.8f);  // EMA 0.5: half way
  CHECK(t.hand && t.box.x0 == 110 && t.box.x1 == 210 && t.score == 0.8f);
  for (int i = 1; i < FS_DET_HOLD_FRAMES; ++i) {  // held for the misses before the limit
    fs_track_update(&t, false, NULL, 0);
    CHECK(t.hand && !t.fresh && t.misses == i && t.box.x0 == 110);
  }
  fs_track_update(&t, false, NULL, 0);  // the FS_DET_HOLD_FRAMES-th miss: no hand
  CHECK(!t.hand && t.misses == FS_DET_HOLD_FRAMES);
  fs_track_update(&t, true, &a, 0.7f);  // re-acquired: starts from the new box, no EMA
  CHECK(t.hand && t.fresh && t.box.x0 == 100);
  printf("fs_detect track: %d checks, %d failed\n", g_checks, g_failures);
  return g_failures ? 1 : 0;
}

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "mode?\n"); return 2; }
  const char* mode = argv[1];
  if (strcmp(mode, "anchors") == 0) {
    for (int i = 0; i < FS_DET_ANCHORS; ++i) {
      float cx, cy;
      fs_detect_anchor(i, &cx, &cy);
      printf("%d %.6f %.6f\n", i, (double)cx, (double)cy);
    }
    return 0;
  }
  if (strcmp(mode, "decode") == 0 && argc == 7) {
    float* head = read_file(argv[2], (size_t)FS_DET_ANCHORS * (FS_DET_COORDS + 1) * sizeof(float));
    const fs_quant_t rq = {(float)atof(argv[3]), atoi(argv[4])}, lq = {(float)atof(argv[5]), atoi(argv[6])};
    static int8_t qreg[FS_DET_ANCHORS * FS_DET_COORDS], qlog[FS_DET_ANCHORS];
    for (size_t i = 0; i < (size_t)FS_DET_ANCHORS * FS_DET_COORDS; ++i) qreg[i] = quantize(head[i], rq);
    for (size_t i = 0; i < (size_t)FS_DET_ANCHORS; ++i) qlog[i] = quantize(head[(size_t)FS_DET_ANCHORS * FS_DET_COORDS + i], lq);
    fs_detection_t d;
    const bool hit = fs_detect_decode(qreg, rq, qlog, lq, &d);
    const fs_box_t hb = fs_detect_hand_box(&d.palm, (float)FS_DET_FRAME / FS_DET_SIZE);
    printf("%d %.6f %d %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f", hit, (double)d.score, d.anchor, (double)d.palm.x0,
           (double)d.palm.y0, (double)d.palm.x1, (double)d.palm.y1, (double)hb.x0, (double)hb.y0, (double)hb.x1,
           (double)hb.y1);
    for (int k = 0; k < FS_DET_KEYPOINTS; ++k) printf(" %.4f %.4f", (double)d.keypoints[k][0], (double)d.keypoints[k][1]);
    printf("\n");
    free(head);
    return 0;
  }
  if (strcmp(mode, "resize") == 0 && argc == 6) {
    uint16_t* frame = read_file(argv[2], (size_t)FS_DET_FRAME * FS_DET_FRAME * 2);
    const fs_quant_t q = {(float)atof(argv[3]), atoi(argv[4])};
    static int8_t out[FS_DET_SIZE * FS_DET_SIZE * 3];
    fs_detect_preprocess(frame, out, q);
    write_file(argv[5], out, sizeof(out));
    free(frame);
    return 0;
  }
  if (strcmp(mode, "crop") == 0 && argc == 8) {
    uint16_t* frame = read_file(argv[2], (size_t)FS_DET_FRAME * FS_DET_FRAME * 2);
    const fs_box_t box = {(float)atof(argv[3]), (float)atof(argv[4]), (float)atof(argv[5]), (float)atof(argv[6])};
    static uint8_t rgb[FS_SIZE * FS_SIZE * 3];
    fs_detect_crop(frame, &box, rgb);
    write_file(argv[7], rgb, sizeof(rgb));
    free(frame);
    return 0;
  }
  if (strcmp(mode, "track") == 0) return mode_track();
  fprintf(stderr, "bad arguments\n");
  return 2;
}
