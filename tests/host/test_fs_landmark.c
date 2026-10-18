// Host harness for src/fs_landmark.c, driven by tests/test_landmark.py. Modes:
//   crop   <frame.bin> cx cy side ux uy scale zp <out.bin>
//          320x320 RGB565 (u16 LE) -> 224x224x3 int8 of the rotated crop; (ux, uy) = unit
//          vector toward the fingers in frame coordinates (the crop's "up")
//   decode <lm63.txt> presence handedness lscale lzp pscale pzp hscale hzp cx cy side ux uy [mode]
//          63 floats (crop px) quantized here -> frame px landmarks, presence, handedness,
//          then features (63) -> one line each; mode = auto | right | left (the handedness
//          rule, FS_LM_HANDEDNESS_*; default auto: mirror by the handedness score)
//   presence <values.txt>   one presence per line, or "reset" -> 0 / 1 per line: whether
//          fs_landmark_presence_gate says there is a hand (the hold through low frames)
//   mlp    <features.txt> <weights.txt>   features 63 floats; weights: hidden, then W1 b1 W2 b2
//          -> 24 probabilities
//   smooth alpha <crops.txt>   one crop per line as "cx cy side ux uy", or "reset" -> the
//          averaged crop after each line: cx cy side rx ry dx dy (fs_landmark_crop_smooth)
//
//   clang -std=c99 -Wall -Wextra -Werror -pedantic -I src src/fs_detect.c src/fs_landmark.c \
//         tests/host/test_fs_landmark.c -lm -o /tmp/test_fs_landmark
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "fs_landmark.h"

static void* read_file(const char* path, size_t want) {
  FILE* f = fopen(path, "rb");
  if (!f) { perror(path); exit(2); }
  void* buf = malloc(want);
  if (fread(buf, 1, want, f) != want) { fprintf(stderr, "%s: short read\n", path); exit(2); }
  fclose(f);
  return buf;
}

static float* read_floats(const char* path, size_t n) {
  FILE* f = fopen(path, "r");
  if (!f) { perror(path); exit(2); }
  float* v = malloc(n * sizeof(float));
  for (size_t i = 0; i < n; ++i)
    if (fscanf(f, "%f", &v[i]) != 1) { fprintf(stderr, "%s: value %zu\n", path, i); exit(2); }
  fclose(f);
  return v;
}

static int8_t quantize(float v, fs_quant_t q) {
  const float x = v / q.scale + (float)q.zero_point;
  const int r = (int)(x >= 0.0f ? x + 0.5f : x - 0.5f);
  return (int8_t)(r < -128 ? -128 : r > 127 ? 127 : r);
}

static fs_lm_crop_t crop_of(char** a) {  // cx cy side ux uy
  fs_lm_crop_t c;
  c.cx = (float)atof(a[0]); c.cy = (float)atof(a[1]); c.side = (float)atof(a[2]);
  const float ux = (float)atof(a[3]), uy = (float)atof(a[4]);
  c.rx = -uy; c.ry = ux; c.dx = -ux; c.dy = -uy;
  return c;
}

int main(int argc, char** argv) {
  if (argc < 2) return 2;
  if (strcmp(argv[1], "crop") == 0 && argc == 11) {
    uint16_t* frame = read_file(argv[2], (size_t)FS_DET_FRAME * FS_DET_FRAME * 2);
    const fs_lm_crop_t c = crop_of(argv + 3);
    const fs_quant_t q = {(float)atof(argv[8]), atoi(argv[9])};
    static int8_t out[FS_LM_SIZE * FS_LM_SIZE * 3];
    fs_landmark_crop(frame, &c, out, q);
    FILE* f = fopen(argv[10], "wb");
    fwrite(out, 1, sizeof(out), f);
    fclose(f);
    free(frame);
    return 0;
  }
  if (strcmp(argv[1], "decode") == 0 && (argc == 16 || argc == 17)) {
    int mode = FS_LM_HANDEDNESS_AUTO;
    if (argc == 17) {
      if (strcmp(argv[16], "right") == 0) mode = FS_LM_HANDEDNESS_RIGHT;
      else if (strcmp(argv[16], "left") == 0) mode = FS_LM_HANDEDNESS_LEFT;
      else if (strcmp(argv[16], "auto") != 0) return 2;
    }
    float* lm = read_floats(argv[2], FS_LM_FEATURES);
    const fs_quant_t lq = {(float)atof(argv[5]), atoi(argv[6])}, pq = {(float)atof(argv[7]), atoi(argv[8])},
                     hq = {(float)atof(argv[9]), atoi(argv[10])};
    const fs_lm_crop_t c = crop_of(argv + 11);
    int8_t q63[FS_LM_FEATURES];
    for (int i = 0; i < FS_LM_FEATURES; ++i) q63[i] = quantize(lm[i], lq);
    fs_landmarks_t out;
    fs_landmark_decode(q63, lq, quantize((float)atof(argv[3]), pq), pq, quantize((float)atof(argv[4]), hq), hq, &c, &out);
    for (int i = 0; i < FS_LM_COUNT; ++i) printf("%.4f %.4f %.4f ", (double)out.xyz[i][0], (double)out.xyz[i][1], (double)out.xyz[i][2]);
    printf("\n%.4f %.4f\n", (double)out.presence, (double)out.right);
    float f[FS_LM_FEATURES];
    fs_landmark_features_as(&out, mode, f);
    for (int i = 0; i < FS_LM_FEATURES; ++i) printf("%.5f ", (double)f[i]);
    printf("\n");
    free(lm);
    return 0;
  }
  if (strcmp(argv[1], "presence") == 0 && argc == 3) {
    FILE* f = fopen(argv[2], "r");
    if (!f) { perror(argv[2]); return 2; }
    fs_lm_presence_t g;
    fs_landmark_presence_reset(&g);
    char line[64];
    while (fgets(line, sizeof line, f)) {
      if (strncmp(line, "reset", 5) == 0) {
        fs_landmark_presence_reset(&g);
        continue;
      }
      if (line[0] == '\n' || line[0] == '\0') continue;
      printf("%d\n", fs_landmark_presence_gate(&g, (float)atof(line)) ? 1 : 0);
    }
    fclose(f);
    return 0;
  }
  if (strcmp(argv[1], "mlp") == 0 && argc == 4) {
    float* f = read_floats(argv[2], FS_LM_FEATURES);
    FILE* w = fopen(argv[3], "r");
    if (!w) { perror(argv[3]); return 2; }
    int hidden = 0;
    if (fscanf(w, "%d", &hidden) != 1 || hidden <= 0 || hidden > 256) return 2;
    float* W1 = malloc((size_t)hidden * FS_LM_FEATURES * sizeof(float));
    float* b1 = malloc((size_t)hidden * sizeof(float));
    float* W2 = malloc((size_t)FS_CLASSES * hidden * sizeof(float));
    float b2[FS_CLASSES], probs[FS_CLASSES];
    for (size_t i = 0; i < (size_t)hidden * FS_LM_FEATURES; ++i) if (fscanf(w, "%f", &W1[i]) != 1) return 2;
    for (int i = 0; i < hidden; ++i) if (fscanf(w, "%f", &b1[i]) != 1) return 2;
    for (size_t i = 0; i < (size_t)FS_CLASSES * hidden; ++i) if (fscanf(w, "%f", &W2[i]) != 1) return 2;
    for (int i = 0; i < FS_CLASSES; ++i) if (fscanf(w, "%f", &b2[i]) != 1) return 2;
    fclose(w);
    fs_landmark_letters(f, W1, b1, W2, b2, hidden, probs);
    for (int k = 0; k < FS_CLASSES; ++k) printf("%.6f ", (double)probs[k]);
    printf("\n");
    return 0;
  }
  if (strcmp(argv[1], "smooth") == 0 && argc == 4) {
    const float alpha = (float)atof(argv[2]);
    FILE* f = fopen(argv[3], "r");
    if (!f) { perror(argv[3]); return 2; }
    fs_lm_crop_t avg = {0, 0, 0, 0, 0, 0, 0};
    bool valid = false;
    char line[256];
    while (fgets(line, sizeof line, f)) {
      if (strncmp(line, "reset", 5) == 0) {
        valid = false;
        continue;
      }
      char* a[5];
      char* save = NULL;
      int n = 0;
      for (char* tok = strtok_r(line, " \t\r\n", &save); tok && n < 5; tok = strtok_r(NULL, " \t\r\n", &save)) a[n++] = tok;
      if (n == 0) continue;
      if (n != 5) { fprintf(stderr, "bad crop line\n"); return 2; }
      const fs_lm_crop_t raw = crop_of(a);
      fs_landmark_crop_smooth(&avg, &valid, &raw, alpha);
      printf("%.5f %.5f %.5f %.6f %.6f %.6f %.6f\n", (double)avg.cx, (double)avg.cy, (double)avg.side, (double)avg.rx,
             (double)avg.ry, (double)avg.dx, (double)avg.dy);
    }
    fclose(f);
    return 0;
  }
  fprintf(stderr, "bad arguments\n");
  return 2;
}
