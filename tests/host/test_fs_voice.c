// Host harness for src/fs_voice.c, driven by tests/test_voice.py. Modes:
//   mfcc   <clips.bin> N            N clips of 16000 int16 LE -> 49 lines of 10 coefficients per clip
//   stream <clips.bin> N chunk scale zp   the same clips through fs_voice_stream_push in chunks of
//          `chunk` samples (the window is reset per clip) -> 490 int8 per clip, one line each,
//          preceded by "frames <n>"
//   rms    <clips.bin> N            one RMS per clip
//   softmax <logits.txt> scale zp   12 int8 logits -> 12 probabilities
//   decide threshold refractory <probs.txt>   one line per step: "ms p0 .. p11" -> class or -1
//
//   clang -std=c99 -Wall -Wextra -Werror -pedantic -I src src/fs_voice.c tests/host/test_fs_voice.c \
//         -lm -o /tmp/test_fs_voice
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "fs_voice.h"

#define CLIP 16000

static int16_t* read_clips(const char* path, int n) {
  FILE* f = fopen(path, "rb");
  if (!f) { perror(path); exit(2); }
  int16_t* v = malloc((size_t)n * CLIP * sizeof(int16_t));
  if (fread(v, sizeof(int16_t), (size_t)n * CLIP, f) != (size_t)n * CLIP) { fprintf(stderr, "short read\n"); exit(2); }
  fclose(f);
  return v;
}

int main(int argc, char** argv) {
  if (argc < 2) return 2;
  fs_voice_mfcc_init();
  if (strcmp(argv[1], "mfcc") == 0 && argc == 4) {
    const int n = atoi(argv[3]);
    int16_t* clips = read_clips(argv[2], n);
    for (int c = 0; c < n; ++c)
      for (int fr = 0; fr < FS_VOICE_FRAMES; ++fr) {
        float co[FS_VOICE_COEFFS];
        fs_voice_mfcc_frame(clips + c * CLIP + fr * FS_VOICE_STRIDE, co);
        for (int j = 0; j < FS_VOICE_COEFFS; ++j) printf("%s%.5f", j ? " " : "", (double)co[j]);
        printf("\n");
      }
    free(clips);
    return 0;
  }
  if (strcmp(argv[1], "stream") == 0 && argc == 7) {
    const int n = atoi(argv[3]), chunk = atoi(argv[4]);
    const fs_voice_quant_t q = {(float)atof(argv[5]), atoi(argv[6])};
    int16_t* clips = read_clips(argv[2], n);
    static fs_voice_stream_t s;
    for (int c = 0; c < n; ++c) {
      fs_voice_stream_reset(&s, q);
      int frames = 0;
      for (int i = 0; i < CLIP; i += chunk) frames += fs_voice_stream_push(&s, clips + c * CLIP + i, CLIP - i < chunk ? CLIP - i : chunk);
      int8_t in[FS_VOICE_INPUT];
      fs_voice_stream_input(&s, in);
      printf("frames %d full %d\n", frames, fs_voice_stream_full(&s) ? 1 : 0);
      for (int i = 0; i < FS_VOICE_INPUT; ++i) printf("%s%d", i ? " " : "", in[i]);
      printf("\n");
    }
    free(clips);
    return 0;
  }
  if (strcmp(argv[1], "rms") == 0 && argc == 4) {
    const int n = atoi(argv[3]);
    int16_t* clips = read_clips(argv[2], n);
    for (int c = 0; c < n; ++c) printf("%.3f\n", (double)fs_voice_rms(clips + c * CLIP, CLIP));
    free(clips);
    return 0;
  }
  if (strcmp(argv[1], "softmax") == 0 && argc == 5) {
    FILE* f = fopen(argv[2], "r");
    if (!f) { perror(argv[2]); exit(2); }
    int8_t logits[FS_VOICE_CLASSES];
    for (int i = 0; i < FS_VOICE_CLASSES; ++i) { int v; if (fscanf(f, "%d", &v) != 1) exit(2); logits[i] = (int8_t)v; }
    fclose(f);
    const fs_voice_quant_t q = {(float)atof(argv[3]), atoi(argv[4])};
    float p[FS_VOICE_CLASSES];
    fs_voice_softmax(logits, q, p);
    for (int i = 0; i < FS_VOICE_CLASSES; ++i) printf("%s%.6f", i ? " " : "", (double)p[i]);
    printf("\n");
    return 0;
  }
  if (strcmp(argv[1], "decide") == 0 && argc == 5) {
    fs_voice_decider_t d;
    fs_voice_decider_init(&d, (float)atof(argv[2]), (uint32_t)atoi(argv[3]));
    FILE* f = fopen(argv[4], "r");
    if (!f) { perror(argv[4]); exit(2); }
    unsigned ms;
    float p[FS_VOICE_CLASSES];
    while (fscanf(f, "%u", &ms) == 1) {
      for (int i = 0; i < FS_VOICE_CLASSES; ++i) if (fscanf(f, "%f", &p[i]) != 1) exit(2);
      const int k = fs_voice_decide(&d, p, ms);
      printf("%d %s\n", k, k >= 0 ? fs_voice_labels[k] : "-");
    }
    fclose(f);
    return 0;
  }
  fprintf(stderr, "usage: see the header of %s\n", __FILE__);
  return 2;
}
