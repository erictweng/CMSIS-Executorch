// Host harness for src/fs_orient.c, driven by tests/test_orient.py. Modes:
//   roll  dx dy dz rx ry rz  gx gy gz        -> the roll in degrees (calibration down / right
//                                              vectors, then a gravity reading)
//   level roll_deg ax ay <points.txt>         -> each "x y z" line rotated about (ax, ay)
//   gravity alpha <readings.txt>              -> the averaged gravity after each "x y z" line
//                                              ("reset" restarts)
//
//   clang -std=c99 -Wall -Wextra -Werror -pedantic -I src src/fs_orient.c \
//         tests/host/test_fs_orient.c -lm -o /tmp/test_fs_orient
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "fs_orient.h"

#define DEG (3.14159265358979f / 180.0f)

int main(int argc, char** argv) {
  if (argc < 2) return 2;
  if (strcmp(argv[1], "roll") == 0 && argc == 11) {
    fs_orient_cal_t cal;
    float g[3];
    for (int i = 0; i < 3; ++i) {
      cal.down[i] = (float)atof(argv[2 + i]);
      cal.right[i] = (float)atof(argv[5 + i]);
      g[i] = (float)atof(argv[8 + i]);
    }
    printf("%.5f\n", (double)(fs_orient_roll(&cal, g) / DEG));
    return 0;
  }
  if (strcmp(argv[1], "level") == 0 && argc == 6) {
    const float roll = (float)atof(argv[2]) * DEG;
    const float about[2] = {(float)atof(argv[3]), (float)atof(argv[4])};
    FILE* f = fopen(argv[5], "r");
    if (!f) { perror(argv[5]); return 2; }
    float pts[64][3];
    int n = 0;
    while (n < 64 && fscanf(f, "%f %f %f", &pts[n][0], &pts[n][1], &pts[n][2]) == 3) n++;
    fclose(f);
    fs_orient_level(pts, n, roll, about);
    for (int i = 0; i < n; ++i) printf("%.5f %.5f %.5f\n", (double)pts[i][0], (double)pts[i][1], (double)pts[i][2]);
    return 0;
  }
  if (strcmp(argv[1], "gravity") == 0 && argc == 4) {
    const float alpha = (float)atof(argv[2]);
    FILE* f = fopen(argv[3], "r");
    if (!f) { perror(argv[3]); return 2; }
    fs_orient_t s;
    fs_orient_reset(&s);
    char line[128];
    while (fgets(line, sizeof line, f)) {
      if (strncmp(line, "reset", 5) == 0) {
        fs_orient_reset(&s);
        continue;
      }
      float g[3];
      if (sscanf(line, "%f %f %f", &g[0], &g[1], &g[2]) != 3) continue;
      fs_orient_gravity(&s, g, alpha);
      printf("%.6f %.6f %.6f\n", (double)s.g[0], (double)s.g[1], (double)s.g[2]);
    }
    fclose(f);
    return 0;
  }
  fprintf(stderr, "bad arguments\n");
  return 2;
}
