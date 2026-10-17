// fs_decoder.c — Fingerspell decoder: one frame of letter probabilities in, committed text out.
// Implements the Decoder contract of fs_api.h. Pure C99 with no hardware dependency so it
// unit-tests on a laptop (tests/host/test_fs_decoder.c) with the same code the board runs.
//
// Rules (SPEC.md §4 F2, §7):
//   commit   the top letter is appended when its probability is >= commit_threshold for
//            commit_frames consecutive frames; it commits once, however long it is held
//   release  the same letter commits again only after a gap of >= release_ms with no
//            confident letter (the hand dropped or the confidence dipped), so "LL" works
//   space    a gap of >= space_ms with no confident letter inserts one space
//   text     oldest characters drop off when the buffer is full

#include <string.h>

#include "fs_api.h"

static const fs_decoder_cfg_t kDefaultCfg = {
    .commit_threshold = 0.60f, .commit_frames = 3, .space_ms = 1000, .release_ms = 300};

static void append(fs_decoder_t* d, char c) {
  if (d->len >= FS_TEXT_MAX - 1) {
    memmove(d->text, d->text + 1, (size_t)(d->len - 1));
    d->len--;
  }
  d->text[d->len++] = c;
  d->text[d->len] = '\0';
}

void fs_decoder_init(fs_decoder_t* d, const fs_decoder_cfg_t* cfg) {
  memset(d, 0, sizeof *d);
  d->cfg = cfg ? *cfg : kDefaultCfg;
  if (d->cfg.commit_frames < 1) d->cfg.commit_frames = 1;
}

fs_event_t fs_decoder_step(fs_decoder_t* d, const float probs[FS_CLASSES], uint32_t now_ms) {
  int top = 0;
  for (int i = 1; i < FS_CLASSES; ++i)
    if (probs[i] > probs[top]) top = i;
  const float conf = probs[top];
  const bool confident = conf >= d->cfg.commit_threshold;

  d->live = confident ? FS_LABELS[top] : 0;
  d->live_conf = conf;

  if (!confident) {
    d->cand = 0;
    d->streak = 0;
    if (d->len > 0 && d->text[d->len - 1] != ' ' && now_ms - d->last_conf_ms >= d->cfg.space_ms) {
      append(d, ' ');
      return FS_EV_SPACE;
    }
    return FS_EV_NONE;
  }

  if (now_ms - d->last_conf_ms >= d->cfg.release_ms) d->released = true;
  d->last_conf_ms = now_ms;

  const char letter = FS_LABELS[top];
  if (letter == d->cand) {
    d->streak++;
  } else {
    d->cand = letter;
    d->streak = 1;
  }

  if (d->streak != d->cfg.commit_frames) return FS_EV_NONE;
  if (letter == d->last_committed && !d->released) return FS_EV_NONE;

  append(d, letter);
  d->last_committed = letter;
  d->released = false;
  return FS_EV_LETTER;
}

fs_event_t fs_decoder_backspace(fs_decoder_t* d) {
  if (d->len == 0) return FS_EV_NONE;
  d->text[--d->len] = '\0';
  return FS_EV_BACKSPACE;
}

fs_event_t fs_decoder_clear(fs_decoder_t* d) {
  d->len = 0;
  d->text[0] = '\0';
  return FS_EV_CLEAR;
}
