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
//   motion   (fs_decoder_step_motion) D and I are the static shapes of Z and J, which are
//            drawn in the air; instead of committing, they become pending and the fingertip
//            (index for D, pinky for I) accumulates its path in hand sizes, ignoring jitter
//            below a dead band. When the signer moves on (another letter, no hand, or a
//            timeout) the pending letter commits as Z / J if the path reached the motion
//            threshold, else as D / I.

#include <math.h>
#include <string.h>

#include "fs_api.h"

static const fs_decoder_cfg_t kDefaultCfg = {
    .commit_threshold = 0.60f, .commit_frames = 3, .space_ms = 1000, .release_ms = 300,
    .motion_threshold = 1.0f, .motion_deadband = 0.06f, .pending_timeout_ms = 1500, .pending_drop_frames = 3};

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
  if (d->cfg.pending_drop_frames < 1) d->cfg.pending_drop_frames = 1;
}

static char motion_letter(char still) { return still == 'D' ? 'Z' : still == 'I' ? 'J' : still; }

// Commit `letter` under the release rule; the event (FS_EV_NONE when the rule blocks it).
static fs_event_t commit(fs_decoder_t* d, char letter) {
  if (letter == d->last_committed && !d->released) return FS_EV_NONE;
  append(d, letter);
  d->last_committed = letter;
  d->released = false;
  return FS_EV_LETTER;
}

// The pending letter commits as its still or motion form; clears the pending state.
static fs_event_t resolve_pending(fs_decoder_t* d) {
  const char letter = d->pending_path >= d->cfg.motion_threshold ? motion_letter(d->pending) : d->pending;
  d->pending = 0;
  d->pending_path = 0.0f;
  d->pending_drop = 0;
  d->last_tip_valid = false;
  return commit(d, letter);
}

// Add this frame's fingertip step to the pending path (hand sizes, dead band applied).
static void accumulate(fs_decoder_t* d, const fs_motion_t* m) {
  if (m == NULL || !m->valid || m->hand_size <= 1e-3f) {
    d->last_tip_valid = false;
    return;
  }
  const float* tip = d->pending == 'I' ? m->pinky_tip : m->index_tip;
  if (d->last_tip_valid) {
    const float dx = (tip[0] - d->last_tip[0]) / m->hand_size, dy = (tip[1] - d->last_tip[1]) / m->hand_size;
    const float step = sqrtf(dx * dx + dy * dy);
    if (step >= d->cfg.motion_deadband) d->pending_path += step;
  }
  d->last_tip[0] = tip[0];
  d->last_tip[1] = tip[1];
  d->last_tip_valid = true;
}

fs_event_t fs_decoder_step(fs_decoder_t* d, const float probs[FS_CLASSES], uint32_t now_ms) {
  return fs_decoder_step_motion(d, probs, NULL, now_ms);
}

fs_event_t fs_decoder_step_motion(fs_decoder_t* d, const float probs[FS_CLASSES], const fs_motion_t* motion,
                                  uint32_t now_ms) {
  int top = 0;
  for (int i = 1; i < FS_CLASSES; ++i)
    if (probs[i] > probs[top]) top = i;
  const float conf = probs[top];
  const bool confident = conf >= d->cfg.commit_threshold;

  d->live = confident ? FS_LABELS[top] : 0;
  d->live_conf = conf;

  // A pending D / I: keep measuring while the shape holds, resolve when the signer moves on
  // (the hand is lost, another letter or no letter for pending_drop_frames, or the timeout).
  // A different confident letter keeps counting its own streak meanwhile, so the next letter
  // is not delayed by the dropout tolerance; it commits right after the pending one resolves.
  fs_event_t resolved = FS_EV_NONE;
  if (d->pending) {
    const bool same = confident && FS_LABELS[top] == d->pending;
    const bool hand_lost = motion != NULL && !motion->valid;
    if (same) {
      d->pending_drop = 0;
      accumulate(d, motion);
    } else {
      d->pending_drop++;
    }
    const bool resolve = hand_lost || d->pending_drop >= d->cfg.pending_drop_frames ||
                         now_ms - d->pending_since_ms >= d->cfg.pending_timeout_ms;
    if (!resolve && (same || !confident)) {
      d->live = d->pending;  // shown while pending
      if (confident) d->last_conf_ms = now_ms;
      return FS_EV_PENDING;
    }
    if (resolve) resolved = resolve_pending(d);
  }

  if (!confident) {
    d->cand = 0;
    d->streak = 0;
    if (resolved != FS_EV_NONE) return resolved;
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

  if (d->streak != d->cfg.commit_frames) return resolved != FS_EV_NONE ? resolved : d->pending ? FS_EV_PENDING : FS_EV_NONE;
  if (d->pending) resolved = resolve_pending(d);  // the next letter is here: the pending one goes first
  if (motion != NULL && (letter == 'D' || letter == 'I')) {  // a motion letter's still shape: wait
    d->pending = letter;
    d->pending_since_ms = now_ms;
    d->pending_path = 0.0f;
    d->pending_drop = 0;
    d->last_tip_valid = false;
    accumulate(d, motion);
    return resolved != FS_EV_NONE ? resolved : FS_EV_PENDING;
  }
  const fs_event_t ev = commit(d, letter);
  return ev != FS_EV_NONE ? ev : resolved;
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
