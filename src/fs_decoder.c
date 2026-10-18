// fs_decoder.c — Fingerspell decoder: one frame of letter probabilities in, committed text out.
// Implements the Decoder contract of fs_api.h. Pure C99 with no hardware dependency so it
// unit-tests on a laptop (tests/host/test_fs_decoder.c) with the same code the board runs.
//
// Rules (SPEC.md §4 F2, §7; temporal averaging per documentation/temporal-average-plan.md):
//   average  the probabilities are averaged over frames (exponential, weight prob_alpha on
//            the new frame). The first frame with a hand copies the raw probabilities, so a
//            steady hand is not delayed; a frame without a hand (motion->valid false, or the
//            probabilities sum to under 0.5) resets the average. Every rule below reads the
//            averaged stream, and live_conf shows it.
//   commit   the top letter is appended when its probability is >= commit_threshold for
//            commit_frames consecutive frames; it commits once, however long it is held
//   hold     the candidate stays the live letter while its averaged probability is
//            >= hold_threshold (hysteresis); such frames keep the streak alive but do not
//            count toward commit_frames, and they count as confident for the space timer
//   release  the same letter commits again only after the hand was lost for >= release_ms
//            (a confidence dip never re-arms it), so "LL" is signed by bouncing the hand
//   space    the hand out of view for >= space_ms inserts one space; a pause with the hand
//            in view never does (slow signing must not break a word)
//   buttons  backspace / clear re-arm the decoder: the held letter commits again, a pending
//            D / I is dropped, and the space rule waits until the hand is next in view
//   text     oldest characters drop off when the buffer is full
//   motion   (fs_decoder_step_motion) D and I are the static shapes of Z and J, which are
//            drawn in the air; instead of committing, they become pending and the fingertip
//            (index for D, pinky for I) is watched: its displacement from where the hold
//            began, in hand sizes (keypoint jitter cancels in a displacement; a path sum
//            of jittery steps read a held D as Z). Z / J commit the moment the displacement
//            reaches the motion threshold; the still shape that follows the stroke neither
//            pends nor commits again until the hand is lost or another letter commits.
//            When the signer moves on first (another letter, no hand, or a timeout) the
//            pending letter commits as D / I.

#include <math.h>
#include <string.h>

#include "fs_api.h"

static const fs_decoder_cfg_t kDefaultCfg = {
    .commit_threshold = 0.50f, .hold_threshold = 0.40f, .prob_alpha = 0.4f, .commit_frames = 2,
    .space_ms = 600, .release_ms = 300,
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
  if (d->cfg.hold_threshold <= 0.0f || d->cfg.hold_threshold > d->cfg.commit_threshold)
    d->cfg.hold_threshold = d->cfg.commit_threshold;  // no hysteresis
  if (d->cfg.prob_alpha <= 0.0f || d->cfg.prob_alpha > 1.0f) d->cfg.prob_alpha = 1.0f;  // no averaging
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
  d->start_tip_valid = false;
  return commit(d, letter);
}

// Update the pending displacement: how far the fingertip is from where the hold began (hand
// sizes, dead band applied), keeping the largest value seen. The first frame with a usable
// tip sets the start.
static void accumulate(fs_decoder_t* d, const fs_motion_t* m) {
  if (m == NULL || !m->valid || m->hand_size <= 1e-3f) return;
  const float* tip = d->pending == 'I' ? m->pinky_tip : m->index_tip;
  if (!d->start_tip_valid) {
    d->start_tip[0] = tip[0];
    d->start_tip[1] = tip[1];
    d->start_tip_valid = true;
    return;
  }
  const float dx = (tip[0] - d->start_tip[0]) / m->hand_size, dy = (tip[1] - d->start_tip[1]) / m->hand_size;
  const float dist = sqrtf(dx * dx + dy * dy);
  if (dist >= d->cfg.motion_deadband && dist > d->pending_path) d->pending_path = dist;
}

fs_event_t fs_decoder_step(fs_decoder_t* d, const float probs[FS_CLASSES], uint32_t now_ms) {
  return fs_decoder_step_motion(d, probs, NULL, now_ms);
}

// The averaged probabilities for this frame; `hand` says whether there is one to average.
static const float* average(fs_decoder_t* d, const float probs[FS_CLASSES], bool hand) {
  if (!hand) {
    d->avg_valid = false;
    return probs;
  }
  if (!d->avg_valid) {
    memcpy(d->avg, probs, sizeof d->avg);
    d->avg_valid = true;
  } else {
    for (int i = 0; i < FS_CLASSES; ++i) d->avg[i] += d->cfg.prob_alpha * (probs[i] - d->avg[i]);
  }
  return d->avg;
}

fs_event_t fs_decoder_step_motion(fs_decoder_t* d, const float probs[FS_CLASSES], const fs_motion_t* motion,
                                  uint32_t now_ms) {
  float sum = 0.0f;
  for (int i = 0; i < FS_CLASSES; ++i) sum += probs[i];
  const bool hand = (motion == NULL || motion->valid) && sum >= 0.5f;
  if (!hand && now_ms - d->last_hand_ms >= d->cfg.release_ms) d->released = true;
  if (hand) {
    d->last_hand_ms = now_ms;
    d->space_blocked = false;
  }

  const float* p = average(d, probs, hand);
  int top = 0;
  for (int i = 1; i < FS_CLASSES; ++i)
    if (p[i] > p[top]) top = i;
  float conf = p[top];
  const bool above = hand && conf >= d->cfg.commit_threshold;
  bool confident = above;
  if (!confident && hand && d->cand) {  // hysteresis: the candidate holds on down to hold_threshold
    const int ci = (int)(strchr(FS_LABELS, d->cand) - FS_LABELS);
    if (p[ci] >= d->cfg.hold_threshold) {
      top = ci;
      conf = p[ci];
      confident = true;
    }
  }

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
      if (d->pending_path >= d->cfg.motion_threshold) {  // the stroke is drawn: Z / J now
        d->live = d->pending;
        d->cand = d->pending;  // the still shape continues; its streak must not re-pend it
        d->streak = d->cfg.commit_frames + 1;
        return resolve_pending(d);
      }
    } else {
      d->pending_drop++;
    }
    const bool resolve = hand_lost || d->pending_drop >= d->cfg.pending_drop_frames ||
                         now_ms - d->pending_since_ms >= d->cfg.pending_timeout_ms;
    if (!resolve && (same || !confident)) {
      d->live = d->pending;  // shown while pending
      return FS_EV_PENDING;
    }
    if (resolve) resolved = resolve_pending(d);
  }

  if (!confident) {
    d->cand = 0;
    d->streak = 0;
    if (resolved != FS_EV_NONE) return resolved;
    if (!hand && !d->space_blocked && d->len > 0 && d->text[d->len - 1] != ' ' &&
        now_ms - d->last_hand_ms >= d->cfg.space_ms) {
      append(d, ' ');
      return FS_EV_SPACE;
    }
    return FS_EV_NONE;
  }

  const char letter = FS_LABELS[top];
  if (letter != d->cand) {
    d->cand = letter;
    d->streak = 0;
  }
  if (above) d->streak++;  // a held frame keeps the candidate but does not count

  if (!above || d->streak != d->cfg.commit_frames)
    return resolved != FS_EV_NONE ? resolved : d->pending ? FS_EV_PENDING : FS_EV_NONE;
  if (d->pending) resolved = resolve_pending(d);  // the next letter is here: the pending one goes first
  if (motion != NULL && (letter == 'D' || letter == 'I')) {  // a motion letter's still shape: wait
    if (d->last_committed == motion_letter(letter) && !d->released)  // the tail of the stroke just committed
      return resolved;
    d->pending = letter;
    d->pending_since_ms = now_ms;
    d->pending_path = 0.0f;
    d->pending_drop = 0;
    d->start_tip_valid = false;
    accumulate(d, motion);
    return resolved != FS_EV_NONE ? resolved : FS_EV_PENDING;
  }
  const fs_event_t ev = commit(d, letter);
  return ev != FS_EV_NONE ? ev : resolved;
}

// After a button edit the signer keeps holding the shape: let it commit again, forget a
// pending D / I, and hold the space rule until the hand is next seen.
static void rearm(fs_decoder_t* d) {
  d->cand = 0;
  d->streak = 0;
  d->released = true;
  d->pending = 0;
  d->pending_path = 0.0f;
  d->pending_drop = 0;
  d->start_tip_valid = false;
  d->space_blocked = true;
}

fs_event_t fs_decoder_backspace(fs_decoder_t* d) {
  if (d->len == 0) return FS_EV_NONE;
  d->text[--d->len] = '\0';
  rearm(d);
  return FS_EV_BACKSPACE;
}

fs_event_t fs_decoder_clear(fs_decoder_t* d) {
  d->len = 0;
  d->text[0] = '\0';
  rearm(d);
  return FS_EV_CLEAR;
}
