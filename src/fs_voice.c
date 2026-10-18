// fs_voice.c -- see fs_voice.h. Pure C99: builds on the host for the tests and on the M55.
//
// The tables are computed at init rather than stored, in double so they match the numpy
// reference (ml/kws.py) bit-for-bit where it matters (the mel band edges are integer decisions
// on the mel scale; a float table could round a bin into the neighbouring band).
#include "fs_voice.h"

#include <math.h>
#include <string.h>

const char* const fs_voice_labels[FS_VOICE_CLASSES] = {"silence", "unknown", "yes",  "no",  "up",   "down",
                                                       "left",    "right",   "on",   "off", "stop", "go"};

#define LOW_HZ 20.0
#define HIGH_HZ 4000.0
#define FLOOR 1e-12f
#define PI 3.14159265358979323846

static float s_hann[FS_VOICE_WINDOW];
static float s_cos[FS_VOICE_FFT / 2], s_sin[FS_VOICE_FFT / 2];
static uint16_t s_bitrev[FS_VOICE_FFT];
static int16_t s_band[FS_VOICE_BINS];   // mel channel of each bin (-2: outside 20..4000 Hz)
static float s_weight[FS_VOICE_BINS];   // share of the bin that goes to its channel (the rest to channel + 1)
static int s_bin_start, s_bin_end;
static float s_dct[FS_VOICE_COEFFS][FS_VOICE_MELS];
static float s_re[FS_VOICE_FFT], s_im[FS_VOICE_FFT];  // FFT work
static bool s_ready = false;

static double hz_to_mel(double f) { return 1127.0 * log1p(f / 700.0); }

void fs_voice_mfcc_init(void) {
  for (int n = 0; n < FS_VOICE_WINDOW; ++n)  // periodic Hann, as TF's spectrogram
    s_hann[n] = (float)(0.5 - 0.5 * cos(2.0 * PI * n / FS_VOICE_WINDOW));
  for (int k = 0; k < FS_VOICE_FFT / 2; ++k) {
    s_cos[k] = (float)cos(2.0 * PI * k / FS_VOICE_FFT);
    s_sin[k] = (float)sin(2.0 * PI * k / FS_VOICE_FFT);
  }
  for (int i = 0; i < FS_VOICE_FFT; ++i) {
    unsigned r = 0;
    for (int b = 0; b < 10; ++b) r |= ((unsigned)i >> b & 1U) << (9 - b);
    s_bitrev[i] = (uint16_t)r;
  }
  // TensorFlow's MfccMelFilterbank::Initialize, verbatim in its decisions.
  const double mel_low = hz_to_mel(LOW_HZ), mel_high = hz_to_mel(HIGH_HZ);
  const double spacing = (mel_high - mel_low) / (FS_VOICE_MELS + 1);
  double centres[FS_VOICE_MELS + 1];
  for (int i = 0; i <= FS_VOICE_MELS; ++i) centres[i] = mel_low + spacing * (i + 1);
  const double hz_per_bin = 0.5 * FS_VOICE_RATE / (FS_VOICE_BINS - 1);
  s_bin_start = (int)(1.5 + LOW_HZ / hz_per_bin);
  s_bin_end = (int)(HIGH_HZ / hz_per_bin);
  for (int i = 0; i < FS_VOICE_BINS; ++i) {
    s_band[i] = -2;
    s_weight[i] = 0.f;
    if (i < s_bin_start || i > s_bin_end) continue;
    const double melf = hz_to_mel(i * hz_per_bin);
    int ch = 0;
    while (ch < FS_VOICE_MELS && centres[ch] < melf) ++ch;
    s_band[i] = (int16_t)(ch - 1);
    if (ch > 0)
      s_weight[i] = (float)((centres[ch] - melf) / (centres[ch] - centres[ch - 1]));
    else
      s_weight[i] = (float)((centres[0] - melf) / (centres[0] - mel_low));
  }
  for (int j = 0; j < FS_VOICE_COEFFS; ++j)
    for (int i = 0; i < FS_VOICE_MELS; ++i)
      s_dct[j][i] = (float)(cos(PI * j * (i + 0.5) / FS_VOICE_MELS) * sqrt(2.0 / FS_VOICE_MELS));
  s_ready = true;
}

// In-place radix-2 decimation-in-time FFT of s_re / s_im (N = 1024).
static void fft(void) {
  for (int i = 0; i < FS_VOICE_FFT; ++i) {
    const int j = s_bitrev[i];
    if (j > i) {
      float t = s_re[i]; s_re[i] = s_re[j]; s_re[j] = t;
      t = s_im[i]; s_im[i] = s_im[j]; s_im[j] = t;
    }
  }
  for (int len = 2; len <= FS_VOICE_FFT; len <<= 1) {
    const int half = len >> 1, step = FS_VOICE_FFT / len;
    for (int start = 0; start < FS_VOICE_FFT; start += len)
      for (int k = 0; k < half; ++k) {
        const float wr = s_cos[k * step], wi = -s_sin[k * step];
        const int a = start + k, b = a + half;
        const float xr = s_re[b] * wr - s_im[b] * wi, xi = s_re[b] * wi + s_im[b] * wr;
        s_re[b] = s_re[a] - xr; s_im[b] = s_im[a] - xi;
        s_re[a] += xr; s_im[a] += xi;
      }
  }
}

void fs_voice_mfcc_frame(const int16_t frame[FS_VOICE_WINDOW], float coeffs[FS_VOICE_COEFFS]) {
  if (!s_ready) fs_voice_mfcc_init();
  for (int n = 0; n < FS_VOICE_WINDOW; ++n) s_re[n] = (float)frame[n] * (1.0f / 32768.0f) * s_hann[n];
  memset(s_re + FS_VOICE_WINDOW, 0, (FS_VOICE_FFT - FS_VOICE_WINDOW) * sizeof(float));
  memset(s_im, 0, sizeof(s_im));
  fft();
  float mel[FS_VOICE_MELS] = {0};
  for (int i = s_bin_start; i <= s_bin_end; ++i) {  // the filterbank reads the magnitude
    const float mag = sqrtf(s_re[i] * s_re[i] + s_im[i] * s_im[i]);
    const int ch = s_band[i];
    if (ch >= 0) mel[ch] += mag * s_weight[i];
    if (ch + 1 < FS_VOICE_MELS) mel[ch + 1] += mag * (1.0f - s_weight[i]);
  }
  for (int i = 0; i < FS_VOICE_MELS; ++i) mel[i] = logf(mel[i] > FLOOR ? mel[i] : FLOOR);
  for (int j = 0; j < FS_VOICE_COEFFS; ++j) {
    float acc = 0.f;
    for (int i = 0; i < FS_VOICE_MELS; ++i) acc += s_dct[j][i] * mel[i];
    coeffs[j] = acc;
  }
}

static int8_t quantize(float v, fs_voice_quant_t q) {
  const float x = v / q.scale + (float)q.zero_point;
  const int r = (int)(x >= 0.0f ? x + 0.5f : x - 0.5f);
  return (int8_t)(r < -128 ? -128 : r > 127 ? 127 : r);
}

void fs_voice_stream_reset(fs_voice_stream_t* s, fs_voice_quant_t q) {
  memset(s, 0, sizeof(*s));
  s->q = q;
}

int fs_voice_stream_push(fs_voice_stream_t* s, const int16_t* samples, int n) {
  int produced = 0;
  while (n > 0) {
    const int take = n < FS_VOICE_WINDOW - s->filled ? n : FS_VOICE_WINDOW - s->filled;
    memcpy(s->buf + s->filled, samples, (size_t)take * sizeof(int16_t));
    s->filled += take;
    samples += take;
    n -= take;
    if (s->filled == FS_VOICE_WINDOW) {
      float c[FS_VOICE_COEFFS];
      fs_voice_mfcc_frame(s->buf, c);
      for (int j = 0; j < FS_VOICE_COEFFS; ++j) s->feat[s->head][j] = quantize(c[j], s->q);
      s->head = (s->head + 1) % FS_VOICE_FRAMES;
      s->frames++;
      produced++;
      memmove(s->buf, s->buf + FS_VOICE_STRIDE, (FS_VOICE_WINDOW - FS_VOICE_STRIDE) * sizeof(int16_t));
      s->filled = FS_VOICE_WINDOW - FS_VOICE_STRIDE;
    }
  }
  return produced;
}

void fs_voice_stream_input(const fs_voice_stream_t* s, int8_t out[FS_VOICE_INPUT]) {
  // head is the oldest row once the ring has wrapped (frames >= FS_VOICE_FRAMES); before that the
  // rows are 0..frames-1 and the rest of the window is zero-filled (the caller checks full()).
  const int oldest = s->frames >= FS_VOICE_FRAMES ? s->head : 0;
  for (int f = 0; f < FS_VOICE_FRAMES; ++f)
    memcpy(out + f * FS_VOICE_COEFFS, s->feat[(oldest + f) % FS_VOICE_FRAMES], FS_VOICE_COEFFS);
}

float fs_voice_rms(const int16_t* x, int n) {
  if (n <= 0) return 0.f;
  double acc = 0.0;
  for (int i = 0; i < n; ++i) acc += (double)x[i] * x[i];
  return (float)sqrt(acc / n);
}

void fs_voice_softmax(const int8_t logits[FS_VOICE_CLASSES], fs_voice_quant_t q, float probs[FS_VOICE_CLASSES]) {
  float z[FS_VOICE_CLASSES], m = -1e30f, sum = 0.f;
  for (int i = 0; i < FS_VOICE_CLASSES; ++i) {
    z[i] = (float)(logits[i] - q.zero_point) * q.scale;
    if (z[i] > m) m = z[i];
  }
  for (int i = 0; i < FS_VOICE_CLASSES; ++i) {
    probs[i] = expf(z[i] - m);
    sum += probs[i];
  }
  for (int i = 0; i < FS_VOICE_CLASSES; ++i) probs[i] /= sum;
}

void fs_voice_decider_init(fs_voice_decider_t* d, float threshold, uint32_t refractory_ms) {
  d->threshold = threshold;
  d->refractory_ms = refractory_ms;
  d->last_ms = 0;
  d->reported = false;
}

int fs_voice_decide(fs_voice_decider_t* d, const float probs[FS_VOICE_CLASSES], uint32_t now_ms) {
  int k = 0;
  for (int i = 1; i < FS_VOICE_CLASSES; ++i)
    if (probs[i] > probs[k]) k = i;
  if (k < FS_VOICE_WORD_FIRST || probs[k] < d->threshold) return -1;
  if (d->reported && now_ms - d->last_ms < d->refractory_ms) return -1;
  d->last_ms = now_ms;
  d->reported = true;
  return k;
}
