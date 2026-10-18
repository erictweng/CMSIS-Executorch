// fs_voice.h -- keyword spotting on the CPU side: the MFCC front end, the rolling feature window
// the "kws" method takes, softmax and the decision rule. Pure C99, no hardware, host-tested
// (tests/host/test_fs_voice.c, tests/test_voice.py) against the numpy reference ml/kws.py, which
// is checked against TensorFlow's audio_ops (what the DS-CNN was trained with). The microphone
// itself is src/fs_voice_capture.c (DevKit-E8 only).
//
// Front end (ml/kws.py has the derivation): 16 kHz int16 audio, 640-sample periodic Hann window
// every 320 samples, 1024-point FFT, 40 HTK-mel channels 20..4000 Hz on the magnitude spectrum,
// log with a 1e-12 floor, DCT-II (sqrt(2/40)), coefficients 0..9. 49 frames = one 1 s input.
#pragma once
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FS_VOICE_RATE 16000
#define FS_VOICE_WINDOW 640
#define FS_VOICE_STRIDE 320
#define FS_VOICE_FFT 1024
#define FS_VOICE_BINS (FS_VOICE_FFT / 2 + 1)
#define FS_VOICE_MELS 40
#define FS_VOICE_FRAMES 49
#define FS_VOICE_COEFFS 10
#define FS_VOICE_CLASSES 12
#define FS_VOICE_INPUT (FS_VOICE_FRAMES * FS_VOICE_COEFFS)
#define FS_VOICE_WORD_FIRST 2  // classes 0 and 1 are silence and unknown: never reported

typedef struct {
  float scale;
  int zero_point;
} fs_voice_quant_t;

// Class names in the model's output order (model/model.py KWS_LABELS).
extern const char* const fs_voice_labels[FS_VOICE_CLASSES];

// Build the window, FFT, mel and DCT tables (~13 KB of static data). Call once before any frame.
void fs_voice_mfcc_init(void);
// One 640-sample frame -> 10 MFCC coefficients.
void fs_voice_mfcc_frame(const int16_t frame[FS_VOICE_WINDOW], float coeffs[FS_VOICE_COEFFS]);

// Streaming front end: samples in (any chunk size), int8 MFCC frames into a rolling window of
// the last FS_VOICE_FRAMES frames, quantized with the method's input scale / zero point.
typedef struct {
  int16_t buf[FS_VOICE_WINDOW];
  int filled;
  int8_t feat[FS_VOICE_FRAMES][FS_VOICE_COEFFS];  // ring of frames
  int head;                                       // the row the next frame goes to
  uint32_t frames;                                // frames since reset
  fs_voice_quant_t q;
} fs_voice_stream_t;

void fs_voice_stream_reset(fs_voice_stream_t* s, fs_voice_quant_t q);
// Returns the number of new frames produced from these samples.
int fs_voice_stream_push(fs_voice_stream_t* s, const int16_t* samples, int n);
static inline bool fs_voice_stream_full(const fs_voice_stream_t* s) { return s->frames >= FS_VOICE_FRAMES; }
// The method's input: the last 49 frames, oldest first, time-major (490 int8).
void fs_voice_stream_input(const fs_voice_stream_t* s, int8_t out[FS_VOICE_INPUT]);

// Root mean square of n samples (0..32767), for the level print and the mic test.
float fs_voice_rms(const int16_t* x, int n);

// The method's int8 logits -> probabilities.
void fs_voice_softmax(const int8_t logits[FS_VOICE_CLASSES], fs_voice_quant_t q, float probs[FS_VOICE_CLASSES]);

// Decision rule: report the top class when it is a word (not silence / unknown), its probability
// is at least threshold, and refractory_ms have passed since the last report. Returns the class
// index or -1.
typedef struct {
  float threshold;
  uint32_t refractory_ms;
  uint32_t last_ms;
  bool reported;
} fs_voice_decider_t;

void fs_voice_decider_init(fs_voice_decider_t* d, float threshold, uint32_t refractory_ms);
int fs_voice_decide(fs_voice_decider_t* d, const float probs[FS_VOICE_CLASSES], uint32_t now_ms);

#ifdef __cplusplus
}
#endif
