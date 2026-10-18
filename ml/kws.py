"""Keyword spotting reference (numpy): the DS-CNN front end and decision rule the firmware
implements in src/fs_voice.c. Pure numpy, no TensorFlow, so the board tests run in the project venv.

The front end is TensorFlow's audio_ops as Arm ML-Zoo's DS-CNN training used it
(data_processing/data_preprocessing.py: audio_spectrogram(window 640, stride 320,
magnitude_squared=True) then mfcc(sample_rate 16000, dct_coefficient_count 10) with the op defaults
40 mel channels, 20..4000 Hz):

- samples int16 -> float / 32768, zero-padded or cropped to 16000 (1 s)
- 49 frames: periodic Hann window of 640, 1024-point FFT, power spectrum of 513 bins
- mel filterbank on the *magnitude* (sqrt of the power), HTK mel 1127 ln(1 + f/700), 40 triangular
  channels between 20 and 4000 Hz computed the way tensorflow/core/kernels/mfcc_mel_filterbank.cc
  does (band edges from mel-spaced centre frequencies, bins below 20 Hz / above 4000 Hz dropped)
- natural log with a 1e-12 floor, DCT-II scaled by sqrt(2/40), coefficients 0..9 (c0 kept)
- no mean/variance normalisation; the model takes the 49x10 matrix time-major (490 values)

tests/test_voice.py checks this file against TensorFlow's own output (tests/fixtures/kws_ref.npz)
and the C implementation against this file.
"""
from __future__ import annotations

import numpy as np

RATE = 16000
CLIP = 16000           # 1 s
WINDOW = 640           # 40 ms
STRIDE = 320           # 20 ms
FFT = 1024
BINS = FFT // 2 + 1    # 513
MELS = 40
LOW_HZ, HIGH_HZ = 20.0, 4000.0
COEFFS = 10
FRAMES = 1 + (CLIP - WINDOW) // STRIDE  # 49
FLOOR = 1e-12
LABELS = ["_silence_", "_unknown_", "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
WORDS = LABELS[2:]  # the ten words the board may show


def hz_to_mel(f):
    return 1127.0 * np.log1p(np.asarray(f, dtype=np.float64) / 700.0)


def hann() -> np.ndarray:
    n = np.arange(WINDOW)
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * n / WINDOW)).astype(np.float32)


def mel_weights() -> tuple[np.ndarray, np.ndarray]:
    """TF's MfccMelFilterbank tables: band_mapper[bin] (channel index or -2) and weight[bin]."""
    mel_low, mel_high = hz_to_mel(LOW_HZ), hz_to_mel(HIGH_HZ)
    spacing = (mel_high - mel_low) / (MELS + 1)
    centres = mel_low + spacing * (np.arange(MELS + 1) + 1)  # MELS + 1 edges
    hz_per_bin = 0.5 * RATE / (BINS - 1)
    start = int(1.5 + LOW_HZ / hz_per_bin)
    end = int(HIGH_HZ / hz_per_bin)
    band = np.full(BINS, -2, dtype=np.int64)
    weight = np.zeros(BINS, dtype=np.float64)
    for i in range(BINS):
        melf = hz_to_mel(i * hz_per_bin)
        if i < start or i > end:
            continue
        ch = 0
        while ch < MELS and centres[ch] < melf:
            ch += 1
        band[i] = ch - 1
        if ch - 1 >= 0:
            weight[i] = (centres[ch] - melf) / (centres[ch] - centres[ch - 1])
        else:
            weight[i] = (centres[0] - melf) / (centres[0] - mel_low)
    return band, weight


def mel_matrix() -> np.ndarray:
    """The same filterbank as a dense MELS x BINS matrix applied to the magnitude spectrum."""
    band, weight = mel_weights()
    m = np.zeros((MELS, BINS), dtype=np.float64)
    for i in range(BINS):
        ch = band[i]
        if ch == -2:
            continue
        if ch >= 0:
            m[ch, i] += weight[i]
        if ch + 1 < MELS:
            m[ch + 1, i] += 1.0 - weight[i]
    return m


def dct_matrix() -> np.ndarray:
    """DCT-II rows for coefficients 0..COEFFS-1 over MELS inputs, TF's sqrt(2/N) scaling on every row."""
    i = np.arange(MELS)
    return np.array([np.cos(np.pi * j * (i + 0.5) / MELS) for j in range(COEFFS)]) * np.sqrt(2.0 / MELS)


_HANN = hann()
_MEL = mel_matrix()
_DCT = dct_matrix()


def pad_clip(samples: np.ndarray) -> np.ndarray:
    s = np.asarray(samples, dtype=np.int16).ravel()[:CLIP]
    return np.pad(s, (0, CLIP - len(s)))


def frame_features(frame: np.ndarray) -> np.ndarray:
    """One 640-sample int16 frame -> 10 MFCC coefficients (float32)."""
    x = np.asarray(frame, dtype=np.float64) / 32768.0 * _HANN
    spec = np.fft.rfft(x, FFT)
    power = spec.real ** 2 + spec.imag ** 2
    mel = _MEL @ np.sqrt(power)
    return (_DCT @ np.log(np.maximum(mel, FLOOR))).astype(np.float32)


def mfcc(samples: np.ndarray) -> np.ndarray:
    """1 s of int16 audio -> 49 x 10 float32, like TF audio_spectrogram + mfcc."""
    s = pad_clip(samples)
    return np.stack([frame_features(s[i * STRIDE:i * STRIDE + WINDOW]) for i in range(FRAMES)])


def quantize(features: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    return np.clip(np.round(features / scale) + zero_point, -128, 127).astype(np.int8)


def softmax(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64) - np.max(logits)
    e = np.exp(z)
    return e / e.sum()


class Decider:
    """The board's rule: a word is reported when its probability is >= threshold and at least
    refractory_ms have passed since the last report; silence and unknown are never reported."""

    def __init__(self, threshold: float = 0.7, refractory_ms: int = 1000):
        self.threshold = threshold
        self.refractory_ms = refractory_ms
        self.last_ms = -10 ** 9

    def step(self, probs: np.ndarray, now_ms: int) -> str | None:
        k = int(np.argmax(probs))
        if k < 2 or probs[k] < self.threshold or now_ms - self.last_ms < self.refractory_ms:
            return None
        self.last_ms = now_ms
        return LABELS[k]


def read_wav(path) -> np.ndarray:
    """16 kHz mono 16-bit PCM WAV -> int16 samples (stdlib only)."""
    import wave
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == RATE and w.getnchannels() == 1 and w.getsampwidth() == 2, path
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
