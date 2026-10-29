"""Log-mel feature extractor. The C++ twin is firmware/kws_esp32/dsp.h -- keep them in sync."""
import numpy as np
from config import *

HANN = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(WIN) / WIN)).astype(np.float32)


def mel_filterbank():
    hz2mel = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)
    mel2hz = lambda m: 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    pts = mel2hz(np.linspace(hz2mel(FMIN), hz2mel(FMAX), NMEL + 2))
    freqs = np.arange(NFFT // 2 + 1) * SR / NFFT
    W = np.zeros((NMEL, NFFT // 2 + 1), np.float32)
    for m in range(NMEL):
        lo, c, hi = pts[m], pts[m + 1], pts[m + 2]
        W[m] = np.maximum(0.0, np.minimum((freqs - lo) / (c - lo), (hi - freqs) / (hi - c)))
    return W


MEL = mel_filterbank()
_IDX = np.arange(NFRAMES)[:, None] * HOP + np.arange(WIN)[None, :]


def fix_len(x, n=CLIP):
    x = np.asarray(x, np.float32)
    return np.pad(x, (0, n - len(x))) if len(x) < n else x[:n]


def logmel(x):
    """x: float32 array [..., 16000] in [-1, 1]  ->  [..., 49, 40] natural-log mel energies."""
    x = np.asarray(x, np.float32)
    fr = x[..., _IDX].astype(np.float32)                 # [..., 49, 480]
    fr -= fr.mean(-1, keepdims=True)                     # per-frame DC removal (ADC has an offset)
    fr *= HANN
    spec = np.fft.rfft(fr, n=NFFT, axis=-1)
    p = (spec.real ** 2 + spec.imag ** 2).astype(np.float32)
    return np.log(p @ MEL.T + 1e-6).astype(np.float32)
